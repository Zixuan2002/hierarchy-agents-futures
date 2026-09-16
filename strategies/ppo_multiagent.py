# strategies/ppo_multiagent.py

from __future__ import annotations

import numpy as np
import pandas as pd

from stable_baselines3 import PPO as SB3PPO
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from .base import StrategyBase, Decision, AccountState
from .registry import register


@register("ppo_multiagent")
class PPOMultiAgentStrategy(StrategyBase):
    """
    Multi-agent PPO strategy:

    - long_model:  PPO trained on LongSignalEnv (outputs long confidence in [0,1])
    - short_model: PPO trained on ShortSignalEnv (outputs short confidence in [-1,0])
    - trader_model: PPO trained on TradingEnvWithSignals, which takes:

        base_obs  = [cash, price, holding, macd, rsi, cci, adx]
        signals   = [long_signal, short_signal]
        norm_pv   = V_t / V_0

        trader_obs = [base_obs, long_signal, short_signal, norm_pv]  # len=10

    Parameters (from --strategy-param):

        trader_model_path : str
        long_model_path   : str
        short_model_path  : str
        max_position      : int (default 3)
    """

    def __init__(
        self,
        trader_model_path: str,
        long_model_path: str,
        short_model_path: str,
        max_position: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.trader_model_path = trader_model_path
        self.long_model_path = long_model_path
        self.short_model_path = short_model_path
        self.max_position = int(max_position)

        self.trader_model: SB3PPO | None = None
        self.long_model: SB3PPO | None = None
        self.short_model: SB3PPO | None = None

        # We no longer depend on a precomputed self.features from on_start;
        # we’ll compute indicators directly from `bars` in on_bar.
        self.initial_value: float | None = None

    @property
    def warmup_bars(self) -> int:
        return 30

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_models_loaded(self) -> None:
        """Lazy-load PPO models on first use."""
        if (
            self.long_model is not None
            and self.short_model is not None
            and self.trader_model is not None
        ):
            return

        self.long_model = SB3PPO.load(self.long_model_path)
        self.short_model = SB3PPO.load(self.short_model_path)
        self.trader_model = SB3PPO.load(self.trader_model_path)

    def _compute_indicator_row(self, bars: pd.DataFrame) -> pd.Series | None:
        """
        Compute MACD, RSI, CCI, ADX on the *current* visible bars, and
        return the last row (current bar) with these columns.
        """
        if bars.empty:
            return None

        df = bars.copy()

        # Make sure necessary columns exist
        if not {"close", "high", "low"}.issubset(df.columns):
            return None

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        df["macd"] = MACD(close).macd()
        df["rsi"] = RSIIndicator(close).rsi()
        df["cci"] = CCIIndicator(high=high, low=low, close=close).cci()
        df["adx"] = ADXIndicator(high=high, low=low, close=close).adx()

        row = df.iloc[-1]

        # Warmup check: if any indicator is NaN at the current bar, skip trading
        if any(np.isnan(row[k]) for k in ["macd", "rsi", "cci", "adx"]):
            return None

        return row

    def _signal_obs_from_row(self, row: pd.Series) -> np.ndarray:
        return np.array(
            [row["macd"], row["rsi"], row["cci"], row["adx"]],
            dtype=np.float32,
        )

    def _base_obs_from_row(self, row: pd.Series, cash: float, holding: float) -> np.ndarray:
        price = float(row["close"])
        return np.array(
            [cash, price, holding, row["macd"], row["rsi"], row["cci"], row["adx"]],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Main decision logic each bar
    # ------------------------------------------------------------------
    def on_bar(self, bars: pd.DataFrame, account: AccountState) -> Decision:
        """
        bars: DataFrame of recent OHLCV bars; we use the last row as current bar.
        account: AccountState with .cash_account, .position, etc.
        """
        # 0) No bars: do nothing
        if bars.empty or len(bars) < 30:  # MACD needs ~26 bars minimum
            return Decision(
                target=account.position,
                info={"reason": f"Not enough bars: {len(bars)}"},
            )

        # 1) Ensure models are loaded (lazy init)
        self._ensure_models_loaded()

        # 2) Compute indicators for the current bar
        row = self._compute_indicator_row(bars)
        if row is None:
            # indicators not ready or data missing → hold
            return Decision(
                target=account.position,
                info={"reason": "Indicators not ready or OHLC missing"},
            )

        ts = bars.index[-1]

        cash = float(account.cash_account)
        holding = float(account.position)
        price = float(row["close"])

        # 3) Initialize reference portfolio value the first time
        if self.initial_value is None:
            self.initial_value = float(account.equity)
            if self.initial_value <= 0:
                self.initial_value = 1.0  # avoid division by zero

        portfolio_value = float(account.equity)
        norm_pv = float(portfolio_value / self.initial_value)

        # 4) Signal observation: [macd, rsi, cci, adx]
        sig_obs = self._signal_obs_from_row(row)

        # 5) Long / short signals
        long_sig, _ = self.long_model.predict(sig_obs, deterministic=True)
        short_sig, _ = self.short_model.predict(sig_obs, deterministic=True)
        long_sig = float(np.asarray(long_sig).reshape(()))
        short_sig = float(np.asarray(short_sig).reshape(()))

        # 6) Base trading observation (like StockEnv):
        #    [cash, price, holding, macd, rsi, cci, adx]
        base_obs = self._base_obs_from_row(row, cash=cash, holding=holding)

        # 7) Trader observation: [base_obs, long_sig, short_sig, norm_pv]
        trader_obs = np.concatenate(
            [
                base_obs,
                np.array([long_sig, short_sig], dtype=np.float32),
                np.array([norm_pv], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        # 8) Trader PPO action in [-1, 1]
        action, _ = self.trader_model.predict(trader_obs, deterministic=True)
        action = float(np.asarray(action).reshape(()))
        action = max(-1.0, min(1.0, action))

        # Map action fraction to an absolute bounded target position.
        target_float = action * self.max_position
        target_units = int(round(target_float))

        info = {
            "action_raw": action,
            "target_float": target_float,
            "target_units": target_units,
            "long_signal": long_sig,
            "short_signal": short_sig,
            "norm_pv": norm_pv,
        }

        return Decision(target=target_units, info=info)
