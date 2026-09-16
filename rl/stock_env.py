# rl/stock_env.py

from __future__ import annotations
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class StockEnvConfig:
    initial_balance: float = 1_000_000.0
    transaction_fee: float = 0.001
    hmax: int = 100
    reward_scaling: float = 1e-4
    nb_stock: int = 1  # fixed to 1 in our use-case
    multiplier: float = 1.0
    margin_ratio: float = 0.1


class StockEnv(gym.Env):
    """
    Single-asset trading environment similar to the original StockEnv,
    but specialized to nb_stock = 1 and using our futures dataset.

    Observation (shape = 7):
        [cash, price, holding, macd, rsi, cci, adx]

    Action (shape = (1,)):
        continuous in [-1, 1]; we interpret it as *target fraction of max volume*.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: StockEnvConfig,
        start_day: int | None = None,
        end_day: int | None = None,
    ) -> None:
        super().__init__()

        assert config.nb_stock == 1, "This StockEnv is single-asset only."

        self.dataframe = dataframe
        self.config = config

        self.start_day = int(start_day) if start_day is not None else int(dataframe.index.min())
        self.end_day = int(end_day) if end_day is not None else int(dataframe.index.max())
        self.day = self.start_day

        self.nb_stock = 1
        self.initial_balance = float(config.initial_balance)
        self.transaction_fee = float(config.transaction_fee)
        self.hmax = int(config.hmax)
        self.reward_scaling = float(config.reward_scaling)
        self.multiplier = float(config.multiplier)
        self.margin_ratio = float(config.margin_ratio)

        # State: [cash, price, holding, macd, rsi, cci, adx]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )
        # Action: one scalar in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self._reset_internal_state()

    # ---------------------------------------------------
    # Helpers
    # ---------------------------------------------------
    def _get_current_row(self) -> pd.Series:
        return self.dataframe.loc[self.day, :]

    def _get_price(self) -> float:
        return float(self._get_current_row()["adjcp"])

    def _get_indicators(self) -> tuple[float, float, float, float]:
        row = self._get_current_row()
        return (
            float(row["macd"]),
            float(row["rsi"]),
            float(row["cci"]),
            float(row["adx"]),
        )

    def _compute_portfolio_value(self) -> float:
        price = self._get_price()
        unrealized = (price - self.avg_price) * self.multiplier * self.holding
        return self.cash + unrealized

    def _reset_internal_state(self) -> None:
        self.day = self.start_day
        self.cash = self.initial_balance
        self.holding = 0.0
        self.avg_price = 0.0
        self.asset_memory: list[float] = [self.initial_balance]
        self.reward_mem: list[float] = []
        self._update_state()

    def _update_state(self) -> None:
        price = self._get_price()
        macd, rsi, cci, adx = self._get_indicators()
        self.state = np.array(
            [self.cash, price, self.holding, macd, rsi, cci, adx],
            dtype=np.float32,
        )

    # ---------------------------------------------------
    # Gym API
    # ---------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal_state()
        return self.state.copy(), {}

    def step(self, action: np.ndarray):
        """
        action: np.array of shape (1,) in [-1, 1].
        We interpret it as an absolute target position in [-hmax, hmax].
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_scalar = float(action[0])

        target = int(round(action_scalar * self.hmax))
        current = int(self.holding)
        trade_units = target - current

        price = self._get_price()

        if trade_units != 0:
            if current > 0 and trade_units < 0:
                closing = min(current, -trade_units)
                self.cash += (price - self.avg_price) * self.multiplier * closing
            elif current < 0 and trade_units > 0:
                closing = min(-current, trade_units)
                self.cash += (self.avg_price - price) * self.multiplier * closing

            self.cash -= abs(trade_units) * price * self.multiplier * self.transaction_fee
            new_position = current + trade_units
            if new_position == 0:
                self.avg_price = 0.0
            elif current == 0 or (current > 0 > new_position) or (current < 0 < new_position):
                self.avg_price = price
            elif (current > 0 and trade_units > 0) or (current < 0 and trade_units < 0):
                self.avg_price = (
                    abs(current) * self.avg_price + abs(trade_units) * price
                ) / abs(new_position)
            self.holding = float(new_position)

        # Advance time
        if self.day >= self.end_day:
            raise RuntimeError("step() called after the episode terminated")
        self.day += 1
        done = self.day >= self.end_day

        # Compute new portfolio value
        new_portfolio_value = self._compute_portfolio_value()
        self.asset_memory.append(new_portfolio_value)

        # Reward: portfolio return scaled
        if len(self.asset_memory) >= 2 and self.asset_memory[-2] != 0:
            rtn = (self.asset_memory[-1] - self.asset_memory[-2]) / self.asset_memory[-2]
        else:
            rtn = 0.0
        reward = rtn * self.reward_scaling
        self.reward_mem.append(reward)

        # Update state
        self._update_state()

        terminated = done
        truncated = False
        info = {"portfolio_value": new_portfolio_value}
        return self.state.copy(), reward, terminated, truncated, info
