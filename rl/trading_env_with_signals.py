# rl/trading_env_with_signals.py

from __future__ import annotations
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO as SB3PPO

from .stock_env import StockEnv, StockEnvConfig
from .signal_envs import _extract_technical_features


class TradingEnvWithSignals(gym.Env):
    """
    Wrap StockEnv and augment its observation with:

    - long_signal: output of a pre-trained long-signal PPO agent
    - short_signal: output of a pre-trained short-signal PPO agent
    - norm_pv: normalized portfolio value = V_t / initial_balance

    Underlying StockEnv observation (single asset):
        base_obs = [cash, price, holding, macd, rsi, cci, adx] (len=7)

    Augmented observation (len=10):
        [base_obs, long_signal, short_signal, norm_pv]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: StockEnvConfig,
        long_model: SB3PPO,
        short_model: SB3PPO,
        start_day: int | None = None,
        end_day: int | None = None,
    ) -> None:
        super().__init__()

        self.dataframe = dataframe
        self.config = config

        self.base_env = StockEnv(
            dataframe=dataframe,
            config=config,
            start_day=start_day,
            end_day=end_day,
        )

        self.long_model = long_model
        self.short_model = short_model

        # Observation space: base_obs (7) + 2 signals + 1 norm_pv = 10
        low = np.full(10, -np.inf, dtype=np.float32)
        high = np.full(10, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: same as base_env (Box(-1,1,(1,)))
        self.action_space = self.base_env.action_space

        self._last_signals: Optional[np.ndarray] = None

    # ---------------------------------------------
    def _get_portfolio_value(self) -> float:
        return self.base_env._compute_portfolio_value()

    def _compute_signals(self) -> np.ndarray:
        """
        Compute [long_signal, short_signal] at the current timestep
        using the technical indicators as input.
        """
        day = self.base_env.day
        data_slice = self.dataframe.loc[day, :]
        tech_obs = _extract_technical_features(data_slice)

        long_signal, _ = self.long_model.predict(tech_obs, deterministic=True)
        short_signal, _ = self.short_model.predict(tech_obs, deterministic=True)

        long_signal = float(np.asarray(long_signal).reshape(()))
        short_signal = float(np.asarray(short_signal).reshape(()))

        signals = np.array([long_signal, short_signal], dtype=np.float32)
        self._last_signals = signals
        return signals

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        # Signals
        signals = self._compute_signals() if (self.long_model is not None and self.short_model is not None) else (
            self._last_signals if self._last_signals is not None else np.array([0.0, 0.0], dtype=np.float32)
        )

        # Normalized portfolio value
        current_value = self._get_portfolio_value()
        norm_pv = current_value / self.config.initial_balance if self.config.initial_balance > 0 else 0.0
        pv_feature = np.array([norm_pv], dtype=np.float32)

        return np.concatenate(
            [base_obs.astype(np.float32), signals.astype(np.float32), pv_feature],
            axis=0,
        )

    # ---------------------------------------------
    def reset(self, *, seed=None, options=None):
        base_obs, info = self.base_env.reset(seed=seed, options=options)
        aug_obs = self._augment_obs(np.asarray(base_obs, dtype=np.float32))
        return aug_obs, info

    def step(self, action: np.ndarray):
        """
        Step the underlying StockEnv, then:
        - recompute reward as portfolio return
        - augment observation with signals + normalized PV
        """
        base_obs, _, terminated, truncated, info = self.base_env.step(action)

        # Portfolio-return reward
        reward = 0.0
        if hasattr(self.base_env, "asset_memory") and len(self.base_env.asset_memory) >= 2:
            prev_v, curr_v = self.base_env.asset_memory[-2], self.base_env.asset_memory[-1]
            if prev_v != 0:
                reward = (curr_v - prev_v) / prev_v

        aug_obs = self._augment_obs(np.asarray(base_obs, dtype=np.float32))
        return aug_obs, float(reward), terminated, truncated, info
