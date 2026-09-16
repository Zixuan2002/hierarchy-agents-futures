# rl/signal_envs.py

from __future__ import annotations
from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def _extract_technical_features(data_slice: pd.Series | pd.DataFrame) -> np.ndarray:
    """
    Build the technical indicator feature vector for a single timestep.

    For nb_stock = 1, this is simply [macd, rsi, cci, adx] of that contract.
    """
    if isinstance(data_slice, pd.DataFrame):
        # If multiple rows, assume single-asset and take first row
        row = data_slice.iloc[0]
    else:
        row = data_slice

    macd = float(row["macd"])
    rsi = float(row["rsi"])
    cci = float(row["cci"])
    adx = float(row["adx"])

    return np.array([macd, rsi, cci, adx], dtype=np.float32)


class BaseSignalEnv(gym.Env):
    """
    Single-asset environment that learns a *signal* (position strength)
    from technical indicators, without handling cash / inventory explicitly.

    Observation (shape = 4):
        [macd, rsi, cci, adx]

    Action:
        mode='long'  -> Box(0, 1, (1,))   (long strength)
        mode='short' -> Box(-1, 0, (1,))  (short strength)

    Reward:
        approx. action * next return
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        mode: Literal["long", "short"] = "long",
        start_day: int | None = None,
        end_day: int | None = None,
    ) -> None:
        super().__init__()

        assert mode in {"long", "short"}
        self.mode = mode

        self.dataframe = dataframe
        self.start_day = int(start_day) if start_day is not None else int(dataframe.index.min())
        self.end_day = int(end_day) if end_day is not None else int(dataframe.index.max())
        self.day = self.start_day

        # Observation space: [macd, rsi, cci, adx]
        example_slice = dataframe.loc[self.day, :]
        obs_example = _extract_technical_features(example_slice)
        low = np.full_like(obs_example, -np.inf, dtype=np.float32)
        high = np.full_like(obs_example, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space
        if mode == "long":
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=0.0, shape=(1,), dtype=np.float32)

        self._last_obs: np.ndarray | None = None

    # -----------------------------------------------
    def _get_price(self, day: int) -> float:
        row = self.dataframe.loc[day, :]
        return float(row["adjcp"])

    def _get_obs(self) -> np.ndarray:
        return _extract_technical_features(self.dataframe.loc[self.day, :])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.day = self.start_day
        obs = self._get_obs()
        self._last_obs = obs
        return obs.copy(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_scalar = float(action[0])

        # Move to next day
        if self.day >= self.end_day:
            raise RuntimeError("step() called after the episode terminated")
        self.day += 1
        done = self.day >= self.end_day

        # Reward based on next-period return times action
        price_prev = self._get_price(self.day - 1)
        price_curr = self._get_price(self.day)
        if price_prev != 0:
            ret = (price_curr - price_prev) / price_prev
        else:
            ret = 0.0

        # For long mode, positive returns with positive actions are rewarded
        # For short mode, negative actions should be rewarded when returns are negative
        reward = action_scalar * ret

        obs = self._get_obs()
        self._last_obs = obs

        terminated = done
        truncated = False
        info = {"return": ret}
        return obs.copy(), float(reward), terminated, truncated, info


class LongSignalEnv(BaseSignalEnv):
    def __init__(self, dataframe: pd.DataFrame, start_day: int | None = None, end_day: int | None = None):
        super().__init__(dataframe=dataframe, mode="long", start_day=start_day, end_day=end_day)


class ShortSignalEnv(BaseSignalEnv):
    def __init__(self, dataframe: pd.DataFrame, start_day: int | None = None, end_day: int | None = None):
        super().__init__(dataframe=dataframe, mode="short", start_day=start_day, end_day=end_day)
