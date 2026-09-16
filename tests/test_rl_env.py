from __future__ import annotations

import numpy as np
import pandas as pd

from rl.stock_env import StockEnv, StockEnvConfig


def _dataset():
    return pd.DataFrame(
        {
            "adjcp": [100.0, 90.0, 80.0],
            "macd": [0.0, 0.0, 0.0],
            "rsi": [50.0, 50.0, 50.0],
            "cci": [0.0, 0.0, 0.0],
            "adx": [20.0, 20.0, 20.0],
        },
        index=[0, 1, 2],
    )


def test_rl_environment_can_hold_a_bounded_short():
    config = StockEnvConfig(
        initial_balance=1_000.0,
        transaction_fee=0.0,
        hmax=4,
        multiplier=10.0,
    )
    env = StockEnv(_dataset(), config)
    env.reset()
    _, _, terminated, _, info = env.step(np.array([-1.0], dtype=np.float32))
    assert env.holding == -4
    assert not terminated
    assert info["portfolio_value"] == 1_400.0


def test_rl_action_is_target_not_repeated_delta():
    config = StockEnvConfig(initial_balance=1_000.0, transaction_fee=0.0, hmax=4)
    env = StockEnv(_dataset(), config)
    env.reset()
    env.step(np.array([1.0], dtype=np.float32))
    env.step(np.array([1.0], dtype=np.float32))
    assert env.holding == 4
