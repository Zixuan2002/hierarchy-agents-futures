from __future__ import annotations

import pandas as pd

from engine.backtest_runner import _simulate_trade
from engine.market_loader import iter_time_slices


def test_long_round_trip_realizes_pnl_once():
    position, avg, cash, realized = _simulate_trade(0, 0.0, 1_000.0, 1, 100.0, 10.0)
    assert (position, avg, cash, realized) == (1, 100.0, 1_000.0, 0.0)

    position, avg, cash, realized = _simulate_trade(position, avg, cash, -1, 110.0, 10.0)
    assert position == 0
    assert cash == 1_100.0
    assert realized == 100.0


def test_short_round_trip_realizes_pnl_once():
    position, avg, cash, _ = _simulate_trade(0, 0.0, 1_000.0, -2, 100.0, 10.0)
    position, avg, cash, realized = _simulate_trade(position, avg, cash, 2, 90.0, 10.0)
    assert position == 0
    assert cash == 1_200.0
    assert realized == 200.0


def test_time_slices_honor_step_minutes():
    idx = pd.date_range("2024-01-01 09:00", periods=10, freq="1min", tz="Asia/Shanghai")
    bars = pd.DataFrame({"close": range(10)}, index=idx)
    slices = list(iter_time_slices(bars, simulate_step_min=5))
    assert [ts for ts, _ in slices] == [idx[0], idx[5]]
