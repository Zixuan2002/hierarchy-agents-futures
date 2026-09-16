from __future__ import annotations

import pandas as pd

from .base import StrategyBase, Decision, AccountState
from .registry import register


@register("cash_weighted_ma")
class CashWeightedMovingAverage(StrategyBase):
    """
    MA-based direction, position size scaled by available cash.

    Parameters
    ----------
    fast : int = 5
    slow : int = 20
    price_col : str = "close"
    unit_cash : float = 150000.0  # cash required per additional contract
    max_contracts : int = 5
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.fast = int(self.params.get("fast", 5))
        self.slow = int(self.params.get("slow", 20))
        if self.fast <= 0 or self.slow <= 0:
            raise ValueError("fast/slow must be positive integers")
        self.price_col = str(self.params.get("price_col", "close"))
        self.unit_cash = float(self.params.get("unit_cash", 150000.0))
        if self.unit_cash <= 0:
            raise ValueError("unit_cash must be positive")
        self.max_contracts = max(1, int(self.params.get("max_contracts", 5)))

    @property
    def warmup_bars(self) -> int:
        return max(self.fast, self.slow)

    def on_bar(self, bars: pd.DataFrame, account: AccountState) -> Decision:
        if len(bars) < self.warmup_bars:
            decided = self.remember_target(0)
            return Decision(target=decided, info={"reason": "warmup"})

        series = bars[self.price_col]
        fast_ma = series.rolling(self.fast, min_periods=self.fast).mean().iloc[-1]
        slow_ma = series.rolling(self.slow, min_periods=self.slow).mean().iloc[-1]
        price = float(series.iloc[-1])

        direction = 0
        if fast_ma > slow_ma:
            direction = 1
        elif fast_ma < slow_ma:
            direction = -1

        available_units = max(
            0, min(self.max_contracts, int(account.available_cash // self.unit_cash))
        )

        if direction == 0 or available_units == 0:
            target = 0
        else:
            target = direction * available_units

        decided = self.remember_target(target)
        info = {
            "fast_ma": float(fast_ma),
            "slow_ma": float(slow_ma),
            "price": price,
            "direction": direction,
            "available_cash": account.available_cash,
            "chosen_contracts": abs(decided) if decided is not None else 0,
        }
        return Decision(target=decided, info=info)
