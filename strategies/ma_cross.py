# strategies/ma_cross.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
from .base import StrategyBase, Decision, AccountState
from .registry import register

@register("ma_cross")
class MovingAverageCross(StrategyBase):
    """
    Simple moving-average cross strategy.

    Params
    ------
    fast : int = 10
    slow : int = 30
    price_col : str = "close"
    flat_on_equal : bool = False   # if True, target=0 when fast==slow; else keep last
    """
    def __init__(self, **params):
        super().__init__(**params)
        self.fast = int(self.params.get("fast", 10))
        self.slow = int(self.params.get("slow", 30))
        self.price_col = str(self.params.get("price_col", "close"))
        self.flat_on_equal = bool(self.params.get("flat_on_equal", False))
        if self.fast <= 0 or self.slow <= 0:
            raise ValueError("fast/slow must be positive")
        if self.fast >= self.slow:
            # Allow but warn via doc: usually fast < slow
            pass

    @property
    def warmup_bars(self) -> int:
        return max(self.fast, self.slow)

    def on_bar(self, bars: pd.DataFrame, account: AccountState) -> Decision:
        if len(bars) < self.warmup_bars:
            decided = self.remember_target(0)
            return Decision(target=decided, info={"reason": "warmup"})

        px = bars[self.price_col]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        f, s = float(fast_ma.iloc[-1]), float(slow_ma.iloc[-1])
        target: int | None
        if f > s:
            target = 1
        elif f < s:
            target = -1
        else:
            target = 0 if self.flat_on_equal else None  # keep last or flat

        decided = self.remember_target(target)
        return Decision(
            target=decided,
            info={
                "fast": self.fast,
                "slow": self.slow,
                "fast_ma": f,
                "slow_ma": s,
                "price": float(px.iloc[-1]),
            },
        )
