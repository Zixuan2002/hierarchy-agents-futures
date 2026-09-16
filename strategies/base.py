# strategies/base.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class Decision:
    """
    Strategy output instructions.

    Options
    -------
    target : Optional[int]
        Desired absolute position size (can be >1). If omitted, the strategy
        keeps the previously recorded target.
    delta : Optional[int]
        Direct change in position (positive to increase long exposure, negative
        to add shorts or reduce longs). Cannot be set together with `target`.
    info : dict
        Extra diagnostics for logging/plotting.
    """
    target: Optional[int] = None
    delta: Optional[int] = None
    info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.target is not None and self.delta is not None:
            raise ValueError("Decision cannot set both 'target' and 'delta'.")
        if self.target is not None:
            self.target = int(self.target)
        if self.delta is not None:
            self.delta = int(self.delta)


@dataclass
class AccountState:
    """
    Snapshot of account metrics visible to the strategy.
    """
    timestamp: pd.Timestamp
    contract: str
    price: float
    position: int
    avg_price: float
    cash_account: float
    available_cash: float
    margin: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    multiplier: float

class StrategyBase:
    """
    Base interface for strategy plugins.
    """
    name: str = "base"

    def __init__(self, **params):
        self.params = params
        self._last_target: Optional[int] = 0

    @property
    def warmup_bars(self) -> int:
        """
        How many bars are needed before the strategy can output signals.
        Override in subclass if needed.
        """
        return 0

    def on_bar(self, bars: pd.DataFrame, account: AccountState) -> Decision:
        """
        Called each step with all visible bars (incremental reveal).
        Must be overridden by subclass to return a Decision.
        """
        raise NotImplementedError

    def remember_target(self, candidate: Optional[int]) -> Optional[int]:
        """
        Helper to persist the last absolute target position.
        """
        if candidate is None:
            return self._last_target
        self._last_target = int(candidate)
        return self._last_target
