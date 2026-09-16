from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

# Use non-interactive backend before importing pyplot
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.market_loader import (
    iter_time_slices,
    load_contract_meta,
    load_ohlcv,
)
from engine.rules_loader import query as load_rules
from strategies.base import StrategyBase, Decision, AccountState
from strategies.registry import create_strategy

_THIS_FILE = Path(__file__).resolve()
ENGINE_DIR = _THIS_FILE.parent
PROJECT_ROOT = ENGINE_DIR.parent

DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports" / "backtests"
BEIJING_TZ = "Asia/Shanghai"


@dataclass
class BacktestConfig:
    contract: str
    start: str | datetime
    end: str | datetime
    strategy: str
    step_min: int = 1
    bar_interval_min: Optional[int] = None
    initial_cash: float = 1_000_000.0
    slippage: int = 1
    strategy_params: Optional[Dict[str, Any]] = None
    report_root: Path = field(default_factory=lambda: DEFAULT_REPORT_ROOT)


@dataclass
class BacktestResult:
    config: BacktestConfig
    summary: Dict[str, Any]
    output_dir: Path
    trades_path: Path
    equity_path: Path
    summary_path: Path
    chart_path: Path
    kline_path: Path


def _extract_symbol(contract: str) -> str:
    clean = contract.strip()
    if not clean:
        raise ValueError("Contract code is empty.")
    idx = 0
    while idx < len(clean) and clean[idx].isalpha():
        idx += 1
    if idx == 0:
        raise ValueError(f"Contract code has no leading letters: {contract}")
    return clean[:idx].upper()


def _format_date(date_like: str | datetime) -> str:
    if isinstance(date_like, datetime):
        return date_like.strftime("%Y%m%d")
    text = str(date_like).strip()
    if not text:
        raise ValueError("Date string cannot be empty.")
    text = text.replace("-", "").replace("/", "")
    if len(text) >= 8:
        return text[:8]
    raise ValueError(f"Unexpected date format: {date_like}")


def _compute_fee(
    quantity: float,
    fill_price: float,
    multiplier: float,
    fee_method: str,
    fee_value: float,
) -> float:
    qty = abs(quantity)
    if qty == 0:
        return 0.0
    method = (fee_method or "AMOUNT").upper()
    if method == "AMOUNT":
        return qty * fee_value
    if method == "RATIO":
        return qty * fill_price * multiplier * fee_value
    # Unknown fee method; fall back to zero but keep signal for debugging.
    return 0.0


def _simulate_trade(
    position: int,
    avg_price: float,
    cash_account: float,
    trade_qty: int,
    fill_price: float,
    multiplier: float,
) -> Tuple[int, float, float, float]:
    """
    Simulate applying trade_qty contracts at fill_price.

    Returns (new_position, new_avg_price, new_cash_account, realized_pnl).
    """
    if trade_qty == 0:
        return position, avg_price, cash_account, 0.0

    new_position = position
    new_avg = avg_price
    new_cash = cash_account
    realized = 0.0

    if position == 0:
        new_position = trade_qty
        new_avg = fill_price
        return new_position, new_avg, new_cash, realized

    # Long position adjustments
    if position > 0:
        if trade_qty > 0:
            new_position = position + trade_qty
            new_avg = ((position * avg_price) + (trade_qty * fill_price)) / new_position
            return new_position, new_avg, new_cash, realized

        closing = min(position, abs(trade_qty))
        realized += (fill_price - avg_price) * multiplier * closing
        new_cash += realized
        new_position = position + trade_qty
        if new_position > 0:
            new_avg = avg_price
        elif new_position == 0:
            new_avg = 0.0
        else:
            new_avg = fill_price
        return new_position, new_avg, new_cash, realized

    # Short position adjustments
    if trade_qty < 0:
        new_position = position + trade_qty
        new_avg = (
            (abs(position) * avg_price) + (abs(trade_qty) * fill_price)
        ) / abs(new_position)
        return new_position, new_avg, new_cash, realized

    closing = min(abs(position), trade_qty)
    realized += (avg_price - fill_price) * multiplier * closing
    new_cash += realized
    new_position = position + trade_qty
    if new_position < 0:
        new_avg = avg_price
    elif new_position == 0:
        new_avg = 0.0
    else:
        new_avg = fill_price
    return new_position, new_avg, new_cash, realized


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_equity(equity_df: pd.DataFrame, out_path: Path) -> None:
    if equity_df.empty:
        return
    equity_series = equity_df["equity"].astype(float)
    ts_series = pd.to_datetime(equity_df["timestamp"])
    try:
        if ts_series.dt.tz is not None:
            ts_series = ts_series.dt.tz_convert(BEIJING_TZ)
        else:
            ts_series = ts_series.dt.tz_localize(BEIJING_TZ)
    except (AttributeError, TypeError):
        ts_series = ts_series.tz_localize(BEIJING_TZ)

    x = np.arange(len(equity_series))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, equity_series, label="Equity")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Bars")
    ax.set_ylabel("Equity")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    if len(x) > 0:
        if len(x) <= 12:
            tick_idx = x
        else:
            step = max(1, len(x) // 8)
            tick_idx = x[::step]
            if tick_idx[-1] != x[-1]:
                tick_idx = np.append(tick_idx, x[-1])
        tick_labels = []
        for idx in tick_idx:
            ts_value = ts_series.iloc[int(idx)]
            tick_labels.append(ts_value.strftime("%Y-%m-%d %H:%M"))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_kline_with_trades(
    ohlcv: pd.DataFrame,
    trades_df: pd.DataFrame,
    out_path: Path,
    step_minutes: int,
) -> None:
    if ohlcv.empty:
        return

    df = ohlcv.copy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx in range(len(df)):
        ax.vlines(x[idx], lows[idx], highs[idx], color="black", linewidth=0.8, zorder=1)

    width = max(0.3, min(0.7, 600 / max(len(df), 1)))
    up = closes >= opens
    down = ~up
    ax.bar(
        x[up],
        closes[up] - opens[up],
        bottom=opens[up],
        width=width,
        edgecolor="black",
        linewidth=0.5,
        facecolor="none",
        zorder=2,
        label="Up" if np.any(up) else None,
    )
    ax.bar(
        x[down],
        opens[down] - closes[down],
        bottom=closes[down],
        width=width,
        edgecolor="black",
        linewidth=0.5,
        facecolor="black",
        alpha=0.25,
        zorder=2,
        label="Down" if np.any(down) else None,
    )

    legend_seen: set[str] = set()
    if not trades_df.empty:
        trades = trades_df.copy()
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        try:
            trades["timestamp"] = trades["timestamp"].dt.tz_convert(BEIJING_TZ)
        except TypeError:
            trades["timestamp"] = trades["timestamp"].dt.tz_localize(BEIJING_TZ)

        index_positions = pd.Series(np.arange(len(df)), index=df.index)

        style_map = {
            "open_long": ("OL", "green", "^"),
            "open_short": ("OS", "red", "v"),
            "close_long": ("CL", "blue", "o"),
            "close_short": ("CS", "purple", "o"),
            "increase_long": ("AL", "green", "^"),
            "increase_short": ("AS", "red", "v"),
            "reduce_long": ("RL", "orange", "o"),
            "reduce_short": ("RS", "brown", "o"),
        }

        for _, trade in trades.iterrows():
            ts = trade["timestamp"]
            if ts in index_positions.index:
                x_pos = int(index_positions.loc[ts])
            else:
                loc = df.index.get_indexer([ts], method="nearest")
                if loc.size == 0 or loc[0] == -1:
                    continue
                x_pos = int(loc[0])

            price = float(trade["fill_price"])
            before = int(trade["position_before"])
            after = int(trade["position_after"])

            if before == 0 and after > 0:
                key = "open_long"
            elif before == 0 and after < 0:
                key = "open_short"
            elif after == 0 and before > 0:
                key = "close_long"
            elif after == 0 and before < 0:
                key = "close_short"
            elif abs(after) > abs(before):
                key = "increase_long" if after > 0 else "increase_short"
            elif abs(after) < abs(before):
                key = "reduce_long" if before > 0 else "reduce_short"
            else:
                key = "increase_long" if after > 0 else "increase_short"

            label_text, color, marker = style_map.get(key, ("TR", "black", "x"))
            legend_label = {
                "open_long": "Open Long",
                "open_short": "Open Short",
                "close_long": "Close Long",
                "close_short": "Close Short",
                "increase_long": "Increase Long",
                "increase_short": "Increase Short",
                "reduce_long": "Reduce Long",
                "reduce_short": "Reduce Short",
            }.get(key, "Trade")

            ax.scatter(
                x_pos,
                price,
                color=color,
                marker=marker,
                s=80,
                zorder=3,
                label=legend_label if legend_label not in legend_seen else None,
            )
            legend_seen.add(legend_label)
            ax.text(
                x_pos,
                price,
                label_text,
                color=color,
                fontsize=8,
                ha="center",
                va="bottom",
                zorder=4,
            )

    ax.set_title("Price with Trades")
    ax.set_xlabel(f"Bars (step={step_minutes} min)")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)

    if len(x) > 0:
        if len(x) <= 15:
            tick_idx = x
        else:
            step = max(1, len(x) // 10)
            tick_idx = x[::step]
            if tick_idx[-1] != x[-1]:
                tick_idx = np.append(tick_idx, x[-1])
        tick_labels = []
        for idx in tick_idx:
            ts_value = df.index[int(idx)]
            tick_labels.append(ts_value.tz_convert(BEIJING_TZ).strftime("%Y-%m-%d %H:%M"))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _calc_summary(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    config: BacktestConfig,
    contract_meta: Dict[str, Any],
    rule_info: Dict[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "contract": config.contract,
        "strategy": config.strategy,
        "start": str(config.start),
        "end": str(config.end),
        "step_min": config.step_min,
        "bar_interval_min": config.bar_interval_min or config.step_min,
        "initial_cash": config.initial_cash,
        "slippage_tick": config.slippage,
        "timezone": BEIJING_TZ,
        "exchange": contract_meta.get("exchange"),
        "multiplier": contract_meta.get("multiplier"),
        "tick_size": contract_meta.get("tick_size"),
        "margin_ratio": rule_info.get("margin_ratio"),
        "fee_method": rule_info.get("fee_method"),
        "fee_value": rule_info.get("fee_value"),
    }

    if equity_df.empty:
        summary.update(
            {
                "bars": 0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": None,
                "num_trades": 0,
                "win_rate": None,
                "total_fee": 0.0,
                "profit_factor": None,
            }
        )
        return summary

    equity_series = equity_df["equity"].astype(float)
    start_equity = float(equity_series.iloc[0])
    end_equity = float(equity_series.iloc[-1])
    total_return = (
        end_equity / config.initial_cash - 1.0 if config.initial_cash != 0 else 0.0
    )

    equity_with_initial = pd.concat(
        [pd.Series([config.initial_cash]), equity_series.reset_index(drop=True)],
        ignore_index=True,
    )
    running_max = equity_with_initial.cummax()
    drawdowns = equity_with_initial / running_max - 1.0
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

    timestamps = pd.to_datetime(equity_df["timestamp"])
    daily_equity = pd.Series(equity_series.to_numpy(), index=timestamps).resample("1D").last().dropna()
    returns = daily_equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe: Optional[float]
    if returns.empty or returns.std(ddof=0) == 0:
        sharpe = None
    else:
        sharpe = float(returns.mean() / returns.std(ddof=0) * math.sqrt(252))

    total_fee = float(equity_df["total_fee"].iloc[-1])

    num_trades = len(trades_df)
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    if num_trades > 0:
        realized = trades_df["realized_pnl"].astype(float)
        closing_events = realized[realized != 0]
        wins = (closing_events > 0).sum()
        win_rate = float(wins / len(closing_events)) if len(closing_events) else None
        gross_positive = realized[realized > 0].sum()
        gross_negative = realized[realized < 0].sum()
        if gross_negative != 0:
            profit_factor = float(gross_positive / abs(gross_negative))

    summary.update(
        {
            "bars": int(len(equity_df)),
            "start_equity": start_equity,
            "end_equity": end_equity,
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "sharpe": sharpe,
            "num_trades": int(num_trades),
            "win_rate": win_rate,
            "total_fee": total_fee,
            "profit_factor": profit_factor,
        }
    )
    return summary


def run_backtest(config: BacktestConfig) -> BacktestResult:
    strategy_params = config.strategy_params or {}
    bar_interval = config.bar_interval_min or config.step_min

    bars = load_ohlcv(
        contract=config.contract,
        start=config.start,
        end=config.end,
        bar_interval_min=bar_interval,
        tz=BEIJING_TZ,
    )
    if bars.empty:
        raise ValueError("No OHLCV data returned for the specified window.")

    if bars.index.tz is None:
        bars.index = bars.index.tz_localize(BEIJING_TZ)
    else:
        bars.index = bars.index.tz_convert(BEIJING_TZ)

    symbol = _extract_symbol(config.contract)
    meta = load_contract_meta(symbol)
    multiplier = float(meta.get("multiplier", 1.0))
    tick_size = float(meta.get("tick_size", 1.0))

    rule_info = load_rules(config.contract, config.start)
    margin_ratio = float(rule_info.get("margin_ratio", 0.1))
    fee_method = str(rule_info.get("fee_method", "AMOUNT"))
    fee_value = float(rule_info.get("fee_value", 0.0))

    strategy: StrategyBase = create_strategy(config.strategy, **strategy_params)

    report_root = config.report_root or DEFAULT_REPORT_ROOT
    _ensure_directory(report_root)
    start_label = _format_date(config.start)
    end_label = _format_date(config.end)
    folder_name = (
        f"{config.contract}_{start_label}_{end_label}_"
        f"{config.strategy}_{config.step_min}m"
    )
    out_dir = report_root / folder_name
    _ensure_directory(out_dir)

    position = 0
    avg_price = 0.0
    cash_account = float(config.initial_cash)
    margin = 0.0
    cumulative_fee = 0.0
    cumulative_realized = 0.0
    equity_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    log_rows: List[Dict[str, Any]] = []

    warmup = max(0, int(strategy.warmup_bars))

    for ts, visible in iter_time_slices(
        bars,
        simulate_step_min=config.step_min,
        min_warmup_bars=warmup,
    ):
        window = visible.copy()
        if ts.tzinfo is not None:
            ts_local = ts.tz_convert(BEIJING_TZ)
        else:
            ts_local = ts.tz_localize(BEIJING_TZ)
        price = float(window["close"].iloc[-1])
        position_before = position
        avg_before = avg_price

        # Cost-basis accounting: cash changes only when P&L is realized and
        # when fees are charged. Open P&L is represented by `unrealized`.
        margin_current = abs(position_before) * price * multiplier * margin_ratio
        available_cash_current = cash_account - margin_current
        unrealized_before = (price - avg_price) * multiplier * position_before
        equity_before = cash_account + unrealized_before
        account_state = AccountState(
            timestamp=ts_local,
            contract=config.contract,
            price=price,
            position=position_before,
            avg_price=avg_price,
            cash_account=cash_account,
            available_cash=available_cash_current,
            margin=margin_current,
            equity=equity_before,
            realized_pnl=cumulative_realized,
            unrealized_pnl=unrealized_before,
            multiplier=multiplier,
        )

        decision: Decision
        try:
            decision = strategy.on_bar(window, account_state)
        except Exception as exc:  # pragma: no cover - defensive logging
            log_rows.append(
                {
                    "timestamp": ts_local.isoformat(),
                    "level": "ERROR",
                    "message": f"Strategy raised exception: {exc}",
                }
            )
            decision = Decision(target=position_before, info={"error": str(exc)})

        if decision.delta is not None:
            trade_qty = int(decision.delta)
            target = position_before + trade_qty
            strategy.remember_target(target)
        else:
            remembered = strategy.remember_target(decision.target)
            if remembered is None:
                remembered = position_before
            target = int(remembered)
            trade_qty = target - position_before

        executed_trade = False
        fill_price = price
        realized = 0.0
        fee = 0.0

        if trade_qty != 0:
            slip_price = config.slippage * tick_size
            if trade_qty > 0:
                fill_price = price + slip_price
            else:
                fill_price = max(price - slip_price, 0.0)

            sim_position, sim_avg, sim_cash, realized = _simulate_trade(
                position, avg_price, cash_account, trade_qty, fill_price, multiplier
            )

            fee = _compute_fee(
                trade_qty,
                fill_price,
                multiplier,
                fee_method,
                fee_value,
            )
            sim_cash -= fee
            margin_required = abs(sim_position) * price * multiplier * margin_ratio

            if sim_cash < margin_required:
                log_rows.append(
                    {
                        "timestamp": ts_local.isoformat(),
                        "level": "WARNING",
                        "message": (
                            "Trade skipped due to insufficient cash for margin: "
                            f"target={target}, required_margin={margin_required:.2f}, "
                            f"cash_after={sim_cash:.2f}"
                        ),
                    }
                )
            else:
                executed_trade = True
                position = sim_position
                avg_price = sim_avg
                cash_account = sim_cash
                margin = margin_required
                cumulative_fee += fee
                cumulative_realized += realized
                trade_rows.append(
                    {
                        "timestamp": ts_local.isoformat(),
                        "price": price,
                        "fill_price": fill_price,
                        "trade_qty": trade_qty,
                        "position_before": position_before,
                        "position_after": position,
                        "realized_pnl": realized,
                        "fee": fee,
                        "cash_account": cash_account,
                        "margin": margin,
                        "decision_target": decision.target,
                        "decision_delta": decision.delta,
                        "decision_info": json.dumps(decision.info or {}),
                    }
                )

        margin = abs(position) * price * multiplier * margin_ratio
        available_cash = cash_account - margin
        cash_report = cash_account

        if available_cash < 0:
            log_rows.append(
                {
                    "timestamp": ts_local.isoformat(),
                    "level": "WARNING",
                    "message": (
                        "Negative available cash after margin requirement. "
                        f"Position={position}, available_cash={available_cash:.2f}"
                    ),
                }
            )

        unrealized = (price - avg_price) * multiplier * position
        equity_value = cash_account + unrealized

        equity_rows.append(
            {
                "timestamp": ts_local.isoformat(),
                "price": price,
                "position": position,
                "target": target,
                "executed_trade": executed_trade,
                "avg_price": avg_price,
                "cash": cash_report,
                "cash_account": cash_account,
                "margin": margin,
                "available_cash": available_cash,
                "unrealized_pnl": unrealized,
                "cumulative_realized_pnl": cumulative_realized,
                "total_fee": cumulative_fee,
                "equity": equity_value,
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)
    logs_df = pd.DataFrame(log_rows)

    trades_path = out_dir / "trades.csv"
    equity_path = out_dir / "equity.csv"
    summary_path = out_dir / "summary.json"
    chart_path = out_dir / "chart_equity.png"
    kline_path = out_dir / "chart_kline.png"
    log_path = out_dir / "logs.csv"

    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
    else:
        trades_df.to_csv(trades_path, index=False, columns=[
            "timestamp",
            "price",
            "fill_price",
            "trade_qty",
            "position_before",
            "position_after",
            "realized_pnl",
            "fee",
            "cash_account",
            "margin",
            "decision_target",
            "decision_delta",
            "decision_info",
        ])

    if not equity_df.empty:
        equity_df.to_csv(equity_path, index=False)
    else:
        equity_df.to_csv(equity_path, index=False, columns=[
            "timestamp",
            "price",
            "position",
            "target",
            "executed_trade",
            "avg_price",
            "cash",
            "cash_account",
            "margin",
            "available_cash",
            "unrealized_pnl",
            "cumulative_realized_pnl",
            "total_fee",
            "equity",
        ])

    if not logs_df.empty:
        logs_df.to_csv(log_path, index=False)

    summary = _calc_summary(equity_df, trades_df, config, meta, rule_info)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    _plot_equity(equity_df, chart_path)
    _plot_kline_with_trades(
        ohlcv=bars,
        trades_df=trades_df,
        out_path=kline_path,
        step_minutes=config.step_min,
    )

    return BacktestResult(
        config=config,
        summary=summary,
        output_dir=out_dir,
        trades_path=trades_path,
        equity_path=equity_path,
        summary_path=summary_path,
        chart_path=chart_path,
        kline_path=kline_path,
    )
