"""
python -m engine.run_backtest \
    --contract rb2305 \
    --start 2023-02-09 \
    --end 2023-02-10 \
    --strategy ma_cross \
    --step-min 1
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from engine.backtest_runner import BacktestConfig, run_backtest, DEFAULT_REPORT_ROOT
from strategies.registry import available_strategies


def _load_strategy_plugins() -> None:
    pkg = importlib.import_module("strategies")
    pkg_path = Path(pkg.__file__).parent
    for _finder, name, is_pkg in pkgutil.iter_modules([str(pkg_path)]):
        if is_pkg or name.startswith("_"):
            continue
        importlib.import_module(f"strategies.{name}")


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if lowered.startswith("0x"):
            return int(lowered, 16)
        if "." in lowered:
            return float(lowered)
        return int(lowered)
    except ValueError:
        return value


def _parse_params(param_items: Iterable[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in param_items:
        if "=" not in item:
            raise ValueError(
                f"Strategy parameter must be in key=value format. Received: '{item}'"
            )
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Strategy parameter has empty key: '{item}'")
        parsed[key] = _coerce_value(raw.strip())
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single backtest job.")
    parser.add_argument("--contract", required=True, help="Contract code, e.g., rb2305")
    parser.add_argument("--start", required=True, help="Start date (YYYYMMDD)")
    parser.add_argument("--end", required=True, help="End date (YYYYMMDD)")
    parser.add_argument("--strategy", required=True, help="Registered strategy name")
    parser.add_argument("--step-min", type=int, default=1, help="Simulation step in minutes")
    parser.add_argument(
        "--bar-interval-min",
        type=int,
        default=None,
        help="Bar aggregation interval in minutes (default: step-min)",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=1_000_000.0,
        help="Initial cash balance (default: 1_000_000)",
    )
    parser.add_argument(
        "--slippage",
        type=int,
        default=1,
        help="Slippage in ticks applied to fills (default: 1)",
    )
    parser.add_argument(
        "--strategy-param",
        action="append",
        default=[],
        help="Strategy parameter in key=value form. Can be repeated.",
    )
    parser.add_argument(
        "--report-root",
        type=str,
        default=None,
        help="Optional output root (default: reports/backtests)",
    )
    return parser


def _normalize_dates(start: str, end: str) -> Tuple[str, str]:
    def _clean(value: str, name: str) -> str:
        text = str(value).strip()
        text = text.replace("-", "").replace("/", "")
        if len(text) != 8 or not text.isdigit():
            raise ValueError(f"{name} must be in YYYYMMDD format (got '{value}')")
        return text

    start_clean = _clean(start, "start")
    end_clean = _clean(end, "end")
    if end_clean < start_clean:
        raise ValueError(f"end ({end_clean}) must be >= start ({start_clean})")
    return start_clean, end_clean


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    _load_strategy_plugins()

    registered = available_strategies()
    if args.strategy not in registered:
        parser.error(
            f"Strategy '{args.strategy}' is not registered. Available: {registered}"
        )

    try:
        strategy_params = _parse_params(args.strategy_param)
    except ValueError as exc:
        parser.error(str(exc))
        return

    report_root = Path(args.report_root) if args.report_root else DEFAULT_REPORT_ROOT

    try:
        start_date, end_date = _normalize_dates(args.start, args.end)
    except ValueError as exc:
        parser.error(str(exc))
        return

    config = BacktestConfig(
        contract=args.contract,
        start=start_date,
        end=end_date,
        strategy=args.strategy,
        step_min=args.step_min,
        bar_interval_min=args.bar_interval_min,
        initial_cash=args.cash,
        slippage=args.slippage,
        strategy_params=strategy_params or None,
        report_root=report_root,
    )

    result = run_backtest(config)

    print(f"[OK] Backtest completed. Reports saved to: {result.output_dir}")
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
