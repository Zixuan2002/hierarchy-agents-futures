from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional
import pandas as pd
import sys

_THIS_FILE = Path(__file__).resolve()
ENGINE_DIR = _THIS_FILE.parent
PROJECT_ROOT = ENGINE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.plot_kline import read_ticks, build_ohlcv  # type: ignore
except Exception as e:
    raise ImportError(
        f"Failed to import scripts.plot_kline: {e}\n"
        f"Ensure scripts/plot_kline.py exists under {PROJECT_ROOT}"
    )

def load_ohlcv(
    contract: str,
    start: str | datetime,
    end: str | datetime,
    bar_interval_min: int = 1,
    tz: str = "Asia/Shanghai",
    **kwargs,
) -> pd.DataFrame:
    """
    Read ticks and aggregate to OHLCV bars.

    Parameters
    ----------
    contract : str
        Instrument code like 'rb2511', 'IF2503', etc.
    start, end : str|datetime
        Time window. Accepts 'YYYYMMDD' or 'YYYY-MM-DD[ ...]'.
    bar_interval_min : int
        Bar size in minutes for aggregation.
    tz : str
        Target timezone for output index.
    kwargs :
        Extra arguments passed through to your read_ticks/build_ohlcv if needed.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (tz-aware if your builders do that), columns like:
        ['open','high','low','close','volume', ...]
    """
    # Your read_ticks() should handle contract + time window
    ticks = read_ticks(contract, start, end, tz=tz, **kwargs)
    if ticks is None or len(ticks) == 0:
        raise ValueError(f"No ticks returned for {contract} between {start} and {end}")

    # Your build_ohlcv() should accept (ticks, interval_min=?, tz=?)
    ohlcv = build_ohlcv(ticks, interval_min=bar_interval_min, tz=tz, **kwargs)

    # Robust index coercion to DatetimeIndex (in case)
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        if "dt" in ohlcv.columns:
            ohlcv = ohlcv.set_index(pd.to_datetime(ohlcv["dt"]))
        else:
            raise ValueError("OHLCV does not have a DatetimeIndex or 'dt' column.")

    # Ensure index is sorted and unique (basic hygiene)
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")].sort_index()
    return ohlcv

def iter_time_slices(
    ohlcv: pd.DataFrame,
    simulate_step_min: int = 1,
    min_warmup_bars: int = 0,
) -> Generator[tuple[pd.Timestamp, pd.DataFrame], None, None]:
    """
    Incrementally reveal bars like a live feed.

    Yields
    ------
    (current_end_ts, visible_df)

    Notes
    -----
    - If min_warmup_bars>0, the first yield will start after that many bars.
    - Step is in bars, not seconds (since index is bar-closed timestamps).
    """
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("iter_time_slices expects a DatetimeIndex on ohlcv.")

    if len(ohlcv) == 0:
        return

    # Number of bars to advance each step (simulate_step_min is in minutes per bar)
    step = max(1, int(round(simulate_step_min / max(1, int(round((ohlcv.index[1] - ohlcv.index[0]).total_seconds() / 60))) )))
    # Above: guards in case bar size != simulate step; we still move >=1 bar.

    start_idx = max(0, min_warmup_bars)
    for i in range(start_idx, len(ohlcv), step):
        # Reveal up to bar i (inclusive)
        end_ts = ohlcv.index[i]
        yield end_ts, ohlcv.iloc[: i + 1, :]

def debug_paths() -> dict:
    return {
        "this_file": str(_THIS_FILE),
        "engine_dir": str(ENGINE_DIR),
        "project_root": str(PROJECT_ROOT),
    }


from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import time as dtime


# === NEW: load contract meta (exchange, multiplier, tick_size, trading_hours...) ===
def load_contract_meta(symbol_prefix: str, meta_path: Path | None = None) -> dict:
    """
    Read contract meta from data/meta/contracts.json and return the section for `symbol_prefix`.
    The `symbol_prefix` will be uppercased before lookup (e.g., "rb" -> "RB").
    """
    # project root: <...>/futures-agent/
    project_root = Path(__file__).resolve().parents[1]
    if meta_path is None:
        meta_path = project_root / "data" / "meta" / "contracts.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    key = symbol_prefix.upper()
    if key not in meta:
        raise KeyError(f"[contracts.json] key not found: {key}. Available: {list(meta.keys())[:8]}")
    return meta[key]


# === NEW: build an in-session boolean mask for a tz-aware DatetimeIndex ===
def session_mask(index: pd.DatetimeIndex,
                 trading_hours: list[list[str]] | tuple[tuple[str, str], ...],
                 tz: str = "Asia/Shanghai") -> pd.Series:
    """
    Return a boolean Series indicating whether each timestamp is inside any trading session window.

    Parameters
    ----------
    index : tz-aware DatetimeIndex
    trading_hours : list of [start, end] strings like [["09:00","10:15"],["10:30","11:30"],...]
    tz : timezone string; index will be converted to this tz for comparison

    Notes
    -----
    - Handles same-day sessions (e.g., 21:00–23:00). If you ever have overnight spans (e.g., 21:00–02:30),
      you can extend this by splitting into two windows [21:00, 23:59:59] and [00:00, 02:30].
    """
    if index.tz is None:
        raise ValueError("index must be tz-aware. Provide tz in your data loader (e.g., Asia/Shanghai).")

    # convert to local tz once
    idx_local = index.tz_convert(tz)
    t = idx_local.time  # numpy array of datetime.time

    # pre-parse session bounds
    sessions: list[tuple[dtime, dtime]] = []
    for s, e in trading_hours:
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        sessions.append((dtime(sh, sm), dtime(eh, em)))

    mask = np.zeros(len(index), dtype=bool)
    # OR across sessions
    for s, e in sessions:
        # same-day window (start <= time < end)
        in_win = (t >= s) & (t < e)
        mask |= in_win

    return pd.Series(mask, index=index)
