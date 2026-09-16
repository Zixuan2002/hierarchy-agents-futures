"""
Build minute-based OHLCV from per-instrument SQLite DBs and render a large candlestick chart.

Example:
  python scripts/plot_kline.py --instrument al2303 --interval 5 --tz Asia/Shanghai
  python scripts/plot_kline.py --instrument rb2305 --interval 60 --start 2023-02-09 --end 2023-02-10
"""

import argparse
import os
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "data" / "db"
OUT_DIR = PROJECT_ROOT / "reports" / "kline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_ticks(instrument: str,
               start: str | None,
               end: str | None,
               columns=("ts_event_ms","last_price","volume"),
               tz: str = "Asia/Shanghai") -> pd.DataFrame:
    """Read ticks from per-instrument SQLite database."""
    db_path = DB_DIR / f"{instrument.lower()}.db"
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    where = []
    if start:
        # start/end are in calendar local date; compare by UTC ms boundaries if needed
        start_dt = pd.Timestamp(start).tz_localize(tz).tz_convert("UTC")
        start_ms = int(start_dt.timestamp() * 1000)
        where.append(f"ts_event_ms >= {start_ms}")
    if end:
        end_dt = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
        end_ms = int(end_dt.timestamp() * 1000)
        where.append(f"ts_event_ms < {end_ms}")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    col_sql = ", ".join(columns)
    query = f"SELECT {col_sql} FROM ticks {where_sql} ORDER BY ts_event_ms ASC"
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn)
    return df

def to_timezone(dt_utc: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Convert UTC naive series to target timezone."""
    # ts_event_ms is UTC milliseconds; localize to UTC then convert
    dt = pd.to_datetime(dt_utc, utc=True)
    try:
        return dt.tz_convert(tz)
    except Exception:
        # fallback: return UTC if timezone not available
        return dt

def build_ohlcv(df: pd.DataFrame, interval_min: int, tz: str) -> pd.DataFrame:
    """Aggregate ticks to OHLCV with minute buckets."""
    if df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    # index = localized datetime
    dt_utc = pd.to_datetime(df["ts_event_ms"], unit="ms", utc=True)
    dt_local = to_timezone(dt_utc, tz)
    df = df.copy()
    df["dt"] = dt_local
    df = df.set_index("dt").sort_index()

    # Resample by interval minutes
    rule = f"{int(interval_min)}min"

    # Volume is cumulative within a session/day. Convert it to non-negative
    # per-tick increments before aggregation so the first tick in each bucket
    # is not silently discarded.
    if "volume" in df.columns:
        increments = df["volume"].astype(float).diff()
        resets = increments < 0
        increments.loc[resets] = df.loc[resets, "volume"].astype(float)
        df["volume_increment"] = increments.fillna(0.0).clip(lower=0.0)
    def agg_bucket(g: pd.DataFrame) -> pd.Series:
        if g.empty:
            return pd.Series({"open": np.nan, "high": np.nan, "low": np.nan,
                              "close": np.nan, "volume": 0})
        o = g["last_price"].iloc[0]
        h = g["last_price"].max()
        l = g["last_price"].min()
        c = g["last_price"].iloc[-1]
        vol = float(g["volume_increment"].sum()) if "volume_increment" in g else 0.0
        return pd.Series({"open": o, "high": h, "low": l, "close": c, "volume": vol})

    ohlcv = df.resample(rule).apply(agg_bucket)
    # Drop empty rows (no ticks)
    ohlcv = ohlcv.dropna(subset=["open","high","low","close"], how="any")
    return ohlcv

def plot_candles(ohlcv: pd.DataFrame,
                 title: str,
                 out_path: Path,
                 width_px: int = 3200,
                 height_px: int = 1600,
                 dpi: int = 100):
    """Render a large candlestick chart using pure Matplotlib (no seaborn)."""
    if ohlcv.empty:
        raise ValueError("No data to plot.")
    # Prepare canvas
    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # X values
    x = np.arange(len(ohlcv))
    o = ohlcv["open"].to_numpy()
    h = ohlcv["high"].to_numpy()
    l = ohlcv["low"].to_numpy()
    c = ohlcv["close"].to_numpy()

    # Wick lines
    for i in range(len(x)):
        ax.vlines(x[i], l[i], h[i], linewidth=1)

    # Candle bodies
    width = max(0.3, min(0.7, 600 / len(x)))  # auto width
    up = c >= o
    down = ~up

    # Up candles (close >= open): unfilled rectangles
    ax.bar(x[up], (c[up] - o[up]), bottom=o[up], width=width, edgecolor="black", linewidth=0.5, fill=False)
    # Down candles (close < open): filled rectangles
    ax.bar(x[down], (o[down] - c[down]), bottom=c[down], width=width, edgecolor="black", linewidth=0.5)

    # Title & axes
    ax.set_title(title)
    ax.set_xlim(-1, len(x) + 1)
    ax.set_xlabel("bars")
    ax.set_ylabel("price")
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

    # Some x-ticks
    step = max(1, len(x) // 10)
    xticks = np.arange(0, len(x), step)
    ax.set_xticks(xticks)
    # Show datetime labels for reference
    idx = ohlcv.index
    labels = [str(idx[i].tz_convert(None)) if hasattr(idx[i], "tzinfo") else str(idx[i]) for i in xticks]
    ax.set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot minute-based K-line from per-instrument SQLite DB.")
    ap.add_argument("--instrument", required=True, help="Instrument code (e.g., al2303)")
    ap.add_argument("--interval", type=int, required=True, help="Bar interval in minutes, e.g., 1,5,60")
    ap.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD) in local calendar")
    ap.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD) inclusive in local calendar")
    ap.add_argument("--tz", type=str, default="Asia/Shanghai", help="Timezone for bars (default: Asia/Shanghai)")
    ap.add_argument("--width", type=int, default=3200, help="Output width in pixels")
    ap.add_argument("--height", type=int, default=1600, help="Output height in pixels")
    ap.add_argument("--dpi", type=int, default=100, help="Figure DPI")
    args = ap.parse_args()

    df = read_ticks(args.instrument, args.start, args.end, tz=args.tz)
    ohlcv = build_ohlcv(df, args.interval, args.tz)

    out_name = f"{args.instrument.lower()}_{args.interval}min"
    if args.start or args.end:
        s = (args.start or "").replace("-", "")
        e = (args.end or "").replace("-", "")
        if s or e:
            out_name += f"_{s}-{e}"
    out_path = OUT_DIR / f"{out_name}.png"

    title = f"{args.instrument.upper()}  {args.interval}min  ({args.tz})"
    plot_candles(ohlcv, title, out_path, width_px=args.width, height_px=args.height, dpi=args.dpi)
    print(f"[OK] Saved K-line: {out_path}")

if __name__ == "__main__":
    main()
