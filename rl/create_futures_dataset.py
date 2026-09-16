# rl/create_futures_dataset.py

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from engine.market_loader import load_ohlcv  # your existing loader


def build_futures_dataset(
    contract: str,
    bar_interval_min: int = 1,
) -> pd.DataFrame:
    """
    Load *all available* OHLCV bars for a single futures contract
    at the given bar size, then compute the technical indicators
    required by the RL environments.

    Dataset format (nb_stock = 1):
        index : integer "day" (factorized from 'date')
        columns:
            ['date','tic','open','high','low','close','adjcp','volume',
             'macd','rsi','cci','adx']
    """
    # Use very wide date range so underlying loader/DB returns all available data.
    # Assumes your ticks / bars fall within [2000-01-01, 2100-01-01].
    start = "20000101"
    end = "21000101"

    ohlcv = load_ohlcv(
        contract=contract,
        start=start,
        end=end,
        bar_interval_min=bar_interval_min,
    )
    if len(ohlcv) == 0:
        raise ValueError(f"No OHLCV bars returned for {contract} in full range {start}–{end}")

    ohlcv = ohlcv.sort_index()

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    volume = ohlcv["volume"].astype(float) if "volume" in ohlcv.columns else np.nan

    # Technical indicators (same library as original project)
    macd = MACD(close).macd()
    rsi = RSIIndicator(close).rsi()
    cci = CCIIndicator(high=high, low=low, close=close).cci()
    adx = ADXIndicator(high=high, low=low, close=close).adx()

    # Build dataset DataFrame
    idx = ohlcv.index
    # Normalize to string timestamps; if already tz-aware, convert to Asia/Shanghai
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Shanghai")
    date_str = idx.strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(
        {
            "date": date_str,
            "tic": contract,
            "open": ohlcv["open"].astype(float).values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "adjcp": close.values,  # futures: no corporate actions, so adjcp = close
            "volume": volume.values,
            "macd": macd.values,
            "rsi": rsi.values,
            "cci": cci.values,
            "adx": adx.values,
        }
    )

    # Drop rows with NaNs in indicators (warmup)
    df = df.dropna(subset=["macd", "rsi", "cci", "adx"]).reset_index(drop=True)

    # Factorize 'date' into integer "day" index as expected by StockEnv/signal envs
    df.index = df["date"].factorize()[0]

    return df


def main():
    parser = argparse.ArgumentParser(description="Build RL dataset from futures OHLCV.")
    parser.add_argument(
        "--contract",
        type=str,
        required=True,
        help="Futures contract code, e.g., al2406",
    )
    parser.add_argument(
        "--bar-interval-min",
        type=int,
        default=1,
        help="Bar size in minutes (default 1).",
    )
    args = parser.parse_args()

    df = build_futures_dataset(
        contract=args.contract,
        bar_interval_min=args.bar_interval_min,
    )

    # Fixed output convention: data/rl/processed_dataset_{contract}.csv
    out_dir = Path("data") / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"processed_dataset_{args.contract}.csv"

    df.to_csv(out_path, index=True)  # keep integer 'day' index

    print(f"[create_futures_dataset] Saved dataset for {args.contract} to {out_path}")


if __name__ == "__main__":
    main()
