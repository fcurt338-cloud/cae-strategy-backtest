"""
Order-flow imbalance ranking signal: fraction of traded volume driven by
aggressive buyers (market buys hitting the ask) vs aggressive sellers.
Genuinely different from price action -- two bars can have identical OHLC
and totally different buy/sell aggression underneath.

Hypothesis (stated precisely, tested as-is, not adjusted after seeing results):
sustained aggressive buying pressure (elevated buy_ratio over a trailing
window) predicts continued upward price movement -- informed/urgent buyers
accumulating. Long the most buy-imbalanced names.
"""
import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_orderflow_wide(symbols=None) -> pd.DataFrame:
    """Daily buy_ratio per symbol, wide format. buy_ratio = sum(taker_buy_base_volume) /
    sum(volume) over each calendar day -- volume-weighted, not a simple mean of hourly
    ratios (a day with one huge hour should count more than a day of quiet hours)."""
    files = glob.glob(os.path.join(RAW_DIR, "*_orderflow.csv"))
    series = {}
    for f in files:
        sym = os.path.basename(f).replace("_orderflow.csv", "")
        if symbols is not None and sym not in symbols:
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time").sort_index()
        daily_vol = df["volume"].resample("1D").sum()
        daily_buy = df["taker_buy_base_volume"].resample("1D").sum()
        ratio = (daily_buy / daily_vol.replace(0, pd.NA))
        series[sym] = ratio
    return pd.DataFrame(series)


def compute_orderflow_rank_at(wide_ratio: pd.DataFrame, as_of_date: pd.Timestamp, lookback_days: int,
                               direction: str = "aggressive_buy") -> pd.Series:
    """Trailing volume-weighted buy_ratio over `lookback_days` ending at as_of_date
    (inclusive) -- as_of_date must already be a fully-closed day (caller's responsibility,
    same convention as the momentum/funding signal functions).

    direction:
      - "aggressive_buy": long the highest buy_ratio (sustained aggressive buying)
      - "aggressive_sell_fade": long the LOWEST buy_ratio (fade aggressive selling,
        i.e. a contrarian read -- heavy selling exhausting itself)
    """
    window = wide_ratio.loc[:as_of_date].tail(lookback_days)
    if len(window) < lookback_days:
        return pd.Series(dtype=float)
    avg = window.mean().dropna()
    if direction == "aggressive_buy":
        return avg
    elif direction == "aggressive_sell_fade":
        return -avg
    else:
        raise ValueError(f"unknown direction {direction!r}")
