"""
Funding-rate ranking signal: genuinely new information (positioning/crowding),
not derivable from OHLCV price action. Two competing hypotheses, both real and
both worth testing rather than assuming one:

  - Contrarian ("squeeze"): extremely negative funding = shorts paying longs =
    crowded shorts, often a setup for a squeeze higher. Long the most-negative
    funding names.
  - Momentum-aligned ("carry"): positive funding often coincides with a real
    ongoing uptrend (longs pay because there's genuine bullish pressure). Long
    the most-positive funding names.

`direction` picks which one a given backtest run tests -- both get run and
compared honestly rather than picking the one that "should" work.
"""
import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_funding_wide(symbols=None) -> pd.DataFrame:
    """Daily-mean funding rate per symbol, wide format (columns=symbols, index=date).
    Binance funding events aren't calendar-aligned across symbols (some 1h/4h/8h
    cadence), so daily mean is the natural common granularity, matching the daily
    rebalance cadence used elsewhere in this project."""
    files = glob.glob(os.path.join(RAW_DIR, "*_funding.csv"))
    series = {}
    for f in files:
        sym = os.path.basename(f).replace("_funding.csv", "")
        if symbols is not None and sym not in symbols:
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
        df = df.set_index("funding_time").sort_index()
        daily = df["funding_rate"].resample("1D").mean()
        series[sym] = daily
    return pd.DataFrame(series)


def compute_funding_rank_at(wide_funding: pd.DataFrame, as_of_date: pd.Timestamp, lookback_days: int,
                             direction: str = "contrarian") -> pd.Series:
    """Trailing mean funding rate over `lookback_days` ending at as_of_date (inclusive) --
    as_of_date must already be a fully-closed day (enforced by caller, same convention
    as compute_returns_at in momentum_signals.py).

    Returns a series where HIGHER value = more attractive to go long, so downstream
    top-K selection logic (identical to the momentum engine's) works unchanged:
      - contrarian: negate trailing funding, so most-negative-funding symbols rank highest
      - carry: return trailing funding as-is, so most-positive-funding symbols rank highest
    """
    window = wide_funding.loc[:as_of_date].tail(lookback_days)
    if len(window) < lookback_days:
        return pd.Series(dtype=float)
    avg = window.mean()
    avg = avg.dropna()
    if direction == "contrarian":
        return -avg
    elif direction == "carry":
        return avg
    else:
        raise ValueError(f"unknown direction {direction!r}")
