"""
On-chain activity ranking signal: weekly Transfer-event count and unique
active address count per token, from Etherscan. Genuinely different from
price, volume, funding, and order-flow -- this is raw network participation,
observable on-chain before (or independent of) any price reaction.

Two hypotheses, both real and testable rather than assumed:
  - "activity surge": a spike in transfer count / unique addresses relative
    to the token's own recent baseline predicts continued upward price
    movement (growing organic interest / accumulation).
  - "activity fade": the opposite -- declining activity relative to baseline
    predicts strength (less distribution pressure, quieter accumulation).
"""
import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_onchain_wide(metric="transfer_count") -> pd.DataFrame:
    """Weekly on-chain metric per symbol, wide format, indexed by week start (UTC date)."""
    files = glob.glob(os.path.join(RAW_DIR, "*_onchain.csv"))
    series = {}
    for f in files:
        # on-chain files are saved under bare symbol names (AI), but every other
        # module (price, funding, order-flow) keys on the ticker+USDT pair (AIUSDT) --
        # normalize here so rank_fn output actually matches price-lookup columns.
        sym = os.path.basename(f).replace("_onchain.csv", "") + "USDT"
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["week_start"] = pd.to_datetime(df["week_start_ts"], unit="s", utc=True)
        df = df.set_index("week_start").sort_index()
        series[sym] = df[metric]
    return pd.DataFrame(series)


def compute_onchain_rank_at(wide_metric: pd.DataFrame, as_of_date: pd.Timestamp,
                             baseline_weeks: int = 8, direction: str = "activity_surge") -> pd.Series:
    """Ratio of the most recent value to the trailing baseline_weeks average (excluding the
    most recent week itself, to avoid the current observation dominating its own baseline).
    as_of_date must land on or after a fully-available weekly data point -- uses the last
    row with index <= as_of_date, same causality convention as the other signal modules."""
    window = wide_metric.loc[:as_of_date]
    if len(window) < baseline_weeks + 1:
        return pd.Series(dtype=float)
    latest = window.iloc[-1]
    baseline = window.iloc[-(baseline_weeks + 1):-1].mean()
    ratio = (latest / baseline.replace(0, pd.NA)).dropna()
    if direction == "activity_surge":
        return ratio
    elif direction == "activity_fade":
        return -ratio
    else:
        raise ValueError(f"unknown direction {direction!r}")
