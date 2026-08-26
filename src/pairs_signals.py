"""
Pairs trading / statistical arbitrage: market-neutral, not the long-only
rotation used everywhere else tonight. Genuinely different risk profile --
profits from the SPREAD between two correlated assets reverting, not from
either asset's direction.

Precise definitions:
  - Spread(t) = log(price_A(t)) - log(price_B(t))  -- a 1:1 log-ratio spread,
    not a regression-fitted hedge ratio, to keep this as few-parameter and
    verifiable as possible (a fitted beta is itself a place to overfit).
  - z-score(t) = (Spread(t) - rolling_mean(Spread, W)) / rolling_std(Spread, W)
  - Entry: |z-score| crosses above `entry_z` -- if z > entry_z, A is rich
    relative to B: short A, long B. If z < -entry_z, the reverse.
  - Exit: z-score reverts to within `exit_z` of zero, OR a stop if it keeps
    diverging past `stop_z` (the classic pairs-trading risk: the spread
    doesn't revert, it breaks structurally).

Pair selection is itself a place this can overfit -- selecting the "best"
pairs using the same data you test on guarantees something looks good by
construction. Selection uses only a TRAINING window; backtest results are
reported on the FULL period including that training window's own outcome,
so the reader can see how the selected pairs performed where they were
chosen vs. where they weren't.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class PairsConfig:
    zscore_window: int = 30       # days, rolling window for spread mean/std
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0           # divergence stop -- spread broke, cut losses
    taker_fee_pct: float = 0.001
    slippage_pct: float = 0.015
    risk_pct_per_trade: float = 0.0075
    max_concurrent_pairs: int = 4
    max_holding_days: int = None  # hard time stop -- force-exit regardless of z after this many days.
                                   # None = disabled (original behavior). A separate risk mechanism from
                                   # stop_z: caps how long a trade can wait for reversion, independent of
                                   # how far the spread has moved.


def daily_log_prices(symbol_dfs: dict) -> pd.DataFrame:
    closes = {sym: df["close"].resample("1D").last() for sym, df in symbol_dfs.items()}
    wide = pd.DataFrame(closes)
    return np.log(wide)


def select_pairs_by_correlation(log_prices: pd.DataFrame, train_end: pd.Timestamp,
                                 top_n: int = 8, min_days: int = 90) -> list:
    """Rank all possible pairs by correlation of daily log-returns over the training
    window only (log_prices up to train_end). Returns [(symA, symB, corr), ...] sorted
    by correlation descending. Does not look past train_end."""
    train = log_prices.loc[:train_end]
    rets = train.diff().dropna(how="all")
    valid_cols = [c for c in rets.columns if rets[c].notna().sum() >= min_days]
    corr_matrix = rets[valid_cols].corr()

    scored = []
    for a, b in combinations(valid_cols, 2):
        c = corr_matrix.loc[a, b]
        if pd.isna(c):
            continue
        scored.append((a, b, c))
    scored.sort(key=lambda x: -x[2])
    return scored[:top_n]


def compute_spread_zscore(log_prices: pd.DataFrame, sym_a: str, sym_b: str, window: int) -> pd.Series:
    spread = log_prices[sym_a] - log_prices[sym_b]
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z = (spread - mean) / std.replace(0, np.nan)
    return z
