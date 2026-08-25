"""
Smart-money-concept primitives: swing structure, liquidity pools, sweeps,
SMT divergence, and Fair Value Gaps. Pure price (and volume where noted) —
no oscillators, no moving averages.

Everything here respects causality: a value at bar i only uses information
knowable at bar i's close (swing points are only confirmed `swing_k` bars
after they occur, and that lag is preserved explicitly).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SMCConfig:
    swing_k: int = 3                  # bars each side to confirm a fractal swing point
    pool_tolerance_pct: float = 0.006  # swing lows/highs within this %% cluster into one pool
    pool_lookback_bars: int = 300      # how far back to look for clusterable swings
    sweep_reclaim_bars: int = 3        # must close back above pool within this many bars
    sweep_min_penetration_pct: float = 0.0  # min wick-below-pool depth as %% of price (0 = any)
    smt_window_bars: int = 4           # +/- bars to match reference asset's swing to the sweep time
    smt_reference_min_gap_pct: float = 0.0  # how far reference's low must stay above ITS prior low
    fvg_watch_bars: int = 20           # bars after sweep to watch for a bullish FVG to form
    fvg_fill_window_bars: int = 40     # bars after FVG forms for price to retrace into it and fill
    fvg_entry_level: str = "midpoint"  # "midpoint" | "far_edge" (candle3 low) | "near_edge" (candle1 high)
    stop_atr_buffer_mult: float = 0.5
    stop_atr_period: int = 14
    target_lookback_bars: int = 400    # how far ahead of entry to search for an opposing liquidity pool
    min_rr_multiple: float = 1.5       # skip setups whose best available target gives less than this R:R
    require_displacement_volume: bool = False  # quality filter: displacement candle needs real volume
    displacement_vol_mult: float = 1.5
    displacement_vol_sma_period: int = 20


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    return true_range(df).rolling(period, min_periods=period).mean()


def swing_points(df: pd.DataFrame, k: int):
    """Fractal swing highs/lows: bar i is a swing low if its low is the min of [i-k, i+k].
    Returned as boolean arrays aligned to bar i, but only "knowable" (usable in a signal)
    from bar i+k onward -- callers must respect that lag themselves."""
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    n = len(df)
    is_swing_low = np.zeros(n, dtype=bool)
    is_swing_high = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        window_low = low[i - k : i + k + 1]
        window_high = high[i - k : i + k + 1]
        if low[i] == window_low.min() and (window_low == low[i]).sum() == 1:
            is_swing_low[i] = True
        if high[i] == window_high.max() and (window_high == high[i]).sum() == 1:
            is_swing_high[i] = True
    return is_swing_low, is_swing_high


def cluster_liquidity_pools(times, prices, tolerance_pct):
    """Greedy clustering of swing prices that are within tolerance_pct of each other.
    Returns list of dicts: {price (mean), first_idx, last_idx, touches}."""
    order = np.argsort(prices)
    pools = []
    used = np.zeros(len(prices), dtype=bool)
    for oi in order:
        if used[oi]:
            continue
        p = prices[oi]
        members = [oi]
        used[oi] = True
        for oj in order:
            if used[oj]:
                continue
            if abs(prices[oj] - p) / p <= tolerance_pct:
                members.append(oj)
                used[oj] = True
        idxs = sorted(members)
        pools.append({
            "price": float(np.mean([prices[m] for m in members])),
            "idxs": idxs,
            "touches": len(idxs),
        })
    return pools


def bullish_fvg(df: pd.DataFrame):
    """Standard 3-candle bullish FVG: high[i-2] < low[i], gap = (high[i-2], low[i]).
    Available/knowable as of bar i's close (no lookahead)."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    gap_low = np.full(n, np.nan)
    gap_high = np.full(n, np.nan)
    is_fvg = np.zeros(n, dtype=bool)
    for i in range(2, n):
        c1_high = high[i - 2]
        c3_low = low[i]
        if c3_low > c1_high:
            is_fvg[i] = True
            gap_low[i] = c1_high
            gap_high[i] = c3_low
    return is_fvg, gap_low, gap_high
