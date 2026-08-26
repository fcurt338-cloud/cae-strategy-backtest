"""
Liquidation-wick reversal: an original strategy grounded in a specific market
mechanism, not a generic pattern. A leveraged perpetual futures market
periodically produces forced-liquidation cascades -- price moves sharply
because over-leveraged positions are being force-closed by the exchange,
not because of new information. That kind of move is mechanical: it
exhausts itself once the leveraged positions are cleared, and price tends
to snap back. A genuine information-driven move (a hack, a regulatory
shock, a real re-rating) does not behave this way -- it persists.

This matters because the project's prior single-asset mean-reversion test
(single_asset_mr_signals.py) failed at ~48% win rate: it treated EVERY
extreme multi-bar return the same, with no way to distinguish forced-flow
exhaustion from genuine directional moves. This signal targets the specific
fingerprint forced liquidation leaves in OHLCV data that a generic return
z-score does not capture:

  1. Range z-score: (high-low) unusually wide vs. this symbol's own recent
     bars -- a real spike in intrabar volatility, not just a big candle.
  2. Volume z-score: volume unusually high vs. recent bars -- confirms the
     move was accompanied by unusually heavy forced execution, not just a
     quiet drift on thin volume.
  3. Rejection fraction: the bar closed back away from its extreme by a
     meaningful fraction of its own range -- the forced flow was absorbed
     WITHIN the same bar, not merely paused. A bar that closes near its low
     (for a down-move) is continuation, not exhaustion, and must NOT
     qualify no matter how wide or high-volume it was.

All three conditions are required together. Any one alone is a much weaker,
more common, and less mechanically-specific signal (e.g. a wide-range bar
alone is just "high volatility"; a rejection alone without volume could be
routine noise, not a real forced-flow event).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LiqWickConfig:
    baseline_window_bars: int = 48   # 2 days at 1h, rolling baseline for range/volume
    entry_range_z: float = 2.0
    entry_vol_z: float = 2.0
    rejection_thresh: float = 0.66   # fraction of the bar's range reclaimed by the close
    atr_period: int = 20
    stop_atr_mult: float = 1.5
    take_profit_r_mult: float = 2.0
    max_holding_bars: int = 24       # 1 day hard time-stop
    taker_fee_pct: float = 0.0004
    slippage_pct: float = 0.0005
    risk_pct_per_trade: float = 0.0075
    max_concurrent_positions: int = 8


def compute_signal_arrays(df: pd.DataFrame, cfg: LiqWickConfig):
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    n = len(df)

    bar_range = h - l
    range_s = pd.Series(bar_range)
    range_mean = range_s.rolling(cfg.baseline_window_bars).mean()
    range_std = range_s.rolling(cfg.baseline_window_bars).std()
    range_z = ((range_s - range_mean) / range_std.replace(0, np.nan)).values

    vol_s = pd.Series(v)
    vol_mean = vol_s.rolling(cfg.baseline_window_bars).mean()
    vol_std = vol_s.rolling(cfg.baseline_window_bars).std()
    vol_z = ((vol_s - vol_mean) / vol_std.replace(0, np.nan)).values

    with np.errstate(divide="ignore", invalid="ignore"):
        rejection_up = np.where(bar_range > 0, (c - l) / bar_range, np.nan)    # close reclaimed from the low
        rejection_down = np.where(bar_range > 0, (h - c) / bar_range, np.nan)  # close reclaimed from the high

    wide_and_loud = (range_z > cfg.entry_range_z) & (vol_z > cfg.entry_vol_z)
    long_entry = wide_and_loud & (rejection_up > cfg.rejection_thresh)
    short_entry = wide_and_loud & (rejection_down > cfg.rejection_thresh)
    long_entry = np.nan_to_num(long_entry, nan=False).astype(bool)
    short_entry = np.nan_to_num(short_entry, nan=False).astype(bool)

    prev_close = np.roll(c, 1)
    prev_close[0] = np.nan
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    atr = pd.Series(tr).rolling(cfg.atr_period).mean().values

    return {
        "long_entry": long_entry, "short_entry": short_entry,
        "wick_low": l, "wick_high": h, "atr": atr,
    }
