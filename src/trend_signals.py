"""
Donchian-channel trend-following breakout: the actual, historically-documented
"Turtle Trader" system, not a repackaged chart pattern.

Core idea: most breakout attempts fail (win rate in the teens/twenties is
normal and expected for this style), so the entire edge comes from payoff
asymmetry -- cut losers small and fast, and never cap a winner with a fixed
profit target. The only exit is a trailing channel; it only ratchets in the
trader's favor.

Entry: close breaks above the highest high of the preceding `entry_lookback`
bars (a closing breakout, slightly more robust to single-wick noise than an
intrabar trigger).
Initial stop: entry - stop_atr_mult * ATR(atr_period).
Trailing stop: ratchets up to the lowest low of the preceding `exit_lookback`
bars once that level is higher than the current stop -- never loosens.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smc_signals import atr


@dataclass
class TrendConfig:
    entry_lookback: int = 480   # bars; ~20 days on 1h, translate per timeframe
    exit_lookback: int = 240    # bars; ~10 days on 1h
    stop_atr_mult: float = 2.0
    atr_period: int = 20
    entry_fill: str = "close"   # "close" | "next_open"
    max_hold_bars: int = 0      # 0 = no outer time cap (pure trend system)

    # Regime filter: skip entries when price is too far below its own trailing
    # long-term high (severe structural breakdown) -- 0 disables the filter.
    regime_lookback_bars: int = 0
    regime_min_pct_of_high: float = 0.4


def compute_trend_arrays(df: pd.DataFrame, cfg: TrendConfig):
    """Returns arrays aligned to df's index:
      entry_signal: bool, True if bar i is a valid breakout entry bar
      initial_stop: the ATR-based initial stop as of bar i (for use if bar i is an entry)
      trail_ref: rolling lowest-low of the preceding exit_lookback bars, usable each bar
                 for trailing-stop management once in a position.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    entry_channel_high = high.rolling(cfg.entry_lookback).max().shift(1)
    entry_signal = (close > entry_channel_high).fillna(False)

    if cfg.regime_lookback_bars > 0:
        trailing_high = high.rolling(cfg.regime_lookback_bars, min_periods=cfg.regime_lookback_bars // 2).max().shift(1)
        regime_ok = (close >= cfg.regime_min_pct_of_high * trailing_high).fillna(False)
        entry_signal = entry_signal & regime_ok

    a = atr(df, cfg.atr_period)
    initial_stop = close - cfg.stop_atr_mult * a

    trail_ref = low.rolling(cfg.exit_lookback).min().shift(1)

    return {
        "entry_signal": entry_signal.to_numpy(dtype=bool),
        "initial_stop": initial_stop.to_numpy(dtype=float),
        "trail_ref": trail_ref.to_numpy(dtype=float),
    }
