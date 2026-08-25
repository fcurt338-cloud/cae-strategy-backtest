"""
New-listing effect: an event-driven signal, not a rolling one -- mechanically
different from all seven prior tests, which all ranked/scored symbols
continuously. Here the event is fixed (each symbol's own listing date) and
the question is what happens in the days/weeks after.

Two well-documented, competing hypotheses in crypto specifically (both
stated precisely, tested as-is):
  - "initial pump fade": listings often see an initial speculative pop that
    fades as early unlocks/market-making inventory gets distributed --
    predicts a SHORT opportunity, or for a long-only book, an AVOID window.
  - "post-dump recovery": after the initial volatility and any pump-fade
    settles (a fixed number of days post-listing), a token that's held a
    floor and shows renewed volume may be entering genuine price discovery
    -- predicts a LONG entry window some fixed number of days after listing,
    not at the listing itself.

This tests the second hypothesis: enter long at a fixed offset after each
symbol's own first-available bar, hold for a fixed period, no per-symbol
lookahead (each symbol's own listing date is used, which is legitimately
knowable in real time -- you know a coin listed the day it lists).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ListingEffectConfig:
    entry_offset_days: int = 21     # enter this many days after the symbol's first bar
    holding_days: int = 14          # hold for this long
    min_avg_dollar_volume: float = 100_000  # liquidity floor over the holding window, using close*volume
    stop_pct: float = 0.30          # hard stop, simple fixed-percent (not ATR -- deliberately minimal rules)
    taker_fee_pct: float = 0.001
    slippage_pct: float = 0.015
    risk_pct_per_trade: float = 0.0075
    max_concurrent_positions: int = 4
