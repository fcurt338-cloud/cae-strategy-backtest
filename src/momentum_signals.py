"""
Cross-sectional momentum: rank the universe by trailing return, hold the
strongest names, rotate on a fixed schedule. Mechanically different from
every strategy tested so far in this project (CAE, SMT+FVG, Donchian trend)
-- those all read one coin's own chart in isolation; this ranks coins
*against each other*, which is one of the most replicated results in
quantitative finance (equities, currencies, commodities, and specifically
studied in crypto).

Precision matters here more than usual, per explicit instruction: every
rule below is stated exactly, with the causality boundary spelled out,
because an ambiguous rule silently corrupts a backtest without ever
raising an error.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MomentumConfig:
    lookback_days: int = 14       # trailing return window used for ranking
    holding_days: int = 7         # rebalance / rotation period
    top_k: int = 8                 # number of names held at once (equal-weight)
    min_lookback_history_days: int = 21  # symbol must have this much history before it's eligible at all
    taker_fee_pct: float = 0.001
    slippage_pct: float = 0.015


def daily_closes(symbol_dfs: dict) -> pd.DataFrame:
    """Resample each symbol's 1h OHLCV to daily close, aligned into one wide DataFrame.
    A day's close is only the actual daily-bar close (last 1h close of that UTC day) --
    if a day is still in progress (today, before 23:00 UTC), pandas resample('1D').last()
    would silently use a partial day. Callers must never rank on the LAST row unless
    they've confirmed that day is complete -- see build_rebalance_dates below, which only
    ever selects dates strictly in the past relative to the data's own max timestamp."""
    closes = {}
    for sym, df in symbol_dfs.items():
        daily = df["close"].resample("1D").last()
        closes[sym] = daily
    wide = pd.DataFrame(closes)
    return wide


def build_rebalance_dates(index: pd.DatetimeIndex, holding_days: int, lookback_days: int, min_history_days: int):
    """Every `holding_days`-th day, starting once enough lookback history exists.
    The LAST calendar day in `index` is excluded unless we can confirm it's a fully-closed
    day -- callers pass an index already trimmed to fully-closed days only."""
    start_idx = max(lookback_days, min_history_days)
    return index[start_idx::holding_days]


def compute_returns_at(wide_closes: pd.DataFrame, as_of_date: pd.Timestamp, lookback_days: int) -> pd.Series:
    """Trailing return ending at `as_of_date` (inclusive), using only rows with
    index <= as_of_date -- as_of_date itself must already be a fully-closed day
    (enforced by the caller via build_rebalance_dates trimming)."""
    window = wide_closes.loc[:as_of_date].tail(lookback_days + 1)
    if len(window) < lookback_days + 1:
        return pd.Series(dtype=float)
    start = window.iloc[0]
    end = window.iloc[-1]
    valid = start.notna() & end.notna() & (start > 0)
    ret = (end[valid] / start[valid]) - 1.0
    return ret
