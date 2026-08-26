"""
Funding-rate arbitrage / cash-and-carry basis trade: long spot + short the
perpetual future, dollar-neutral. Delta-neutral by construction (spot and
perp move together, so price P&L on the two legs roughly cancels) -- the
return comes purely from the funding payment, which Binance perpetuals pay
to shorts whenever the funding rate is positive. This is a structurally
different mechanism from every earlier strategy tonight: no prediction of
direction, no bet on a spread reverting -- just harvesting a persistent,
real cash flow.

Caveat stated plainly: this backtest does not model basis risk (the
difference between spot and perp price at entry/exit), since we only have
one price series per symbol and are treating it as a stand-in for both
legs. In practice spot and perp prices differ by a small, usually tight
amount; ignoring it is optimistic. It also assumes cash collateral with no
separate borrow/margin cost.
"""
import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_funding_daily(symbol: str) -> pd.Series:
    """Daily-summed funding rate for one symbol (sum of the 3 daily 8h fundings)."""
    f = os.path.join(RAW_DIR, f"{symbol}_funding.csv")
    if not os.path.exists(f):
        return pd.Series(dtype=float)
    df = pd.read_csv(f)
    if df.empty:
        return pd.Series(dtype=float)
    df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    df = df.set_index("funding_time").sort_index()
    return df["funding_rate"].resample("1D").sum()


def load_funding_wide(symbols: list) -> pd.DataFrame:
    series = {sym: load_funding_daily(sym) for sym in symbols}
    series = {k: v for k, v in series.items() if not v.empty}
    return pd.DataFrame(series)
