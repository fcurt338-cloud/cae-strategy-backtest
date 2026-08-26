"""
Single-asset short-term mean reversion: after an unusually sharp move in ONE
asset (relative to its OWN recent volatility, not relative to another asset),
does price tend to partially revert over the following hours/days? This is a
structurally different mechanism from everything else tested in this project:
  - not a breakout/continuation bet (CAE, SMC, Donchian trend -- all failed)
  - not a cross-asset relative-value bet (pairs trading -- real edge, but
    lost to tail risk after 5 different fix attempts)
It's the classic "short-term reversal" effect documented in equities and
futures markets, tested here on crypto majors for the first time this
project (majors, not low-caps -- low-cap breakout strategies failed WORSE
than majors' equivalents all project, so there's no reason to expect
low-caps would be better for this either).

Signal: rolling z-score of a short trailing return, normalized by that same
asset's own rolling return volatility (adaptive per-symbol, not a fixed %
threshold that would mean very different things for BTC vs a small alt).

  ret_N(t)  = log(close(t)) - log(close(t-N))
  z(t)      = (ret_N(t) - rolling_mean(ret_N, W)) / rolling_std(ret_N, W)

Entry: |z| > entry_z -- CONTRARIAN (long after an extreme down move, short
after an extreme up move). Exit: |z| back within exit_z (reverted), OR a
hard time-stop, OR an ATR stop-loss -- unlike a pairs spread, a single
asset's momentum CAN keep going rather than reverting, so a real stop is
required from the start (lesson from every earlier strategy: don't bolt on
risk control after the fact, design it in).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SingleAssetMRConfig:
    return_lookback_bars: int = 12   # ~12h trailing return
    zscore_window_bars: int = 720    # ~30 days of 1h bars, for normalizing "how extreme"
    entry_z: float = 2.0
    exit_z: float = 0.5
    atr_period: int = 20
    stop_atr_mult: float = 2.0
    max_holding_bars: int = 48       # ~2 days hard time-stop, learned from pairs trading
    taker_fee_pct: float = 0.0004
    slippage_pct: float = 0.0005
    risk_pct_per_trade: float = 0.0075
    max_concurrent_positions: int = 8


def compute_signal_arrays(df: pd.DataFrame, cfg: SingleAssetMRConfig):
    """df must have 'close','high','low' columns, 1h bars. Returns a dict of aligned
    numpy arrays: z (reversion signal), atr (for stop sizing)."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    log_close = np.log(close)
    ret_n = np.full(len(close), np.nan)
    ret_n[cfg.return_lookback_bars:] = log_close[cfg.return_lookback_bars:] - log_close[:-cfg.return_lookback_bars]

    ret_series = pd.Series(ret_n)
    roll_mean = ret_series.rolling(cfg.zscore_window_bars).mean()
    roll_std = ret_series.rolling(cfg.zscore_window_bars).std()
    z = (ret_series - roll_mean) / roll_std.replace(0, np.nan)

    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(cfg.atr_period).mean()

    return {"z": z.values, "atr": atr.values, "close": close}
