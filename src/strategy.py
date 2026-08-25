"""
Coil Absorption Expansion (CAE) signal generation.

Pure price + volume. No RSI/MACD/Stochastic/Bollinger/MA-crossover/oscillators.
Only: rolling range, rolling volume averages, ATR (built from raw High/Low/Close,
which is a volatility-of-range measure, not a momentum oscillator), and candle
anatomy (open/close/high/low position).

All functions are vectorized with pandas/numpy and operate on a DataFrame with
columns: open, high, low, close, volume (indexed by open_time, ascending).
"""
import numpy as np
import pandas as pd

from config import CAEConfig


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(period, min_periods=period).mean()


def compute_signals(df: pd.DataFrame, cfg: CAEConfig) -> pd.DataFrame:
    """
    Returns df with added columns, most importantly:
      - coil_valid: bool, True if bars [i-N, i-1] form a valid coil
      - entry_signal: bool, True if bar i is a valid CAE breakout entry bar
      - coil_high, coil_low: the coil range bounds used for stop/invalidation
    """
    N = cfg.coil_lookback
    df = df.copy()

    df["atr_range"] = atr(df, cfg.atr_period_for_range)
    df["atr_stop"] = atr(df, cfg.atr_period_for_stop)
    df["atr_trail"] = atr(df, cfg.trail_atr_period)
    df["vol_sma20"] = df["volume"].rolling(cfg.vol_fade_sma_period, min_periods=cfg.vol_fade_sma_period).mean()

    # Coil window = bars [i-N .. i-1] (shifted by 1 so it excludes the current/signal bar)
    coil_high = df["high"].rolling(N).max().shift(1)
    coil_low = df["low"].rolling(N).min().shift(1)
    coil_avg_vol = df["volume"].rolling(N).mean().shift(1)
    prev_2n_avg_vol = df["volume"].rolling(2 * N).mean().shift(1 + N)

    # ATR of the coil bars themselves (avg true range over the N bars preceding signal bar)
    tr = true_range(df)
    coil_atr = tr.rolling(N).mean().shift(1)
    coil_avg_range = (df["high"] - df["low"]).rolling(N).mean().shift(1)

    df["coil_high"] = coil_high
    df["coil_low"] = coil_low
    df["coil_avg_vol"] = coil_avg_vol
    df["coil_atr"] = coil_atr
    df["coil_avg_range"] = coil_avg_range

    vol_dryup_ok = coil_avg_vol <= cfg.vol_dryup_mult * prev_2n_avg_vol

    coil_range_pct = (coil_high - coil_low) / df["close"].shift(1)
    range_ok_pct = coil_range_pct <= cfg.max_coil_range_pct
    range_ok_atr = (coil_high - coil_low) <= cfg.max_coil_range_atr_mult * df["atr_range"].shift(1)
    if cfg.max_coil_range_mode == "pct":
        range_ok = range_ok_pct
    elif cfg.max_coil_range_mode == "atr":
        range_ok = range_ok_atr
    else:  # either
        range_ok = range_ok_pct | range_ok_atr

    coil_valid = vol_dryup_ok & range_ok
    df["coil_valid"] = coil_valid.fillna(False)

    # --- Breakout trigger conditions (evaluated on the current/signal bar) ---
    if cfg.breakout_confirm_bars > 1:
        # require N consecutive closes (ending at the signal bar) all above THIS bar's
        # coil_high level -- filters single-bar spike-and-fade fakeouts using price only
        close_above_coil_high = df["close"].rolling(cfg.breakout_confirm_bars).min() > coil_high
    else:
        close_above_coil_high = df["close"] > coil_high

    if cfg.breakout_vol_ref == "sma20":
        vol_breakout_ok = df["volume"] >= cfg.breakout_vol_mult_sma20 * df["vol_sma20"]
    else:
        vol_breakout_ok = df["volume"] >= cfg.breakout_vol_mult_coil * coil_avg_vol

    bar_range = df["high"] - df["low"]
    range_expansion_ok = bar_range >= cfg.trigger_range_atr_mult * coil_atr

    bullish_candle = df["close"] > df["open"]
    upper_half = (df["close"] - df["low"]) >= 0.5 * (df["high"] - df["low"])

    entry_signal = (
        df["coil_valid"]
        & close_above_coil_high
        & vol_breakout_ok
        & range_expansion_ok
        & bullish_candle
        & upper_half
    )

    if cfg.use_upper_range_filter:
        roll_high = df["high"].rolling(cfg.upper_range_lookback, min_periods=cfg.upper_range_lookback // 2).max()
        roll_low = df["low"].rolling(cfg.upper_range_lookback, min_periods=cfg.upper_range_lookback // 2).min()
        pos_in_range = (df["close"] - roll_low) / (roll_high - roll_low).replace(0, np.nan)
        upper_range_ok = pos_in_range >= (1 - cfg.upper_range_pct)
        entry_signal = entry_signal & upper_range_ok.fillna(False)

    df["entry_signal"] = entry_signal.fillna(False)
    return df
