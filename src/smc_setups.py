"""
Chains the SMC primitives into concrete trade setups:

  key-level liquidity pool (equal lows)
    -> sweep (wick below, close back above)
    -> SMT divergence vs BTC (BTC does NOT make a corresponding new low)
    -> bullish FVG forms within a watch window
    -> price retraces into the FVG -> filled entry, with stop below the
       sweep low and a target at the nearest confirmed opposing swing high

Everything is computed in a single forward pass per symbol so every decision
only uses information knowable as of that bar's close (swing points lag their
own confirmation by `swing_k` bars, matching real-time chart behavior).
"""
import numpy as np
import pandas as pd

from smc_signals import SMCConfig, swing_points, bullish_fvg, atr


def _flag_key_levels(swing_idxs, prices, tolerance_pct, lookback_bars):
    """A swing point is a 'key level' (liquidity pool) if another swing point of the
    same type occurred within tolerance_pct of it in the preceding lookback_bars.

    Sliding-window scan: swing_idxs is sorted ascending, so only swings within
    lookback_bars of the current one can ever match -- an O(n * window_size) scan,
    not O(n^2) (which is unusable at ~15-20k fractal swings per 15m symbol)."""
    n = len(swing_idxs)
    is_key = np.zeros(n, dtype=bool)
    left = 0
    for a in range(n):
        while swing_idxs[a] - swing_idxs[left] > lookback_bars:
            left += 1
        pa = prices[a]
        for b in range(left, a):
            if abs(pa - prices[b]) / pa <= tolerance_pct:
                is_key[a] = True
                is_key[b] = True
    return is_key


def find_setups(df: pd.DataFrame, ref_df: pd.DataFrame, cfg: SMCConfig):
    """df, ref_df must share the same index (ref_df pre-aligned/reindexed to df's index).
    Returns a list of dicts, one per filled setup:
      entry_idx, entry_price, stop_price, target_price, gap_low, sweep_idx
    """
    n = len(df)
    k = cfg.swing_k
    is_sw_low, is_sw_high = swing_points(df, k)
    ref_is_sw_low, _ = swing_points(ref_df, k)

    atr14 = atr(df, cfg.stop_atr_period).to_numpy()
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ref_low = ref_df["low"].to_numpy()

    fvg_flag, fvg_low, fvg_high = bullish_fvg(df)
    volume = df["volume"].to_numpy()
    vol_sma = df["volume"].rolling(cfg.displacement_vol_sma_period, min_periods=cfg.displacement_vol_sma_period).mean().to_numpy()

    sw_low_idxs = np.where(is_sw_low)[0]
    sw_low_prices = low[sw_low_idxs]
    key_low_flags = _flag_key_levels(sw_low_idxs, sw_low_prices, cfg.pool_tolerance_pct, cfg.pool_lookback_bars)
    key_low_idxs = sw_low_idxs[key_low_flags]  # only "equal lows" liquidity pools

    ref_sw_low_idxs = np.where(ref_is_sw_low)[0]

    sw_high_idxs = np.where(is_sw_high)[0]

    results = []

    # rolling pointers into the sorted key_low_idxs / ref_sw_low_idxs / sw_high_idxs arrays
    key_low_ptr = 0          # next key-low swing not yet "confirmed" (available) as of bar i
    ref_low_ptr = 0
    sw_high_ptr = 0
    available_key_lows = []   # confirmed key-level swing lows not yet swept, as (idx, price)
    available_ref_lows = []   # confirmed reference swing lows, as (idx, price)
    available_sw_highs = []   # confirmed swing highs (targets), as (idx, price)

    pending_sweep = None      # {'sweep_idx', 'pool_price', 'sweep_low_price'}
    smt_watch_until = None    # bar index deadline to find an FVG, or None
    smt_sweep_low_price = None
    smt_sweep_idx = None
    pending_fvg = None        # {'entry_level','gap_low','fill_deadline','sweep_low_price','sweep_idx'}

    for i in range(n):
        j = i - k
        # confirm new key-level swing lows knowable as of bar i
        while key_low_ptr < len(key_low_idxs) and key_low_idxs[key_low_ptr] <= j:
            idx = key_low_idxs[key_low_ptr]
            available_key_lows.append((idx, low[idx]))
            key_low_ptr += 1
        # trim old pools
        available_key_lows = [(x, p) for x, p in available_key_lows if x >= i - cfg.pool_lookback_bars]

        while ref_low_ptr < len(ref_sw_low_idxs) and ref_sw_low_idxs[ref_low_ptr] <= j:
            idx = ref_sw_low_idxs[ref_low_ptr]
            available_ref_lows.append((idx, ref_low[idx]))
            ref_low_ptr += 1
        available_ref_lows = [(x, p) for x, p in available_ref_lows if x >= i - cfg.pool_lookback_bars]

        while sw_high_ptr < len(sw_high_idxs) and sw_high_idxs[sw_high_ptr] <= j:
            idx = sw_high_idxs[sw_high_ptr]
            available_sw_highs.append((idx, high[idx]))
            sw_high_ptr += 1
        available_sw_highs = [(x, p) for x, p in available_sw_highs if x >= i - cfg.target_lookback_bars]

        # --- 1) check pending FVG fill ---
        if pending_fvg is not None:
            if i > pending_fvg["fill_deadline"]:
                pending_fvg = None
            else:
                lvl = pending_fvg["entry_level"]
                if low[i] <= lvl <= high[i]:
                    sweep_idx = pending_fvg["sweep_idx"]
                    a14 = atr14[sweep_idx]
                    buffer = cfg.stop_atr_buffer_mult * a14 if not np.isnan(a14) else pending_fvg["sweep_low_price"] * 0.01
                    stop_price = pending_fvg["sweep_low_price"] - buffer
                    risk = lvl - stop_price
                    min_target = lvl + cfg.min_rr_multiple * risk if risk > 0 else None
                    # nearest confirmed opposing swing high that still clears the min R:R bar
                    qualifying = sorted(p for (_, p) in available_sw_highs if p >= (min_target or np.inf))
                    if qualifying:
                        target_price = qualifying[0]
                    elif min_target is not None:
                        target_price = min_target  # fallback: exactly the min-R:R fixed target
                    else:
                        target_price = None
                    if target_price is not None and stop_price < lvl < target_price:
                        results.append({
                            "entry_idx": i, "entry_price": lvl, "stop_price": stop_price,
                            "target_price": target_price, "gap_low": pending_fvg["gap_low"],
                            "sweep_idx": sweep_idx,
                        })
                    pending_fvg = None
                    continue

        # --- 2) check FVG watch window ---
        if smt_watch_until is not None:
            if i > smt_watch_until:
                smt_watch_until = None
            elif fvg_flag[i] and (
                not cfg.require_displacement_volume
                or (not np.isnan(vol_sma[i - 1]) and volume[i - 1] >= cfg.displacement_vol_mult * vol_sma[i - 1])
            ):
                gl, gh = fvg_low[i], fvg_high[i]
                if cfg.fvg_entry_level == "far_edge":
                    lvl = gl
                elif cfg.fvg_entry_level == "near_edge":
                    lvl = gh
                else:
                    lvl = (gl + gh) / 2
                pending_fvg = {
                    "entry_level": lvl, "gap_low": gl,
                    "fill_deadline": i + cfg.fvg_fill_window_bars,
                    "sweep_low_price": smt_sweep_low_price, "sweep_idx": smt_sweep_idx,
                }
                smt_watch_until = None

        # --- 3) check pending sweep reclaim -> SMT check ---
        if pending_sweep is not None:
            if close[i] > pending_sweep["pool_price"]:
                sweep_idx = pending_sweep["sweep_idx"]
                prior_ref = [p for (x, p) in available_ref_lows if x < sweep_idx]
                if prior_ref:
                    ref_prior_low = prior_ref[-1]
                    w_lo = max(0, sweep_idx - cfg.smt_window_bars)
                    w_hi = min(n - 1, sweep_idx + cfg.smt_window_bars)
                    ref_min_in_window = ref_low[w_lo : w_hi + 1].min()
                    if ref_min_in_window > ref_prior_low * (1 + cfg.smt_reference_min_gap_pct):
                        smt_watch_until = i + cfg.fvg_watch_bars
                        smt_sweep_low_price = pending_sweep["sweep_low_price"]
                        smt_sweep_idx = sweep_idx
                pending_sweep = None
            elif i - pending_sweep["sweep_idx"] > cfg.sweep_reclaim_bars:
                pending_sweep = None

        # --- 4) look for a new sweep at this bar (only one setup pipeline active at a time) ---
        if pending_sweep is None and smt_watch_until is None and pending_fvg is None and available_key_lows:
            pool_idx, pool_price = available_key_lows[-1]
            min_pen = cfg.sweep_min_penetration_pct
            if low[i] < pool_price * (1 - min_pen):
                pending_sweep = {"sweep_idx": i, "pool_price": pool_price, "sweep_low_price": low[i]}
                available_key_lows.pop()  # consume this pool so it isn't re-swept repeatedly

    return results
