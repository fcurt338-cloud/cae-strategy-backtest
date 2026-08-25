"""Tunable parameters for the Coil Absorption Expansion (CAE) strategy."""
from dataclasses import dataclass, field


@dataclass
class CAEConfig:
    # --- Coil detection ---
    coil_lookback: int = 24          # N bars; 24 default on 15m, 12-18 on 1h
    vol_dryup_mult: float = 0.65     # avg(vol, last N) <= mult * avg(vol, prev 2N)
    max_coil_range_mode: str = "either"   # "pct" | "atr" | "either" (pass if EITHER condition true)
    max_coil_range_pct: float = 0.10      # (HH-LL)/close <= 10%
    max_coil_range_atr_mult: float = 1.5  # (HH-LL) <= 1.5 * ATR(20)
    atr_period_for_range: int = 20

    # --- Breakout trigger ---
    breakout_vol_ref: str = "coil_avg"    # "coil_avg" (>=3.0x coil avg vol) | "sma20" (>=2.5x 20-bar vol SMA)
    breakout_vol_mult_coil: float = 3.0
    breakout_vol_mult_sma20: float = 2.5
    trigger_range_atr_mult: float = 1.5   # trigger bar range >= 1.5x ATR of coil bars

    # --- Optional filters ---
    use_upper_range_filter: bool = False
    upper_range_lookback: int = 7 * 96    # in bars; caller should pass bars-per-day aware value
    upper_range_pct: float = 0.40         # price must be in upper 40% of that range

    # --- Entry ---
    entry_fill: str = "close"        # "close" (enter at signal bar close) | "next_open"
    breakout_confirm_bars: int = 1   # require this many consecutive closes above coil_high (1 = literal spec)

    # --- Exit: stop ---
    stop_mode: str = "structure"     # "structure" (max(coil_low, entry-1.8*ATR14)) | "fixed15"
    stop_atr_mult: float = 1.8
    atr_period_for_stop: int = 14
    fixed_stop_pct: float = 0.15

    # --- Exit: take profit / trailing ---
    tp1_pct: float = 0.40            # scale out 50% at +40%
    tp1_fraction: float = 0.50
    trail_mode: str = "atr"          # "atr" (2x ATR trail) | "swing_low"
    trail_atr_mult: float = 2.0
    trail_atr_period: int = 14
    swing_low_lookback: int = 5

    # --- Exit: time stop ---
    time_stop_hours: float = 48.0

    # --- Exit: volume fade ---
    vol_fade_sma_period: int = 20
    vol_fade_no_new_high_bars: int = 3

    # --- Refinement knob (default 0 = literal spec behavior): bars after entry during
    # which volume_fade/time_stop exits are suppressed, giving a breakout room to develop
    # before the "no progress" exits can fire. Coil-reentry and hard stop are unaffected.
    min_hold_bars_before_fade_or_time_exit: int = 0

    # --- Risk / position management ---
    risk_pct_per_trade: float = 0.0075
    max_concurrent_positions: int = 4
    taker_fee_pct: float = 0.001
    slippage_pct: float = 0.015

    # --- Market filter ---
    use_btc_filter: bool = False
    btc_drop_threshold_pct: float = 0.035   # skip new entries if BTC 1h return <= -3.5%

    # --- Bars-per-day, set per timeframe for volume/time math ---
    bars_per_day: int = 96           # 96 for 15m, 24 for 1h
    timeframe: str = "15m"

    def clone(self, **overrides):
        import dataclasses
        return dataclasses.replace(self, **overrides)


def default_config_for_timeframe(tf: str) -> CAEConfig:
    if tf == "15m":
        return CAEConfig(coil_lookback=24, bars_per_day=96, timeframe="15m",
                          upper_range_lookback=7 * 96)
    elif tf == "1h":
        return CAEConfig(coil_lookback=15, bars_per_day=24, timeframe="1h",
                          upper_range_lookback=7 * 24)
    else:
        raise ValueError(f"unknown timeframe {tf}")
