"""
Portfolio-level, multi-symbol, event-driven backtest engine for the
liquidity-sweep + SMT-divergence + FVG strategy.

Exit rules:
  - Hard stop: below the sweep low (with an ATR buffer), from smc_setups.
  - Target: scale out 50% at the nearest confirmed opposing swing high
    (or a 2R fallback if none exists), then trail the remainder with the
    most recent confirmed swing low formed after entry (pure structure,
    no ATR/oscillator).
  - Invalidation: close back below the FVG's low bound (gap_low) -> the
    demand zone that was supposed to hold failed -> full exit.
  - Time stop: max_hold_hours with no target hit -> exit (bounds exposure;
    this strategy has no volume-fade/coil-style rules, since it's built
    from swing/liquidity structure rather than volume dry-up).

Same realism model as the CAE engine: equity updated on realized P&L only,
0.1% taker fee + 1.5% slippage per fill, 0.75% risk/trade, notional capped
to an equal-weight slice of equity (same tight-stop hazard as CAE coils).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from smc_signals import SMCConfig, swing_points
from smc_setups import find_setups


@dataclass
class SMCEngineConfig:
    risk_pct_per_trade: float = 0.0075
    max_concurrent_positions: int = 4
    taker_fee_pct: float = 0.001
    slippage_pct: float = 0.015
    tp1_fraction: float = 0.50
    max_hold_hours: float = 96.0
    swing_k: int = 3  # must match SMCConfig.swing_k used for setup detection (for the trail)


@dataclass
class Trade:
    symbol: str
    entry_time: object
    entry_price: float
    stop_price_initial: float
    size_initial: float
    gap_low: float
    exit_events: list = field(default_factory=list)

    def realized_pnl(self, fee_pct, slip_pct):
        entry_fill = self.entry_price * (1 + slip_pct)
        pnl = 0.0
        for (_, price, frac, _reason) in self.exit_events:
            exit_fill = price * (1 - slip_pct)
            sz = self.size_initial * frac
            pnl += sz * (exit_fill - entry_fill) - exit_fill * sz * fee_pct - entry_fill * sz * fee_pct
        return pnl

    def notional_entry(self, slip_pct):
        return self.entry_price * (1 + slip_pct) * self.size_initial

    def holding_hours(self):
        if not self.exit_events:
            return None
        return (self.exit_events[-1][0] - self.entry_time) / pd.Timedelta(hours=1)

    def exit_reason_final(self):
        return self.exit_events[-1][3] if self.exit_events else None


class SymArrays:
    def __init__(self, df: pd.DataFrame, setups: list, swing_k: int):
        self.times = df.index.values.astype("datetime64[ns]").view("int64")
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.n = len(df)

        is_sw_low, _ = swing_points(df, swing_k)
        self.is_sw_low = is_sw_low  # bar i is a confirmed swing low as of i (usable from i+swing_k onward)

        self.entry_signal = np.zeros(self.n, dtype=bool)
        self.entry_price = np.full(self.n, np.nan)
        self.stop_price = np.full(self.n, np.nan)
        self.target_price = np.full(self.n, np.nan)
        self.gap_low = np.full(self.n, np.nan)
        for s in setups:
            i = s["entry_idx"]
            self.entry_signal[i] = True
            self.entry_price[i] = s["entry_price"]
            self.stop_price[i] = s["stop_price"]
            self.target_price[i] = s["target_price"]
            self.gap_low[i] = s["gap_low"]


class Position:
    __slots__ = [
        "symbol", "entry_price", "entry_time_ns", "size", "remaining_frac",
        "stop_price", "tp1_hit", "gap_low", "trade",
    ]

    def __init__(self, symbol, entry_time_ns, entry_price, size, stop_price, gap_low, trade):
        self.symbol = symbol
        self.entry_time_ns = entry_time_ns
        self.entry_price = entry_price
        self.size = size
        self.remaining_frac = 1.0
        self.stop_price = stop_price
        self.tp1_hit = False
        self.gap_low = gap_low
        self.trade = trade


HOUR_NS = 3_600_000_000_000


class SMCPortfolioBacktester:
    def __init__(self, cfg: SMCEngineConfig, symbol_setups: Dict[str, tuple], starting_equity: float = 10_000.0):
        """symbol_setups: {symbol: (df, setups_list)}"""
        self.cfg = cfg
        self.equity = starting_equity
        self.arrays: Dict[str, SymArrays] = {}
        self.target_prices_cache: Dict[str, np.ndarray] = {}
        for sym, (df, setups) in symbol_setups.items():
            self.arrays[sym] = SymArrays(df, setups, cfg.swing_k)
        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.equity_curve = []

    def run(self):
        cfg = self.cfg
        symbols = list(self.arrays.keys())
        ptr = {s: 0 for s in symbols}
        arrs = self.arrays

        all_times = np.unique(np.concatenate([a.times for a in arrs.values()]))

        for ts in all_times:
            active = []
            for s in symbols:
                a = arrs[s]
                p = ptr[s]
                if p < a.n and a.times[p] == ts:
                    active.append((s, p))

            for s, p in active:
                if s in self.open_positions:
                    self._manage_position(s, ts, p)

            if len(self.open_positions) < cfg.max_concurrent_positions:
                for s, p in active:
                    if len(self.open_positions) >= cfg.max_concurrent_positions:
                        break
                    if s in self.open_positions:
                        continue
                    a = arrs[s]
                    if not a.entry_signal[p]:
                        continue
                    self._try_open_position(s, ts, p)

            for s, p in active:
                ptr[s] = p + 1

            self.equity_curve.append((ts, self.equity))

        for sym, pos in list(self.open_positions.items()):
            a = arrs[sym]
            last_idx = a.n - 1
            self._close_remaining(sym, a.times[last_idx], a.close[last_idx], "end_of_data")

        return self.equity_curve, self.closed_trades

    def _try_open_position(self, sym, ts, p):
        cfg = self.cfg
        a = self.arrays[sym]
        entry_price = a.entry_price[p]
        stop_price = a.stop_price[p]
        gap_low = a.gap_low[p]
        if np.isnan(entry_price) or np.isnan(stop_price) or stop_price >= entry_price:
            return

        risk_amount = self.equity * cfg.risk_pct_per_trade
        stop_distance = entry_price - stop_price
        size = risk_amount / stop_distance
        if size <= 0 or not np.isfinite(size):
            return

        max_notional = self.equity / cfg.max_concurrent_positions
        size = min(size, max_notional / entry_price)
        if size <= 0:
            return

        trade = Trade(symbol=sym, entry_time=pd.Timestamp(ts), entry_price=entry_price,
                       stop_price_initial=stop_price, size_initial=size, gap_low=gap_low)
        entry_fill_price = entry_price * (1 + cfg.slippage_pct)
        self.equity -= entry_fill_price * size * cfg.taker_fee_pct

        pos = Position(sym, ts, entry_price, size, stop_price, gap_low, trade)
        self.open_positions[sym] = pos
        self.target_prices_cache[sym] = a.target_price[p]

    def _record_exit(self, pos: Position, ts, price, frac, reason):
        frac = min(frac, pos.remaining_frac)
        if frac <= 0:
            return
        cfg = self.cfg
        exit_fill_price = price * (1 - cfg.slippage_pct)
        sz = pos.size * frac
        entry_fill_price = pos.entry_price * (1 + cfg.slippage_pct)
        pnl = sz * (exit_fill_price - entry_fill_price) - exit_fill_price * sz * cfg.taker_fee_pct - entry_fill_price * sz * cfg.taker_fee_pct
        self.equity += pnl
        pos.trade.exit_events.append((pd.Timestamp(ts), price, frac, reason))
        pos.remaining_frac -= frac

    def _close_remaining(self, sym, ts, price, reason):
        pos = self.open_positions.get(sym)
        if pos is None:
            return
        if pos.remaining_frac > 0:
            self._record_exit(pos, ts, price, pos.remaining_frac, reason)
        self.closed_trades.append(pos.trade)
        del self.open_positions[sym]
        self.target_prices_cache.pop(sym, None)

    def _manage_position(self, sym, ts, p):
        cfg = self.cfg
        a = self.arrays[sym]
        pos = self.open_positions[sym]
        high, low, close = a.high[p], a.low[p], a.close[p]

        # 1) invalidation: close back below the FVG's low bound
        if close < pos.gap_low:
            self._close_remaining(sym, ts, close, "fvg_invalidated")
            return

        # 2) hard stop
        if low <= pos.stop_price:
            self._close_remaining(sym, ts, pos.stop_price, "stop")
            return

        # 3) target (scale-out), once
        if not pos.tp1_hit:
            target_price = self.target_prices_cache.get(sym)
            if target_price is not None and not np.isnan(target_price) and high >= target_price:
                self._record_exit(pos, ts, target_price, cfg.tp1_fraction, "target")
                pos.tp1_hit = True
                if pos.remaining_frac <= 0:
                    self.closed_trades.append(pos.trade)
                    del self.open_positions[sym]
                    self.target_prices_cache.pop(sym, None)
                    return

        # 4) structure trail: once target hit, trail stop up to the most recent confirmed swing low
        if pos.tp1_hit:
            confirm_idx = p - cfg.swing_k
            if confirm_idx >= 0 and a.is_sw_low[confirm_idx]:
                candidate = a.low[confirm_idx]
                if candidate > pos.entry_price and candidate > pos.stop_price:
                    pos.stop_price = candidate

        # 5) time stop
        elapsed_hours = (ts - pos.entry_time_ns) / HOUR_NS
        if elapsed_hours >= cfg.max_hold_hours:
            self._close_remaining(sym, ts, close, "time_stop")
            return
