"""
Portfolio-level, multi-symbol, event-driven backtest engine for the CAE strategy.

Performance note: with ~50 symbols x tens-of-thousands of bars each, a naive
pandas `.loc[timestamp]`-per-bar loop is too slow (tens of millions of slow
lookups). Instead each symbol's series are converted to numpy arrays once, and
the global chronological walk uses a merge-style monotonic pointer per symbol
(classic sorted-merge pattern) so every bar is visited exactly once in O(total
bars) overall.

Design notes / simplifications (documented so results aren't oversold):
  - Equity is updated on realized P&L only (at trade closes / partial scale-outs),
    not marked-to-market intrabar. Position sizing at entry uses this last realized
    equity. Standard, conservative-ish backtest simplification.
  - Intrabar fills: a bar's low is checked for stop-outs and its high for the TP1
    target; if both would trigger in the same bar we assume the stop fills first
    (conservative, since low-caps gap).
  - "Time stop ... if no further progress": operationalized as — once a position
    has been open >= time_stop_hours, exit on the first subsequent bar whose high
    is NOT a new high since entry.
  - "Volume fade exit": volume < 20-bar average AND no new high for 3 consecutive bars.
  - "Closes back inside the original coil range": close < coil_high (the breakout
    level) after entry.
  - Costs: taker fee applied to notional on each fill; slippage applied as an
    adverse price shift on each fill (buy fills higher, sell fills lower).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import CAEConfig
from strategy import compute_signals


@dataclass
class Trade:
    symbol: str
    entry_time: object
    entry_price: float
    stop_price_initial: float
    size_initial: float
    exit_events: list = field(default_factory=list)  # (time, price, size_fraction_of_initial, reason)
    coil_low: float = 0.0
    coil_high: float = 0.0

    def realized_pnl(self, fee_pct, slip_pct):
        entry_fill = self.entry_price * (1 + slip_pct)
        pnl = 0.0
        for (_, price, frac, _reason) in self.exit_events:
            exit_fill = price * (1 - slip_pct)
            sz = self.size_initial * frac
            entry_fee = entry_fill * sz * fee_pct
            exit_fee = exit_fill * sz * fee_pct
            pnl += sz * (exit_fill - entry_fill) - entry_fee - exit_fee
        return pnl

    def notional_entry(self, slip_pct):
        return self.entry_price * (1 + slip_pct) * self.size_initial

    def holding_hours(self):
        if not self.exit_events:
            return None
        last_time = self.exit_events[-1][0]
        return (last_time - self.entry_time) / pd.Timedelta(hours=1)

    def exit_reason_final(self):
        return self.exit_events[-1][3] if self.exit_events else None


class SymArrays:
    """Numpy views of a symbol's signal-computed dataframe, for fast indexed access."""
    def __init__(self, df: pd.DataFrame):
        self.times = df.index.values.astype("datetime64[ns]").view("int64")
        self.index = df.index
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.volume = df["volume"].to_numpy(dtype=float)
        entry_signal = df["entry_signal"].to_numpy(dtype=bool)
        if "liquidity_ok" in df.columns:
            entry_signal = entry_signal & df["liquidity_ok"].fillna(False).to_numpy(dtype=bool)
        self.entry_signal = entry_signal
        self.coil_high = df["coil_high"].to_numpy(dtype=float)
        self.coil_low = df["coil_low"].to_numpy(dtype=float)
        self.atr_stop = df["atr_stop"].to_numpy(dtype=float)
        self.atr_trail = df["atr_trail"].to_numpy(dtype=float)
        self.vol_sma20 = df["vol_sma20"].to_numpy(dtype=float)
        self.n = len(df)


class Position:
    __slots__ = [
        "symbol", "entry_price", "entry_time_ns", "size", "remaining_frac",
        "stop_price", "tp1_hit", "coil_high", "coil_low", "max_high_since_entry",
        "no_new_high_streak", "time_stop_active", "trade", "bars_held",
    ]

    def __init__(self, symbol, entry_time_ns, entry_price, size, stop_price, coil_high, coil_low, trade):
        self.symbol = symbol
        self.entry_time_ns = entry_time_ns
        self.entry_price = entry_price
        self.size = size
        self.remaining_frac = 1.0
        self.stop_price = stop_price
        self.tp1_hit = False
        self.coil_high = coil_high
        self.coil_low = coil_low
        self.max_high_since_entry = entry_price
        self.no_new_high_streak = 0
        self.time_stop_active = False
        self.trade = trade
        self.bars_held = 0


HOUR_NS = 3_600_000_000_000


class PortfolioBacktester:
    def __init__(self, cfg: CAEConfig, symbol_dfs: Dict[str, pd.DataFrame], btc_df: Optional[pd.DataFrame] = None,
                 starting_equity: float = 10_000.0):
        self.cfg = cfg
        self.starting_equity = starting_equity
        self.equity = starting_equity
        self.arrays: Dict[str, SymArrays] = {}
        for sym, df in symbol_dfs.items():
            sig = compute_signals(df, cfg)
            self.arrays[sym] = SymArrays(sig)
        self.btc_arr = None
        if btc_df is not None and cfg.use_btc_filter:
            btc = btc_df.copy()
            btc["ret_1h"] = btc["close"].pct_change()
            self.btc_times = btc.index.values.astype("datetime64[ns]").view("int64")
            self.btc_ret = btc["ret_1h"].to_numpy(dtype=float)
        else:
            self.btc_times = None
            self.btc_ret = None

        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.equity_curve = []  # (time_ns, equity)

    def _btc_blocks(self, ts_ns, btc_ptr):
        if self.btc_times is None:
            return False, btc_ptr
        n = len(self.btc_times)
        while btc_ptr < n and self.btc_times[btc_ptr] < ts_ns:
            btc_ptr += 1
        if btc_ptr < n and self.btc_times[btc_ptr] == ts_ns:
            r = self.btc_ret[btc_ptr]
            if not np.isnan(r) and r <= -self.cfg.btc_drop_threshold_pct:
                return True, btc_ptr
        return False, btc_ptr

    def run(self):
        cfg = self.cfg
        symbols = list(self.arrays.keys())
        ptr = {s: 0 for s in symbols}
        arrs = self.arrays

        all_times = np.unique(np.concatenate([a.times for a in arrs.values()]))
        btc_ptr = 0

        for ts in all_times:
            # advance pointers to symbols with a bar exactly at ts, collecting them
            active_syms = []
            for s in symbols:
                a = arrs[s]
                p = ptr[s]
                if p < a.n and a.times[p] == ts:
                    active_syms.append((s, p))

            # 1) manage existing open positions with a bar this timestamp
            for s, p in active_syms:
                if s in self.open_positions:
                    self._manage_position(s, ts, p)

            # 2) consider new entries
            if len(self.open_positions) < cfg.max_concurrent_positions:
                blocked, btc_ptr = self._btc_blocks(ts, btc_ptr)
                if not blocked:
                    for s, p in active_syms:
                        if len(self.open_positions) >= cfg.max_concurrent_positions:
                            break
                        if s in self.open_positions:
                            continue
                        a = arrs[s]
                        if not a.entry_signal[p]:
                            continue
                        self._try_open_position(s, ts, p)

            # advance pointers past processed bars
            for s, p in active_syms:
                ptr[s] = p + 1

            self.equity_curve.append((ts, self.equity))

        # Force-close any positions still open at end of data
        for sym, pos in list(self.open_positions.items()):
            a = arrs[sym]
            last_idx = a.n - 1
            self._close_remaining(sym, a.times[last_idx], a.close[last_idx], "end_of_data")

        return self.equity_curve, self.closed_trades

    def _try_open_position(self, sym, ts, p):
        cfg = self.cfg
        a = self.arrays[sym]

        if cfg.entry_fill == "next_open":
            if p + 1 >= a.n:
                return
            entry_price = a.open[p + 1]
            ts_fill = a.times[p + 1]
        else:
            entry_price = a.close[p]
            ts_fill = ts

        if not np.isfinite(entry_price) or entry_price <= 0:
            return

        coil_low = a.coil_low[p]
        coil_high = a.coil_high[p]
        atr_stop = a.atr_stop[p]
        if np.isnan(coil_low) or np.isnan(atr_stop):
            return

        if cfg.stop_mode == "fixed15":
            stop_price = entry_price * (1 - cfg.fixed_stop_pct)
        else:
            structure_stop = coil_low
            atr_stop_price = entry_price - cfg.stop_atr_mult * atr_stop
            stop_price = max(structure_stop, atr_stop_price)  # tighter of the two

        if stop_price >= entry_price:
            return

        risk_amount = self.equity * cfg.risk_pct_per_trade
        stop_distance = entry_price - stop_price
        size = risk_amount / stop_distance
        if size <= 0 or not np.isfinite(size):
            return

        # Cap notional: on a tight coil the structural/ATR stop can be only ~1-2% away,
        # which would make pure risk_amount/stop_distance sizing put a huge fraction of
        # equity (or, across positions, more than 100% of equity) into one low-cap spot
        # position. This is a spot account, not leveraged — no single position should be
        # able to exceed an equal-weight slice of equity, and this also naturally throttles
        # size when stops are unrealistically tight relative to fee+slippage costs.
        max_notional = self.equity / cfg.max_concurrent_positions
        max_size = max_notional / entry_price
        size = min(size, max_size)
        if size <= 0 or not np.isfinite(size):
            return

        trade = Trade(
            symbol=sym, entry_time=pd.Timestamp(ts_fill), entry_price=entry_price,
            stop_price_initial=stop_price, size_initial=size,
            coil_low=coil_low, coil_high=coil_high,
        )
        entry_fill_price = entry_price * (1 + cfg.slippage_pct)
        entry_fee = entry_fill_price * size * cfg.taker_fee_pct
        self.equity -= entry_fee

        pos = Position(sym, ts_fill, entry_price, size, stop_price, coil_high, coil_low, trade)
        self.open_positions[sym] = pos

    def _record_exit(self, pos: Position, ts, price, frac, reason):
        frac = min(frac, pos.remaining_frac)
        if frac <= 0:
            return
        cfg = self.cfg
        exit_fill_price = price * (1 - cfg.slippage_pct)
        sz = pos.size * frac
        exit_fee = exit_fill_price * sz * cfg.taker_fee_pct
        entry_fill_price = pos.entry_price * (1 + cfg.slippage_pct)
        entry_fee_alloc = entry_fill_price * sz * cfg.taker_fee_pct
        pnl = sz * (exit_fill_price - entry_fill_price) - exit_fee - entry_fee_alloc
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

    def _manage_position(self, sym, ts, p):
        cfg = self.cfg
        a = self.arrays[sym]
        pos = self.open_positions[sym]

        pos.bars_held += 1
        high, low, close, volume = a.high[p], a.low[p], a.close[p], a.volume[p]
        made_new_high = high > pos.max_high_since_entry
        if made_new_high:
            pos.max_high_since_entry = high
            pos.no_new_high_streak = 0
        else:
            pos.no_new_high_streak += 1

        # 1) Coil re-entry invalidation
        if close < pos.coil_high:
            self._close_remaining(sym, ts, close, "coil_reentry")
            return

        # 2) Hard stop (intrabar low touch)
        if low <= pos.stop_price:
            self._close_remaining(sym, ts, pos.stop_price, "stop")
            return

        # 3) TP1 (intrabar high touch), only once
        if not pos.tp1_hit:
            tp1_price = pos.entry_price * (1 + cfg.tp1_pct)
            if high >= tp1_price:
                self._record_exit(pos, ts, tp1_price, cfg.tp1_fraction, "tp1")
                pos.tp1_hit = True
                if pos.remaining_frac <= 0:
                    self.closed_trades.append(pos.trade)
                    del self.open_positions[sym]
                    return

        # 4) Trailing stop update (applies once TP1 hit)
        if pos.tp1_hit:
            if cfg.trail_mode == "swing_low":
                start = max(0, p - cfg.swing_low_lookback)
                new_stop = a.low[start:p].min() if p > start else pos.stop_price
            else:
                atr_trail = a.atr_trail[p]
                new_stop = close - cfg.trail_atr_mult * atr_trail if not np.isnan(atr_trail) else pos.stop_price
            pos.stop_price = max(pos.stop_price, new_stop)

        past_grace_period = pos.bars_held >= cfg.min_hold_bars_before_fade_or_time_exit

        # 5) Time stop
        if past_grace_period:
            elapsed_hours = (ts - pos.entry_time_ns) / HOUR_NS
            if elapsed_hours >= cfg.time_stop_hours:
                if pos.time_stop_active:
                    if not made_new_high:
                        self._close_remaining(sym, ts, close, "time_stop")
                        return
                else:
                    pos.time_stop_active = True

        # 6) Volume fade exit
        if past_grace_period:
            vol_sma20 = a.vol_sma20[p]
            if not np.isnan(vol_sma20) and volume < vol_sma20 and pos.no_new_high_streak >= cfg.vol_fade_no_new_high_bars:
                self._close_remaining(sym, ts, close, "volume_fade")
                return
