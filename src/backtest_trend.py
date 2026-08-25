"""
Portfolio-level, multi-symbol, event-driven backtest engine for the Donchian
trend-following system. Deliberately the simplest exit logic of the three
strategies tested in this project: hard initial stop, then a trailing channel
stop that only ratchets in the position's favor. No fixed profit target, no
partial scale-outs, no pattern-based invalidation rules -- the asymmetry is
the whole point.

Same realism model as the other two engines: equity updated on realized P&L
only, 0.1% taker fee + 1.5% slippage per fill, 0.75% risk/trade, notional
capped to an equal-weight slice of equity.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from trend_signals import TrendConfig, compute_trend_arrays


@dataclass
class TrendEngineConfig:
    risk_pct_per_trade: float = 0.0075
    max_concurrent_positions: int = 4
    taker_fee_pct: float = 0.001
    slippage_pct: float = 0.015


@dataclass
class Trade:
    symbol: str
    entry_time: object
    entry_price: float
    stop_price_initial: float
    size_initial: float
    exit_events: list = field(default_factory=list)
    max_favorable_price: float = 0.0

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

    def r_multiple(self, fee_pct, slip_pct):
        risk = self.entry_price - self.stop_price_initial
        if risk <= 0:
            return None
        return self.realized_pnl(fee_pct, slip_pct) / (risk * self.size_initial)


class SymArrays:
    def __init__(self, df: pd.DataFrame, cfg: TrendConfig):
        self.times = df.index.values.astype("datetime64[ns]").view("int64")
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.n = len(df)
        arrs = compute_trend_arrays(df, cfg)
        self.entry_signal = arrs["entry_signal"]
        self.initial_stop = arrs["initial_stop"]
        self.trail_ref = arrs["trail_ref"]


class Position:
    __slots__ = ["symbol", "entry_price", "entry_idx", "size", "stop_price", "trade", "bars_held"]

    def __init__(self, symbol, entry_price, entry_idx, size, stop_price, trade):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_idx = entry_idx
        self.size = size
        self.stop_price = stop_price
        self.trade = trade
        self.bars_held = 0


class TrendPortfolioBacktester:
    def __init__(self, cfg: TrendEngineConfig, symbol_dfs: Dict[str, pd.DataFrame], trend_cfg: TrendConfig,
                 starting_equity: float = 10_000.0):
        self.cfg = cfg
        self.trend_cfg = trend_cfg
        self.equity = starting_equity
        self.arrays: Dict[str, SymArrays] = {sym: SymArrays(df, trend_cfg) for sym, df in symbol_dfs.items()}
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
            self._close_position(sym, a.times[last_idx], a.close[last_idx], "end_of_data")

        return self.equity_curve, self.closed_trades

    def _try_open_position(self, sym, ts, p):
        cfg = self.cfg
        tcfg = self.trend_cfg
        a = self.arrays[sym]

        if tcfg.entry_fill == "next_open":
            if p + 1 >= a.n:
                return
            entry_price = a.open[p + 1]
            ts_fill = a.times[p + 1]
            entry_idx = p + 1
        else:
            entry_price = a.close[p]
            ts_fill = ts
            entry_idx = p

        stop_price = a.initial_stop[p]
        if not np.isfinite(entry_price) or not np.isfinite(stop_price) or stop_price >= entry_price:
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

        trade = Trade(symbol=sym, entry_time=pd.Timestamp(ts_fill), entry_price=entry_price,
                       stop_price_initial=stop_price, size_initial=size, max_favorable_price=entry_price)
        entry_fill_price = entry_price * (1 + cfg.slippage_pct)
        self.equity -= entry_fill_price * size * cfg.taker_fee_pct

        pos = Position(sym, entry_price, entry_idx, size, stop_price, trade)
        self.open_positions[sym] = pos

    def _close_position(self, sym, ts, price, reason):
        cfg = self.cfg
        pos = self.open_positions.get(sym)
        if pos is None:
            return
        exit_fill_price = price * (1 - cfg.slippage_pct)
        entry_fill_price = pos.entry_price * (1 + cfg.slippage_pct)
        pnl = pos.size * (exit_fill_price - entry_fill_price) - exit_fill_price * pos.size * cfg.taker_fee_pct \
            - entry_fill_price * pos.size * cfg.taker_fee_pct
        self.equity += pnl
        pos.trade.exit_events.append((pd.Timestamp(ts), price, 1.0, reason))
        self.closed_trades.append(pos.trade)
        del self.open_positions[sym]

    def _manage_position(self, sym, ts, p):
        a = self.arrays[sym]
        pos = self.open_positions[sym]
        pos.bars_held += 1
        high, low = a.high[p], a.low[p]

        pos.trade.max_favorable_price = max(pos.trade.max_favorable_price, high)

        # hard/trailing stop (unified: trail only ratchets stop_price upward)
        if low <= pos.stop_price:
            self._close_position(sym, ts, pos.stop_price, "trail_stop" if pos.bars_held > 1 else "stop")
            return

        trail_candidate = a.trail_ref[p]
        if not np.isnan(trail_candidate) and trail_candidate > pos.stop_price:
            pos.stop_price = trail_candidate

        if self.trend_cfg.max_hold_bars and pos.bars_held >= self.trend_cfg.max_hold_bars:
            self._close_position(sym, ts, a.close[p], "time_stop")
