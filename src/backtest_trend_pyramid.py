"""
Trend engine with pyramiding: adds a fixed-risk unit each time price advances
`add_atr_step` * ATR-at-entry beyond the last add, up to `max_units`. All units
in a position share one common trailing stop (the same Donchian-channel trail
as the base engine) -- when it's hit, everything exits together. This is the
classic Turtle unit-adding mechanism: the position itself grows into a strong
trend instead of being sized once at entry and left alone.

Reuses trend_signals.py unchanged. Deliberately a separate module from
backtest_trend.py rather than bolted onto it, so the (already-checked) base
engine stays simple and this more complex variant can be validated on its own.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from trend_signals import TrendConfig, compute_trend_arrays
from smc_signals import atr as atr_series


@dataclass
class PyramidEngineConfig:
    risk_pct_per_trade: float = 0.0075   # risk per UNIT (not per whole position)
    max_concurrent_positions: int = 4
    taker_fee_pct: float = 0.001
    slippage_pct: float = 0.015
    max_units: int = 4
    add_atr_step: float = 0.5            # add a unit every 0.5*ATR(at entry) of favorable move
    atr_period_for_add: int = 20


@dataclass
class Trade:
    symbol: str
    entry_time: object
    entry_price: float           # price of the FIRST unit
    stop_price_initial: float
    size_initial: float          # size of the first unit only (for reference)
    total_size: float = 0.0      # filled in at close: sum of all units
    n_units: int = 1
    exit_events: list = field(default_factory=list)
    unit_entries: list = field(default_factory=list)  # [(time, price, size)]

    def realized_pnl(self, fee_pct, slip_pct):
        # All units exit together at the same price/time in this design (frac is always
        # 1.0 in practice), so P&L is just each unit's own entry vs. that shared exit fill.
        pnl = 0.0
        for (_, price, frac, _reason) in self.exit_events:
            exit_fill = price * (1 - slip_pct)
            for (_, entry_price, unit_size) in self.unit_entries:
                entry_fill = entry_price * (1 + slip_pct)
                sz = unit_size * frac
                pnl += sz * (exit_fill - entry_fill) - exit_fill * sz * fee_pct - entry_fill * sz * fee_pct
        return pnl

    def notional_entry(self, slip_pct):
        return sum(price * (1 + slip_pct) * size for (_, price, size) in self.unit_entries)

    def holding_hours(self):
        if not self.exit_events:
            return None
        return (self.exit_events[-1][0] - self.entry_time) / pd.Timedelta(hours=1)

    def exit_reason_final(self):
        return self.exit_events[-1][3] if self.exit_events else None


class SymArrays:
    def __init__(self, df: pd.DataFrame, cfg: TrendConfig, add_atr_period: int):
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
        self.atr_add = atr_series(df, add_atr_period).to_numpy(dtype=float)


class Position:
    __slots__ = ["symbol", "stop_price", "trade", "n_units", "last_add_price", "add_step"]

    def __init__(self, symbol, stop_price, trade, first_unit_price, add_step):
        self.symbol = symbol
        self.stop_price = stop_price
        self.trade = trade
        self.n_units = 1
        self.last_add_price = first_unit_price
        self.add_step = add_step


class PyramidPortfolioBacktester:
    def __init__(self, cfg: PyramidEngineConfig, symbol_dfs: Dict[str, pd.DataFrame], trend_cfg: TrendConfig,
                 starting_equity: float = 10_000.0):
        self.cfg = cfg
        self.trend_cfg = trend_cfg
        self.equity = starting_equity
        self.arrays: Dict[str, SymArrays] = {
            sym: SymArrays(df, trend_cfg, cfg.atr_period_for_add) for sym, df in symbol_dfs.items()
        }
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

    def _unit_size(self, entry_price, stop_price):
        if stop_price >= entry_price:
            return 0.0
        risk_amount = self.equity * self.cfg.risk_pct_per_trade
        size = risk_amount / (entry_price - stop_price)
        max_notional = self.equity / self.cfg.max_concurrent_positions / self.cfg.max_units
        return min(size, max_notional / entry_price)

    def _try_open_position(self, sym, ts, p):
        cfg = self.cfg
        a = self.arrays[sym]
        entry_price = a.close[p]
        stop_price = a.initial_stop[p]
        add_step_ref = a.atr_add[p]
        if not np.isfinite(entry_price) or not np.isfinite(stop_price) or not np.isfinite(add_step_ref):
            return
        size = self._unit_size(entry_price, stop_price)
        if size <= 0:
            return

        trade = Trade(symbol=sym, entry_time=pd.Timestamp(ts), entry_price=entry_price,
                       stop_price_initial=stop_price, size_initial=size)
        trade.unit_entries.append((pd.Timestamp(ts), entry_price, size))
        entry_fill_price = entry_price * (1 + cfg.slippage_pct)
        self.equity -= entry_fill_price * size * cfg.taker_fee_pct

        pos = Position(sym, stop_price, trade, entry_price, cfg.add_atr_step * add_step_ref)
        self.open_positions[sym] = pos

    def _close_position(self, sym, ts, price, reason):
        cfg = self.cfg
        pos = self.open_positions.get(sym)
        if pos is None:
            return
        trade = pos.trade
        trade.total_size = sum(sz for (_, _, sz) in trade.unit_entries)
        trade.n_units = len(trade.unit_entries)
        exit_fill_price = price * (1 - cfg.slippage_pct)
        pnl = 0.0
        for (_, entry_price, unit_size) in trade.unit_entries:
            entry_fill_price = entry_price * (1 + cfg.slippage_pct)
            pnl += unit_size * (exit_fill_price - entry_fill_price) \
                - exit_fill_price * unit_size * cfg.taker_fee_pct \
                - entry_fill_price * unit_size * cfg.taker_fee_pct
        self.equity += pnl
        trade.exit_events.append((pd.Timestamp(ts), price, 1.0, reason))
        self.closed_trades.append(trade)
        del self.open_positions[sym]

    def _manage_position(self, sym, ts, p):
        cfg = self.cfg
        a = self.arrays[sym]
        pos = self.open_positions[sym]
        high, low = a.high[p], a.low[p]

        if low <= pos.stop_price:
            self._close_position(sym, ts, pos.stop_price, "trail_stop")
            return

        trail_candidate = a.trail_ref[p]
        if not np.isnan(trail_candidate) and trail_candidate > pos.stop_price:
            pos.stop_price = trail_candidate

        if pos.n_units < cfg.max_units and pos.add_step > 0:
            add_trigger_price = pos.last_add_price + pos.add_step
            if high >= add_trigger_price:
                add_price = add_trigger_price
                size = self._unit_size(add_price, pos.stop_price)
                if size > 0:
                    pos.trade.unit_entries.append((pd.Timestamp(ts), add_price, size))
                    entry_fill_price = add_price * (1 + cfg.slippage_pct)
                    self.equity -= entry_fill_price * size * cfg.taker_fee_pct
                    pos.n_units += 1
                    pos.last_add_price = add_price
