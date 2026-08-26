"""
Portfolio-level, multi-symbol engine for the liquidation-wick reversal
strategy (liq_wick_signals.py). Same timestamp-synchronized multi-symbol
loop as backtest_single_asset_mr.py / backtest_trend.py, but with a fixed
ATR stop-loss placed beyond the rejected wick, a fixed R-multiple profit
target, and a short (24h) hard time-stop -- this is meant to be a fast
exhaustion-bounce trade, not a slow drift back to a mean.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from liq_wick_signals import LiqWickConfig, compute_signal_arrays


@dataclass
class Trade:
    symbol: str
    direction: int  # +1 long, -1 short
    entry_time: object
    entry_price: float
    stop_price: float
    target_price: float
    size: float
    exit_time: object = None
    exit_price: float = None
    exit_reason: str = None

    def realized_pnl(self, fee_pct, slip_pct):
        if self.exit_price is None:
            return 0.0
        if self.direction > 0:
            entry_fill = self.entry_price * (1 + slip_pct)
            exit_fill = self.exit_price * (1 - slip_pct)
            pnl = self.size * (exit_fill - entry_fill)
        else:
            entry_fill = self.entry_price * (1 - slip_pct)
            exit_fill = self.exit_price * (1 + slip_pct)
            pnl = self.size * (entry_fill - exit_fill)
        fees = fee_pct * self.size * (entry_fill + exit_fill)
        return pnl - fees


class SymArrays:
    def __init__(self, df: pd.DataFrame, cfg: LiqWickConfig):
        self.times = df.index.values.astype("datetime64[ns]").view("int64")
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.n = len(df)
        sig = compute_signal_arrays(df, cfg)
        self.long_entry = sig["long_entry"]
        self.short_entry = sig["short_entry"]
        self.wick_low = sig["wick_low"]
        self.wick_high = sig["wick_high"]
        self.atr = sig["atr"]


class Position:
    __slots__ = ["symbol", "direction", "entry_price", "size", "stop_price", "target_price", "trade", "bars_held"]

    def __init__(self, symbol, direction, entry_price, size, stop_price, target_price, trade):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.size = size
        self.stop_price = stop_price
        self.target_price = target_price
        self.trade = trade
        self.bars_held = 0


class LiqWickBacktester:
    def __init__(self, cfg: LiqWickConfig, symbol_dfs: Dict[str, pd.DataFrame], starting_equity: float = 10_000.0):
        self.cfg = cfg
        self.equity = starting_equity
        self.arrays: Dict[str, SymArrays] = {sym: SymArrays(df, cfg) for sym, df in symbol_dfs.items()}
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
                    if a.long_entry[p]:
                        self._try_open_position(s, p, 1)
                    elif a.short_entry[p]:
                        self._try_open_position(s, p, -1)

            for s, p in active:
                ptr[s] = p + 1

            self.equity_curve.append((pd.Timestamp(ts), self.equity))

        for sym, pos in list(self.open_positions.items()):
            a = arrs[sym]
            last_idx = a.n - 1
            self._close_position(sym, a.times[last_idx], a.close[last_idx], "end_of_data")

        return self.equity_curve, self.closed_trades

    def _try_open_position(self, sym, p, direction):
        cfg = self.cfg
        a = self.arrays[sym]
        if p + 1 >= a.n:
            return
        entry_price = a.open[p + 1]
        ts_fill = a.times[p + 1]
        atr_now = a.atr[p]
        if not np.isfinite(entry_price) or not np.isfinite(atr_now) or atr_now <= 0:
            return

        if direction > 0:
            stop_price = a.wick_low[p] - cfg.stop_atr_mult * atr_now
            if stop_price >= entry_price:
                return
            risk_distance = entry_price - stop_price
            target_price = entry_price + cfg.take_profit_r_mult * risk_distance
        else:
            stop_price = a.wick_high[p] + cfg.stop_atr_mult * atr_now
            if stop_price <= entry_price:
                return
            risk_distance = stop_price - entry_price
            target_price = entry_price - cfg.take_profit_r_mult * risk_distance

        risk_amount = self.equity * cfg.risk_pct_per_trade
        size = risk_amount / risk_distance
        if size <= 0 or not np.isfinite(size):
            return
        max_notional = self.equity / cfg.max_concurrent_positions
        size = min(size, max_notional / entry_price)
        if size <= 0:
            return

        trade = Trade(symbol=sym, direction=direction, entry_time=pd.Timestamp(ts_fill),
                       entry_price=entry_price, stop_price=stop_price, target_price=target_price, size=size)
        entry_fill_price = entry_price * (1 + cfg.slippage_pct if direction > 0 else 1 - cfg.slippage_pct)
        self.equity -= entry_fill_price * size * cfg.taker_fee_pct

        pos = Position(sym, direction, entry_price, size, stop_price, target_price, trade)
        self.open_positions[sym] = pos

    def _close_position(self, sym, ts, price, reason):
        cfg = self.cfg
        pos = self.open_positions.get(sym)
        if pos is None:
            return
        pos.trade.exit_time = pd.Timestamp(ts)
        pos.trade.exit_price = price
        pos.trade.exit_reason = reason
        self.equity += pos.trade.realized_pnl(cfg.taker_fee_pct, cfg.slippage_pct)
        self.closed_trades.append(pos.trade)
        del self.open_positions[sym]

    def _manage_position(self, sym, ts, p):
        cfg = self.cfg
        a = self.arrays[sym]
        pos = self.open_positions[sym]
        pos.bars_held += 1
        high, low = a.high[p], a.low[p]

        if pos.direction > 0:
            if low <= pos.stop_price:
                self._close_position(sym, ts, pos.stop_price, "stop")
                return
            if high >= pos.target_price:
                self._close_position(sym, ts, pos.target_price, "target")
                return
        else:
            if high >= pos.stop_price:
                self._close_position(sym, ts, pos.stop_price, "stop")
                return
            if low <= pos.target_price:
                self._close_position(sym, ts, pos.target_price, "target")
                return

        if pos.bars_held >= cfg.max_holding_bars:
            self._close_position(sym, ts, a.close[p], "time_stop")
