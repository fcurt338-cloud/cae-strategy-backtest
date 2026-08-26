"""
Market-neutral pairs-trading backtest. Short the rich leg, long the cheap leg,
dollar-neutral, profiting from the spread reverting -- not from either asset's
own direction. This is a genuinely different risk mechanism from every long-
only strategy tested earlier tonight.

Backtesting caveat, stated plainly: shorting spot isn't literally available
without margin/futures. This simulates the short leg with the same fee+
slippage cost structure used throughout this project as a reasonable
approximation; real execution would need a margin or perpetual-futures
account, and perp funding costs on the short leg are not modeled here.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from pairs_signals import PairsConfig, daily_log_prices, compute_spread_zscore


@dataclass
class PairTrade:
    sym_long: str
    sym_short: str
    entry_time: object
    entry_time2: object
    notional_per_leg: float
    price_long_entry: float
    price_short_entry: float
    exit_time: object = None
    price_long_exit: float = None
    price_short_exit: float = None
    exit_reason: str = None

    def realized_pnl(self, fee_pct, slip_pct):
        if self.exit_time is None:
            return 0.0
        long_entry_fill = self.price_long_entry * (1 + slip_pct)
        short_entry_fill = self.price_short_entry * (1 - slip_pct)  # short entry = sell = receive less
        long_exit_fill = self.price_long_exit * (1 - slip_pct)      # long exit = sell = receive less
        short_exit_fill = self.price_short_exit * (1 + slip_pct)    # short exit = buy back = pay more

        shares_long = self.notional_per_leg / long_entry_fill
        shares_short = self.notional_per_leg / short_entry_fill

        long_pnl = shares_long * (long_exit_fill - long_entry_fill)
        short_pnl = shares_short * (short_entry_fill - short_exit_fill)

        fees = fee_pct * (self.notional_per_leg * 2 + shares_long * long_exit_fill + shares_short * short_exit_fill)
        return long_pnl + short_pnl - fees

    def holding_days(self):
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).days


def run_pairs_backtest(symbol_dfs: Dict[str, pd.DataFrame], cfg: PairsConfig, pairs: list,
                        starting_equity: float = 10_000.0):
    """pairs: [(symA, symB, corr), ...] from select_pairs_by_correlation."""
    log_prices = daily_log_prices(symbol_dfs)
    wide_open = pd.DataFrame({sym: df["open"].resample("1D").first() for sym, df in symbol_dfs.items()})
    dates = log_prices.index[:-1]  # exclude last (possibly partial) day

    zscores = {}
    for a, b, _ in pairs:
        zscores[(a, b)] = compute_spread_zscore(log_prices, a, b, cfg.zscore_window)

    equity = starting_equity
    equity_curve = []
    open_trades: Dict[tuple, dict] = {}  # (a,b) -> state dict
    closed_trades: List[PairTrade] = []

    for d in dates:
        loc = log_prices.index.get_loc(d)
        if loc + 1 >= len(log_prices.index):
            break
        exec_date = log_prices.index[loc + 1]
        if exec_date not in wide_open.index:
            continue

        # 1) manage open pair trades
        for key in list(open_trades.keys()):
            a, b = key
            z_series = zscores[key]
            if d not in z_series.index:
                continue
            z = z_series.loc[d]
            if pd.isna(z):
                continue
            state = open_trades[key]
            trade = state["trade"]

            should_exit = False
            reason = None
            if abs(z) <= cfg.exit_z:
                should_exit = True
                reason = "reverted"
            elif abs(z) >= cfg.stop_z:
                should_exit = True
                reason = "stop"

            if should_exit:
                price_long_exit = wide_open.loc[exec_date, trade.sym_long] if trade.sym_long in wide_open.columns else np.nan
                price_short_exit = wide_open.loc[exec_date, trade.sym_short] if trade.sym_short in wide_open.columns else np.nan
                if not pd.isna(price_long_exit) and not pd.isna(price_short_exit):
                    trade.exit_time = exec_date
                    trade.price_long_exit = price_long_exit
                    trade.price_short_exit = price_short_exit
                    trade.exit_reason = reason
                    pnl = trade.realized_pnl(cfg.taker_fee_pct, cfg.slippage_pct)
                    equity += pnl
                    closed_trades.append(trade)
                    del open_trades[key]

        # 2) consider new entries
        if len(open_trades) < cfg.max_concurrent_pairs:
            candidates = []
            for a, b, _ in pairs:
                key = (a, b)
                if key in open_trades:
                    continue
                z_series = zscores[key]
                if d not in z_series.index:
                    continue
                z = z_series.loc[d]
                if pd.isna(z) or abs(z) < cfg.entry_z:
                    continue
                candidates.append((key, z))
            candidates.sort(key=lambda x: -abs(x[1]))  # strongest signal first

            for key, z in candidates:
                if len(open_trades) >= cfg.max_concurrent_pairs:
                    break
                a, b = key
                std_now = log_prices[a].loc[:d].tail(cfg.zscore_window).sub(
                    log_prices[b].loc[:d].tail(cfg.zscore_window)
                ).std()
                if pd.isna(std_now) or std_now <= 0:
                    continue
                spread_dist_to_stop = (cfg.stop_z - abs(z)) * std_now if abs(z) < cfg.stop_z else std_now
                if spread_dist_to_stop <= 0:
                    continue
                risk_amount = equity * cfg.risk_pct_per_trade
                notional = risk_amount / spread_dist_to_stop
                max_notional = equity / cfg.max_concurrent_pairs
                notional = min(notional, max_notional)
                if notional <= 0:
                    continue

                price_a = wide_open.loc[exec_date, a] if a in wide_open.columns else np.nan
                price_b = wide_open.loc[exec_date, b] if b in wide_open.columns else np.nan
                if pd.isna(price_a) or pd.isna(price_b):
                    continue

                if z > 0:  # A rich vs B -> short A, long B
                    trade = PairTrade(sym_long=b, sym_short=a, entry_time=exec_date, entry_time2=exec_date,
                                       notional_per_leg=notional, price_long_entry=price_b, price_short_entry=price_a)
                else:  # A cheap vs B -> long A, short B
                    trade = PairTrade(sym_long=a, sym_short=b, entry_time=exec_date, entry_time2=exec_date,
                                       notional_per_leg=notional, price_long_entry=price_a, price_short_entry=price_b)
                open_trades[key] = {"trade": trade}

        equity_curve.append((exec_date, equity))

    # close remaining at last price
    last_date = log_prices.index[-1]
    for key, state in list(open_trades.items()):
        trade = state["trade"]
        pl = wide_open.get(trade.sym_long, pd.Series(dtype=float)).get(last_date, np.nan)
        ps = wide_open.get(trade.sym_short, pd.Series(dtype=float)).get(last_date, np.nan)
        if pd.isna(pl) or pd.isna(ps):
            continue
        trade.exit_time = last_date
        trade.price_long_exit = pl
        trade.price_short_exit = ps
        trade.exit_reason = "end_of_data"
        equity += trade.realized_pnl(cfg.taker_fee_pct, cfg.slippage_pct)
        closed_trades.append(trade)

    return equity_curve, closed_trades
