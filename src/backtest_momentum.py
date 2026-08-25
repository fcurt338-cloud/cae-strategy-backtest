"""
Cross-sectional momentum backtest: full liquidate-and-rebuy at each rebalance
date, equal-weighted across the top-K names by trailing return. Deliberately
the simplest possible execution model (no partial rebalancing, no carried
positions) precisely because that simplicity is easiest to verify correct --
per instruction, precision over cleverness here.

Execution timing: ranks are computed from data through rebalance date D's
close; trades EXECUTE at D+1's open (not D's own close) -- deciding on a
close and then claiming you traded at that same close is a subtle lookahead
that real trading can't achieve (the close isn't known until it happens).
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from momentum_signals import MomentumConfig, daily_closes, build_rebalance_dates, compute_returns_at


@dataclass
class RebalanceEvent:
    date: pd.Timestamp
    exec_date: pd.Timestamp
    held: List[str]
    ranks: pd.Series
    equity_before: float
    equity_after: float


def run_momentum_backtest(symbol_dfs: Dict[str, pd.DataFrame], cfg: MomentumConfig,
                           starting_equity: float = 10_000.0,
                           start_date: pd.Timestamp = None, end_date: pd.Timestamp = None,
                           partial_rebalance: bool = False, rank_fn=None):
    """partial_rebalance=True: only trade names entering/exiting the top-K list each
    rebalance; continuing holdings ride untouched (no fee/slippage paid on them). This
    is how real momentum strategies control turnover cost -- full liquidate-and-rebuy
    every period (the default) pays transaction costs even on names that didn't need
    to change, which turned out to be the dominant cost driver in the full-rebalance
    version.

    rank_fn: optional callable(as_of_date) -> pd.Series ranking score (higher = more
    attractive long). Only honored when partial_rebalance=True. Defaults to price
    momentum. Pass funding_signals.compute_funding_rank_at (wrapped) to rank by
    funding rate instead."""
    if partial_rebalance:
        return _run_partial_rebalance(symbol_dfs, cfg, starting_equity, start_date, end_date, rank_fn=rank_fn)
    return _run_full_rebalance(symbol_dfs, cfg, starting_equity, start_date, end_date)


def _run_full_rebalance(symbol_dfs: Dict[str, pd.DataFrame], cfg: MomentumConfig,
                         starting_equity: float = 10_000.0,
                         start_date: pd.Timestamp = None, end_date: pd.Timestamp = None):
    wide_close = daily_closes(symbol_dfs)
    # daily open, for next-day execution pricing
    wide_open = pd.DataFrame({sym: df["open"].resample("1D").first() for sym, df in symbol_dfs.items()})

    full_index = wide_close.index
    # only fully-closed days: drop the last day of overall data (the run may have started mid-day)
    fully_closed_index = full_index[:-1]
    if start_date is not None:
        fully_closed_index = fully_closed_index[fully_closed_index >= start_date]
    if end_date is not None:
        fully_closed_index = fully_closed_index[fully_closed_index <= end_date]

    rebalance_dates = build_rebalance_dates(
        fully_closed_index, cfg.holding_days, cfg.lookback_days, cfg.min_lookback_history_days
    )

    equity = starting_equity
    equity_curve = []
    events: List[RebalanceEvent] = []
    current_positions = {}  # symbol -> shares (in "unit" terms, tracked via dollar value instead)
    current_dollar_alloc = {}  # symbol -> dollars invested at last rebalance's exec price

    for d in rebalance_dates:
        # find execution date = next available index date after d in full_index
        loc = full_index.get_indexer([d])[0]
        if loc + 1 >= len(full_index):
            break
        exec_date = full_index[loc + 1]
        if exec_date not in wide_open.index:
            continue

        # 1) close existing positions at exec_date's open
        if current_dollar_alloc:
            realized_equity = 0.0
            for sym, (entry_price, dollars) in current_dollar_alloc.items():
                exit_price = wide_open.loc[exec_date, sym] if sym in wide_open.columns else np.nan
                if pd.isna(exit_price) or pd.isna(entry_price) or entry_price <= 0:
                    realized_equity += dollars  # can't price it, carry forward at cost (rare/edge case)
                    continue
                shares = dollars / (entry_price * (1 + cfg.slippage_pct))  # shares bought at entry
                sell_fill = exit_price * (1 - cfg.slippage_pct)
                proceeds = shares * sell_fill
                proceeds -= proceeds * cfg.taker_fee_pct
                realized_equity += proceeds
            equity = realized_equity
            current_dollar_alloc = {}

        equity_before = equity

        # 2) rank and select
        rets = compute_returns_at(wide_close, d, cfg.lookback_days)
        rets = rets.dropna()
        if len(rets) == 0:
            equity_curve.append((exec_date, equity))
            continue
        ranked = rets.sort_values(ascending=False)
        top = ranked.head(cfg.top_k)
        held = list(top.index)

        # 3) open new equal-weight positions at exec_date's open
        if held:
            per_position = equity / len(held)
            for sym in held:
                entry_price = wide_open.loc[exec_date, sym] if sym in wide_open.columns else np.nan
                if pd.isna(entry_price) or entry_price <= 0:
                    continue
                buy_fill = entry_price * (1 + cfg.slippage_pct)
                fee = per_position * cfg.taker_fee_pct
                dollars_net = per_position - fee
                current_dollar_alloc[sym] = (entry_price, dollars_net)

        events.append(RebalanceEvent(date=d, exec_date=exec_date, held=held, ranks=top,
                                      equity_before=equity_before, equity_after=equity))
        equity_curve.append((exec_date, equity))

    # mark final equity at last available prices
    if current_dollar_alloc:
        last_date = full_index[-1]
        final_equity = 0.0
        for sym, (entry_price, dollars) in current_dollar_alloc.items():
            last_price = wide_close.loc[last_date, sym] if sym in wide_close.columns else np.nan
            if pd.isna(last_price) or pd.isna(entry_price) or entry_price <= 0:
                final_equity += dollars
                continue
            shares = dollars / (entry_price * (1 + cfg.slippage_pct))
            final_equity += shares * last_price * (1 - cfg.slippage_pct)
        equity = final_equity
        equity_curve.append((last_date, equity))

    return equity_curve, events, starting_equity


def _run_partial_rebalance(symbol_dfs: Dict[str, pd.DataFrame], cfg: MomentumConfig,
                            starting_equity: float = 10_000.0,
                            start_date: pd.Timestamp = None, end_date: pd.Timestamp = None,
                            rank_fn=None):
    """rank_fn: optional callable(as_of_date) -> pd.Series of rank scores (higher = more
    attractive long candidate). Defaults to the price-momentum ranking (compute_returns_at)
    if not provided, preserving prior behavior exactly."""
    wide_close = daily_closes(symbol_dfs)
    wide_open = pd.DataFrame({sym: df["open"].resample("1D").first() for sym, df in symbol_dfs.items()})
    if rank_fn is None:
        rank_fn = lambda d: compute_returns_at(wide_close, d, cfg.lookback_days)

    full_index = wide_close.index
    fully_closed_index = full_index[:-1]
    if start_date is not None:
        fully_closed_index = fully_closed_index[fully_closed_index >= start_date]
    if end_date is not None:
        fully_closed_index = fully_closed_index[fully_closed_index <= end_date]

    rebalance_dates = build_rebalance_dates(
        fully_closed_index, cfg.holding_days, cfg.lookback_days, cfg.min_lookback_history_days
    )

    holdings = {}  # symbol -> shares (fixed at entry, held until sold)
    cash = starting_equity
    equity_curve = []
    events: List[RebalanceEvent] = []

    def mark_value(as_of_date):
        total = cash
        for sym, shares in holdings.items():
            px = wide_close.loc[as_of_date, sym] if sym in wide_close.columns else np.nan
            if not pd.isna(px):
                total += shares * px
        return total

    for d in rebalance_dates:
        loc = full_index.get_indexer([d])[0]
        if loc + 1 >= len(full_index):
            break
        exec_date = full_index[loc + 1]
        if exec_date not in wide_open.index:
            continue

        equity_before = mark_value(exec_date)

        rets = rank_fn(d).dropna()
        if len(rets) == 0:
            equity_curve.append((exec_date, mark_value(exec_date)))
            continue
        ranked = rets.sort_values(ascending=False)
        top = ranked.head(cfg.top_k)
        target = set(top.index)
        current = set(holdings.keys())

        to_sell = current - target
        to_keep = current & target
        to_buy = target - current

        # 1) sell exits
        for sym in to_sell:
            exit_price = wide_open.loc[exec_date, sym] if sym in wide_open.columns else np.nan
            shares = holdings.pop(sym)
            if pd.isna(exit_price) or exit_price <= 0:
                continue
            sell_fill = exit_price * (1 - cfg.slippage_pct)
            proceeds = shares * sell_fill
            proceeds -= proceeds * cfg.taker_fee_pct
            cash += proceeds

        # 2) buy new entries with available cash, equal-weighted among them
        if to_buy:
            per_position = cash / len(to_buy)
            for sym in to_buy:
                entry_price = wide_open.loc[exec_date, sym] if sym in wide_open.columns else np.nan
                if pd.isna(entry_price) or entry_price <= 0:
                    continue
                buy_fill = entry_price * (1 + cfg.slippage_pct)
                fee = per_position * cfg.taker_fee_pct
                dollars_net = per_position - fee
                shares = dollars_net / buy_fill
                holdings[sym] = shares
                cash -= per_position

        held_list = list(to_keep) + list(to_buy)
        equity_after = mark_value(exec_date)
        events.append(RebalanceEvent(date=d, exec_date=exec_date, held=held_list, ranks=top,
                                      equity_before=equity_before, equity_after=equity_after))
        equity_curve.append((exec_date, equity_after))

    if holdings:
        last_date = full_index[-1]
        equity_curve.append((last_date, mark_value(last_date)))

    return equity_curve, events, starting_equity
