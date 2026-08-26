"""
Pairs-trading variant that scales EXPOSURE down continuously as the spread
keeps moving against the position, instead of holding full size until a
stop is hit. This targets a specific, now well-diagnosed failure mode of
the baseline engine (backtest_pairs.py): across the full 2018-2026 walk-
forward, the trade-level win rate stayed genuinely elevated (>50% in 11 of
13 folds) but overall P&L was still negative, because a minority of trades
that don't revert lose much more than the many small trades win. Three
different fixes to WHEN the trade exits (freezing the exit reference,
tightening stop_z, a hard time-stop) either failed outright or were
cosmetic -- none of them touched HOW MUCH is at risk as the trade goes
against you, which is the lever this variant changes.

Scaling rule: at entry, target exposure = 100% of the position's notional.
As |z| moves from entry_z toward stop_z, target exposure shrinks linearly:
    target_fraction = clip(1 - (|z| - entry_z) / (stop_z - entry_z), 0, 1)
Exposure only ever trims down (a spread moving back toward the mean does
NOT re-add size -- avoids whipsaw and avoids re-risking into a trade that
already proved it could go the wrong way). By the time |z| would reach the
old hard stop, the remaining position is already ~0, so no single trade can
produce the -$3,000+ losses that drove the baseline's worst folds.

Cost model and position sizing (initial notional, entry/exit fee+slippage
per leg per side) are unchanged from backtest_pairs.py, decomposed into
partial fills so trims are charged fees/slippage the same way a real
partial close would be.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from pairs_signals import PairsConfig, daily_log_prices, compute_spread_zscore


@dataclass
class PairTradeScaled:
    sym_long: str
    sym_short: str
    entry_time: object
    original_notional: float
    price_long_entry: float
    price_short_entry: float
    long_entry_fill: float
    short_entry_fill: float
    shares_long_total: float
    shares_short_total: float
    shares_long_remaining: float
    shares_short_remaining: float
    remaining_fraction: float = 1.0
    realized_pnl_so_far: float = 0.0
    trim_events: int = 0
    exit_time: object = None
    exit_reason: str = None
    fraction_history: list = field(default_factory=list)  # [(date, fraction_held_from_this_date), ...]

    def fraction_on(self, d) -> float:
        """Fraction of original_notional held as of date d, for funding-cost purposes."""
        frac = 1.0
        for dt, f in self.fraction_history:
            if dt <= d:
                frac = f
            else:
                break
        return frac

    def trim(self, target_fraction: float, price_long_now: float, price_short_now: float,
             fee_pct: float, slip_pct: float, when) -> float:
        """Close (self.remaining_fraction - target_fraction) of the ORIGINAL position.
        Returns the P&L realized by this trim (already net of exit-side fee)."""
        frac_to_close = self.remaining_fraction - target_fraction
        if frac_to_close <= 1e-9:
            return 0.0
        shares_long_close = self.shares_long_total * frac_to_close
        shares_short_close = self.shares_short_total * frac_to_close

        long_exit_fill = price_long_now * (1 - slip_pct)   # sell the long -- receive less
        short_exit_fill = price_short_now * (1 + slip_pct)  # buy back the short -- pay more

        long_pnl = shares_long_close * (long_exit_fill - self.long_entry_fill)
        short_pnl = shares_short_close * (self.short_entry_fill - short_exit_fill)
        exit_fee = fee_pct * (shares_long_close * long_exit_fill + shares_short_close * short_exit_fill)
        pnl = long_pnl + short_pnl - exit_fee

        self.shares_long_remaining -= shares_long_close
        self.shares_short_remaining -= shares_short_close
        self.remaining_fraction = target_fraction
        self.realized_pnl_so_far += pnl
        self.trim_events += 1
        self.fraction_history.append((when, target_fraction))
        return pnl


def run_pairs_backtest_scaled(symbol_dfs: Dict[str, pd.DataFrame], cfg: PairsConfig, pairs: list,
                               starting_equity: float = 10_000.0):
    """Same structure as backtest_pairs.run_pairs_backtest, but positions scale down
    continuously toward stop_z instead of holding full size until a hard stop."""
    log_prices = daily_log_prices(symbol_dfs)
    wide_open = pd.DataFrame({sym: df["open"].resample("1D").first() for sym, df in symbol_dfs.items()})
    dates = log_prices.index[:-1]

    zscores = {}
    for a, b, _ in pairs:
        zscores[(a, b)] = compute_spread_zscore(log_prices, a, b, cfg.zscore_window)

    equity = starting_equity
    equity_curve = []
    open_trades: Dict[tuple, PairTradeScaled] = {}
    closed_trades: List[PairTradeScaled] = []

    for d in dates:
        loc = log_prices.index.get_loc(d)
        if loc + 1 >= len(log_prices.index):
            break
        exec_date = log_prices.index[loc + 1]
        if exec_date not in wide_open.index:
            continue

        # 1) manage open trades: scale down or fully exit
        for key in list(open_trades.keys()):
            a, b = key
            z_series = zscores[key]
            if d not in z_series.index:
                continue
            z = z_series.loc[d]
            if pd.isna(z):
                continue
            trade = open_trades[key]

            price_long_now = wide_open.loc[exec_date, trade.sym_long] if trade.sym_long in wide_open.columns else np.nan
            price_short_now = wide_open.loc[exec_date, trade.sym_short] if trade.sym_short in wide_open.columns else np.nan
            if pd.isna(price_long_now) or pd.isna(price_short_now):
                continue

            if abs(z) > cfg.entry_z:
                target = max(0.0, 1.0 - (abs(z) - cfg.entry_z) / (cfg.stop_z - cfg.entry_z))
            else:
                target = 1.0
            target = min(target, trade.remaining_fraction)  # never scale back up

            fully_reverted = abs(z) <= cfg.exit_z

            if fully_reverted or target <= 1e-6:
                pnl = trade.trim(0.0, price_long_now, price_short_now, cfg.taker_fee_pct, cfg.slippage_pct, exec_date)
                equity += pnl
                trade.exit_time = exec_date
                trade.exit_reason = "reverted" if fully_reverted else "scaled_out"
                closed_trades.append(trade)
                del open_trades[key]
            elif target < trade.remaining_fraction - 1e-9:
                pnl = trade.trim(target, price_long_now, price_short_now, cfg.taker_fee_pct, cfg.slippage_pct, exec_date)
                equity += pnl

        # 2) new entries (same candidate logic as the baseline engine)
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
            candidates.sort(key=lambda x: -abs(x[1]))

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
                notional = min(notional, equity / cfg.max_concurrent_pairs)
                if notional <= 0:
                    continue

                price_a = wide_open.loc[exec_date, a] if a in wide_open.columns else np.nan
                price_b = wide_open.loc[exec_date, b] if b in wide_open.columns else np.nan
                if pd.isna(price_a) or pd.isna(price_b):
                    continue

                if z > 0:
                    sym_long, sym_short = b, a
                    price_long_entry, price_short_entry = price_b, price_a
                else:
                    sym_long, sym_short = a, b
                    price_long_entry, price_short_entry = price_a, price_b

                long_entry_fill = price_long_entry * (1 + cfg.slippage_pct)
                short_entry_fill = price_short_entry * (1 - cfg.slippage_pct)
                shares_long = notional / long_entry_fill
                shares_short = notional / short_entry_fill
                entry_fee = cfg.taker_fee_pct * notional * 2
                equity -= entry_fee

                trade = PairTradeScaled(
                    sym_long=sym_long, sym_short=sym_short, entry_time=exec_date,
                    original_notional=notional, price_long_entry=price_long_entry,
                    price_short_entry=price_short_entry, long_entry_fill=long_entry_fill,
                    short_entry_fill=short_entry_fill, shares_long_total=shares_long,
                    shares_short_total=shares_short, shares_long_remaining=shares_long,
                    shares_short_remaining=shares_short,
                )
                trade.fraction_history.append((exec_date, 1.0))
                open_trades[key] = trade

        equity_curve.append((exec_date, equity))

    # close remaining at last price
    last_date = log_prices.index[-1]
    for key, trade in list(open_trades.items()):
        pl = wide_open.get(trade.sym_long, pd.Series(dtype=float)).get(last_date, np.nan)
        ps = wide_open.get(trade.sym_short, pd.Series(dtype=float)).get(last_date, np.nan)
        if pd.isna(pl) or pd.isna(ps):
            continue
        pnl = trade.trim(0.0, pl, ps, cfg.taker_fee_pct, cfg.slippage_pct, last_date)
        equity += pnl
        trade.exit_time = last_date
        trade.exit_reason = "end_of_data"
        closed_trades.append(trade)

    return equity_curve, closed_trades
