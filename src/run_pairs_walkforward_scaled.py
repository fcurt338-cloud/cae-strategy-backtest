"""
Same 5-fold (or N-fold) rolling walk-forward as run_pairs_walkforward.py, but
using the scaled-exposure engine (backtest_pairs_scaled.py) instead of the
baseline fixed-size-until-stop engine. Funding cost is prorated day-by-day by
the fraction of the position actually held that day, since exposure now
shrinks over a trade's life instead of staying fixed until exit.
"""
import os

import pandas as pd

from run_pairs_walkforward import load_majors, load_funding_rates
from pairs_signals import PairsConfig, daily_log_prices, select_pairs_by_correlation
from backtest_pairs_scaled import run_pairs_backtest_scaled


def short_leg_funding_pnl_scaled(funding_rates: dict, symbol: str, trade, notional: float) -> float:
    """Same Binance sign convention as the baseline (positive rate = income to the
    short leg), but weighted by the fraction of the position actually held each day,
    since this engine's exposure shrinks over the trade's life."""
    if symbol not in funding_rates:
        return 0.0
    series = funding_rates[symbol]
    window = series.loc[trade.entry_time:trade.exit_time]
    total = 0.0
    for d, rate in window.items():
        total += rate * notional * trade.fraction_on(d)
    return total


def run_walkforward_scaled(n_folds=6, top_n=8, zscore_window=30, entry_z=2.0, exit_z=0.5, stop_z=4.0,
                            fee=0.0004, slip=0.0005, starting_equity=10_000.0, verbose=True):
    symbols = load_majors()
    funding_rates = load_funding_rates()
    log_prices = daily_log_prices(symbols)
    all_dates = log_prices.index
    start, end = all_dates[0], all_dates[-1]
    total_days = (end - start).days
    fold_len = total_days // n_folds

    cfg = PairsConfig(zscore_window=zscore_window, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
                       taker_fee_pct=fee, slippage_pct=slip)

    fold_results = []
    all_oos_trades = []
    for i in range(1, n_folds):
        train_end = start + pd.Timedelta(days=i * fold_len)
        fold_end = start + pd.Timedelta(days=(i + 1) * fold_len) if i < n_folds - 1 else end

        pairs = select_pairs_by_correlation(log_prices, train_end, top_n=top_n)
        if not pairs:
            continue

        eq, trades = run_pairs_backtest_scaled(symbols, cfg, pairs, starting_equity=starting_equity)
        fold_trades = [t for t in trades if train_end < t.entry_time <= fold_end]

        pnls = []
        for t in fold_trades:
            gross = t.realized_pnl_so_far
            funding_pnl = short_leg_funding_pnl_scaled(funding_rates, t.sym_short, t, t.original_notional)
            pnls.append(gross + funding_pnl)

        n = len(fold_trades)
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        win_rate = 100 * wins / n if n else None
        fold_results.append({
            "fold": i, "train_end": str(train_end.date()), "fold_end": str(fold_end.date()),
            "n_trades": n, "win_rate": win_rate, "total_pnl": total_pnl,
        })
        all_oos_trades.extend(list(zip(fold_trades, pnls)))
        if verbose:
            print(f"fold {i} [{train_end.date()} -> {fold_end.date()}]: n={n} win_rate={win_rate} pnl=${total_pnl:.0f}")

    total_n = sum(r["n_trades"] for r in fold_results)
    total_wins = sum(1 for _, p in all_oos_trades if p > 0)
    total_pnl = sum(p for _, p in all_oos_trades)
    print(f"\n=== SCALED ENGINE, ALL FOLDS COMBINED ===")
    print(f"n_trades={total_n} win_rate={100*total_wins/total_n if total_n else None:.1f}% total_pnl=${total_pnl:.0f}")
    n_profitable_folds = sum(1 for r in fold_results if r["total_pnl"] > 0)
    print(f"profitable folds: {n_profitable_folds}/{len(fold_results)}")

    return fold_results, all_oos_trades


if __name__ == "__main__":
    run_walkforward_scaled(n_folds=14)
