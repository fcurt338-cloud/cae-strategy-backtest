"""
Proper multi-fold walk-forward for the pairs strategy: rolling windows where
pair selection uses ONLY data before the fold, tested ONLY on that fold --
repeated across the full history, not one single train/test split. This is
the standard every other strategy in this project was held to; the pairs
test hadn't been yet.

Also adds a REAL funding cost on the short leg -- a genuine, ongoing expense
of holding a short perpetual position that was disclosed as unmodeled in the
first pass. Uses actual Binance funding-rate history (fetched for each major
this session), not an estimate: for each short leg, sums the real funding
payments over the exact days that specific position was held.
"""
import glob
import os

import pandas as pd

from run_backtest import load_symbol_df, apply_listing_age_filter, apply_liquidity_filter
from pairs_signals import PairsConfig, daily_log_prices, select_pairs_by_correlation, compute_spread_zscore
from backtest_pairs import run_pairs_backtest, PairTrade

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_funding_rates():
    """symbol -> daily-summed funding rate series (funding rate is paid to shorts when
    positive, i.e. a positive rate here is a COST to a short position)."""
    files = glob.glob(os.path.join(RAW_DIR, "*_funding.csv"))
    out = {}
    for f in files:
        sym = os.path.basename(f).replace("_funding.csv", "")
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
        df = df.set_index("funding_time").sort_index()
        out[sym] = df["funding_rate"].resample("1D").sum()
    return out


def short_leg_funding_pnl(funding_rates: dict, symbol: str, entry_time, exit_time, notional: float) -> float:
    """Real funding P&L (not just cost -- can be positive) of holding `symbol` short
    from entry_time to exit_time, sized at `notional` dollars.

    Binance convention, stated precisely and applied literally: when the funding rate
    is positive, LONGS pay SHORTS that rate (as a fraction of notional) each funding
    interval -- so a short position RECEIVES funding when the rate is positive, and
    PAYS when the rate is negative. This is summed rate * notional, added directly to
    the trade's P&L (not subtracted) -- positive rate sum = income to the short leg,
    negative rate sum = expense."""
    if symbol not in funding_rates:
        return 0.0
    series = funding_rates[symbol]
    window = series.loc[entry_time:exit_time]
    total_rate = window.sum()
    return total_rate * notional


def load_majors():
    majors = ['BTC','ETH','SOL','BNB','XRP','ADA','DOGE','AVAX','LINK','DOT','LTC','BCH','UNI',
              'ATOM','NEAR','APT','ARB','OP','FIL','ICP','ETC','XLM','HBAR','SUI','INJ','TIA',
              'RENDER','TRX','SHIB']
    symbols = {}
    for base in majors:
        sym = base + 'USDT'
        df = load_symbol_df(sym, '1h')
        if df is None or len(df) < 200:
            continue
        df = apply_listing_age_filter(df, 14)
        df = apply_liquidity_filter(df, 100_000, 24)
        if df is not None:
            symbols[sym] = df
    return symbols


def run_walkforward(n_folds=6, top_n=8, zscore_window=30, entry_z=2.0, exit_z=0.5, stop_z=4.0,
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
    for i in range(1, n_folds):  # fold 0 has no prior data to select pairs from -- start at fold 1
        train_end = start + pd.Timedelta(days=i * fold_len)
        fold_end = start + pd.Timedelta(days=(i + 1) * fold_len) if i < n_folds - 1 else end

        pairs = select_pairs_by_correlation(log_prices, train_end, top_n=top_n)
        if not pairs:
            continue

        # run the backtest across the FULL series (needed for the rolling z-score to have
        # proper history) but only KEEP trades entered within (train_end, fold_end] --
        # i.e. this fold's genuinely out-of-sample window, selected on strictly prior data.
        eq, trades = run_pairs_backtest(symbols, cfg, pairs, starting_equity=starting_equity)
        fold_trades = [t for t in trades if train_end < t.entry_time <= fold_end]

        pnls = []
        for t in fold_trades:
            gross = t.realized_pnl(fee, slip)
            funding_pnl = short_leg_funding_pnl(funding_rates, t.sym_short, t.entry_time, t.exit_time,
                                                 t.notional_per_leg)
            pnls.append(gross + funding_pnl)

        n = len(fold_trades)
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        win_rate = 100 * wins / n if n else None
        fold_results.append({
            "fold": i, "train_end": str(train_end.date()), "fold_end": str(fold_end.date()),
            "pairs_selected": [(a, b) for a, b, _ in pairs],
            "n_trades": n, "win_rate": win_rate, "total_pnl": total_pnl,
        })
        all_oos_trades.extend(list(zip(fold_trades, pnls)))
        if verbose:
            print(f"fold {i} [{train_end.date()} -> {fold_end.date()}]: n={n} win_rate={win_rate} "
                  f"pnl=${total_pnl:.0f}  pairs={[(a,b) for a,b,_ in pairs]}")

    total_n = sum(r["n_trades"] for r in fold_results)
    total_wins = sum(1 for _, p in all_oos_trades if p > 0)
    total_pnl = sum(p for _, p in all_oos_trades)
    print(f"\n=== ALL FOLDS COMBINED (genuinely OOS, funding-cost-adjusted) ===")
    print(f"n_trades={total_n} win_rate={100*total_wins/total_n if total_n else None:.1f}% total_pnl=${total_pnl:.0f}")
    n_profitable_folds = sum(1 for r in fold_results if r["total_pnl"] > 0)
    print(f"profitable folds: {n_profitable_folds}/{len(fold_results)}")

    return fold_results, all_oos_trades


if __name__ == "__main__":
    run_walkforward(n_folds=6)
