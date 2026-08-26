from collections import Counter, defaultdict

from run_single_asset_mr_backtest import load_majors
from liq_wick_signals import LiqWickConfig
from backtest_liq_wick import LiqWickBacktester


if __name__ == "__main__":
    symbols = load_majors()
    print(f"{len(symbols)} symbols loaded")

    cfg = LiqWickConfig()
    bt = LiqWickBacktester(cfg, symbols)
    equity_curve, trades = bt.run()

    n = len(trades)
    pnls = [t.realized_pnl(cfg.taker_fee_pct, cfg.slippage_pct) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = sum(pnls)
    print(f"\n=== FULL PERIOD ===")
    print(f"n_trades={n}  win_rate={100*wins/n:.1f}%  total_pnl=${total_pnl:.0f}  final_equity=${10_000+total_pnl:.0f}")

    reasons = Counter(t.exit_reason for t in trades)
    print(f"exit reasons: {dict(reasons)}")

    print(f"\n=== BY YEAR ===")
    by_year = defaultdict(list)
    for t, p in zip(trades, pnls):
        by_year[t.entry_time.year].append(p)
    for year in sorted(by_year):
        yp = by_year[year]
        w = sum(1 for x in yp if x > 0)
        print(f"  {year}: n={len(yp):4d}  win_rate={100*w/len(yp):.1f}%  pnl=${sum(yp):.0f}")

    print(f"\n=== BY SYMBOL (concentration check) ===")
    by_sym = defaultdict(list)
    for t, p in zip(trades, pnls):
        by_sym[t.symbol].append(p)
    sym_totals = sorted(by_sym.items(), key=lambda kv: -sum(kv[1]))
    for sym, sp in sym_totals:
        w = sum(1 for x in sp if x > 0)
        print(f"  {sym:10s} n={len(sp):4d}  win_rate={100*w/len(sp):5.1f}%  pnl=${sum(sp):8.0f}")
