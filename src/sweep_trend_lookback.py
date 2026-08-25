"""
Experiment 1: sweep Donchian entry/exit lookback pairs on 1h, loading the
universe once and reusing it across configs (avoids redundant CSV parsing).
"""
import time

from run_backtest import load_universe
from trend_signals import TrendConfig
from backtest_trend import TrendEngineConfig, TrendPortfolioBacktester
from metrics import trades_to_dataframe, compute_performance

PAIRS_DAYS = [
    (20, 10), (35, 15), (55, 20), (75, 30), (100, 40), (130, 50),
]

def breakeven_gap(perf):
    aw, al, wr = perf.get("avg_winner"), perf.get("avg_loser"), perf.get("win_rate_pct")
    if not aw or not al or wr is None:
        return None
    al = abs(al)
    breakeven = al / (aw + al) * 100
    return wr - breakeven, breakeven

if __name__ == "__main__":
    t0 = time.time()
    symbol_dfs = load_universe("1h")
    print(f"Universe loaded: {len(symbol_dfs)} symbols ({time.time()-t0:.1f}s)")

    engine_cfg = TrendEngineConfig()
    results = []
    for entry_d, exit_d in PAIRS_DAYS:
        tcfg = TrendConfig(entry_lookback=entry_d * 24, exit_lookback=exit_d * 24,
                            stop_atr_mult=2.0, atr_period=20)
        bt = TrendPortfolioBacktester(engine_cfg, symbol_dfs, tcfg, starting_equity=10_000.0)
        eq, trades = bt.run()
        tdf = trades_to_dataframe(trades, engine_cfg.taker_fee_pct, engine_cfg.slippage_pct)
        perf = compute_performance(eq, tdf, 10_000.0)
        gap = breakeven_gap(perf)
        n_profitable = int((tdf.groupby("symbol")["pnl"].sum() > 0).sum()) if len(tdf) else 0
        n_symbols_traded = tdf["symbol"].nunique() if len(tdf) else 0
        row = {
            "entry_d": entry_d, "exit_d": exit_d,
            "n_trades": perf.get("n_trades"),
            "return_pct": perf.get("total_return_pct"),
            "win_rate": perf.get("win_rate_pct"),
            "pf": perf.get("profit_factor"),
            "avg_win": perf.get("avg_winner"),
            "avg_loss": perf.get("avg_loser"),
            "maxdd": perf.get("max_drawdown_pct"),
            "gap_pp": gap[0] if gap else None,
            "breakeven_wr": gap[1] if gap else None,
            "profitable_syms": f"{n_profitable}/{n_symbols_traded}",
        }
        results.append(row)
        print(f"entry={entry_d}d exit={exit_d}d: n={row['n_trades']} ret={row['return_pct']:.2f}% "
              f"wr={row['win_rate']:.2f}% pf={row['pf']:.3f} avgW=${row['avg_win']:.2f} "
              f"avgL=${row['avg_loss']:.2f} maxdd={row['maxdd']:.2f}% "
              f"gap={row['gap_pp']:.1f}pp profitable={row['profitable_syms']}")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
