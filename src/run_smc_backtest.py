"""
Orchestrates the liquidity-sweep + SMT-divergence + FVG strategy:
load cached OHLCV -> per-symbol setup detection -> portfolio backtest -> report.

Usage: python run_smc_backtest.py --timeframe 15m
"""
import argparse
import json
import os
import time

import pandas as pd

from run_backtest import load_universe, load_btc, RESULTS_DIR
from smc_signals import SMCConfig
from smc_setups import find_setups
from backtest_smc import SMCEngineConfig, SMCPortfolioBacktester
from metrics import trades_to_dataframe, compute_performance, performance_by_year


def build_symbol_setups(timeframe, smc_cfg: SMCConfig, verbose=True):
    symbol_dfs = load_universe(timeframe)
    btc_df = load_btc(timeframe)
    symbol_setups = {}
    t0 = time.time()
    for i, (sym, df) in enumerate(symbol_dfs.items()):
        ref_aligned = btc_df.reindex(df.index, method="ffill")
        setups = find_setups(df, ref_aligned, smc_cfg)
        symbol_setups[sym] = (df, setups)
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(symbol_dfs)}] setups so far: {sym} -> {len(setups)} "
                  f"(elapsed {time.time()-t0:.1f}s)")
    total_setups = sum(len(s) for _, s in symbol_setups.values())
    print(f"[{timeframe}] {len(symbol_dfs)} symbols, {total_setups} total setups found, "
          f"{time.time()-t0:.1f}s")
    return symbol_setups


def run_one(timeframe, smc_cfg=None, engine_cfg=None, starting_equity=10_000.0, verbose=True):
    if smc_cfg is None:
        smc_cfg = SMCConfig() if timeframe == "1h" else SMCConfig(
            pool_lookback_bars=600, target_lookback_bars=800, fvg_watch_bars=40, fvg_fill_window_bars=80,
        )
    if engine_cfg is None:
        engine_cfg = SMCEngineConfig(swing_k=smc_cfg.swing_k,
                                      max_hold_hours=96.0 if timeframe == "1h" else 48.0)

    symbol_setups = build_symbol_setups(timeframe, smc_cfg, verbose=verbose)

    bt = SMCPortfolioBacktester(engine_cfg, symbol_setups, starting_equity=starting_equity)
    equity_curve, trades = bt.run()

    trades_df = trades_to_dataframe(trades, engine_cfg.taker_fee_pct, engine_cfg.slippage_pct)
    perf = compute_performance(equity_curve, trades_df, starting_equity)
    by_year = performance_by_year(trades_df)

    return {
        "timeframe": timeframe, "n_symbols": len(symbol_setups),
        "performance": perf, "by_year": by_year, "trades_df": trades_df, "equity_curve": equity_curve,
        "smc_cfg": smc_cfg, "engine_cfg": engine_cfg,
    }


def print_report(res):
    p = res["performance"]
    print("=" * 70)
    print(f"SMC strategy -- Timeframe: {res['timeframe']}   Symbols: {res['n_symbols']}")
    print("=" * 70)
    for k in ["starting_equity", "final_equity", "total_return_pct", "max_drawdown_pct",
              "n_trades", "win_rate_pct", "profit_factor", "avg_winner", "avg_loser",
              "avg_holding_hours", "sharpe", "sortino"]:
        print(f"  {k:24s}: {p.get(k)}")
    print("  exit_reason_counts     :", p.get("exit_reason_counts"))
    print("-" * 70)
    for y, stats in sorted(res["by_year"].items()):
        print(f"  {y}: n={stats['n_trades']:4d}  pnl=${stats['total_pnl']:.2f}  "
              f"win_rate={stats['win_rate_pct']:.1f}%  pf={stats['profit_factor']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", choices=["15m", "1h"], required=True)
    args = ap.parse_args()

    res = run_one(args.timeframe)
    print_report(res)

    res["trades_df"].to_csv(os.path.join(RESULTS_DIR, f"smc_trades_{args.timeframe}.csv"), index=False)
    eq_df = pd.DataFrame(res["equity_curve"], columns=["time_ns", "equity"])
    eq_df["time"] = pd.to_datetime(eq_df["time_ns"])
    eq_df[["time", "equity"]].to_csv(os.path.join(RESULTS_DIR, f"smc_equity_{args.timeframe}.csv"), index=False)
    print("Saved.")
