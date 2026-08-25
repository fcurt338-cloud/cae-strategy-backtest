"""
Orchestrates the Donchian trend-following backtest: load cached OHLCV -> run -> report.

Usage: python run_trend_backtest.py --timeframe 1h [--system 1|2]
"""
import argparse
import os

import pandas as pd

from run_backtest import load_universe, RESULTS_DIR
from trend_signals import TrendConfig
from backtest_trend import TrendEngineConfig, TrendPortfolioBacktester
from metrics import trades_to_dataframe, compute_performance, performance_by_year


def default_trend_config(timeframe: str, system: int = 1) -> TrendConfig:
    bars_per_day = 96 if timeframe == "15m" else 24
    if system == 1:
        entry_days, exit_days = 20, 10
    else:
        entry_days, exit_days = 55, 20
    return TrendConfig(
        entry_lookback=entry_days * bars_per_day,
        exit_lookback=exit_days * bars_per_day,
        stop_atr_mult=2.0,
        atr_period=20,
    )


def run_one(timeframe, trend_cfg=None, engine_cfg=None, starting_equity=10_000.0, system=1):
    if trend_cfg is None:
        trend_cfg = default_trend_config(timeframe, system)
    if engine_cfg is None:
        engine_cfg = TrendEngineConfig()

    symbol_dfs = load_universe(timeframe)
    print(f"[{timeframe}] {len(symbol_dfs)} symbols, entry_lookback={trend_cfg.entry_lookback} "
          f"exit_lookback={trend_cfg.exit_lookback} bars")

    bt = TrendPortfolioBacktester(engine_cfg, symbol_dfs, trend_cfg, starting_equity=starting_equity)
    equity_curve, trades = bt.run()

    trades_df = trades_to_dataframe(trades, engine_cfg.taker_fee_pct, engine_cfg.slippage_pct)
    perf = compute_performance(equity_curve, trades_df, starting_equity)
    by_year = performance_by_year(trades_df)

    return {
        "timeframe": timeframe, "n_symbols": len(symbol_dfs),
        "performance": perf, "by_year": by_year, "trades_df": trades_df, "equity_curve": equity_curve,
        "trend_cfg": trend_cfg, "engine_cfg": engine_cfg,
    }


def print_report(res):
    p = res["performance"]
    print("=" * 70)
    print(f"Trend system -- Timeframe: {res['timeframe']}   Symbols: {res['n_symbols']}")
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
    ap.add_argument("--system", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()

    res = run_one(args.timeframe, system=args.system)
    print_report(res)

    suffix = f"{args.timeframe}_sys{args.system}"
    res["trades_df"].to_csv(os.path.join(RESULTS_DIR, f"trend_trades_{suffix}.csv"), index=False)
    eq_df = pd.DataFrame(res["equity_curve"], columns=["time_ns", "equity"])
    eq_df["time"] = pd.to_datetime(eq_df["time_ns"])
    eq_df[["time", "equity"]].to_csv(os.path.join(RESULTS_DIR, f"trend_equity_{suffix}.csv"), index=False)
    print("Saved.")
