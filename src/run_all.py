"""
End-to-end driver: for each timeframe, run the baseline (spec-faithful) backtest,
the parameter sweep, and one refinement variant (min_hold_bars_before_fade_or_time_exit
> 0), then dump everything needed for the report to results/.
"""
import json
import os
import time

import pandas as pd

from run_backtest import run_one, print_report, RESULTS_DIR, load_universe, load_btc
from sweep import run_sweep
from backtest import PortfolioBacktester
from metrics import trades_to_dataframe, compute_performance, performance_by_year


def run_refinement(timeframe, starting_equity=10_000.0):
    from config import default_config_for_timeframe
    symbol_dfs = load_universe(timeframe)
    btc_df = load_btc(timeframe)
    base_cfg = default_config_for_timeframe(timeframe)
    grace = 8 if timeframe == "15m" else 3  # ~2h grace on 15m, ~3h grace on 1h
    cfg = base_cfg.clone(min_hold_bars_before_fade_or_time_exit=grace)
    bt = PortfolioBacktester(cfg, symbol_dfs, btc_df=btc_df, starting_equity=starting_equity)
    equity_curve, trades = bt.run()
    trades_df = trades_to_dataframe(trades, cfg.taker_fee_pct, cfg.slippage_pct)
    perf = compute_performance(equity_curve, trades_df, starting_equity)
    by_year = performance_by_year(trades_df)
    return {
        "timeframe": timeframe, "grace_bars": grace, "n_symbols": len(symbol_dfs),
        "performance": perf, "by_year": by_year, "trades_df": trades_df, "equity_curve": equity_curve,
    }


def main():
    all_results = {}
    for tf in ["15m", "1h"]:
        t0 = time.time()
        print(f"\n\n########## BASELINE {tf} ##########")
        res = run_one(tf)
        print_report(res)
        res["trades_df"].to_csv(os.path.join(RESULTS_DIR, f"trades_{tf}.csv"), index=False)
        eq_df = pd.DataFrame(res["equity_curve"], columns=["time_ns", "equity"])
        eq_df["time"] = pd.to_datetime(eq_df["time_ns"])
        eq_df[["time", "equity"]].to_csv(os.path.join(RESULTS_DIR, f"equity_{tf}.csv"), index=False)

        print(f"\n########## SWEEP {tf} ##########")
        sweep_res = run_sweep(tf)

        print(f"\n########## REFINEMENT {tf} (grace period) ##########")
        ref = run_refinement(tf)
        print(f"Refinement perf: {ref['performance']}")
        ref["trades_df"].to_csv(os.path.join(RESULTS_DIR, f"trades_refined_{tf}.csv"), index=False)

        all_results[tf] = {
            "baseline_performance": res["performance"],
            "baseline_by_year": res["by_year"],
            "n_symbols": res["n_symbols"],
            "symbols": res["symbols"],
            "sweep": sweep_res,
            "refinement_performance": ref["performance"],
            "refinement_by_year": ref["by_year"],
            "refinement_grace_bars": ref["grace_bars"],
        }
        print(f"[{tf}] total time: {time.time()-t0:.1f}s")

    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved consolidated results -> results/all_results.json")


if __name__ == "__main__":
    main()
