"""
Parameter sensitivity sweep for the CAE strategy.

Sweeps one parameter at a time around the default config (holding others fixed)
so results are readable as "how sensitive is performance to N / vol multiplier /
range threshold", as requested. Reuses the already-loaded universe across runs
to avoid re-reading CSVs for every combination.
"""
import argparse
import json
import os

from config import default_config_for_timeframe
from backtest import PortfolioBacktester
from metrics import trades_to_dataframe, compute_performance
from run_backtest import load_universe, load_btc, RESULTS_DIR


SWEEP_GRID = {
    "coil_lookback": {
        "15m": [12, 18, 24, 32, 48],
        "1h": [8, 12, 15, 18, 24],
    },
    "breakout_vol_mult_coil": [2.0, 2.5, 3.0, 4.0, 5.0],
    "max_coil_range_pct": [0.06, 0.08, 0.10, 0.14, 0.20],
    "vol_dryup_mult": [0.45, 0.55, 0.65, 0.75, 0.85],
    "entry_fill": ["close", "next_open"],
    "stop_mode": ["structure", "fixed15"],
    "breakout_vol_ref": ["coil_avg", "sma20"],
    "trail_mode": ["atr", "swing_low"],
}


def run_sweep(timeframe, starting_equity=10_000.0):
    symbol_dfs = load_universe(timeframe)
    btc_df = load_btc(timeframe)
    base_cfg = default_config_for_timeframe(timeframe)

    print(f"Universe: {len(symbol_dfs)} symbols for {timeframe} sweep")

    results = {}
    for param, values in SWEEP_GRID.items():
        vals = values[timeframe] if isinstance(values, dict) else values
        results[param] = []
        for v in vals:
            cfg = base_cfg.clone(**{param: v})
            bt = PortfolioBacktester(cfg, symbol_dfs, btc_df=btc_df, starting_equity=starting_equity)
            equity_curve, trades = bt.run()
            trades_df = trades_to_dataframe(trades, cfg.taker_fee_pct, cfg.slippage_pct)
            perf = compute_performance(equity_curve, trades_df, starting_equity)
            row = {
                "value": v,
                "n_trades": perf.get("n_trades"),
                "total_return_pct": perf.get("total_return_pct"),
                "win_rate_pct": perf.get("win_rate_pct"),
                "profit_factor": perf.get("profit_factor"),
                "max_drawdown_pct": perf.get("max_drawdown_pct"),
                "sharpe": perf.get("sharpe"),
            }
            results[param].append(row)
            print(f"  {param}={v}: trades={row['n_trades']}, return={row['total_return_pct']:.2f}%, "
                  f"win_rate={row['win_rate_pct']}, pf={row['profit_factor']}, "
                  f"maxdd={row['max_drawdown_pct']:.2f}%, sharpe={row['sharpe']}")

    out_path = os.path.join(RESULTS_DIR, f"sweep_{timeframe}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved sweep -> {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", choices=["15m", "1h"], required=True)
    args = ap.parse_args()
    run_sweep(args.timeframe)
