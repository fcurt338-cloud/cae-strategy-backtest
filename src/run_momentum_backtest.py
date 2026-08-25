"""
Orchestrates the momentum backtest with walk-forward validation built in
from the start -- report full-period AND rolling out-of-sample numbers
together, always, so a config can't be reported as "promising" without
already having been checked the way the trend system should have been.
"""
import math

import numpy as np
import pandas as pd

from run_backtest import load_universe
from momentum_signals import MomentumConfig
from backtest_momentum import run_momentum_backtest


def compute_period_metrics(equity_curve, starting_equity):
    if not equity_curve or len(equity_curve) < 2:
        return {"n_periods": 0}
    eq = np.array([e for _, e in equity_curve])
    times = [t for t, _ in equity_curve]
    period_rets = eq[1:] / eq[:-1] - 1.0
    total_return_pct = (eq[-1] / starting_equity - 1) * 100
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    max_dd_pct = dd.min() * 100 if len(dd) else 0.0
    hit_rate = 100 * (period_rets > 0).mean() if len(period_rets) else None
    avg_ret = period_rets.mean() if len(period_rets) else None
    std_ret = period_rets.std() if len(period_rets) else None
    sharpe = None
    if std_ret and std_ret > 0:
        periods_per_year = 365 / max((times[-1] - times[0]).days / max(len(period_rets), 1), 1)
        sharpe = (avg_ret / std_ret) * math.sqrt(periods_per_year)
    return {
        "n_periods": len(period_rets),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "hit_rate_pct": hit_rate,
        "avg_period_return_pct": avg_ret * 100 if avg_ret is not None else None,
        "sharpe": sharpe,
        "final_equity": eq[-1],
    }


def run_full_and_walkforward(cfg: MomentumConfig, timeframe="1h", n_folds=4, starting_equity=10_000.0,
                              verbose=True, partial_rebalance=False):
    symbol_dfs = load_universe(timeframe)
    if verbose:
        print(f"Universe: {len(symbol_dfs)} symbols")

    # Full period
    full_curve, full_events, _ = run_momentum_backtest(symbol_dfs, cfg, starting_equity,
                                                         partial_rebalance=partial_rebalance)
    full_metrics = compute_period_metrics(full_curve, starting_equity)

    # Determine overall date range for folds
    all_dates = sorted(set().union(*[set(df.index.date) for df in symbol_dfs.values()]))
    start, end = pd.Timestamp(all_dates[0], tz="UTC"), pd.Timestamp(all_dates[-1], tz="UTC")
    total_days = (end - start).days
    fold_len = total_days // n_folds

    fold_results = []
    for i in range(n_folds):
        fold_start = start + pd.Timedelta(days=i * fold_len)
        fold_end = start + pd.Timedelta(days=(i + 1) * fold_len) if i < n_folds - 1 else end
        curve, events, _ = run_momentum_backtest(symbol_dfs, cfg, starting_equity,
                                                   start_date=fold_start, end_date=fold_end,
                                                   partial_rebalance=partial_rebalance)
        m = compute_period_metrics(curve, starting_equity)
        fold_results.append({"fold": i + 1, "start": str(fold_start.date()), "end": str(fold_end.date()), **m})
        if verbose:
            print(f"  fold {i+1} [{fold_start.date()} - {fold_end.date()}]: "
                  f"n_periods={m.get('n_periods')} ret={m.get('total_return_pct')} "
                  f"hit_rate={m.get('hit_rate_pct')} sharpe={m.get('sharpe')}")

    return {"full_period": full_metrics, "folds": fold_results, "config": cfg}


if __name__ == "__main__":
    cfg = MomentumConfig(lookback_days=14, holding_days=7, top_k=8)
    print(f"Config: lookback={cfg.lookback_days}d hold={cfg.holding_days}d top_k={cfg.top_k}")
    res = run_full_and_walkforward(cfg)
    print("\nFull period:", res["full_period"])
