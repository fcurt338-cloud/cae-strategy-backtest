"""Test the ensemble agreement-filter signal, walk-forward from the start."""
import pandas as pd

from run_backtest import load_universe
from momentum_signals import MomentumConfig
from backtest_momentum import run_momentum_backtest
from ensemble_signals import build_ensemble_rank_fn
from run_momentum_backtest import compute_period_metrics


def run_ensemble_full_and_wf(cfg: MomentumConfig, agreement_percentile=0.5, n_folds=4,
                              starting_equity=10_000.0, verbose=True):
    symbol_dfs = load_universe("1h")
    rank_fn = build_ensemble_rank_fn(symbol_dfs, momentum_lookback=cfg.lookback_days,
                                      agreement_percentile=agreement_percentile)

    full_curve, full_events, _ = run_momentum_backtest(
        symbol_dfs, cfg, starting_equity, partial_rebalance=True, rank_fn=rank_fn
    )
    full_metrics = compute_period_metrics(full_curve, starting_equity)
    if verbose:
        n_with_holdings = sum(1 for e in full_events if e.held)
        print(f"  rebalance events with >=1 qualifying holding: {n_with_holdings}/{len(full_events)}")

    all_dates = sorted(set().union(*[set(df.index.date) for df in symbol_dfs.values()]))
    start = pd.Timestamp(all_dates[0], tz="UTC")
    end = pd.Timestamp(all_dates[-1], tz="UTC")
    total_days = (end - start).days
    fold_len = total_days // n_folds

    fold_results = []
    for i in range(n_folds):
        fold_start = start + pd.Timedelta(days=i * fold_len)
        fold_end = start + pd.Timedelta(days=(i + 1) * fold_len) if i < n_folds - 1 else end
        curve, events, _ = run_momentum_backtest(
            symbol_dfs, cfg, starting_equity, start_date=fold_start, end_date=fold_end,
            partial_rebalance=True, rank_fn=rank_fn
        )
        m = compute_period_metrics(curve, starting_equity)
        fold_results.append({"fold": i + 1, "start": str(fold_start.date()), "end": str(fold_end.date()), **m})
        if verbose:
            print(f"  fold {i+1} [{fold_start.date()} - {fold_end.date()}]: "
                  f"n_periods={m.get('n_periods')} ret={m.get('total_return_pct')} "
                  f"hit_rate={m.get('hit_rate_pct')} sharpe={m.get('sharpe')}")

    return {"full_period": full_metrics, "folds": fold_results}


if __name__ == "__main__":
    for top_k in [8, 15]:
        for agreement_pct in [0.5, 0.66]:
            cfg = MomentumConfig(lookback_days=14, holding_days=7, top_k=top_k)
            print(f"\n=== top_k={top_k} agreement_percentile={agreement_pct} ===")
            res = run_ensemble_full_and_wf(cfg, agreement_percentile=agreement_pct)
            print("Full period:", res["full_period"])
