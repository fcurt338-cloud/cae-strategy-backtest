"""Test the on-chain activity ranking signal, walk-forward from the start."""
import pandas as pd

from run_backtest import load_universe
from momentum_signals import MomentumConfig
from backtest_momentum import run_momentum_backtest
from onchain_signals import load_onchain_wide, compute_onchain_rank_at
from run_momentum_backtest import compute_period_metrics


def run_onchain_full_and_wf(cfg: MomentumConfig, direction: str, metric="transfer_count",
                             baseline_weeks=8, n_folds=4, starting_equity=10_000.0, verbose=True):
    symbol_dfs = load_universe("1h")
    wide_metric = load_onchain_wide(metric=metric)
    if verbose:
        print(f"Universe: {len(symbol_dfs)} symbols priced, {len(wide_metric.columns)} with on-chain data")

    rank_fn = lambda d: compute_onchain_rank_at(wide_metric, d, baseline_weeks=baseline_weeks, direction=direction)

    full_curve, full_events, _ = run_momentum_backtest(
        symbol_dfs, cfg, starting_equity, partial_rebalance=True, rank_fn=rank_fn
    )
    full_metrics = compute_period_metrics(full_curve, starting_equity)

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
                  f"ret={m.get('total_return_pct')} hit_rate={m.get('hit_rate_pct')} sharpe={m.get('sharpe')}")

    return {"full_period": full_metrics, "folds": fold_results}


if __name__ == "__main__":
    for metric in ["transfer_count", "unique_addresses"]:
        for direction in ["activity_surge", "activity_fade"]:
            cfg = MomentumConfig(lookback_days=7, holding_days=7, top_k=8)
            print(f"\n=== metric={metric} direction={direction} ===")
            res = run_onchain_full_and_wf(cfg, direction, metric=metric)
            print("Full period:", res["full_period"])
