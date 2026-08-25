"""
Orchestrates: load cached OHLCV -> run CAE backtest -> print/save results.

Usage:
    python run_backtest.py --timeframe 15m
    python run_backtest.py --timeframe 1h
    python run_backtest.py --timeframe 15m --sweep
"""
import argparse
import glob
import json
import os

import pandas as pd

from config import default_config_for_timeframe
from backtest import PortfolioBacktester
from metrics import trades_to_dataframe, compute_performance, performance_by_year

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_BARS_REQUIRED = 200  # need enough history for warmup (coil windows, ATR, upper-range lookback)


def load_symbol_df(symbol, timeframe):
    path = os.path.join(RAW_DIR, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def discover_symbols(timeframe):
    files = glob.glob(os.path.join(RAW_DIR, f"*_{timeframe}.csv"))
    syms = [os.path.basename(f)[: -len(f"_{timeframe}.csv")] for f in files]
    return sorted(s for s in syms if s != "BTCUSDT")


def apply_listing_age_filter(df, min_days=14):
    """Drop the first min_days of a symbol's own history (listing-age requirement)."""
    if df is None or df.empty:
        return df
    cutoff = df.index[0] + pd.Timedelta(days=min_days)
    trimmed = df[df.index >= cutoff]
    return trimmed if len(trimmed) >= MIN_BARS_REQUIRED else None


def apply_liquidity_filter(df, min_avg_daily_usd=100_000, bars_per_day=96):
    """Require trailing-7d average daily $ volume >= threshold.

    Important: this must NOT remove rows from the series — doing so would create
    calendar gaps that silently corrupt every rolling computation (coil lookback,
    ATR) computed downstream on positional windows. Instead it adds a boolean
    'liquidity_ok' column, which the backtester ANDs into entry_signal so illiquid
    periods simply can't trigger entries while indicators still see continuous data.
    """
    if df is None or df.empty:
        return None
    dollar_vol = df["close"] * df["volume"]
    daily = dollar_vol.resample("1D").sum()
    roll7 = daily.rolling(7, min_periods=7).mean()
    qualifying_days = set(roll7[roll7 >= min_avg_daily_usd].index.date)
    if not qualifying_days:
        return None
    df = df.copy()
    df["liquidity_ok"] = df.index.to_series().dt.date.isin(qualifying_days).values
    return df if df["liquidity_ok"].any() else None


def load_universe(timeframe):
    symbols = discover_symbols(timeframe)
    data = {}
    for sym in symbols:
        df = load_symbol_df(sym, timeframe)
        df = apply_listing_age_filter(df, min_days=14)
        df = apply_liquidity_filter(df, min_avg_daily_usd=100_000, bars_per_day=(96 if timeframe == "15m" else 24))
        if df is not None:
            data[sym] = df
    return data


def load_btc(timeframe):
    return load_symbol_df("BTCUSDT", timeframe)


def run_one(timeframe, cfg_overrides=None, starting_equity=10_000.0, verbose=True):
    cfg = default_config_for_timeframe(timeframe)
    if cfg_overrides:
        cfg = cfg.clone(**cfg_overrides)

    symbol_dfs = load_universe(timeframe)
    btc_df = load_btc(timeframe)
    if verbose:
        print(f"[{timeframe}] {len(symbol_dfs)} symbols pass listing-age + liquidity filters")

    bt = PortfolioBacktester(cfg, symbol_dfs, btc_df=btc_df, starting_equity=starting_equity)
    equity_curve, trades = bt.run()

    trades_df = trades_to_dataframe(trades, cfg.taker_fee_pct, cfg.slippage_pct)
    perf = compute_performance(equity_curve, trades_df, starting_equity)
    by_year = performance_by_year(trades_df)

    return {
        "timeframe": timeframe,
        "n_symbols": len(symbol_dfs),
        "symbols": sorted(symbol_dfs.keys()),
        "performance": perf,
        "by_year": by_year,
        "trades_df": trades_df,
        "equity_curve": equity_curve,
        "config": cfg,
    }


def print_report(res):
    p = res["performance"]
    print("=" * 70)
    print(f"Timeframe: {res['timeframe']}   Symbols in universe: {res['n_symbols']}")
    print("=" * 70)
    for k in ["starting_equity", "final_equity", "total_return_pct", "max_drawdown_pct",
              "n_trades", "win_rate_pct", "profit_factor", "avg_winner", "avg_loser",
              "avg_holding_hours", "sharpe", "sortino"]:
        print(f"  {k:24s}: {p.get(k)}")
    print("  exit_reason_counts     :", p.get("exit_reason_counts"))
    print("-" * 70)
    print("By year:")
    for y, stats in sorted(res["by_year"].items()):
        print(f"  {y}: n={stats['n_trades']:4d}  pnl=${stats['total_pnl']:.2f}  "
              f"win_rate={stats['win_rate_pct']:.1f}%  pf={stats['profit_factor']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", choices=["15m", "1h"], required=True)
    ap.add_argument("--equity", type=float, default=10_000.0)
    args = ap.parse_args()

    res = run_one(args.timeframe, starting_equity=args.equity)
    print_report(res)

    out_path = os.path.join(RESULTS_DIR, f"trades_{args.timeframe}.csv")
    res["trades_df"].to_csv(out_path, index=False)
    print(f"Saved trades -> {out_path}")

    eq_path = os.path.join(RESULTS_DIR, f"equity_{args.timeframe}.csv")
    eq_df = pd.DataFrame(res["equity_curve"], columns=["time_ns", "equity"])
    eq_df["time"] = pd.to_datetime(eq_df["time_ns"])
    eq_df[["time", "equity"]].to_csv(eq_path, index=False)
    print(f"Saved equity curve -> {eq_path}")

    summary_path = os.path.join(RESULTS_DIR, f"summary_{args.timeframe}.json")
    perf_serializable = {k: (v if not isinstance(v, dict) else v) for k, v in res["performance"].items()}
    with open(summary_path, "w") as f:
        json.dump({"performance": perf_serializable, "by_year": res["by_year"], "n_symbols": res["n_symbols"]}, f, indent=2, default=str)
    print(f"Saved summary -> {summary_path}")
