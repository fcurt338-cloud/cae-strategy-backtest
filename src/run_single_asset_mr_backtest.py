import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from run_backtest import load_symbol_df, apply_listing_age_filter, apply_liquidity_filter
from single_asset_mr_signals import SingleAssetMRConfig
from backtest_single_asset_mr import SingleAssetMRBacktester


def load_majors():
    majors = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'LINK', 'DOT', 'LTC', 'BCH', 'UNI',
              'ATOM', 'NEAR', 'APT', 'ARB', 'OP', 'FIL', 'ICP', 'ETC', 'XLM', 'HBAR', 'SUI', 'INJ', 'TIA',
              'RENDER', 'TRX', 'SHIB']
    symbols = {}
    for base in majors:
        sym = base + 'USDT'
        df = load_symbol_df(sym, '1h')
        if df is None or len(df) < 2000:
            continue
        df = apply_listing_age_filter(df, 14)
        df = apply_liquidity_filter(df, 100_000, 24)
        if df is not None:
            symbols[sym] = df
    return symbols


if __name__ == "__main__":
    symbols = load_majors()
    print(f"{len(symbols)} symbols loaded")

    cfg = SingleAssetMRConfig()
    bt = SingleAssetMRBacktester(cfg, symbols)
    equity_curve, trades = bt.run()

    n = len(trades)
    wins = sum(1 for t in trades if t.realized_pnl(cfg.taker_fee_pct, cfg.slippage_pct) > 0)
    total_pnl = sum(t.realized_pnl(cfg.taker_fee_pct, cfg.slippage_pct) for t in trades)
    print(f"\n=== FULL PERIOD ===")
    print(f"n_trades={n}  win_rate={100*wins/n:.1f}%  total_pnl=${total_pnl:.0f}  final_equity=${10_000+total_pnl:.0f}")

    reasons = Counter(t.exit_reason for t in trades)
    print(f"exit reasons: {dict(reasons)}")

    print(f"\n=== BY YEAR ===")
    by_year = defaultdict(list)
    for t in trades:
        pnl = t.realized_pnl(cfg.taker_fee_pct, cfg.slippage_pct)
        by_year[t.entry_time.year].append(pnl)
    for year in sorted(by_year):
        pnls = by_year[year]
        w = sum(1 for p in pnls if p > 0)
        print(f"  {year}: n={len(pnls):4d}  win_rate={100*w/len(pnls):.1f}%  pnl=${sum(pnls):.0f}")

    print(f"\n=== BY SYMBOL (concentration check) ===")
    by_sym = defaultdict(list)
    for t in trades:
        pnl = t.realized_pnl(cfg.taker_fee_pct, cfg.slippage_pct)
        by_sym[t.symbol].append(pnl)
    sym_totals = sorted(by_sym.items(), key=lambda kv: -sum(kv[1]))
    for sym, pnls in sym_totals:
        w = sum(1 for p in pnls if p > 0)
        print(f"  {sym:10s} n={len(pnls):4d}  win_rate={100*w/len(pnls):5.1f}%  pnl=${sum(pnls):8.0f}")
