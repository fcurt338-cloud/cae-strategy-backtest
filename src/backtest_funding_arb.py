"""
Backtest for the funding-rate carry trade (see funding_arb_signals.py for the
mechanism). Two variants:

  - "always_on": hold the delta-neutral position (short perp / long spot) for
    every day funding data exists, for every symbol. Pays entry+exit cost
    once. This is the raw, unfiltered size of the opportunity.
  - "timed": only hold the position while the trailing N-day average funding
    rate is above `on_threshold`; go flat otherwise. Pays entry+exit cost
    every time the position opens/closes. Tests whether avoiding negative-
    or low-funding stretches is worth the extra round-trip costs.

Position sizing: equal-weight notional across all symbols with funding data
(fixed at equity / n_symbols, not rebalanced), since this is a market-neutral
carry strategy, not a risk-of-loss-driven one -- sizing isn't stop-distance
based the way the pairs trade was.

Cost model: matches the pairs-trading majors cost assumption (0.04% taker fee
+ 0.05% slippage), applied per leg per side -- i.e. 2 legs x 2 sides x
(fee+slip) = 4x(fee+slip) per full round trip, since both the spot and perp
leg need to be opened and closed.
"""
import pandas as pd

from funding_arb_signals import load_funding_wide


def backtest_always_on(funding_wide: pd.DataFrame, starting_equity: float = 10_000.0,
                        fee_pct: float = 0.0004, slip_pct: float = 0.0005):
    """Hold every symbol's carry position for its entire available history."""
    n_symbols = funding_wide.shape[1]
    notional_per_symbol = starting_equity / n_symbols
    cost_per_leg_side = fee_pct + slip_pct
    round_trip_cost_pct = 4 * cost_per_leg_side  # 2 legs x 2 sides

    results = []
    for sym in funding_wide.columns:
        series = funding_wide[sym].dropna()
        if series.empty:
            continue
        total_funding_pct = series.sum()
        net_pct = total_funding_pct - round_trip_cost_pct
        n_days = len(series)
        years = n_days / 365.0
        annualized_pct = net_pct / years if years > 0 else float("nan")
        results.append({
            "symbol": sym, "n_days": n_days,
            "gross_funding_pct": total_funding_pct * 100,
            "round_trip_cost_pct": round_trip_cost_pct * 100,
            "net_pct": net_pct * 100,
            "annualized_pct": annualized_pct * 100,
            "net_dollars": net_pct * notional_per_symbol,
        })

    total_net_dollars = sum(r["net_dollars"] for r in results)
    return results, total_net_dollars, notional_per_symbol


def backtest_timed(funding_wide: pd.DataFrame, trailing_window: int = 7, on_threshold: float = 0.0,
                    starting_equity: float = 10_000.0, fee_pct: float = 0.0004, slip_pct: float = 0.0005):
    """Only hold the carry position while trailing avg funding > on_threshold.
    Signal at day d uses trailing window ending at d-1 (no lookahead: you decide
    whether to hold TODAY based on funding observed strictly before today)."""
    n_symbols = funding_wide.shape[1]
    notional_per_symbol = starting_equity / n_symbols
    cost_per_side = fee_pct + slip_pct  # one leg-pair, one side (entry OR exit)
    entry_exit_cost_pct = 2 * cost_per_side  # 2 legs, one side each

    all_results = []
    for sym in funding_wide.columns:
        series = funding_wide[sym].dropna()
        if len(series) < trailing_window + 5:
            continue
        trailing_avg = series.rolling(trailing_window).mean().shift(1)  # strictly causal
        held = trailing_avg > on_threshold

        net_pct = 0.0
        n_trades = 0
        n_days_held = 0
        in_position = False
        for d in series.index:
            want_hold = bool(held.get(d, False)) if pd.notna(held.get(d, float("nan"))) else False
            if want_hold and not in_position:
                net_pct -= entry_exit_cost_pct
                n_trades += 1
                in_position = True
            elif not want_hold and in_position:
                net_pct -= entry_exit_cost_pct
                in_position = False
            if in_position:
                net_pct += series.loc[d]
                n_days_held += 1
        if in_position:  # close out at the end
            net_pct -= entry_exit_cost_pct

        n_days_total = len(series)
        years = n_days_total / 365.0
        annualized_pct = net_pct / years if years > 0 else float("nan")
        all_results.append({
            "symbol": sym, "n_days_total": n_days_total, "n_days_held": n_days_held,
            "pct_time_held": 100 * n_days_held / n_days_total if n_days_total else 0,
            "n_trades": n_trades, "net_pct": net_pct * 100, "annualized_pct": annualized_pct * 100,
            "net_dollars": net_pct * notional_per_symbol,
        })

    total_net_dollars = sum(r["net_dollars"] for r in all_results)
    return all_results, total_net_dollars, notional_per_symbol


if __name__ == "__main__":
    majors = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'LINK', 'DOT', 'LTC', 'BCH', 'UNI',
              'ATOM', 'NEAR', 'APT', 'ARB', 'OP', 'FIL', 'ICP', 'ETC', 'XLM', 'HBAR', 'SUI', 'INJ', 'TIA',
              'RENDER', 'TRX']
    symbols = [m + "USDT" for m in majors]
    funding_wide = load_funding_wide(symbols)
    print(f"funding data for {funding_wide.shape[1]} symbols, {funding_wide.shape[0]} days\n")

    print("=== ALWAYS-ON (naive, unfiltered) ===")
    results, total, notional = backtest_always_on(funding_wide)
    for r in sorted(results, key=lambda x: -x["net_pct"]):
        print(f"  {r['symbol']:10s} n_days={r['n_days']:5d}  gross={r['gross_funding_pct']:7.2f}%  "
              f"net={r['net_pct']:7.2f}%  annualized={r['annualized_pct']:6.2f}%  ${r['net_dollars']:8.1f}")
    print(f"\nTotal net P&L across {len(results)} symbols (${notional:.0f} notional each): ${total:.0f}")
