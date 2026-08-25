"""
Event study: for every symbol, align its own price history to "days since
listing" (day 0 = first available bar), then compute the cross-sectional
average/median return from day D to day D+H for a range of D values. This
reveals the SHAPE of any post-listing effect using every symbol's full
trajectory, rather than betting the whole test on one arbitrarily-chosen
entry offset.

Uses log returns for correct cross-sectional averaging (arithmetic averaging
of simple returns is upward-biased for volatile assets -- the same variance-
drain issue diagnosed in the momentum test earlier tonight).
"""
import numpy as np
import pandas as pd

from run_backtest import load_symbol_df


def build_days_since_listing_matrix(symbols, max_days=120):
    """Returns a DataFrame: rows = day-since-listing (0..max_days), columns = symbol,
    values = daily close price. Day 0 = the symbol's own first available 1h bar's day."""
    data = {}
    for sym in symbols:
        df = load_symbol_df(sym, "1h")
        if df is None or df.empty:
            continue
        daily = df["close"].resample("1D").last().dropna()
        if len(daily) < 5:
            continue
        vals = daily.to_numpy()[:max_days + 1]
        data[sym] = pd.Series(vals, index=np.arange(len(vals)))
    return pd.DataFrame(data)


def forward_return_by_day(price_matrix: pd.DataFrame, holding_days: int):
    """For each day D (row) and symbol (col), the log return from day D to day D+holding_days.
    NaN where either endpoint is missing (symbol's history doesn't reach that far)."""
    shifted = price_matrix.shift(-holding_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(shifted / price_matrix)
    return log_ret


def event_study_summary(symbols, holding_days=7, max_days=90):
    price_matrix = build_days_since_listing_matrix(symbols, max_days=max_days + holding_days + 5)
    log_ret = forward_return_by_day(price_matrix, holding_days)
    rows = []
    for d in range(0, max_days + 1):
        if d not in log_ret.index:
            continue
        row = log_ret.loc[d].dropna()
        if len(row) < 5:  # need a reasonable cross-section to trust the average
            continue
        rows.append({
            "day": d,
            "n_symbols": len(row),
            "mean_log_return_pct": row.mean() * 100,
            "median_log_return_pct": row.median() * 100,
            "pct_positive": (row > 0).mean() * 100,
            "std_pct": row.std() * 100,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import glob, os
    files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "*_1h.csv"))
    symbols = sorted(set(os.path.basename(f).replace("_1h.csv", "") for f in files) - {"BTCUSDT"})
    print(f"{len(symbols)} symbols")

    for holding_days in [3, 7, 14, 21]:
        print(f"\n=== forward {holding_days}-day return by days-since-listing ===")
        summary = event_study_summary(symbols, holding_days=holding_days, max_days=90)
        # print every 7th day to keep it scannable
        for _, row in summary.iterrows():
            if int(row["day"]) % 7 == 0:
                print(f"  day {int(row['day']):3d}: n={int(row['n_symbols']):2d}  "
                      f"mean={row['mean_log_return_pct']:7.2f}%  median={row['median_log_return_pct']:7.2f}%  "
                      f"pct_pos={row['pct_positive']:5.1f}%  std={row['std_pct']:6.2f}%")
