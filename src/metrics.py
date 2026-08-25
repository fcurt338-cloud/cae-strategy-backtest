"""Performance metrics computed from a PortfolioBacktester run's outputs."""
import math
from typing import List

import numpy as np
import pandas as pd

from backtest import Trade


def trades_to_dataframe(trades: List[Trade], fee_pct: float, slip_pct: float) -> pd.DataFrame:
    rows = []
    for t in trades:
        if not t.exit_events:
            continue
        pnl = t.realized_pnl(fee_pct, slip_pct)
        notional = t.notional_entry(slip_pct)
        last_time = t.exit_events[-1][0]
        rows.append({
            "symbol": t.symbol,
            "entry_time": t.entry_time,
            "exit_time": last_time,
            "entry_price": t.entry_price,
            "size": t.size_initial,
            "notional": notional,
            "pnl": pnl,
            "pnl_pct_of_notional": pnl / notional if notional else np.nan,
            "holding_hours": t.holding_hours(),
            "exit_reason": t.exit_reason_final(),
            "n_fills": len(t.exit_events),
        })
    return pd.DataFrame(rows)


def compute_performance(equity_curve, trades_df: pd.DataFrame, starting_equity: float, bars_per_year_for_sharpe=None):
    result = {}
    if not equity_curve:
        return {"error": "no equity curve"}

    times_ns = np.array([t for t, _ in equity_curve])
    eq = np.array([e for _, e in equity_curve])
    final_equity = eq[-1] if len(eq) else starting_equity

    result["starting_equity"] = starting_equity
    result["final_equity"] = final_equity
    result["total_return_pct"] = (final_equity / starting_equity - 1) * 100

    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max
    result["max_drawdown_pct"] = drawdown.min() * 100 if len(drawdown) else 0.0

    n_trades = len(trades_df)
    result["n_trades"] = n_trades
    if n_trades == 0:
        result.update({
            "win_rate_pct": None, "profit_factor": None, "avg_winner": None,
            "avg_loser": None, "avg_holding_hours": None, "sharpe": None, "sortino": None,
        })
        return result

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    result["win_rate_pct"] = 100 * len(wins) / n_trades
    gross_win = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    result["profit_factor"] = (gross_win / gross_loss) if gross_loss > 0 else np.inf
    result["avg_winner"] = wins["pnl"].mean() if len(wins) else 0.0
    result["avg_loser"] = losses["pnl"].mean() if len(losses) else 0.0
    result["avg_holding_hours"] = trades_df["holding_hours"].mean()

    # Sharpe/Sortino from per-trade returns (pct of notional), annualized by trade frequency.
    rets = trades_df["pnl_pct_of_notional"].dropna()
    if len(rets) > 1 and rets.std() > 0:
        span_days = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).total_seconds() / 86400
        trades_per_year = n_trades / span_days * 365 if span_days > 0 else n_trades
        result["sharpe"] = (rets.mean() / rets.std()) * math.sqrt(max(trades_per_year, 1))
        downside = rets[rets < 0]
        if len(downside) > 1 and downside.std() > 0:
            result["sortino"] = (rets.mean() / downside.std()) * math.sqrt(max(trades_per_year, 1))
        else:
            result["sortino"] = None
    else:
        result["sharpe"] = None
        result["sortino"] = None

    result["exit_reason_counts"] = trades_df["exit_reason"].value_counts().to_dict()
    return result


def performance_by_year(trades_df: pd.DataFrame):
    if trades_df.empty:
        return {}
    trades_df = trades_df.copy()
    trades_df["year"] = trades_df["exit_time"].dt.year
    out = {}
    for year, grp in trades_df.groupby("year"):
        wins = grp[grp["pnl"] > 0]
        losses = grp[grp["pnl"] <= 0]
        gross_win = wins["pnl"].sum()
        gross_loss = -losses["pnl"].sum()
        out[int(year)] = {
            "n_trades": len(grp),
            "total_pnl": grp["pnl"].sum(),
            "win_rate_pct": 100 * len(wins) / len(grp) if len(grp) else None,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else np.inf,
        }
    return out
