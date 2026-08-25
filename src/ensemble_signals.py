"""
Signal combination: require agreement across all three independently-tested
signal families before considering a symbol a long candidate. This is a
mechanically different bet from any single signal -- the hypothesis is that
each signal alone is too noisy, but simultaneous agreement filters out the
noise. It is NOT a blended average score (which would let one strong signal
paper over two weak/disagreeing ones); a symbol must independently clear the
bar on each signal.

Directions used are the primary, a-priori theoretically-motivated hypothesis
for each signal family -- chosen before looking at which direction had the
least-bad standalone number, specifically to avoid re-introducing the
overfitting risk this project has been careful about all night:
  - momentum: continuation (long trailing winners) -- the standard hypothesis
  - funding: contrarian (long most-negative funding, crowded-short squeeze)
    -- the more established effect in funding-rate-arbitrage literature
  - order-flow: continuation (long highest taker-buy ratio, aggressive
    buying persists) -- the standard order-flow-momentum hypothesis
"""
import pandas as pd

from momentum_signals import daily_closes, compute_returns_at
from funding_signals import load_funding_wide, compute_funding_rank_at
from orderflow_signals import load_orderflow_wide, compute_orderflow_rank_at


def build_ensemble_rank_fn(symbol_dfs, momentum_lookback=14, funding_lookback=3, orderflow_lookback=3,
                            agreement_percentile=0.5):
    """Returns a callable(as_of_date) -> pd.Series of combined scores for symbols that
    clear `agreement_percentile` (default: above-median) on ALL THREE signals
    independently. Score for qualifying symbols = mean of their three percentile ranks
    (used only to order within the agreeing set, not to let one signal compensate for
    another failing the threshold)."""
    wide_close = daily_closes(symbol_dfs)
    wide_funding = load_funding_wide()
    wide_orderflow = load_orderflow_wide()

    def rank_fn(d):
        mom = compute_returns_at(wide_close, d, momentum_lookback).dropna()
        fund = compute_funding_rank_at(wide_funding, d, funding_lookback, direction="contrarian").dropna()
        flow = compute_orderflow_rank_at(wide_orderflow, d, orderflow_lookback, direction="aggressive_buy").dropna()

        common = mom.index.intersection(fund.index).intersection(flow.index)
        if len(common) == 0:
            return pd.Series(dtype=float)

        mom_pct = mom.loc[common].rank(pct=True)
        fund_pct = fund.loc[common].rank(pct=True)
        flow_pct = flow.loc[common].rank(pct=True)

        qualifies = (mom_pct >= agreement_percentile) & (fund_pct >= agreement_percentile) & (flow_pct >= agreement_percentile)
        agreeing = common[qualifies]
        if len(agreeing) == 0:
            return pd.Series(dtype=float)

        combined = (mom_pct[agreeing] + fund_pct[agreeing] + flow_pct[agreeing]) / 3.0
        return combined

    return rank_fn
