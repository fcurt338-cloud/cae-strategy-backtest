# Low-cap crypto trend-strategy research — progress log

Read this fully before doing anything. This is a continuing research project; you're picking it up mid-stream. The user is asleep and unreachable — work autonomously, do not wait for input.

## Setup
- `pip install pandas numpy` (only deps needed; stdlib `urllib` used elsewhere, avoid `requests` if it errors).
- Cached 1h OHLCV for 50 low-cap USDT pairs + BTC is in `data/raw/*_1h.csv` (already in this repo). 15m data was NOT pushed (too large) — 1h has consistently outperformed 15m in every strategy tested, so this isn't a real loss. If you want 15m or fresher data and have working internet access in this environment, `src/fetch_ohlcv.py` / `src/fetch_ohlcv_parallel.py` can fetch it (Binance primary, OKX fallback) — but do not assume internet access works; verify before relying on it, and don't block on it if it doesn't.
- Code layout: `src/run_backtest.py` (universe loading + filters), `src/trend_signals.py` + `src/backtest_trend.py` (the promising strategy), `src/backtest_trend_pyramid.py` (tested, rejected), `src/sweep_trend_lookback.py` (lookback sweep pattern to reuse), `src/strategy.py`+`src/backtest.py` (CAE, rejected), `src/smc_*.py` (liquidity/SMT/FVG, rejected).

## What's been learned (three full strategies tested on this universe, 2023-2026, 15m+1h)

1. **CAE (coil breakout)**: catastrophic. -100%/-99.9999% return, win rate 3.85%/9.24%, PF 0.02/0.22, 0/50 symbols profitable. Root cause: ~46% of trades exit via false-breakout ("coil_reentry").
2. **SMT divergence + Fair Value Gap**: also catastrophic. -100%/-99.99999%, win rate 3.34%/8.28%, PF 0.024/0.051, 0/50 profitable. ~81% of trades exit via "fvg_invalidated" (price fills the gap and keeps going).
3. **Donchian trend-following (Turtle-style) — THE PROMISING ONE, CONTINUE THIS.** Entry: close breaks above highest high of prior N bars. Stop: entry - 2*ATR(20). Exit: NO fixed target, only a trailing stop (lowest low of prior M bars, ratchets up only). Best validated so far: **1h, entry=75 days (1800 bars), exit=30 days (720 bars), plus a regime filter (skip entries when close < 0.4 * trailing-180-day-high)**: profit_factor 0.854, win rate 7.33%, breakeven-win-rate gap only -1.1pp (win rate needed ~8.4%), 8/31 symbols profitable, 2 of 4 years net profitable (2024 PF 1.14, 2026 PF 1.76). Still net negative overall (-53% total return) — closest to breakeven found, not a working strategy yet.

Config for the best-known result:
```python
TrendConfig(entry_lookback=1800, exit_lookback=720, stop_atr_mult=2.0, atr_period=20,
            regime_lookback_bars=4320, regime_min_pct_of_high=0.4)
TrendEngineConfig()  # defaults: risk_pct_per_trade=0.0075, max_concurrent_positions=4,
                      # taker_fee_pct=0.001, slippage_pct=0.015
```

Full 6-point lookback sweep (1h, no regime filter) — entry_lookback/exit_lookback in DAYS, PF = profit factor:
- 20d/10d: PF 0.451, gap -6.9pp
- 35d/15d: PF 0.732, gap -2.3pp
- 55d/20d: PF 0.408, gap -9.3pp
- **75d/30d: PF 0.795, gap -1.6pp** (best without filter)
- 100d/40d: PF 0.738, gap -2.5pp
- 130d/50d: PF 0.386, gap -8.0pp
Not monotonic — sweet spot around 75-100 days, falls off both sides. `src/sweep_trend_lookback.py` has this pattern, reusable.

Regime filter sweep (at 75d/30d) — lookback/floor:
- No filter: PF 0.795
- **180d lookback, 40% floor: PF 0.854 (best)**
- 180d lookback, 60% floor: PF 0.371 (much worse — too strict)
- 365d lookback, 40% floor: PF 0.710
- 365d lookback, 60% floor: PF 0.464
Loose filtering helps; strict filtering hurts a lot. Don't tighten past 40% without re-testing broadly.

**Pyramiding: tested properly, rejected.** `src/backtest_trend_pyramid.py` implements classic Turtle unit-adding (add a risk unit every K*ATR of favorable move, up to 4 units, shared trailing stop). Tested K=0.5, 1.5, 3.0 at 75d/30d — every spacing was WORSE than the no-pyramid baseline (PF 0.65-0.72 vs 0.795). Do not pursue further without a genuinely new idea for the mechanism — this exact approach doesn't work here.

**Stress-test discipline that must continue**: at 75d/30d, one trade (SUPERUSDT, Nov 2023-Jan 2024, +$7,221, a real 369% gain) drives most of the positive side. Hand-verified against raw daily closes — genuine gradual rally, not a data bug — but it means results lean heavily on rare, large winners with a thin sample (~32-50 symbols, ~3.5 years). Any new "promising" result needs the same treatment before being trusted: per-symbol distribution (not one lucky symbol), parameter sensitivity to nearby values (not just one winning cell), and hand-tracing a few actual trades against raw OHLCV.

## Priority queue (work through in order; check this file's "Session history" below first — don't redo completed items)

1. **Walk-forward validation**: split 2023-2026 into rolling train/test segments; fit/select the lookback+filter on a trailing window, test out-of-sample on the next segment, roll forward. 2024 and 2026 were profitable years, 2023 and 2025 were not — this looks regime-dependent. Check whether a fixed 75d/30d+filter config would have been selectable in advance, or whether it's fit to hindsight.
2. **Re-run the 6-point lookback sweep WITH the 180d/40% regime filter applied at every point** (it was only tested at 75d/30d so far).
3. If anything clears PF > 1.0 with 50+ trades: full stress test (per-symbol distribution, parameter sensitivity, hand-trace 5-10 trades) before calling it real. Update this log and the report honestly either way.
4. If the priority queue is exhausted with nothing further to try: write a clear "CONCLUDED" final summary (see below) and stop.

## Session history (newest last — append, don't overwrite)

- 2026-08-25 ~01:52 UTC: First cloud attempt failed immediately — that sandbox had zero general internet access (confirmed via Binance/OKX/CoinGecko/even google.com all EGRESS_BLOCKED). Correctly refused to fabricate results and stopped. No work done.
- 2026-08-25 ~03:00 UTC: Repo created and pushed with code + 1h OHLCV cache specifically to route around the network block via git clone (which may use a different provisioning path than the agent's own blocked runtime network). This routine now fires hourly. If you're reading this as a fresh run: try a small network test early (e.g. `pip install` already proves pypi.org works; try a Binance API call to see if data fetching is ALSO possible now, but don't block on it — the pushed 1h CSVs are sufficient to keep making real progress either way).

## When you're done for this session (each hourly fire)

1. Update this file's "Session history" with what you did and found (append, don't delete prior entries).
2. If you produced a new/updated `results/report_trend.html`, make sure it's committed (same design system: Fraunces/IBM Plex Sans/IBM Plex Mono headers already in the file — follow existing patterns, don't redesign).
3. `git add -A && git commit -m "..." && git push origin main` — this is how the user (and the next hourly run) sees your progress. If push fails (e.g. no credentials in this environment), that's fine — say so clearly in your session's final message, don't silently lose the work; at minimum leave the finished files in the working tree and explain the situation in your response.
4. If you've concluded the research (validated something real, or exhausted the priority queue with nothing further to try), write `## CONCLUDED` as a new top-level heading right under the main title of this file, with the final verdict, so the next hourly fire sees it immediately and exits without redoing work.
