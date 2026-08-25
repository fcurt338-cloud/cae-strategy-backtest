"""
Parallel wrapper around fetch_ohlcv.download_symbol using a thread pool.
Each symbol/interval download is I/O bound (mostly waiting on HTTP round trips,
worsened by VPN latency), so threads give a large wall-clock speedup without
coming close to Binance's real rate limit (1200 request-weight/min).
"""
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fetch_ohlcv import download_symbol, RAW_DIR

CAND_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "universe", "candidates.csv")


def main(n_symbols=50, n_workers=8, intervals=("15m", "1h")):
    rows = list(csv.DictReader(open(CAND_PATH)))
    rows.sort(key=lambda r: -float(r["binance_24h_quote_volume_usd"]))
    symbols = [r["symbol"] for r in rows[:n_symbols]]

    jobs = [(sym, interval) for interval in intervals for sym in symbols]
    end_ms = int(time.time() * 1000)

    print(f"Downloading {len(symbols)} symbols x {len(intervals)} intervals = {len(jobs)} jobs, {n_workers} workers")

    def work(job):
        sym, interval = job
        try:
            n = download_symbol(sym, interval, end_ms)
            return (sym, interval, n, None)
        except Exception as e:
            return (sym, interval, 0, str(e))

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            sym, interval, n, err = fut.result()
            done += 1
            if err:
                print(f"[{done}/{len(jobs)}] {sym} {interval}: FAILED ({err})")
            else:
                print(f"[{done}/{len(jobs)}] {sym} {interval}: +{n} bars", flush=True)


if __name__ == "__main__":
    main()
