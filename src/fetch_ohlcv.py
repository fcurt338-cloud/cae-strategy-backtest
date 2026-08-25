"""
Download historical OHLCV (klines) from Binance for a list of symbols,
at 15m and 1h, from each symbol's listing date (or a floor date) to now.

Uses stdlib urllib only (requests/pip are broken on this machine).
Caches each symbol/interval to data/raw/{symbol}_{interval}.csv and is
resumable: re-running skips symbol/interval pairs already fully cached
up to (near) the current time.
"""
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE = "https://api.binance.com/api/v3/klines"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
}

FLOOR_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
FLOOR_MS = int(FLOOR_DATE.timestamp() * 1000)


def get_json(url, retries=6, backoff=1.5):
    last_err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 or e.code == 418:
                time.sleep(backoff * (i + 1) * 5)
            else:
                time.sleep(backoff)
        except Exception as e:
            last_err = e
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {retries} retries: {url} ({last_err})")


def fetch_klines(symbol, interval, start_ms, end_ms):
    """Yield raw kline rows between start_ms and end_ms (inclusive-ish), paginated."""
    cur = start_ms
    step = INTERVAL_MS[interval]
    while cur < end_ms:
        url = f"{BASE}?symbol={symbol}&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000"
        data = get_json(url)
        if not data:
            break
        for row in data:
            yield row
        last_open = data[-1][0]
        next_cur = last_open + step
        if next_cur <= cur:
            break
        cur = next_cur
        if len(data) < 1000:
            break
        time.sleep(0.15)


def csv_path(symbol, interval):
    return os.path.join(RAW_DIR, f"{symbol}_{interval}.csv")


def existing_last_ts(path):
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if row:
                last = int(row[0])
    return last


def download_symbol(symbol, interval, end_ms):
    path = csv_path(symbol, interval)
    last_ts = existing_last_ts(path)
    step = INTERVAL_MS[interval]
    start_ms = (last_ts + step) if last_ts else FLOOR_MS

    if start_ms >= end_ms:
        return 0

    write_header = not os.path.exists(path)
    n = 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n_trades"])
        for row in fetch_klines(symbol, interval, start_ms, end_ms):
            w.writerow([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]])
            n += 1
    return n


def download_universe(symbols, intervals=("15m", "1h")):
    end_ms = int(time.time() * 1000)
    total = len(symbols) * len(intervals)
    done = 0
    for interval in intervals:
        for symbol in symbols:
            done += 1
            try:
                n = download_symbol(symbol, interval, end_ms)
                print(f"[{done}/{total}] {symbol} {interval}: +{n} bars -> {csv_path(symbol, interval)}")
            except Exception as e:
                print(f"[{done}/{total}] {symbol} {interval}: FAILED ({e})")


if __name__ == "__main__":
    import csv as _csv
    cand_path = os.path.join(os.path.dirname(__file__), "..", "data", "universe", "candidates.csv")
    rows = list(_csv.DictReader(open(cand_path)))
    rows.sort(key=lambda r: -float(r["binance_24h_quote_volume_usd"]))
    top_symbols = [r["symbol"] for r in rows[:50]]
    print(f"Downloading {len(top_symbols)} symbols: {top_symbols}")
    download_universe(top_symbols)
