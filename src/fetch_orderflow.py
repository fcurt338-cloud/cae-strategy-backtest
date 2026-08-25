"""
Fetch taker buy volume from Binance 1h klines -- a field present in the raw
kline response (index 9: taker_buy_base_asset_volume) that wasn't saved by
the original OHLCV fetcher. This is genuinely different information from
price/volume totals: it tells you what FRACTION of traded volume was driven
by aggressive buyers (market-order buys hitting the ask) vs aggressive
sellers, i.e. who was more urgent to trade, not just how much traded.

Stores only [open_time, volume, taker_buy_base_volume] -- volume is included
for convenience (buy_ratio = taker_buy_base_volume / volume) even though it
duplicates the existing OHLCV cache, since joining is easier this way than
re-deriving.
"""
import csv
import glob
import os
import time

from fetch_ohlcv import get_json, RAW_DIR, INTERVAL_MS, FLOOR_MS


def csv_path(symbol):
    return os.path.join(RAW_DIR, f"{symbol}_orderflow.csv")


def existing_last_ts(path):
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if row:
                last = int(row[0])
    return last


def download_orderflow(symbol, interval="1h", end_ms=None):
    path = csv_path(symbol)
    last_ts = existing_last_ts(path)
    step = INTERVAL_MS[interval]
    start_ms = (last_ts + step) if last_ts else FLOOR_MS
    end_ms = end_ms or int(time.time() * 1000)
    if start_ms >= end_ms:
        return 0

    write_header = not os.path.exists(path)
    n = 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["open_time", "volume", "taker_buy_base_volume"])
        cur = start_ms
        while cur < end_ms:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000"
            data = get_json(url)
            if not data:
                break
            for row in data:
                w.writerow([row[0], row[5], row[9]])
                n += 1
            last_open = data[-1][0]
            next_cur = last_open + step
            if next_cur <= cur:
                break
            cur = next_cur
            if len(data) < 1000:
                break
            time.sleep(0.12)
    return n


def our_universe_symbols():
    files = glob.glob(os.path.join(RAW_DIR, "*_1h.csv"))
    return sorted(set(os.path.basename(f).replace("_1h.csv", "") for f in files) - {"BTCUSDT"} | {"BTCUSDT"})


if __name__ == "__main__":
    symbols = sorted(our_universe_symbols())
    print(f"Fetching order-flow data for {len(symbols)} symbols")
    end_ms = int(time.time() * 1000)
    for i, sym in enumerate(symbols):
        try:
            n = download_orderflow(sym, "1h", end_ms)
            print(f"[{i+1}/{len(symbols)}] {sym}: +{n} rows")
        except Exception as e:
            print(f"[{i+1}/{len(symbols)}] {sym}: FAILED ({e})")
