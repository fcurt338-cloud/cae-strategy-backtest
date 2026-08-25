"""
Fetch historical funding rate data from Binance USDⓈ-M futures for the
subset of our universe with USDT perpetual listings. Standard forward
pagination via startTime (unlike OKX's inverted before/after convention).
"""
import csv
import glob
import json
import os
import time
import urllib.error
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE = "https://fapi.binance.com/fapi/v1/fundingRate"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
FLOOR_MS = 1673740800000  # 2023-01-15, matches our OHLCV floor


def get_json(url, retries=6, backoff=1.5):
    last_err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last_err = e
            time.sleep(backoff * (i + 1) * (5 if e.code in (429, 418) else 1))
        except Exception as e:
            last_err = e
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {retries} retries: {url} ({last_err})")


def get_perp_symbols():
    req = urllib.request.Request(
        "https://fapi.binance.com/fapi/v1/exchangeInfo", headers=HEADERS
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read())
    return {s["symbol"] for s in d["symbols"]
            if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}


def our_universe_symbols():
    files = glob.glob(os.path.join(RAW_DIR, "*_1h.csv"))
    return sorted(set(os.path.basename(f).replace("_1h.csv", "") for f in files) - {"BTCUSDT"})


def csv_path(symbol):
    return os.path.join(RAW_DIR, f"{symbol}_funding.csv")


def download_funding(symbol, end_ms=None):
    path = csv_path(symbol)
    last_ts = None
    if os.path.exists(path):
        with open(path) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    last_ts = int(row[0])

    end_ms = end_ms or int(time.time() * 1000)
    start_ms = (last_ts + 1) if last_ts else FLOOR_MS
    if start_ms >= end_ms:
        return 0

    write_header = not os.path.exists(path)
    n = 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["funding_time", "funding_rate"])
        cur = start_ms
        while cur < end_ms:
            url = f"{BASE}?symbol={symbol}&startTime={cur}&endTime={end_ms}&limit=1000"
            data = get_json(url)
            if not data:
                break
            for row in data:
                w.writerow([row["fundingTime"], row["fundingRate"]])
                n += 1
            last_time = int(data[-1]["fundingTime"])
            next_cur = last_time + 1
            if next_cur <= cur:
                break
            cur = next_cur
            if len(data) < 1000:
                break
            time.sleep(0.12)
    return n


if __name__ == "__main__":
    perp_symbols = get_perp_symbols()
    our_symbols = our_universe_symbols()
    matched = [s for s in our_symbols if s in perp_symbols]
    print(f"Fetching funding history for {len(matched)} symbols with Binance USDT perps")
    end_ms = int(time.time() * 1000)
    for i, symbol in enumerate(matched):
        try:
            n = download_funding(symbol, end_ms)
            print(f"[{i+1}/{len(matched)}] {symbol}: +{n} rows")
        except Exception as e:
            print(f"[{i+1}/{len(matched)}] {symbol}: FAILED ({e})")
