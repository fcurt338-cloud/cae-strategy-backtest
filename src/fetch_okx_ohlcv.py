"""
Fetch 1h OHLCV from OKX for the large-cap universe test (BTC, ETH, and other
majors) -- OKX instead of Binance since the VPN needed for Binance isn't
reliably connected. Applies the pagination lesson learned earlier tonight:
OKX's `after` parameter pages OLDER (into the past), `before` pages NEWER --
inverted from the intuitive reading, confirmed the hard way on the funding-
rate fetch.
"""
import csv
import json
import os
import time
import urllib.error
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0"}
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
FLOOR_MS = 1673740800000  # 2023-01-15, matches project floor


def get_json(url, retries=6, backoff=1.0):
    last_err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"Failed after {retries} retries: {url} ({last_err})")


def csv_path(base_symbol):
    return os.path.join(RAW_DIR, f"{base_symbol}USDT_1h.csv")


def download_symbol(inst_id, base_symbol, end_ms=None):
    """inst_id like 'BTC-USDT'. Writes to the SAME {SYMBOL}USDT_1h.csv naming
    convention as the Binance fetcher, so it drops straight into load_universe()."""
    path = csv_path(base_symbol)
    if os.path.exists(path):
        # already have it (from the original low-cap fetch or a prior run) -- skip
        with open(path) as f:
            if sum(1 for _ in f) > 100:
                return 0

    end_ms = end_ms or int(time.time() * 1000)
    rows = []
    after = None
    while True:
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst_id}&bar=1H&limit=300"
        if after:
            url += f"&after={after}"
        d = get_json(url)
        data = d.get("data", [])
        if not data:
            break
        for row in data:
            ts = int(row[0])
            if ts < FLOOR_MS:
                continue
            rows.append(row)
        oldest = int(data[-1][0])
        if oldest < FLOOR_MS:
            break
        after = oldest
        time.sleep(0.12)

    rows.sort(key=lambda r: int(r[0]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n_trades"])
        for row in rows:
            ts = int(row[0])
            w.writerow([ts, row[1], row[2], row[3], row[4], row[5], ts + 3599999, row[6] if len(row) > 6 else "", 0])
    return len(rows)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    majors = ['ETH','SOL','BNB','XRP','ADA','DOGE','AVAX','LINK','DOT','LTC','BCH','UNI',
              'ATOM','NEAR','APT','ARB','OP','FIL','ICP','ETC','XLM','HBAR','SUI','INJ','TIA',
              'RENDER','TRX','SHIB']  # BTC already cached from the original low-cap fetch
    print(f"Fetching {len(majors)} large-cap symbols from OKX (parallel)")

    def work(base):
        try:
            n = download_symbol(f"{base}-USDT", base)
            return (base, n, None)
        except Exception as e:
            return (base, 0, str(e))

    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(work, b) for b in majors]
        for fut in as_completed(futures):
            base, n, err = fut.result()
            done += 1
            if err:
                print(f"[{done}/{len(majors)}] {base}: FAILED ({err})")
            else:
                print(f"[{done}/{len(majors)}] {base}: +{n} rows", flush=True)
