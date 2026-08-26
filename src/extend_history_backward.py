"""
Extend the cached 1h OHLCV further back in time than the original 2023-01-15
floor (which was an arbitrary constant inherited from the low-cap project,
not an actual data or exchange limit). Continues paginating OKX history-
candles BACKWARD from the oldest timestamp already in each symbol's CSV,
down to a much earlier floor (2018-01-01, before any of these symbols
existed, so it naturally stops at each symbol's real listing date), then
merges (prepends) the new older rows into the existing file.

Same OKX pagination lesson as fetch_okx_ohlcv.py: `after` pages older.
"""
import csv
import json
import os
import time
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0"}
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
NEW_FLOOR_MS = 1514764800000  # 2018-01-01 -- earlier than any of these symbols existed


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


def oldest_ts_in_file(path):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)
        return int(first_row[0])


def extend_symbol(base_symbol, verbose=True):
    inst_id = f"{base_symbol}-USDT"
    path = csv_path(base_symbol)
    if not os.path.exists(path):
        return (base_symbol, 0, "no existing file")

    start_before = oldest_ts_in_file(path)
    older_rows = []
    after = start_before
    while True:
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst_id}&bar=1H&limit=300&after={after}"
        try:
            d = get_json(url)
        except Exception as e:
            return (base_symbol, len(older_rows), f"error mid-fetch: {e}")
        data = d.get("data", [])
        if not data:
            break
        for row in data:
            ts = int(row[0])
            if ts < NEW_FLOOR_MS or ts >= start_before:
                continue
            older_rows.append(row)
        oldest = int(data[-1][0])
        if oldest >= after:  # no progress, stop
            break
        if oldest < NEW_FLOOR_MS:
            break
        after = oldest
        time.sleep(0.12)

    if not older_rows:
        return (base_symbol, 0, None)

    older_rows.sort(key=lambda r: int(r[0]))
    with open(path) as f:
        existing_lines = f.readlines()
    header = existing_lines[0]
    existing_body = existing_lines[1:]

    new_lines = []
    for row in older_rows:
        ts = int(row[0])
        new_lines.append(f"{ts},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{ts+3599999},{row[6] if len(row) > 6 else ''},0\n")

    with open(path, "w", newline="") as f:
        f.write(header)
        f.writelines(new_lines)
        f.writelines(existing_body)

    return (base_symbol, len(older_rows), None)


if __name__ == "__main__":
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    majors = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'LINK', 'DOT', 'LTC', 'BCH', 'UNI',
              'ATOM', 'NEAR', 'APT', 'ARB', 'OP', 'FIL', 'ICP', 'ETC', 'XLM', 'HBAR', 'SUI', 'INJ', 'TIA',
              'RENDER', 'TRX', 'SHIB']
    if len(sys.argv) > 1:
        majors = sys.argv[1:]

    print(f"Extending {len(majors)} symbols backward from their current earliest date, floor=2018-01-01")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(extend_symbol, m) for m in majors]
        done = 0
        for fut in as_completed(futures):
            base, n, err = fut.result()
            done += 1
            if err:
                print(f"[{done}/{len(majors)}] {base}: +{n} older rows (error: {err})")
            else:
                print(f"[{done}/{len(majors)}] {base}: +{n} older rows", flush=True)
