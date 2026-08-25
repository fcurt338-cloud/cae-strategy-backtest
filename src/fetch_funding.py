"""
Fetch historical funding rate data from OKX for the subset of our universe
that has USDT-settled perpetual swaps. Funding rate is genuinely different
information from OHLCV price action: it reflects the cost longs pay shorts
(or vice versa) to hold a perpetual position, which directly measures how
crowded/one-sided current positioning is -- something no amount of candle
pattern analysis can see.
"""
import csv
import glob
import json
import os
import time
import urllib.error
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE = "https://www.okx.com/api/v5/public/funding-rate-history"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def get_json(url, retries=6, backoff=1.5):
    last_err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last_err = e
            time.sleep(backoff * (i + 1) * (5 if e.code == 429 else 1))
        except Exception as e:
            last_err = e
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {retries} retries: {url} ({last_err})")


def get_swap_bases():
    req = urllib.request.Request(
        "https://www.okx.com/api/v5/public/instruments?instType=SWAP", headers=HEADERS
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read())
    return {x["instId"].split("-")[0]: x["instId"] for x in d["data"]
            if x["settleCcy"] == "USDT" and x["state"] == "live"}


def our_universe_bases():
    files = glob.glob(os.path.join(RAW_DIR, "*_1h.csv"))
    bases = sorted(set(os.path.basename(f).replace("USDT_1h.csv", "") for f in files) - {"BTC"})
    return bases


def csv_path(inst_id):
    return os.path.join(RAW_DIR, f"{inst_id.replace('-', '_')}_funding.csv")


def download_funding(inst_id, end_ms=None):
    path = csv_path(inst_id)
    existing_times = set()
    if os.path.exists(path):
        with open(path) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    existing_times.add(int(row[0]))

    end_ms = end_ms or int(time.time() * 1000)
    rows = []
    after = None  # OKX pagination convention: "after" = older records (paginate into the past);
                  # "before" = newer records -- opposite of the intuitive reading.
    while True:
        url = f"{BASE}?instId={inst_id}&limit=100"
        if after:
            url += f"&after={after}"
        data = get_json(url)["data"]
        if not data:
            break
        for row in data:
            ft = int(row["fundingTime"])
            rows.append((ft, float(row["fundingRate"])))
        oldest = int(data[-1]["fundingTime"])
        after = oldest
        time.sleep(0.12)
        if len(rows) > 20000:  # safety cap
            break

    new_rows = [(t, r) for t, r in rows if t not in existing_times]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["funding_time", "funding_rate"])
        for t, r in sorted(new_rows):
            w.writerow([t, r])
    return len(new_rows), (min(t for t, _ in rows) if rows else None)


if __name__ == "__main__":
    swap_map = get_swap_bases()
    our_bases = our_universe_bases()
    matched = [(b, swap_map[b]) for b in our_bases if b in swap_map]
    print(f"Fetching funding history for {len(matched)} symbols with OKX perp swaps")
    for i, (base, inst_id) in enumerate(matched):
        try:
            n, oldest = download_funding(inst_id)
            oldest_str = time.strftime("%Y-%m-%d", time.gmtime(oldest / 1000)) if oldest else "n/a"
            print(f"[{i+1}/{len(matched)}] {inst_id}: +{n} rows, oldest so far {oldest_str}")
        except Exception as e:
            print(f"[{i+1}/{len(matched)}] {inst_id}: FAILED ({e})")
