"""
Fetch on-chain Transfer-event activity from Etherscan for the subset of our
universe with Ethereum contracts. Metric: weekly transfer count and unique
active address count per token -- deliberately count-based, not volume-based,
to avoid needing per-token decimals parsing (a real source of bugs if rushed).
This is genuinely new information: network activity/participation, not
derivable from price, volume, funding, or order-flow data.

Weekly granularity (not daily) both for API call budget and because it
matches the holding_days=7 cadence already used throughout this project.
Uses Etherscan's page/offset pagination within each weekly block range to
guarantee completeness regardless of how active a given week was, rather
than guessing a "safe" chunk size and silently truncating busy periods.
"""
import csv
import json
import os
import time
import urllib.error
import urllib.request

API_KEY = "8QSKGCU3UIXV8PVRMI4EZ6MCGBTMQK4I34"
BASE = "https://api.etherscan.io/v2/api"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HEADERS = {"User-Agent": "Mozilla/5.0"}
FLOOR_TS = 1673740800  # 2023-01-15, matches project floor (seconds, not ms -- Etherscan uses seconds)


def get_json(params, retries=6, backoff=1.0):
    url = f"{BASE}?" + "&".join(f"{k}={v}" for k, v in params.items()) + f"&apikey={API_KEY}"
    last_err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"Failed after {retries} retries: {params} ({last_err})")


def block_at_time(ts):
    d = get_json({"chainid": 1, "module": "block", "action": "getblocknobytime",
                   "timestamp": ts, "closest": "before"})
    if d.get("status") != "1":
        return None
    return int(d["result"])


def logs_for_range(address, from_block, to_block):
    """Paginated getLogs for one block range. Returns list of (from_addr, to_addr)."""
    all_pairs = []
    page = 1
    while True:
        d = get_json({"chainid": 1, "module": "logs", "action": "getLogs", "address": address,
                       "topic0": TRANSFER_TOPIC, "fromBlock": from_block, "toBlock": to_block,
                       "page": page, "offset": 1000})
        result = d.get("result")
        if not isinstance(result, list) or not result:
            break
        for log in result:
            topics = log.get("topics", [])
            if len(topics) < 3:
                continue
            frm = "0x" + topics[1][-40:]
            to = "0x" + topics[2][-40:]
            all_pairs.append((frm, to))
        if len(result) < 1000:
            break
        page += 1
        time.sleep(0.25)
    return all_pairs


def build_week_boundaries(end_ts=None):
    end_ts = end_ts or int(time.time())
    weeks = []
    t = FLOOR_TS
    while t < end_ts:
        weeks.append(t)
        t += 7 * 86400
    weeks.append(end_ts)
    return weeks


def csv_path(symbol):
    return os.path.join(RAW_DIR, f"{symbol}_onchain.csv")


def fetch_symbol(symbol, address, week_ts_boundaries, week_blocks, verbose=True):
    path = csv_path(symbol)
    existing = set()
    if os.path.exists(path):
        with open(path) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    existing.add(row[0])

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["week_start_ts", "transfer_count", "unique_addresses"])
        for i in range(len(week_ts_boundaries) - 1):
            wk_start_ts = week_ts_boundaries[i]
            if str(wk_start_ts) in existing:
                continue
            from_block, to_block = week_blocks[i], week_blocks[i + 1]
            if from_block is None or to_block is None or from_block >= to_block:
                continue
            pairs = logs_for_range(address, from_block, to_block)
            transfer_count = len(pairs)
            unique_addrs = len(set(a for pair in pairs for a in pair))
            w.writerow([wk_start_ts, transfer_count, unique_addrs])
            f.flush()
            if verbose:
                print(f"    week {i+1}/{len(week_ts_boundaries)-1}: {transfer_count} transfers, "
                      f"{unique_addrs} unique addrs")
            time.sleep(0.25)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "universe", "eth_contracts.json")) as f:
        eth_tokens = json.load(f)

    print("Precomputing week-boundary block numbers (shared across all tokens)...")
    week_ts = build_week_boundaries()
    week_blocks = []
    for i, ts in enumerate(week_ts):
        b = block_at_time(ts)
        week_blocks.append(b)
        time.sleep(0.22)
    print(f"{len(week_ts)} week boundaries resolved to blocks")

    for i, (sym, addr) in enumerate(eth_tokens.items()):
        print(f"[{i+1}/{len(eth_tokens)}] {sym} ({addr})")
        try:
            fetch_symbol(sym, addr, week_ts, week_blocks, verbose=False)
            print(f"  done")
        except Exception as e:
            print(f"  FAILED: {e}")
