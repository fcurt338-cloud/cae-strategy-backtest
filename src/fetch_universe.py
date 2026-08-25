"""
Build the low-cap USDT universe for the CAE backtest.

Data sources (no third-party HTTP libs needed — stdlib urllib only,
because this machine's `requests`/`pip` install is corrupted):
  - Binance /api/v3/exchangeInfo   -> list of live USDT spot pairs
  - Binance /api/v3/ticker/24hr    -> current 24h quote volume (coarse liquidity screen)
  - CoinGecko /coins/markets       -> current market cap (used as a *current-snapshot*
                                       low-cap screen — see caveat in README)

Output: data/universe/candidates.csv with columns:
  symbol, base, market_cap_usd, cg_id, cg_24h_volume, binance_24h_quote_volume
"""
import json
import time
import urllib.request
import urllib.error
import csv
import os

HEADERS = {"User-Agent": "Mozilla/5.0"}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "universe")
os.makedirs(OUT_DIR, exist_ok=True)

STABLECOIN_BASES = {
    "USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDP", "BUSD", "EUR", "EURI",
    "GUSD", "USDD", "USDE", "PYUSD", "FRAX", "LUSD", "SUSD", "USTC", "AEUR",
}


def get_json(url, retries=5, backoff=2.0):
    last_err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                time.sleep(backoff * (i + 1) * 3)
            else:
                time.sleep(backoff)
        except Exception as e:
            last_err = e
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {retries} retries: {url} ({last_err})")


def fetch_binance_usdt_pairs():
    d = get_json("https://api.binance.com/api/v3/exchangeInfo")
    pairs = []
    for s in d["symbols"]:
        if (
            s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
            and s["isSpotTradingAllowed"]
            and s["baseAsset"] not in STABLECOIN_BASES
        ):
            pairs.append(s["symbol"])
    return sorted(pairs)


def fetch_binance_24h_volumes(symbols):
    d = get_json("https://api.binance.com/api/v3/ticker/24hr")
    vol = {x["symbol"]: float(x["quoteVolume"]) for x in d if x["symbol"] in set(symbols)}
    return vol


def fetch_coingecko_markets(max_pages=12, per_page=250, min_mcap=1_000_000, max_mcap=100_000_000):
    """Fetch coins sorted by market cap desc, stop once we're well past the low-cap band."""
    all_coins = []
    consecutive_below_floor = 0
    for page in range(1, max_pages + 1):
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            f"?vs_currency=usd&order=market_cap_desc&per_page={per_page}&page={page}&sparkline=false"
        )
        try:
            d = get_json(url)
        except RuntimeError as e:
            print(f"  page {page} failed permanently: {e}")
            break
        if not d:
            break
        all_coins.extend(d)
        below = sum(1 for c in d if c.get("market_cap") and c["market_cap"] < min_mcap)
        if below > per_page * 0.9:
            consecutive_below_floor += 1
        else:
            consecutive_below_floor = 0
        if consecutive_below_floor >= 2:
            break
        time.sleep(1.5)
    return all_coins


def build_universe():
    print("Fetching Binance USDT spot pairs...")
    binance_symbols = fetch_binance_usdt_pairs()
    print(f"  {len(binance_symbols)} live USDT spot pairs")

    print("Fetching Binance 24h volumes...")
    vol24h = fetch_binance_24h_volumes(binance_symbols)

    print("Fetching CoinGecko market caps (paginated, rate-limited)...")
    coins = fetch_coingecko_markets()
    print(f"  {len(coins)} coins fetched from CoinGecko")

    cg_by_symbol = {}
    for c in coins:
        sym = c["symbol"].upper()
        # keep the highest-market-cap entry if a ticker collides
        if sym not in cg_by_symbol or (c.get("market_cap") or 0) > (cg_by_symbol[sym].get("market_cap") or 0):
            cg_by_symbol[sym] = c

    rows = []
    for sym in binance_symbols:
        base = sym[: -len("USDT")]
        cg = cg_by_symbol.get(base)
        if not cg:
            continue
        mcap = cg.get("market_cap")
        if mcap is None or mcap < 1_000_000 or mcap > 100_000_000:
            continue
        bvol = vol24h.get(sym, 0.0)
        rows.append(
            {
                "symbol": sym,
                "base": base,
                "market_cap_usd": mcap,
                "cg_id": cg["id"],
                "cg_24h_volume_usd": cg.get("total_volume") or 0,
                "binance_24h_quote_volume_usd": bvol,
            }
        )

    rows.sort(key=lambda r: -r["binance_24h_quote_volume_usd"])

    out_path = os.path.join(OUT_DIR, "candidates.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "symbol", "base", "market_cap_usd", "cg_id", "cg_24h_volume_usd", "binance_24h_quote_volume_usd"
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} candidate low-cap USDT pairs -> {out_path}")
    return rows


if __name__ == "__main__":
    build_universe()
