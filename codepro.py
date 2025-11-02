#!/usr/bin/env python3
"""
fetch_wallet_txs.py

Fetch all transactions for a Provenance wallet by paging through the Explorer API.

Usage:
  - Install dependencies:
      pip install requests

  - Run:
      PROVENANCE_API="https://explorer.provenance.io" \
      python fetch_wallet_txs.py \
        --address pb1tp4559l0gy6d5cv004gw6a3h3fqd6kcvzqhm6nzfey7p6hxn8qkqd70va9 \
        --from-date 2023-01-01 --to-date 2025-11-01 \
        --out wallet_txs.json --page-size 100

This script:
 - Pages through the Explorer endpoint /v2/txs/address/{address}
 - Automatically continues until no more pages or until reported totalPages is reached
 - Saves a JSON file with all fetched transaction summaries
 - Has retry/backoff and polite sleeping between requests
"""

from __future__ import annotations
import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Default wallet and dates (same values you provided earlier)
DEFAULT_WALLET = "pb1tp4559l0gy6d5cv004gw6a3h3fqd6kcvzqhm6nzfey7p6hxn8qkqd70va9"
DEFAULT_FROM = "2023-01-01"
DEFAULT_TO = "2025-11-01"
DEFAULT_BASE = os.getenv("PROVENANCE_API", "https://explorer.provenance.io")
DEFAULT_PAGE_SIZE = 100
DEFAULT_OUTPUT = "wallet_txs.json"


def get_session_with_retries(
    total_retries: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


def extract_items_and_paging(json_resp: Dict[str, Any]) -> Tuple[List[Any], Optional[int], Optional[int]]:
    """
    Inspect the JSON response and extract:
      - items: list of tx summaries (or empty)
      - page: current page if provided
      - total_pages: totalPages if provided

    Known shapes:
      { "results": [...], "page": 1, "totalPages": 5, ... }
      { "items": [...], "page": 1, "total_pages": 5, ... }
      { "data": { "results": [...], "page": 1, "totalPages": 5 } }
      Some APIs return list directly (rare here) -- then items == json_resp if it's a list.
    """
    # If the top-level response is a list, return it directly.
    if isinstance(json_resp, list):
        return json_resp, None, None

    # Common keys to look for
    candidates_for_items = ["results", "items", "data", "txs", "transactions"]
    # Try top-level keys first
    for key in candidates_for_items:
        if key in json_resp:
            val = json_resp[key]
            # Some APIs wrap results inside 'data' -> { results: [...] }
            if isinstance(val, list):
                items = val
                break
            if isinstance(val, dict):
                # inside data/results
                if "results" in val and isinstance(val["results"], list):
                    items = val["results"]
                    # try to get paging from val
                    page = val.get("page") or val.get("current_page") or val.get("currentPage")
                    total_pages = val.get("totalPages") or val.get("total_pages") or val.get("totalPages")
                    return items, (page if isinstance(page, int) else None), (total_pages if isinstance(total_pages, int) else None)
                # if val itself looks like a single object, not the list we want, continue
            # continue searching
    else:
        # If no candidates matched, try known direct keys
        items = json_resp.get("results") or json_resp.get("items") or []

    # get paging information from top-level if present
    page = json_resp.get("page") or json_resp.get("current_page") or json_resp.get("currentPage")
    total_pages = json_resp.get("totalPages") or json_resp.get("total_pages") or json_resp.get("totalPages") or json_resp.get("totalPages")

    # Defensive: ensure items is a list
    if not isinstance(items, list):
        # fallback: sometimes the structure is { "data": { "items": [...] } }
        items = []
        if isinstance(json_resp.get("data"), dict):
            items = json_resp["data"].get("results") or json_resp["data"].get("items") or []
    return items, (page if isinstance(page, int) else None), (total_pages if isinstance(total_pages, int) else None)


def fetch_page(
    session: requests.Session,
    base_url: str,
    address: str,
    page: int = 1,
    count: int = 100,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    api_prefix: str = "/v2",
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Fetch a single page and return the parsed JSON.
    Raises requests.HTTPError for non-2xx after retries are exhausted.
    """
    # build URL
    # expected: {BASE}/v2/txs/address/{address}
    url = f"{base_url.rstrip('/')}{api_prefix}/txs/address/{address}"
    params = {"count": count, "page": page}
    if from_date:
        params["fromDate"] = from_date
    if to_date:
        params["toDate"] = to_date

    resp = session.get(url, params=params, timeout=timeout)
    # raise_for_status only after retries; we want to capture JSON on error sometimes but keep behavior simple:
    resp.raise_for_status()
    return resp.json()


def fetch_all_transactions(
    base_url: str,
    address: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sleep_between_requests: float = 0.15,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    session = get_session_with_retries()
    all_items: List[Dict[str, Any]] = []
    page = 1

    while True:
        try:
            resp_json = fetch_page(
                session=session,
                base_url=base_url,
                address=address,
                page=page,
                count=page_size,
                from_date=from_date,
                to_date=to_date,
            )
        except requests.HTTPError as e:
            # If the server returns an error (e.g., 404 for unknown address), propagate useful message.
            raise RuntimeError(f"HTTP error while fetching page {page}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error while fetching page {page}: {e}") from e

        items, current_page, total_pages = extract_items_and_paging(resp_json)

        # If we couldn't detect items, try interpreting the entire resp as a single list
        if items is None:
            items = []

        # debug/diagnostic print
        got = len(items)
        cp = current_page or page
        tp = total_pages or "unknown"
        print(f"Fetched page {cp} -> {got} items (totalPages={tp})")

        if got == 0:
            # no more items -> break
            break

        all_items.extend(items)

        # if API reports total pages, and we've reached it -> stop
        if isinstance(total_pages, int) and isinstance(current_page, int):
            if current_page >= total_pages:
                break

        # if max_pages argument provided, stop at max_pages
        if max_pages and page >= max_pages:
            print(f"Reached max_pages limit ({max_pages}); stopping early.")
            break

        page += 1
        time.sleep(sleep_between_requests)

    return all_items


def save_json(data: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} items to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch all Provenance wallet transactions via the Explorer API.")
    p.add_argument("--base-url", default=DEFAULT_BASE, help=f"Explorer base URL (default env PROVENANCE_API or {DEFAULT_BASE})")
    p.add_argument("--address", required=False, default=DEFAULT_WALLET, help=f"Wallet address (default {DEFAULT_WALLET})")
    p.add_argument("--from-date", required=False, default=DEFAULT_FROM, help="Start date (ISO YYYY-MM-DD) inclusive")
    p.add_argument("--to-date", required=False, default=DEFAULT_TO, help="End date (ISO YYYY-MM-DD) inclusive")
    p.add_argument("--out", "-o", default=DEFAULT_OUTPUT, help="Output JSON file path (default wallet_txs.json)")
    p.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Page size (count) per request")
    p.add_argument("--sleep", type=float, default=0.15, help="Seconds to sleep between requests (default 0.15)")
    p.add_argument("--max-pages", type=int, default=0, help="Optional: stop after this many pages (0 = no limit)")
    return p.parse_args()


def main():
    args = parse_args()
    max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else None

    print(f"Base URL: {args.base_url}")
    print(f"Address : {args.address}")
    print(f"From    : {args.from_date}")
    print(f"To      : {args.to_date}")
    print(f"PageSize: {args.page_size}")
    print(f"Output  : {args.out}")
    print()

    try:
        all_txs = fetch_all_transactions(
            base_url=args.base_url,
            address=args.address,
            from_date=args.from_date,
            to_date=args.to_date,
            page_size=args.page_size,
            sleep_between_requests=args.sleep,
            max_pages=max_pages,
        )
    except Exception as e:
        print(f"Failed to fetch transactions: {e}")
        return

    save_json(all_txs, args.out)


if __name__ == "__main__":
    main()
