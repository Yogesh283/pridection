"""Hang on WinGo history API — poll fast, print the instant a new result appears.

Best practical way to KNOW the settled result early (not predict it).
Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics import big_small_label
from data.database import Database

HIST = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
DEAR = "https://dearapi.tashanwin.fit/wingo30"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def fetch_hist() -> dict | None:
    url = f"{HIST}?ts={int(time.time() * 1000)}"
    r = requests.get(url, timeout=8, headers=HEADERS)
    r.raise_for_status()
    rows = (r.json().get("data") or {}).get("list") or []
    return rows[0] if rows else None


def fetch_dear() -> dict | None:
    j = requests.get(DEAR, timeout=8, headers=HEADERS).json()
    if not j.get("issueNumber"):
        return None
    return {
        "issueNumber": j.get("issueNumber"),
        "number": j.get("number"),
        "color": j.get("colour") or j.get("color"),
    }


def store(db: Database | None, row: dict) -> None:
    if db is None:
        return
    try:
        n = int(row["number"])
        db.upsert_round(
            period=str(row["issueNumber"]),
            number=n,
            color=str(row.get("color") or "").lower() or None,
            timestamp=None,
            raw_json=json.dumps(row, ensure_ascii=False),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  db note: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Hang-poll WinGo history API")
    ap.add_argument("--interval", type=float, default=0.4, help="poll seconds")
    ap.add_argument("--minutes", type=float, default=0, help="0 = forever")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--compare-dear", action="store_true", help="also show dear lag")
    args = ap.parse_args()

    db = None if args.no_db else Database()
    last: str | None = None
    deadline = time.time() + args.minutes * 60 if args.minutes > 0 else None
    print("=" * 56)
    print("  HANG API — WinGo_30S history poller")
    print(f"  URL: {HIST}")
    print(f"  poll every {args.interval}s | Ctrl+C stop")
    print("=" * 56)

    try:
        while True:
            if deadline and time.time() >= deadline:
                print("time limit reached")
                break
            try:
                row = fetch_hist()
            except Exception as exc:  # noqa: BLE001
                print(f"[{now()}] hist error: {exc}")
                time.sleep(max(0.5, args.interval))
                continue

            if not row:
                time.sleep(args.interval)
                continue

            period = str(row.get("issueNumber") or "")
            if period and period != last:
                n = int(row["number"])
                color = row.get("color")
                bs = big_small_label(n)
                tag = "NEW" if last is not None else "START"
                print(
                    f"[{now()}] {tag} period={period} "
                    f"number={n} {bs} color={color}"
                )
                store(db, row)

                if args.compare_dear:
                    try:
                        d = fetch_dear()
                        if d:
                            same = str(d.get("issueNumber")) == period
                            print(
                                f"         dear: period={d.get('issueNumber')} "
                                f"n={d.get('number')} same={same}"
                            )
                    except Exception as exc:  # noqa: BLE001
                        print(f"         dear error: {exc}")

                last = period
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
