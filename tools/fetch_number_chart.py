"""Build chart-style Number Statistic (last N) from history API + MySQL DB.

The in-game chart (Missing / Avg missing / Frequency / Max consecutive)
is computed client-side from recent results — there is no public
GetNoaverageEmerdList on draw.ar-lottery01 for WinGo_30S.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics import big_small_label
from api.client import primary_color
from data.database import Database

HIST_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
WINDOW = 100


def fetch_history_api() -> list[dict]:
    url = f"{HIST_URL}?ts={int(time.time() * 1000)}"
    r = requests.get(url, timeout=20, headers=HEADERS)
    r.raise_for_status()
    rows = (r.json().get("data") or {}).get("list") or []
    out = []
    for row in rows:
        period = str(row.get("issueNumber") or "")
        if not period:
            continue
        number = int(row["number"])
        color = str(row.get("color") or row.get("colour") or "")
        out.append(
            {
                "period": period,
                "number": number,
                "color": color,
                "big_small": big_small_label(number),
                "source": "history_api",
            }
        )
    return out


def sync_history_into_db(db: Database, rows: list[dict]) -> int:
    n = 0
    for row in rows:
        ok = db.upsert_round(
            period=row["period"],
            number=int(row["number"]),
            color=row.get("color"),
            timestamp=None,
            raw_json=json.dumps(row, ensure_ascii=False),
        )
        if ok:
            n += 1
    return n


def _max_consecutive(seq: list[int], digit: int) -> int:
    best = cur = 0
    for x in seq:
        if x == digit:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def build_chart(rounds: list[dict], window: int = WINDOW) -> dict:
    """
    Same fields as the game UI Statistic tab for last `window` periods.
    rounds must be chronological ascending (oldest -> newest).
    """
    if not rounds:
        return {"window": window, "digits": {}, "recent": []}

    tail = rounds[-window:]
    nums = [int(r["number"]) for r in tail]
    n = len(nums)

    digits: dict[str, dict] = {}
    for d in range(10):
        positions = [i for i, x in enumerate(nums) if x == d]
        freq = len(positions)
        if not positions:
            missing = n  # never in window
            avg_missing = float(n)
        else:
            missing = (n - 1) - positions[-1]
            # gaps between consecutive appearances (+ leading gap optional)
            gaps = []
            prev = -1
            for pos in positions:
                gaps.append(pos - prev - 1)
                prev = pos
            # trailing gap to end is `missing`; avg missing in UI usually
            # averages inter-appearance gaps (and sometimes includes edges).
            avg_missing = sum(gaps) / len(gaps) if gaps else float(n)
        digits[str(d)] = {
            "missing": int(missing),
            "avg_missing": round(float(avg_missing), 2),
            "frequency": int(freq),
            "max_consecutive": int(_max_consecutive(nums, d)),
        }

    recent = []
    for r in reversed(tail[-20:]):
        num = int(r["number"])
        col = primary_color(r.get("color")) or str(r.get("color") or "")
        recent.append(
            {
                "period": str(r["period"]),
                "number": num,
                "color": col,
                "big_small": big_small_label(num),
            }
        )

    return {
        "window": n,
        "requested_window": window,
        "latest_period": str(tail[-1]["period"]),
        "digits": digits,
        "recent": recent,
    }


def print_chart(chart: dict) -> None:
    digits = chart["digits"]
    print(f"Statistic (last {chart['window']} Periods)")
    print(f"Latest period: {chart.get('latest_period')}")
    print("")
    header = "          " + " ".join(f"{d:>4}" for d in range(10))
    print(header)
    missing = "Missing   " + " ".join(
        f"{digits[str(d)]['missing']:>4}" for d in range(10)
    )
    avg = "Avg miss  " + " ".join(
        f"{digits[str(d)]['avg_missing']:>4.0f}" for d in range(10)
    )
    freq = "Frequency " + " ".join(
        f"{digits[str(d)]['frequency']:>4}" for d in range(10)
    )
    mx = "Max consec" + " ".join(
        f"{digits[str(d)]['max_consecutive']:>4}" for d in range(10)
    )
    print(missing)
    print(avg)
    print(freq)
    print(mx)
    print("")
    print("Recent periods (newest first):")
    for row in chart["recent"][:10]:
        print(
            f"  {row['period']}  num={row['number']}  "
            f"{row['big_small'][0]}  {row['color']}"
        )


def main() -> int:
    print("=== Fetch history API + build chart from DB ===")
    db = Database()
    try:
        hist = fetch_history_api()
        inserted = sync_history_into_db(db, hist)
        print(f"History API rows : {len(hist)} (inserted/updated flag={inserted})")
    except Exception as exc:  # noqa: BLE001
        print(f"History API note : {exc}")

    rounds = db.get_rounds()
    print(f"DB rounds        : {len(rounds)}")
    if len(rounds) < 10:
        print("DB mein data kam hai.")
        return 1

    chart = build_chart(rounds, WINDOW)
    print("")
    print_chart(chart)

    out = ROOT / "exports" / "number_chart.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chart, indent=2), encoding="utf-8")
    print("")
    print(f"Saved: {out}")
    print(
        "Note: game UI chart API is not public on this host; "
        "stats are computed from DB last 100 (same formulas)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
