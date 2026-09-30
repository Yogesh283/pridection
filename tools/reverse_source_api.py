"""Reverse-engineer what WinGo source APIs actually publish vs client chart math."""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics import big_small_label
from api.client import primary_color
from data.database import Database
from models.predictor import MajorityWindowModel

H = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}
HIST = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
DEAR = "https://dearapi.tashanwin.fit/wingo30"
EXPORT = ROOT / "exports" / "api_source_reverse.json"


def get(url: str, timeout: int = 20) -> requests.Response:
    return requests.get(url, timeout=timeout, headers=H, allow_redirects=True)


def probe_sites() -> list[dict]:
    sites = [
        "https://tashanwin.co/",
        "https://www.tashanwin.co/",
        "https://tashanwin.com/",
        "https://www.tashanwin.com/",
        "https://dearapi.tashanwin.fit/",
        "https://dearapi.tashanwin.fit/wingo30",
        "https://dearapi.tashanwin.fit/wingo1",
        "https://dearapi.tashanwin.fit/wingo3",
        "https://dearapi.tashanwin.fit/wingo5",
    ]
    out = []
    for url in sites:
        row = {"url": url}
        try:
            r = get(url)
            text = r.text or ""
            row.update(
                {
                    "status": r.status_code,
                    "final_url": r.url,
                    "content_type": (r.headers.get("content-type") or "")[:60],
                    "bytes": len(r.content),
                    "snippet": text[:220].replace("\n", " "),
                }
            )
            scripts = re.findall(r"""src=["']([^"']+)["']""", text, flags=re.I)[:12]
            apis = re.findall(
                r"""https?://[^\s"'<>]+(?:api|lottery|draw|wingo)[^\s"'<>]*""",
                text,
                flags=re.I,
            )
            row["scripts"] = scripts
            row["inline_apis"] = list(dict.fromkeys(apis))[:10]
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
        print(f"SITE {url} -> {row.get('status', row.get('error'))}")
    return out


def scrape_source_apis() -> dict:
    ts = int(time.time() * 1000)
    hist = get(f"{HIST}?ts={ts}").json()
    dear = get(DEAR).json()
    rows = list((hist.get("data") or {}).get("list") or [])
    latest = rows[0] if rows else {}
    return {
        "history_api": {
            "url": HIST,
            "meta": {
                "pageNo": (hist.get("data") or {}).get("pageNo"),
                "totalPage": (hist.get("data") or {}).get("totalPage"),
                "totalCount": (hist.get("data") or {}).get("totalCount"),
                "fetched": len(rows),
            },
            "record_keys": list(latest.keys()) if latest else [],
            "latest": latest,
            "note": (
                "CDN JSON of SETTLED past draws only. "
                "No predict / no algorithm / no chart Missing fields."
            ),
        },
        "dear_api": {
            "url": DEAR,
            "record_keys": list(dear.keys()),
            "latest": dear,
            "note": (
                "Thin mirror of latest settled draw "
                "(issueNumber/number/colour). Not a prediction engine."
            ),
        },
        "same_period": str(latest.get("issueNumber")) == str(dear.get("issueNumber")),
        "same_number": str(latest.get("number")) == str(dear.get("number")),
    }


def client_side_rules(number: int) -> dict:
    """Exact rules the game UI applies after receiving a number (not predictive)."""
    n = int(number)
    if n in (0, 5):
        color = "violet+red" if n == 0 else "violet+green"
        primary = "RED" if n == 0 else "GREEN"
    elif n % 2 == 0:
        color = "red"
        primary = "RED"
    else:
        color = "green"
        primary = "GREEN"
    return {
        "number": n,
        "big_small": "BIG" if n >= 5 else "SMALL",
        "color_rule": color,
        "primary_color": primary,
        "source": "deterministic mapping from digit — not an AI/algorithm",
    }


def chart_from_history(nums: list[int], window: int = 100) -> dict:
    seq = nums[-window:]
    missing = {}
    freq = Counter(seq)
    for d in range(10):
        dist = next((i for i, x in enumerate(reversed(seq)) if x == d), len(seq))
        missing[d] = dist
    return {
        "window": len(seq),
        "missing": missing,
        "frequency": {str(k): freq.get(k, 0) for k in range(10)},
        "note": "Same stats as in-game Chart tab — computed AFTER results, not before.",
    }


def accuracy_on_our_db() -> dict:
    db = Database()
    rounds = db.get_rounds()
    if len(rounds) < 30:
        return {"rounds": len(rounds), "note": "not enough data"}
    model = MajorityWindowModel()
    sample = rounds[-200:] if len(rounds) > 200 else rounds
    bs_ok = col_ok = total = 0
    for i in range(15, len(sample)):
        pred = model.predict(sample[:i])
        actual = sample[i]
        a_bs = big_small_label(int(actual["number"]))
        a_col = primary_color(actual.get("color"))
        p_bs = max(pred["big_small"], key=pred["big_small"].get)
        p_col = max(pred["colors"], key=pred["colors"].get)
        total += 1
        bs_ok += int(p_bs == a_bs)
        if a_col:
            col_ok += int(p_col == a_col)
    return {
        "db_rounds": len(rounds),
        "tested": total,
        "big_small_accuracy_pct": round(100.0 * bs_ok / total, 2),
        "color_accuracy_pct": round(100.0 * col_ok / total, 2),
        "meaning": "Our code copying chart-style patterns — NOT website secret accuracy.",
    }


def main() -> int:
    print("=" * 60)
    print("  REVERSE: WinGo website / API — what is actually created?")
    print("=" * 60)

    print("\n[1] Probe related websites")
    sites = probe_sites()

    print("\n[2] Scrape source API payloads")
    apis = scrape_source_apis()
    print("  History keys:", apis["history_api"]["record_keys"])
    print("  History latest:", apis["history_api"]["latest"])
    print("  Dear keys:", apis["dear_api"]["record_keys"])
    print("  Dear latest:", apis["dear_api"]["latest"])
    print("  Same draw now:", apis["same_period"], apis["same_number"])

    latest_n = int(
        (apis["dear_api"]["latest"] or {}).get("number")
        or (apis["history_api"]["latest"] or {}).get("number")
        or 0
    )
    print("\n[3] Client mapping rules (what UI 'creates' from the digit)")
    rules = client_side_rules(latest_n)
    print(" ", rules)

    print("\n[4] Chart algorithm (client-side from last results)")
    hist_list = (apis["history_api"].get("latest") and []) or []
    # prefer DB for full window
    db = Database()
    rounds = db.get_rounds()
    nums = [int(r["number"]) for r in rounds] if rounds else []
    if not nums and apis["history_api"]["latest"]:
        # fallback: only API page (10 rows) — fetch already in apis meta
        ts = int(time.time() * 1000)
        rows = (get(f"{HIST}?ts={ts}").json().get("data") or {}).get("list") or []
        nums = [int(x["number"]) for x in reversed(rows)]
    chart = chart_from_history(nums)
    print(f"  window={chart['window']} missing={chart['missing']}")

    print("\n[5] Accuracy from OUR reverse of their public rules/patterns")
    acc = accuracy_on_our_db()
    print(" ", acc)

    report = {
        "sites": sites,
        "apis": apis,
        "client_rules": rules,
        "chart": chart,
        "accuracy": acc,
        "conclusion": {
            "api_creates_prediction": False,
            "api_publishes": "settled issueNumber + number + color only",
            "color_big_small": "deterministic from number (0-9)",
            "chart_missing_freq": "client-side stats on past draws",
            "secret_algo_found": False,
            "realistic_accuracy": "~50% Big/Small and Color",
        },
    }
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    EXPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {EXPORT}")
    print("\nCONCLUSION:")
    for k, v in report["conclusion"].items():
        print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
