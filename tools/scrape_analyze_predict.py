"""Scrape WinGo APIs -> store in MySQL -> analyze -> next Big/Small + Color prediction.

Sources:
  1) https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json
  2) https://dearapi.tashanwin.fit/wingo30  (WINGO_API_URL)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.accuracy import compute_accuracy_report
from analysis.statistics import big_small_label, streak_length, summarize_rounds
from api.client import WingoAPIClient, primary_color
from data.database import Database
from models.ensemble import EnsemblePredictor, build_prediction_export, write_prediction_json
from models.predictor import MajorityWindowModel

HIST_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def scrape_history() -> tuple[list[dict], dict]:
    url = f"{HIST_URL}?ts={int(time.time() * 1000)}"
    r = requests.get(url, timeout=20, headers=HEADERS)
    r.raise_for_status()
    data = r.json().get("data") or {}
    rows = list(data.get("list") or [])
    meta = {
        "pageNo": data.get("pageNo"),
        "totalPage": data.get("totalPage"),
        "totalCount": data.get("totalCount"),
        "fetched": len(rows),
        "url": HIST_URL,
    }
    return rows, meta


def scrape_dear() -> dict | None:
    try:
        norm = WingoAPIClient().fetch_normalized()
        return norm.get("current")
    except Exception as exc:  # noqa: BLE001
        print(f"  dearapi note: {exc}")
        return None


def color_from_hist(row: dict) -> str | None:
    raw = row.get("color") or row.get("colour")
    if raw:
        return str(raw).lower()
    try:
        n = int(row["number"])
    except (KeyError, TypeError, ValueError):
        return None
    from config import COLOR_MAP

    mapped = COLOR_MAP.get(n)
    return mapped.lower() if mapped else None


def store_scraped(db: Database) -> dict:
    hist_rows, meta = scrape_history()
    hist_ins = 0
    for row in hist_rows:
        period = str(row.get("issueNumber") or "")
        if not period:
            continue
        if db.upsert_round(
            period=period,
            number=int(row["number"]),
            color=color_from_hist(row),
            timestamp=None,
            raw_json=json.dumps(row, ensure_ascii=False),
        ):
            hist_ins += 1

    dear = scrape_dear()
    dear_ins = 0
    if dear:
        if db.upsert_round(
            period=str(dear["period"]),
            number=int(dear["number"]),
            color=dear.get("color"),
            timestamp=dear.get("timestamp"),
            raw_json=dear.get("raw_json"),
        ):
            dear_ins += 1

    return {
        "history_meta": meta,
        "history_rows": hist_rows,
        "history_inserted": hist_ins,
        "dear": dear,
        "dear_inserted": dear_ins,
        "db_rounds": db.count_rounds(),
    }


def light_backtest(rounds: list[dict], window: int = 120) -> dict:
    """Fast majority-window walk-forward on last N rounds only."""
    sample = rounds[-window:] if len(rounds) > window else rounds
    if len(sample) < 20:
        return {"total": 0}
    model = MajorityWindowModel()
    bs_ok = col_ok = total = 0
    min_h = 10
    for i in range(min_h, len(sample)):
        pred = model.predict(sample[:i])
        actual = sample[i]
        actual_bs = big_small_label(int(actual["number"]))
        actual_col = primary_color(actual.get("color"))
        pred_bs = max(pred["big_small"], key=pred["big_small"].get)
        pred_col = max(pred["colors"], key=pred["colors"].get)
        total += 1
        bs_ok += int(pred_bs == actual_bs)
        if actual_col:
            col_ok += int(pred_col == actual_col)
    return {
        "total": total,
        "big_small_accuracy": round(100.0 * bs_ok / total, 2) if total else 0.0,
        "color_accuracy": round(100.0 * col_ok / total, 2) if total else 0.0,
        "window": len(sample),
    }


def analyze_data(rounds: list[dict]) -> dict:
    if not rounds:
        return {}
    summary = summarize_rounds(rounds)
    last20 = rounds[-20:]
    last50 = rounds[-50:]
    nums = [int(r["number"]) for r in rounds]
    bs = [big_small_label(n) for n in nums]
    cols = [primary_color(r.get("color")) for r in rounds]
    cols = [c for c in cols if c]

    missing = {}
    for d in range(10):
        dist = None
        for i, n in enumerate(reversed(nums[-100:])):
            if n == d:
                dist = i
                break
        missing[d] = dist if dist is not None else len(nums[-100:])

    due = min(missing, key=lambda d: -missing[d])
    return {
        "summary": summary,
        "last": rounds[-1],
        "last5": [
            {
                "period": r["period"][-6:],
                "n": int(r["number"]),
                "bs": big_small_label(int(r["number"])),
                "c": primary_color(r.get("color")),
            }
            for r in rounds[-5:]
        ],
        "bs_streak": streak_length(bs),
        "color_streak": streak_length(cols) if cols else 0,
        "bs_last20": dict(Counter(big_small_label(int(r["number"])) for r in last20)),
        "bs_last50": dict(Counter(big_small_label(int(r["number"])) for r in last50)),
        "color_last20": dict(
            Counter(c for c in (primary_color(r.get("color")) for r in last20) if c)
        ),
        "digit_missing": missing,
        "due_digit": due,
        "due_missing": missing[due],
        "backtest": light_backtest(rounds),
    }


def predict_next(db: Database, rounds: list[dict]) -> dict | None:
    from api.history_sync import guess_next_period, sync_history_into_db

    try:
        sync_history_into_db(db)
        rounds = db.get_rounds()
    except Exception as exc:  # noqa: BLE001
        print(f"  history sync note: {exc}")

    if len(rounds) < 5:
        return None
    latest = rounds[-1]
    current = str(latest["period"])
    target = guess_next_period(current)
    hist_bs = None
    resolved = db.get_resolved_predictions()
    if resolved:
        hist_bs = float(
            compute_accuracy_report(resolved).get("big_small_accuracy") or 0.0
        )
    result = EnsemblePredictor().predict(
        rounds,
        historical_big_small_accuracy=hist_bs,
        train_ml=True,
        focus="big_small",
    )
    if not result:
        return None
    write_prediction_json(
        build_prediction_export(target, result, current_period=current)
    )
    db.save_prediction(
        {
            "target_period": target,
            "predicted_number": result["top_number"],
            "predicted_color": result["top_color"],
            "predicted_big_small": result["top_big_small"],
            "number_probability": result["number_probability"],
            "color_probability": result["color_probability"],
            "big_small_probability": result["big_small_probability"],
            "model_name": "ensemble",
        },
        update_existing=True,
    )
    return {"target": target, "result": result, "latest": latest}


def main() -> int:
    print("=" * 56)
    print("  WINGO SCRAPE -> ANALYZE -> PREDICT")
    print("=" * 56)

    print("\n[1] SCRAPE WinGo APIs")
    db = Database()
    sync = store_scraped(db)
    meta = sync["history_meta"]
    print(f"  History API : {meta['url']}")
    print(f"  Hist rows   : {meta['fetched']} (totalCount={meta.get('totalCount')})")
    print(f"  Hist upsert : {sync['history_inserted']}")
    dear = sync["dear"]
    if dear:
        print(
            f"  Dear API    : period={dear['period']} "
            f"n={dear['number']} color={dear.get('color')}"
        )
    print(f"  Dear upsert : {sync['dear_inserted']}")
    print(f"  DB rounds   : {sync['db_rounds']}")

    rounds = db.get_rounds()
    print("\n[2] ANALYZE scraped data")
    report = analyze_data(rounds)
    if not report:
        print("  No rounds in DB.")
        return 1
    last = report["last"]
    print(
        f"  Latest      : {last['period']} -> {last['number']} "
        f"({big_small_label(int(last['number']))}, {primary_color(last.get('color'))})"
    )
    print(f"  Last 5      : {report['last5']}")
    print(f"  BS streak   : {report['bs_streak']} | Color streak: {report['color_streak']}")
    print(f"  BS last20   : {report['bs_last20']}")
    print(f"  BS last50   : {report['bs_last50']}")
    print(f"  Color last20: {report['color_last20']}")
    print(
        f"  Due digit   : {report['due_digit']} "
        f"(missing {report['due_missing']} in last 100)"
    )
    bt = report["backtest"]
    if bt.get("total"):
        print(
            f"  Fast BT     : BS {bt['big_small_accuracy']}% | "
            f"Color {bt['color_accuracy']}%  (n={bt['total']}, window={bt['window']})"
        )

    live = db.get_resolved_predictions()
    if live:
        acc = compute_accuracy_report(live)
        print(
            f"  Live settled: BS {acc.get('big_small_accuracy', 0):.1f}% | "
            f"Color {acc.get('color_accuracy', 0):.1f}% "
            f"({acc.get('total_predictions', 0)} preds)"
        )

    print("\n[3] NEXT PREDICTION")
    pred = predict_next(db, rounds)
    if not pred:
        print("  Failed — need more rounds.")
        return 1
    r = pred["result"]
    print(f"  Period      : {pred['target']}")
    print(f"  Big/Small   : {r['top_big_small']}")
    print(
        f"  Confidence  : {r['confidence_level']} "
        f"({r['confidence_score'] * 100:.0f}%)"
    )
    print(f"  Prob        : BS {r['big_small_probability'] * 100:.0f}%")
    if r.get("bs_source"):
        print(f"  Strategy    : {r['bs_source']}")
    print("\n  Saved -> prediction.json")
    print("  Note: Big/Small only — estimate ~50% long-run.")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
