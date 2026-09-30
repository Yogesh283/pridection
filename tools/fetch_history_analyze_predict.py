"""Fetch history API + dearapi into MySQL, then analyze and predict Color/Big-Small."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.accuracy import compute_accuracy_report
from analysis.statistics import big_small_label, summarize_rounds
from api.client import WingoAPIClient, primary_color
from data.database import Database
from models.backtest import backtest_history
from models.backtest import _guess_next_period
from models.ensemble import EnsemblePredictor, build_prediction_export, write_prediction_json

HIST_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_history_rows() -> tuple[list[dict], dict]:
    url = f"{HIST_URL}?ts={int(time.time() * 1000)}"
    r = requests.get(url, timeout=20, headers=HEADERS)
    r.raise_for_status()
    data = (r.json().get("data") or {})
    rows = list(data.get("list") or [])
    meta = {
        "pageNo": data.get("pageNo"),
        "totalPage": data.get("totalPage"),
        "totalCount": data.get("totalCount"),
        "fetched": len(rows),
    }
    return rows, meta


def color_from_row(row: dict) -> str | None:
    raw = row.get("color") or row.get("colour")
    if raw:
        return str(raw).lower()
    try:
        n = int(row["number"])
    except (KeyError, TypeError, ValueError):
        return None
    # Fallback same as project COLOR_MAP primary
    from config import COLOR_MAP

    mapped = COLOR_MAP.get(n)
    return mapped.lower() if mapped else None


def upsert_history(db: Database) -> dict:
    rows, meta = fetch_history_rows()
    inserted = 0
    for row in rows:
        period = str(row.get("issueNumber") or "")
        if not period:
            continue
        number = int(row["number"])
        color = color_from_row(row)
        ok = db.upsert_round(
            period=period,
            number=number,
            color=color,
            timestamp=None,
            raw_json=json.dumps(row, ensure_ascii=False),
        )
        if ok:
            inserted += 1

    # Also pull dearapi current.
    dear_inserted = 0
    try:
        client = WingoAPIClient()
        norm = client.fetch_normalized()
        current = norm.get("current")
        if current:
            ok = db.upsert_round(
                period=str(current["period"]),
                number=int(current["number"]),
                color=current.get("color"),
                timestamp=current.get("timestamp"),
                raw_json=current.get("raw_json"),
            )
            if ok:
                dear_inserted += 1
    except Exception as exc:  # noqa: BLE001
        print(f"dearapi note: {exc}")

    return {
        "history_meta": meta,
        "history_rows": len(rows),
        "history_inserted": inserted,
        "dear_inserted": dear_inserted,
        "db_rounds": db.count_rounds(),
    }


def analyze(db: Database) -> dict:
    rounds = db.get_rounds()
    summary = summarize_rounds(rounds) if rounds else {}
    resolved = db.get_resolved_predictions()
    live_acc = compute_accuracy_report(resolved) if resolved else {}
    bt = {}
    # Full ensemble walk-forward on 900+ rounds hangs; use last 120 only.
    if len(rounds) >= 40:
        sample = rounds[-120:] if len(rounds) > 120 else rounds
        bt = backtest_history(
            sample,
            min_history=min(30, max(10, len(sample) // 4)),
            use_ensemble=True,
        )
    return {
        "rounds": len(rounds),
        "summary": summary,
        "live_acc": live_acc,
        "backtest": bt,
    }


def predict_next(db: Database) -> dict | None:
    rounds = db.get_rounds()
    if len(rounds) < 5:
        return None
    latest = rounds[-1]
    target = _guess_next_period(str(latest["period"]))
    resolved = db.get_resolved_predictions()
    hist_bs = None
    if resolved:
        hist_bs = float(compute_accuracy_report(resolved).get("big_small_accuracy") or 0.0)
    result = EnsemblePredictor().predict(
        rounds,
        historical_big_small_accuracy=hist_bs,
        train_ml=True,
        focus="big_small",
    )
    if not result:
        return None
    write_prediction_json(build_prediction_export(target, result))
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
    print("=== 1) FETCH history API + dearapi -> DB ===")
    db = Database()
    sync = upsert_history(db)
    print(f"History API rows : {sync['history_rows']}")
    print(f"History meta     : {sync['history_meta']}")
    print(f"Inserted hist    : {sync['history_inserted']}")
    print(f"Inserted dearapi : {sync['dear_inserted']}")
    print(f"DB rounds now    : {sync['db_rounds']}")
    print(
        "Note: public history file only returns last ~10 rows "
        "(totalCount shown but pageNo not open)."
    )

    print("")
    print("=== 2) ANALYZE DB ===")
    report = analyze(db)
    summary = report["summary"]
    print(f"Stored rounds    : {report['rounds']}")
    if summary:
        print(f"Big ratio        : {summary.get('big_ratio')}")
        print(f"Color frequency  : {summary.get('color_frequency')}")
    live = report["live_acc"]
    if live:
        print(
            f"Live settled BS  : {live.get('big_small_accuracy', 0):.2f}% "
            f"({live.get('correct_big_small', 0)}/{live.get('total_predictions', 0)})"
        )
        print(
            f"Live settled Col : {live.get('color_accuracy', 0):.2f}% "
            f"({live.get('correct_colors', 0)}/{live.get('total_predictions', 0)})"
        )
    bt = report["backtest"]
    if bt:
        print(
            f"Backtest BS      : {bt.get('big_small_accuracy', 0):.2f}% "
            f"n={bt.get('total_predictions', 0)}"
        )
        print(
            f"Backtest Color   : {bt.get('color_accuracy', 0):.2f}% "
            f"n={bt.get('total_predictions', 0)}"
        )

    print("")
    print("=== 3) NEXT PREDICTION (Color + Big/Small) ===")
    pred = predict_next(db)
    if not pred:
        print("Prediction failed — not enough rounds.")
        return 1
    r = pred["result"]
    print(f"Period     : {pred['target']}")
    print(f"Big/Small  : {r['top_big_small']}")
    print(f"Color      : {r['top_color']}")
    print(
        f"Confidence : {r['confidence_level']} ({r['confidence_score'] * 100:.0f}%)"
    )
    print(
        f"Chance     : BS {r['big_small_probability'] * 100:.0f}% | "
        f"Color {r['color_probability'] * 100:.0f}%"
    )
    if r.get("bs_source"):
        print(f"Strategy   : {r['bs_source']}")
    print("Note: estimate only — more history pages are not public on this API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
