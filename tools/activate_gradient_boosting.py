"""Activate Gradient Boosting as production Big/Small authority.

Runs chronological walk-forward smoke test, trains on full history,
emits a live prediction, and writes the production report.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.history_sync import guess_next_period
from config import (
    EXPORTS_DIR,
    GB_LEARNING_RATE,
    GB_MAX_DEPTH,
    GB_MIN_HISTORY,
    GB_N_ESTIMATORS,
    GB_RETRAIN_EVERY,
    PREDICTION_MODEL,
)
from data.database import Database
from models.ensemble import EnsemblePredictor, build_prediction_export, write_prediction_json
from models.gb_features import FEATURE_NAMES
from models.gradient_boosting import (
    MODEL_NAME,
    MODEL_VERSION,
    walkforward_smoke_test,
)


def main() -> int:
    db = Database()
    rounds = db.get_rounds()
    n_rounds = len(rounds)
    print(f"Loaded rounds: {n_rounds}")
    print(f"PREDICTION_MODEL config: {PREDICTION_MODEL}")

    if n_rounds < GB_MIN_HISTORY + 20:
        print("ERROR: insufficient rounds for activation")
        return 1

    print("\n=== Walk-forward smoke test ===")
    smoke = walkforward_smoke_test(
        rounds,
        min_history=GB_MIN_HISTORY,
        retrain_every=GB_RETRAIN_EVERY,
    )
    if not smoke.get("ok"):
        print("SMOKE FAILED:", smoke)
        return 1

    print(
        f"WF accuracy={smoke.get('accuracy')}% "
        f"scored={smoke.get('scored')} "
        f"unavailable={smoke.get('unavailable')} "
        f"training_rows_final={smoke.get('training_rows_final')}"
    )

    print("\n=== Activate live model ===")
    ens = EnsemblePredictor()
    ok = ens.gb.ensure_ready(rounds, force_retrain=True)
    if not ok:
        print("TRAIN FAILED:", ens.gb.last_error)
        return 1

    resolved = db.get_resolved_predictions()
    metrics = ens.gb.refresh_metrics_from_predictions(resolved)

    result = ens.predict(rounds, train_ml=False, focus="big_small")
    if not result:
        print("PREDICT FAILED: empty result")
        return 1

    latest = rounds[-1]
    target = guess_next_period(str(latest["period"]))
    export = build_prediction_export(
        target, result, current_period=str(latest["period"])
    )
    export["gb_metrics"] = metrics
    export["model_status"] = ens.gb.model_status()
    export["walkforward_smoke"] = {
        "accuracy": smoke.get("accuracy"),
        "scored": smoke.get("scored"),
        "correct": smoke.get("correct"),
        "incorrect": smoke.get("incorrect"),
    }
    write_prediction_json(export)

    # Persist live tip if available (does not rewrite historical resolutions).
    if not result.get("skip_tip") and result.get("top_big_small"):
        db.save_prediction(
            {
                "target_period": target,
                "predicted_number": result.get("top_number"),
                "predicted_color": result.get("top_color"),
                "predicted_big_small": result["top_big_small"],
                "number_probability": result.get("number_probability"),
                "color_probability": result.get("color_probability"),
                "big_small_probability": result.get("big_small_probability"),
                "model_name": "gradient_boosting",
                "decision_strategy": "gradient_boosting",
                "status": "TIP",
            },
            update_existing=True,
        )
        db.upsert_model_metrics(
            "gradient_boosting",
            sample_size=int(metrics.get("resolved_predictions") or 0),
            number_accuracy=0.0,
            color_accuracy=0.0,
            big_small_accuracy=float(metrics.get("accuracy") or 0.0),
        )

    status = ens.gb.model_status()
    live_acc = metrics.get("accuracy")
    live_n = metrics.get("resolved_predictions") or 0
    cur_pred = result.get("top_big_small") or result.get("decision_strategy")
    cur_conf = result.get("confidence_score")

    report_lines = [
        "# Gradient Boosting Production Report",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Model configuration",
        f"- PREDICTION_MODEL: **{PREDICTION_MODEL}**",
        f"- model_name: **{MODEL_NAME}**",
        f"- model_version: **{MODEL_VERSION}**",
        f"- n_estimators: {GB_N_ESTIMATORS}",
        f"- learning_rate: {GB_LEARNING_RATE}",
        f"- max_depth: {GB_MAX_DEPTH}",
        f"- min_history: {GB_MIN_HISTORY}",
        f"- retrain_every: {GB_RETRAIN_EVERY}",
        "",
        "## Features",
        f"- count: {len(FEATURE_NAMES)}",
        f"- names: {', '.join(FEATURE_NAMES)}",
        "",
        "## Training",
        f"- confirmed rounds in DB: {n_rounds}",
        f"- training_rows (supervised): {ens.gb.training_rows}",
        f"- last_trained_at: {ens.gb.last_trained_at}",
        f"- gap_detected: {ens.gb.gap_info.get('gap_detected')} "
        f"(count={ens.gb.gap_info.get('gap_count')})",
        "",
        "## Walk-forward smoke test",
        f"- scored: {smoke.get('scored')}",
        f"- correct: {smoke.get('correct')}",
        f"- incorrect: {smoke.get('incorrect')}",
        f"- unavailable: {smoke.get('unavailable')}",
        f"- accuracy: {smoke.get('accuracy')}%",
        f"- note: chronological expanding window, retrain every {GB_RETRAIN_EVERY}; "
        "not a profit claim.",
        "",
        "## Current live status",
        f"- model_status: **{status.get('model_status')}**",
        f"- decision_strategy: **{result.get('decision_strategy')}**",
        f"- current tip: {cur_pred}",
        f"- confidence: {None if cur_conf is None else round(float(cur_conf) * 100, 2)}%",
        f"- probability_big: {result.get('probability_big')}",
        f"- probability_small: {result.get('probability_small')}",
        f"- target_period: {target}",
        "",
        "## Resolved live predictions (GB only)",
        f"- resolved: {live_n}",
        f"- correct: {metrics.get('correct')}",
        f"- incorrect: {metrics.get('incorrect')}",
        f"- accuracy: {live_acc if live_acc is not None else 'n/a (insufficient sample)'}",
        f"- average_confidence: {metrics.get('average_confidence')}",
        f"- last_10/25/50/100: {metrics.get('last_10')}/"
        f"{metrics.get('last_25')}/{metrics.get('last_50')}/{metrics.get('last_100')}",
        "",
        "## Confidence distribution",
        json.dumps(metrics.get("confidence_bins") or {}, indent=2),
        "",
        "## Errors / failures",
        f"- last_error: {ens.gb.last_error}",
        f"- smoke unavailable count: {smoke.get('unavailable')}",
        "",
        "## Retraining status",
        f"- retrain_every: {GB_RETRAIN_EVERY} new confirmed rounds",
        f"- rounds_at_last_train: {ens.gb._rounds_at_last_train}",
        "",
        "## Production decision",
        "- PRODUCTION CHANGE: **YES**",
        "- PREVIOUS MODEL: majority_w8",
        "- FINAL STRATEGY: **gradient_boosting**",
        "- majority_w8: diagnostic only (no silent fallback)",
        "",
        "Integrity: historical rounds and prior prediction resolutions were not modified.",
    ]
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EXPORTS_DIR / "GRADIENT_BOOSTING_PRODUCTION_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    results_path = EXPORTS_DIR / "gradient_boosting_activation.json"
    results_path.write_text(
        json.dumps(
            {
                "activated": True,
                "prediction_model": PREDICTION_MODEL,
                "smoke": smoke,
                "model_status": status,
                "metrics": metrics,
                "current_prediction": {
                    "target_period": target,
                    "predicted_big_small": result.get("top_big_small"),
                    "decision_strategy": result.get("decision_strategy"),
                    "confidence": result.get("confidence_score"),
                    "probability_big": result.get("probability_big"),
                    "probability_small": result.get("probability_small"),
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("")
    print("MODEL ACTIVATED:")
    print("gradient_boosting")
    print("")
    print("PREVIOUS MODEL:")
    print("majority_w8")
    print("")
    print("TRAINING ROWS:")
    print(ens.gb.training_rows)
    print("")
    print("WALK-FORWARD ACCURACY:")
    print(f"{smoke.get('accuracy'):.2f}%" if smoke.get("accuracy") is not None else "n/a")
    print("")
    print("RESOLVED LIVE PREDICTIONS:")
    print(live_n)
    print("")
    print("LIVE ACCURACY:")
    if live_acc is None:
        print("n/a (insufficient resolved GB tips)")
    else:
        print(f"{live_acc:.2f}%")
    print("")
    print("CURRENT PREDICTION:")
    print(cur_pred)
    print("")
    print("CURRENT CONFIDENCE:")
    if cur_conf is None:
        print("n/a")
    else:
        print(f"{float(cur_conf) * 100:.2f}%")
    print("")
    print("MODEL STATUS:")
    print(status.get("model_status"))
    print("")
    print("PRODUCTION MODEL:")
    print("gradient_boosting")
    print("")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
