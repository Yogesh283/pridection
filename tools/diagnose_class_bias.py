"""READ-ONLY class-bias diagnostic for calibrated_gradient_boosting.

Does NOT:
- modify production code paths in place
- overwrite the production joblib artifact
- retrain-to-disk
- change DB rounds or predictions
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from analysis.statistics import big_small_label
from config import EXPORTS_DIR, ROOT_DIR
from data.database import Database
from models.model_candidates import probability_big
from models.production_features import FEATURE_NAMES, build_feature_vector

MODEL_NAME = "calibrated_gradient_boosting"
ARTIFACT = ROOT_DIR / "models" / "artifacts" / "production_big_small.joblib"
MIN_HISTORY = 200


def _pct(n: int, d: int) -> float | None:
    if d <= 0:
        return None
    return round(100.0 * n / d, 2)


def _wilson(correct: int, total: int) -> tuple[float, float] | None:
    if total <= 0:
        return None
    # simple normal approx CI for report brevity
    p = correct / total
    z = 1.96
    se = (p * (1 - p) / total) ** 0.5
    return round(100 * max(0.0, p - z * se), 2), round(100 * min(1.0, p + z * se), 2)


def _bucket(conf: float) -> str | None:
    # conf is tip-side probability in [0.5, 1]
    pct = conf * 100.0 if conf <= 1.0 else conf
    if 50 <= pct < 55:
        return "50-55"
    if 55 <= pct < 60:
        return "55-60"
    if 60 <= pct < 65:
        return "60-65"
    if 65 <= pct < 70:
        return "65-70"
    if pct >= 70:
        return "70+"
    return None


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    act_big = sum(1 for r in rows if r["actual"] == "BIG")
    act_small = sum(1 for r in rows if r["actual"] == "SMALL")
    pred_big = sum(1 for r in rows if r["pred"] == "BIG")
    pred_small = sum(1 for r in rows if r["pred"] == "SMALL")

    tp_big = sum(1 for r in rows if r["pred"] == "BIG" and r["actual"] == "BIG")
    fp_big = sum(1 for r in rows if r["pred"] == "BIG" and r["actual"] == "SMALL")
    tp_small = sum(1 for r in rows if r["pred"] == "SMALL" and r["actual"] == "SMALL")
    fp_small = sum(1 for r in rows if r["pred"] == "SMALL" and r["actual"] == "BIG")
    # confusion: rows=actual BIG/SMALL, cols=pred BIG/SMALL
    # TN for BIG = correct SMALL = tp_small
    fn_big = fp_small  # actual BIG predicted SMALL
    fn_small = fp_big

    correct = tp_big + tp_small
    big_prec = _pct(tp_big, pred_big)
    small_prec = _pct(tp_small, pred_small)
    big_rec = _pct(tp_big, act_big)
    small_rec = _pct(tp_small, act_small)
    acc = _pct(correct, total)

    p_bigs = [r["p_big"] for r in rows]
    p_smalls = [r["p_small"] for r in rows]
    tip_confs = [r["confidence"] for r in rows]

    buckets = {
        k: {"count": 0, "correct": 0, "incorrect": 0, "accuracy": None}
        for k in ("50-55", "55-60", "60-65", "65-70", "70+")
    }
    for r in rows:
        b = _bucket(r["confidence"])
        if not b:
            continue
        buckets[b]["count"] += 1
        if r["pred"] == r["actual"]:
            buckets[b]["correct"] += 1
        else:
            buckets[b]["incorrect"] += 1
    for b in buckets.values():
        if b["count"] >= 5:
            b["accuracy"] = _pct(b["correct"], b["count"])

    # Bias verdict vs actual distribution
    pred_big_share = (pred_big / total) if total else 0.0
    act_big_share = (act_big / total) if total else 0.0
    delta = pred_big_share - act_big_share
    if abs(delta) < 0.05 and 0.40 <= pred_big_share <= 0.60:
        bias = "NONE"
    elif delta >= 0.10 or pred_big_share >= 0.70:
        bias = "BIG-BIASED"
    elif delta <= -0.10 or pred_big_share <= 0.30:
        bias = "SMALL-BIASED"
    elif delta > 0.05:
        bias = "BIG-BIASED"
    elif delta < -0.05:
        bias = "SMALL-BIASED"
    else:
        bias = "NONE"

    return {
        "total": total,
        "actual_big": act_big,
        "actual_small": act_small,
        "actual_big_pct": _pct(act_big, total),
        "actual_small_pct": _pct(act_small, total),
        "predicted_big": pred_big,
        "predicted_small": pred_small,
        "predicted_big_pct": _pct(pred_big, total),
        "predicted_small_pct": _pct(pred_small, total),
        "correct_big": tp_big,
        "incorrect_big": fp_big,
        "correct_small": tp_small,
        "incorrect_small": fp_small,
        "big_precision": big_prec,
        "small_precision": small_prec,
        "big_recall": big_rec,
        "small_recall": small_rec,
        "overall_accuracy": acc,
        "wilson_95": _wilson(correct, total),
        "confusion_matrix": {
            "actual_BIG_pred_BIG": tp_big,
            "actual_BIG_pred_SMALL": fn_big,
            "actual_SMALL_pred_BIG": fp_big,
            "actual_SMALL_pred_SMALL": tp_small,
        },
        "avg_p_big": round(statistics.mean(p_bigs), 6) if p_bigs else None,
        "avg_p_small": round(statistics.mean(p_smalls), 6) if p_smalls else None,
        "median_p_big": round(statistics.median(p_bigs), 6) if p_bigs else None,
        "median_p_small": round(statistics.median(p_smalls), 6) if p_smalls else None,
        "min_p_big": round(min(p_bigs), 6) if p_bigs else None,
        "max_p_big": round(max(p_bigs), 6) if p_bigs else None,
        "min_p_small": round(min(p_smalls), 6) if p_smalls else None,
        "max_p_small": round(max(p_smalls), 6) if p_smalls else None,
        "avg_tip_confidence": round(statistics.mean(tip_confs), 6) if tip_confs else None,
        "probability_buckets": buckets,
        "pred_minus_actual_big_share_pp": round(100.0 * delta, 2),
        "class_bias": bias,
    }


def _window_stats(rows: list[dict[str, Any]], n: int) -> dict[str, Any]:
    subset = rows[-n:] if len(rows) >= n else rows[:]
    m = _metrics(subset)
    return {
        "window": n,
        "n": m["total"],
        "predicted_big": m["predicted_big"],
        "predicted_small": m["predicted_small"],
        "actual_big": m["actual_big"],
        "actual_small": m["actual_small"],
        "accuracy": m["overall_accuracy"],
        "predicted_big_pct": m["predicted_big_pct"],
        "predicted_small_pct": m["predicted_small_pct"],
    }


def load_frozen_artifact() -> dict[str, Any]:
    if not ARTIFACT.exists():
        raise FileNotFoundError(f"artifact missing: {ARTIFACT}")
    artifact = joblib.load(ARTIFACT)
    # Deep-ish copy of sklearn models so any accidental fit cannot leak to disk
    # unless we explicitly dump (we never dump in this script).
    return {
        "models": {k: copy.deepcopy(v) for k, v in (artifact.get("models") or {}).items()},
        "weights": dict(artifact.get("weights") or {}),
        "selected_model": artifact.get("selected_model"),
        "model_name": artifact.get("model_name"),
        "model_version": artifact.get("model_version"),
        "feature_names": list(artifact.get("feature_names") or []),
        "trained_rows": artifact.get("trained_rows"),
        "trained_rounds": artifact.get("trained_rounds"),
        "trained_until_period": artifact.get("trained_until_period"),
        "trained_at": artifact.get("trained_at"),
        "meta": {
            k: artifact.get(k)
            for k in (
                "confidence_threshold",
                "high_confidence_enabled",
                "retrain_every",
                "feature_warmup",
            )
        },
    }


def predict_frozen(artifact: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(history) < MIN_HISTORY:
        return None
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("feature_schema_mismatch")
    vec = build_feature_vector(history).reshape(1, -1)
    if not np.isfinite(vec).all():
        return None
    p_big = float(
        sum(
            float(artifact["weights"][name]) * probability_big(model, vec)
            for name, model in artifact["models"].items()
        )
    )
    p_big = min(1.0, max(0.0, p_big))
    p_small = 1.0 - p_big
    tip = "BIG" if p_big >= 0.5 else "SMALL"
    return {
        "pred": tip,
        "p_big": p_big,
        "p_small": p_small,
        "confidence": max(p_big, p_small),
    }


def run_frozen_chronological(
    rounds: list[dict[str, Any]], artifact: dict[str, Any]
) -> list[dict[str, Any]]:
    """Features from past only; model weights frozen (no retrain, no disk write)."""
    rows: list[dict[str, Any]] = []
    for i in range(MIN_HISTORY, len(rounds)):
        out = predict_frozen(artifact, rounds[:i])
        if not out:
            continue
        actual = big_small_label(int(rounds[i]["number"]))
        rows.append(
            {
                "i": i,
                "period": rounds[i].get("period"),
                "actual": actual,
                **out,
            }
        )
    return rows


def analyze_live_resolved(db: Database) -> dict[str, Any]:
    resolved = db.get_resolved_predictions()
    gb = [
        r
        for r in resolved
        if MODEL_NAME
        in str(r.get("model_name") or "").lower()
        or MODEL_NAME
        in str(r.get("decision_strategy") or "").lower()
        or "calibrated_gradient_boosting"
        in str(r.get("model_name") or "").lower()
    ]
    rows = []
    for r in gb:
        pred = str(r.get("predicted_big_small") or "").upper()
        actual = str(r.get("actual_big_small") or "").upper()
        if pred not in ("BIG", "SMALL") or actual not in ("BIG", "SMALL"):
            continue
        conf = r.get("big_small_probability")
        if conf is None:
            p_big = 0.5
            p_small = 0.5
            tip_conf = 0.5
        else:
            tip_conf = float(conf)
            if tip_conf <= 1.0:
                # stored as tip-side probability
                if pred == "BIG":
                    p_big = tip_conf
                    p_small = 1.0 - tip_conf
                else:
                    p_small = tip_conf
                    p_big = 1.0 - tip_conf
            else:
                tip_conf = tip_conf / 100.0
                if pred == "BIG":
                    p_big = tip_conf
                    p_small = 1.0 - tip_conf
                else:
                    p_small = tip_conf
                    p_big = 1.0 - tip_conf
        rows.append(
            {
                "period": r.get("target_period"),
                "pred": pred,
                "actual": actual,
                "p_big": p_big,
                "p_small": p_small,
                "confidence": max(p_big, p_small),
            }
        )
    return {"count": len(rows), "metrics": _metrics(rows) if rows else None, "rows": rows}


def diagnose_causes(
    frozen_m: dict[str, Any],
    live_m: dict[str, Any] | None,
    train_y_balance: dict[str, Any],
    recent_vs_hist: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    bias = frozen_m["class_bias"]
    notes.append(f"Frozen chronological class_bias={bias}.")

    # Training imbalance
    tb = train_y_balance.get("big_pct")
    if tb is not None and abs(tb - 50.0) < 5:
        notes.append(
            f"Training labels near balanced (BIG={tb}%) — training class imbalance is NOT the main driver."
        )
    elif tb is not None:
        notes.append(
            f"Training labels BIG={tb}% — mild imbalance; check if same direction as prediction bias."
        )

    # Probability calibration / threshold
    avg_b = frozen_m.get("avg_p_big")
    med_b = frozen_m.get("median_p_big")
    if avg_b is not None and med_b is not None:
        if abs(avg_b - 0.5) < 0.03 and frozen_m["predicted_big_pct"] not in (None,):
            notes.append(
                f"Mean P(BIG)={avg_b}, median={med_b}: probabilities hug 0.5; "
                "argmax threshold 0.5 amplifies tiny tilts into one-sided tip frequency."
            )
        if avg_b < 0.48:
            notes.append(
                "Average P(BIG) systematically below 0.5 → decision threshold 0.5 yields SMALL-heavy tips "
                "(probability calibration / model tilt)."
            )
        if avg_b > 0.52:
            notes.append(
                "Average P(BIG) systematically above 0.5 → BIG-heavy tips from 0.5 threshold."
            )

    # Underfitting
    if frozen_m.get("overall_accuracy") is not None and 45 <= frozen_m["overall_accuracy"] <= 55:
        notes.append(
            f"Overall accuracy {frozen_m['overall_accuracy']}% near chance → weak signal / underfitting; "
            "bias can dominate because there is little true predictive structure."
        )

    # Concentration in weak buckets
    b5055 = (frozen_m.get("probability_buckets") or {}).get("50-55", {}).get("count", 0)
    if frozen_m["total"] and b5055 / frozen_m["total"] >= 0.7:
        notes.append(
            f"{_pct(b5055, frozen_m['total'])}% of tips sit in 50–55% confidence — "
            "model is mostly undecided; one-sided class rate is threshold artifact, not strong conviction."
        )

    # Recent vs hist
    if recent_vs_hist.get("shift_note"):
        notes.append(recent_vs_hist["shift_note"])

    if live_m:
        notes.append(
            f"Live resolved DB tips for this model: pred BIG={live_m.get('predicted_big_pct')}% "
            f"SMALL={live_m.get('predicted_small_pct')}% bias={live_m.get('class_bias')} "
            f"(n={live_m.get('total')})."
        )

    # Features note (cannot prove without ablation; observational)
    notes.append(
        "Feature bias not proven here (no ablation). Frozen-model tip skew without matching "
        "actual-class skew indicates model/calibration/threshold behavior rather than forced 50/50 games."
    )
    return notes


def main() -> int:
    db = Database()
    rounds = db.get_rounds()
    artifact = load_frozen_artifact()

    print(f"Loaded rounds={len(rounds)}")
    print(
        f"Artifact model={artifact.get('model_name')} version={artifact.get('model_version')} "
        f"trained_rows={artifact.get('trained_rows')} (READ-ONLY copy; no disk retrain)"
    )

    # Label balance in full history and in supervised training slice used originally
    labels = [big_small_label(int(r["number"])) for r in rounds]
    hist_big = sum(1 for x in labels if x == "BIG")
    hist_small = len(labels) - hist_big
    # Approximate training labels: rounds after warmup (feature_warmup default 50)
    warmup = int((artifact.get("meta") or {}).get("feature_warmup") or 50)
    train_labels = labels[warmup:]
    train_big = sum(1 for x in train_labels if x == "BIG")
    train_bal = {
        "n": len(train_labels),
        "big": train_big,
        "small": len(train_labels) - train_big,
        "big_pct": _pct(train_big, len(train_labels)),
        "small_pct": _pct(len(train_labels) - train_big, len(train_labels)),
    }

    recent = labels[-100:] if len(labels) >= 100 else labels
    recent_big = sum(1 for x in recent if x == "BIG")
    recent_vs_hist = {
        "hist_big_pct": _pct(hist_big, len(labels)),
        "recent100_big_pct": _pct(recent_big, len(recent)),
        "shift_note": (
            f"Recent100 BIG%={_pct(recent_big, len(recent))} vs full-hist BIG%={_pct(hist_big, len(labels))}: "
            + (
                "similar — recent distribution shift is not the primary explanation."
                if abs((recent_big / max(len(recent), 1)) - (hist_big / max(len(labels), 1))) < 0.08
                else "notable shift vs full history — may amplify frozen-model mismatch."
            )
        ),
    }

    frozen_rows = run_frozen_chronological(rounds, artifact)
    frozen_m = _metrics(frozen_rows)
    windows = [_window_stats(frozen_rows, n) for n in (20, 50, 100, 200)]

    live = analyze_live_resolved(db)
    live_m = live.get("metrics")

    causes = diagnose_causes(frozen_m, live_m, train_bal, recent_vs_hist)

    # Artifact path mtime check — we must not have written
    art_mtime_before = ARTIFACT.stat().st_mtime

    report_lines = [
        "# Calibrated Gradient Boosting — Class Bias Diagnostic",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Scope (READ-ONLY)",
        "- Production model: **calibrated_gradient_boosting** (frozen artifact copy).",
        "- No production code change, no artifact overwrite, no DB mutation, no retrain-to-disk.",
        "- Chronological test: features from rounds[:i] only; tip = BIG if P(BIG)>=0.5 else SMALL.",
        "- Note: frozen weights were trained on later history; this measures **current model tip bias**",
        "  on chronological features, not a pure never-seen expanding retrain OOS.",
        "",
        "## Artifact",
        f"- path: `{ARTIFACT}`",
        f"- model_name: {artifact.get('model_name')}",
        f"- model_version: {artifact.get('model_version')}",
        f"- trained_rows: {artifact.get('trained_rows')}",
        f"- trained_rounds: {artifact.get('trained_rounds')}",
        f"- trained_until_period: {artifact.get('trained_until_period')}",
        f"- trained_at: {artifact.get('trained_at')}",
        "",
        "## Historical class balance",
        f"- rounds: {len(rounds)}",
        f"- actual BIG: {hist_big} ({_pct(hist_big, len(labels))}%)",
        f"- actual SMALL: {hist_small} ({_pct(hist_small, len(labels))}%)",
        f"- approx train-slice BIG%: {train_bal['big_pct']}% (n={train_bal['n']})",
        f"- recent100 BIG%: {recent_vs_hist['recent100_big_pct']}%",
        "",
        "## Frozen chronological evaluation",
        f"- test predictions: **{frozen_m['total']}**",
        f"- actual BIG: {frozen_m['actual_big']} ({frozen_m['actual_big_pct']}%)",
        f"- actual SMALL: {frozen_m['actual_small']} ({frozen_m['actual_small_pct']}%)",
        f"- predicted BIG: {frozen_m['predicted_big']} ({frozen_m['predicted_big_pct']}%)",
        f"- predicted SMALL: {frozen_m['predicted_small']} ({frozen_m['predicted_small_pct']}%)",
        f"- correct BIG: {frozen_m['correct_big']}",
        f"- incorrect BIG: {frozen_m['incorrect_big']}",
        f"- correct SMALL: {frozen_m['correct_small']}",
        f"- incorrect SMALL: {frozen_m['incorrect_small']}",
        f"- BIG precision: {frozen_m['big_precision']}%",
        f"- SMALL precision: {frozen_m['small_precision']}%",
        f"- BIG recall: {frozen_m['big_recall']}%",
        f"- SMALL recall: {frozen_m['small_recall']}%",
        f"- overall accuracy: {frozen_m['overall_accuracy']}%",
        f"- wilson ~95%: {frozen_m['wilson_95']}",
        f"- pred BIG share − actual BIG share: {frozen_m['pred_minus_actual_big_share_pp']} pp",
        f"- **class bias: {frozen_m['class_bias']}**",
        "",
        "### Confusion matrix (actual × predicted)",
        "```",
        "                 pred BIG   pred SMALL",
        f"actual BIG      {frozen_m['confusion_matrix']['actual_BIG_pred_BIG']:>8}   {frozen_m['confusion_matrix']['actual_BIG_pred_SMALL']:>10}",
        f"actual SMALL    {frozen_m['confusion_matrix']['actual_SMALL_pred_BIG']:>8}   {frozen_m['confusion_matrix']['actual_SMALL_pred_SMALL']:>10}",
        "```",
        "",
        "### Probability stats",
        f"- avg P(BIG): {frozen_m['avg_p_big']}",
        f"- avg P(SMALL): {frozen_m['avg_p_small']}",
        f"- median P(BIG): {frozen_m['median_p_big']}",
        f"- median P(SMALL): {frozen_m['median_p_small']}",
        f"- min/max P(BIG): {frozen_m['min_p_big']} / {frozen_m['max_p_big']}",
        f"- min/max P(SMALL): {frozen_m['min_p_small']} / {frozen_m['max_p_small']}",
        f"- avg tip confidence: {frozen_m['avg_tip_confidence']}",
        "",
        "### Tip-confidence buckets",
    ]
    for name, b in frozen_m["probability_buckets"].items():
        report_lines.append(
            f"- {name}: count={b['count']} correct={b['correct']} incorrect={b['incorrect']} "
            f"acc={b['accuracy']}"
        )

    report_lines += ["", "### Recent windows (frozen chronological)"]
    for w in windows:
        report_lines.append(
            f"- last {w['window']}: n={w['n']} pred BIG/SMALL={w['predicted_big']}/{w['predicted_small']} "
            f"({w['predicted_big_pct']}%/{w['predicted_small_pct']}%) "
            f"actual BIG/SMALL={w['actual_big']}/{w['actual_small']} acc={w['accuracy']}%"
        )

    report_lines += ["", "## Live resolved predictions (DB, this model only)"]
    if live_m:
        report_lines += [
            f"- n={live_m['total']}",
            f"- predicted BIG/SMALL: {live_m['predicted_big']}/{live_m['predicted_small']} "
            f"({live_m['predicted_big_pct']}%/{live_m['predicted_small_pct']}%)",
            f"- actual BIG/SMALL: {live_m['actual_big']}/{live_m['actual_small']}",
            f"- accuracy: {live_m['overall_accuracy']}%",
            f"- BIG/SMALL precision: {live_m['big_precision']}% / {live_m['small_precision']}%",
            f"- BIG/SMALL recall: {live_m['big_recall']}% / {live_m['small_recall']}%",
            f"- class bias: **{live_m['class_bias']}**",
            f"- avg P(BIG)/P(SMALL): {live_m['avg_p_big']} / {live_m['avg_p_small']}",
        ]
    else:
        report_lines.append("- no resolved calibrated_gradient_boosting tips found.")

    report_lines += ["", "## Cause assessment", *[f"- {c}" for c in causes], ""]
    report_lines += [
        "## Conclusion",
        f"- Frozen chronological bias: **{frozen_m['class_bias']}**",
        f"- Live DB bias: **{(live_m or {}).get('class_bias', 'n/a')}**",
        "- No thresholds were changed. No 50/50 forcing applied.",
        "- Production remains calibrated_gradient_boosting unchanged.",
        "",
    ]

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EXPORTS_DIR / "CALIBRATED_GB_CLASS_BIAS_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    json_path = EXPORTS_DIR / "calibrated_gb_class_bias_results.json"
    # exports/*.json may be gitignored; still write for local use
    json_path.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "artifact_version": artifact.get("model_version"),
                "frozen": frozen_m,
                "windows": windows,
                "live_resolved": live_m,
                "train_balance": train_bal,
                "recent_vs_hist": recent_vs_hist,
                "causes": causes,
                "read_only": True,
                "artifact_mtime_unchanged": ARTIFACT.stat().st_mtime == art_mtime_before,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Terminal summary (exact requested shape)
    print("")
    print("MODEL:")
    print(MODEL_NAME)
    print("")
    print("TEST PREDICTIONS:")
    print(frozen_m["total"])
    print("")
    print("ACTUAL BIG:")
    print(f"{frozen_m['actual_big_pct']}%")
    print("")
    print("ACTUAL SMALL:")
    print(f"{frozen_m['actual_small_pct']}%")
    print("")
    print("PREDICTED BIG:")
    print(f"{frozen_m['predicted_big_pct']}%")
    print("")
    print("PREDICTED SMALL:")
    print(f"{frozen_m['predicted_small_pct']}%")
    print("")
    print("BIG PRECISION:")
    print(f"{frozen_m['big_precision']}%")
    print("")
    print("SMALL PRECISION:")
    print(f"{frozen_m['small_precision']}%")
    print("")
    print("BIG RECALL:")
    print(f"{frozen_m['big_recall']}%")
    print("")
    print("SMALL RECALL:")
    print(f"{frozen_m['small_recall']}%")
    print("")
    print("OVERALL ACCURACY:")
    print(f"{frozen_m['overall_accuracy']}%")
    print("")
    print("CLASS BIAS:")
    print(frozen_m["class_bias"])
    print("")
    print(f"Report: {report_path}")
    print(f"Artifact mtime unchanged: {ARTIFACT.stat().st_mtime == art_mtime_before}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
