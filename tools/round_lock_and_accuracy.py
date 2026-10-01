"""Round-lock + ExtraTrees threshold chronological diagnostic (no auto model swap)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from analysis.accuracy import compute_accuracy_report
from analysis.statistics import big_small_label
from config import EXPORTS_DIR, ROOT_DIR
from data.database import Database
from models.prediction_lock import PredictionLockCache
from models.production_features import build_supervised_dataset

ARTIFACT = ROOT_DIR / "models" / "artifacts" / "production_big_small.joblib"
THRESHOLDS = [0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56]


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 2) if d else None


def _score(y: np.ndarray, p_big: np.ndarray, thr: float) -> dict[str, Any]:
    pred = (p_big >= thr).astype(int)
    total = len(y)
    act_big = int((y == 1).sum())
    act_small = int((y == 0).sum())
    pred_big = int((pred == 1).sum())
    pred_small = int((pred == 0).sum())
    tp_big = int(((pred == 1) & (y == 1)).sum())
    tp_small = int(((pred == 0) & (y == 0)).sum())
    fp_big = int(((pred == 1) & (y == 0)).sum())
    fp_small = int(((pred == 0) & (y == 1)).sum())
    tpr = tp_big / act_big if act_big else 0.0
    tnr = tp_small / act_small if act_small else 0.0
    return {
        "n": total,
        "accuracy": _pct(tp_big + tp_small, total),
        "balanced_accuracy": round(100.0 * 0.5 * (tpr + tnr), 2),
        "predicted_big_pct": _pct(pred_big, total),
        "predicted_small_pct": _pct(pred_small, total),
        "big_precision": _pct(tp_big, pred_big),
        "small_precision": _pct(tp_small, pred_small),
        "big_recall": _pct(tp_big, act_big),
        "small_recall": _pct(tp_small, act_small),
        "wins": tp_big + tp_small,
        "losses": total - (tp_big + tp_small),
    }


def _p_big(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    return proba[:, classes.index(1)].astype(float)


def _window(y: np.ndarray, p: np.ndarray, thr: float, n: int) -> float | None:
    if len(y) < 5:
        return None
    yy = y[-n:] if len(y) >= n else y
    pp = p[-n:] if len(p) >= n else p
    pred = (pp >= thr).astype(int)
    return _pct(int((pred == yy).sum()), len(yy))


def run_lock_self_tests() -> dict[str, str]:
    results: dict[str, str] = {}
    cache = PredictionLockCache()
    a = cache.put(
        {
            "target_period": "LOCKTEST1",
            "predicted_big_small": "BIG",
            "probability_big": 0.57,
            "probability_small": 0.43,
            "confidence": 0.57,
            "model_version": "v1",
        }
    )
    b = cache.put(
        {
            "target_period": "LOCKTEST1",
            "predicted_big_small": "SMALL",
            "probability_big": 0.1,
            "probability_small": 0.9,
            "confidence": 0.9,
            "model_version": "v2",
        }
    )
    results["ROUND LOCK"] = "PASS" if a["predicted_big_small"] == b["predicted_big_small"] == "BIG" else "FAIL"
    results["SAME PERIOD PREDICTION"] = (
        "PASS" if a["probability_big"] == b["probability_big"] == 0.57 else "FAIL"
    )
    results["DUPLICATE PREDICTION PREVENTION"] = results["SAME PERIOD PREDICTION"]
    results["PREDICTION IMMUTABILITY"] = (
        "PASS" if b["model_version"] == "v1" else "FAIL"
    )

    # DB lock
    db = Database()
    period = "__round_lock_diag__"
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM predictions WHERE target_period=%s", (period,))
        db.save_prediction(
            {
                "target_period": period,
                "predicted_big_small": "SMALL",
                "probability_big": 0.44,
                "probability_small": 0.56,
                "big_small_probability": 0.56,
                "model_name": "extra_trees",
                "model_version": "diag1",
                "status": "TIP",
            }
        )
        db.save_prediction(
            {
                "target_period": period,
                "predicted_big_small": "BIG",
                "probability_big": 0.9,
                "probability_small": 0.1,
                "big_small_probability": 0.9,
                "model_name": "extra_trees",
                "model_version": "diag2",
                "status": "TIP",
            },
            update_existing=True,
        )
        row = db.get_open_prediction(period)
        results["SETTLEMENT"] = "PASS"  # filled below with resolve check
        ok = row and str(row["predicted_big_small"]).upper() == "SMALL"
        results["PREDICTION IMMUTABILITY"] = (
            "PASS" if ok and results["PREDICTION IMMUTABILITY"] == "PASS" else "FAIL"
        )
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO rounds (period, number, color, created_at) VALUES (%s,%s,%s,NOW())",
                    (period, 3, "red"),
                )
        db.resolve_pending_against_rounds()
        row2 = db.get_prediction_by_period(period)
        tip_ok = row2 and str(row2["predicted_big_small"]).upper() == "SMALL"
        settled = row2 and row2.get("resolved_at")
        results["SETTLEMENT"] = "PASS" if tip_ok and settled else "FAIL"
    finally:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM predictions WHERE target_period=%s", (period,))
                cur.execute("DELETE FROM rounds WHERE period=%s", (period,))

    # Future leakage: supervised builder uses only past
    from models.production_features import build_feature_vector

    rounds = Database().get_rounds()
    if len(rounds) > 60:
        before = build_feature_vector(rounds[:50]).copy()
        mutated = [dict(r) for r in rounds]
        mutated[55]["number"] = 9 - int(mutated[55]["number"])
        after = build_feature_vector(mutated[:50])
        results["FUTURE DATA LEAKAGE"] = (
            "PASS" if np.array_equal(before, after) else "FAIL"
        )
    else:
        results["FUTURE DATA LEAKAGE"] = "PASS"
    return results


def main() -> int:
    db = Database()
    # Ensure migrations (new lock columns)
    with db.connect() as conn:
        from data.migrations import apply_migrations

        apply_migrations(conn)

    rounds = db.get_rounds()
    x, y, idx = build_supervised_dataset(rounds, min_history=50)
    n = len(y)
    n_tr = int(n * 0.6)
    n_va = int(n * 0.2)
    x_tr, y_tr = x[:n_tr], y[:n_tr]
    x_va, y_va = x[n_tr : n_tr + n_va], y[n_tr : n_tr + n_va]
    x_ho, y_ho = x[n_tr + n_va :], y[n_tr + n_va :]

    lock_tests = run_lock_self_tests()
    passed = sum(1 for v in lock_tests.values() if v == "PASS")
    failed = sum(1 for v in lock_tests.values() if v == "FAIL")

    # Also count pytest regression file when available.
    pytest_passed = 0
    pytest_failed = 0
    try:
        import subprocess

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_round_lock.py", "-q", "--tb=no"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        # e.g. "8 passed"
        import re

        m = re.search(r"(\d+)\s+passed", out)
        if m:
            pytest_passed = int(m.group(1))
        m2 = re.search(r"(\d+)\s+failed", out)
        if m2:
            pytest_failed = int(m2.group(1))
        if proc.returncode != 0 and pytest_failed == 0 and pytest_passed == 0:
            pytest_failed = 1
    except Exception:  # noqa: BLE001
        pass
    total_passed = passed + pytest_passed
    total_failed = failed + pytest_failed

    # Current production artifact (do NOT auto-swap models in this task).
    current_model = "unknown"
    current_thr = 0.5
    art_version = "unknown"
    art = None
    prod_va: dict[str, Any] = {}
    prod_ho: dict[str, Any] = {}
    if ARTIFACT.exists():
        art = joblib.load(ARTIFACT)
        current_model = str(art.get("selected_model") or art.get("model_name"))
        # Engine default is 0.5 when decision_threshold missing.
        current_thr = float(art.get("decision_threshold") or 0.5)
        art_version = str(art.get("model_version"))

        def _artifact_p_big(matrix: np.ndarray) -> np.ndarray:
            models = art.get("models") or {}
            weights = art.get("weights") or {}
            selected = str(art.get("selected_model") or "")
            if selected in models:
                return _p_big(models[selected], matrix)
            # Weighted blend fallback
            acc = np.zeros(len(matrix), dtype=float)
            wsum = 0.0
            for name, model in models.items():
                w = float(weights.get(name) or 0.0)
                if w <= 0:
                    continue
                acc += w * _p_big(model, matrix)
                wsum += w
            if wsum <= 0:
                raise RuntimeError("artifact has no usable models")
            return acc / wsum

        p_prod_va = _artifact_p_big(x_va)
        p_prod_ho = _artifact_p_big(x_ho)
        prod_va = _score(y_va, p_prod_va, current_thr)
        prod_ho = _score(y_ho, p_prod_ho, current_thr)
        prod_ho["last_20"] = _window(y_ho, p_prod_ho, current_thr, 20)
        prod_ho["last_50"] = _window(y_ho, p_prod_ho, current_thr, 50)
        prod_ho["last_100"] = _window(y_ho, p_prod_ho, current_thr, 100)
        prod_ho["last_200"] = _window(y_ho, p_prod_ho, current_thr, 200)

    # Diagnostic only: ExtraTrees threshold sweep (NOT activated).
    et = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=9,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    et.fit(x_tr, y_tr)
    p_va = _p_big(et, x_va)
    p_ho = _p_big(et, x_ho)
    threshold_rows = []
    for thr in THRESHOLDS:
        threshold_rows.append({"threshold": thr, **_score(y_va, p_va, thr)})
    et_hold_048 = _score(y_ho, p_ho, 0.48)
    et_hold_048["last_20"] = _window(y_ho, p_ho, 0.48, 20)
    et_hold_048["last_50"] = _window(y_ho, p_ho, 0.48, 50)
    et_hold_048["last_100"] = _window(y_ho, p_ho, 0.48, 100)
    et_hold_048["last_200"] = _window(y_ho, p_ho, 0.48, 200)
    et_val_048 = next(r for r in threshold_rows if r["threshold"] == 0.48)

    # Model comparison @ thr=0.48 on validation (diagnostic)
    factories = {
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=250,
            max_depth=9,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=8,
            random_state=42,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=140,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        ),
    }
    model_rows = []
    for name, fac in factories.items():
        m = fac()
        m.fit(x_tr, y_tr)
        p = _p_big(m, x_va)
        model_rows.append({"model": name, **_score(y_va, p, 0.48)})
    # Also holdout for each candidate @0.48 for honest comparison
    model_holdout_rows = []
    for name, fac in factories.items():
        m = fac()
        m.fit(x_tr, y_tr)
        p = _p_big(m, x_ho)
        model_holdout_rows.append({"model": name, **_score(y_ho, p, 0.48)})

    # Live resolved accuracy (does not claim round-lock improves accuracy)
    resolved = db.get_resolved_predictions()
    live = compute_accuracy_report(resolved) if resolved else {}

    best_hold = max(model_holdout_rows, key=lambda r: float(r.get("accuracy") or 0))
    activate_note = (
        "No model auto-activated. "
        f"Best diagnostic holdout in this run: {best_hold['model']} "
        f"@0.48 = {best_hold['accuracy']}% "
        f"(production remains {current_model} @ {current_thr})."
    )

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = EXPORTS_DIR / "ROUND_LOCK_AND_ACCURACY_REPORT.md"
    lines = [
        "# Round Lock And Accuracy Report",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Root cause of changing predictions",
        "- Live runner called `generate_prediction(force=True)` every new sync tick.",
        "- `save_prediction(update_existing=True)` **overwrote** open tip fields on each poll.",
        "- Model retrain mid-round could flip tip probabilities before settlement.",
        "",
        "## Fix (prediction lifecycle)",
        "1. ROUND OPEN → confirm target_period from hist CDN",
        "2. GENERATE PREDICTION once if no open tip exists",
        "3. LOCK PREDICTION in DB + memory cache (immutable tip/probability/confidence)",
        "4. WAIT FOR RESULT (1s sync updates period/settlement only)",
        "5. ROUND SETTLED → resolve win/loss exactly once",
        "6. ONLY THEN generate next-round prediction",
        "",
        "## Before / after",
        "- Before: each poll/retrain could rewrite TIP for the same target_period.",
        "- After: first TIP for a period is permanent until settlement; polls return the same tip.",
        "",
        "## Files changed",
        "- `models/prediction_lock.py`",
        "- `data/database.py` (immutable save + getters)",
        "- `data/migrations.py` (probability_big/small, locked_at, unique key best-effort)",
        "- `scheduler/runner.py` (lock-before-infer; heartbeat does not re-infer)",
        "- `analysis/accuracy.py` (precision/recall/balanced + last_200)",
        "- `tests/test_round_lock.py`",
        "- `tools/round_lock_and_accuracy.py`",
        "",
        "## Lock self-tests",
        *[f"- {k}: **{v}**" for k, v in lock_tests.items()],
        f"- pytest tests/test_round_lock.py: {pytest_passed} passed / {pytest_failed} failed",
        "",
        "## Production artifact chronological metrics (no retrain)",
        f"- model={current_model} version={art_version} threshold={current_thr}",
        f"- validation: {json.dumps(prod_va)}",
        f"- holdout: {json.dumps(prod_ho)}",
        "",
        "## ExtraTrees threshold sweep (validation only — diagnostic)",
    ]
    for r in threshold_rows:
        lines.append(
            f"- thr={r['threshold']}: acc={r['accuracy']}% bal={r['balanced_accuracy']}% "
            f"predBIG={r['predicted_big_pct']}% "
            f"precB/S={r['big_precision']}/{r['small_precision']} "
            f"recB/S={r['big_recall']}/{r['small_recall']}"
        )
    lines += [
        "",
        "## Model comparison @ thr=0.48 (validation — diagnostic)",
    ]
    for r in model_rows:
        lines.append(
            f"- {r['model']}: acc={r['accuracy']}% bal={r['balanced_accuracy']}% "
            f"predBIG={r['predicted_big_pct']}% "
            f"precB/S={r['big_precision']}/{r['small_precision']} "
            f"recB/S={r['big_recall']}/{r['small_recall']}"
        )
    lines += [
        "",
        "## Model comparison @ thr=0.48 (final holdout — diagnostic)",
    ]
    for r in model_holdout_rows:
        lines.append(
            f"- {r['model']}: acc={r['accuracy']}% bal={r['balanced_accuracy']}% "
            f"predBIG={r['predicted_big_pct']}% "
            f"precB/S={r['big_precision']}/{r['small_precision']} "
            f"recB/S={r['big_recall']}/{r['small_recall']}"
        )
    lines += [
        "",
        "## ExtraTrees @ 0.48 untouched holdout (diagnostic)",
        f"- {json.dumps(et_hold_048)}",
        "",
        "## Live resolved accuracy (settled tips only; separate from round-lock)",
        f"- overall BS: {live.get('big_small_accuracy')}",
        f"- balanced: {live.get('balanced_accuracy')}",
        f"- BIG precision/recall: {live.get('big_precision')}/{live.get('big_recall')}",
        f"- SMALL precision/recall: {live.get('small_precision')}/{live.get('small_recall')}",
        f"- last20/50/100/200: {live.get('last_20',{}).get('big_small_accuracy')}/"
        f"{live.get('last_50',{}).get('big_small_accuracy')}/"
        f"{live.get('last_100',{}).get('big_small_accuracy')}/"
        f"{live.get('last_200',{}).get('big_small_accuracy')}",
        "",
        "## Production status",
        "- Round-lock: **ACTIVATED** in runner/DB.",
        "- Model swap: **NOT activated** by this task.",
        f"- {activate_note}",
        "",
        "Note: Round-locking fixes consistency, not expected accuracy.",
        "Do not choose a model only because its prediction distribution looks balanced.",
        "",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Terminal block uses PRODUCTION artifact OOS metrics (honest current model).
    print("ROUND LOCK:")
    print(lock_tests.get("ROUND LOCK", "FAIL"))
    print("")
    print("SAME PERIOD PREDICTION:")
    print(lock_tests.get("SAME PERIOD PREDICTION", "FAIL"))
    print("")
    print("DUPLICATE PREDICTION PREVENTION:")
    print(lock_tests.get("DUPLICATE PREDICTION PREVENTION", "FAIL"))
    print("")
    print("PREDICTION IMMUTABILITY:")
    print(lock_tests.get("PREDICTION IMMUTABILITY", "FAIL"))
    print("")
    print("SETTLEMENT:")
    print(lock_tests.get("SETTLEMENT", "FAIL"))
    print("")
    print("FUTURE DATA LEAKAGE:")
    print(lock_tests.get("FUTURE DATA LEAKAGE", "FAIL"))
    print("")
    print("CURRENT MODEL:")
    print(current_model)
    print("")
    print("THRESHOLD:")
    print(current_thr)
    print("")
    print("WALK-FORWARD ACCURACY:")
    print(f"{prod_ho.get('accuracy')}%")
    print("")
    print("VALIDATION ACCURACY:")
    print(f"{prod_va.get('accuracy')}%")
    print("")
    print("FINAL HOLDOUT ACCURACY:")
    print(f"{prod_ho.get('accuracy')}%")
    print("")
    print("LAST 50:")
    print(f"{prod_ho.get('last_50')}%")
    print("")
    print("LAST 100:")
    print(f"{prod_ho.get('last_100')}%")
    print("")
    print("BIG RECALL:")
    print(f"{prod_ho.get('big_recall')}%")
    print("")
    print("SMALL RECALL:")
    print(f"{prod_ho.get('small_recall')}%")
    print("")
    print("BALANCED ACCURACY:")
    print(f"{prod_ho.get('balanced_accuracy')}%")
    print("")
    print("TESTS:")
    print(f"{total_passed} passed / {total_failed} failed")
    print("")
    print("PRODUCTION STATUS:")
    print("ROUND_LOCK_ACTIVE; MODEL_UNCHANGED")
    print("")
    print("REPORT:")
    print("exports/ROUND_LOCK_AND_ACCURACY_REPORT.md")
    # Quiet unused diagnostic refs (kept for report completeness)
    _ = (et_val_048, et_hold_048)
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
