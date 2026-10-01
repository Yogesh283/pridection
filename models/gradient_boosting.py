"""Live Gradient Boosting Big/Small predictor (production authority).

No silent fallback to majority_w8. On failure returns PREDICTION_UNAVAILABLE.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from analysis.statistics import big_small_label
from config import EXPORTS_DIR, ROOT_DIR
from models.gb_features import (
    FEATURE_NAMES,
    build_gb_supervised,
    build_gb_vector,
    detect_period_gaps,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "gradient_boosting"
MODEL_VERSION = "gb_v1"
DEFAULT_MIN_HISTORY = 200
DEFAULT_RETRAIN_EVERY = 25
DEFAULT_FEATURE_WARMUP = 20

STATUS_TRAINING = "TRAINING"
STATUS_READY = "READY"
STATUS_STALE = "STALE"
STATUS_ERROR = "ERROR"
STATUS_UNAVAILABLE = "PREDICTION_UNAVAILABLE"

_MODEL_PATH = ROOT_DIR / "models" / "artifacts" / "gradient_boosting_state.json"
_ARTIFACT_DIR = ROOT_DIR / "models" / "artifacts"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class GradientBoostingLivePredictor:
    """Expanding-window GradientBoostingClassifier for Big/Small."""

    def __init__(
        self,
        *,
        min_history: int = DEFAULT_MIN_HISTORY,
        retrain_every: int = DEFAULT_RETRAIN_EVERY,
        feature_warmup: int = DEFAULT_FEATURE_WARMUP,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 3,
    ) -> None:
        self.min_history = int(min_history)
        self.retrain_every = int(retrain_every)
        self.feature_warmup = int(feature_warmup)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)

        self.model: Any = None
        self.trained = False
        self.status = STATUS_ERROR
        self.last_error: str | None = None
        self.last_trained_at: str | None = None
        self.training_rows = 0
        self.trained_on_rounds = 0
        self.model_version = MODEL_VERSION
        self.last_prediction_at: str | None = None
        self.last_prediction: dict[str, Any] | None = None
        self._rounds_at_last_train = 0
        self._lock = threading.Lock()
        self.gap_info: dict[str, Any] = {"gap_detected": False, "gap_count": 0}

        # Live metrics (in-memory; refreshed from DB by callers).
        self.metrics: dict[str, Any] = {
            "total_predictions": 0,
            "resolved_predictions": 0,
            "correct": 0,
            "incorrect": 0,
            "accuracy": None,
            "average_confidence": None,
            "big_accuracy": None,
            "small_accuracy": None,
            "last_10": None,
            "last_25": None,
            "last_50": None,
            "last_100": None,
            "confidence_bins": {},
        }

    def _make_model(self) -> Any:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )

    def needs_retrain(self, history_len: int) -> bool:
        if not self.trained or self.model is None:
            return True
        new_rounds = history_len - self._rounds_at_last_train
        return new_rounds >= self.retrain_every

    def fit(self, rounds: list[dict[str, Any]], *, force: bool = False) -> bool:
        """Train on confirmed chronological rounds only."""
        with self._lock:
            return self._fit_unlocked(rounds, force=force)

    def _fit_unlocked(self, rounds: list[dict[str, Any]], *, force: bool = False) -> bool:
        self.gap_info = detect_period_gaps(rounds)
        if self.gap_info.get("gap_detected"):
            logger.info(
                "gap_detected=true gap_count=%s (observed rows only; no fill)",
                self.gap_info.get("gap_count"),
            )

        if len(rounds) < self.min_history:
            self.trained = False
            self.status = STATUS_ERROR
            self.last_error = (
                f"insufficient_history:{len(rounds)}<{self.min_history}"
            )
            logger.warning("GB train skipped: %s", self.last_error)
            return False

        if (
            not force
            and self.trained
            and not self.needs_retrain(len(rounds))
        ):
            self.status = STATUS_READY
            return True

        self.status = STATUS_TRAINING
        try:
            x, y = build_gb_supervised(rounds, min_history=self.feature_warmup)
            if len(x) < 80:
                self.trained = False
                self.status = STATUS_ERROR
                self.last_error = f"insufficient_supervised_rows:{len(x)}"
                return False
            if len(set(y.tolist())) < 2:
                self.trained = False
                self.status = STATUS_ERROR
                self.last_error = "single_class_labels"
                return False

            clf = self._make_model()
            clf.fit(x, y)
            self.model = clf
            self.trained = True
            self.training_rows = int(len(x))
            self.trained_on_rounds = int(len(rounds))
            self._rounds_at_last_train = int(len(rounds))
            self.last_trained_at = _utc_now()
            self.last_error = None
            self.status = STATUS_READY
            self._persist_meta()
            logger.info(
                "GB trained: rows=%s rounds=%s version=%s",
                self.training_rows,
                self.trained_on_rounds,
                self.model_version,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self.trained = False
            self.model = None
            self.status = STATUS_ERROR
            self.last_error = f"training_failure:{exc}"
            logger.exception("GB training failed: %s", exc)
            return False

    def ensure_ready(
        self, rounds: list[dict[str, Any]], *, force_retrain: bool = False
    ) -> bool:
        if force_retrain or self.needs_retrain(len(rounds)):
            return self.fit(rounds, force=force_retrain)
        if self.trained and self.model is not None:
            self.status = STATUS_READY
            return True
        return self.fit(rounds, force=True)

    def predict(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Predict next Big/Small from settled history.

        Never uses majority_w8. On failure returns status PREDICTION_UNAVAILABLE.
        """
        self.gap_info = detect_period_gaps(history)
        if self.gap_info.get("gap_detected"):
            logger.info(
                "gap_detected=true gap_count=%s at predict",
                self.gap_info.get("gap_count"),
            )

        if len(history) < self.min_history:
            return self._unavailable(
                f"insufficient_history:{len(history)}<{self.min_history}"
            )
        if not self.trained or self.model is None:
            return self._unavailable("model_not_trained")

        try:
            vec = build_gb_vector(history).reshape(1, -1)
            if not np.all(np.isfinite(vec)):
                return self._unavailable("invalid_feature_vector")

            proba = self.model.predict_proba(vec)[0]
            classes = list(self.model.classes_)
            p_big = 0.5
            p_small = 0.5
            if 1 in classes:
                p_big = float(proba[classes.index(1)])
            if 0 in classes:
                p_small = float(proba[classes.index(0)])
            else:
                p_small = 1.0 - p_big
            # Normalize
            total = p_big + p_small
            if total <= 0:
                return self._unavailable("zero_probability_mass")
            p_big /= total
            p_small /= total

            tip = "BIG" if p_big >= p_small else "SMALL"
            confidence = max(p_big, p_small)
            out = {
                "ok": True,
                "status": STATUS_READY,
                "unavailable_reason": None,
                "predicted_big_small": tip,
                "probability_big": round(p_big, 6),
                "probability_small": round(p_small, 6),
                "big_small": {"BIG": p_big, "SMALL": p_small},
                "confidence": round(confidence, 6),
                "model_name": MODEL_NAME,
                "model_version": self.model_version,
                "trained_on_rows": self.training_rows,
                "trained_on_rounds": self.trained_on_rounds,
                "prediction_timestamp": _utc_now(),
                "decision_strategy": MODEL_NAME,
                "gap_detected": bool(self.gap_info.get("gap_detected")),
                "gap_count": int(self.gap_info.get("gap_count") or 0),
            }
            self.last_prediction_at = out["prediction_timestamp"]
            self.last_prediction = out
            self.status = STATUS_READY
            return out
        except Exception as exc:  # noqa: BLE001
            logger.exception("GB predict failed: %s", exc)
            return self._unavailable(f"predict_failure:{exc}")

    def _unavailable(self, reason: str) -> dict[str, Any]:
        self.last_error = reason
        self.status = STATUS_ERROR
        logger.error("PREDICTION_UNAVAILABLE reason=%s", reason)
        return {
            "ok": False,
            "status": STATUS_UNAVAILABLE,
            "unavailable_reason": reason,
            "predicted_big_small": None,
            "probability_big": None,
            "probability_small": None,
            "big_small": None,
            "confidence": None,
            "model_name": MODEL_NAME,
            "model_version": self.model_version,
            "trained_on_rows": self.training_rows,
            "trained_on_rounds": self.trained_on_rounds,
            "prediction_timestamp": _utc_now(),
            "decision_strategy": STATUS_UNAVAILABLE,
            "gap_detected": bool(self.gap_info.get("gap_detected")),
            "gap_count": int(self.gap_info.get("gap_count") or 0),
        }

    def model_status(self) -> dict[str, Any]:
        stale = False
        if self.trained and self._rounds_at_last_train > 0:
            # Mark STALE if more than 2x retrain interval behind (informational).
            # Actual status field still READY until retrain happens.
            pass
        status = self.status
        if status == STATUS_READY and stale:
            status = STATUS_STALE
        return {
            "model_status": status,
            "model_name": MODEL_NAME,
            "model_version": self.model_version,
            "last_trained_at": self.last_trained_at,
            "training_rows": self.training_rows,
            "trained_on_rounds": self.trained_on_rounds,
            "last_prediction_at": self.last_prediction_at,
            "last_prediction": (
                (self.last_prediction or {}).get("predicted_big_small")
            ),
            "last_error": self.last_error,
            "resolved_predictions": self.metrics.get("resolved_predictions"),
            "accuracy": self.metrics.get("accuracy"),
            "retrain_every": self.retrain_every,
            "min_history": self.min_history,
            "gap_info": self.gap_info,
            "feature_count": len(FEATURE_NAMES),
            "features": list(FEATURE_NAMES),
            "hyperparams": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
            },
        }

    def refresh_metrics_from_predictions(
        self, resolved_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute live GB metrics from resolved prediction rows (GB only)."""
        gb_rows = [
            r
            for r in resolved_rows
            if str(r.get("model_name") or r.get("decision_strategy") or "")
            .lower()
            .startswith("gradient_boosting")
            and r.get("predicted_big_small")
            and r.get("big_small_correct") is not None
        ]
        total_pred = len(
            [
                r
                for r in resolved_rows
                if str(r.get("model_name") or "")
                .lower()
                .startswith("gradient_boosting")
            ]
        )
        correct = sum(1 for r in gb_rows if int(r["big_small_correct"]) == 1)
        incorrect = sum(1 for r in gb_rows if int(r["big_small_correct"]) == 0)
        n = correct + incorrect
        confs = [
            float(r["big_small_probability"])
            for r in gb_rows
            if r.get("big_small_probability") is not None
        ]

        def _last_n(k: int) -> float | None:
            subset = gb_rows[-k:]
            if len(subset) < min(5, k):
                return None
            ok = sum(1 for r in subset if int(r["big_small_correct"]) == 1)
            return round(100.0 * ok / len(subset), 2) if subset else None

        big_rows = [
            r for r in gb_rows if str(r.get("predicted_big_small")).upper() == "BIG"
        ]
        small_rows = [
            r
            for r in gb_rows
            if str(r.get("predicted_big_small")).upper() == "SMALL"
        ]

        def _side_acc(rows: list[dict[str, Any]]) -> float | None:
            if len(rows) < 5:
                return None
            ok = sum(1 for r in rows if int(r["big_small_correct"]) == 1)
            return round(100.0 * ok / len(rows), 2)

        bins = {
            "50-55": {"count": 0, "correct": 0, "incorrect": 0, "accuracy": None},
            "55-60": {"count": 0, "correct": 0, "incorrect": 0, "accuracy": None},
            "60-65": {"count": 0, "correct": 0, "incorrect": 0, "accuracy": None},
            "65-70": {"count": 0, "correct": 0, "incorrect": 0, "accuracy": None},
            "70-75": {"count": 0, "correct": 0, "incorrect": 0, "accuracy": None},
            "75+": {"count": 0, "correct": 0, "incorrect": 0, "accuracy": None},
        }

        def _bin_key(c: float) -> str | None:
            pct = c * 100.0 if c <= 1.0 else c
            if 50 <= pct < 55:
                return "50-55"
            if 55 <= pct < 60:
                return "55-60"
            if 60 <= pct < 65:
                return "60-65"
            if 65 <= pct < 70:
                return "65-70"
            if 70 <= pct < 75:
                return "70-75"
            if pct >= 75:
                return "75+"
            return None

        for r in gb_rows:
            c = r.get("big_small_probability")
            if c is None:
                continue
            key = _bin_key(float(c))
            if not key:
                continue
            bins[key]["count"] += 1
            if int(r["big_small_correct"]) == 1:
                bins[key]["correct"] += 1
            else:
                bins[key]["incorrect"] += 1
        for b in bins.values():
            if b["count"] >= 5:
                b["accuracy"] = round(
                    100.0 * b["correct"] / b["count"], 2
                )

        acc = round(100.0 * correct / n, 2) if n >= 5 else None
        self.metrics = {
            "total_predictions": total_pred,
            "resolved_predictions": n,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": acc,
            "average_confidence": (
                round(sum(confs) / len(confs), 4) if confs else None
            ),
            "big_accuracy": _side_acc(big_rows),
            "small_accuracy": _side_acc(small_rows),
            "last_10": _last_n(10),
            "last_25": _last_n(25),
            "last_50": _last_n(50),
            "last_100": _last_n(100),
            "confidence_bins": bins,
        }
        return self.metrics

    def _persist_meta(self) -> None:
        try:
            _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_name": MODEL_NAME,
                "model_version": self.model_version,
                "last_trained_at": self.last_trained_at,
                "training_rows": self.training_rows,
                "trained_on_rounds": self.trained_on_rounds,
                "status": self.status,
                "hyperparams": {
                    "n_estimators": self.n_estimators,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                },
                "feature_names": FEATURE_NAMES,
            }
            _MODEL_PATH.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GB meta persist failed: %s", exc)


def walkforward_smoke_test(
    rounds: list[dict[str, Any]],
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    retrain_every: int = DEFAULT_RETRAIN_EVERY,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """
    Chronological expanding-window smoke test (no shuffle, no leakage).
    Retrains every `retrain_every` new rounds after initial fit.
    """
    gb = GradientBoostingLivePredictor(
        min_history=min_history, retrain_every=retrain_every
    )
    if len(rounds) <= min_history + 5:
        return {
            "ok": False,
            "error": "not_enough_rounds",
            "rounds": len(rounds),
            "min_history": min_history,
        }

    correct = 0
    total = 0
    unavailable = 0
    details: list[dict[str, Any]] = []
    start = min_history
    end = len(rounds)
    if max_steps is not None:
        end = min(end, start + int(max_steps))

    # Initial train on first min_history rounds.
    ok_fit = gb.fit(rounds[:start], force=True)
    if not ok_fit:
        return {
            "ok": False,
            "error": gb.last_error or "initial_train_failed",
            "status": gb.status,
        }

    for i in range(start, end):
        hist = rounds[:i]
        if gb.needs_retrain(len(hist)):
            gb.fit(hist, force=False)
        pred = gb.predict(hist)
        if not pred.get("ok"):
            unavailable += 1
            continue
        tip = pred["predicted_big_small"]
        actual = big_small_label(int(rounds[i]["number"]))
        hit = tip == actual
        correct += int(hit)
        total += 1
        if len(details) < 20 or i >= end - 5:
            details.append(
                {
                    "i": i,
                    "period": rounds[i].get("period"),
                    "pred": tip,
                    "actual": actual,
                    "correct": hit,
                    "confidence": pred.get("confidence"),
                }
            )

    acc = round(100.0 * correct / total, 2) if total else None
    return {
        "ok": True,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "min_history": min_history,
        "retrain_every": retrain_every,
        "training_rows_final": gb.training_rows,
        "rounds_used": len(rounds),
        "scored": total,
        "correct": correct,
        "incorrect": total - correct,
        "unavailable": unavailable,
        "accuracy": acc,
        "gap_info": detect_period_gaps(rounds),
        "model_status": gb.model_status(),
        "sample_details": details,
    }


# Process-wide singleton used by EnsemblePredictor / runner.
_GLOBAL_GB: GradientBoostingLivePredictor | None = None


def get_live_gb() -> GradientBoostingLivePredictor:
    global _GLOBAL_GB
    if _GLOBAL_GB is None:
        from config import (
            GB_MAX_DEPTH,
            GB_LEARNING_RATE,
            GB_MIN_HISTORY,
            GB_N_ESTIMATORS,
            GB_RETRAIN_EVERY,
        )

        _GLOBAL_GB = GradientBoostingLivePredictor(
            min_history=GB_MIN_HISTORY,
            retrain_every=GB_RETRAIN_EVERY,
            n_estimators=GB_N_ESTIMATORS,
            learning_rate=GB_LEARNING_RATE,
            max_depth=GB_MAX_DEPTH,
        )
    return _GLOBAL_GB
