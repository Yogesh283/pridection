"""Serialized, restart-safe production Big/Small predictor.

This module has no majority, follow/flip, or frequency fallback.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from config import ROOT_DIR
from models.model_candidates import candidate_factories, probability_big
from models.production_features import (
    FEATURE_NAMES,
    build_feature_vector,
    build_supervised_dataset,
)

logger = logging.getLogger(__name__)

STATUS_AVAILABLE = "PREDICTION_AVAILABLE"
STATUS_UNAVAILABLE = "PREDICTION_UNAVAILABLE"
MODEL_PATH = ROOT_DIR / "models" / "artifacts" / "production_big_small.joblib"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ProductionBigSmallEngine:
    def __init__(self, artifact_path: Path | str = MODEL_PATH) -> None:
        self.artifact_path = Path(artifact_path)
        self.artifact: dict[str, Any] | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self.load()

    def load(self) -> bool:
        if not self.artifact_path.exists():
            self.last_error = "model_artifact_missing"
            return False
        try:
            artifact = joblib.load(self.artifact_path)
            if artifact.get("feature_names") != FEATURE_NAMES:
                self.last_error = "feature_schema_mismatch"
                return False
            if not artifact.get("models") or not artifact.get("weights"):
                self.last_error = "invalid_model_artifact"
                return False
            artifact.setdefault("high_confidence_enabled", False)
            self.artifact = artifact
            self.last_error = None
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unable to load production model")
            self.last_error = f"artifact_load_failure:{exc}"
            return False

    def _unavailable(self, reason: str) -> dict[str, Any]:
        self.last_error = reason
        artifact = self.artifact or {}
        return {
            "ok": False,
            "status": STATUS_UNAVAILABLE,
            "unavailable_reason": reason,
            "prediction": None,
            "predicted_big_small": None,
            "probability_big": None,
            "probability_small": None,
            "confidence": None,
            "model_name": artifact.get("model_name"),
            "model_version": artifact.get("model_version"),
            "decision_strategy": artifact.get("selected_model"),
            "trained_rows": artifact.get("trained_rows", 0),
            "trained_until_period": artifact.get("trained_until_period"),
        }

    def _probability_big(self, vector: np.ndarray) -> float:
        if not self.artifact:
            raise RuntimeError("model_not_loaded")
        row = vector.reshape(1, -1)
        return float(
            sum(
                float(self.artifact["weights"][name])
                * probability_big(model, row)
                for name, model in self.artifact["models"].items()
            )
        )

    def needs_retrain(self, rounds: list[dict[str, Any]]) -> bool:
        if not self.artifact:
            return False
        trained_rounds = int(self.artifact.get("trained_rounds") or 0)
        interval = int(self.artifact.get("retrain_every") or 25)
        return len(rounds) - trained_rounds >= interval

    def retrain(self, rounds: list[dict[str, Any]]) -> bool:
        """Refit the frozen selected strategy using confirmed rows only."""
        with self._lock:
            if not self.artifact and not self.load():
                return False
            if len(rounds) < 200:
                self.last_error = f"insufficient_history:{len(rounds)}<200"
                return False
            try:
                assert self.artifact is not None
                x, y, _ = build_supervised_dataset(
                    rounds, min_history=int(self.artifact.get("feature_warmup") or 50)
                )
                factories = candidate_factories()
                names = list(self.artifact["models"])
                models: dict[str, Any] = {}
                for name in names:
                    if name not in factories:
                        self.last_error = f"model_dependency_unavailable:{name}"
                        return False
                    model = factories[name]()
                    model.fit(x, y)
                    models[name] = model
                selected = str(self.artifact["selected_model"])
                now = datetime.now(timezone.utc)
                self.artifact.update(
                    {
                        "models": models,
                        "trained_rows": int(len(y)),
                        "trained_rounds": len(rounds),
                        "trained_until_period": str(rounds[-1]["period"]),
                        "trained_at": _utc_now(),
                        "model_version": f"{selected}_{now:%Y%m%d}_{len(rounds)}",
                        # Preserve bias-fix decision threshold / bands across retrain.
                        "decision_threshold": float(
                            self.artifact.get("decision_threshold") or 0.5
                        ),
                        "confidence_bands": self.artifact.get("confidence_bands")
                        or {
                            "LOW": [0.50, 0.55],
                            "MEDIUM": [0.55, 0.65],
                            "HIGH": [0.65, 0.75],
                            "VERY_HIGH": [0.75, 1.01],
                        },
                    }
                )
                self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.artifact_path.with_suffix(".tmp")
                joblib.dump(self.artifact, temporary)
                os.replace(temporary, self.artifact_path)
                self.last_error = None
                return True
            except Exception as exc:  # noqa: BLE001
                logger.exception("Production model retraining failed")
                self.last_error = f"retraining_failure:{exc}"
                return False

    def ensure_ready(self, rounds: list[dict[str, Any]]) -> bool:
        if not self.artifact and not self.load():
            return False
        if self.needs_retrain(rounds):
            return self.retrain(rounds)
        return True

    def predict(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        if len(history) < 200:
            return self._unavailable(f"insufficient_history:{len(history)}<200")
        if not self.ensure_ready(history):
            return self._unavailable(self.last_error or "model_not_ready")
        try:
            vector = build_feature_vector(history)
            if vector.shape[0] != len(FEATURE_NAMES) or not np.isfinite(vector).all():
                return self._unavailable("invalid_feature_vector")
            p_big = min(1.0, max(0.0, self._probability_big(vector)))
            p_small = 1.0 - p_big
            confidence = max(p_big, p_small)
            assert self.artifact is not None
            decision_threshold = float(self.artifact.get("decision_threshold") or 0.5)
            configured = os.getenv("PRODUCTION_CONFIDENCE_THRESHOLD", "").strip()
            threshold = (
                float(configured)
                if configured
                else float(self.artifact.get("confidence_threshold") or 0.55)
            )
            high_confidence_mode = os.getenv(
                "PRODUCTION_HIGH_CONFIDENCE_MODE", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            if not os.getenv("PRODUCTION_HIGH_CONFIDENCE_MODE", "").strip():
                high_confidence_mode = bool(
                    self.artifact.get("high_confidence_enabled", False)
                )
            if high_confidence_mode and confidence < threshold:
                unavailable = self._unavailable(
                    f"confidence_below_threshold:{confidence:.6f}<{threshold:.6f}"
                )
                unavailable.update(
                    {
                        "probability_big": round(p_big, 6),
                        "probability_small": round(p_small, 6),
                        "confidence": round(confidence, 6),
                        "confidence_threshold": threshold,
                        "decision_threshold": decision_threshold,
                    }
                )
                return unavailable
            # Documented: p_big == decision_threshold => BIG
            prediction = "BIG" if p_big >= decision_threshold else "SMALL"
            # Explicit confidence bands (never label ~51% as HIGH).
            if confidence < 0.55:
                confidence_level = "LOW"
            elif confidence < 0.65:
                confidence_level = "MEDIUM"
            elif confidence < 0.75:
                confidence_level = "HIGH"
            else:
                confidence_level = "VERY_HIGH"
            return {
                "ok": True,
                "status": STATUS_AVAILABLE,
                "unavailable_reason": None,
                "prediction": prediction,
                "predicted_big_small": prediction,
                "probability_big": round(p_big, 6),
                "probability_small": round(p_small, 6),
                "confidence": round(confidence, 6),
                "confidence_level": confidence_level,
                "confidence_threshold": threshold,
                "decision_threshold": decision_threshold,
                "high_confidence_mode": high_confidence_mode,
                "model_name": self.artifact["model_name"],
                "model_version": self.artifact["model_version"],
                "decision_strategy": self.artifact["selected_model"],
                "trained_rows": int(self.artifact["trained_rows"]),
                "trained_until_period": self.artifact["trained_until_period"],
                "prediction_timestamp": _utc_now(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Production prediction failed")
            return self._unavailable(f"prediction_failure:{exc}")

    def status(self) -> dict[str, Any]:
        artifact = self.artifact or {}
        return {
            "status": STATUS_AVAILABLE if self.artifact else STATUS_UNAVAILABLE,
            "last_error": self.last_error,
            "model_name": artifact.get("model_name"),
            "model_version": artifact.get("model_version"),
            "trained_rows": artifact.get("trained_rows", 0),
            "trained_rounds": artifact.get("trained_rounds", 0),
            "trained_until_period": artifact.get("trained_until_period"),
            "retrain_every": artifact.get("retrain_every", 25),
            "confidence_threshold": artifact.get("confidence_threshold"),
            "high_confidence_enabled": artifact.get(
                "high_confidence_enabled", False
            ),
        }


_GLOBAL_ENGINE: ProductionBigSmallEngine | None = None


def get_production_engine() -> ProductionBigSmallEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = ProductionBigSmallEngine()
    return _GLOBAL_ENGINE
