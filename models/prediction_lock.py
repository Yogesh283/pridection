"""Round-locked prediction cache: one target_period => one immutable tip."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PredictionLockCache:
    """In-memory immutable cache keyed by target_period + model_version."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_period: dict[str, dict[str, Any]] = {}

    def get(self, target_period: str, model_version: str | None = None) -> dict[str, Any] | None:
        key = str(target_period)
        with self._lock:
            entry = self._by_period.get(key)
            if not entry:
                return None
            if model_version and entry.get("model_version") != model_version:
                # Locked tip for this period stays even if model version differs —
                # round-lock wins over retrain mid-round.
                return dict(entry)
            return dict(entry)

    def put(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = str(payload["target_period"])
        with self._lock:
            existing = self._by_period.get(key)
            if existing:
                return dict(existing)
            stored = {
                "target_period": key,
                "predicted_big_small": payload.get("predicted_big_small"),
                "probability_big": payload.get("probability_big"),
                "probability_small": payload.get("probability_small"),
                "confidence": payload.get("confidence"),
                "confidence_level": payload.get("confidence_level"),
                "model_name": payload.get("model_name"),
                "model_version": payload.get("model_version"),
                "decision_strategy": payload.get("decision_strategy"),
                "decision_threshold": payload.get("decision_threshold"),
                "trained_on_rows": payload.get("trained_on_rows"),
                "trained_until_period": payload.get("trained_until_period"),
                "predicted_number": payload.get("predicted_number"),
                "predicted_color": payload.get("predicted_color"),
                "locked_at": payload.get("locked_at") or _utc(),
                "status": payload.get("status") or "TIP",
            }
            self._by_period[key] = stored
            return dict(stored)

    def clear_period(self, target_period: str) -> None:
        with self._lock:
            self._by_period.pop(str(target_period), None)


_GLOBAL_CACHE = PredictionLockCache()


def get_prediction_lock_cache() -> PredictionLockCache:
    return _GLOBAL_CACHE


def locked_row_to_result(row: dict[str, Any]) -> dict[str, Any]:
    """Build an ensemble-compatible result dict from a locked DB/cache row."""
    tip = row.get("predicted_big_small")
    skip = tip is None or str(row.get("status") or "").upper() == "WAIT"
    p_big = row.get("probability_big")
    p_small = row.get("probability_small")
    conf = row.get("big_small_probability") or row.get("confidence")
    if p_big is None and conf is not None and tip:
        # Recover side probs from stored tip confidence when legacy rows lack columns.
        tip_u = str(tip).upper()
        c = float(conf)
        if tip_u == "BIG":
            p_big, p_small = c, 1.0 - c
        else:
            p_small, p_big = c, 1.0 - c
    if p_big is None:
        p_big = 0.5
    if p_small is None:
        p_small = 0.5
    if conf is None:
        conf = max(float(p_big), float(p_small))
    tip_u = None if skip else str(tip).upper()
    conf_f = float(conf)
    if conf_f < 0.55:
        level = "LOW"
    elif conf_f < 0.65:
        level = "MEDIUM"
    elif conf_f < 0.75:
        level = "HIGH"
    else:
        level = "VERY_HIGH"
    decision = str(
        row.get("decision_strategy") or row.get("model_name") or "locked"
    )
    return {
        "focus": "big_small",
        "status": "PREDICTION_AVAILABLE" if not skip else "WAIT",
        "top_number": row.get("predicted_number"),
        "top_color": row.get("predicted_color"),
        "top_big_small": tip_u,
        "bs_source": decision,
        "decision_strategy": decision,
        "model_name": row.get("model_name") or decision,
        "model_version": row.get("model_version"),
        "trained_on_rows": row.get("trained_rows") or row.get("trained_on_rows"),
        "trained_until_period": row.get("trained_until_period"),
        "probability_big": float(p_big),
        "probability_small": float(p_small),
        "big_small_probability": conf_f,
        "confidence_score": conf_f,
        "confidence_level": level if not skip else "LOW",
        "number_probability": row.get("number_probability"),
        "color_probability": row.get("color_probability"),
        "skip_tip": skip,
        "prediction_unavailable": False,
        "ml_is_tip_authority": not skip,
        "round_locked": True,
        "decision_threshold": row.get("decision_threshold"),
        "candidate_models": [decision],
        "big_small_predictions": [
            {"big_small": "BIG", "probability": round(float(p_big), 6)},
            {"big_small": "SMALL", "probability": round(float(p_small), 6)},
        ],
        "disclaimer": (
            "Round-locked tip: immutable for this target_period until settlement."
        ),
    }
