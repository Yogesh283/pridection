"""Integrity tests for the production-only Big/Small engine."""

from __future__ import annotations

import inspect

import joblib
import numpy as np

from models.model_candidates import candidate_factories
from models.production_engine import ProductionBigSmallEngine
from models.production_features import (
    FEATURE_NAMES,
    build_feature_vector,
    build_supervised_dataset,
)


def _rounds(count: int) -> list[dict]:
    return [
        {
            "period": str(20261001000000000 + i),
            "number": (i * 7 + i // 3) % 10,
            "color": "RED" if ((i * 7 + i // 3) % 10) % 2 == 0 else "GREEN",
        }
        for i in range(count)
    ]


def test_target_and_future_rows_cannot_change_feature_row():
    rounds = _rounds(80)
    before = build_feature_vector(rounds[:50])
    rounds[50]["number"] = 9 - rounds[50]["number"]
    rounds[70]["number"] = 9 - rounds[70]["number"]
    after = build_feature_vector(rounds[:50])
    assert np.array_equal(before, after)

    x, y, indices = build_supervised_dataset(rounds, min_history=50)
    assert indices[0] == 50
    assert len(x) == len(y) == 30


def test_production_module_contains_no_legacy_fallback_calls():
    import models.production_engine as module

    source = inspect.getsource(module).lower()
    assert "majority_w8" not in source
    assert "last10" not in source
    assert "frequencymodel" not in source
    assert "train_test_split" not in source


def test_model_survives_restart_and_retrains(tmp_path):
    rounds = _rounds(225)
    x, y, _ = build_supervised_dataset(rounds[:200], min_history=50)
    model = candidate_factories()["gradient_boosting"]()
    model.fit(x, y)
    path = tmp_path / "production.joblib"
    joblib.dump(
        {
            "models": {"gradient_boosting": model},
            "weights": {"gradient_boosting": 1.0},
            "selected_model": "gradient_boosting",
            "model_name": "gradient_boosting",
            "model_version": "test_v1",
            "feature_names": FEATURE_NAMES,
            "feature_warmup": 50,
            "trained_rows": len(y),
            "trained_rounds": 200,
            "trained_until_period": rounds[199]["period"],
            "retrain_every": 25,
            "confidence_threshold": 0.50,
        },
        path,
    )
    first = ProductionBigSmallEngine(path)
    assert first.artifact is not None
    assert first.needs_retrain(rounds) is True
    assert first.retrain(rounds) is True
    assert first.artifact["trained_rounds"] == 225

    restarted = ProductionBigSmallEngine(path)
    output = restarted.predict(rounds)
    assert restarted.artifact is not None
    assert output["model_name"] == "gradient_boosting"
    assert output["status"] == "PREDICTION_AVAILABLE"
