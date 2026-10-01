"""Prediction and probability tests."""

from __future__ import annotations

from models.ensemble import EnsemblePredictor, compute_confidence
from models.predictor import (
    FrequencyModel,
    MarkovModel,
    PatternModel,
    RandomBaseline,
    RecentWeightedModel,
    _normalize,
)


def _rounds(n: int = 40):
    rows = []
    for i in range(n):
        number = (i * 3) % 10
        color = "GREEN" if number % 2 else "RED"
        rows.append({"period": str(1000 + i), "number": number, "color": color})
    return rows


def test_probability_normalization():
    dist = _normalize({0: 2.0, 1: 2.0, 2: 0.0})
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist[0] == dist[1]


def test_baseline_models_return_distributions():
    history = _rounds(30)
    for model in (
        FrequencyModel(),
        MarkovModel(),
        RecentWeightedModel(),
        PatternModel(),
        RandomBaseline(seed=1),
    ):
        out = model.predict(history)
        assert abs(sum(out["numbers"].values()) - 1.0) < 1e-6
        assert abs(sum(out["colors"].values()) - 1.0) < 1e-6
        assert abs(sum(out["big_small"].values()) - 1.0) < 1e-6
        assert set(out["numbers"]) == set(range(10))
        assert set(out["big_small"]) == {"SMALL", "BIG"}


def test_production_engine_refuses_insufficient_history_without_fallback():
    predictor = EnsemblePredictor()
    result = predictor.predict(_rounds(50), train_ml=False)
    assert result is not None
    assert result["top_number"] is None
    assert result["top_color"] is None
    assert result["top_big_small"] is None
    assert result["status"] == "PREDICTION_UNAVAILABLE"
    assert result["skip_tip"] is True
    assert "insufficient_history" in result["unavailable_reason"]
    assert abs(sum(p["probability"] for p in result["big_small_predictions"]) - 1.0) < 1e-5


def test_insufficient_data_no_fake_prediction():
    predictor = EnsemblePredictor()
    assert predictor.predict([], train_ml=False) is None


def test_confidence_not_raw_probability():
    number_probs = {i: 0.05 for i in range(10)}
    number_probs[7] = 0.55
    color_probs = {"RED": 0.4, "GREEN": 0.4, "VIOLET": 0.2}
    outputs = {
        "a": {"numbers": number_probs, "colors": color_probs},
        "b": {"numbers": number_probs, "colors": color_probs},
    }
    conf = compute_confidence(number_probs, color_probs, outputs, sample_size=10)
    assert conf["confidence_score"] != number_probs[7]
    assert conf["confidence_level"] in {"LOW", "MEDIUM", "HIGH"}
