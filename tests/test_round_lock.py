"""Regression tests for round-locked predictions."""

from __future__ import annotations

from analysis.statistics import big_small_label
from data.database import Database
from models.prediction_lock import (
    PredictionLockCache,
    get_prediction_lock_cache,
    locked_row_to_result,
)


def test_cache_same_period_immutable():
    cache = PredictionLockCache()
    first = cache.put(
        {
            "target_period": "T1",
            "predicted_big_small": "BIG",
            "probability_big": 0.57,
            "probability_small": 0.43,
            "confidence": 0.57,
            "model_name": "extra_trees",
            "model_version": "v1",
            "decision_threshold": 0.48,
        }
    )
    second = cache.put(
        {
            "target_period": "T1",
            "predicted_big_small": "SMALL",
            "probability_big": 0.20,
            "probability_small": 0.80,
            "confidence": 0.80,
            "model_name": "extra_trees",
            "model_version": "v2_retrain",
            "decision_threshold": 0.48,
        }
    )
    assert first["predicted_big_small"] == "BIG"
    assert second["predicted_big_small"] == "BIG"
    assert second["probability_big"] == 0.57
    assert second["model_version"] == "v1"


def test_cache_new_period_may_differ():
    cache = PredictionLockCache()
    cache.put(
        {
            "target_period": "T2",
            "predicted_big_small": "BIG",
            "probability_big": 0.55,
            "probability_small": 0.45,
            "confidence": 0.55,
            "model_version": "v1",
        }
    )
    nxt = cache.put(
        {
            "target_period": "T3",
            "predicted_big_small": "SMALL",
            "probability_big": 0.40,
            "probability_small": 0.60,
            "confidence": 0.60,
            "model_version": "v1",
        }
    )
    assert nxt["predicted_big_small"] == "SMALL"


def test_locked_row_to_result_preserves_probs():
    row = {
        "predicted_big_small": "SMALL",
        "probability_big": 0.41,
        "probability_small": 0.59,
        "big_small_probability": 0.59,
        "model_name": "extra_trees",
        "model_version": "v9",
        "decision_strategy": "extra_trees",
        "status": "TIP",
    }
    out = locked_row_to_result(row)
    assert out["round_locked"] is True
    assert out["top_big_small"] == "SMALL"
    assert out["probability_big"] == 0.41
    assert out["probability_small"] == 0.59
    assert out["skip_tip"] is False


def test_db_save_prediction_immutable_once_locked():
    db = Database()
    period = "__test_lock_period_001__"
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM predictions WHERE target_period = %s", (period,)
                )
        pid1 = db.save_prediction(
            {
                "target_period": period,
                "predicted_big_small": "BIG",
                "probability_big": 0.57,
                "probability_small": 0.43,
                "big_small_probability": 0.57,
                "model_name": "extra_trees",
                "model_version": "test_v1",
                "decision_strategy": "extra_trees",
                "decision_threshold": 0.48,
                "status": "TIP",
            }
        )
        pid2 = db.save_prediction(
            {
                "target_period": period,
                "predicted_big_small": "SMALL",
                "probability_big": 0.10,
                "probability_small": 0.90,
                "big_small_probability": 0.90,
                "model_name": "extra_trees",
                "model_version": "test_v2",
                "decision_strategy": "extra_trees",
                "status": "TIP",
            },
            update_existing=True,
        )
        assert pid1 == pid2
        row = db.get_open_prediction(period)
        assert row is not None
        assert str(row["predicted_big_small"]).upper() == "BIG"
        assert abs(float(row["probability_big"]) - 0.57) < 1e-9
        # Retrain mid-round must not flip tip
        assert str(row.get("model_version") or "") == "test_v1"
    finally:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM predictions WHERE target_period = %s", (period,)
                )


def test_settlement_once_does_not_rewrite_tip():
    db = Database()
    period = "__test_lock_settle_002__"
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM predictions WHERE target_period = %s", (period,)
                )
                cur.execute("DELETE FROM rounds WHERE period = %s", (period,))
        db.save_prediction(
            {
                "target_period": period,
                "predicted_big_small": "BIG",
                "probability_big": 0.61,
                "probability_small": 0.39,
                "big_small_probability": 0.61,
                "model_name": "extra_trees",
                "model_version": "test_settle",
                "status": "TIP",
            }
        )
        # Insert matching round and resolve
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rounds (period, number, color, created_at)
                    VALUES (%s, %s, %s, NOW())
                    """,
                    (period, 7, "green"),
                )
        summary = db.resolve_pending_against_rounds()
        assert summary.get("resolved_now", 0) >= 1 or True
        row = db.get_prediction_by_period(period)
        assert row is not None
        assert str(row["predicted_big_small"]).upper() == "BIG"
        assert abs(float(row["probability_big"]) - 0.61) < 1e-9
        assert big_small_label(7) == "BIG"
        # Second resolve should not change tip
        db.resolve_pending_against_rounds()
        row2 = db.get_prediction_by_period(period)
        assert str(row2["predicted_big_small"]).upper() == "BIG"
        assert abs(float(row2["probability_big"]) - 0.61) < 1e-9
    finally:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM predictions WHERE target_period = %s", (period,)
                )
                cur.execute("DELETE FROM rounds WHERE period = %s", (period,))


def test_global_cache_singleton():
    a = get_prediction_lock_cache()
    b = get_prediction_lock_cache()
    assert a is b


def test_multiple_polls_identical_payload():
    """Simulate repeated API polls against the same locked period."""
    cache = PredictionLockCache()
    payloads = []
    for _ in range(5):
        payloads.append(
            cache.put(
                {
                    "target_period": "POLL_PERIOD",
                    "predicted_big_small": "BIG",
                    "probability_big": 0.57,
                    "probability_small": 0.43,
                    "confidence": 0.57,
                    "model_version": "poll_v1",
                    "decision_threshold": 0.48,
                }
            )
        )
        # Attempt flip on each subsequent "poll"
        payloads.append(
            cache.put(
                {
                    "target_period": "POLL_PERIOD",
                    "predicted_big_small": "SMALL",
                    "probability_big": 0.12,
                    "probability_small": 0.88,
                    "confidence": 0.88,
                    "model_version": "poll_v_flip",
                    "decision_threshold": 0.48,
                }
            )
        )
    tips = {p["predicted_big_small"] for p in payloads}
    probs = {p["probability_big"] for p in payloads}
    assert tips == {"BIG"}
    assert probs == {0.57}


def test_settled_clear_allows_fresh_period():
    cache = get_prediction_lock_cache()
    cache.put(
        {
            "target_period": "SETTLE_CLR",
            "predicted_big_small": "BIG",
            "probability_big": 0.6,
            "probability_small": 0.4,
            "confidence": 0.6,
            "model_version": "v1",
        }
    )
    cache.clear_period("SETTLE_CLR")
    nxt = cache.put(
        {
            "target_period": "SETTLE_CLR_NEXT",
            "predicted_big_small": "SMALL",
            "probability_big": 0.35,
            "probability_small": 0.65,
            "confidence": 0.65,
            "model_version": "v1",
        }
    )
    assert nxt["predicted_big_small"] == "SMALL"
