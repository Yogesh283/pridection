"""Integrity tests for resolution, color, WAIT, upsert, BS labels."""

from __future__ import annotations

from analysis.accuracy import compute_accuracy_report, is_scored_big_small
from analysis.statistics import big_small_label
from api.client import primary_color
from api.color_canon import canonical_color, source_rank
from api.history_sync import guess_next_period
from data.database import Database
from models.ensemble import EnsemblePredictor


def test_big_small_mapping():
    for n in range(5):
        assert big_small_label(n) == "SMALL"
    for n in range(5, 10):
        assert big_small_label(n) == "BIG"


def test_color_normalization_case_and_compound():
    assert canonical_color("red") == "RED"
    assert canonical_color("RED") == "RED"
    assert canonical_color("green") == "GREEN"
    assert canonical_color("GREEN") == "GREEN"
    assert canonical_color("red,violet") == "RED,VIOLET"
    assert canonical_color("RED,VIOLET") == "RED,VIOLET"
    assert canonical_color("green,violet") == "GREEN,VIOLET"
    assert primary_color("red") == primary_color("RED") == "RED"
    assert primary_color("red,violet") == "RED"
    assert primary_color("green,violet") == "GREEN"


def test_source_trust_order():
    assert source_rank("hist_cdn") > source_rank("dearapi") > source_rank("unknown")


def test_guess_next_period():
    assert guess_next_period("20260930100050897") == "20260930100050898"


def test_wait_not_scored_as_big():
    row = {
        "predicted_big_small": None,
        "predicted_number": 8,
        "actual_number": 8,
        "status": "RESOLVED_WAIT",
        "big_small_correct": None,
    }
    assert is_scored_big_small(row) is False
    report = compute_accuracy_report(
        [
            {
                "predicted_big_small": "BIG",
                "big_small_correct": 1,
                "number_correct": 1,
                "color_correct": 1,
                "actual_number": 8,
                "predicted_number": 8,
                "model_name": "majority_w8",
            },
            row,
        ]
    )
    assert report["big_small_scored"] == 1
    assert report["wait_excluded"] == 1
    assert report["big_small_accuracy"] == 100.0


def test_upsert_no_duplicate_and_trust(tmp_path=None):
    db = Database()
    period = "__test_integrity_period_999999"
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rounds WHERE period = %s", (period,))
        assert db.upsert_round(period, 1, "red", source="dearapi", color_raw="red")
        assert db.get_round_by_period(period)["number"] == 1
        assert db.get_round_by_period(period)["color"] == "RED"
        # Lower trust cannot overwrite.
        assert (
            db.upsert_round(period, 9, "green", source="unknown", color_raw="green")
            is False
        )
        assert db.get_round_by_period(period)["number"] == 1
        # Higher trust can correct.
        assert db.upsert_round(
            period, 9, "green,violet", source="hist_cdn", color_raw="green,violet"
        )
        row = db.get_round_by_period(period)
        assert row["number"] == 9
        assert row["color"] == "GREEN,VIOLET"
        assert row["source"] == "hist_cdn"
        # No duplicates.
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS c FROM rounds WHERE period = %s", (period,)
                )
                assert int(cur.fetchone()["c"]) == 1
    finally:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rounds WHERE period = %s", (period,))


def test_resolve_idempotent_and_exact_period():
    db = Database()
    period = "__test_resolve_period_888888"
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM predictions WHERE target_period = %s", (period,))
                cur.execute("DELETE FROM rounds WHERE period = %s", (period,))
        db.upsert_round(period, 3, "GREEN", source="hist_cdn")
        pid = db.save_prediction(
            {
                "target_period": period,
                "predicted_number": 3,
                "predicted_color": "GREEN",
                "predicted_big_small": "SMALL",
                "model_name": "majority_w8",
                "decision_strategy": "majority_w8",
                "status": "TIP",
            }
        )
        out1 = db.resolve_prediction(pid, 3, "GREEN")
        assert out1["changed"] is True
        out2 = db.resolve_prediction(pid, 3, "GREEN")
        assert out2["reason"] == "already_resolved"
        assert out2["changed"] is False
        row = None
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM predictions WHERE id = %s", (pid,))
                row = cur.fetchone()
        assert row["big_small_correct"] == 1
        assert row["actual_number"] == 3
        # Orphan stays open.
        orphan_period = "__test_orphan_period_777777"
        oid = db.save_prediction(
            {
                "target_period": orphan_period,
                "predicted_number": 1,
                "predicted_big_small": "SMALL",
                "model_name": "majority_w8",
                "decision_strategy": "majority_w8",
            }
        )
        summary = db.resolve_pending_against_rounds()
        unresolved = {p["id"] for p in db.get_unresolved_predictions()}
        assert oid in unresolved
        assert summary["orphans_left"] >= 1
    finally:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM predictions WHERE target_period LIKE %s",
                    ("__test_%",),
                )
                cur.execute(
                    "DELETE FROM rounds WHERE period LIKE %s", ("__test_%",)
                )


def test_live_decision_strategy_never_uses_legacy_fallback():
    db = Database()
    rounds = db.get_rounds()
    if len(rounds) < 220:
        return
    ens = EnsemblePredictor()
    out = ens.predict(rounds[-220:], train_ml=False, focus="big_small")
    assert out is not None
    assert "decision_strategy" in out
    strategy = str(out.get("decision_strategy") or "")
    assert "majority" not in strategy.lower()
    assert "frequency" not in strategy.lower()
    if out.get("skip_tip"):
        assert out.get("prediction_unavailable") is True
    else:
        assert out.get("ml_is_tip_authority") is True
        assert out.get("top_big_small") in ("BIG", "SMALL")
