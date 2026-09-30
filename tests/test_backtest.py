"""Backtest and database integrity tests."""

from __future__ import annotations

import pytest
import pymysql

from data.database import Database
from models.backtest import backtest_history


def _rounds(n: int = 80):
    rows = []
    for i in range(n):
        number = (i * 7) % 10
        color = "RED" if number in (0, 2, 4, 6, 8) else "GREEN"
        if number in (0, 5):
            color = f"{color},VIOLET"
        rows.append({"period": str(5000 + i), "number": number, "color": color})
    return rows


def test_backtest_no_future_and_metrics():
    report = backtest_history(_rounds(80), min_history=20, use_ensemble=True)
    assert report["total_predictions"] > 0
    assert 0.0 <= report["number_accuracy"] <= 100.0
    assert 0.0 <= report["color_accuracy"] <= 100.0
    assert "last_20" in report
    assert "confusion_matrix" in report
    # Details length matches total predictions
    assert len(report["details"]) == report["total_predictions"]
    # Each detail uses an actual period from history
    periods = {r["period"] for r in _rounds(80)}
    for detail in report["details"]:
        assert detail["target_period"] in periods


def test_backtest_insufficient_data():
    report = backtest_history(_rounds(10), min_history=30)
    assert report["total_predictions"] == 0
    assert report["insufficient_sample_size"] is True


def test_duplicate_period_prevention():
    try:
        db = Database(database="wingo30_test")
    except pymysql.MySQLError as exc:
        pytest.skip(f"XAMPP MySQL not available: {exc}")

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rounds WHERE period = %s", ("P1",))

    assert db.upsert_round("P1", 1, "RED") is True
    assert db.upsert_round("P1", 2, "GREEN") is False
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT number FROM rounds WHERE period = %s", ("P1",))
            row = cur.fetchone()
    assert row["number"] == 1
