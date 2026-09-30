"""Unit tests for API parsing and normalization."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests

from api.client import (
    APIError,
    SchemaChangedError,
    WingoAPIClient,
    normalize_color_value,
    normalize_number,
    normalize_result,
    primary_color,
)


def test_normalize_number():
    assert normalize_number("7") == 7
    assert normalize_number(0) == 0
    assert normalize_number("10") is None
    assert normalize_number(None) is None


def test_normalize_color_variants():
    assert normalize_color_value("red") == "RED"
    assert normalize_color_value("green") == "GREEN"
    assert normalize_color_value("violet") == "VIOLET"
    assert normalize_color_value("purple") == "VIOLET"
    assert normalize_color_value("red,violet") == "RED,VIOLET"
    assert normalize_color_value("green,violet") == "GREEN,VIOLET"


def test_primary_color_prefers_red_green():
    assert primary_color("RED,VIOLET") == "RED"
    assert primary_color("GREEN,VIOLET") == "GREEN"
    assert primary_color("VIOLET") == "VIOLET"


def test_normalize_result_live_schema():
    raw = {
        "status": "ok",
        "issueNumber": "20260918100050199",
        "number": "0",
        "colour": "red,violet",
    }
    out = normalize_result(raw)
    assert out["current"]["period"] == "20260918100050199"
    assert out["current"]["number"] == 0
    assert out["current"]["color"] == "RED,VIOLET"
    assert out["current"]["primary_color"] == "RED"


def test_normalize_result_nested_history():
    raw = {
        "data": {
            "current": {
                "period": "1002",
                "result": "3",
                "color": "green",
            },
            "history": [
                {"period": "1001", "number": 1, "color": "green"},
                {"period": "1000", "number": 2, "color": "red"},
            ],
        }
    }
    out = normalize_result(raw)
    assert out["current"]["period"] == "1002"
    assert out["current"]["number"] == 3
    assert len(out["history"]) == 2


def test_schema_changed_raises():
    with pytest.raises(SchemaChangedError):
        normalize_result({"foo": "bar"})


def test_api_failure_retries(monkeypatch):
    client = WingoAPIClient(retries=3, timeout=1)
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.RequestException("network down")
    client.session = mock_session

    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("api.client.time.sleep", fake_sleep)
    with pytest.raises(APIError):
        client.fetch_raw()
    assert mock_session.get.call_count == 3
    assert len(sleeps) == 2
