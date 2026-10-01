"""Gradient Boosting feature set for Big/Small (no target leakage).

Features for predicting round i use only rounds strictly before i.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.statistics import big_small_label
from api.client import primary_color

# Stable ordered feature names for GradientBoostingClassifier.
FEATURE_NAMES: list[str] = [
    # Big/Small lags
    "lag_bs_1",
    "lag_bs_2",
    "lag_bs_3",
    "lag_bs_4",
    "lag_bs_5",
    "lag_bs_6",
    "lag_bs_7",
    "lag_bs_8",
    "lag_bs_10",
    "lag_bs_12",
    # Number lags
    "lag_number_1",
    "lag_number_2",
    "lag_number_3",
    "lag_number_4",
    "lag_number_5",
    # Recent number stats
    "mean_3",
    "mean_5",
    "mean_8",
    "mean_12",
    "min_5",
    "max_5",
    "min_8",
    "max_8",
    # Big/Small counts
    "big_count_3",
    "big_count_5",
    "big_count_8",
    "big_count_12",
    "small_count_3",
    "small_count_5",
    "small_count_8",
    "small_count_12",
    # Streaks
    "current_big_streak",
    "current_small_streak",
    "previous_streak_length",
    # Color (encoded)
    "previous_color",
    "color_lag_2",
    "color_lag_3",
    "color_red_count_5",
    "color_green_count_5",
    "color_violet_count_5",
    "color_red_count_8",
    "color_green_count_8",
    "color_violet_count_8",
]

_COLOR_CODE = {"RED": 0.0, "GREEN": 1.0, "VIOLET": 2.0, "UNKNOWN": -1.0}
_BS_LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)
_NUM_LAGS = (1, 2, 3, 4, 5)


def _primary(color: Any) -> str:
    return primary_color(color) or "UNKNOWN"


def _bs_code(label: str) -> float:
    return 1.0 if label == "BIG" else 0.0


def _current_streaks(bs: list[str]) -> tuple[int, int, int]:
    """Return (current_big, current_small, previous_streak_length)."""
    if not bs:
        return 0, 0, 0
    cur = bs[-1]
    cur_len = 1
    for i in range(len(bs) - 2, -1, -1):
        if bs[i] == cur:
            cur_len += 1
        else:
            break
    big_streak = cur_len if cur == "BIG" else 0
    small_streak = cur_len if cur == "SMALL" else 0
    # Previous streak: run ending just before the current streak.
    prev_len = 0
    end = len(bs) - cur_len - 1
    if end >= 0:
        prev = bs[end]
        prev_len = 1
        for i in range(end - 1, -1, -1):
            if bs[i] == prev:
                prev_len += 1
            else:
                break
    return big_streak, small_streak, prev_len


def build_gb_feature_dict(history: list[dict[str, Any]]) -> dict[str, float]:
    """Build GB features from settled history only (chronological ascending)."""
    feats: dict[str, float] = {k: 0.0 for k in FEATURE_NAMES}
    if not history:
        return feats

    numbers = [int(r["number"]) for r in history]
    colors = [_primary(r.get("color")) for r in history]
    bs = [big_small_label(n) for n in numbers]
    n = len(numbers)

    for lag in _BS_LAGS:
        key = f"lag_bs_{lag}"
        if n >= lag:
            feats[key] = _bs_code(bs[-lag])
        else:
            feats[key] = 0.5

    for lag in _NUM_LAGS:
        key = f"lag_number_{lag}"
        if n >= lag:
            feats[key] = float(numbers[-lag])
        else:
            feats[key] = 4.5

    for w in (3, 5, 8, 12):
        chunk = numbers[-w:] if n else []
        if chunk:
            feats[f"mean_{w}"] = float(sum(chunk) / len(chunk))
        else:
            feats[f"mean_{w}"] = 4.5
    for w in (5, 8):
        chunk = numbers[-w:] if n else []
        if chunk:
            feats[f"min_{w}"] = float(min(chunk))
            feats[f"max_{w}"] = float(max(chunk))
        else:
            feats[f"min_{w}"] = 0.0
            feats[f"max_{w}"] = 9.0

    for w in (3, 5, 8, 12):
        chunk = bs[-w:] if n else []
        feats[f"big_count_{w}"] = float(sum(1 for x in chunk if x == "BIG"))
        feats[f"small_count_{w}"] = float(sum(1 for x in chunk if x == "SMALL"))

    big_s, small_s, prev_s = _current_streaks(bs)
    feats["current_big_streak"] = float(big_s)
    feats["current_small_streak"] = float(small_s)
    feats["previous_streak_length"] = float(prev_s)

    feats["previous_color"] = _COLOR_CODE.get(colors[-1], -1.0) if colors else -1.0
    feats["color_lag_2"] = (
        _COLOR_CODE.get(colors[-2], -1.0) if n >= 2 else -1.0
    )
    feats["color_lag_3"] = (
        _COLOR_CODE.get(colors[-3], -1.0) if n >= 3 else -1.0
    )
    for w in (5, 8):
        chunk = colors[-w:] if n else []
        feats[f"color_red_count_{w}"] = float(sum(1 for c in chunk if c == "RED"))
        feats[f"color_green_count_{w}"] = float(
            sum(1 for c in chunk if c == "GREEN")
        )
        feats[f"color_violet_count_{w}"] = float(
            sum(1 for c in chunk if c == "VIOLET")
        )

    return feats


def build_gb_vector(history: list[dict[str, Any]]) -> np.ndarray:
    feats = build_gb_feature_dict(history)
    return np.array([float(feats[k]) for k in FEATURE_NAMES], dtype=float)


def build_gb_supervised(
    rounds: list[dict[str, Any]],
    min_history: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chronological supervised rows.
    X[i] from rounds[:i]; y[i] = 1 if BIG else 0 for rounds[i].
    """
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for i in range(min_history, len(rounds)):
        x_rows.append(build_gb_vector(rounds[:i]))
        y_rows.append(1 if big_small_label(int(rounds[i]["number"])) == "BIG" else 0)
    if not x_rows:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,), dtype=int)
    return np.vstack(x_rows), np.array(y_rows, dtype=int)


def detect_period_gaps(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect numeric period gaps without fabricating missing rounds."""
    gaps = 0
    last_p: int | None = None
    for r in rounds:
        p = str(r.get("period") or "").strip()
        if not p.isdigit():
            continue
        cur = int(p)
        if last_p is not None and cur > last_p + 1:
            gaps += cur - last_p - 1
        last_p = cur
    return {"gap_detected": gaps > 0, "gap_count": gaps}
