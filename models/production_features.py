"""Causal feature engineering for the production Big/Small engine.

Every vector is computed from settled rows strictly before the target row.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from analysis.statistics import big_small_label
from api.color_canon import canonical_color

WINDOWS = (3, 5, 8, 10, 15, 20, 30, 50)
BS_LAGS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
NUMBER_LAGS = tuple(range(1, 21))
COLOR_LAGS = (1, 2, 3, 4, 5)
COLORS = ("RED", "GREEN", "VIOLET")


def _color_flags(value: Any, number: int) -> tuple[int, int, int]:
    parts = set((canonical_color(value, number) or "").split(","))
    return tuple(int(color in parts) for color in COLORS)  # type: ignore[return-value]


def _streak_lengths(labels: list[str]) -> tuple[int, int]:
    if not labels:
        return 0, 0
    current = labels[-1]
    current_len = 1
    for value in reversed(labels[:-1]):
        if value != current:
            break
        current_len += 1
    previous_len = 0
    previous_end = len(labels) - current_len - 1
    if previous_end >= 0:
        previous = labels[previous_end]
        previous_len = 1
        for i in range(previous_end - 1, -1, -1):
            if labels[i] != previous:
                break
            previous_len += 1
    return current_len, previous_len


def build_feature_dict(history: list[dict[str, Any]]) -> dict[str, float]:
    """Return a stable, finite feature dictionary from chronological history."""
    if not history:
        return {}

    numbers = [int(row["number"]) for row in history]
    labels = [big_small_label(number) for number in numbers]
    bs = [1 if label == "BIG" else 0 for label in labels]
    color_flags = [
        _color_flags(row.get("color"), int(row["number"])) for row in history
    ]
    features: dict[str, float] = {"history_size": float(len(history))}

    for lag in BS_LAGS:
        features[f"bs_lag_{lag}"] = float(bs[-lag]) if len(bs) >= lag else 0.5
    for lag in NUMBER_LAGS:
        features[f"number_lag_{lag}"] = (
            float(numbers[-lag]) if len(numbers) >= lag else 4.5
        )
    for lag in COLOR_LAGS:
        flags = color_flags[-lag] if len(color_flags) >= lag else (0, 0, 0)
        for index, color in enumerate(COLORS):
            features[f"color_{color.lower()}_lag_{lag}"] = float(flags[index])

    current_streak, previous_streak = _streak_lengths(labels)
    features["bs_current_streak"] = float(current_streak)
    features["bs_previous_streak"] = float(previous_streak)
    features["number_repeat_last"] = float(
        len(numbers) >= 2 and numbers[-1] == numbers[-2]
    )

    for window in WINDOWS:
        nums = numbers[-window:]
        sides = bs[-window:]
        flags = color_flags[-window:]
        size = max(1, len(nums))
        prefix = f"w{window}"

        big_pct = sum(sides) / size
        features[f"bs_big_pct_{prefix}"] = big_pct
        features[f"bs_small_pct_{prefix}"] = 1.0 - big_pct
        features[f"number_mean_{prefix}"] = float(np.mean(nums))
        features[f"number_median_{prefix}"] = float(np.median(nums))
        features[f"number_std_{prefix}"] = float(np.std(nums))
        features[f"number_min_{prefix}"] = float(min(nums))
        features[f"number_max_{prefix}"] = float(max(nums))

        counts = Counter(nums)
        for digit in range(10):
            features[f"number_{digit}_pct_{prefix}"] = counts[digit] / size
        features[f"number_unique_pct_{prefix}"] = len(counts) / size
        features[f"number_repeat_rate_{prefix}"] = (
            sum(nums[i] == nums[i - 1] for i in range(1, len(nums)))
            / max(1, len(nums) - 1)
        )
        deltas = [nums[i] - nums[i - 1] for i in range(1, len(nums))]
        features[f"number_delta_mean_{prefix}"] = (
            float(np.mean(deltas)) if deltas else 0.0
        )
        features[f"number_delta_abs_mean_{prefix}"] = (
            float(np.mean(np.abs(deltas))) if deltas else 0.0
        )
        features[f"number_transition_up_{prefix}"] = (
            sum(delta > 0 for delta in deltas) / max(1, len(deltas))
        )
        features[f"number_transition_down_{prefix}"] = (
            sum(delta < 0 for delta in deltas) / max(1, len(deltas))
        )
        features[f"number_transition_same_{prefix}"] = (
            sum(delta == 0 for delta in deltas) / max(1, len(deltas))
        )

        transitions = list(zip(sides[:-1], sides[1:]))
        transition_count = max(1, len(transitions))
        for left, right, name in (
            (1, 1, "big_big"),
            (1, 0, "big_small"),
            (0, 1, "small_big"),
            (0, 0, "small_small"),
        ):
            features[f"bs_transition_{name}_{prefix}"] = (
                sum(a == left and b == right for a, b in transitions)
                / transition_count
            )
        features[f"bs_alternating_rate_{prefix}"] = (
            sum(a != b for a, b in transitions) / transition_count
        )
        features[f"bs_repeat_rate_{prefix}"] = (
            sum(a == b for a, b in transitions) / transition_count
        )
        features[f"bs_strict_alternating_{prefix}"] = float(
            bool(transitions) and all(a != b for a, b in transitions)
        )
        features[f"bs_strict_repeat_{prefix}"] = float(
            bool(transitions) and all(a == b for a, b in transitions)
        )

        for color_index, color in enumerate(COLORS):
            features[f"color_{color.lower()}_pct_{prefix}"] = (
                sum(row[color_index] for row in flags) / size
            )
        color_pairs = list(zip(flags[:-1], flags[1:]))
        color_pair_count = max(1, len(color_pairs))
        for from_index, from_color in enumerate(COLORS):
            for to_index, to_color in enumerate(COLORS):
                features[
                    f"color_transition_{from_color.lower()}_"
                    f"{to_color.lower()}_{prefix}"
                ] = (
                    sum(
                        left[from_index] == 1 and right[to_index] == 1
                        for left, right in color_pairs
                    )
                    / color_pair_count
                )
        features[f"color_exact_repeat_rate_{prefix}"] = (
            sum(left == right for left, right in color_pairs) / color_pair_count
        )

    return features


def feature_names() -> list[str]:
    """Feature order is derived once from a synthetic 50-row history."""
    rows = [
        {
            "period": str(i),
            "number": i % 10,
            "color": "RED,VIOLET" if i % 10 == 0 else (
                "GREEN,VIOLET" if i % 10 == 5 else (
                    "RED" if i % 2 == 0 else "GREEN"
                )
            ),
        }
        for i in range(50)
    ]
    return sorted(build_feature_dict(rows))


FEATURE_NAMES = feature_names()


def build_feature_vector(history: list[dict[str, Any]]) -> np.ndarray:
    values = build_feature_dict(history)
    return np.asarray([float(values[name]) for name in FEATURE_NAMES], dtype=float)


def build_supervised_dataset(
    rounds: list[dict[str, Any]], min_history: int = 50
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build X/y where target index i uses only ``rounds[:i]``."""
    x_rows: list[np.ndarray] = []
    targets: list[int] = []
    target_indices: list[int] = []
    for i in range(min_history, len(rounds)):
        x_rows.append(build_feature_vector(rounds[:i]))
        targets.append(
            1 if big_small_label(int(rounds[i]["number"])) == "BIG" else 0
        )
        target_indices.append(i)
    if not x_rows:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=float),
            np.empty((0,), dtype=int),
            [],
        )
    return np.vstack(x_rows), np.asarray(targets, dtype=int), target_indices
