"""Feature engineering with strict no-leakage rules.

Features for predicting round i use only rounds strictly before i.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.patterns import alternating_score, sequence_similarity, transition_matrix
from analysis.statistics import (
    big_small_label,
    distribution_deviation,
    last_occurrence_distance,
    number_frequency,
    odd_even_label,
    rolling_entropy,
    streak_length,
)
from api.client import primary_color
from config import FEATURE_WINDOWS


def _safe_primary(color: str | None) -> str:
    return primary_color(color) or "UNKNOWN"


def build_feature_vector(history: list[dict[str, Any]]) -> dict[str, float]:
    """
    Build features from settled history only.

    `history` must contain only past rounds (chronological ascending).
    """
    features: dict[str, float] = {}
    numbers = [int(r["number"]) for r in history]
    colors = [_safe_primary(r.get("color")) for r in history]
    big_small = [big_small_label(n) for n in numbers]
    odd_even = [odd_even_label(n) for n in numbers]

    if not numbers:
        return features

    features["history_size"] = float(len(numbers))
    features["last_number"] = float(numbers[-1])
    features["last_is_big"] = 1.0 if numbers[-1] >= 5 else 0.0
    features["last_is_odd"] = 1.0 if numbers[-1] % 2 else 0.0

    for window in FEATURE_WINDOWS:
        window_nums = numbers[-window:]
        window_cols = colors[-window:]
        window_bs = big_small[-window:]
        freq = number_frequency(window_nums)
        for n in range(10):
            features[f"num_freq_w{window}_{n}"] = freq.get(n, 0.0)
        features[f"entropy_w{window}"] = rolling_entropy(freq)
        features[f"dev_w{window}"] = distribution_deviation(freq)

        red = sum(1 for c in window_cols if c == "RED") / (len(window_cols) or 1)
        green = sum(1 for c in window_cols if c == "GREEN") / (len(window_cols) or 1)
        violet = sum(1 for c in window_cols if c == "VIOLET") / (len(window_cols) or 1)
        features[f"color_red_w{window}"] = red
        features[f"color_green_w{window}"] = green
        features[f"color_violet_w{window}"] = violet
        features[f"big_freq_w{window}"] = sum(
            1 for x in window_bs if x == "BIG"
        ) / (len(window_bs) or 1)
        features[f"odd_freq_w{window}"] = sum(
            1 for x in odd_even[-window:] if x == "ODD"
        ) / (len(window_nums) or 1)

    features["number_streak"] = float(streak_length(numbers))
    features["color_streak"] = float(streak_length(colors))
    features["big_small_streak"] = float(streak_length(big_small))
    features["alternating_color"] = alternating_score(colors)
    features["alternating_bs"] = alternating_score(big_small)

    # Explicit Big/Small recent one-hots + transitions (primary market).
    features["last_bs_big"] = 1.0 if big_small[-1] == "BIG" else 0.0
    features["prev2_bs_big"] = (
        1.0 if len(big_small) >= 2 and big_small[-2] == "BIG" else 0.0
    )
    features["prev3_bs_big"] = (
        1.0 if len(big_small) >= 3 and big_small[-3] == "BIG" else 0.0
    )
    if len(big_small) >= 2:
        bs_tm = transition_matrix(big_small)
        prev_bs = big_small[-1]
        features["trans_bs_to_big"] = float(bs_tm.get(prev_bs, {}).get("BIG", 0.0))
        features["trans_bs_to_small"] = float(bs_tm.get(prev_bs, {}).get("SMALL", 0.0))
    else:
        features["trans_bs_to_big"] = 0.5
        features["trans_bs_to_small"] = 0.5

    # Mean of last k encoded as BIG=1.
    for k in (3, 5, 8, 12):
        chunk = big_small[-k:]
        features[f"bs_mean_big_w{k}"] = sum(1 for x in chunk if x == "BIG") / (
            len(chunk) or 1
        )

    for n in range(10):
        dist = last_occurrence_distance(numbers, n)
        features[f"gap_number_{n}"] = float(dist if dist is not None else len(numbers))

    # Transition frequencies from previous number/color.
    if len(numbers) >= 2:
        num_tm = transition_matrix(numbers)
        col_tm = transition_matrix(colors)
        prev_n = numbers[-1]
        prev_c = colors[-1]
        for n in range(10):
            features[f"trans_num_{n}"] = float(num_tm.get(prev_n, {}).get(n, 0.0))
        for cname, key in (("RED", "red"), ("GREEN", "green"), ("VIOLET", "violet")):
            features[f"trans_color_{key}"] = float(col_tm.get(prev_c, {}).get(cname, 0.0))
        features["prev_color_red"] = 1.0 if prev_c == "RED" else 0.0
        features["prev_color_green"] = 1.0 if prev_c == "GREEN" else 0.0
        features["prev_color_violet"] = 1.0 if prev_c == "VIOLET" else 0.0
    else:
        for n in range(10):
            features[f"trans_num_{n}"] = 0.1
        features["trans_color_red"] = 1 / 3
        features["trans_color_green"] = 1 / 3
        features["trans_color_violet"] = 1 / 3
        features["prev_color_red"] = 0.0
        features["prev_color_green"] = 0.0
        features["prev_color_violet"] = 0.0

    # Sequence similarity vs recent short patterns.
    if len(numbers) >= 8:
        features["seq_sim_4"] = sequence_similarity(numbers[-8:-4], numbers[-4:])
    else:
        features["seq_sim_4"] = 0.0

    return features


def features_to_vector(features: dict[str, float], keys: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    ordered_keys = keys or sorted(features.keys())
    vector = np.array([float(features.get(k, 0.0)) for k in ordered_keys], dtype=float)
    return vector, ordered_keys


def build_supervised_dataset(
    rounds: list[dict[str, Any]],
    min_history: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Chronological supervised dataset.

    For each index i >= min_history:
      X[i] from rounds[:i]
      y_number = rounds[i].number
      y_color primary of rounds[i]
      y_bs = 1 if BIG else 0
    """
    color_to_id = {"RED": 0, "GREEN": 1, "VIOLET": 2}
    x_rows: list[np.ndarray] = []
    y_num: list[int] = []
    y_col: list[int] = []
    y_bs: list[int] = []
    feature_keys: list[str] | None = None

    for i in range(min_history, len(rounds)):
        hist = rounds[:i]
        feats = build_feature_vector(hist)
        vec, feature_keys = features_to_vector(feats, feature_keys)
        x_rows.append(vec)
        num = int(rounds[i]["number"])
        y_num.append(num)
        pc = _safe_primary(rounds[i].get("color"))
        y_col.append(color_to_id.get(pc, 0))
        y_bs.append(1 if big_small_label(num) == "BIG" else 0)

    if not x_rows:
        return (
            np.empty((0, 0)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
            feature_keys or [],
        )

    return (
        np.vstack(x_rows),
        np.array(y_num, dtype=int),
        np.array(y_col, dtype=int),
        np.array(y_bs, dtype=int),
        feature_keys or [],
    )
