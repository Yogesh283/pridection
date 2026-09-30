"""Baseline and ML prediction strategies."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np

from analysis.patterns import repeating_sequence_scores, transition_matrix
from analysis.statistics import big_small_label, number_frequency
from api.client import primary_color
from config import MIN_TRAINING_SAMPLES
from models.features import build_feature_vector, build_supervised_dataset, features_to_vector

logger = logging.getLogger(__name__)

COLOR_LABELS = ["RED", "GREEN", "VIOLET"]
BIG_SMALL_LABELS = ["SMALL", "BIG"]


def _uniform_numbers() -> dict[int, float]:
    return {n: 0.1 for n in range(10)}


def _uniform_colors() -> dict[str, float]:
    return {c: 1 / 3 for c in COLOR_LABELS}


def _uniform_big_small() -> dict[str, float]:
    return {"SMALL": 0.5, "BIG": 0.5}


def _normalize(dist: dict[Any, float]) -> dict[Any, float]:
    total = sum(max(0.0, float(v)) for v in dist.values())
    if total <= 0:
        keys = list(dist.keys())
        if not keys:
            return {}
        return {k: 1.0 / len(keys) for k in keys}
    return {k: max(0.0, float(v)) / total for k, v in dist.items()}


def big_small_from_numbers(number_probs: dict[int, float]) -> dict[str, float]:
    small = sum(float(number_probs.get(n, 0.0)) for n in range(0, 5))
    big = sum(float(number_probs.get(n, 0.0)) for n in range(5, 10))
    return _normalize({"SMALL": small, "BIG": big})


def _attach_big_small(result: dict[str, Any]) -> dict[str, Any]:
    if "big_small" not in result and "numbers" in result:
        result["big_small"] = big_small_from_numbers(result["numbers"])
    return result


def _colors_from_rounds(rounds: list[dict[str, Any]]) -> list[str]:
    out = []
    for r in rounds:
        pc = primary_color(r.get("color"))
        if pc:
            out.append(pc)
    return out


def _big_small_from_rounds(rounds: list[dict[str, Any]]) -> list[str]:
    return [big_small_label(int(r["number"])) for r in rounds]


class MajorityWindowModel:
    """
    Tuned on local historical DB:
    Big/Small majority of last 8 (best walk-forward on stored rounds),
    Color majority of last 10.
    Empirically stronger than noisy multi-model blend on this dataset.
    """

    name = "majority"

    def __init__(self, bs_window: int = 8, color_window: int = 10) -> None:
        self.bs_window = bs_window
        self.color_window = color_window

    def predict(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            return {
                "numbers": _uniform_numbers(),
                "colors": _uniform_colors(),
                "big_small": _uniform_big_small(),
            }
        bs_list = _big_small_from_rounds(history[-self.bs_window :])
        col_list = [
            c
            for c in (
                primary_color(r.get("color"))
                for r in history[-self.color_window :]
            )
            if c
        ]
        bs_counts = Counter(bs_list)
        col_counts = Counter(col_list)
        # Laplace smooth so unanimous windows are not shown as 100% certainty.
        big_small = {
            "SMALL": (bs_counts.get("SMALL", 0) + 1) / (len(bs_list) + 2),
            "BIG": (bs_counts.get("BIG", 0) + 1) / (len(bs_list) + 2),
        }
        col_total = sum(col_counts.values()) or 1
        colors = {
            k: col_counts.get(k, 0) / col_total for k in COLOR_LABELS
        }
        # Prefer clearer majority; on exact tie do NOT default to SMALL
        # (dict order previously forced SMALL and created live bias).
        if bs_counts.get("BIG", 0) == bs_counts.get("SMALL", 0):
            last = bs_list[-1]
            top_bs = "SMALL" if last == "BIG" else "BIG"
            big_small = {"SMALL": 0.5, "BIG": 0.5}
        else:
            top_bs = "BIG" if bs_counts.get("BIG", 0) > bs_counts.get("SMALL", 0) else "SMALL"
        side_nums = [
            int(r["number"])
            for r in history[-self.bs_window :]
            if big_small_label(int(r["number"])) == top_bs
        ]
        num_counts = Counter(side_nums)
        num_total = sum(num_counts.values()) or 1
        numbers = {i: num_counts.get(i, 0) / num_total for i in range(10)}
        if sum(numbers.values()) <= 0:
            numbers = _uniform_numbers()
        return {
            "numbers": _normalize(numbers),
            "colors": _normalize(colors),
            "big_small": _normalize(big_small),
            "top_big_small": top_bs,
        }


class FrequencyModel:
    name = "frequency"

    def predict(self, history: list[dict[str, Any]], window: int = 50) -> dict[str, Any]:
        if not history:
            return {
                "numbers": _uniform_numbers(),
                "colors": _uniform_colors(),
                "big_small": _uniform_big_small(),
            }
        window_rounds = history[-window:]
        numbers = [int(r["number"]) for r in window_rounds]
        colors = _colors_from_rounds(window_rounds)
        bs = _big_small_from_rounds(window_rounds)
        num_probs = number_frequency(numbers)
        color_counts = Counter(colors)
        total = sum(color_counts.values()) or 1
        color_probs = {c: color_counts.get(c, 0) / total for c in COLOR_LABELS}
        bs_counts = Counter(bs)
        bs_total = sum(bs_counts.values()) or 1
        bs_probs = {k: bs_counts.get(k, 0) / bs_total for k in BIG_SMALL_LABELS}
        return _attach_big_small(
            {
                "numbers": _normalize(num_probs),
                "colors": _normalize(color_probs),
                "big_small": _normalize(bs_probs),
            }
        )


class MarkovModel:
    name = "markov"

    def predict(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        if len(history) < 2:
            return {
                "numbers": _uniform_numbers(),
                "colors": _uniform_colors(),
                "big_small": _uniform_big_small(),
            }
        numbers = [int(r["number"]) for r in history]
        colors = _colors_from_rounds(history)
        bs = _big_small_from_rounds(history)
        num_tm = transition_matrix(numbers)
        col_tm = transition_matrix(colors)
        bs_tm = transition_matrix(bs)
        prev_n = numbers[-1]
        prev_c = colors[-1] if colors else "RED"
        prev_bs = bs[-1]
        num_probs = {n: float(num_tm.get(prev_n, {}).get(n, 0.0)) for n in range(10)}
        if sum(num_probs.values()) <= 0:
            num_probs = _uniform_numbers()
        color_probs = {c: float(col_tm.get(prev_c, {}).get(c, 0.0)) for c in COLOR_LABELS}
        if sum(color_probs.values()) <= 0:
            color_probs = _uniform_colors()
        bs_probs = {
            k: float(bs_tm.get(prev_bs, {}).get(k, 0.0)) for k in BIG_SMALL_LABELS
        }
        if sum(bs_probs.values()) <= 0:
            bs_probs = _uniform_big_small()
        return {
            "numbers": _normalize(num_probs),
            "colors": _normalize(color_probs),
            "big_small": _normalize(bs_probs),
        }


class RecentWeightedModel:
    name = "recent"

    def predict(self, history: list[dict[str, Any]], half_life: float = 20.0) -> dict[str, Any]:
        if not history:
            return {
                "numbers": _uniform_numbers(),
                "colors": _uniform_colors(),
                "big_small": _uniform_big_small(),
            }
        numbers = [int(r["number"]) for r in history]
        colors = _colors_from_rounds(history)
        bs = _big_small_from_rounds(history)
        n = len(numbers)
        weights = np.array([0.5 ** ((n - 1 - i) / half_life) for i in range(n)], dtype=float)
        num_scores = {k: 0.0 for k in range(10)}
        for value, weight in zip(numbers, weights):
            num_scores[value] += float(weight)
        color_scores = {c: 0.0 for c in COLOR_LABELS}
        for value, weight in zip(colors, weights[-len(colors) :]):
            if value in color_scores:
                color_scores[value] += float(weight)
        bs_scores = {k: 0.0 for k in BIG_SMALL_LABELS}
        for value, weight in zip(bs, weights):
            bs_scores[value] += float(weight)
        return {
            "numbers": _normalize(num_scores),
            "colors": _normalize(color_scores),
            "big_small": _normalize(bs_scores),
        }


class PatternModel:
    name = "pattern"

    def predict(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        if len(history) < 3:
            return {
                "numbers": _uniform_numbers(),
                "colors": _uniform_colors(),
                "big_small": _uniform_big_small(),
            }
        numbers = [int(r["number"]) for r in history]
        colors = _colors_from_rounds(history)
        bs = _big_small_from_rounds(history)
        return {
            "numbers": _normalize(
                repeating_sequence_scores(numbers, list(range(10)))
            ),
            "colors": _normalize(
                repeating_sequence_scores(colors, COLOR_LABELS)
            ),
            "big_small": _normalize(
                repeating_sequence_scores(bs, BIG_SMALL_LABELS)
            ),
        }


class RandomBaseline:
    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        num = self.rng.random(10)
        col = self.rng.random(3)
        bs = self.rng.random(2)
        return {
            "numbers": _normalize({i: float(num[i]) for i in range(10)}),
            "colors": _normalize({c: float(col[i]) for i, c in enumerate(COLOR_LABELS)}),
            "big_small": _normalize(
                {k: float(bs[i]) for i, k in enumerate(BIG_SMALL_LABELS)}
            ),
        }


class MLModelBundle:
    """Chronological ML models trained only on past data."""

    name = "ml"

    def __init__(self) -> None:
        self.number_models: dict[str, Any] = {}
        self.color_models: dict[str, Any] = {}
        self.bs_models: dict[str, Any] = {}
        self.feature_keys: list[str] = []
        self.trained = False
        self.train_samples = 0
        self.bs_val_accuracy: float | None = None

    def _make_estimators(self) -> dict[str, Any]:
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return {
            "logreg": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000, solver="lbfgs", C=0.5
                        ),
                    ),
                ]
            ),
            "rf": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=6,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
            ),
            "hgb": HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.05,
                max_iter=200,
                min_samples_leaf=15,
                l2_regularization=0.1,
                random_state=42,
            ),
        }

    def fit(self, rounds: list[dict[str, Any]]) -> bool:
        if len(rounds) < MIN_TRAINING_SAMPLES:
            logger.info(
                "Skipping ML training: %s < %s samples",
                len(rounds),
                MIN_TRAINING_SAMPLES,
            )
            self.trained = False
            return False

        x, y_num, y_col, y_bs, keys = build_supervised_dataset(
            rounds, min_history=20
        )
        if len(x) < MIN_TRAINING_SAMPLES:
            self.trained = False
            return False

        # Chronological split: first 80% train, last 20% held out (not shuffled).
        split = int(len(x) * 0.8)
        if split < 50:
            self.trained = False
            return False

        x_train = x[:split]
        y_num_train, y_col_train, y_bs_train = (
            y_num[:split],
            y_col[:split],
            y_bs[:split],
        )
        x_val, y_bs_val = x[split:], y_bs[split:]
        self.feature_keys = keys
        self.number_models = {}
        self.color_models = {}
        self.bs_models = {}

        for name in self._make_estimators():
            try:
                num_est = self._make_estimators()[name]
                col_est = self._make_estimators()[name]
                bs_est = self._make_estimators()[name]
                num_est.fit(x_train, y_num_train)
                col_est.fit(x_train, y_col_train)
                bs_est.fit(x_train, y_bs_train)
                self.number_models[name] = num_est
                self.color_models[name] = col_est
                self.bs_models[name] = bs_est
            except Exception as exc:  # pragma: no cover - estimator edge cases
                logger.warning("ML estimator %s failed: %s", name, exc)

        self.trained = bool(self.bs_models) or bool(self.number_models)
        self.train_samples = len(x_train)

        # Held-out Big/Small accuracy (honest).
        if self.bs_models and len(x_val) > 0:
            votes = np.zeros(len(x_val), dtype=float)
            for model in self.bs_models.values():
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x_val)
                    classes = list(model.classes_)
                    if 1 in classes:
                        votes += proba[:, classes.index(1)]
                    elif True in classes:
                        votes += proba[:, list(classes).index(True)]
            votes /= max(len(self.bs_models), 1)
            pred = (votes >= 0.5).astype(int)
            self.bs_val_accuracy = float((pred == y_bs_val).mean() * 100.0)
            logger.info(
                "ML Big/Small held-out accuracy: %.2f%% (n=%s)",
                self.bs_val_accuracy,
                len(y_bs_val),
            )
        return self.trained

    def predict(self, history: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not self.trained or not self.feature_keys:
            return None
        feats = build_feature_vector(history)
        vec, _ = features_to_vector(feats, self.feature_keys)
        x = vec.reshape(1, -1)

        num_acc = np.zeros(10, dtype=float)
        col_acc = np.zeros(3, dtype=float)
        bs_big = 0.0
        count = 0
        bs_count = 0
        for name, model in self.number_models.items():
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x)[0]
                    classes = list(model.classes_)
                    for cls, p in zip(classes, proba):
                        num_acc[int(cls)] += float(p)
                color_model = self.color_models[name]
                cproba = color_model.predict_proba(x)[0]
                cclasses = list(color_model.classes_)
                for cls, p in zip(cclasses, cproba):
                    col_acc[int(cls)] += float(p)
                count += 1
            except Exception as exc:
                logger.warning("ML predict failed for %s: %s", name, exc)

        for name, model in self.bs_models.items():
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x)[0]
                    classes = list(model.classes_)
                    if 1 in classes:
                        bs_big += float(proba[classes.index(1)])
                    elif 0 in classes and len(classes) == 1:
                        bs_big += 0.0
                    else:
                        # Fallback: class order unknown
                        bs_big += float(proba[-1]) if len(proba) > 1 else 0.5
                    bs_count += 1
            except Exception as exc:
                logger.warning("ML BS predict failed for %s: %s", name, exc)

        if count == 0 and bs_count == 0:
            return None
        if count:
            num_acc /= count
            col_acc /= count
            numbers = _normalize({i: float(num_acc[i]) for i in range(10)})
            colors = _normalize(
                {COLOR_LABELS[i]: float(col_acc[i]) for i in range(3)}
            )
        else:
            numbers = _uniform_numbers()
            colors = _uniform_colors()

        if bs_count:
            p_big = bs_big / bs_count
            big_small = _normalize({"BIG": float(p_big), "SMALL": float(1.0 - p_big)})
        else:
            big_small = big_small_from_numbers(numbers)

        return {
            "numbers": numbers,
            "colors": colors,
            "big_small": big_small,
            "bs_val_accuracy": self.bs_val_accuracy,
        }


def walk_forward_validate(
    rounds: list[dict[str, Any]],
    min_train: int = 200,
    step: int = 20,
) -> dict[str, float]:
    """Simple walk-forward validation for number accuracy using frequency model."""
    if len(rounds) < min_train + 10:
        return {"sample_size": 0.0, "number_accuracy": 0.0, "color_accuracy": 0.0}

    model = FrequencyModel()
    correct_n = 0
    correct_c = 0
    total = 0
    for i in range(min_train, len(rounds), step):
        hist = rounds[:i]
        pred = model.predict(hist)
        top_n = max(pred["numbers"], key=pred["numbers"].get)
        top_c = max(pred["colors"], key=pred["colors"].get)
        actual_n = int(rounds[i]["number"])
        actual_c = primary_color(rounds[i].get("color"))
        correct_n += int(top_n == actual_n)
        correct_c += int(actual_c == top_c)
        total += 1

    return {
        "sample_size": float(total),
        "number_accuracy": 100.0 * correct_n / total if total else 0.0,
        "color_accuracy": 100.0 * correct_c / total if total else 0.0,
    }
