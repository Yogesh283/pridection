"""Probability ensemble and confidence scoring."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.statistics import big_small_label
from api.client import primary_color
from config import (
    ENSEMBLE_WEIGHTS,
    MIN_SAMPLE_FOR_ACCURACY_CLAIM,
    PREDICTION_FOCUS,
    PREDICTION_JSON_PATH,
)
from models.predictor import (
    FrequencyModel,
    MajorityWindowModel,
    MarkovModel,
    MLModelBundle,
    PatternModel,
    RandomBaseline,
    RecentWeightedModel,
    _normalize,
    big_small_from_numbers,
)

logger = logging.getLogger(__name__)


def _combine_distributions(
    parts: list[tuple[str, dict[Any, float], float]],
) -> dict[Any, float]:
    combined: dict[Any, float] = {}
    weight_sum = 0.0
    for _name, dist, weight in parts:
        if not dist or weight <= 0:
            continue
        weight_sum += weight
        for key, value in dist.items():
            combined[key] = combined.get(key, 0.0) + weight * float(value)
    if weight_sum <= 0:
        return {}
    return _normalize({k: v / weight_sum for k, v in combined.items()})


def compute_confidence(
    number_probs: dict[int, float],
    color_probs: dict[str, float],
    model_outputs: dict[str, dict[str, Any]],
    sample_size: int,
    historical_accuracy: float | None = None,
    big_small_probs: dict[str, float] | None = None,
    focus: str | None = None,
) -> dict[str, Any]:
    """
    Confidence is NOT probability * 100.

    When focus is big_small, agreement/margin use BIG/SMALL.
    """
    focus = (focus or PREDICTION_FOCUS or "big_small").lower()
    if not number_probs:
        return {
            "confidence_score": 0.0,
            "confidence_level": "LOW",
            "model_agreement": "0/0",
            "agreement_count": 0,
            "model_count": 0,
        }

    ranked_numbers = sorted(number_probs.items(), key=lambda x: x[1], reverse=True)
    ranked_colors = sorted(color_probs.items(), key=lambda x: x[1], reverse=True)
    bs_probs = big_small_probs or big_small_from_numbers(number_probs)
    ranked_bs = sorted(bs_probs.items(), key=lambda x: x[1], reverse=True)

    top_number = ranked_numbers[0][0]
    top_color = ranked_colors[0][0]
    top_bs = ranked_bs[0][0]

    if focus == "big_small":
        top_p = ranked_bs[0][1]
        second_p = ranked_bs[1][1] if len(ranked_bs) > 1 else 0.0
        agree = 0
        model_count = 0
        for _name, out in model_outputs.items():
            if not out:
                continue
            model_count += 1
            bs = out.get("big_small") or big_small_from_numbers(out["numbers"])
            agree += int(max(bs, key=bs.get) == top_bs)
    elif focus == "color":
        top_p = ranked_colors[0][1]
        second_p = ranked_colors[1][1] if len(ranked_colors) > 1 else 0.0
        agree = 0
        model_count = 0
        for _name, out in model_outputs.items():
            if not out:
                continue
            model_count += 1
            agree += int(max(out["colors"], key=out["colors"].get) == top_color)
    else:
        top_p = ranked_numbers[0][1]
        second_p = ranked_numbers[1][1] if len(ranked_numbers) > 1 else 0.0
        agree = 0
        model_count = 0
        for _name, out in model_outputs.items():
            if not out:
                continue
            model_count += 1
            agree += int(max(out["numbers"], key=out["numbers"].get) == top_number)

    agreement_ratio = (agree / model_count) if model_count else 0.0
    margin = max(0.0, top_p - second_p)
    # With ~200+ rounds, sample factor should not crush confidence.
    sample_factor = min(1.0, sample_size / 200.0)
    hist_factor = 0.5
    if historical_accuracy is not None:
        if focus in {"big_small", "color"}:
            # Map 48%-62% historical accuracy into 0..1
            hist_factor = max(0.0, min(1.0, (historical_accuracy - 48.0) / 14.0))
        else:
            hist_factor = max(0.0, min(1.0, (historical_accuracy - 10.0) / 30.0))

    # Strength of top probability itself (e.g. 0.58 majority share).
    strength = max(0.0, min(1.0, (top_p - 0.50) / 0.25)) if focus in {"big_small", "color"} else top_p

    score = (
        0.30 * agreement_ratio
        + 0.25 * min(1.0, margin * 3.0)
        + 0.20 * strength
        + 0.15 * sample_factor
        + 0.10 * hist_factor
    )
    score = float(max(0.05, min(0.92, score)))

    if sample_size < MIN_SAMPLE_FOR_ACCURACY_CLAIM:
        level = "LOW"
    elif score < 0.38:
        level = "LOW"
    elif score < 0.62:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "confidence_score": round(score, 4),
        "confidence_level": level,
        "model_agreement": f"{agree}/{model_count}",
        "agreement_count": agree,
        "model_count": model_count,
        "top_number": top_number,
        "top_color": top_color,
        "top_big_small": top_bs,
        "focus": focus,
    }


class EnsemblePredictor:
    def __init__(self) -> None:
        # bs_window=8 beat 12 on local walk-forward (~53% vs ~51%).
        self.majority = MajorityWindowModel(bs_window=8, color_window=10)
        self.frequency = FrequencyModel()
        self.markov = MarkovModel()
        self.recent = RecentWeightedModel()
        self.pattern = PatternModel()
        self.random = RandomBaseline()
        self.ml = MLModelBundle()
        self._last_train_size = 0
        self._adaptive_weights: dict[str, float] = dict(ENSEMBLE_WEIGHTS)
        self._bs_tie_margin = 0.08
        self._regime_window = 20
        self._cold_threshold = 0.48

    def _maj8_hit_rate(self, history: list[dict[str, Any]], window: int) -> float:
        if len(history) < window + 2:
            return 0.5
        hits = 0
        total = 0
        start = len(history) - window
        for i in range(start, len(history)):
            past = history[:i]
            if len(past) < 3:
                continue
            out = self.majority.predict(past)
            pred = str(
                out.get("top_big_small")
                or max(out["big_small"], key=out["big_small"].get)
            )
            actual = big_small_label(int(history[i]["number"]))
            hits += int(pred == actual)
            total += 1
        return (hits / total) if total else 0.5

    def _color_maj_hit_rate(self, history: list[dict[str, Any]], window: int) -> float:
        if len(history) < window + 2:
            return 0.5
        hits = 0
        total = 0
        start = len(history) - window
        for i in range(start, len(history)):
            past = history[:i]
            if len(past) < 3:
                continue
            out = self.majority.predict(past)
            maj_col = {
                k: v for k, v in out["colors"].items() if k in ("RED", "GREEN")
            } or out["colors"]
            pred = max(maj_col, key=maj_col.get)
            actual = primary_color(history[i].get("color"))
            if not actual:
                continue
            hits += int(pred == actual)
            total += 1
        return (hits / total) if total else 0.5

    def _pick_big_small(self, history: list[dict[str, Any]]) -> tuple[str, dict[str, float], str]:
        """
        Big/Small picker:
        1) Dedicated ML BS models when trained and probability edge >= 8%
        2) Else follow last / anti-streak3
        """
        # Ensure ML has a chance to train on current history.
        if not self.ml.trained and len(history) >= 200:
            self.maybe_train_ml(history)

        ml_out = self.ml.predict(history) if self.ml.trained else None
        if ml_out and ml_out.get("big_small"):
            bs = {k: float(v) for k, v in ml_out["big_small"].items()}
            top = max(bs, key=bs.get)
            edge = abs(float(bs.get(top, 0.5)) - 0.5)
            if edge >= 0.08:
                return str(top), _normalize(bs), "ml_bs"

        last = big_small_label(int(history[-1]["number"]))
        flip = "SMALL" if last == "BIG" else "BIG"
        top = last
        source = "follow_last"
        if len(history) >= 3:
            a = big_small_label(int(history[-1]["number"]))
            b = big_small_label(int(history[-2]["number"]))
            c = big_small_label(int(history[-3]["number"]))
            if a == b == c:
                top = flip
                source = "anti_streak3"
        other = "SMALL" if top == "BIG" else "BIG"
        maj_bs = {top: 0.58, other: 0.42}
        return top, _normalize(maj_bs), source

    def _pick_color(self, history: list[dict[str, Any]]) -> tuple[str, dict[str, float], str]:
        """
        Adaptive Color (RED/GREEN):
        - Default: majority window
        - If majority cold on last ~20: flip last color (stronger on recent stretch)
        """
        maj_out = self.majority.predict(history)
        maj_col = {
            k: float(v) for k, v in maj_out["colors"].items() if k in ("RED", "GREEN")
        } or {k: float(v) for k, v in maj_out["colors"].items()}
        top = max(maj_col, key=maj_col.get)
        source = "majority_color"
        last = primary_color(history[-1].get("color"))
        if not last:
            last = "RED" if int(history[-1]["number"]) % 2 == 0 else "GREEN"
        flip = "GREEN" if last == "RED" else "RED"

        col_rate = self._color_maj_hit_rate(history, self._regime_window)
        if col_rate < self._cold_threshold:
            top = flip
            other = last
            maj_col = {top: 0.58, other: 0.42}
            source = "flip_color_cold"

        other = "GREEN" if top == "RED" else "RED"
        if top not in maj_col or other not in maj_col:
            maj_col = {top: float(maj_col.get(top, 0.58)), other: float(maj_col.get(other, 0.42))}
        return str(top), _normalize({"RED": maj_col.get("RED", 0.0), "GREEN": maj_col.get("GREEN", 0.0)}), source

    def maybe_train_ml(self, history: list[dict[str, Any]]) -> None:
        if len(history) < 200:
            return
        if self.ml.trained and abs(len(history) - self._last_train_size) < 50:
            return
        if self.ml.fit(history):
            self._last_train_size = len(history)

    def _base_models(self) -> dict[str, Any]:
        return {
            "majority": self.majority,
            "frequency": self.frequency,
            "markov": self.markov,
            "recent": self.recent,
            "pattern": self.pattern,
        }

    def refresh_adaptive_weights(
        self,
        history: list[dict[str, Any]],
        focus: str = "big_small",
        window: int = 40,
    ) -> dict[str, float]:
        """
        Re-weight models from recent chronological hits only (no future leakage).
        Better recent Big/Small or Color models get higher live weight.
        """
        weights = dict(ENSEMBLE_WEIGHTS)
        weights["random"] = 0.0
        if len(history) < 25:
            self._adaptive_weights = weights
            return weights

        start = max(15, len(history) - window)
        scores: dict[str, list[int]] = {name: [] for name in self._base_models()}

        for i in range(start, len(history)):
            past = history[:i]
            actual = history[i]
            actual_n = int(actual["number"])
            actual_bs = big_small_label(actual_n)
            actual_c = primary_color(actual.get("color"))
            for name, model in self._base_models().items():
                out = model.predict(past)
                if focus == "color":
                    top = max(out["colors"], key=out["colors"].get)
                    hit = int(bool(actual_c) and top == actual_c)
                else:
                    bs = out.get("big_small") or big_small_from_numbers(out["numbers"])
                    top = max(bs, key=bs.get)
                    hit = int(top == actual_bs)
                scores[name].append(hit)

        for name, hits in scores.items():
            if not hits:
                continue
            acc = sum(hits) / len(hits)
            # Emphasize models clearly above coin-flip; keep a floor.
            weights[name] = max(0.15, (acc ** 2) * 4.0)
        self._adaptive_weights = weights
        return weights

    def predict(
        self,
        history: list[dict[str, Any]],
        historical_number_accuracy: float | None = None,
        historical_big_small_accuracy: float | None = None,
        train_ml: bool = True,
        focus: str | None = None,
    ) -> dict[str, Any] | None:
        if not history:
            logger.warning("No history available; refusing to invent a prediction.")
            return None

        focus = (focus or PREDICTION_FOCUS or "big_small").lower()
        if train_ml:
            self.maybe_train_ml(history)

        weights = self.refresh_adaptive_weights(history, focus=focus)

        outputs: dict[str, dict[str, Any]] = {
            name: model.predict(history) for name, model in self._base_models().items()
        }
        # Random baseline excluded from live blend (adds noise).
        ml_out = self.ml.predict(history)
        if ml_out:
            outputs["ml"] = ml_out

        def parts(key: str) -> list[tuple[str, dict[Any, float], float]]:
            rows = []
            for name, out in outputs.items():
                if key == "big_small":
                    dist = out.get("big_small") or big_small_from_numbers(out["numbers"])
                else:
                    dist = out[key]
                w = weights.get(name, ENSEMBLE_WEIGHTS.get(name, 1.0))
                if w <= 0:
                    continue
                rows.append((name, dist, w))
            return rows

        number_probs = _combine_distributions(parts("numbers"))
        color_probs = _combine_distributions(parts("colors"))
        big_small_probs = _combine_distributions(parts("big_small"))
        if not number_probs or not color_probs or not big_small_probs:
            return None

        # Majority vote boost for Big/Small when models agree.
        votes: dict[str, float] = {"SMALL": 0.0, "BIG": 0.0}
        for name, out in outputs.items():
            w = weights.get(name, 1.0)
            if w <= 0:
                continue
            bs = out.get("big_small") or big_small_from_numbers(out["numbers"])
            top = max(bs, key=bs.get)
            votes[str(top)] = votes.get(str(top), 0.0) + w
        vote_total = sum(votes.values()) or 1.0
        vote_probs = {k: v / vote_total for k, v in votes.items()}
        # Blend ensemble probs with vote (60/40) for stability.
        big_small_probs = _normalize(
            {
                k: 0.6 * float(big_small_probs.get(k, 0.0)) + 0.4 * float(vote_probs.get(k, 0.0))
                for k in ("SMALL", "BIG")
            }
        )

        number_predictions = [
            {"number": int(n), "probability": round(float(p), 6)}
            for n, p in sorted(number_probs.items(), key=lambda x: x[1], reverse=True)
        ]
        color_predictions = [
            {"color": str(c), "probability": round(float(p), 6)}
            for c, p in sorted(color_probs.items(), key=lambda x: x[1], reverse=True)
        ]
        big_small_predictions = [
            {"big_small": str(k), "probability": round(float(p), 6)}
            for k, p in sorted(big_small_probs.items(), key=lambda x: x[1], reverse=True)
        ]

        top_number = number_predictions[0]["number"]
        top_color = color_predictions[0]["color"]
        top_big_small = big_small_predictions[0]["big_small"]
        bs_source = "ensemble"

        maj_out = self.majority.predict(history)
        maj_nums = maj_out["numbers"]
        maj_col = {
            k: v for k, v in maj_out["colors"].items() if k in ("RED", "GREEN")
        } or maj_out["colors"]
        maj_bs = {k: float(v) for k, v in maj_out["big_small"].items()}

        if focus == "number":
            # Number-first: ensemble blend of digit probs (not forced by Big/Small).
            top_number = int(number_predictions[0]["number"])
            top_big_small = big_small_label(top_number)
            top_color = max(maj_col, key=maj_col.get)
            color_predictions = [
                {"color": str(c), "probability": round(float(p), 6)}
                for c, p in sorted(maj_col.items(), key=lambda x: x[1], reverse=True)
            ]
            big_small_predictions = [
                {
                    "big_small": str(top_big_small),
                    "probability": round(
                        float(maj_bs.get(top_big_small, 0.5)), 6
                    ),
                },
                {
                    "big_small": "SMALL" if top_big_small == "BIG" else "BIG",
                    "probability": round(
                        float(
                            maj_bs.get(
                                "SMALL" if top_big_small == "BIG" else "BIG", 0.5
                            )
                        ),
                        6,
                    ),
                },
            ]
            bs_source = "number_ensemble"
            conf_nums = {int(k): float(v) for k, v in number_probs.items()}
        else:
            # Adaptive Big/Small + Color (switches when majority goes cold).
            top_big_small, maj_bs, bs_source = self._pick_big_small(history)
            top_color, maj_col, color_source = self._pick_color(history)
            bs_source = f"{bs_source}+{color_source}"
            big_small_predictions = [
                {"big_small": str(k), "probability": round(float(p), 6)}
                for k, p in sorted(maj_bs.items(), key=lambda x: x[1], reverse=True)
            ]
            color_predictions = [
                {"color": str(c), "probability": round(float(p), 6)}
                for c, p in sorted(maj_col.items(), key=lambda x: x[1], reverse=True)
            ]
            side = [
                (n, p)
                for n, p in maj_nums.items()
                if big_small_label(int(n)) == top_big_small
            ]
            if side:
                top_number = int(max(side, key=lambda x: x[1])[0])
                number_predictions = [
                    {"number": int(n), "probability": round(float(p), 6)}
                    for n, p in sorted(maj_nums.items(), key=lambda x: x[1], reverse=True)
                ]
            elif focus == "big_small":
                preferred = [
                    p
                    for p in number_predictions
                    if big_small_label(int(p["number"])) == top_big_small
                ]
                if preferred:
                    top_number = preferred[0]["number"]
            conf_nums = {int(k): float(v) for k, v in maj_nums.items()}

        hist_acc = (
            historical_big_small_accuracy
            if focus == "big_small"
            else historical_number_accuracy
        )
        confidence = compute_confidence(
            conf_nums,
            {str(k): float(v) for k, v in maj_col.items()},
            outputs,
            sample_size=len(history),
            historical_accuracy=hist_acc,
            big_small_probs={str(k): float(v) for k, v in maj_bs.items()},
            focus=focus,
        )

        return {
            "focus": focus,
            "number_predictions": number_predictions,
            "color_predictions": color_predictions,
            "big_small_predictions": big_small_predictions,
            "top_number": top_number,
            "top_color": top_color,
            "top_big_small": top_big_small,
            "bs_source": bs_source,
            "number_probability": next(
                p["probability"]
                for p in number_predictions
                if p["number"] == top_number
            ),
            "color_probability": color_predictions[0]["probability"],
            "big_small_probability": big_small_predictions[0]["probability"],
            "confidence_score": confidence["confidence_score"],
            "confidence_level": confidence["confidence_level"],
            "model_agreement": confidence["model_agreement"],
            "adaptive_weights": {k: round(v, 3) for k, v in weights.items() if v > 0},
            "model_outputs": {
                name: {
                    "top_number": max(out["numbers"], key=out["numbers"].get),
                    "top_color": max(out["colors"], key=out["colors"].get),
                    "top_big_small": max(
                        out.get("big_small")
                        or big_small_from_numbers(out["numbers"]),
                        key=(
                            out.get("big_small")
                            or big_small_from_numbers(out["numbers"])
                        ).get,
                    ),
                }
                for name, out in outputs.items()
            },
            "sample_size": len(history),
            "models_used": list(outputs.keys()),
            "disclaimer": (
                "Probability is not certainty. This is a statistical estimate "
                "based on historical patterns only."
            ),
        }


def write_prediction_json(
    payload: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    out = Path(path) if path else Path(PREDICTION_JSON_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def build_prediction_export(
    target_period: str,
    ensemble_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "period": target_period,
        "focus": ensemble_result.get("focus", PREDICTION_FOCUS),
        "prediction": {
            "big_small": ensemble_result["top_big_small"],
            "number": ensemble_result["top_number"],
            "color": ensemble_result["top_color"],
        },
        "probabilities": {
            "big_small": ensemble_result["big_small_probability"],
            "number": ensemble_result["number_probability"],
            "color": ensemble_result["color_probability"],
        },
        "confidence": ensemble_result["confidence_score"],
        "confidence_level": ensemble_result["confidence_level"],
        "model_agreement": ensemble_result.get("model_agreement"),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "disclaimer": ensemble_result.get("disclaimer"),
    }
