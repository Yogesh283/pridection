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
    PREDICTION_MODEL,
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
    elif score < 0.45:
        level = "LOW"
    else:
        # Never advertise HIGH on this near-random market.
        level = "MEDIUM"
    # Cap displayed confidence near tip probability (avoid "74% sure" look).
    tip_cap = 0.50 + min(0.12, max(0.0, top_p - 0.50) * 1.2)
    score = float(max(0.05, min(tip_cap, score, 0.62)))

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
        # Legacy predictors above are retained for diagnostics only.
        from models.gradient_boosting import get_live_gb
        from models.production_engine import get_production_engine

        self.gb = get_live_gb()
        self.production = get_production_engine()
        self._last_bs_analysis: dict[str, Any] | None = None

    def _predict_production_big_small(
        self, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        output = self.production.predict(history)
        available = bool(output.get("ok"))
        p_big = output.get("probability_big")
        p_small = output.get("probability_small")
        probabilities = (
            {"BIG": float(p_big), "SMALL": float(p_small)}
            if p_big is not None and p_small is not None
            else {"BIG": 0.5, "SMALL": 0.5}
        )
        strategy = output.get("decision_strategy") or output.get("model_name")
        return {
            "focus": "big_small",
            "status": output.get("status", "PREDICTION_UNAVAILABLE"),
            "number_predictions": [],
            "color_predictions": [],
            "big_small_predictions": [
                {"big_small": side, "probability": round(probability, 6)}
                for side, probability in sorted(
                    probabilities.items(), key=lambda item: item[1], reverse=True
                )
            ],
            "top_number": None,
            "top_color": None,
            "top_big_small": output.get("prediction") if available else None,
            "bs_source": strategy,
            "decision_strategy": strategy,
            "candidate_models": list((self.production.artifact or {}).get("models", {})),
            "ml_is_tip_authority": available,
            "skip_tip": not available,
            "prediction_unavailable": not available,
            "unavailable_reason": output.get("unavailable_reason"),
            "bs_analysis": None,
            "model_name": output.get("model_name"),
            "model_version": output.get("model_version"),
            "trained_on_rows": output.get("trained_rows"),
            "trained_until_period": output.get("trained_until_period"),
            "probability_big": p_big,
            "probability_small": p_small,
            "number_probability": None,
            "color_probability": None,
            "big_small_probability": output.get("confidence"),
            "confidence_score": float(output.get("confidence") or 0.0),
            "confidence_level": "HIGH" if available else "UNAVAILABLE",
            "model_agreement": None,
            "adaptive_weights": dict(
                (self.production.artifact or {}).get("weights", {})
            ),
            "model_outputs": {},
            "sample_size": len(history),
            "models_used": list((self.production.artifact or {}).get("models", {})),
            "disclaimer": (
                "Probability is a model output, not a guaranteed accuracy rate. "
                "No majority or frequency fallback is used."
            ),
        }

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

    def _pick_big_small(
        self, history: list[dict[str, Any]]
    ) -> tuple[str | None, dict[str, float], str, bool, int | None]:
        """
        Live Big/Small tip authority.

        Production: gradient_boosting (PREDICTION_MODEL).
        majority_w8 is diagnostic only — never silent fallback.
        """
        from analysis.big_small_analyze import analyze_then_predict

        # Keep last-10 analysis for UI/diagnostics (not for tip selection).
        bundle = analyze_then_predict(history)
        analysis = bundle.get("analysis") or {}

        maj_out = self.majority.predict(history)
        maj_bs = {str(k): float(v) for k, v in maj_out["big_small"].items()}
        top_num_any = int(max(maj_out["numbers"], key=maj_out["numbers"].get))
        num_alts = [
            {"number": int(n), "pct": round(100.0 * float(p), 1)}
            for n, p in sorted(
                maj_out["numbers"].items(), key=lambda x: x[1], reverse=True
            )[:3]
        ]

        authority = (PREDICTION_MODEL or "gradient_boosting").strip().lower()
        if authority == "gradient_boosting":
            # Controlled retrain (every N new rounds), then predict.
            self.gb.ensure_ready(history, force_retrain=False)
            gb_out = self.gb.predict(history)
            if not gb_out.get("ok"):
                reason = gb_out.get("unavailable_reason") or "unknown"
                tip_meta = {
                    "big_small": None,
                    "number": None,
                    "numbers": num_alts,
                    "number_pct": None,
                    "skip": True,
                    "action": "PREDICTION_UNAVAILABLE",
                    "source": "PREDICTION_UNAVAILABLE",
                    "probs": None,
                    "reason": f"PREDICTION_UNAVAILABLE: {reason}",
                    "gb": gb_out,
                    "diagnostic_majority_w8": maj_bs,
                }
                self._last_bs_analysis = {"analysis": analysis, "tip": tip_meta}
                return (
                    None,
                    {"BIG": 0.5, "SMALL": 0.5},
                    "PREDICTION_UNAVAILABLE",
                    True,
                    None,
                )

            tip = str(gb_out["predicted_big_small"])
            probs = {
                "BIG": float(gb_out["probability_big"]),
                "SMALL": float(gb_out["probability_small"]),
            }
            side = [
                (n, p)
                for n, p in maj_out["numbers"].items()
                if big_small_label(int(n)) == tip
            ]
            tip_num = int(max(side, key=lambda x: x[1])[0]) if side else top_num_any
            tip_meta = {
                "big_small": tip,
                "number": tip_num,
                "numbers": num_alts,
                "number_pct": round(
                    100.0 * float(maj_out["numbers"].get(tip_num, 0.0)), 1
                ),
                "skip": False,
                "action": "TIP",
                "source": "gradient_boosting",
                "probs": probs,
                "reason": (
                    f"gradient_boosting tip={tip} "
                    f"P(BIG)={probs['BIG']:.3f} P(SMALL)={probs['SMALL']:.3f} "
                    f"rows={gb_out.get('trained_on_rows')} "
                    f"ver={gb_out.get('model_version')}"
                ),
                "gb": gb_out,
                "diagnostic_majority_w8": maj_bs,
            }
            self._last_bs_analysis = {"analysis": analysis, "tip": tip_meta}
            return tip, _normalize(probs), "gradient_boosting", False, tip_num

        # Diagnostic-only path if PREDICTION_MODEL explicitly set to majority_w8.
        win = history[-self.majority.bs_window :]
        bs_counts = {
            "BIG": sum(
                1 for r in win if big_small_label(int(r["number"])) == "BIG"
            ),
            "SMALL": sum(
                1 for r in win if big_small_label(int(r["number"])) == "SMALL"
            ),
        }
        if bs_counts["BIG"] == bs_counts["SMALL"]:
            tip_meta = {
                "big_small": None,
                "number": top_num_any,
                "numbers": num_alts,
                "number_pct": round(
                    100.0 * float(maj_out["numbers"].get(top_num_any, 0.0)), 1
                ),
                "skip": True,
                "action": "WAIT",
                "source": "majority_tie",
                "probs": {"BIG": 0.5, "SMALL": 0.5},
                "reason": (
                    f"majority w{self.majority.bs_window} "
                    f"BIG={bs_counts['BIG']} SMALL={bs_counts['SMALL']} tie -> WAIT"
                ),
            }
            self._last_bs_analysis = {"analysis": analysis, "tip": tip_meta}
            return (
                "BIG",
                {"BIG": 0.5, "SMALL": 0.5},
                "majority_tie",
                True,
                top_num_any,
            )

        tip = "BIG" if bs_counts["BIG"] > bs_counts["SMALL"] else "SMALL"
        side = [
            (n, p)
            for n, p in maj_out["numbers"].items()
            if big_small_label(int(n)) == tip
        ]
        tip_num = int(max(side, key=lambda x: x[1])[0]) if side else top_num_any
        tip_meta = {
            "big_small": tip,
            "number": tip_num,
            "numbers": num_alts,
            "number_pct": round(100.0 * float(maj_out["numbers"].get(tip_num, 0.0)), 1),
            "skip": False,
            "action": "TIP",
            "source": "majority_w8",
            "probs": maj_bs,
            "reason": (
                f"majority w{self.majority.bs_window} "
                f"BIG={bs_counts['BIG']} SMALL={bs_counts['SMALL']} -> tip {tip}"
            ),
        }
        self._last_bs_analysis = {"analysis": analysis, "tip": tip_meta}
        return tip, _normalize(maj_bs), "majority_w8", False, tip_num

    def _pick_color(self, history: list[dict[str, Any]]) -> tuple[str, dict[str, float], str]:
        """Color kept for internal metrics only (UI is Big/Small-only)."""
        maj_out = self.majority.predict(history)
        maj_col = {
            k: float(v) for k, v in maj_out["colors"].items() if k in ("RED", "GREEN")
        } or {k: float(v) for k, v in maj_out["colors"].items()}
        top = max(maj_col, key=maj_col.get)
        return str(top), _normalize(maj_col), "majority_color"

    def maybe_train_ml(self, history: list[dict[str, Any]]) -> None:
        # Supervised rows ~= len(history) - warm-up; allow training before 200 raw rows.
        if len(history) < 120:
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
        if focus == "big_small":
            return self._predict_production_big_small(history)
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
        skip_tip = False

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
            # Production Big/Small authority (gradient_boosting by default).
            top_big_small, maj_bs, bs_source, skip_tip, tip_number = self._pick_big_small(
                history
            )
            top_color, maj_col, color_source = self._pick_color(history)
            decision_name = str(bs_source)
            bs_source = f"{bs_source}+{color_source}"
            if top_big_small is None:
                top_big_small = "BIG"  # placeholder; skip_tip prevents scoring
            big_small_predictions = [
                {"big_small": str(k), "probability": round(float(p), 6)}
                for k, p in sorted(maj_bs.items(), key=lambda x: x[1], reverse=True)
            ]
            color_predictions = [
                {"color": str(c), "probability": round(float(p), 6)}
                for c, p in sorted(maj_col.items(), key=lambda x: x[1], reverse=True)
            ]
            if tip_number is not None:
                top_number = int(tip_number)
            side = [
                (n, p)
                for n, p in maj_nums.items()
                if big_small_label(int(n)) == top_big_small
            ]
            if tip_number is None and side:
                top_number = int(max(side, key=lambda x: x[1])[0])
                number_predictions = [
                    {"number": int(n), "probability": round(float(p), 6)}
                    for n, p in sorted(maj_nums.items(), key=lambda x: x[1], reverse=True)
                ]
            elif focus == "big_small" and tip_number is None:
                preferred = [
                    p
                    for p in number_predictions
                    if big_small_label(int(p["number"])) == top_big_small
                ]
                if preferred:
                    top_number = preferred[0]["number"]
            # Keep number_predictions ordered with tip number first when set.
            if tip_number is not None:
                tip_meta = (getattr(self, "_last_bs_analysis", None) or {}).get("tip") or {}
                alts = tip_meta.get("numbers") or []
                if alts:
                    number_predictions = [
                        {
                            "number": int(a["number"]),
                            "probability": round(float(a.get("pct", 0)) / 100.0, 6),
                        }
                        for a in alts
                    ]
                else:
                    number_predictions = [
                        {"number": int(top_number), "probability": 0.1}
                    ]
            conf_nums = {int(k): float(v) for k, v in maj_nums.items()}
            if tip_number is not None:
                conf_nums[int(top_number)] = max(
                    conf_nums.get(int(top_number), 0.0), 0.2
                )

        hist_acc = (
            historical_big_small_accuracy
            if focus == "big_small"
            else historical_number_accuracy
        )
        # For GB authority, confidence tracks model probability (not ensemble vote).
        tip_meta_full = (getattr(self, "_last_bs_analysis", None) or {}).get("tip") or {}
        gb_payload = tip_meta_full.get("gb") if isinstance(tip_meta_full, dict) else None
        if (
            focus == "big_small"
            and isinstance(gb_payload, dict)
            and gb_payload.get("ok")
            and gb_payload.get("confidence") is not None
        ):
            gb_conf = float(gb_payload["confidence"])
            confidence = {
                "confidence_score": round(gb_conf, 4),
                "confidence_level": (
                    "MEDIUM" if gb_conf >= 0.55 else "LOW"
                ),
                "model_agreement": "gb/1",
                "agreement_count": 1,
                "model_count": 1,
                "top_number": top_number,
                "top_color": top_color,
                "top_big_small": top_big_small,
                "focus": focus,
            }
        else:
            confidence = compute_confidence(
                conf_nums,
                {str(k): float(v) for k, v in maj_col.items()},
                outputs,
                sample_size=len(history),
                historical_accuracy=hist_acc,
                big_small_probs={str(k): float(v) for k, v in maj_bs.items()},
                focus=focus,
            )
        if skip_tip:
            confidence["confidence_level"] = "LOW"
            confidence["confidence_score"] = min(
                float(confidence["confidence_score"]), 0.45
            )

        # Resolve decision_strategy for production naming.
        if focus == "big_small":
            if str(bs_source).startswith("PREDICTION_UNAVAILABLE") or skip_tip and (
                "PREDICTION_UNAVAILABLE" in str(bs_source)
            ):
                decision_strategy = "PREDICTION_UNAVAILABLE"
            elif "gradient_boosting" in str(bs_source):
                decision_strategy = "gradient_boosting"
            elif skip_tip and "tie" in str(bs_source):
                decision_strategy = "majority_tie"
            elif str(bs_source).startswith("majority"):
                decision_strategy = "majority_w8"
            else:
                decision_strategy = str(bs_source).split("+")[0]
        else:
            decision_strategy = str(bs_source).split("+")[0] if bs_source else "ensemble"

        gb_is_authority = decision_strategy == "gradient_boosting"

        return {
            "focus": focus,
            "number_predictions": number_predictions,
            "color_predictions": color_predictions,
            "big_small_predictions": big_small_predictions,
            "top_number": top_number,
            "top_color": top_color,
            "top_big_small": None if skip_tip else top_big_small,
            "bs_source": bs_source,
            "decision_strategy": decision_strategy,
            "candidate_models": list(outputs.keys()),
            "ml_is_tip_authority": gb_is_authority,
            "skip_tip": bool(skip_tip),
            "prediction_unavailable": decision_strategy == "PREDICTION_UNAVAILABLE",
            "unavailable_reason": (
                tip_meta_full.get("reason")
                if decision_strategy == "PREDICTION_UNAVAILABLE"
                else None
            ),
            "bs_analysis": getattr(self, "_last_bs_analysis", None),
            "gb_status": self.gb.model_status() if hasattr(self, "gb") else None,
            "model_name": decision_strategy,
            "model_version": (
                (gb_payload or {}).get("model_version")
                if gb_is_authority
                else None
            ),
            "trained_on_rows": (
                (gb_payload or {}).get("trained_on_rows")
                if gb_is_authority
                else None
            ),
            "probability_big": (
                float(maj_bs.get("BIG", 0.5)) if maj_bs else None
            ),
            "probability_small": (
                float(maj_bs.get("SMALL", 0.5)) if maj_bs else None
            ),
            "number_probability": next(
                (
                    p["probability"]
                    for p in number_predictions
                    if p["number"] == top_number
                ),
                number_predictions[0]["probability"] if number_predictions else 0.1,
            ),
            "color_probability": color_predictions[0]["probability"],
            "big_small_probability": (
                None
                if skip_tip
                else big_small_predictions[0]["probability"]
            ),
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
                "Production tip authority is gradient_boosting. "
                "majority_w8 is diagnostic only and never used as silent fallback."
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
    *,
    current_period: str | None = None,
    live_accuracy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from api.history_sync import period_serial

    current = str(current_period) if current_period else None
    target = str(target_period)
    acc = live_accuracy or {}
    skip = bool(ensemble_result.get("skip_tip"))
    unavailable = bool(ensemble_result.get("prediction_unavailable"))
    tip_bs = None if skip else ensemble_result.get("top_big_small")
    bs_analysis = ensemble_result.get("bs_analysis") or {}
    analysis_block = bs_analysis.get("analysis") if isinstance(bs_analysis, dict) else None
    tip_meta = bs_analysis.get("tip") if isinstance(bs_analysis, dict) else None
    tip_number = None
    tip_numbers: list[Any] = []
    tip_number_pct = None
    if isinstance(tip_meta, dict):
        tip_number = tip_meta.get("number")
        tip_numbers = tip_meta.get("numbers") or []
        tip_number_pct = tip_meta.get("number_pct")
    if tip_number is None and not skip:
        tip_number = ensemble_result.get("top_number")

    decision = (
        ensemble_result.get("decision_strategy")
        or ensemble_result.get("bs_source")
        or PREDICTION_MODEL
    )
    if unavailable:
        action = "PREDICTION_UNAVAILABLE"
    elif skip:
        action = "WAIT"
    else:
        action = "TIP"

    return {
        "target_period": target,
        "period": target,
        "next_period": target,
        "next_serial": period_serial(target),
        "current_period": current,
        "current_serial": period_serial(current) if current else None,
        "focus": "big_small",
        "skip": skip,
        "action": action,
        "status": (
            "PREDICTION_AVAILABLE"
            if not skip
            else "PREDICTION_UNAVAILABLE"
        ),
        "analysis": analysis_block,
        "tip_reason": (tip_meta or {}).get("reason")
        if isinstance(tip_meta, dict)
        else None,
        "prediction": tip_bs,
        "prediction_detail": {
            "big_small": tip_bs,
            "number": tip_number,
            "numbers": tip_numbers,
        },
        "probability_big": ensemble_result.get("probability_big"),
        "probability_small": ensemble_result.get("probability_small"),
        "probabilities": {
            "big_small": ensemble_result.get("big_small_probability"),
            "probability_big": ensemble_result.get("probability_big"),
            "probability_small": ensemble_result.get("probability_small"),
            "number": round(float(tip_number_pct or 0) / 100.0, 4)
            if tip_number_pct is not None
            else ensemble_result.get("number_probability"),
        },
        "confidence": ensemble_result.get("confidence_score"),
        "confidence_level": ensemble_result.get("confidence_level"),
        "model_agreement": ensemble_result.get("model_agreement"),
        "strategy": decision,
        "decision_strategy": decision,
        "model_name": ensemble_result.get("model_name") or decision,
        "model_version": ensemble_result.get("model_version"),
        "trained_rows": ensemble_result.get("trained_on_rows"),
        "trained_on_rows": ensemble_result.get("trained_on_rows"),
        "trained_until_period": ensemble_result.get("trained_until_period"),
        "candidate_models": ensemble_result.get("candidate_models"),
        "ml_is_tip_authority": bool(ensemble_result.get("ml_is_tip_authority")),
        "live_accuracy": {
            "big_small_pct": acc.get("big_small_accuracy"),
            "number_pct": acc.get("number_accuracy"),
            "sample": acc.get("big_small_scored") or acc.get("total_predictions"),
            "wait_excluded": acc.get("wait_excluded"),
            "last_10": acc.get("last_10"),
            "last_25": acc.get("last_25"),
            "last_50": acc.get("last_50"),
            "last_100": acc.get("last_100"),
            "note": "WAIT/UNAVAILABLE excluded from BS accuracy. Tip = decision_strategy.",
        },
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "prediction_timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "disclaimer": (
            "Probability is a model output, not guaranteed accuracy. "
            "The production engine has no majority or frequency fallback."
        ),
    }
