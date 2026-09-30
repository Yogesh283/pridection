"""Walk-forward historical backtesting without future leakage."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.accuracy import format_accuracy_text
from analysis.statistics import big_small_label
from api.client import primary_color
from models.ensemble import EnsemblePredictor
from models.predictor import FrequencyModel, MarkovModel, PatternModel, RecentWeightedModel


def _guess_next_period(period: str) -> str:
    if period.isdigit():
        width = len(period)
        return str(int(period) + 1).zfill(width)
    return f"{period}+1"


def backtest_history(
    rounds: list[dict[str, Any]],
    min_history: int = 30,
    use_ensemble: bool = True,
) -> dict[str, Any]:
    """
    For every historical round i (starting after min_history):
      1. Use only rounds before i
      2. Generate prediction for period i
      3. Compare with actual result
    """
    if len(rounds) <= min_history:
        return {
            "total_predictions": 0,
            "correct_numbers": 0,
            "wrong_numbers": 0,
            "number_accuracy": 0.0,
            "correct_colors": 0,
            "wrong_colors": 0,
            "color_accuracy": 0.0,
            "big_small_accuracy": 0.0,
            "last_20": {"number_accuracy": 0.0, "color_accuracy": 0.0},
            "last_50": {"number_accuracy": 0.0, "color_accuracy": 0.0},
            "last_100": {"number_accuracy": 0.0, "color_accuracy": 0.0},
            "overall": {"number_accuracy": 0.0, "color_accuracy": 0.0, "sample_size": 0},
            "model_accuracy": {},
            "confusion_matrix": {},
            "details": [],
            "insufficient_sample_size": True,
            "note": "Insufficient sample size.",
        }

    ensemble = EnsemblePredictor() if use_ensemble else None
    baselines = {
        "frequency": FrequencyModel(),
        "markov": MarkovModel(),
        "recent": RecentWeightedModel(),
        "pattern": PatternModel(),
    }

    details: list[dict[str, Any]] = []
    model_hits: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_correct": 0, "c_correct": 0, "total": 0}
    )
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Optional one-time ML train on early history only (no future).
    if ensemble and len(rounds) > 220:
        ensemble.maybe_train_ml(rounds[:200])

    for i in range(min_history, len(rounds)):
        history = rounds[:i]
        actual = rounds[i]
        actual_number = int(actual["number"])
        actual_color = primary_color(actual.get("color")) or actual.get("color")
        actual_bs = big_small_label(actual_number)

        if ensemble:
            result = ensemble.predict(history, train_ml=False, focus="big_small")
            if not result:
                continue
            pred_number = int(result["top_number"])
            pred_color = str(result["top_color"])
            pred_bs = str(result.get("top_big_small") or big_small_label(pred_number))
            model_name = "ensemble"
        else:
            result = baselines["frequency"].predict(history)
            pred_number = max(result["numbers"], key=result["numbers"].get)
            pred_color = max(result["colors"], key=result["colors"].get)
            pred_bs = max(
                result.get("big_small")
                or {
                    "SMALL": sum(result["numbers"].get(n, 0) for n in range(5)),
                    "BIG": sum(result["numbers"].get(n, 0) for n in range(5, 10)),
                },
                key=(
                    result.get("big_small")
                    or {
                        "SMALL": sum(result["numbers"].get(n, 0) for n in range(5)),
                        "BIG": sum(result["numbers"].get(n, 0) for n in range(5, 10)),
                    }
                ).get,
            )
            model_name = "frequency"

        number_correct = int(pred_number == actual_number)
        color_correct = int(
            bool(pred_color)
            and bool(actual_color)
            and (
                pred_color.upper() in str(actual_color).upper()
                or str(actual_color).upper() in pred_color.upper()
            )
        )
        bs_correct = int(pred_bs == actual_bs)

        details.append(
            {
                "target_period": actual["period"],
                "predicted_number": pred_number,
                "predicted_color": pred_color,
                "predicted_big_small": pred_bs,
                "actual_number": actual_number,
                "actual_color": actual_color,
                "actual_big_small": actual_bs,
                "number_correct": number_correct,
                "color_correct": color_correct,
                "big_small_correct": bs_correct,
                "model_name": model_name,
            }
        )
        confusion[str(pred_number)][str(actual_number)] += 1
        model_hits[model_name]["total"] += 1
        model_hits[model_name]["n_correct"] += number_correct
        model_hits[model_name]["c_correct"] += color_correct

        # Track individual baseline agreement for reporting.
        for bname, bmodel in baselines.items():
            bout = bmodel.predict(history)
            bn = max(bout["numbers"], key=bout["numbers"].get)
            bc = max(bout["colors"], key=bout["colors"].get)
            model_hits[bname]["total"] += 1
            model_hits[bname]["n_correct"] += int(bn == actual_number)
            model_hits[bname]["c_correct"] += int(
                bc and actual_color and bc.upper() == str(actual_color).upper()
            )

    total = len(details)
    correct_numbers = sum(d["number_correct"] for d in details)
    correct_colors = sum(d["color_correct"] for d in details)
    correct_bs = sum(d["big_small_correct"] for d in details)

    def window_stats(n: int) -> dict[str, float]:
        subset = details[-n:]
        t = len(subset) or 1
        return {
            "sample_size": float(len(subset)),
            "number_accuracy": 100.0 * sum(d["number_correct"] for d in subset) / t,
            "color_accuracy": 100.0 * sum(d["color_correct"] for d in subset) / t,
            "big_small_accuracy": 100.0
            * sum(d["big_small_correct"] for d in subset)
            / t,
        }

    model_accuracy = {}
    for name, stats in model_hits.items():
        t = stats["total"] or 1
        model_accuracy[name] = {
            "sample_size": stats["total"],
            "number_accuracy": 100.0 * stats["n_correct"] / t,
            "color_accuracy": 100.0 * stats["c_correct"] / t,
        }

    report = {
        "total_predictions": total,
        "correct_numbers": correct_numbers,
        "wrong_numbers": total - correct_numbers,
        "number_accuracy": 100.0 * correct_numbers / total if total else 0.0,
        "correct_colors": correct_colors,
        "wrong_colors": total - correct_colors,
        "color_accuracy": 100.0 * correct_colors / total if total else 0.0,
        "correct_big_small": correct_bs,
        "wrong_big_small": total - correct_bs,
        "big_small_accuracy": 100.0 * correct_bs / total if total else 0.0,
        "last_20": window_stats(20),
        "last_50": window_stats(50),
        "last_100": window_stats(100),
        "overall": {
            "sample_size": total,
            "number_accuracy": 100.0 * correct_numbers / total if total else 0.0,
            "color_accuracy": 100.0 * correct_colors / total if total else 0.0,
            "big_small_accuracy": 100.0 * correct_bs / total if total else 0.0,
        },
        "model_accuracy": model_accuracy,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "details": details,
        "insufficient_sample_size": total < 20,
        "note": (
            "Insufficient sample size."
            if total < 20
            else "Backtest uses only past data for each prediction."
        ),
    }
    return report


def print_backtest_report(report: dict[str, Any]) -> None:
    print(format_accuracy_text(report))
    if report.get("insufficient_sample_size"):
        print("Insufficient sample size.")
