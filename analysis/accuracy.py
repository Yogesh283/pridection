"""Accuracy reporting from settled predictions only."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.statistics import big_small_label
from config import MIN_SAMPLE_FOR_ACCURACY_CLAIM


def _pct(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * correct / total


def _big_small_correct_flag(row: dict[str, Any]) -> int:
    if row.get("big_small_correct") is not None:
        return int(row["big_small_correct"])
    pred_n = row.get("predicted_number")
    act_n = row.get("actual_number")
    if pred_n is None or act_n is None:
        return 0
    return int(big_small_label(int(pred_n)) == big_small_label(int(act_n)))


def window_accuracy(rows: list[dict[str, Any]], n: int) -> dict[str, float]:
    subset = rows[-n:] if n > 0 else rows
    total = len(subset)
    number_correct = sum(1 for r in subset if r.get("number_correct") == 1)
    color_correct = sum(1 for r in subset if r.get("color_correct") == 1)
    bs_correct = sum(_big_small_correct_flag(r) for r in subset)
    return {
        "sample_size": float(total),
        "number_accuracy": _pct(number_correct, total),
        "color_accuracy": _pct(color_correct, total),
        "big_small_accuracy": _pct(bs_correct, total),
    }


def confusion_matrix_numbers(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        pred = row.get("predicted_number")
        actual = row.get("actual_number")
        if pred is None or actual is None:
            continue
        matrix[str(pred)][str(actual)] += 1
    return {k: dict(v) for k, v in matrix.items()}


def compute_accuracy_report(resolved: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(resolved)
    number_correct = sum(1 for r in resolved if r.get("number_correct") == 1)
    color_correct = sum(1 for r in resolved if r.get("color_correct") == 1)
    bs_correct = sum(_big_small_correct_flag(r) for r in resolved)

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in resolved:
        by_model[row.get("model_name") or "ensemble"].append(row)

    model_accuracy = {}
    for name, rows in by_model.items():
        nc = sum(1 for r in rows if r.get("number_correct") == 1)
        cc = sum(1 for r in rows if r.get("color_correct") == 1)
        bc = sum(_big_small_correct_flag(r) for r in rows)
        model_accuracy[name] = {
            "sample_size": len(rows),
            "number_accuracy": _pct(nc, len(rows)),
            "color_accuracy": _pct(cc, len(rows)),
            "big_small_accuracy": _pct(bc, len(rows)),
        }

    insufficient = total < MIN_SAMPLE_FOR_ACCURACY_CLAIM
    return {
        "total_predictions": total,
        "correct_numbers": number_correct,
        "wrong_numbers": total - number_correct,
        "number_accuracy": _pct(number_correct, total),
        "correct_colors": color_correct,
        "wrong_colors": total - color_correct,
        "color_accuracy": _pct(color_correct, total),
        "correct_big_small": bs_correct,
        "wrong_big_small": total - bs_correct,
        "big_small_accuracy": _pct(bs_correct, total),
        "last_20": window_accuracy(resolved, 20),
        "last_50": window_accuracy(resolved, 50),
        "last_100": window_accuracy(resolved, 100),
        "overall": {
            "sample_size": total,
            "number_accuracy": _pct(number_correct, total),
            "color_accuracy": _pct(color_correct, total),
            "big_small_accuracy": _pct(bs_correct, total),
        },
        "model_accuracy": model_accuracy,
        "confusion_matrix": confusion_matrix_numbers(resolved),
        "insufficient_sample_size": insufficient,
        "note": (
            "Insufficient sample size."
            if insufficient
            else "Accuracy reflects settled historical predictions only."
        ),
    }


def format_accuracy_text(report: dict[str, Any]) -> str:
    lines = [
        f"Total predictions : {report['total_predictions']}",
        f"Correct numbers   : {report['correct_numbers']}",
        f"Wrong numbers     : {report['wrong_numbers']}",
        f"Number accuracy   : {report['number_accuracy']:.2f}%",
        f"Correct colors    : {report['correct_colors']}",
        f"Wrong colors      : {report['wrong_colors']}",
        f"Color accuracy    : {report['color_accuracy']:.2f}%",
        f"Correct Big/Small : {report.get('correct_big_small', 0)}",
        f"Wrong Big/Small   : {report.get('wrong_big_small', 0)}",
        f"Big/Small accuracy: {report.get('big_small_accuracy', 0.0):.2f}%",
    ]
    if report.get("insufficient_sample_size"):
        lines.append("Note: Insufficient sample size.")
    return "\n".join(lines)
