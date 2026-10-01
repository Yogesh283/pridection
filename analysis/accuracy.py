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


def is_scored_big_small(row: dict[str, Any]) -> bool:
    """WAIT / null BS tip is never scored."""
    pred = row.get("predicted_big_small")
    if pred is None:
        return False
    if str(pred).strip() == "":
        return False
    if str(row.get("status") or "").upper() in {"WAIT", "RESOLVED_WAIT"}:
        # RESOLVED_WAIT still has null tip — exclude from BS accuracy.
        if str(row.get("status") or "").upper() == "RESOLVED_WAIT":
            return False
    return True


def scored_big_small_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if is_scored_big_small(r)]


def _big_small_correct_flag(row: dict[str, Any]) -> int | None:
    """Return 0/1 for scored tips, None for WAIT (excluded)."""
    if not is_scored_big_small(row):
        return None
    if row.get("big_small_correct") is not None:
        return int(row["big_small_correct"])
    act_n = row.get("actual_number")
    if act_n is None:
        return None
    actual_bs = big_small_label(int(act_n))
    pred_bs = str(row.get("predicted_big_small") or "").upper()
    if not pred_bs:
        return None
    return int(pred_bs == actual_bs)


def window_accuracy(rows: list[dict[str, Any]], n: int) -> dict[str, float]:
    subset = rows[-n:] if n > 0 else rows
    scored = scored_big_small_rows(subset)
    total = len(subset)
    number_correct = sum(1 for r in subset if r.get("number_correct") == 1)
    color_correct = sum(1 for r in subset if r.get("color_correct") == 1)
    bs_flags = [_big_small_correct_flag(r) for r in scored]
    bs_correct = sum(1 for f in bs_flags if f == 1)
    bs_n = len(bs_flags)
    confidences = [
        float(row["big_small_probability"])
        for row in scored
        if row.get("big_small_probability") is not None
    ]
    return {
        "sample_size": float(total),
        "number_accuracy": _pct(number_correct, total),
        "color_accuracy": _pct(color_correct, total),
        "big_small_accuracy": _pct(bs_correct, bs_n),
        "big_small_scored": float(bs_n),
        "correct": float(bs_correct),
        "incorrect": float(bs_n - bs_correct),
        "wait_excluded": float(total - bs_n),
        "coverage": _pct(bs_n, total),
        "average_confidence": (
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
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

    scored = scored_big_small_rows(resolved)
    bs_flags = [_big_small_correct_flag(r) for r in scored]
    bs_correct = sum(1 for f in bs_flags if f == 1)
    bs_n = len(bs_flags)
    wait_n = total - bs_n

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in resolved:
        by_model[row.get("model_name") or row.get("decision_strategy") or "unknown"].append(
            row
        )

    model_accuracy = {}
    for name, rows in by_model.items():
        nc = sum(1 for r in rows if r.get("number_correct") == 1)
        cc = sum(1 for r in rows if r.get("color_correct") == 1)
        srows = scored_big_small_rows(rows)
        bc = sum(1 for r in srows if _big_small_correct_flag(r) == 1)
        model_accuracy[name] = {
            "sample_size": len(rows),
            "number_accuracy": _pct(nc, len(rows)),
            "color_accuracy": _pct(cc, len(rows)),
            "big_small_accuracy": _pct(bc, len(srows)),
            "big_small_scored": len(srows),
            "wait_excluded": len(rows) - len(srows),
        }

    insufficient = bs_n < MIN_SAMPLE_FOR_ACCURACY_CLAIM

    # Class metrics on scored tips only.
    tip_big = [r for r in scored if str(r.get("predicted_big_small") or "").upper() == "BIG"]
    tip_small = [
        r for r in scored if str(r.get("predicted_big_small") or "").upper() == "SMALL"
    ]
    act_big = [
        r for r in scored if str(r.get("actual_big_small") or "").upper() == "BIG"
        or (
            r.get("actual_number") is not None
            and big_small_label(int(r["actual_number"])) == "BIG"
        )
    ]
    act_small = [
        r for r in scored if str(r.get("actual_big_small") or "").upper() == "SMALL"
        or (
            r.get("actual_number") is not None
            and big_small_label(int(r["actual_number"])) == "SMALL"
        )
    ]
    tp_big = sum(1 for r in tip_big if _big_small_correct_flag(r) == 1)
    tp_small = sum(1 for r in tip_small if _big_small_correct_flag(r) == 1)
    big_prec = _pct(tp_big, len(tip_big))
    small_prec = _pct(tp_small, len(tip_small))
    big_rec = _pct(tp_big, len(act_big))
    small_rec = _pct(tp_small, len(act_small))
    tpr = (tp_big / len(act_big)) if act_big else 0.0
    tnr = (tp_small / len(act_small)) if act_small else 0.0
    balanced = 100.0 * 0.5 * (tpr + tnr) if (act_big or act_small) else 0.0

    return {
        "total_predictions": total,
        "correct_numbers": number_correct,
        "wrong_numbers": total - number_correct,
        "number_accuracy": _pct(number_correct, total),
        "correct_colors": color_correct,
        "wrong_colors": total - color_correct,
        "color_accuracy": _pct(color_correct, total),
        "correct_big_small": bs_correct,
        "wrong_big_small": bs_n - bs_correct,
        "big_small_accuracy": _pct(bs_correct, bs_n),
        "big_small_scored": bs_n,
        "wait_excluded": wait_n,
        "big_precision": big_prec,
        "small_precision": small_prec,
        "big_recall": big_rec,
        "small_recall": small_rec,
        "balanced_accuracy": round(balanced, 2),
        "predicted_big": len(tip_big),
        "predicted_small": len(tip_small),
        "actual_big": len(act_big),
        "actual_small": len(act_small),
        "last_10": window_accuracy(resolved, 10),
        "last_20": window_accuracy(resolved, 20),
        "last_25": window_accuracy(resolved, 25),
        "last_50": window_accuracy(resolved, 50),
        "last_100": window_accuracy(resolved, 100),
        "last_200": window_accuracy(resolved, 200),
        "overall": {
            "sample_size": total,
            "number_accuracy": _pct(number_correct, total),
            "color_accuracy": _pct(color_correct, total),
            "big_small_accuracy": _pct(bs_correct, bs_n),
            "big_small_scored": bs_n,
            "wait_excluded": wait_n,
        },
        "model_accuracy": model_accuracy,
        "confusion_matrix": confusion_matrix_numbers(resolved),
        "insufficient_sample_size": insufficient,
        "note": (
            "Insufficient scored Big/Small sample."
            if insufficient
            else "Accuracy excludes WAIT tips from Big/Small numerator/denominator."
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
        f"BS scored / WAIT  : {report.get('big_small_scored', 0)} / {report.get('wait_excluded', 0)}",
        f"Correct Big/Small : {report.get('correct_big_small', 0)}",
        f"Wrong Big/Small   : {report.get('wrong_big_small', 0)}",
        f"Big/Small accuracy: {report.get('big_small_accuracy', 0.0):.2f}%",
    ]
    if report.get("insufficient_sample_size"):
        lines.append("Note: Insufficient sample size.")
    return "\n".join(lines)
