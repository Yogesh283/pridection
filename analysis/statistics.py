"""Descriptive statistics helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from api.client import primary_color


def number_frequency(numbers: Iterable[int]) -> dict[int, float]:
    values = list(numbers)
    total = len(values) or 1
    counts = Counter(values)
    return {n: counts.get(n, 0) / total for n in range(10)}


def color_frequency(colors: Iterable[str | None]) -> dict[str, float]:
    primaries = [primary_color(c) for c in colors]
    primaries = [c for c in primaries if c]
    total = len(primaries) or 1
    counts = Counter(primaries)
    return {
        "RED": counts.get("RED", 0) / total,
        "GREEN": counts.get("GREEN", 0) / total,
        "VIOLET": counts.get("VIOLET", 0) / total,
    }


def big_small_label(number: int) -> str:
    return "SMALL" if 0 <= number <= 4 else "BIG"


def odd_even_label(number: int) -> str:
    return "EVEN" if number % 2 == 0 else "ODD"


def rolling_entropy(probs: dict[Any, float]) -> float:
    values = np.array(list(probs.values()), dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-(values * np.log2(values)).sum())


def distribution_deviation(probs: dict[int, float], expected: float = 0.1) -> float:
    return float(sum(abs(probs.get(n, 0.0) - expected) for n in range(10)))


def streak_length(sequence: list[Any]) -> int:
    if not sequence:
        return 0
    last = sequence[-1]
    streak = 0
    for item in reversed(sequence):
        if item == last:
            streak += 1
        else:
            break
    return streak


def last_occurrence_distance(sequence: list[Any], value: Any) -> int | None:
    for idx in range(len(sequence) - 1, -1, -1):
        if sequence[idx] == value:
            return len(sequence) - 1 - idx
    return None


def summarize_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    numbers = [int(r["number"]) for r in rounds]
    colors = [r.get("color") for r in rounds]
    return {
        "count": len(rounds),
        "number_frequency": number_frequency(numbers),
        "color_frequency": color_frequency(colors),
        "entropy": rolling_entropy(number_frequency(numbers)),
        "big_ratio": sum(1 for n in numbers if n >= 5) / (len(numbers) or 1),
        "odd_ratio": sum(1 for n in numbers if n % 2 == 1) / (len(numbers) or 1),
    }
