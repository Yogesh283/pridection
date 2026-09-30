"""Pattern helpers for sequence analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Hashable


def transition_matrix(sequence: list[Hashable]) -> dict[Hashable, dict[Hashable, float]]:
    counts: dict[Hashable, Counter] = defaultdict(Counter)
    for prev, nxt in zip(sequence, sequence[1:]):
        counts[prev][nxt] += 1
    matrix: dict[Hashable, dict[Hashable, float]] = {}
    for prev, counter in counts.items():
        total = sum(counter.values()) or 1
        matrix[prev] = {k: v / total for k, v in counter.items()}
    return matrix


def repeating_sequence_scores(
    sequence: list[Hashable],
    next_candidates: list[Hashable],
    max_pattern_len: int = 5,
) -> dict[Hashable, float]:
    """
    Score candidates by how often they follow matching recent tails.

    Uses only past data in `sequence`.
    """
    scores = {c: 0.0 for c in next_candidates}
    if len(sequence) < 2:
        return scores

    for length in range(1, min(max_pattern_len, len(sequence)) + 1):
        pattern = tuple(sequence[-length:])
        weight = float(length)
        for i in range(0, len(sequence) - length):
            window = tuple(sequence[i : i + length])
            following_idx = i + length
            if window == pattern and following_idx < len(sequence):
                nxt = sequence[following_idx]
                if nxt in scores:
                    scores[nxt] += weight
    total = sum(scores.values())
    if total <= 0:
        return {c: 1.0 / len(next_candidates) for c in next_candidates}
    return {c: scores[c] / total for c in next_candidates}


def alternating_score(colors: list[str]) -> float:
    if len(colors) < 2:
        return 0.0
    alternations = sum(1 for a, b in zip(colors, colors[1:]) if a != b)
    return alternations / (len(colors) - 1)


def sequence_similarity(a: list[Any], b: list[Any]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    matches = sum(1 for x, y in zip(a[-n:], b[-n:]) if x == y)
    return matches / n
