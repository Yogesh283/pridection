"""Canonical color helpers for storage and evaluation."""

from __future__ import annotations

from typing import Any

from config import COLOR_ALIASES, COLOR_MAP


def canonical_color(raw: Any, number: int | None = None) -> str | None:
    """
    Canonical compound color for model/eval use.
    Always UPPER, comma-joined, no spaces. Examples: RED | GREEN | RED,VIOLET.
    Preserves VIOLET in compounds; does not invent outcomes.
    """
    if raw is None or raw == "":
        if number is not None and int(number) in COLOR_MAP:
            return str(COLOR_MAP[int(number)]).upper().replace(" ", "")
        return None

    if isinstance(raw, (list, tuple)):
        parts = [_token(str(x)) for x in raw]
        parts = [p for p in parts if p]
        return ",".join(_ordered_unique(parts)) if parts else None

    text = str(raw).strip()
    for sep in (",", "/", "|", "+"):
        if sep in text:
            parts = [_token(p) for p in text.split(sep)]
            parts = [p for p in parts if p]
            return ",".join(_ordered_unique(parts)) if parts else None

    single = _token(text)
    if single:
        return single
    return text.upper().replace(" ", "") or None


def _token(part: str) -> str | None:
    key = part.strip().lower()
    if not key:
        return None
    if key in COLOR_ALIASES:
        return COLOR_ALIASES[key]
    upper = key.upper()
    if upper in {"RED", "GREEN", "VIOLET"}:
        return upper
    return None


def _ordered_unique(parts: list[str]) -> list[str]:
    # Prefer RED/GREEN before VIOLET for stable compound order.
    order = {"RED": 0, "GREEN": 1, "VIOLET": 2}
    seen: set[str] = set()
    out: list[str] = []
    for p in sorted(parts, key=lambda x: order.get(x, 9)):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def primary_from_canonical(color: str | None) -> str | None:
    """Single primary label: prefer RED/GREEN over VIOLET."""
    if not color:
        return None
    parts = [p.strip().upper() for p in str(color).split(",") if p.strip()]
    if not parts:
        return None
    for preferred in ("RED", "GREEN"):
        if preferred in parts:
            return preferred
    if "VIOLET" in parts:
        return "VIOLET"
    return parts[0]


# Source trust for round ingestion (higher wins corrections).
SOURCE_TRUST: dict[str, int] = {
    "hist_cdn": 100,
    "dearapi": 50,
    "manual": 40,
    "unknown": 10,
}


def source_rank(source: str | None) -> int:
    if not source:
        return SOURCE_TRUST["unknown"]
    return int(SOURCE_TRUST.get(str(source).strip().lower(), SOURCE_TRUST["unknown"]))
