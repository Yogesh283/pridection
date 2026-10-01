"""HTTP client and dynamic schema normalization for the Wingo 30 API."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

from config import (
    COLOR_ALIASES,
    COLOR_MAP,
    REQUEST_RETRIES,
    REQUEST_TIMEOUT,
    WINGO_API_URL,
)

logger = logging.getLogger(__name__)

# Known field aliases discovered from live inspection + defensive fallbacks.
PERIOD_KEYS = (
    "issueNumber",
    "issue_number",
    "period",
    "periodId",
    "period_id",
    "round",
    "roundId",
    "round_id",
    "gameId",
    "id",
)
NUMBER_KEYS = (
    "number",
    "result",
    "resultNumber",
    "result_number",
    "openCode",
    "open_code",
    "code",
    "value",
)
COLOR_KEYS = (
    "colour",
    "color",
    "resultColor",
    "result_color",
    "openColor",
    "open_color",
)
TIMESTAMP_KEYS = (
    "timestamp",
    "time",
    "createdAt",
    "created_at",
    "openTime",
    "open_time",
    "drawTime",
    "draw_time",
)
COUNTDOWN_KEYS = (
    "countdown",
    "countDown",
    "remainSeconds",
    "remain_seconds",
    "remaining",
    "leftSeconds",
    "left_seconds",
    "timeLeft",
    "time_left",
)
STATUS_KEYS = ("status", "state", "gameStatus", "game_status")
HISTORY_KEYS = (
    "history",
    "histories",
    "list",
    "records",
    "results",
    "data",
    "items",
)


class APIError(Exception):
    """Raised when the remote API cannot be fetched or parsed."""


class SchemaChangedError(Exception):
    """Raised when required result fields cannot be located."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def inspect_structure(obj: Any, prefix: str = "", depth: int = 0, max_depth: int = 4) -> list[str]:
    """Return human-readable key/type diagnostics for nested JSON."""
    lines: list[str] = []
    indent = "  " * depth
    if depth > max_depth:
        lines.append(f"{indent}{prefix}: <max depth>")
        return lines

    if isinstance(obj, dict):
        if not obj:
            lines.append(f"{indent}{prefix or 'root'}: {{}}")
            return lines
        label = prefix or "root"
        lines.append(f"{indent}{label}: object keys={list(obj.keys())}")
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else key
            lines.extend(inspect_structure(value, child, depth + 1, max_depth))
    elif isinstance(obj, list):
        lines.append(f"{indent}{prefix or 'root'}: list len={len(obj)}")
        if obj:
            lines.extend(inspect_structure(obj[0], f"{prefix}[0]", depth + 1, max_depth))
    else:
        sample = repr(obj)
        if len(sample) > 80:
            sample = sample[:77] + "..."
        lines.append(f"{indent}{prefix}: {type(obj).__name__} = {sample}")
    return lines


def print_schema_diagnostic(raw_response: Any) -> None:
    print("API schema diagnostic:")
    for line in inspect_structure(raw_response):
        print(line)


def _find_first(obj: Any, keys: tuple[str, ...]) -> Any:
    """Recursively find the first matching key in nested dict/list structures."""
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] not in (None, ""):
                return obj[key]
        for value in obj.values():
            found = _find_first(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first(item, keys)
            if found is not None:
                return found
    return None


def _collect_history_candidates(obj: Any) -> list[dict[str, Any]]:
    """Find nested lists that look like historical round records."""
    candidates: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in HISTORY_KEYS and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            candidates.append(item)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return candidates


def normalize_color_token(token: str) -> str | None:
    cleaned = token.strip().lower()
    return COLOR_ALIASES.get(cleaned)


def normalize_color_value(raw_color: Any, number: int | None = None) -> str | None:
    """Canonical UPPER color for storage/eval (preserves VIOLET compounds)."""
    from api.color_canon import canonical_color

    return canonical_color(raw_color, number)


def primary_color(color: str | None) -> str | None:
    """Choose a single primary color for classification/metrics."""
    from api.color_canon import canonical_color, primary_from_canonical

    return primary_from_canonical(canonical_color(color))


def has_violet(color: str | None) -> bool:
    if not color:
        return False
    return "VIOLET" in str(color).upper()


def normalize_number(raw_number: Any) -> int | None:
    if raw_number is None or raw_number == "":
        return None
    try:
        value = int(str(raw_number).strip())
    except (TypeError, ValueError):
        return None
    if 0 <= value <= 9:
        return value
    return None


def _normalize_one_record(record: dict[str, Any], source: str = "record") -> dict[str, Any] | None:
    period = _find_first(record, PERIOD_KEYS)
    number = normalize_number(_find_first(record, NUMBER_KEYS))
    raw_color = _find_first(record, COLOR_KEYS)
    color = normalize_color_value(raw_color, number)
    timestamp = _find_first(record, TIMESTAMP_KEYS)
    countdown = _find_first(record, COUNTDOWN_KEYS)
    status = _find_first(record, STATUS_KEYS)

    if period is None or number is None:
        return None

    return {
        "period": str(period),
        "number": number,
        "color": color,
        "primary_color": primary_color(color),
        "has_violet": has_violet(color),
        "timestamp": str(timestamp) if timestamp is not None else _utcnow_iso(),
        "countdown": float(countdown) if countdown is not None else None,
        "status": str(status) if status is not None else None,
        "raw_json": json.dumps(record, ensure_ascii=True),
        "source": source,
    }


def _extract_current_record(raw_response: Any) -> dict[str, Any] | None:
    """Prefer explicit current/latest objects before whole-payload parsing."""
    if isinstance(raw_response, list):
        if raw_response and isinstance(raw_response[0], dict):
            return _normalize_one_record(raw_response[0], source="current")
        return None

    if not isinstance(raw_response, dict):
        return None

    for key in ("current", "latest", "result", "data"):
        node = raw_response.get(key)
        if isinstance(node, dict):
            if key == "data" and isinstance(node.get("current"), dict):
                parsed = _normalize_one_record(node["current"], source="current")
                if parsed:
                    return parsed
            parsed = _normalize_one_record(node, source="current")
            if parsed:
                return parsed

    return _normalize_one_record(raw_response, source="current")


def normalize_result(raw_response: Any) -> dict[str, Any]:
    """
    Normalize an unknown JSON payload into a stable internal structure.

    Observed live schema (2026-09-18):
    {
      "status": "ok",
      "issueNumber": "...",
      "number": "0",
      "colour": "red,violet"
    }
    """
    if not isinstance(raw_response, (dict, list)):
        raise SchemaChangedError("API response is not JSON object/list")

    current = _extract_current_record(raw_response)

    history_records: list[dict[str, Any]] = []
    for item in _collect_history_candidates(raw_response):
        normalized = _normalize_one_record(item, source="history")
        if normalized:
            history_records.append(normalized)

    # Deduplicate history by period, keep first occurrence.
    seen: set[str] = set()
    unique_history: list[dict[str, Any]] = []
    for item in history_records:
        if item["period"] in seen:
            continue
        seen.add(item["period"])
        unique_history.append(item)

    if current is None and not unique_history:
        print_schema_diagnostic(raw_response)
        raise SchemaChangedError(
            "API schema changed — prediction paused. "
            "Could not locate period/number fields."
        )

    return {
        "current": current,
        "history": unique_history,
        "countdown": current.get("countdown") if current else None,
        "status": current.get("status") if current else None,
        "fetched_at": _utcnow_iso(),
        "raw": raw_response,
    }


class WingoAPIClient:
    """Resilient client for publicly available Wingo 30 result data."""

    def __init__(
        self,
        url: str | None = None,
        timeout: int | None = None,
        retries: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.url = url or WINGO_API_URL
        self.timeout = timeout if timeout is not None else REQUEST_TIMEOUT
        self.retries = retries if retries is not None else REQUEST_RETRIES
        self.session = session or requests.Session()

    def fetch_raw(self) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(self.url, timeout=self.timeout)
                if response.status_code != 200:
                    raise APIError(f"HTTP {response.status_code}: {response.text[:200]}")
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise APIError(f"Invalid JSON response: {exc}") from exc
                logger.info("Fetched API successfully on attempt %s", attempt)
                return payload
            except (requests.RequestException, APIError) as exc:
                last_error = exc
                wait = min(2 ** (attempt - 1), 8)
                logger.error(
                    "API request failed (attempt %s/%s): %s",
                    attempt,
                    self.retries,
                    exc,
                )
                if attempt < self.retries:
                    time.sleep(wait)
        raise APIError(f"API failed after {self.retries} retries: {last_error}")

    def fetch_normalized(self) -> dict[str, Any]:
        raw = self.fetch_raw()
        return normalize_result(raw)

    def inspect(self) -> dict[str, Any]:
        raw = self.fetch_raw()
        print_schema_diagnostic(raw)
        return normalize_result(raw)
