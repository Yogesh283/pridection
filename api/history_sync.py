"""Sync official WinGo history CDN into MySQL (authoritative issueNumber)."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from analysis.statistics import big_small_label
from api.color_canon import canonical_color
from config import COLOR_MAP
from data.database import Database

HIST_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_history_list() -> list[dict[str, Any]]:
    url = f"{HIST_URL}?ts={int(time.time() * 1000)}"
    r = requests.get(url, timeout=15, headers=HEADERS)
    r.raise_for_status()
    return list((r.json().get("data") or {}).get("list") or [])


def color_from_hist_row(row: dict[str, Any]) -> str | None:
    raw = row.get("color") or row.get("colour")
    try:
        n = int(row["number"])
    except (KeyError, TypeError, ValueError):
        n = None
    if raw:
        return canonical_color(raw, n)
    if n is not None:
        return canonical_color(COLOR_MAP.get(n), n)
    return None


def sync_history_into_db(db: Database | None = None) -> dict[str, Any]:
    """
    Upsert latest history page from official CDN (highest trust for issueNumber).
    Returns latest settled row (API newest-first).
    """
    db = db or Database()
    rows = fetch_history_list()
    inserted = 0
    for row in rows:
        period = str(row.get("issueNumber") or "")
        if not period:
            continue
        number = int(row["number"])
        raw_color = row.get("color") or row.get("colour")
        ok = db.upsert_round(
            period=period,
            number=number,
            color=color_from_hist_row(row),
            timestamp=None,
            raw_json=json.dumps(row, ensure_ascii=False),
            source="hist_cdn",
            color_raw=str(raw_color) if raw_color is not None else None,
        )
        if ok:
            inserted += 1

    latest = rows[0] if rows else None
    if latest:
        latest = {
            "period": str(latest["issueNumber"]),
            "number": int(latest["number"]),
            "color": color_from_hist_row(latest),
            "big_small": big_small_label(int(latest["number"])),
            "serial": period_serial(str(latest["issueNumber"])),
        }
    return {"inserted": inserted, "fetched": len(rows), "latest": latest}


def period_serial(period: str, digits: int = 5) -> str:
    """Game UI usually highlights the trailing serial of issueNumber."""
    p = str(period)
    if p.isdigit() and len(p) >= digits:
        return p[-digits:]
    return p


def guess_next_period(period: str) -> str:
    """
    Next WinGo_30S issueNumber = current + 1 (zero-padded).

    Official history confirms consecutive periods differ by exactly 1
    within the trading day sequence.
    """
    p = str(period).strip()
    if p.isdigit():
        return str(int(p) + 1).zfill(len(p))
    return f"{p}+1"
