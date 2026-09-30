"""Collect and store publicly available game results."""

from __future__ import annotations

import logging
import time
from typing import Any

from api.client import APIError, SchemaChangedError, WingoAPIClient
from config import POLL_INTERVAL_SECONDS
from data.database import Database

logger = logging.getLogger(__name__)


class Collector:
    def __init__(
        self,
        client: WingoAPIClient | None = None,
        db: Database | None = None,
    ) -> None:
        self.client = client or WingoAPIClient()
        self.db = db or Database()

    def _store_normalized(self, normalized: dict[str, Any]) -> dict[str, Any]:
        inserted = 0
        seen_periods: list[str] = []

        history = list(normalized.get("history") or [])
        current = normalized.get("current")
        records = history + ([current] if current else [])

        for record in records:
            if not record:
                continue
            ok = self.db.upsert_round(
                period=record["period"],
                number=int(record["number"]),
                color=record.get("color"),
                timestamp=record.get("timestamp"),
                raw_json=record.get("raw_json"),
            )
            if ok:
                inserted += 1
                seen_periods.append(record["period"])

        return {
            "inserted": inserted,
            "periods": seen_periods,
            "current": current,
            "countdown": normalized.get("countdown"),
            "fetched_at": normalized.get("fetched_at"),
        }

    def collect_once(self) -> dict[str, Any]:
        """
        Call API once, normalize, store new rounds, return summary.

        Continues gracefully on API failure without inventing results.
        """
        try:
            normalized = self.client.fetch_normalized()
        except SchemaChangedError as exc:
            logger.error("%s", exc)
            return {
                "ok": False,
                "error": str(exc),
                "schema_changed": True,
                "inserted": 0,
            }
        except APIError as exc:
            logger.error("collect_once API failure: %s", exc)
            return {
                "ok": False,
                "error": str(exc),
                "schema_changed": False,
                "inserted": 0,
            }

        summary = self._store_normalized(normalized)
        summary["ok"] = True
        summary["error"] = None
        summary["schema_changed"] = False
        return summary

    def collect_for_duration(
        self,
        hours: float = 1.0,
        poll_interval: float | None = None,
    ) -> dict[str, Any]:
        """
        Auto-fetch for N hours and store every new settled round in MySQL.

        Wingo 30 settles about every 30 seconds, so 1 hour ≈ up to ~120 rounds.
        API only returns the latest result, so continuous polling is required.
        """
        interval = (
            poll_interval if poll_interval is not None else POLL_INTERVAL_SECONDS
        )
        duration_seconds = max(0.0, float(hours) * 3600.0)
        started = time.monotonic()
        deadline = started + duration_seconds

        total_inserted = 0
        fetch_ok = 0
        fetch_fail = 0
        last_period: str | None = None
        cycles = 0

        logger.info(
            "Starting timed collect: %.2f hour(s), poll=%.1fs, target_db=%s",
            hours,
            interval,
            getattr(self.db, "database", "?"),
        )
        print(
            f"Auto-collect started for {hours} hour(s). "
            f"Polling every {interval:.0f}s. Storing into MySQL."
        )

        while True:
            now = time.monotonic()
            if now >= deadline:
                break

            cycle_started = time.monotonic()
            summary = self.collect_once()
            cycles += 1

            if summary.get("ok"):
                fetch_ok += 1
                inserted = int(summary.get("inserted") or 0)
                total_inserted += inserted
                current = summary.get("current") or {}
                period = current.get("period")
                if period and period != last_period:
                    last_period = str(period)
                    print(
                        f"[{cycles}] NEW period={period} "
                        f"number={current.get('number')} "
                        f"color={current.get('color')} "
                        f"(inserted={inserted}, total_new={total_inserted})"
                    )
                elif inserted:
                    print(
                        f"[{cycles}] Stored {inserted} row(s). "
                        f"total_new={total_inserted}"
                    )
                else:
                    logger.info(
                        "Cycle %s: no new period (current=%s)",
                        cycles,
                        period,
                    )
            else:
                fetch_fail += 1
                print(f"[{cycles}] Fetch failed: {summary.get('error')}")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            countdown = None
            current = summary.get("current") if summary.get("ok") else None
            if isinstance(current, dict):
                countdown = current.get("countdown")
            if countdown is None:
                countdown = summary.get("countdown")

            elapsed = time.monotonic() - cycle_started
            if countdown is not None:
                try:
                    sleep_for = max(1.0, float(countdown) + 1.0 - elapsed)
                except (TypeError, ValueError):
                    sleep_for = max(1.0, interval - elapsed)
            else:
                sleep_for = max(1.0, interval - elapsed)

            sleep_for = min(sleep_for, remaining)
            time.sleep(sleep_for)

        result = {
            "ok": True,
            "hours": hours,
            "duration_seconds": time.monotonic() - started,
            "cycles": cycles,
            "fetch_ok": fetch_ok,
            "fetch_fail": fetch_fail,
            "inserted": total_inserted,
            "stored_rounds": self.db.count_rounds(),
            "last_period": last_period,
        }
        print(
            f"Auto-collect finished. cycles={cycles} inserted={total_inserted} "
            f"db_rounds={result['stored_rounds']}"
        )
        logger.info("Timed collect finished: %s", result)
        return result


def collect_once(
    client: WingoAPIClient | None = None, db: Database | None = None
) -> dict[str, Any]:
    return Collector(client=client, db=db).collect_once()


def collect_for_hours(
    hours: float = 1.0,
    poll_interval: float | None = None,
    client: WingoAPIClient | None = None,
    db: Database | None = None,
) -> dict[str, Any]:
    return Collector(client=client, db=db).collect_for_duration(
        hours=hours,
        poll_interval=poll_interval,
    )
