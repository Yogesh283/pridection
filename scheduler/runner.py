"""Continuous live analyzer: DB store -> predict early -> compare -> accuracy."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from rich.console import Console

from analysis.accuracy import compute_accuracy_report
from analysis.statistics import big_small_label
from api.client import primary_color
from api.history_sync import guess_next_period, sync_history_into_db
from config import MIN_SAMPLE_FOR_ACCURACY_CLAIM, POLL_INTERVAL_SECONDS, ROOT_DIR
from data.collector import Collector
from data.database import Database
from models.ensemble import (
    EnsemblePredictor,
    build_prediction_export,
    write_prediction_json,
)

logger = logging.getLogger(__name__)
console = Console()

# Predict roughly this many seconds before the next expected settle.
PREDICT_LEAD_SECONDS = 12.0
ROUND_SECONDS = 30.0
FAST_POLL_SECONDS = 1.0


def seconds_until_predict_window(
    round_seconds: float = ROUND_SECONDS,
    lead_seconds: float = PREDICT_LEAD_SECONDS,
) -> float:
    """
    Align to wall-clock 30s boundaries.

    Wake ~lead_seconds before the next settle (:00/:30), so tip is published
    before the result appears on the game.
    """
    now = time.time()
    into = now % round_seconds
    target = max(1.0, round_seconds - float(lead_seconds))
    if into <= target:
        wait = target - into
    else:
        wait = round_seconds - into + target
    return max(0.5, wait)


class AnalyzerRunner:
    def __init__(
        self,
        collector: Collector | None = None,
        db: Database | None = None,
        predictor: EnsemblePredictor | None = None,
        poll_interval: float | None = None,
    ) -> None:
        self.collector = collector or Collector()
        self.db = db or self.collector.db
        self.predictor = predictor or EnsemblePredictor()
        self.poll_interval = (
            poll_interval if poll_interval is not None else POLL_INTERVAL_SECONDS
        )
        self.schema_paused = False
        self.last_prediction_period: str | None = None
        self.last_seen_period: str | None = None
        self._ml_trained = False
        self._last_export: dict[str, Any] | None = None

    def resolve_pending(self) -> list[dict[str, Any]]:
        """Resolve open predictions; return comparison rows for CLI."""
        comparisons: list[dict[str, Any]] = []
        pending = self.db.get_unresolved_predictions()
        if not pending:
            return comparisons
        rounds_by_period = {r["period"]: r for r in self.db.get_rounds()}
        for pred in pending:
            actual = rounds_by_period.get(pred["target_period"])
            if not actual:
                continue
            self.db.resolve_prediction(
                pred["id"],
                actual_number=int(actual["number"]),
                actual_color=actual.get("color"),
            )
            actual_number = int(actual["number"])
            actual_bs = big_small_label(actual_number)
            actual_color = primary_color(actual.get("color")) or actual.get("color")
            pred_bs = pred.get("predicted_big_small") or (
                big_small_label(int(pred["predicted_number"]))
                if pred.get("predicted_number") is not None
                else None
            )
            comparisons.append(
                {
                    "period": pred["target_period"],
                    "pred_bs": pred_bs,
                    "pred_number": pred.get("predicted_number"),
                    "pred_color": pred.get("predicted_color"),
                    "actual_bs": actual_bs,
                    "actual_number": actual_number,
                    "actual_color": actual_color,
                    "bs_ok": pred_bs == actual_bs,
                    "color_ok": bool(
                        pred.get("predicted_color")
                        and actual_color
                        and set(str(pred["predicted_color"]).upper().split(","))
                        & set(str(actual_color).upper().split(","))
                    ),
                    "number_ok": pred.get("predicted_number") == actual_number,
                }
            )
            logger.info(
                "Resolved %s pred=%s/%s actual=%s/%s",
                pred["target_period"],
                pred.get("predicted_number"),
                pred.get("predicted_color"),
                actual_number,
                actual.get("color"),
            )
        return comparisons

    def historical_number_accuracy(self) -> float | None:
        resolved = self.db.get_resolved_predictions()
        if len(resolved) < MIN_SAMPLE_FOR_ACCURACY_CLAIM:
            return None
        report = compute_accuracy_report(resolved)
        return float(report["number_accuracy"])

    def historical_big_small_accuracy(self) -> float | None:
        resolved = self.db.get_resolved_predictions()
        if len(resolved) < MIN_SAMPLE_FOR_ACCURACY_CLAIM:
            return None
        report = compute_accuracy_report(resolved)
        return float(report.get("big_small_accuracy") or 0.0)

    def print_accuracy(self) -> None:
        # Keep live terminal clean — no rolling win/lost spam.
        return

    def generate_prediction(
        self, force: bool = True, *, train_ml: bool | None = None
    ) -> dict[str, Any] | None:
        # Prefer official history CDN for correct issueNumber / serial.
        try:
            sync = sync_history_into_db(self.db)
            synced = (sync.get("latest") or {}).get("period")
            if synced:
                self.last_seen_period = str(synced)
        except Exception as exc:  # noqa: BLE001
            logger.warning("history sync failed: %s", exc)

        rounds = self.db.get_rounds()
        if len(rounds) < 5:
            print("DB mein 5 rounds se kam hain — wait...")
            return None

        latest = rounds[-1]
        current_period = str(latest["period"])
        self.last_seen_period = current_period
        target_period = guess_next_period(current_period)

        # Skip rewrite if tip already matches latest settled + 1.
        if (
            not force
            and self.last_prediction_period == target_period
        ):
            return None

        if not force:
            for pending in self.db.get_unresolved_predictions():
                if pending["target_period"] == target_period:
                    return None

        do_train = (
            bool(train_ml)
            if train_ml is not None
            else not getattr(self, "_ml_trained", False)
        )
        result = self.predictor.predict(
            rounds,
            historical_number_accuracy=self.historical_number_accuracy(),
            historical_big_small_accuracy=self.historical_big_small_accuracy(),
            train_ml=do_train,
            focus="big_small",
        )
        if not result:
            return None
        if do_train:
            self._ml_trained = True

        pred_id = self.db.save_prediction(
            {
                "target_period": target_period,
                "predicted_number": result["top_number"],
                "predicted_color": result["top_color"],
                "predicted_big_small": result["top_big_small"],
                "number_probability": result["number_probability"],
                "color_probability": result["color_probability"],
                "big_small_probability": result["big_small_probability"],
                "model_name": "ensemble",
            },
            update_existing=True,
        )
        resolved = self.db.get_resolved_predictions()
        live_acc = compute_accuracy_report(resolved) if resolved else None
        export = build_prediction_export(
            target_period,
            result,
            current_period=current_period,
            live_accuracy=live_acc,
        )
        write_prediction_json(export)
        self.last_prediction_period = target_period
        self._last_export = export
        self._write_live_status(export)

        if live_acc:
            self.db.upsert_model_metrics(
                "ensemble",
                sample_size=live_acc["total_predictions"],
                number_accuracy=live_acc["number_accuracy"],
                color_accuracy=live_acc["color_accuracy"],
                big_small_accuracy=live_acc.get("big_small_accuracy", 0.0),
            )

        return {
            "prediction_id": pred_id,
            "target_period": target_period,
            "result": result,
            "export": export,
            "latest": latest,
        }

    def print_prediction(self, bundle: dict[str, Any]) -> None:
        from api.history_sync import period_serial

        result = bundle["result"]
        target = str(bundle["target_period"])
        current = str((bundle.get("latest") or {}).get("period") or "")
        print("")
        print("---------- PREDICT ----------")
        if current:
            print(f"Settled   : {current} (serial {period_serial(current)})")
        print(f"Next      : {target} (serial {period_serial(target)})")
        print(f"Big/Small : {result['top_big_small']}")
        print(f"Color     : {result['top_color']}")
        print(
            f"Conf      : {result['confidence_level']} "
            f"({result['confidence_score'] * 100:.0f}%)"
        )
        print("-----------------------------")
        print("")

    def print_comparison(self, row: dict[str, Any]) -> None:
        # Win/Lost result blocks removed from terminal (fully cleared).
        return

    def render_dashboard(self, prediction_bundle: dict[str, Any] | None = None) -> None:
        if prediction_bundle:
            self.print_prediction(prediction_bundle)

    def _wait_until_near_next_round(self, seconds_before: float = PREDICT_LEAD_SECONDS) -> None:
        """Sleep until ~lead seconds before next 30s wall-clock settle."""
        wait = seconds_until_predict_window(ROUND_SECONDS, seconds_before)
        time.sleep(wait)

    def _poll_until_new_period(self, previous_period: str | None, timeout: float = 45.0) -> dict[str, Any]:
        """Fast-poll history+dear until a new settled period appears or timeout."""
        from api.history_sync import sync_history_into_db

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                sync = sync_history_into_db(self.db)
                latest = sync.get("latest") or {}
                period = latest.get("period")
                if period and period != previous_period:
                    return {
                        "ok": True,
                        "current": {
                            "period": period,
                            "number": latest.get("number"),
                            "color": latest.get("color"),
                        },
                    }
            except Exception as exc:  # noqa: BLE001
                logger.debug("hist poll: %s", exc)
            summary = self.collector.collect_once()
            current = (summary.get("current") or {}) if summary.get("ok") else {}
            period = current.get("period")
            if period and period != previous_period:
                return summary
            time.sleep(FAST_POLL_SECONDS)
        return self.collector.collect_once()

    def _write_live_status(self, export: dict[str, Any] | None = None) -> None:
        """Heartbeat JSON for the website (auto-refresh health)."""
        from pathlib import Path

        payload = {
            "ok": True,
            "server_time": time.time(),
            "server_time_iso": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "last_seen_period": self.last_seen_period,
            "last_prediction_period": self.last_prediction_period,
            "seconds_to_predict_window": round(seconds_until_predict_window(), 1),
            "prediction": export,
        }
        path = Path(ROOT_DIR) / "live_status.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run_live(self, hours: float = 1.0) -> None:
        """
        Fast live loop (1s ticks):
        - sync official history issueNumber every tick
        - as soon as serial changes, publish NEXT tip immediately
        - heartbeat live_status.json so the website auto-refreshes
        """
        import sys

        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass

        forever = float(hours) <= 0
        duration = 0.0 if forever else max(0.1, float(hours)) * 3600.0
        started = time.monotonic()
        end_at = None if forever else started + duration

        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(logging.ERROR)

        print("")
        if forever:
            print("LIVE AUTO forever | 1s hist sync | Ctrl+C stop")
        else:
            print(f"LIVE {hours}h | 1s hist sync | Ctrl+C stop")
        print("")

        self._ml_trained = False
        self.collector.collect_once()
        self.resolve_pending()
        pred = self.generate_prediction(force=True, train_ml=True)
        if pred:
            self.print_prediction(pred)

        last_published_current: str | None = None
        if pred and pred.get("latest"):
            last_published_current = str(pred["latest"]["period"])

        while end_at is None or time.monotonic() < end_at:
            try:
                # Sync authoritative serial from history CDN.
                try:
                    sync = sync_history_into_db(self.db)
                    latest = sync.get("latest") or {}
                    current_period = latest.get("period")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("tick sync failed: %s", exc)
                    current_period = None
                    latest = {}

                if not current_period:
                    self.collector.collect_once()
                    row = self.db.get_latest_round()
                    if row:
                        current_period = str(row["period"])
                        latest = {
                            "period": current_period,
                            "number": row.get("number"),
                            "color": row.get("color"),
                        }

                if current_period:
                    self.last_seen_period = str(current_period)
                    expected_next = guess_next_period(str(current_period))

                    # New settled round -> resolve + fresh next tip immediately.
                    if current_period != last_published_current:
                        self.resolve_pending()
                        nxt = self.generate_prediction(force=True, train_ml=False)
                        if nxt:
                            self.print_prediction(nxt)
                            last_published_current = str(
                                (nxt.get("latest") or {}).get("period")
                                or current_period
                            )
                        else:
                            last_published_current = str(current_period)
                    elif self.last_prediction_period != expected_next:
                        nxt = self.generate_prediction(force=True, train_ml=False)
                        if nxt:
                            self.print_prediction(nxt)
                            last_published_current = str(current_period)
                    else:
                        # Heartbeat only — keeps website "LIVE" indicator fresh.
                        self._write_live_status(self._last_export)

                time.sleep(1.0)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as exc:
                logger.exception("Live loop error: %s", exc)
                time.sleep(2.0)

        print("Live finished.")

    def run_forever(self, stop_after: int | None = None) -> None:
        self.run_live(hours=24.0 if stop_after is None else max(0.1, stop_after / 120.0))


def run_dashboard_snapshot(db: Database | None = None) -> None:
    runner = AnalyzerRunner(db=db or Database())
    rounds = runner.db.get_rounds()
    prediction = None
    if len(rounds) >= 5:
        result = runner.predictor.predict(
            rounds,
            historical_number_accuracy=runner.historical_number_accuracy(),
            historical_big_small_accuracy=runner.historical_big_small_accuracy(),
            train_ml=False,
            focus="big_small",
        )
        if result:
            latest = rounds[-1]
            prediction = {
                "target_period": guess_next_period(str(latest["period"])),
                "result": result,
                "latest": latest,
            }
    runner.render_dashboard(prediction)
