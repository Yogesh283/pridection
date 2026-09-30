"""Continuous live analyzer: DB store -> predict early -> compare -> accuracy."""

from __future__ import annotations

import logging
import time
from typing import Any

from rich.console import Console

from analysis.accuracy import compute_accuracy_report
from analysis.statistics import big_small_label
from api.client import primary_color
from config import MIN_SAMPLE_FOR_ACCURACY_CLAIM, POLL_INTERVAL_SECONDS
from data.collector import Collector
from data.database import Database
from models.backtest import _guess_next_period
from models.ensemble import (
    EnsemblePredictor,
    build_prediction_export,
    write_prediction_json,
)

logger = logging.getLogger(__name__)
console = Console()

# Predict roughly this many seconds before the next expected settle.
PREDICT_LEAD_SECONDS = 10.0
ROUND_SECONDS = 30.0
FAST_POLL_SECONDS = 2.0


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

    def generate_prediction(self, force: bool = True) -> dict[str, Any] | None:
        rounds = self.db.get_rounds()
        if len(rounds) < 5:
            print("DB mein 5 rounds se kam hain — wait...")
            return None

        latest = rounds[-1]
        target_period = _guess_next_period(str(latest["period"]))

        if not force:
            for pending in self.db.get_unresolved_predictions():
                if pending["target_period"] == target_period:
                    return None

        result = self.predictor.predict(
            rounds,
            historical_number_accuracy=self.historical_number_accuracy(),
            historical_big_small_accuracy=self.historical_big_small_accuracy(),
            train_ml=True,
            focus="big_small",
        )
        if not result:
            return None

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
        export = build_prediction_export(target_period, result)
        write_prediction_json(export)
        self.last_prediction_period = target_period

        resolved = self.db.get_resolved_predictions()
        if resolved:
            report = compute_accuracy_report(resolved)
            self.db.upsert_model_metrics(
                "ensemble",
                sample_size=report["total_predictions"],
                number_accuracy=report["number_accuracy"],
                color_accuracy=report["color_accuracy"],
                big_small_accuracy=report.get("big_small_accuracy", 0.0),
            )

        return {
            "prediction_id": pred_id,
            "target_period": target_period,
            "result": result,
            "export": export,
            "latest": latest,
        }

    def print_prediction(self, bundle: dict[str, Any]) -> None:
        result = bundle["result"]
        print("")
        print("---------- PREDICT ----------")
        print(f"Period    : {bundle['target_period']}")
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
        """Sleep until ~10s before next expected 30s settle, then return."""
        wait = max(1.0, ROUND_SECONDS - seconds_before)
        time.sleep(wait)

    def _poll_until_new_period(self, previous_period: str | None, timeout: float = 45.0) -> dict[str, Any]:
        """Fast-poll API until a new settled period appears or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            summary = self.collector.collect_once()
            current = (summary.get("current") or {}) if summary.get("ok") else {}
            period = current.get("period")
            if period and period != previous_period:
                return summary
            time.sleep(FAST_POLL_SECONDS)
        return self.collector.collect_once()

    def run_live(self, hours: float = 1.0) -> None:
        """
        Live loop for N hours:
        1) store API result into MySQL
        2) compare previous prediction vs actual (CLI)
        3) update live accuracy
        4) predict next period ~10s before settle
        """
        import sys

        # Force line-buffered CLI output on Windows.
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass

        duration = max(0.1, float(hours)) * 3600.0
        started = time.monotonic()
        end_at = started + duration

        # Quiet noisy info logs on console for clean CLI.
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(logging.ERROR)

        print("")
        print("LIVE Color + Big/Small | Ctrl+C stop")
        print("")

        # Bootstrap: collect + predict once.
        summary = self.collector.collect_once()
        latest = self.db.get_latest_round()
        self.last_seen_period = latest["period"] if latest else None
        self.resolve_pending()
        pred = self.generate_prediction(force=True)
        if pred:
            self.print_prediction(pred)

        while time.monotonic() < end_at:
            try:
                remaining = end_at - time.monotonic()
                if remaining <= 0:
                    break

                # Wait until ~10 seconds before expected next result.
                self._wait_until_near_next_round(PREDICT_LEAD_SECONDS)

                # Refresh prediction shortly before settle (same open row, update in place).
                # Do NOT create a second prediction row.
                self.collector.collect_once()
                current = self.db.get_latest_round()
                if current and self.last_seen_period and current["period"] != self.last_seen_period:
                    # New result already arrived during wait.
                    self.last_seen_period = str(current["period"])
                    self.resolve_pending()
                    nxt = self.generate_prediction(force=True)
                    if nxt:
                        self.print_prediction(nxt)
                    continue

                pred = self.generate_prediction(force=True)
                if pred:
                    self.print_prediction(pred)

                # Poll until new result arrives, then compare silently.
                summary = self._poll_until_new_period(self.last_seen_period, timeout=40.0)
                current = (summary.get("current") or {}) if summary.get("ok") else {}
                period = current.get("period")
                if period and period != self.last_seen_period:
                    self.last_seen_period = str(period)
                    self.resolve_pending()
                    nxt = self.generate_prediction(force=True)
                    if nxt:
                        self.print_prediction(nxt)

            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as exc:
                logger.exception("Live loop error: %s", exc)
                time.sleep(5.0)

        self.db.export_csv()
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
                "target_period": _guess_next_period(str(latest["period"])),
                "result": result,
                "latest": latest,
            }
    runner.render_dashboard(prediction)
