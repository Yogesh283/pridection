"""MySQL (XAMPP) persistence layer."""

from __future__ import annotations

import csv
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pymysql
from pymysql.cursors import DictCursor

from analysis.statistics import big_small_label
from api.color_canon import canonical_color, primary_from_canonical, source_rank
from config import (
    DB_CHARSET,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    EXPORTS_DIR,
)

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Database:
    """Stores rounds/predictions in XAMPP MySQL."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        charset: str | None = None,
        *,
        create_database: bool = True,
    ) -> None:
        self.host = host if host is not None else DB_HOST
        self.port = port if port is not None else DB_PORT
        self.user = user if user is not None else DB_USER
        self.password = password if password is not None else DB_PASSWORD
        self.database = database if database is not None else DB_NAME
        self.charset = charset if charset is not None else DB_CHARSET
        if create_database:
            self._ensure_database()
        self._ensure_schema()

    def _server_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "charset": self.charset,
            "autocommit": True,
            "cursorclass": DictCursor,
            "connect_timeout": 5,
            "read_timeout": 15,
            "write_timeout": 15,
        }

    def _ensure_database(self) -> None:
        conn = pymysql.connect(**self._server_kwargs())
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        finally:
            conn.close()

    @contextmanager
    def connect(self) -> Iterator[Any]:
        conn = pymysql.connect(database=self.database, **self._server_kwargs())
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        from data.migrations import apply_migrations

        with self.connect() as conn:
            apply_migrations(conn)

    def upsert_round(
        self,
        period: str,
        number: int,
        color: str | None,
        timestamp: str | None = None,
        raw_json: str | None = None,
        *,
        source: str = "unknown",
        color_raw: str | None = None,
    ) -> bool:
        """
        Insert or conditionally update a round by period.

        Trust policy (higher wins): hist_cdn > dearapi > manual > unknown.
        Never creates duplicate periods. Lower-trust sources cannot overwrite
        higher-trust number/color. Returns True if inserted or updated.
        """
        created_at = _utcnow_iso()
        period = str(period)
        number = int(number)
        raw_color_in = color_raw if color_raw is not None else color
        canon = canonical_color(color, number)
        src = (source or "unknown").strip().lower()
        new_rank = source_rank(src)

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, number, color, source, color_raw, raw_json "
                    "FROM rounds WHERE period = %s LIMIT 1",
                    (period,),
                )
                existing = cur.fetchone()
                if not existing:
                    cur.execute(
                        """
                        INSERT INTO rounds
                            (period, number, color, timestamp, raw_json, created_at,
                             source, color_raw)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            period,
                            number,
                            canon,
                            timestamp or created_at,
                            raw_json,
                            created_at,
                            src,
                            str(raw_color_in) if raw_color_in is not None else None,
                        ),
                    )
                    logger.info(
                        "Stored round period=%s number=%s color=%s source=%s",
                        period,
                        number,
                        canon,
                        src,
                    )
                    return True

                old_src = (existing.get("source") or "unknown").strip().lower()
                old_rank = source_rank(old_src)
                old_number = int(existing["number"])
                old_color = existing.get("color")
                conflict = old_number != number or (
                    (canon or "") != (canonical_color(old_color, old_number) or "")
                )
                if conflict:
                    logger.warning(
                        "Round conflict period=%s old=(%s,%s,%s) new=(%s,%s,%s) "
                        "ranks old=%s new=%s",
                        period,
                        old_number,
                        old_color,
                        old_src,
                        number,
                        canon,
                        src,
                        old_rank,
                        new_rank,
                    )
                # Higher trust may correct. Equal trust: same trusted source may
                # refresh corrections; otherwise first-write among equals wins.
                if new_rank < old_rank:
                    return False
                if new_rank == old_rank:
                    same_trusted = src == old_src and src in {"hist_cdn", "dearapi"}
                    if conflict and not same_trusted:
                        return False
                    if not conflict:
                        if src == old_src and raw_json and raw_json != existing.get("raw_json"):
                            cur.execute(
                                "UPDATE rounds SET raw_json = %s WHERE period = %s",
                                (raw_json, period),
                            )
                            return True
                        return False
                    # fall through to UPDATE for same trusted source correction

                cur.execute(
                    """
                    UPDATE rounds
                    SET number = %s,
                        color = %s,
                        color_raw = COALESCE(%s, color_raw),
                        timestamp = COALESCE(%s, timestamp),
                        raw_json = COALESCE(%s, raw_json),
                        source = %s
                    WHERE period = %s
                    """,
                    (
                        number,
                        canon,
                        str(raw_color_in) if raw_color_in is not None else None,
                        timestamp,
                        raw_json,
                        src,
                        period,
                    ),
                )
                logger.info(
                    "Updated round period=%s number=%s color=%s source=%s",
                    period,
                    number,
                    canon,
                    src,
                )
                return True

    def get_rounds(self, limit: int | None = None) -> list[dict[str, Any]]:
        # Fixed-width numeric issueNumbers: CAST keeps chronological order explicit.
        if limit is None:
            query = (
                "SELECT * FROM rounds "
                "ORDER BY CAST(period AS DECIMAL(30,0)) ASC, period ASC"
            )
            params: tuple[Any, ...] = ()
        else:
            query = (
                "SELECT * FROM ("
                "SELECT * FROM rounds "
                "ORDER BY CAST(period AS DECIMAL(30,0)) DESC, period DESC LIMIT %s"
                ") AS recent "
                "ORDER BY CAST(period AS DECIMAL(30,0)) ASC, period ASC"
            )
            params = (limit,)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def get_latest_round(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM rounds "
                    "ORDER BY CAST(period AS DECIMAL(30,0)) DESC, period DESC "
                    "LIMIT 1"
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_round_by_period(self, period: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM rounds WHERE period = %s LIMIT 1",
                    (str(period),),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def count_rounds(self) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM rounds")
                row = cur.fetchone()
        return int(row["c"]) if row else 0

    def period_exists(self, period: str) -> bool:
        return self.get_round_by_period(period) is not None

    def save_prediction(self, prediction: dict[str, Any], update_existing: bool = True) -> int:
        created_at = prediction.get("created_at") or _utcnow_iso()
        predicted_number = prediction.get("predicted_number")
        predicted_big_small = prediction.get("predicted_big_small")
        # WAIT / no-tip: keep predicted_big_small NULL (do not invent from number).
        status = prediction.get("status") or (
            "WAIT" if predicted_big_small is None else "TIP"
        )
        decision = prediction.get("decision_strategy") or prediction.get("model_name") or "unknown"
        model_name = prediction.get("model_name") or decision
        pred_color = canonical_color(prediction.get("predicted_color"), predicted_number)

        with self.connect() as conn:
            with conn.cursor() as cur:
                # Keep only one open prediction per target period.
                cur.execute(
                    """
                    SELECT id FROM predictions
                    WHERE target_period = %s AND resolved_at IS NULL
                    ORDER BY id ASC
                    """,
                    (prediction["target_period"],),
                )
                existing_rows = cur.fetchall() or []
                if len(existing_rows) > 1:
                    keep_id = int(existing_rows[0]["id"])
                    for row in existing_rows[1:]:
                        cur.execute(
                            "DELETE FROM predictions WHERE id = %s",
                            (int(row["id"]),),
                        )
                    existing = {"id": keep_id}
                elif existing_rows:
                    existing = existing_rows[0]
                else:
                    existing = None

                if existing:
                    if update_existing:
                        cur.execute(
                            """
                            UPDATE predictions
                            SET predicted_number = %s,
                                predicted_color = %s,
                                predicted_big_small = %s,
                                number_probability = %s,
                                color_probability = %s,
                                big_small_probability = %s,
                                model_name = %s,
                                model_version = %s,
                                decision_strategy = %s,
                                trained_until_period = %s,
                                trained_rows = %s,
                                status = %s,
                                created_at = %s
                            WHERE id = %s AND resolved_at IS NULL
                            """,
                            (
                                predicted_number,
                                pred_color,
                                predicted_big_small,
                                prediction.get("number_probability"),
                                prediction.get("color_probability"),
                                prediction.get("big_small_probability"),
                                model_name,
                                prediction.get("model_version"),
                                decision,
                                prediction.get("trained_until_period"),
                                prediction.get("trained_rows"),
                                status,
                                created_at,
                                int(existing["id"]),
                            ),
                        )
                    return int(existing["id"])

                cur.execute(
                    """
                    INSERT INTO predictions (
                        target_period, predicted_number, predicted_color,
                        predicted_big_small,
                        number_probability, color_probability, big_small_probability,
                        model_name, model_version, decision_strategy,
                        trained_until_period, trained_rows, status, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        prediction["target_period"],
                        predicted_number,
                        pred_color,
                        predicted_big_small,
                        prediction.get("number_probability"),
                        prediction.get("color_probability"),
                        prediction.get("big_small_probability"),
                        model_name,
                        prediction.get("model_version"),
                        decision,
                        prediction.get("trained_until_period"),
                        prediction.get("trained_rows"),
                        status,
                        created_at,
                    ),
                )
                return int(cur.lastrowid)

    def get_unresolved_predictions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM predictions
                    WHERE resolved_at IS NULL
                    ORDER BY created_at ASC, id ASC
                    """
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def resolve_prediction(
        self,
        prediction_id: int,
        actual_number: int,
        actual_color: str | None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Resolve one prediction against an exact actual result.

        Idempotent: if already resolved and force=False, leaves row unchanged.
        WAIT tips (predicted_big_small NULL) get big_small_correct=NULL (unscored).
        """
        actual_bs = big_small_label(int(actual_number))
        actual_color_canon = canonical_color(actual_color, actual_number)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM predictions WHERE id = %s",
                    (prediction_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {"ok": False, "reason": "missing"}
                if row.get("resolved_at") and not force:
                    return {
                        "ok": True,
                        "reason": "already_resolved",
                        "id": prediction_id,
                        "changed": False,
                    }

                number_correct = (
                    1 if row["predicted_number"] == actual_number else 0
                )
                pred_primary = primary_from_canonical(
                    canonical_color(row.get("predicted_color"), row.get("predicted_number"))
                )
                act_primary = primary_from_canonical(actual_color_canon)
                color_correct = (
                    1
                    if pred_primary and act_primary and pred_primary == act_primary
                    else 0
                )

                pred_bs = (row.get("predicted_big_small") or "").strip().upper() or None
                # WAIT / no BS tip: do not score Big/Small.
                if not pred_bs:
                    big_small_correct = None
                    status = "RESOLVED_WAIT"
                else:
                    big_small_correct = 1 if pred_bs == actual_bs else 0
                    status = "RESOLVED"

                cur.execute(
                    """
                    UPDATE predictions
                    SET actual_number = %s,
                        actual_color = %s,
                        actual_big_small = %s,
                        number_correct = %s,
                        color_correct = %s,
                        big_small_correct = %s,
                        status = %s,
                        resolved_at = %s
                    WHERE id = %s
                    """,
                    (
                        actual_number,
                        actual_color_canon,
                        actual_bs,
                        number_correct,
                        color_correct,
                        big_small_correct,
                        status,
                        _utcnow_iso(),
                        prediction_id,
                    ),
                )
                logger.info(
                    "Resolved prediction id=%s target=%s status=%s bs_correct=%s",
                    prediction_id,
                    row.get("target_period"),
                    status,
                    big_small_correct,
                )
                return {
                    "ok": True,
                    "reason": "resolved",
                    "id": prediction_id,
                    "changed": True,
                    "status": status,
                    "big_small_correct": big_small_correct,
                }

    def resolve_pending_against_rounds(self) -> dict[str, Any]:
        """
        Resolve all open predictions whose target_period exists in rounds.
        Exact period match only. Idempotent. Orphans stay unresolved.
        """
        pending = self.get_unresolved_predictions()
        resolved = 0
        skipped_orphan = 0
        already = 0
        details: list[dict[str, Any]] = []
        for pred in pending:
            actual = self.get_round_by_period(str(pred["target_period"]))
            if not actual:
                skipped_orphan += 1
                continue
            out = self.resolve_prediction(
                int(pred["id"]),
                actual_number=int(actual["number"]),
                actual_color=actual.get("color"),
            )
            if out.get("reason") == "already_resolved":
                already += 1
            elif out.get("changed"):
                resolved += 1
            details.append(
                {
                    "id": pred["id"],
                    "target_period": pred["target_period"],
                    **out,
                }
            )
        return {
            "pending": len(pending),
            "resolved_now": resolved,
            "already_resolved": already,
            "orphans_left": skipped_orphan,
            "details": details,
        }

    def get_resolved_predictions(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is None:
            query = (
                "SELECT * FROM predictions WHERE resolved_at IS NOT NULL "
                "ORDER BY resolved_at ASC, id ASC"
            )
            params: tuple[Any, ...] = ()
        else:
            query = (
                "SELECT * FROM ("
                "SELECT * FROM predictions WHERE resolved_at IS NOT NULL "
                "ORDER BY resolved_at DESC, id DESC LIMIT %s"
                ") AS recent ORDER BY resolved_at ASC, id ASC"
            )
            params = (limit,)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def upsert_model_metrics(
        self,
        model_name: str,
        sample_size: int,
        number_accuracy: float,
        color_accuracy: float,
        big_small_accuracy: float = 0.0,
    ) -> None:
        updated_at = _utcnow_iso()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_metrics (
                        model_name, sample_size, number_accuracy, color_accuracy,
                        big_small_accuracy, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        sample_size = VALUES(sample_size),
                        number_accuracy = VALUES(number_accuracy),
                        color_accuracy = VALUES(color_accuracy),
                        big_small_accuracy = VALUES(big_small_accuracy),
                        updated_at = VALUES(updated_at)
                    """,
                    (
                        model_name,
                        sample_size,
                        number_accuracy,
                        color_accuracy,
                        big_small_accuracy,
                        updated_at,
                    ),
                )

    def get_model_metrics(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM model_metrics ORDER BY model_name ASC"
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def export_csv(self, exports_dir: Path | str | None = None) -> dict[str, Path]:
        out_dir = Path(exports_dir) if exports_dir else Path(EXPORTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "rounds": out_dir / "rounds.csv",
            "predictions": out_dir / "predictions.csv",
            "metrics": out_dir / "metrics.csv",
        }

        with self.connect() as conn:
            with conn.cursor() as cur:
                for table, path in (
                    ("rounds", paths["rounds"]),
                    ("predictions", paths["predictions"]),
                    ("model_metrics", paths["metrics"]),
                ):
                    cur.execute(f"SELECT * FROM {table}")
                    rows = cur.fetchall()
                    with path.open("w", newline="", encoding="utf-8") as fh:
                        if not rows:
                            fh.write("")
                            continue
                        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        for row in rows:
                            writer.writerow(dict(row))
        return paths
