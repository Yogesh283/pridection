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
    ) -> bool:
        """Insert a round if the period is new. Returns True if inserted."""
        created_at = _utcnow_iso()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT IGNORE INTO rounds
                        (period, number, color, timestamp, raw_json, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        period,
                        number,
                        color,
                        timestamp or created_at,
                        raw_json,
                        created_at,
                    ),
                )
                inserted = cur.rowcount > 0
                if inserted:
                    logger.info(
                        "Stored round period=%s number=%s color=%s",
                        period,
                        number,
                        color,
                    )
                return inserted

    def get_rounds(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is None:
            query = "SELECT * FROM rounds ORDER BY period ASC"
            params: tuple[Any, ...] = ()
        else:
            query = (
                "SELECT * FROM ("
                "SELECT * FROM rounds ORDER BY period DESC LIMIT %s"
                ") AS recent ORDER BY period ASC"
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
                    "SELECT * FROM rounds ORDER BY period DESC LIMIT 1"
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
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 AS ok FROM rounds WHERE period = %s LIMIT 1",
                    (period,),
                )
                row = cur.fetchone()
        return row is not None

    def save_prediction(self, prediction: dict[str, Any], update_existing: bool = True) -> int:
        created_at = prediction.get("created_at") or _utcnow_iso()
        predicted_number = prediction.get("predicted_number")
        predicted_big_small = prediction.get("predicted_big_small")
        if predicted_big_small is None and predicted_number is not None:
            predicted_big_small = big_small_label(int(predicted_number))

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
                        cur.execute("DELETE FROM predictions WHERE id = %s", (int(row["id"]),))
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
                                created_at = %s
                            WHERE id = %s
                            """,
                            (
                                predicted_number,
                                prediction.get("predicted_color"),
                                predicted_big_small,
                                prediction.get("number_probability"),
                                prediction.get("color_probability"),
                                prediction.get("big_small_probability"),
                                prediction.get("model_name", "ensemble"),
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
                        model_name, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        prediction["target_period"],
                        predicted_number,
                        prediction.get("predicted_color"),
                        predicted_big_small,
                        prediction.get("number_probability"),
                        prediction.get("color_probability"),
                        prediction.get("big_small_probability"),
                        prediction.get("model_name", "ensemble"),
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
                    ORDER BY created_at ASC
                    """
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def resolve_prediction(
        self,
        prediction_id: int,
        actual_number: int,
        actual_color: str | None,
    ) -> None:
        actual_bs = big_small_label(int(actual_number))
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM predictions WHERE id = %s",
                    (prediction_id,),
                )
                row = cur.fetchone()
                if not row:
                    return
                number_correct = (
                    1 if row["predicted_number"] == actual_number else 0
                )
                pred_color = (row["predicted_color"] or "").upper()
                act_color = (actual_color or "").upper()
                color_correct = 0
                if pred_color and act_color:
                    pred_parts = set(pred_color.replace(" ", "").split(","))
                    act_parts = set(act_color.replace(" ", "").split(","))
                    if pred_parts & act_parts:
                        color_correct = 1

                pred_bs = (row.get("predicted_big_small") or "").upper()
                if not pred_bs and row.get("predicted_number") is not None:
                    pred_bs = big_small_label(int(row["predicted_number"]))
                big_small_correct = 1 if pred_bs == actual_bs else 0

                cur.execute(
                    """
                    UPDATE predictions
                    SET actual_number = %s,
                        actual_color = %s,
                        actual_big_small = %s,
                        number_correct = %s,
                        color_correct = %s,
                        big_small_correct = %s,
                        resolved_at = %s
                    WHERE id = %s
                    """,
                    (
                        actual_number,
                        actual_color,
                        actual_bs,
                        number_correct,
                        color_correct,
                        big_small_correct,
                        _utcnow_iso(),
                        prediction_id,
                    ),
                )

    def get_resolved_predictions(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is None:
            query = (
                "SELECT * FROM predictions WHERE resolved_at IS NOT NULL "
                "ORDER BY resolved_at ASC"
            )
            params: tuple[Any, ...] = ()
        else:
            query = (
                "SELECT * FROM ("
                "SELECT * FROM predictions WHERE resolved_at IS NOT NULL "
                "ORDER BY resolved_at DESC LIMIT %s"
                ") AS recent ORDER BY resolved_at ASC"
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
