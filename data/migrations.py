"""MySQL (XAMPP) schema migrations."""

from __future__ import annotations

from typing import Any


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS rounds (
        id INT NOT NULL AUTO_INCREMENT,
        period VARCHAR(64) NOT NULL,
        number INT NOT NULL,
        color VARCHAR(64) NULL,
        timestamp VARCHAR(64) NULL,
        raw_json LONGTEXT NULL,
        created_at VARCHAR(64) NULL,
        PRIMARY KEY (id),
        UNIQUE KEY uq_rounds_period (period),
        KEY idx_rounds_period (period)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INT NOT NULL AUTO_INCREMENT,
        target_period VARCHAR(64) NOT NULL,
        predicted_number INT NULL,
        predicted_color VARCHAR(64) NULL,
        predicted_big_small VARCHAR(16) NULL,
        number_probability DOUBLE NULL,
        color_probability DOUBLE NULL,
        big_small_probability DOUBLE NULL,
        model_name VARCHAR(64) NULL,
        created_at VARCHAR(64) NULL,
        actual_number INT NULL,
        actual_color VARCHAR(64) NULL,
        actual_big_small VARCHAR(16) NULL,
        number_correct TINYINT NULL,
        color_correct TINYINT NULL,
        big_small_correct TINYINT NULL,
        resolved_at VARCHAR(64) NULL,
        PRIMARY KEY (id),
        KEY idx_predictions_target (target_period),
        KEY idx_predictions_resolved (resolved_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INT NOT NULL AUTO_INCREMENT,
        model_name VARCHAR(64) NOT NULL,
        sample_size INT NULL,
        number_accuracy DOUBLE NULL,
        color_accuracy DOUBLE NULL,
        big_small_accuracy DOUBLE NULL,
        updated_at VARCHAR(64) NULL,
        PRIMARY KEY (id),
        UNIQUE KEY uq_model_metrics_name (model_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

# Safe upgrades for databases created before Big/Small columns existed.
ALTER_COLUMNS = [
    ("predictions", "predicted_big_small", "VARCHAR(16) NULL"),
    ("predictions", "big_small_probability", "DOUBLE NULL"),
    ("predictions", "actual_big_small", "VARCHAR(16) NULL"),
    ("predictions", "big_small_correct", "TINYINT NULL"),
    ("model_metrics", "big_small_accuracy", "DOUBLE NULL"),
    # Integrity / naming (additive only).
    ("rounds", "source", "VARCHAR(32) NULL"),
    ("rounds", "color_raw", "VARCHAR(64) NULL"),
    ("predictions", "decision_strategy", "VARCHAR(64) NULL"),
    ("predictions", "status", "VARCHAR(32) NULL"),
    ("predictions", "model_version", "VARCHAR(128) NULL"),
    ("predictions", "trained_until_period", "VARCHAR(64) NULL"),
    ("predictions", "trained_rows", "INT NULL"),
    ("predictions", "probability_big", "DOUBLE NULL"),
    ("predictions", "probability_small", "DOUBLE NULL"),
    ("predictions", "decision_threshold", "DOUBLE NULL"),
    ("predictions", "locked_at", "VARCHAR(64) NULL"),
]


def _column_exists(cur: Any, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) AS c
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (table, column),
    )
    row = cur.fetchone()
    return int(row["c"] if isinstance(row, dict) else row[0]) > 0


def _index_exists(cur: Any, table: str, index_name: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) AS c
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND INDEX_NAME = %s
        """,
        (table, index_name),
    )
    row = cur.fetchone()
    return int(row["c"] if isinstance(row, dict) else row[0]) > 0


def apply_migrations(conn: Any) -> None:
    with conn.cursor() as cur:
        for statement in SCHEMA_STATEMENTS:
            cur.execute(statement)
        for table, column, coltype in ALTER_COLUMNS:
            if not _column_exists(cur, table, column):
                cur.execute(
                    f"ALTER TABLE `{table}` ADD COLUMN `{column}` {coltype}"
                )
        # Best-effort unique identity for round-lock (target + model + version).
        # Duplicates are collapsed first where possible.
        if not _index_exists(cur, "predictions", "uq_predictions_period_model_ver"):
            try:
                cur.execute(
                    """
                    ALTER TABLE predictions
                    ADD UNIQUE KEY uq_predictions_period_model_ver
                    (target_period, model_name, model_version)
                    """
                )
            except Exception:
                # Existing duplicate keys — application-level lock still enforces immutability.
                pass
