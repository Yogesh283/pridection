"""Configuration for the Wingo 30 statistical analyzer."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

WINGO_API_URL = os.getenv(
    "WINGO_API_URL", "https://dearapi.tashanwin.fit/wingo30"
).strip()
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "3"))
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "200"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "30"))

# XAMPP MySQL / MariaDB settings
DB_HOST = os.getenv("DB_HOST", "127.0.0.1").strip()
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "wingo30").strip()
DB_CHARSET = os.getenv("DB_CHARSET", "utf8mb4").strip()

LOG_PATH = Path(os.getenv("LOG_PATH", str(ROOT_DIR / "logs" / "app.log")))
if not LOG_PATH.is_absolute():
    LOG_PATH = ROOT_DIR / LOG_PATH

PREDICTION_JSON_PATH = Path(
    os.getenv("PREDICTION_JSON_PATH", str(ROOT_DIR / "prediction.json"))
)
if not PREDICTION_JSON_PATH.is_absolute():
    PREDICTION_JSON_PATH = ROOT_DIR / PREDICTION_JSON_PATH

EXPORTS_DIR = Path(os.getenv("EXPORTS_DIR", str(ROOT_DIR / "exports")))
if not EXPORTS_DIR.is_absolute():
    EXPORTS_DIR = ROOT_DIR / EXPORTS_DIR

# Color mapping is configurable. Prefer API-provided settled colours when present.
# Observed API values: "red", "green", "red,violet", "green,violet"
# Typical Wingo association (used only when API colour is missing):
COLOR_MAP = {
    0: "RED,VIOLET",
    1: "GREEN",
    2: "RED",
    3: "GREEN",
    4: "RED",
    5: "GREEN,VIOLET",
    6: "RED",
    7: "GREEN",
    8: "RED",
    9: "GREEN",
}

COLOR_ALIASES = {
    "red": "RED",
    "green": "GREEN",
    "violet": "VIOLET",
    "purple": "VIOLET",
}

PRIMARY_COLORS = ("RED", "GREEN", "VIOLET")

FEATURE_WINDOWS = (10, 20, 50, 100, 200)

ENSEMBLE_WEIGHTS = {
    "majority": 1.5,
    "frequency": 0.6,
    "markov": 0.9,
    "recent": 1.0,
    "pattern": 0.4,
    "random": 0.0,
    "ml": 2.0,  # dedicated Big/Small classifiers
}

MIN_SAMPLE_FOR_ACCURACY_CLAIM = 20
SCHEMA_REQUIRED_HINTS = ("period", "number")

# Primary market focus for live predictions / confidence.
# Options: "big_small", "color", "number"
PREDICTION_FOCUS = os.getenv("PREDICTION_FOCUS", "big_small").strip().lower()
