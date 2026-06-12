"""Configuration values for the Aoi-maintained KinaBot V1 app."""

import os
from pathlib import Path

APP_VERSION = "v1-local-skeleton"
CONSENT_VERSION = "consent-v1"
SCORING_MODEL_VERSION = "score-v1"
OPENAI_TRANSCRIPTION_MODEL = os.getenv("KINABOT_TRANSCRIPTION_MODEL", "gpt-4o-transcribe")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATABASE_PATH = DATA_DIR / "kinabot_v1.sqlite3"

MAX_TESTS_PER_DAY = 2
VERIFICATION_CODE_TTL_MINUTES = 10

SMTP_HOST = os.getenv("KINABOT_SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("KINABOT_SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("KINABOT_SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("KINABOT_SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("KINABOT_SMTP_FROM_EMAIL", SMTP_USERNAME).strip()
SMTP_USE_TLS = os.getenv("KINABOT_SMTP_USE_TLS", "true").strip().lower() != "false"
