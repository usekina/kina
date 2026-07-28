"""Configuration values for the Aoi-maintained KinaBot V1 app."""

import os
from pathlib import Path

APP_VERSION = "v1.1-multilingual-pilot"
CONSENT_VERSION = "consent-v1.1"
SCORING_MODEL_VERSION = "score-v2-multilingual"
OPENAI_INSIGHT_MODEL = os.getenv("KINABOT_INSIGHT_MODEL", "gpt-4.1-mini")
ENVIRONMENT = os.getenv("KINABOT_ENVIRONMENT", "development").strip().lower()
ALLOW_LOCAL_VERIFICATION_CODES = (
    os.getenv("KINABOT_ALLOW_LOCAL_CODES", "true").strip().lower() == "true"
    and ENVIRONMENT != "production"
)
ADMIN_KEY = os.getenv("KINABOT_ADMIN_KEY", "").strip()

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATABASE_PATH = Path(
    os.getenv("KINABOT_DATABASE_PATH", str(DATA_DIR / "kinabot_v1.sqlite3"))
)

MAX_TESTS_PER_DAY = int(os.getenv("KINABOT_MAX_TESTS_PER_DAY", "2"))
MAX_AUDIO_BYTES = int(os.getenv("KINABOT_MAX_AUDIO_MB", "25")) * 1024 * 1024
VERIFICATION_CODE_TTL_MINUTES = 10

SMTP_HOST = os.getenv("KINABOT_SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("KINABOT_SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("KINABOT_SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("KINABOT_SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("KINABOT_SMTP_FROM_EMAIL", SMTP_USERNAME).strip()
SMTP_USE_TLS = os.getenv("KINABOT_SMTP_USE_TLS", "true").strip().lower() != "false"
