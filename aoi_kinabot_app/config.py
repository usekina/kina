"""Configuration values for the Aoi-maintained KinaBot V1 app."""

import os
from pathlib import Path

APP_VERSION = "v1.2-offline-research"
CONSENT_VERSION = "research-pilot-consent-v2.0"
SCORING_MODEL_VERSION = "score-v4-internal-pause-span"
OPENAI_INSIGHT_MODEL = os.getenv("KINABOT_INSIGHT_MODEL", "gpt-5.6-luna")
ENVIRONMENT = os.getenv("KINABOT_ENVIRONMENT", "development").strip().lower()
OFFLINE_RESEARCH_MODE = (
    os.getenv("KINABOT_OFFLINE_RESEARCH_MODE", "false").strip().lower() == "true"
)
OFFLINE_WHISPER_MODEL_PATH = os.getenv(
    "KINABOT_OFFLINE_WHISPER_MODEL_PATH", ""
).strip()
PARTICIPANT_KEY_SECRET = os.getenv("KINABOT_PARTICIPANT_KEY_SECRET", "").strip()
ALLOW_LOCAL_VERIFICATION_CODES = (
    os.getenv("KINABOT_ALLOW_LOCAL_CODES", "true").strip().lower() == "true"
    and ENVIRONMENT != "production"
    and not OFFLINE_RESEARCH_MODE
)
ADMIN_KEY = os.getenv("KINABOT_ADMIN_KEY", "").strip()
LOCAL_API_TOKEN = os.getenv("KINABOT_LOCAL_API_TOKEN", "").strip()

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATABASE_PATH = Path(
    os.getenv("KINABOT_DATABASE_PATH", str(DATA_DIR / "kinabot_v1.sqlite3"))
)

MAX_TESTS_PER_DAY = int(os.getenv("KINABOT_MAX_TESTS_PER_DAY", "3"))
MAX_AUDIO_BYTES = int(os.getenv("KINABOT_MAX_AUDIO_MB", "25")) * 1024 * 1024
VERIFICATION_CODE_TTL_MINUTES = 10

SMTP_HOST = os.getenv("KINABOT_SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("KINABOT_SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("KINABOT_SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("KINABOT_SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("KINABOT_SMTP_FROM_EMAIL", SMTP_USERNAME).strip()
SMTP_USE_TLS = os.getenv("KINABOT_SMTP_USE_TLS", "true").strip().lower() != "false"
