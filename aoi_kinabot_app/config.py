"""Configuration values for the Aoi-maintained KinaBot V1 app."""

from pathlib import Path

APP_VERSION = "v1-local-skeleton"
CONSENT_VERSION = "consent-v1"
SCORING_MODEL_VERSION = "score-v1"

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATABASE_PATH = DATA_DIR / "kinabot_v1.sqlite3"

MAX_TESTS_PER_DAY = 2
VERIFICATION_CODE_TTL_MINUTES = 10
