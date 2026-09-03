"""Private research orchestration built on KinaBot's authoritative core."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import BinaryIO

from config import (APP_VERSION, MAX_TESTS_PER_DAY, PARTICIPANT_KEY_SECRET,
                    SCORING_MODEL_VERSION)
from database import (
    complete_test_session, count_tests_today, delete_user_research_data,
    find_user_id_by_email_hash, get_user_scores, init_db, record_consent,
    save_feature_scores, upsert_user,
)
from language_analysis import LANGUAGE_CODES, analyze_transcript
from offline_identity import participant_key
from speech_to_text import transcribe_audio_upload

NON_DIAGNOSTIC_BOUNDARY = (
    "Descriptive features for this recording only. They do not indicate health, "
    "ability, improvement, decline, diagnosis, cause, or risk."
)


class ResearchServiceError(ValueError):
    """Expected input or processing failure safe to return to a local client."""


def _participant_context(participant_id: str) -> tuple[str, int | None]:
    if len(PARTICIPANT_KEY_SECRET) < 32:
        raise ResearchServiceError(
            "KINABOT_PARTICIPANT_KEY_SECRET must contain at least 32 characters."
        )
    try:
        key = participant_key(participant_id, PARTICIPANT_KEY_SECRET)
    except ValueError as exc:
        raise ResearchServiceError(str(exc)) from exc
    return key, find_user_id_by_email_hash(key)


def analyze_reflection(*, participant_id: str, language: str,
                       consent_version: str, audio_file: BinaryIO,
                       filename: str,
                       session_type: str = "research-reflection",
                       idempotency_key: str | None = None) -> dict:
    """Transcribe and score while retaining only pseudonymous derived data."""
    if language not in LANGUAGE_CODES:
        raise ResearchServiceError("Language must be English, 日本語, or 中文.")
    if not consent_version.strip() or len(consent_version) > 100:
        raise ResearchServiceError("A valid consent_version is required.")
    init_db()
    key, user_id = _participant_context(participant_id)
    today = datetime.now(timezone.utc).date().isoformat()
    if user_id is not None and count_tests_today(user_id, today) >= MAX_TESTS_PER_DAY:
        raise ResearchServiceError("Daily reflection limit reached.")
    ok, transcript_or_error, duration, acoustic_metrics = transcribe_audio_upload(
        audio_file, filename, LANGUAGE_CODES[language]
    )
    if not ok:
        raise ResearchServiceError(transcript_or_error)
    scores, summary = analyze_transcript(
        transcript_or_error, language, duration, acoustic_metrics
    )
    user_id = user_id or upsert_user(key, email=None)
    record_consent(user_id, consent_version.strip())
    session_id, session_number, _ = complete_test_session(
        user_id=user_id, session_date=today, app_version=APP_VERSION,
        consent_version=consent_version.strip(),
        scoring_model_version=SCORING_MODEL_VERSION, scores=scores,
        max_tests_per_day=MAX_TESTS_PER_DAY,
        session_type=session_type.strip()[:100] or "research-reflection",
        language=language, duration_seconds=duration, timezone_name="UTC",
        idempotency_key=idempotency_key,
    )
    return {
        "analysis_id": f"session-{session_id}", "session_number": session_number,
        "language": language, "duration_seconds": duration, "summary": summary,
        "features": scores,
        "provenance": {"app_version": APP_VERSION,
                       "scoring_version": SCORING_MODEL_VERSION,
                       "transcription": "local-faster-whisper"},
        "interpretation": {"type": "longitudinal_reflection",
                           "comparison_scope": "self_only",
                           "non_diagnostic": True,
                           "boundary": NON_DIAGNOSTIC_BOUNDARY},
        "retention": {"audio": "ephemeral", "full_transcript": "not_stored"},
    }


def participant_history(participant_id: str) -> dict:
    """Return derived observations without returning the raw participant ID."""
    _, user_id = _participant_context(participant_id)
    rows = [] if user_id is None else [dict(row) for row in get_user_scores(user_id)]
    return {"sessions": rows,
            "session_count": len({row["session_id"] for row in rows}),
            "comparison_scope": "self_only", "non_diagnostic": True}


def erase_participant(participant_id: str) -> bool:
    """Erase one participant's pseudonymous local research record."""
    _, user_id = _participant_context(participant_id)
    return False if user_id is None else delete_user_research_data(user_id)
