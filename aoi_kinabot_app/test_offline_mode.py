from pathlib import Path

import insight_service
import speech_to_text
from offline_identity import participant_key, valid_participant_id

TEST_SECRET = "test-only-secret-that-is-at-least-32-characters"


def test_school_participant_id_001_is_valid_and_not_stored_in_key():
    assert valid_participant_id("001")
    key = participant_key("001", TEST_SECRET)
    assert key != "001"
    assert "001" not in key
    assert len(key) == 64


def test_participant_ids_are_limited_and_normalized():
    assert participant_key(" study-01 ", TEST_SECRET) == participant_key(
        "STUDY-01", TEST_SECRET
    )
    assert not valid_participant_id("01")
    assert not valid_participant_id("participant@example.org")
    assert not valid_participant_id("name with spaces")


def test_different_studies_get_different_pseudonyms():
    assert participant_key("001", TEST_SECRET) != participant_key(
        "001", "different-study-secret-that-is-32-characters"
    )


def test_offline_mode_uses_local_action_even_when_api_key_exists(monkeypatch):
    monkeypatch.setattr(insight_service, "OFFLINE_RESEARCH_MODE", True)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    history = [
        {
            "feature_name": "Expression Variety",
            "score": score,
            "created_at": f"2026-08-0{index}",
            "session_number": index,
        }
        for index, score in enumerate([60, 58, 57], start=1)
    ]

    result = insight_service.generate_wellness_insight(history, "English")

    assert result["generated_by"] == "KinaBot research action library"


def test_offline_transcription_requires_existing_local_model(tmp_path, monkeypatch):
    monkeypatch.setattr(speech_to_text, "OFFLINE_RESEARCH_MODE", True)
    monkeypatch.setattr(
        speech_to_text,
        "OFFLINE_WHISPER_MODEL_PATH",
        str(tmp_path / "missing-model"),
    )
    assert speech_to_text.speech_to_text_configured() is False

    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    monkeypatch.setattr(
        speech_to_text,
        "OFFLINE_WHISPER_MODEL_PATH",
        str(model_dir),
    )
    assert speech_to_text.speech_to_text_configured() is True


def test_missing_offline_model_fails_before_reading_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(speech_to_text, "OFFLINE_RESEARCH_MODE", True)
    monkeypatch.setattr(
        speech_to_text,
        "OFFLINE_WHISPER_MODEL_PATH",
        str(tmp_path / "missing-model"),
    )
    ok, message, duration, metrics = speech_to_text.transcribe_audio_upload(
        object(), "sample.wav", "en"
    )
    assert ok is False
    assert "Offline Whisper model" in message
    assert duration is None
    assert metrics == {}
