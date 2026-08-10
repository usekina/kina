import importlib
import io

import pytest
from fastapi.testclient import TestClient

TEST_SECRET = "test-participant-secret-that-is-at-least-32-characters"
TEST_TOKEN = "test-local-api-token-that-is-at-least-32-characters"


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("KINABOT_OFFLINE_RESEARCH_MODE", "true")
    monkeypatch.setenv("KINABOT_PARTICIPANT_KEY_SECRET", TEST_SECRET)
    monkeypatch.setenv("KINABOT_LOCAL_API_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("KINABOT_DATABASE_PATH", str(tmp_path / "research.sqlite3"))
    import config
    import database
    import offline_api.service as service
    import offline_api.api as api_module
    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(service)
    importlib.reload(api_module)
    return api_module, service, TestClient(api_module.app)


def test_data_endpoints_require_token(api):
    _, _, client = api
    assert client.get("/v1/participants/001/history").status_code == 401


def test_reflection_is_pseudonymous_versioned_and_deletable(api, monkeypatch):
    _, service, client = api

    def fake_transcription(audio_file, filename, language_code):
        assert audio_file.read() == b"not-real-audio"
        return True, "I spoke with my family today.", 8.0, {"pause_ratio": 0.1}

    monkeypatch.setattr(service, "transcribe_audio_upload", fake_transcription)
    headers = {"X-KinaBot-Token": TEST_TOKEN}
    response = client.post(
        "/v1/reflections", headers=headers,
        data={"participant_id": "001", "language": "English",
              "consent_version": "study-consent-2026-01"},
        files={"audio": ("sample.wav", io.BytesIO(b"not-real-audio"), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["interpretation"]["non_diagnostic"] is True
    assert body["retention"] == {"audio": "ephemeral",
                                  "full_transcript": "not_stored"}
    assert body["provenance"]["scoring_version"]
    assert "001" not in response.text
    assert "family" not in response.text

    history = client.get("/v1/participants/001/history", headers=headers)
    assert history.status_code == 200
    assert history.json()["session_count"] == 1
    assert "001" not in history.text

    assert client.delete("/v1/participants/001", headers=headers).status_code == 200
    empty = client.get("/v1/participants/001/history", headers=headers)
    assert empty.json()["session_count"] == 0


def test_rejects_invalid_id_and_unsupported_audio(api):
    _, _, client = api
    headers = {"X-KinaBot-Token": TEST_TOKEN}
    unsupported = client.post(
        "/v1/reflections", headers=headers,
        data={"participant_id": "001", "language": "English",
              "consent_version": "study-v1"},
        files={"audio": ("sample.exe", b"x", "application/octet-stream")},
    )
    assert unsupported.status_code == 415

    history = client.get("/v1/participants/not valid/history", headers=headers)
    assert history.status_code == 422
