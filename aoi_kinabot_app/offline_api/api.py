"""Local-only HTTP API for approved offline KinaBot research deployments."""

from __future__ import annotations

import io
import secrets

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from config import (APP_VERSION, LOCAL_API_TOKEN, MAX_AUDIO_BYTES,
                    OFFLINE_RESEARCH_MODE, SCORING_MODEL_VERSION)
from offline_api.service import (ResearchServiceError, analyze_reflection,
                                 erase_participant, participant_history)
from speech_to_text import LOCAL_TRANSCRIPTION_TYPES, speech_to_text_configured

app = FastAPI(
    title="KinaBot Offline Research API", version="1.0.0",
    description=("Local, non-diagnostic speech-reflection API for approved "
                 "research. Raw audio and full transcripts are not retained."),
    docs_url="/docs", redoc_url=None,
)


def require_local_token(x_kinabot_token: str = Header(default="")) -> None:
    if len(LOCAL_API_TOKEN) < 32:
        raise HTTPException(status_code=503,
                            detail="Local API token is not configured.")
    if not secrets.compare_digest(x_kinabot_token, LOCAL_API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid local API token.")


def require_offline_mode() -> None:
    if not OFFLINE_RESEARCH_MODE:
        raise HTTPException(
            status_code=503,
            detail="KINABOT_OFFLINE_RESEARCH_MODE=true is required.")


protected = [Depends(require_local_token), Depends(require_offline_mode)]


@app.get("/health")
def health() -> dict:
    ready = OFFLINE_RESEARCH_MODE and speech_to_text_configured()
    return {"status": "ready" if ready else "not_ready",
            "offline_mode": OFFLINE_RESEARCH_MODE,
            "local_transcription_ready": speech_to_text_configured(),
            "app_version": APP_VERSION,
            "scoring_version": SCORING_MODEL_VERSION,
            "non_diagnostic": True}


@app.post("/v1/reflections", dependencies=protected)
async def create_reflection(
    participant_id: str = Form(...), language: str = Form(...),
    consent_version: str = Form(...),
    session_type: str = Form("research-reflection"),
    audio: UploadFile = File(...),
) -> dict:
    extension = (audio.filename or "").rsplit(".", 1)[-1].lower()
    if extension not in LOCAL_TRANSCRIPTION_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type.")
    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413,
                            detail="Audio exceeds the configured size limit.")
    try:
        return analyze_reflection(
            participant_id=participant_id, language=language,
            consent_version=consent_version, audio_file=io.BytesIO(content),
            filename=audio.filename or "recording.audio",
            session_type=session_type)
    except ResearchServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/participants/{participant_id}/history", dependencies=protected)
def get_history(participant_id: str) -> dict:
    try:
        return participant_history(participant_id)
    except ResearchServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/v1/participants/{participant_id}", dependencies=protected)
def delete_participant(participant_id: str) -> dict:
    try:
        deleted = erase_participant(participant_id)
    except ResearchServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404,
                            detail="Participant record not found.")
    return {"deleted": True}
