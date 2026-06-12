"""Speech-to-text helpers for KinaBot V1 audio uploads."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from config import OPENAI_TRANSCRIPTION_MODEL


OPENAI_TRANSCRIPTION_TYPES = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]


def speech_to_text_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def transcribe_audio_upload(uploaded_file: BinaryIO, original_name: str) -> tuple[bool, str]:
    if not speech_to_text_configured():
        return False, "OPENAI_API_KEY is not configured."

    suffix = Path(original_name).suffix or ".audio"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(uploaded_file.getbuffer())

        from openai import OpenAI

        client = OpenAI()
        with temp_path.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIPTION_MODEL,
                file=audio_file,
                response_format="text",
            )

        return True, str(transcript).strip()
    except Exception as exc:
        return False, f"Speech-to-text failed: {exc}"
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
