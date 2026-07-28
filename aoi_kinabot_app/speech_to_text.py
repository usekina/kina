"""Private, server-local speech-to-text helpers for KinaBot."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO


LOCAL_TRANSCRIPTION_TYPES = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]


def speech_to_text_configured() -> bool:
    """Local transcription is part of the app and needs no external API key."""
    return True


@lru_cache(maxsize=1)
def _local_model():
    from faster_whisper import WhisperModel

    model_name = os.getenv("KINABOT_WHISPER_MODEL", "small")
    compute_type = os.getenv("KINABOT_WHISPER_COMPUTE_TYPE", "int8")
    return WhisperModel(model_name, device="cpu", compute_type=compute_type)


def transcribe_audio_upload(
    uploaded_file: BinaryIO,
    original_name: str,
    language_code: str,
) -> tuple[bool, str, float | None]:
    """Transcribe locally and delete the temporary audio file in every outcome."""
    suffix = Path(original_name).suffix or ".audio"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(uploaded_file.getbuffer())

        segments, info = _local_model().transcribe(
            str(temp_path),
            language=language_code,
            beam_size=3,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            return False, "No clear speech was detected. Try a quieter recording.", None
        duration = getattr(info, "duration", None)
        return True, text, float(duration) if duration else None
    except Exception as exc:
        return False, f"Local speech-to-text failed: {exc}", None
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
