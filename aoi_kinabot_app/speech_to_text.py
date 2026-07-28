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
) -> tuple[bool, str, float | None, dict]:
    """Transcribe locally and delete the temporary audio file in every outcome."""
    suffix = Path(original_name).suffix or ".audio"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(uploaded_file.getbuffer())

        segments_iter, info = _local_model().transcribe(
            str(temp_path),
            language=language_code,
            beam_size=3,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments = list(segments_iter)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            return False, "No clear speech was detected. Try a quieter recording.", None, {}
        duration = getattr(info, "duration", None)
        duration_seconds = float(duration) if duration else None
        acoustic_metrics = calculate_pause_metrics(segments, duration_seconds)
        return True, text, duration_seconds, acoustic_metrics
    except Exception as exc:
        return False, f"Local speech-to-text failed: {exc}", None, {}
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def calculate_pause_metrics(segments: list, duration_seconds: float | None) -> dict:
    """Calculate descriptive pause metrics from timestamped speech segments."""
    ordered = sorted(
        (
            (max(0.0, float(segment.start)), max(0.0, float(segment.end)))
            for segment in segments
            if float(segment.end) > float(segment.start)
        ),
        key=lambda item: item[0],
    )
    if not ordered:
        return {}

    voiced_seconds = sum(end - start for start, end in ordered)
    pauses = [
        max(0.0, ordered[index][0] - ordered[index - 1][1])
        for index in range(1, len(ordered))
    ]
    meaningful_pauses = [pause for pause in pauses if pause >= 0.25]
    total_duration = duration_seconds or ordered[-1][1]
    pause_seconds = max(0.0, total_duration - voiced_seconds)
    return {
        "voiced_seconds": round(voiced_seconds, 3),
        "pause_seconds": round(pause_seconds, 3),
        "pause_count": len(meaningful_pauses),
        "mean_pause_seconds": round(
            sum(meaningful_pauses) / len(meaningful_pauses), 3
        )
        if meaningful_pauses
        else 0.0,
        "max_pause_seconds": round(max(meaningful_pauses), 3)
        if meaningful_pauses
        else 0.0,
        "pause_ratio": round(pause_seconds / total_duration, 4)
        if total_duration > 0
        else 0.0,
    }
