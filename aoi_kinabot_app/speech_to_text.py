"""Private, server-local speech-to-text helpers for KinaBot."""

from __future__ import annotations

import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from config import OFFLINE_RESEARCH_MODE, OFFLINE_WHISPER_MODEL_PATH


LOCAL_TRANSCRIPTION_TYPES = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]


def speech_to_text_configured() -> bool:
    """Local transcription is part of the app and needs no external API key."""
    if not OFFLINE_RESEARCH_MODE:
        return True
    return bool(
        OFFLINE_WHISPER_MODEL_PATH
        and Path(OFFLINE_WHISPER_MODEL_PATH).expanduser().is_dir()
    )


@lru_cache(maxsize=1)
def _local_model():
    from faster_whisper import WhisperModel

    if OFFLINE_RESEARCH_MODE:
        if not speech_to_text_configured():
            raise RuntimeError(
                "Offline Whisper model not found. Set "
                "KINABOT_OFFLINE_WHISPER_MODEL_PATH to a local model directory."
            )
        model_name = str(Path(OFFLINE_WHISPER_MODEL_PATH).expanduser().resolve())
    else:
        model_name = os.getenv("KINABOT_WHISPER_MODEL", "small")
    compute_type = os.getenv("KINABOT_WHISPER_COMPUTE_TYPE", "int8")
    return WhisperModel(model_name, device="cpu", compute_type=compute_type)


def transcribe_audio_upload(
    uploaded_file: BinaryIO,
    original_name: str,
    language_code: str,
) -> tuple[bool, str, float | None, dict]:
    """Transcribe locally and delete the temporary audio file in every outcome."""
    if not speech_to_text_configured():
        return (
            False,
            "Offline Whisper model is not configured. Ask the study administrator "
            "to set KINABOT_OFFLINE_WHISPER_MODEL_PATH.",
            None,
            {},
        )
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
    total_duration = max(0.0, float(duration_seconds or 0.0))
    intervals = []
    for segment in segments:
        try:
            start = max(0.0, float(segment.start))
            end = max(0.0, float(segment.end))
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if total_duration > 0:
            start = min(start, total_duration)
            end = min(end, total_duration)
        if end > start:
            intervals.append((start, end))

    if not intervals:
        return {}

    ordered = sorted(intervals, key=lambda item: item[0])
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    voiced_seconds = sum(end - start for start, end in merged)
    pauses = [
        merged[index][0] - merged[index - 1][1]
        for index in range(1, len(merged))
    ]
    meaningful_pauses = [pause for pause in pauses if pause >= 0.25]
    first_speech_start = merged[0][0]
    last_speech_end = merged[-1][1]
    speech_span_seconds = last_speech_end - first_speech_start
    internal_pause_seconds = sum(pauses)
    effective_duration = total_duration or last_speech_end
    leading_silence_seconds = first_speech_start
    trailing_silence_seconds = max(0.0, effective_duration - last_speech_end)
    return {
        "voiced_seconds": round(voiced_seconds, 3),
        "pause_seconds": round(internal_pause_seconds, 3),
        "internal_pause_seconds": round(internal_pause_seconds, 3),
        "speech_span_seconds": round(speech_span_seconds, 3),
        "leading_silence_seconds": round(leading_silence_seconds, 3),
        "trailing_silence_seconds": round(trailing_silence_seconds, 3),
        "pause_count": len(meaningful_pauses),
        "mean_pause_seconds": round(
            sum(meaningful_pauses) / len(meaningful_pauses), 3
        )
        if meaningful_pauses
        else 0.0,
        "max_pause_seconds": round(max(meaningful_pauses), 3)
        if meaningful_pauses
        else 0.0,
        "pause_ratio": round(internal_pause_seconds / speech_span_seconds, 4)
        if speech_span_seconds > 0
        else 0.0,
    }
