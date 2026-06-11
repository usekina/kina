"""Temporary audio handling for KinaBot V1 uploads."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import BinaryIO


SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "aac", "ogg", "flac"]


def _wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return round(frame_count / float(frame_rate), 2)
    except wave.Error:
        return None


def accept_audio_upload(uploaded_file: BinaryIO, original_name: str) -> dict:
    suffix = Path(original_name).suffix or ".audio"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(uploaded_file.getbuffer())

        return {
            "filename": original_name,
            "bytes": temp_path.stat().st_size,
            "duration_seconds": _wav_duration_seconds(temp_path),
            "temporary_file_deleted": False,
        }
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
