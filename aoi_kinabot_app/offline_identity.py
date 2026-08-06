"""Pseudonymous identity helpers for offline research use."""

from __future__ import annotations

import hashlib
import hmac
import re


PARTICIPANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,31}$")


def normalize_participant_id(value: str) -> str:
    return value.strip().upper()


def valid_participant_id(value: str) -> bool:
    return bool(PARTICIPANT_ID_PATTERN.fullmatch(normalize_participant_id(value)))


def participant_key(value: str, secret: str) -> str:
    """Return a keyed pseudonym without retaining the participant ID."""
    normalized = normalize_participant_id(value)
    if not valid_participant_id(normalized):
        raise ValueError("Participant ID must be 3-32 letters, numbers, _ or -.")
    if len(secret) < 32:
        raise ValueError("Participant-key secret must contain at least 32 characters.")
    return hmac.new(
        secret.encode("utf-8"),
        f"kinabot-offline:{normalized}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
