"""Local verification helpers for KinaBot V1."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from config import VERIFICATION_CODE_TTL_MINUTES
from database import find_active_code, mark_code_used, save_verification_code, utc_now_iso


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_verification_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def create_local_verification_code(email: str) -> tuple[str, str]:
    normalized = normalize_email(email)
    email_hash = hash_value(normalized)
    code = make_verification_code()
    code_hash = hash_value(code)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)
    ).isoformat(timespec="seconds")
    save_verification_code(email_hash, code_hash, expires_at)
    return email_hash, code


def verify_code(email: str, code: str) -> str | None:
    normalized = normalize_email(email)
    email_hash = hash_value(normalized)
    row = find_active_code(email_hash, hash_value(code.strip()), utc_now_iso())
    if not row:
        return None
    mark_code_used(int(row["id"]))
    return email_hash
