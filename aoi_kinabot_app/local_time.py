"""User-local calendar helpers while retaining UTC event timestamps."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def safe_zone(timezone_name: str | None):
    try:
        return ZoneInfo(timezone_name or "UTC")
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        return timezone.utc


def local_date_iso(
    timezone_name: str | None,
    now_utc: datetime | None = None,
) -> str:
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(safe_zone(timezone_name)).date().isoformat()


def utc_iso_to_local_date(created_at: str, timezone_name: str | None) -> str:
    timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(safe_zone(timezone_name)).date().isoformat()
