"""Pure helpers for KinaBot's low-pressure 30-day reflection experience."""

from datetime import date


CHALLENGE_DAYS = 30


def challenge_status(session_dates: list[str], today: date) -> dict:
    """Return calendar progress without treating missed days as failure."""
    completed_dates = sorted(
        {date.fromisoformat(value) for value in session_dates if value}
    )
    if not completed_dates:
        return {
            "day": 1,
            "reflection_days": 0,
            "sessions": 0,
            "complete_today": False,
            "challenge_complete": False,
        }

    first_day = completed_dates[0]
    elapsed_day = max(1, (today - first_day).days + 1)
    return {
        "day": min(elapsed_day, CHALLENGE_DAYS),
        "reflection_days": len(completed_dates),
        "sessions": len(session_dates),
        "complete_today": today in completed_dates,
        "challenge_complete": elapsed_day > CHALLENGE_DAYS,
    }
