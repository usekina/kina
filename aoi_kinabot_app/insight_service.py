"""Local compatibility helpers for evidence-bounded wellness information."""

from __future__ import annotations

from typing import Any

from config import OFFLINE_RESEARCH_MODE  # Kept for legacy offline callers/tests

def anonymous_trend_payload(history: list[dict[str, Any]], language: str) -> dict:
    """Legacy summary helper; the new wellness UI does not transmit this data."""
    feature_series: dict[str, list[float]] = {}
    for row in history:
        if row.get("score") is None or row.get("availability_status") == "unavailable":
            continue
        name = str(row["feature_name"])
        feature_series.setdefault(name, []).append(round(float(row["score"]), 1))
    return {
        "language": language,
        "sessions_compared": len(
            {(str(row.get("created_at", "")), int(row.get("session_number", 0))) for row in history}
        ),
        "feature_scores": feature_series,
    }


def generate_wellness_insight(history: list[dict[str, Any]], language: str) -> dict:
    """Compatibility helper: a neutral example, never selected from scores.

    The UI now lets users choose from the whole curated menu. No API request is
    made and no score series is used to select a lifestyle intervention.
    """
    from wellness_guidance import wellness_suggestions
    menu = wellness_suggestions(language, [])
    idea = menu["suggestions"][0]
    return {
        "action": idea["text"], "why": menu["intro"],
        "research_summary": idea["evidence"], "source": idea["source"],
        "boundary": menu["boundary"],
        "generated_by": "KinaBot research action library",
    }
