"""Pure presentation helpers for KinaBot's longitudinal mobile views."""

from __future__ import annotations

from html import escape

import pandas as pd

from scoring import display_feature_name


COMPARISON_KEY_COLUMNS = ("language", "scoring_model_version", "analysis_pipeline_id")


def comparison_key(row: dict) -> tuple[str, ...]:
    """Return the provenance key that makes two sessions comparable."""
    # Legacy sessions predate analysis_pipeline_id; app_version is the safest
    # compatibility discriminator available for those records.
    pipeline_id = row.get("analysis_pipeline_id") or row.get("app_version") or "unknown"
    return (str(row.get("language") or "unknown"),
            str(row.get("scoring_model_version") or "unknown"),
            str(pipeline_id))


def select_latest_comparable_history(
    history: pd.DataFrame, minimum_sessions: int = 3
) -> tuple[pd.DataFrame, tuple[str, ...] | None]:
    """Select the newest compatible group without dropping historical records."""
    if history.empty:
        return history.copy(), None
    working = history.copy()
    working["_comparison_key"] = working.apply(
        lambda row: comparison_key(row.to_dict()), axis=1
    )
    counts = (
        working[["_comparison_key", "session_id"]]
        .drop_duplicates()
        .groupby("_comparison_key")["session_id"]
        .nunique()
    )
    eligible = set(counts[counts >= minimum_sessions].index)
    if not eligible:
        return working.drop(columns=["_comparison_key"]), None
    latest = (
        working[working["_comparison_key"].isin(eligible)]
        .groupby("_comparison_key")["session_id"]
        .max()
        .sort_values()
    )
    selected = latest.index[-1]
    result = working[working["_comparison_key"] == selected].drop(
        columns=["_comparison_key"]
    )
    return result, selected


def metric_grid_html(score_items: list[dict], language: str) -> str:
    """Return a compact two-column score grid suitable for narrow screens."""
    tiles = []
    for item in score_items:
        if item.get("score") is None:
            continue
        score = int(round(float(item["score"])))
        label = escape(display_feature_name(str(item["feature_name"]), language))
        tiles.append(
            f'<div class="metric-tile">'
            f'<div class="metric-tile__top">'
            f'<span class="metric-tile__name">{label}</span>'
            f'<span class="metric-tile__value">{score}</span>'
            f'</div><div class="metric-tile__track">'
            f'<div class="metric-tile__fill" style="width:{score}%"></div>'
            f'</div></div>'
        )
    return f'<div class="metric-grid">{"".join(tiles)}</div>'


def latest_session_scores(history: pd.DataFrame) -> list[dict]:
    """Return score rows belonging to the most recent stored session."""
    if history.empty:
        return []
    if "session_id" in history.columns:
        latest_session_id = history["session_id"].max()
        latest = history[history["session_id"] == latest_session_id]
        return latest.to_dict("records")
    latest_created_at = history["created_at"].max()
    latest = history[history["created_at"] == latest_created_at]
    return latest.to_dict("records")
