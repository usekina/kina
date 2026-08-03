"""Pure presentation helpers for KinaBot's longitudinal mobile views."""

from __future__ import annotations

from html import escape

import pandas as pd

from scoring import display_feature_name


def metric_grid_html(score_items: list[dict], language: str) -> str:
    """Return a compact two-column score grid suitable for narrow screens."""
    tiles = []
    for item in score_items:
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
