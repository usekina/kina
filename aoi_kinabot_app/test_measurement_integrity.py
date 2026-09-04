from __future__ import annotations

import pandas as pd

from history_view import select_latest_comparable_history
from scoring import calculate_feature_scores


def test_unavailable_audio_features_are_explicit_not_neutral_scores():
    scores = calculate_feature_scores("A short sample.", duration_seconds=None)
    by_name = {item["feature_name"]: item for item in scores}
    assert by_name["Speech Pace"]["score"] is None
    assert by_name["Speech Pace"]["availability_status"] == "unavailable"
    assert by_name["Speech Pace"]["failure_reason"] == "duration_unknown"
    assert by_name["Pause Pattern"]["score"] is None


def test_trends_require_pipeline_provenance():
    history = pd.DataFrame([
        {"session_id": 1, "language": "English", "scoring_model_version": "s", "analysis_pipeline_id": "p1"},
        {"session_id": 2, "language": "English", "scoring_model_version": "s", "analysis_pipeline_id": "p1"},
        {"session_id": 3, "language": "English", "scoring_model_version": "s", "analysis_pipeline_id": "p2"},
    ])
    comparable, key = select_latest_comparable_history(history, minimum_sessions=2)
    assert key == ("English", "s", "p1")
    assert set(comparable["session_id"]) == {1, 2}
