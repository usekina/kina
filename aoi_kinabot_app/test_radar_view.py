import math

from radar_view import FEATURES, measured_score, radar_comparison


def records():
    return [dict(session_id=sid, feature_name=name, score=sid * 10,
                 language="English", scoring_model_version="v4", analysis_pipeline_id="p1")
            for sid in range(1, 5) for name in FEATURES]


def test_reference_excludes_current_and_future_sessions():
    data = radar_comparison(records())
    assert data["current"] == [40] * 8
    assert data["reference"] == [20] * 8
    assert data["reference_ids"] == [1, 2, 3]
    assert radar_comparison(records(), 3)["reference"] == [None] * 8


def test_does_not_mix_languages_versions_or_pipelines():
    for field in ("language", "scoring_model_version", "analysis_pipeline_id"):
        rows = records()
        for row in rows:
            if row["session_id"] == 2:
                row[field] = "different"
        data = radar_comparison(rows)
        assert data["reference_ids"] == [1, 3]
        assert data["reference"] == [None] * 8


def test_missing_feature_has_no_zero_or_partial_reference():
    rows = records()
    for row in rows:
        if row["feature_name"] == "Pause Pattern" and row["session_id"] in (2, 4):
            row["score"] = None
    data = radar_comparison(rows)
    assert data["current"][4] is None
    assert data["reference"][4] is None
    assert data["reference"][0] == 20


def test_emotion_defaults_and_invalid_scores_are_not_observations():
    assert measured_score(dict(feature_name="Emotional Tone", score=50, raw_metric="neutral_score_used=true")) is None
    assert measured_score(dict(feature_name="Emotional Tone", score=50)) is None
    assert measured_score(dict(feature_name="Emotional Tone", score=50, raw_metric="positive_words=1; negative_words=1")) == 50
    for value in (math.nan, math.inf, -1, 101, None, "bad"):
        assert measured_score(dict(score=value)) is None
    assert measured_score(dict(score=0)) == 0


def test_unknown_provenance_never_establishes_reference():
    rows = records()
    for row in rows:
        row.pop("analysis_pipeline_id")
    assert radar_comparison(rows)["reference_count"] == 0
    for row in rows:
        row["app_version"] = "legacy-1"
    assert radar_comparison(rows)["reference_count"] == 3


def test_empty_and_missing_session_are_safe():
    assert radar_comparison([]) == {}
    assert radar_comparison(records(), 999) == {}
