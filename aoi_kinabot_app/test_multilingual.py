from pathlib import Path
from datetime import datetime, timezone

import database
import pandas as pd
from language_analysis import LANGUAGE_CODES, analyze_transcript
from local_time import local_date_iso
from reflection_profile import build_reflection_profile
from insight_service import anonymous_trend_payload, generate_wellness_insight
from history_view import latest_session_scores, metric_grid_html
from scoring import calculate_feature_scores, display_feature_name, tokenize
from speech_to_text import calculate_pause_metrics
from wellness_guidance import wellness_suggestions


def test_supported_language_codes():
    assert LANGUAGE_CODES == {"English": "en", "日本語": "ja", "中文": "zh"}


def test_tokenization_supports_all_three_languages():
    assert tokenize("A short English reflection.", "English")
    assert tokenize("今日は家族と散歩しました。", "日本語")
    assert tokenize("今天我和家人一起散步。", "中文")


def test_local_nlp_scores_multilingual():
    for language, text in [
        ("English", "Today I spoke with my family and took a comfortable walk."),
        ("日本語", "今日は家族と話してから、ゆっくり散歩しました。"),
        ("中文", "今天我和家人聊天，然后慢慢散步。"),
    ]:
        scores, summary = analyze_transcript(text, language, 30)
        assert len(scores) == 8
        assert all(0 <= item["score"] <= 100 for item in scores)
        assert summary


def test_first_session_snapshot_is_local_and_language_matched():
    scores = [
        {"feature_name": name, "score": 70}
        for name in [
            "Vocabulary Variety",
            "Response Length",
            "Sentence Complexity",
            "Speech Pace",
            "Pause Pattern",
            "Repetition Pattern",
            "Emotional Tone",
            "Transcription Clarity",
        ]
    ]
    for language in ["English", "日本語", "中文"]:
        snapshot = build_reflection_profile(scores, language)
        assert len(snapshot["dimensions"]) == 4
        assert all(item["score"] == 70 for item in snapshot["dimensions"])
        assert snapshot["takeaway"]
        assert snapshot["action"]


def test_wellness_menu_is_not_selected_from_scores():
    low = wellness_suggestions("English", [{"feature_name": "Expression Variety", "score": 5}])
    high = wellness_suggestions("English", [{"feature_name": "Expression Variety", "score": 95}])
    assert low["suggestions"] == high["suggestions"]
    assert len(low["suggestions"]) == 3


def test_habit_checkins_are_upserted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "kina.sqlite3")
    database.init_db()
    user_id = database.upsert_user("test-user")
    database.save_habit_checkins(
        user_id,
        "2026-07-28",
        {"social_connection": True, "physical_activity": False},
    )
    database.save_habit_checkins(
        user_id,
        "2026-07-28",
        {"social_connection": False, "physical_activity": True},
    )
    rows = database.get_user_habit_checkins(user_id)
    values = {row["habit_name"]: row["completed"] for row in rows}
    assert values == {"physical_activity": 1, "social_connection": 0}


def test_saved_profile_is_restored_for_returning_user(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "kina.sqlite3")
    database.init_db()
    user_id = database.upsert_user("returning-user", "person@example.com")
    database.update_user_profile(user_id, "Kina", "45-59", "Woman", "Chinese", "US")

    same_user_id = database.upsert_user("returning-user", "person@example.com")
    profile = database.get_user_profile(same_user_id)

    assert same_user_id == user_id
    assert profile["display_name"] == "Kina"
    assert profile["age_range"] == "45-59"
    assert profile["gender"] == "Woman"
    assert profile["primary_language"] == "Chinese"
    assert profile["country_region"] == "US"


def test_calculate_feature_scores_accepts_language():
    scores = calculate_feature_scores("今日は友人と話しました。", 20, "日本語")
    assert len(scores) == 8


def test_insight_payload_contains_scores_only(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    history = [
        {
            "created_at": f"2026-07-0{index}",
            "session_number": 1,
            "feature_name": "Vocabulary Variety",
            "score": score,
        }
        for index, score in enumerate([72, 68, 61], start=1)
    ]
    payload = anonymous_trend_payload(history, "English")
    assert set(payload) == {"language", "sessions_compared", "feature_scores"}
    assert "email" not in str(payload).lower()
    insight = generate_wellness_insight(history, "English")
    assert insight["action"]
    assert insight["source"].startswith("https://")


def test_pause_metrics_and_score_use_timestamps():
    class Segment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    metrics = calculate_pause_metrics(
        [Segment(0.0, 2.0), Segment(3.0, 5.0), Segment(6.5, 8.0)],
        10.0,
    )
    assert metrics["pause_count"] == 2
    assert metrics["max_pause_seconds"] == 1.5
    assert metrics["pause_ratio"] == 0.45
    scores = calculate_feature_scores(
        "Today I spoke clearly about my day.",
        10.0,
        "English",
        metrics,
    )
    pause = next(item for item in scores if item["feature_name"] == "Pause Pattern")
    assert "pause_analysis=v1_placeholder" not in pause["raw_metric"]
    assert 0 <= pause["score"] <= 100


def test_feature_copy_matches_selected_language_and_avoids_filler():
    cases = [
        ("English", "Vocabulary variety"),
        ("日本語", "語彙の多様性"),
        ("中文", "词汇多样性"),
    ]
    for language, expected_label in cases:
        scores = calculate_feature_scores("今天和朋友聊天，然后散步。", 20, language)
        first = scores[0]
        assert display_feature_name(first["feature_name"], language) == expected_label
        assert not first["explanation"].lower().startswith("this ")
        assert 0 <= first["score"] <= 100


def test_research_export_excludes_direct_identifiers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "kina.sqlite3")
    database.init_db()
    user_id = database.upsert_user("hash-only", email="private@example.com")
    session_id = database.create_test_session(
        user_id=user_id,
        session_date="2026-07-28",
        session_number=1,
        app_version="test",
        consent_version="test",
        scoring_model_version="test",
        session_type="Daily reflection",
        language="中文",
        duration_seconds=30,
    )
    database.save_feature_scores(
        session_id,
        [
            {
                "feature_name": "Vocabulary Variety",
                "raw_metric": "ratio=0.5",
                "score": 50,
                "explanation": "test",
            }
        ],
    )
    row = dict(database.list_research_records()[0])
    assert row["participant_id"] == "P000001"
    assert "email" not in row
    assert "display_name" not in row


def test_daily_limit_uses_browser_local_midnight(tmp_path: Path, monkeypatch):
    assert local_date_iso(
        "America/New_York",
        datetime(2026, 7, 29, 0, 30, tzinfo=timezone.utc),
    ) == "2026-07-28"

    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "kina.sqlite3")
    database.init_db()
    user_id = database.upsert_user("timezone-user", "timezone@example.com")
    session_id = database.create_test_session(
        user_id=user_id,
        session_date="2026-07-29",
        session_number=1,
        app_version="test",
        consent_version="test",
        scoring_model_version="test",
    )
    with database.get_connection() as conn:
        conn.execute(
            "UPDATE test_sessions SET created_at = ? WHERE id = ?",
            ("2026-07-29T00:30:00+00:00", session_id),
        )

    migrated = database.assign_timezone_to_legacy_sessions(
        user_id,
        "America/New_York",
    )
    assert migrated == 1
    assert database.count_tests_today(user_id, "2026-07-28") == 1
    assert database.count_tests_today(user_id, "2026-07-29") == 0


def test_mobile_metric_grid_contains_eight_compact_tiles():
    scores = [
        {"feature_name": name, "score": 50 + index}
        for index, name in enumerate(
            [
                "Vocabulary Variety",
                "Response Length",
                "Sentence Complexity",
                "Speech Pace",
                "Pause Pattern",
                "Repetition Pattern",
                "Emotional Tone",
                "Transcription Clarity",
            ]
        )
    ]
    markup = metric_grid_html(scores, "English")
    assert markup.count('class="metric-tile"') == 8
    assert "Vocabulary variety" in markup
    assert "57" in markup


def test_latest_session_scores_returns_only_most_recent_session():
    history = pd.DataFrame(
        [
            {"session_id": 1, "created_at": "2026-08-02T10:00:00Z", "feature_name": "A", "score": 40},
            {"session_id": 2, "created_at": "2026-08-02T10:00:00Z", "feature_name": "A", "score": 60},
            {"session_id": 2, "created_at": "2026-08-02T10:00:00Z", "feature_name": "B", "score": 70},
        ]
    )
    latest = latest_session_scores(history)
    assert [item["feature_name"] for item in latest] == ["A", "B"]
    assert [item["score"] for item in latest] == [60, 70]
