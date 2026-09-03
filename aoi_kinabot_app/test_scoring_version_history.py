import database


def _save_session(
    user_id: int, version: str, score: float, language: str, session_number: int
) -> None:
    session_id = database.create_test_session(
        user_id=user_id,
        session_date="2026-08-24",
        session_number=session_number,
        app_version="test",
        consent_version="test",
        scoring_model_version=version,
        language=language,
    )
    database.save_feature_scores(
        session_id,
        [
            {
                "feature_name": "Sentence Complexity",
                "raw_metric": "test",
                "score": score,
                "explanation": "test",
            }
        ],
    )


def test_score_history_combines_languages_and_model_versions_for_one_user(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "kina.sqlite3")
    database.init_db()
    user_id = database.upsert_user("version-test")
    assert database.upsert_user("version-test") == user_id
    _save_session(user_id, "score-v2-multilingual", 24.0, "English", 1)
    _save_session(user_id, "score-v3-connector-boundaries", 14.0, "中文", 2)

    rows = database.get_user_scores(user_id)

    assert len(rows) == 2
    assert {row["score"] for row in rows} == {14.0, 24.0}
    assert {row["language"] for row in rows} == {"English", "中文"}
    assert {row["scoring_model_version"] for row in rows} == {
        "score-v2-multilingual",
        "score-v3-connector-boundaries",
    }
