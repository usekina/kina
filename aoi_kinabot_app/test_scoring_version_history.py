import database


def _save_session(user_id: int, version: str, score: float) -> None:
    session_id = database.create_test_session(
        user_id=user_id,
        session_date="2026-08-24",
        session_number=1,
        app_version="test",
        consent_version="test",
        scoring_model_version=version,
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


def test_score_history_can_be_limited_to_one_model_version(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "kina.sqlite3")
    database.init_db()
    user_id = database.upsert_user("version-test")
    _save_session(user_id, "score-v2-multilingual", 24.0)
    _save_session(user_id, "score-v3-connector-boundaries", 14.0)

    current_rows = database.get_user_scores(
        user_id, scoring_model_version="score-v3-connector-boundaries"
    )

    assert len(current_rows) == 1
    assert current_rows[0]["score"] == 14.0
    assert current_rows[0]["scoring_model_version"] == "score-v3-connector-boundaries"
    assert len(database.get_user_scores(user_id)) == 2
