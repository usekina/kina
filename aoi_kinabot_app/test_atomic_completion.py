from concurrent.futures import ThreadPoolExecutor

import database


def _scores():
    return [
        {
            "feature_name": "Pause Ratio",
            "raw_metric": "test",
            "score": 42.0,
            "explanation": "test",
        }
    ]


def test_concurrent_completion_cannot_exceed_daily_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "kina.sqlite3")
    database.init_db()
    user_id = database.upsert_user("concurrency-test")

    def complete():
        try:
            return database.complete_test_session(
                user_id=user_id,
                session_date="2026-09-03",
                app_version="app-v1",
                consent_version="consent-v1",
                scoring_model_version="score-v1",
                scores=_scores(),
                max_tests_per_day=1,
            )
        except database.DailyLimitReached:
            return "limit"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: complete(), range(2)))
    assert results.count("limit") == 1
    assert database.count_tests_today(user_id, "2026-09-03") == 1


def test_idempotency_key_returns_existing_complete_session(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "kina.sqlite3")
    database.init_db()
    user_id = database.upsert_user("idempotency-test")
    kwargs = dict(
        user_id=user_id,
        session_date="2026-09-03",
        app_version="app-v1",
        consent_version="consent-v1",
        scoring_model_version="score-v1",
        scores=_scores(),
        max_tests_per_day=1,
        idempotency_key="request-123",
    )
    first = database.complete_test_session(**kwargs)
    second = database.complete_test_session(**kwargs)
    assert first[0] == second[0]
    assert second[2] is True
    assert database.count_tests_today(user_id, "2026-09-03") == 1
