import admin_view
from radar_view import FEATURES


def test_admin_key_does_not_replace_verified_owner_identity():
    assert admin_view.is_owner(True, " Owner@Example.com ", "owner@example.com", owner_key_verified=True)
    assert not admin_view.is_owner(False, "owner@example.com", "owner@example.com")
    assert not admin_view.is_owner(True, "other@example.com", "owner@example.com")
    assert not admin_view.is_owner(True, "owner@example.com", "")
    assert not admin_view.is_owner(True, "owner@example.com", "owner@example.com", offline=True)
    assert not admin_view.is_owner(True, "owner@example.com", "owner@example.com", owner_key_verified=False)
    assert admin_view.verify_owner_key("owner@example.com", "owner@example.com", "secret", "secret")
    assert not admin_view.verify_owner_key("other@example.com", "owner@example.com", "secret", "secret")
    assert not admin_view.verify_owner_key("owner@example.com", "owner@example.com", "wrong", "secret")
    assert not admin_view.verify_owner_key("owner@example.com", "owner@example.com", "", "")


def test_non_owner_never_reads_admin_data(monkeypatch):
    def forbidden():
        raise AssertionError("Non-owner accessed admin data")
    for name in ("get_admin_metrics", "list_admin_users", "list_admin_test_records", "list_research_records"):
        monkeypatch.setattr(admin_view, name, forbidden)
    admin_view.render_admin(owner=False, admin_key="correct-key")


def test_sessions_keep_all_features_and_never_merge_equal_timestamps():
    rows = [dict(user_id=9, session_id=sid, created_at="same time", feature_name=name, score=float(i), availability_status="available")
            for sid in [1, 2] for i, name in enumerate(FEATURES)]
    rows[4]["score"] = None
    rows[7]["raw_metric"] = "neutral_score_used=true"
    table = admin_view.session_table(rows)
    assert len(table) == 2
    assert set(FEATURES).issubset(table.columns)
    first = table[table.session_id == 1].iloc[0]
    assert first["feature_rows"] == 8
    assert first["Vocabulary Variety"] == "0"
    assert first["Pause Pattern"] == "Unavailable"
    assert "fallback" in first["Emotional Tone"]
