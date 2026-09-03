import pandas as pd

from history_view import select_latest_comparable_history


def _rows():
    rows = []
    session_id = 1
    for language, count in [("English", 3), ("日本語", 3)]:
        for _ in range(count):
            rows.append(
                {
                    "session_id": session_id,
                    "language": language,
                    "scoring_model_version": "score-v4",
                    "app_version": "v1.2",
                    "feature_name": "Pause Ratio",
                    "score": float(session_id),
                }
            )
            session_id += 1
    return pd.DataFrame(rows)


def test_selects_newest_eligible_group_and_keeps_only_its_rows():
    selected, key = select_latest_comparable_history(_rows())
    assert key == ("日本語", "score-v4", "v1.2")
    assert selected["language"].nunique() == 1
    assert selected["session_id"].nunique() == 3


def test_fewer_than_three_sessions_is_not_comparable():
    frame = _rows().iloc[:2].copy()
    selected, key = select_latest_comparable_history(frame)
    assert key is None
    assert len(selected) == len(frame)
