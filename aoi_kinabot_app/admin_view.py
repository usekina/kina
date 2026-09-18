"""Owner-only administration, grouped by user and recording."""

from __future__ import annotations

import hmac

import pandas as pd
import streamlit as st

from database import get_admin_metrics, list_admin_test_records, list_admin_users, list_research_records
from radar_view import FEATURES


def verify_owner_key(email: str, owner_email: str, supplied: str, expected: str) -> bool:
    return bool(owner_email.strip() and expected and supplied
                and hmac.compare_digest(email.strip().casefold().encode(), owner_email.strip().casefold().encode())
                and hmac.compare_digest(supplied.encode(), expected.encode()))


def is_owner(verified: bool, email: str, owner_email: str, *, offline: bool = False, owner_key_verified: bool = False) -> bool:
    """Fail closed: an admin key alone never grants cross-account access."""
    return bool(verified and owner_key_verified and not offline and owner_email.strip()
                and hmac.compare_digest(email.strip().casefold().encode(), owner_email.strip().casefold().encode()))


def session_table(records: list[dict]) -> pd.DataFrame:
    """One row per session ID, never per timestamp or per feature."""
    sessions = {}
    fields = ("user_id", "session_id", "created_at", "session_date", "session_number",
              "language", "duration_seconds", "scoring_model_version", "analysis_pipeline_id")
    for source in records:
        row = dict(source)
        sid = row["session_id"]
        if sid not in sessions:
            sessions[sid] = {field: row.get(field) for field in fields}
            sessions[sid].update({name: "Not recorded" for name in FEATURES})
            sessions[sid]["feature_rows"] = 0
        feature = row.get("feature_name")
        if feature in FEATURES:
            value = row.get("score")
            if row.get("availability_status") == "unavailable" or value is None:
                label = "Unavailable"
            elif feature == "Emotional Tone" and "neutral_score_used=true" in str(row.get("raw_metric") or ""):
                label = f"{value:g} (fallback)"
            else:
                label = f"{value:g}"
            sessions[sid][feature] = label
            sessions[sid]["feature_rows"] += 1
    return pd.DataFrame([sessions[sid] for sid in sorted(sessions, reverse=True)])


def render_admin(*, owner: bool, admin_key: str) -> None:
    # Authorization is checked inside the data view as well as navigation.
    if not owner or not admin_key:
        return
    st.title("Research administration")
    st.caption("Owner-only access. Each row below is one recording, with all eight feature scores.")
    metrics = get_admin_metrics()
    a, b, c = st.columns(3)
    a.metric("Accounts", metrics["total_users"])
    b.metric("Recordings", metrics["total_tests"])
    c.metric("Feature rows", metrics["total_scores"])
    users = [dict(row) for row in list_admin_users()]
    if not users:
        st.info("No accounts yet.")
        return
    by_id = {row["id"]: row for row in users}
    user_id = st.selectbox("User", list(by_id), format_func=lambda uid: f"#{uid} · {by_id[uid].get('display_name') or 'Unnamed'} · {by_id[uid].get('email') or 'Offline participant'}")
    records = [dict(row) for row in list_admin_test_records()]
    selected = [row for row in records if row["user_id"] == user_id]
    wide = session_table(selected)
    st.subheader("All recordings for this user")
    if wide.empty:
        st.info("This user has no saved recordings.")
    else:
        st.caption("Scroll horizontally for all eight columns. Missing or fallback values are labelled. Time is UTC; session_date is the recording's saved local date.")
        st.dataframe(wide, hide_index=True, use_container_width=True)
        st.download_button("Download this user's recordings (8 scores per row)", wide.to_csv(index=False).encode("utf-8-sig"), f"kinabot_user_{user_id}_sessions.csv", "text/csv")
        ids = list(wide["session_id"])
        sid = st.selectbox("Recording details", ids, format_func=lambda value: f"Session #{value}")
        detail = pd.DataFrame([row for row in selected if row["session_id"] == sid])
        st.dataframe(detail[["feature_name", "score", "availability_status", "failure_reason", "raw_metric"]], hide_index=True, use_container_width=True)
    with st.expander("All-account exports"):
        st.caption("Private exports contain identifying information. Keep them access-restricted.")
        st.download_button("Private user list", pd.DataFrame(users).to_csv(index=False).encode("utf-8-sig"), "kinabot_users_private.csv", "text/csv")
        st.download_button("All recordings — eight scores per row", session_table(records).to_csv(index=False).encode("utf-8-sig"), "kinabot_all_sessions_private.csv", "text/csv")
        research = pd.DataFrame([dict(row) for row in list_research_records()])
        st.download_button("Research feature rows (de-identified)", research.to_csv(index=False).encode("utf-8-sig"), "kinabot_research.csv", "text/csv")
        st.caption("The research export contains available stored records; verify current consent and study eligibility separately.")
