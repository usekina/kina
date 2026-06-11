"""Aoi-maintained KinaBot V1 local skeleton app."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from audio_processing import SUPPORTED_AUDIO_TYPES, accept_audio_upload
from auth import create_local_verification_code, verify_code
from config import APP_VERSION, CONSENT_VERSION, MAX_TESTS_PER_DAY, SCORING_MODEL_VERSION
from database import (
    count_tests_today,
    create_test_session,
    get_admin_metrics,
    get_user_scores,
    init_db,
    list_admin_test_records,
    list_admin_users,
    record_consent,
    save_feature_scores,
    update_user_profile,
    upsert_user,
)
from email_delivery import send_verification_email
from scoring import calculate_feature_scores


st.set_page_config(page_title="KinaBot V1", layout="wide")
init_db()

st.title("KinaBot V1")
st.caption("Speech and language-based cognitive wellness reflection. Not a medical diagnosis.")

if "email" not in st.session_state:
    st.session_state.email = ""
if "email_hash" not in st.session_state:
    st.session_state.email_hash = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "verified" not in st.session_state:
    st.session_state.verified = False


with st.sidebar:
    st.header("Access")
    email = st.text_input("Email", value=st.session_state.email)
    if st.button("Send verification code"):
        if not email.strip():
            st.warning("Enter an email address first.")
        else:
            st.session_state.email = email.strip()
            _, code = create_local_verification_code(email)
            sent, message = send_verification_email(st.session_state.email, code)
            if sent:
                st.success(message)
            else:
                st.warning(message)
                st.info(f"Local dev code: {code}")
                st.caption("Set SMTP environment variables to send this code by email.")

    code = st.text_input("Verification code")
    if st.button("Verify code"):
        email_hash = verify_code(email, code)
        if not email_hash:
            st.error("Invalid or expired code.")
        else:
            st.session_state.email = email.strip()
            st.session_state.email_hash = email_hash
            st.session_state.user_id = upsert_user(email_hash, email=st.session_state.email)
            st.session_state.verified = True
            st.success("Verified.")

    st.divider()
    st.caption(f"App: {APP_VERSION}")
    st.caption(f"Consent: {CONSENT_VERSION}")
    st.caption(f"Scoring: {SCORING_MODEL_VERSION}")

    st.divider()
    with st.expander("Local admin metrics"):
        admin_key = st.text_input("Admin key", type="password")
        if admin_key == "local-admin":
            metrics = get_admin_metrics()
            st.metric("Users", metrics["total_users"])
            st.metric("Tests", metrics["total_tests"])
            st.metric("Scores", metrics["total_scores"])
            st.metric("Active today", metrics["active_users_today"])
        elif admin_key:
            st.warning("Invalid local admin key.")


if not st.session_state.verified:
    st.info("Enter your email and local verification code to start.")
    st.stop()


st.subheader("Pilot Profile")
st.caption("These fields are optional and help KinaBot understand pilot usage. Do not enter medical history.")
profile_col1, profile_col2, profile_col3 = st.columns(3)
with profile_col1:
    age_range = st.selectbox(
        "Age range (optional)",
        ["Prefer not to say", "Under 30", "30-44", "45-59", "60-74", "75+"],
    )
with profile_col2:
    primary_language = st.selectbox(
        "Primary language (optional)",
        ["Prefer not to say", "English", "Japanese", "Chinese", "Spanish", "Other"],
    )
with profile_col3:
    country_region = st.text_input("Country / region (optional)", placeholder="Example: US")

if st.button("Save pilot profile"):
    update_user_profile(
        st.session_state.user_id,
        None if age_range == "Prefer not to say" else age_range,
        None if primary_language == "Prefer not to say" else primary_language,
        country_region.strip() or None,
    )
    st.success("Pilot profile saved.")


st.subheader("Consent")
consent = st.checkbox(
    "I understand KinaBot is for personal reflection, not medical diagnosis. "
    "For the V1 pilot, KinaBot may store my email, optional profile fields, usage records, "
    "and calculated feature scores. Raw audio and transcripts should not be stored by default."
)

if not consent:
    st.warning("Please accept the consent notice before starting a test.")
    st.stop()

record_consent(st.session_state.user_id, CONSENT_VERSION)

today = date.today().isoformat()
tests_today = count_tests_today(st.session_state.user_id, today)
remaining = MAX_TESTS_PER_DAY - tests_today
st.metric("Tests remaining today", max(0, remaining))

if remaining <= 0:
    st.warning("You have reached today's V1 pilot limit.")
    st.stop()

st.subheader("V1 Speech Upload Test")

language = st.radio(
    "Language",
    [
        "English",
        "Japanese (planned)",
        "Chinese (planned)",
        "Spanish (planned)",
    ],
    horizontal=True,
    help="English scoring is active in V1. Japanese, Chinese, and Spanish are planned before the end of 2026.",
)

st.markdown(
    """
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin: 0.5rem 0 1rem 0;">
      <span style="background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7; border-radius:6px; padding:6px 10px; font-size:0.9rem;">English: active</span>
      <span style="background:#f3f4f6; color:#6b7280; border:1px solid #d1d5db; border-radius:6px; padding:6px 10px; font-size:0.9rem;">Japanese: planned by end of 2026</span>
      <span style="background:#f3f4f6; color:#6b7280; border:1px solid #d1d5db; border-radius:6px; padding:6px 10px; font-size:0.9rem;">Chinese: planned by end of 2026</span>
      <span style="background:#f3f4f6; color:#6b7280; border:1px solid #d1d5db; border-radius:6px; padding:6px 10px; font-size:0.9rem;">Spanish: planned by end of 2026</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if language != "English":
    st.info(
        f"{language.replace(' (planned)', '')} support is planned before the end of 2026. "
        "For now, please use English to test the V1 scoring skeleton."
    )
    st.stop()

session_type = st.radio(
    "Session type",
    ["Daily full reflection", "Quick check-in"],
    horizontal=True,
    help="Daily full reflection is intended for 2-3 minutes. Quick check-in is intended for about 60 seconds.",
)
default_duration = 180.0 if session_type == "Daily full reflection" else 60.0

st.caption(
    "Upload a speech audio file for the V1 pilot flow. The local skeleton accepts the file, "
    "temporarily processes it, and deletes the temporary copy. Speech-to-text is not connected yet."
)

uploaded_audio = st.file_uploader(
    "Upload speech audio",
    type=SUPPORTED_AUDIO_TYPES,
    help="Supported local test formats: WAV, MP3, M4A, AAC, OGG, FLAC.",
)

audio_metadata = None
if uploaded_audio is not None:
    st.audio(uploaded_audio)
    st.caption(
        f"Selected file: {uploaded_audio.name} "
        f"({uploaded_audio.size / 1024:.1f} KB). Raw audio will not be stored by this app."
    )

sample_text = st.text_area(
    "Paste transcript for this uploaded speech sample",
    height=160,
    placeholder="Example: Today I went to the store and talked with my family...",
    help="Until speech-to-text is connected, paste a transcript here so the scoring skeleton can run.",
)
duration_seconds = st.number_input(
    "Optional speaking duration in seconds",
    min_value=0.0,
    value=default_duration,
    step=1.0,
)

if st.button("Calculate feature scores"):
    if uploaded_audio is None:
        st.warning("Upload a speech audio file first.")
    elif not sample_text.strip():
        st.warning("Enter transcript text first.")
    else:
        audio_metadata = accept_audio_upload(uploaded_audio, uploaded_audio.name)
        scores = calculate_feature_scores(sample_text, duration_seconds)
        session_number = tests_today + 1
        test_session_id = create_test_session(
            user_id=st.session_state.user_id,
            session_date=today,
            session_number=session_number,
            app_version=APP_VERSION,
            consent_version=CONSENT_VERSION,
            scoring_model_version=SCORING_MODEL_VERSION,
            session_type=session_type,
            language="English",
            duration_seconds=duration_seconds,
        )
        save_feature_scores(test_session_id, scores)

        st.success(
            f"Session {session_number} saved. Audio upload accepted and temporary audio deleted. "
            "No raw audio or transcript stored by this skeleton."
        )
        st.json(
            {
                "audio_filename": audio_metadata["filename"],
                "audio_size_bytes": audio_metadata["bytes"],
                "detected_wav_duration_seconds": audio_metadata["duration_seconds"],
                "temporary_audio_deleted": True,
            }
        )
        df = pd.DataFrame(scores)[["feature_name", "score", "raw_metric", "explanation"]]
        st.dataframe(df, width="stretch", hide_index=True)


st.subheader("Trend History")
rows = get_user_scores(st.session_state.user_id)
if not rows:
    st.caption("No saved scores yet.")
else:
    history = pd.DataFrame([dict(row) for row in rows])
    st.dataframe(history, width="stretch", hide_index=True)
    chart_df = history.pivot_table(
        index="created_at",
        columns="feature_name",
        values="score",
        aggfunc="mean",
    )
    st.line_chart(chart_df)


st.subheader("Local Admin Records")
st.caption("Local development view. Production needs real admin authentication.")
admin_view_key = st.text_input("Admin records key", type="password")
if admin_view_key == "local-admin":
    users_df = pd.DataFrame([dict(row) for row in list_admin_users()])
    tests_df = pd.DataFrame([dict(row) for row in list_admin_test_records()])
    st.markdown("#### Users")
    st.dataframe(users_df, width="stretch", hide_index=True)
    st.markdown("#### Test score records")
    st.dataframe(tests_df, width="stretch", hide_index=True)
elif admin_view_key:
    st.warning("Invalid local admin key.")
