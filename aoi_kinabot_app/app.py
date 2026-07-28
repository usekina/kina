"""Aoi-maintained KinaBot V1 local skeleton app."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from audio_processing import SUPPORTED_AUDIO_TYPES, accept_audio_upload
from auth import create_local_verification_code, verify_code
from config import (
    ADMIN_KEY,
    ALLOW_LOCAL_VERIFICATION_CODES,
    APP_VERSION,
    CONSENT_VERSION,
    MAX_AUDIO_BYTES,
    MAX_TESTS_PER_DAY,
    SCORING_MODEL_VERSION,
)
from database import (
    count_tests_today,
    create_test_session,
    get_admin_metrics,
    get_user_habit_checkins,
    get_user_scores,
    init_db,
    list_admin_test_records,
    list_admin_users,
    record_consent,
    save_feature_scores,
    save_habit_checkins,
    update_user_profile,
    upsert_user,
)
from email_delivery import send_verification_email
from insight_service import generate_wellness_insight
from language_analysis import LANGUAGE_CODES, analyze_transcript
from speech_to_text import (
    LOCAL_TRANSCRIPTION_TYPES,
    speech_to_text_configured,
    transcribe_audio_upload,
)
from wellness_guidance import wellness_suggestions


st.set_page_config(page_title="KinaBot", page_icon="🎙️", layout="centered")
init_db()

st.markdown(
    """
    <style>
    .block-container {max-width: 760px; padding-top: 2rem; padding-bottom: 4rem;}
    h1 {letter-spacing: -0.04em;}
    .privacy-card {
        padding: 0.9rem 1rem; border-radius: 0.8rem;
        background: rgba(46, 160, 67, 0.08);
        border: 1px solid rgba(46, 160, 67, 0.20);
        margin: 0.5rem 0 1rem;
    }
    .privacy-card strong {color: #238636;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("KinaBot")
st.caption("A simple reflection on one voice sample · English · 日本語 · 中文")

if "email" not in st.session_state:
    st.session_state.email = ""
if "email_hash" not in st.session_state:
    st.session_state.email_hash = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "verified" not in st.session_state:
    st.session_state.verified = False
if not st.session_state.verified:
    st.subheader("Sign in")
    st.caption("We use your email to keep your private history together.")
    email = st.text_input("Email", value=st.session_state.email)
    if st.button("Send code", use_container_width=True):
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
                if ALLOW_LOCAL_VERIFICATION_CODES:
                    st.info(f"Local dev code: {code}")
                else:
                    st.error("Email delivery is unavailable. Please try again later.")

    code = st.text_input("6-digit code")
    if st.button("Continue", type="primary", use_container_width=True):
        email_hash = verify_code(email, code)
        if not email_hash:
            st.error("Invalid or expired code.")
        else:
            st.session_state.email = email.strip()
            st.session_state.email_hash = email_hash
            st.session_state.user_id = upsert_user(email_hash, email=st.session_state.email)
            st.session_state.verified = True
            st.rerun()

    st.caption("KinaBot is a wellness reflection tool, not a medical device.")
    st.stop()


with st.expander("Account"):
    st.caption(st.session_state.email)
    display_name = st.text_input("Name", placeholder="Your name")
    age_range = st.selectbox(
        "Age range (optional)",
        ["Prefer not to say", "Under 30", "30-44", "45-59", "60-74", "75+"],
    )
    primary_language = st.selectbox(
        "Primary language (optional)",
        ["Prefer not to say", "English", "Japanese", "Chinese", "Spanish", "Other"],
    )
    country_region = st.text_input("Country / region (optional)", placeholder="Example: US")
    if st.button("Save account"):
        update_user_profile(
            st.session_state.user_id,
            display_name.strip() or None,
            None if age_range == "Prefer not to say" else age_range,
            None if primary_language == "Prefer not to say" else primary_language,
            country_region.strip() or None,
        )
        st.success("Account saved.")

st.markdown(
    """
    <div class="privacy-card">
      <strong>Your recording is not saved.</strong><br>
      Your selected file is processed privately on the KinaBot server with local Python,
      then the temporary copy is deleted. It is not sent to OpenAI. KinaBot stores your account, scores, usage
      history, and optional habit check-ins—not the raw audio or full transcript.
    </div>
    """,
    unsafe_allow_html=True,
)

consent = st.checkbox(
    "I understand and agree. KinaBot describes this sample only; it does not assess health."
)

if not consent:
    st.caption("Please agree before analyzing a recording.")
    st.stop()

record_consent(st.session_state.user_id, CONSENT_VERSION)

today = date.today().isoformat()
tests_today = count_tests_today(st.session_state.user_id, today)
remaining = MAX_TESTS_PER_DAY - tests_today
if remaining <= 0:
    st.info("You have completed today's two reflections. Come back tomorrow.")
    st.stop()

st.subheader("New reflection")

language = st.radio(
    "1 · Choose the language spoken",
    ["English", "日本語", "中文"],
    horizontal=True,
    help="Choose the language actually spoken in the uploaded recording.",
)

session_type = "Daily reflection"

uploaded_audio = st.file_uploader(
    "2 · Choose a recording from your phone or computer",
    type=SUPPORTED_AUDIO_TYPES,
    help="Supported local test formats: WAV, MP3, M4A, AAC, OGG, FLAC.",
)

if uploaded_audio is not None:
    st.audio(uploaded_audio)
    st.caption(
        f"Selected file: {uploaded_audio.name} "
        f"({uploaded_audio.size / 1024:.1f} KB). Raw audio will not be stored by this app."
    )
    audio_extension = uploaded_audio.name.rsplit(".", 1)[-1].lower()
    can_transcribe = audio_extension in LOCAL_TRANSCRIPTION_TYPES
    if not can_transcribe:
        st.info(
            "This file can be uploaded for local flow testing, but automatic transcription supports "
            "MP3, MP4, MPEG, MPGA, M4A, WAV, and WEBM."
        )
st.caption(f"{max(0, remaining)} of {MAX_TESTS_PER_DAY} reflections available today")
if st.button("3 · Analyze", type="primary", use_container_width=True):
    if uploaded_audio is None:
        st.warning("Upload a speech audio file first.")
    elif uploaded_audio.size > MAX_AUDIO_BYTES:
        st.warning(f"Audio must be {MAX_AUDIO_BYTES // (1024 * 1024)} MB or smaller.")
    elif uploaded_audio.name.rsplit(".", 1)[-1].lower() not in LOCAL_TRANSCRIPTION_TYPES:
        st.warning("Use MP3, MP4, MPEG, MPGA, M4A, WAV, or WEBM for automatic analysis.")
    else:
        with st.status("Processing your recording…", expanded=True) as analysis_status:
            st.write("Transcribing privately on the KinaBot server…")
            (
                transcribed,
                transcript_or_error,
                detected_duration,
                acoustic_metrics,
            ) = transcribe_audio_upload(
                uploaded_audio,
                uploaded_audio.name,
                LANGUAGE_CODES[language],
            )
            if not transcribed:
                analysis_status.update(label="Transcription failed", state="error")
                st.error(transcript_or_error)
                st.stop()

            st.write("Analyzing observable communication patterns…")
            scores, session_summary = analyze_transcript(
                transcript_or_error,
                language,
                detected_duration,
                acoustic_metrics,
            )
            audio_metadata = accept_audio_upload(uploaded_audio, uploaded_audio.name)
            session_number = tests_today + 1
            test_session_id = create_test_session(
                user_id=st.session_state.user_id,
                session_date=today,
                session_number=session_number,
                app_version=APP_VERSION,
                consent_version=CONSENT_VERSION,
                scoring_model_version=SCORING_MODEL_VERSION,
                session_type=session_type,
                language=language,
                duration_seconds=detected_duration or audio_metadata["duration_seconds"],
            )
            save_feature_scores(test_session_id, scores)
            analysis_status.update(label="Analysis complete", state="complete", expanded=False)

        st.success(f"Session {session_number} saved. Raw audio and full transcript were not retained.")
        st.info(session_summary)
        score_df = pd.DataFrame(scores)[["feature_name", "score", "explanation"]]
        st.dataframe(
            score_df,
            width="stretch",
            hide_index=True,
            column_config={
                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100)
            },
        )
        st.caption(
            "Scores describe observable features in this sample only. "
            "They do not indicate health, ability, improvement, decline, or risk."
        )

st.subheader("Your history")
rows = get_user_scores(st.session_state.user_id)
if not rows:
    st.caption("No saved scores yet.")
else:
    history = pd.DataFrame([dict(row) for row in rows])
    session_count = len(history[["created_at", "session_number"]].drop_duplicates())
    if session_count < 3:
        st.info(f"{session_count} of 3 sessions completed. Trends unlock after session 3.")
    else:
        chart_df = history.pivot_table(
            index="created_at",
            columns="feature_name",
            values="score",
            aggfunc="mean",
        )
        st.line_chart(chart_df)

        ordered = history.sort_values("created_at")
        first_scores = ordered.groupby("feature_name").first()["score"]
        latest_scores = ordered.groupby("feature_name").last()["score"]
        change = (latest_scores - first_scores).rename("observed_change").reset_index()
        change["pattern"] = change["observed_change"].apply(
            lambda value: "Higher in latest sample"
            if value > 2
            else ("Lower in latest sample" if value < -2 else "Similar")
        )
        st.markdown("#### Observed change since the first sample")
        st.dataframe(change, width="stretch", hide_index=True)
        st.caption(
            "These are descriptive sample-to-sample differences only. "
            "KinaBot does not infer health, improvement, decline, cause, or risk."
        )
        insight = generate_wellness_insight(
            [dict(row) for row in rows],
            language,
        )
        st.markdown("#### One small action")
        if insight.get("encouragement"):
            st.write(insight["encouragement"])
        st.info(insight["action"])
        st.caption(f"{insight['why']} [Research source]({insight['source']})")
        st.caption(insight["boundary"])

st.subheader("Optional wellness habits")
habit_copy = wellness_suggestions(language, [])
st.caption(
    "Optional habit tracking is separate from speech scores. KinaBot does not claim "
    "that a habit caused any score or sample change."
)
habit_values = {
    habit_name: st.checkbox(label, key=f"habit_{today}_{habit_name}")
    for habit_name, label in habit_copy["habit_labels"].items()
}
if st.button("Save today's habit check-in"):
    save_habit_checkins(st.session_state.user_id, today, habit_values)
    st.success("Today's optional wellness habits were saved.")

habit_rows = get_user_habit_checkins(st.session_state.user_id)
if habit_rows:
    habit_history = pd.DataFrame([dict(row) for row in habit_rows])
    habit_daily = (
        habit_history.groupby("checkin_date", as_index=False)["completed"]
        .sum()
        .rename(columns={"completed": "habits_completed"})
    )
    st.bar_chart(habit_daily.set_index("checkin_date"))
    st.caption("This chart shows self-reported habit completion only.")


if ADMIN_KEY:
    with st.sidebar.expander("Admin"):
        admin_view_key = st.text_input("Admin key", type="password")
        if admin_view_key == ADMIN_KEY:
            metrics = get_admin_metrics()
            st.metric("Users", metrics["total_users"])
            st.metric("Tests", metrics["total_tests"])
            users_df = pd.DataFrame([dict(row) for row in list_admin_users()])
            tests_df = pd.DataFrame([dict(row) for row in list_admin_test_records()])
            st.dataframe(users_df, hide_index=True)
            st.dataframe(tests_df, hide_index=True)
        elif admin_view_key:
            st.warning("Invalid admin key.")

st.divider()
st.caption(
    "KinaBot is for personal wellness reflection, not diagnosis or medical advice. "
    f"{APP_VERSION}"
)
