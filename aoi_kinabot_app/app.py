"""Aoi-maintained KinaBot V1 local skeleton app."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from auth import create_local_verification_code, verify_code
from config import APP_VERSION, CONSENT_VERSION, MAX_TESTS_PER_DAY, SCORING_MODEL_VERSION
from database import (
    count_tests_today,
    create_test_session,
    get_user_scores,
    init_db,
    record_consent,
    save_feature_scores,
    upsert_user,
)
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
    if st.button("Generate local verification code"):
        if not email.strip():
            st.warning("Enter an email address first.")
        else:
            st.session_state.email = email.strip()
            _, code = create_local_verification_code(email)
            st.info(f"Local dev code: {code}")
            st.caption("Production will send this code by email.")

    code = st.text_input("Verification code")
    if st.button("Verify code"):
        email_hash = verify_code(email, code)
        if not email_hash:
            st.error("Invalid or expired code.")
        else:
            st.session_state.email = email.strip()
            st.session_state.email_hash = email_hash
            st.session_state.user_id = upsert_user(email_hash)
            st.session_state.verified = True
            st.success("Verified.")

    st.divider()
    st.caption(f"App: {APP_VERSION}")
    st.caption(f"Consent: {CONSENT_VERSION}")
    st.caption(f"Scoring: {SCORING_MODEL_VERSION}")


if not st.session_state.verified:
    st.info("Enter your email and local verification code to start.")
    st.stop()


st.subheader("Consent")
consent = st.checkbox(
    "I understand KinaBot is for personal reflection, not medical diagnosis. "
    "Raw audio and transcripts should not be stored by default in V1."
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

st.subheader("V1 Text-Based Skeleton Test")
st.caption("Audio upload comes next. This skeleton uses transcript text so we can verify scoring and storage first.")

sample_text = st.text_area(
    "Paste sample speech transcript",
    height=160,
    placeholder="Example: Today I went to the store and talked with my family...",
)
duration_seconds = st.number_input(
    "Optional speaking duration in seconds",
    min_value=0.0,
    value=30.0,
    step=1.0,
)

if st.button("Calculate feature scores"):
    if not sample_text.strip():
        st.warning("Enter transcript text first.")
    else:
        scores = calculate_feature_scores(sample_text, duration_seconds)
        session_number = tests_today + 1
        test_session_id = create_test_session(
            user_id=st.session_state.user_id,
            session_date=today,
            session_number=session_number,
            app_version=APP_VERSION,
            consent_version=CONSENT_VERSION,
            scoring_model_version=SCORING_MODEL_VERSION,
        )
        save_feature_scores(test_session_id, scores)

        st.success(f"Session {session_number} saved. No raw audio or transcript stored by this skeleton.")
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
