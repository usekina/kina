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
    get_user_profile,
    get_user_scores,
    init_db,
    list_admin_test_records,
    list_admin_users,
    list_research_records,
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
from scoring import display_feature_name
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
    .score-card {
        padding: 0.9rem 1rem;
        margin: 0.55rem 0;
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 0.85rem;
        background: rgba(250, 250, 250, 0.75);
    }
    .score-card__top {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.45rem;
    }
    .score-card__name {font-weight: 650; font-size: 1.02rem;}
    .score-card__value {font-weight: 750; white-space: nowrap;}
    .score-card__track {
        width: 100%; height: 0.42rem; border-radius: 99px;
        background: rgba(252, 110, 81, 0.15); overflow: hidden;
    }
    .score-card__fill {
        height: 100%; border-radius: 99px; background: #fc6e51;
    }
    .score-card__explanation {
        color: rgba(49, 51, 63, 0.72);
        font-size: 0.9rem; line-height: 1.45; margin-top: 0.55rem;
    }
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
if "code_sent" not in st.session_state:
    st.session_state.code_sent = False
if "staging_code" not in st.session_state:
    st.session_state.staging_code = ""
if "profile" not in st.session_state:
    st.session_state.profile = None
if not st.session_state.verified:
    st.subheader("Start")
    st.caption("Enter an email to keep your scores and history together.")
    email = st.text_input("Email", value=st.session_state.email)
    if not st.session_state.code_sent:
        send_code = st.button("Send code", type="primary", use_container_width=True)
    else:
        send_code = False
    if send_code:
        normalized_email = email.strip().lower()
        if (
            not normalized_email
            or "@" not in normalized_email
            or normalized_email.startswith("@")
            or normalized_email.endswith("@")
        ):
            st.error("Enter a valid email address.")
        else:
            st.session_state.email = normalized_email
            _, code = create_local_verification_code(normalized_email)
            sent, message = send_verification_email(normalized_email, code)
            if sent:
                st.session_state.code_sent = True
                st.session_state.staging_code = ""
                st.success(message)
            elif ALLOW_LOCAL_VERIFICATION_CODES:
                st.session_state.code_sent = True
                st.session_state.staging_code = code
            else:
                st.error("Email delivery is unavailable. Please try again later.")
            st.rerun()

    if st.session_state.code_sent:
        if st.session_state.staging_code:
            st.info(f"Private staging code: {st.session_state.staging_code}")
        code = st.text_input("6-digit code", max_chars=6)
        if st.button("Continue", type="primary", use_container_width=True):
            email_hash = verify_code(st.session_state.email, code)
            if not email_hash:
                st.error("Invalid or expired code.")
            else:
                st.session_state.email_hash = email_hash
                st.session_state.user_id = upsert_user(
                    email_hash,
                    email=st.session_state.email,
                )
                profile = get_user_profile(st.session_state.user_id)
                st.session_state.profile = dict(profile) if profile else {}
                st.session_state.verified = True
                st.session_state.staging_code = ""
                st.rerun()

    st.caption("KinaBot is a wellness reflection tool, not a medical device.")
    st.stop()

if ADMIN_KEY:
    with st.sidebar.expander("Research admin"):
        st.caption("Private owner access")
        admin_view_key = st.text_input("Admin key", type="password")
        if admin_view_key == ADMIN_KEY:
            metrics = get_admin_metrics()
            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric("Users", metrics["total_users"])
            metric_col_2.metric("Sessions", metrics["total_tests"])
            st.caption(f"Active today: {metrics['active_users_today']}")

            users_df = pd.DataFrame([dict(row) for row in list_admin_users()])
            tests_df = pd.DataFrame([dict(row) for row in list_admin_test_records()])
            research_df = pd.DataFrame(
                [dict(row) for row in list_research_records()]
            )
            if not users_df.empty:
                with st.expander("Participant profile summary"):
                    summary_rows = []
                    for field, label in [
                        ("age_range", "Age range"),
                        ("gender", "Gender"),
                        ("primary_language", "Primary language"),
                        ("country_region", "Country / region"),
                    ]:
                        counts = (
                            users_df[field]
                            .fillna("Not provided")
                            .replace("", "Not provided")
                            .value_counts()
                        )
                        summary_rows.extend(
                            {
                                "field": label,
                                "value": value,
                                "users": int(count),
                            }
                            for value, count in counts.items()
                        )
                    st.dataframe(
                        pd.DataFrame(summary_rows),
                        hide_index=True,
                        width="stretch",
                    )

            st.download_button(
                "Download research CSV",
                data=research_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"kinabot_research_{date.today().isoformat()}.csv",
                mime="text/csv",
                use_container_width=True,
                help="De-identified longitudinal records. No email or display name.",
            )
            st.download_button(
                "Download private user list",
                data=users_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"kinabot_users_private_{date.today().isoformat()}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Contains personal information. Store separately from research data.",
            )
            st.markdown("**Recent sessions**")
            st.dataframe(tests_df.head(100), hide_index=True, width="stretch")
            st.caption(
                "Use the de-identified research CSV for analysis. Keep the private "
                "user list access-restricted and never place it in GitHub."
            )
        elif admin_view_key:
            st.warning("Invalid admin key.")


if st.session_state.profile is None:
    profile = get_user_profile(st.session_state.user_id)
    st.session_state.profile = dict(profile) if profile else {}

profile = st.session_state.profile
saved_name = (profile.get("display_name") or "").strip()
age_options = ["Prefer not to say", "Under 30", "30-44", "45-59", "60-74", "75+"]
gender_options = ["Prefer not to say", "Woman", "Man", "Non-binary", "Self-describe"]
language_options = [
    "Prefer not to say",
    "English",
    "Japanese",
    "Chinese",
    "Spanish",
    "Other",
]

if saved_name:
    st.markdown(f"### Welcome back, {saved_name}")

profile_complete = bool(
    saved_name and profile.get("age_range") and profile.get("gender")
)
with st.expander(
    "Account settings" if profile_complete else "Complete your account",
    expanded=not profile_complete,
):
    st.caption(st.session_state.email)
    display_name = st.text_input(
        "Name",
        value=saved_name,
        placeholder="Your name",
    )
    age_range = st.selectbox(
        "Age range",
        age_options,
        index=(
            age_options.index(profile.get("age_range"))
            if profile.get("age_range") in age_options
            else None
        ),
        placeholder="Select an age range",
    )
    gender = st.selectbox(
        "Gender",
        gender_options,
        index=(
            gender_options.index(profile.get("gender"))
            if profile.get("gender") in gender_options
            else None
        ),
        placeholder="Select a gender option",
    )
    primary_language = st.selectbox(
        "Primary language (optional)",
        language_options,
        index=(
            language_options.index(profile.get("primary_language"))
            if profile.get("primary_language") in language_options
            else 0
        ),
    )
    country_region = st.text_input(
        "Country / region (optional)",
        value=profile.get("country_region") or "",
        placeholder="Example: US",
    )
    if st.button("Save account"):
        if not display_name.strip() or age_range is None or gender is None:
            st.error("Please enter your name and select age range and gender.")
        else:
            update_user_profile(
                st.session_state.user_id,
                display_name.strip(),
                age_range,
                gender,
                None if primary_language == "Prefer not to say" else primary_language,
                country_region.strip() or None,
            )
            refreshed_profile = get_user_profile(st.session_state.user_id)
            st.session_state.profile = dict(refreshed_profile) if refreshed_profile else {}
            st.success("Account saved.")
            st.rerun()

if not profile_complete:
    st.info("Complete your account once. KinaBot will remember it for future visits.")
    st.stop()

st.markdown(
    """
    <div class="privacy-card">
      <strong>Your recording is not saved.</strong><br>
      Your selected file is processed privately on the KinaBot server with local Python,
      then the temporary copy is deleted. Raw audio and full transcripts are not
      sent to OpenAI. After three sessions, only anonymous score history may be
      used to select a general wellness action. KinaBot stores your account, scores,
      usage history, and optional habit check-ins—not the raw audio or full transcript.
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
    st.info(
        f"You have completed today's {MAX_TESTS_PER_DAY} reflections. "
        "Come back tomorrow."
    )
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

        result_copy = {
            "English": {
                "saved": f"Session {session_number} saved. Audio and full transcript were not retained.",
                "title": "Your sample",
                "scale": "Each score is a 0–100 sample feature index—not a percentage or health rating.",
                "boundary": (
                    "Scores describe this recording only. They do not indicate health, "
                    "ability, improvement, decline, or risk."
                ),
            },
            "日本語": {
                "saved": f"セッション{session_number}を保存しました。音声と全文は保存していません。",
                "title": "今回の結果",
                "scale": "各スコアは0〜100のサンプル特徴指数です。割合や健康評価ではありません。",
                "boundary": "スコアは今回の録音だけを表し、健康・能力・改善・低下・リスクを示すものではありません。",
            },
            "中文": {
                "saved": f"第 {session_number} 次记录已保存。语音和完整转写文本均未保留。",
                "title": "本次结果",
                "scale": "每项为 0–100 的样本特征分数，不是百分比、健康评分或人群排名。",
                "boundary": "分数只描述本次录音，不代表健康、能力、改善、下降或风险。",
            },
        }[language]
        st.success(result_copy["saved"])
        st.info(session_summary)
        st.markdown(f"### {result_copy['title']}")
        st.caption(result_copy["scale"])
        for item in scores:
            score = int(round(float(item["score"])))
            label = display_feature_name(item["feature_name"], language)
            st.markdown(
                f"""
                <div class="score-card">
                  <div class="score-card__top">
                    <span class="score-card__name">{label}</span>
                    <span class="score-card__value">{score} / 100</span>
                  </div>
                  <div class="score-card__track">
                    <div class="score-card__fill" style="width:{score}%"></div>
                  </div>
                  <div class="score-card__explanation">{item["explanation"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption(result_copy["boundary"])

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

st.subheader("Today's wellness habit")
habit_copy = wellness_suggestions(language, [])
st.caption(
    "Choose the one habit that best matches today. Habit tracking is separate from "
    "speech scores. KinaBot does not claim "
    "that a habit caused any score or sample change."
)
habit_labels = habit_copy["habit_labels"]
selected_habit_label = st.radio(
    "Select one",
    list(habit_labels.values()),
    index=None,
    key=f"habit_{today}",
)
if st.button("Save today's habit check-in"):
    if selected_habit_label is None:
        st.error("Please select one habit.")
    else:
        selected_habit = next(
            name for name, label in habit_labels.items() if label == selected_habit_label
        )
        habit_values = {name: name == selected_habit for name in habit_labels}
        save_habit_checkins(st.session_state.user_id, today, habit_values)
        st.success("Today's wellness habit was saved.")

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


st.divider()
st.caption(
    "KinaBot is for personal wellness reflection, not diagnosis or medical advice. "
    f"{APP_VERSION}"
)
