"""Aoi-maintained KinaBot V1 local skeleton app."""

from __future__ import annotations

from datetime import date
import pandas as pd
import streamlit as st

from audio_processing import SUPPORTED_AUDIO_TYPES, accept_audio_upload
from auth import create_local_verification_code, verify_code
from challenge_progress import CHALLENGE_DAYS, challenge_status
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
    assign_timezone_to_legacy_sessions,
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
from history_view import latest_session_scores, metric_grid_html
from insight_service import generate_wellness_insight
from language_analysis import LANGUAGE_CODES, analyze_transcript
from local_time import local_date_iso
from speech_to_text import (
    LOCAL_TRANSCRIPTION_TYPES,
    speech_to_text_configured,
    transcribe_audio_upload,
)
from scoring import display_feature_name, feature_explanation
from reflection_profile import build_reflection_profile
from wellness_guidance import wellness_suggestions


st.set_page_config(page_title="KinaBot", page_icon="🎙️", layout="centered")
init_db()
browser_timezone = st.context.timezone or "UTC"
today = local_date_iso(browser_timezone)

st.markdown(
    """
    <style>
    .block-container {max-width: 760px; padding-top: 2rem; padding-bottom: 4rem;}
    h1 {letter-spacing: -0.04em;}
    .kinabot-hero {
        padding: 2.6rem 0 1.45rem;
        text-align: center;
    }
    .kinabot-hero__brand {
        color: #f26a21;
        font-size: clamp(1.45rem, 4vw, 1.9rem);
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1;
        margin-bottom: 1.15rem;
    }
    .kinabot-hero__title {
        color: #252832;
        font-size: clamp(2.45rem, 7vw, 4.4rem);
        font-weight: 780;
        letter-spacing: -0.055em;
        line-height: 1.02;
        margin: 0 auto;
        max-width: 720px;
    }
    .kinabot-hero__subtitle {
        color: rgba(49, 51, 63, 0.68);
        font-size: 1.05rem;
        line-height: 1.65;
        margin: 1.2rem auto 0;
        max-width: 600px;
    }
    .kinabot-language-label {
        color: rgba(49, 51, 63, 0.62);
        font-size: 0.88rem;
        font-weight: 650;
        letter-spacing: 0.02em;
        margin: 0.25rem 0 0.2rem;
        text-align: center;
    }
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
    .snapshot-card {
        padding: 0.9rem 1rem; margin: 0.35rem 0;
        border: 1px solid #f4d4c3; border-radius: 1rem;
        background: linear-gradient(145deg, #fff7f1, #ffffff);
    }
    .snapshot-card__top {
        display: flex; justify-content: space-between; align-items: center;
    }
    .snapshot-card__label {font-weight: 700; color: #303642;}
    .snapshot-card__value {font-size: 1.25rem; font-weight: 750; color: #e65f3c;}
    .snapshot-card__track {
        height: 0.45rem; background: #f5e4dc; border-radius: 99px;
        overflow: hidden; margin-top: 0.65rem;
    }
    .snapshot-card__fill {
        height: 100%; background: linear-gradient(90deg, #f28a5c, #e55438);
        border-radius: 99px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.75rem 0 1rem;
    }
    .metric-tile {
        min-width: 0;
        padding: 0.78rem 0.82rem;
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 0.9rem;
        background: linear-gradient(145deg, #fffaf7, #ffffff);
    }
    .metric-tile__top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.55rem;
    }
    .metric-tile__name {
        color: #343741;
        font-size: 0.83rem;
        font-weight: 680;
        line-height: 1.25;
    }
    .metric-tile__value {
        color: #e65f3c;
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1;
        white-space: nowrap;
    }
    .metric-tile__track {
        height: 0.3rem;
        margin-top: 0.65rem;
        overflow: hidden;
        border-radius: 99px;
        background: #f5e4dc;
    }
    .metric-tile__fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #f28a5c, #e55438);
    }
    @media (max-width: 430px) {
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        .metric-grid {gap: 0.5rem;}
        .metric-tile {padding: 0.7rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

LANDING_COPY = {
    "English": {
        "title": "Know your speech patterns over time",
        "subtitle": (
            "Speak naturally. KinaBot helps you reflect on your own speech patterns "
            "over time—with dignity, privacy, and family connection at the center."
        ),
        "language": "Choose your language",
        "start": "Start",
        "login_caption": "Enter your email to keep your reflections together.",
        "email": "Email",
        "send_code": "Send code",
        "code": "6-digit code",
        "continue": "Continue",
        "invalid_email": "Enter a valid email address.",
        "email_unavailable": "Email delivery is unavailable. Please try again later.",
        "invalid_code": "Invalid or expired code.",
        "disclaimer": (
            "KinaBot supports healthy-aging reflection and family conversations. "
            "It is not a medical device or diagnostic tool."
        ),
    },
    "日本語": {
        "title": "話し方のパターンを長期的に振り返る",
        "subtitle": (
            "いつもの言葉で話すだけ。KinaBotがあなた自身の基準をつくり、"
            "尊厳とプライバシー、家族とのつながりを大切にしながら、"
            "話し方のパターンを長期的に振り返ります。"
        ),
        "language": "表示言語を選択",
        "start": "はじめる",
        "login_caption": "メールアドレスで振り返りの記録をまとめます。",
        "email": "メールアドレス",
        "send_code": "認証コードを送信",
        "code": "6桁の認証コード",
        "continue": "次へ",
        "invalid_email": "有効なメールアドレスを入力してください。",
        "email_unavailable": "現在メールを送信できません。後でもう一度お試しください。",
        "invalid_code": "認証コードが無効か、有効期限が切れています。",
        "disclaimer": (
            "KinaBotは健康的な加齢の振り返りと家族との対話を支えます。"
            "医療機器や診断ツールではありません。"
        ),
    },
    "中文": {
        "title": "长期了解自己的语言表达模式",
        "subtitle": (
            "只需用最自然的语言说话。KinaBot建立你的个人基线，帮助你长期了解"
            "自己的语言表达模式，并把尊严、隐私与家庭联结放在中心。"
        ),
        "language": "选择界面语言",
        "start": "开始",
        "login_caption": "输入邮箱，让每次记录连续保存。",
        "email": "邮箱",
        "send_code": "发送验证码",
        "code": "6位验证码",
        "continue": "继续",
        "invalid_email": "请输入有效的邮箱地址。",
        "email_unavailable": "暂时无法发送邮件，请稍后重试。",
        "invalid_code": "验证码无效或已过期。",
        "disclaimer": (
            "KinaBot支持健康老龄化反思与家庭沟通；"
            "它不是医疗器械或诊断工具。"
        ),
    },
}

AUDIO_CAPTURE_COPY = {
    "English": {
        "new": "New reflection",
        "language": "1 · Choose the language spoken",
        "language_help": "Choose the language you will speak in this recording.",
        "method": "2 · Add your voice sample",
        "record": "Record here",
        "upload": "Upload a recording",
        "record_label": "Start recording",
        "record_help": "Allow microphone access, then record 30–90 seconds.",
        "upload_label": "Choose an audio file",
        "upload_help": "Supported formats: WAV, MP3, M4A, AAC, OGG, and FLAC.",
        "fallback": "Microphone unavailable? Choose Upload a recording instead.",
    },
    "日本語": {
        "new": "新しい振り返り",
        "language": "1 · 話す言語を選択",
        "language_help": "今回の録音で話す言語を選択してください。",
        "method": "2 · 音声を追加",
        "record": "ここで録音",
        "upload": "録音ファイルをアップロード",
        "record_label": "録音を開始",
        "record_help": "マイクへのアクセスを許可し、30～90秒録音してください。",
        "upload_label": "音声ファイルを選択",
        "upload_help": "対応形式：WAV、MP3、M4A、AAC、OGG、FLAC。",
        "fallback": "マイクが使えない場合は、録音ファイルのアップロードを選択してください。",
    },
    "中文": {
        "new": "新的语音记录",
        "language": "1 · 选择本次使用的语言",
        "language_help": "请选择这次录音中使用的语言。",
        "method": "2 · 添加语音",
        "record": "在这里直接录音",
        "upload": "上传已有录音",
        "record_label": "开始录音",
        "record_help": "允许使用麦克风，然后录制30至90秒。",
        "upload_label": "选择音频文件",
        "upload_help": "支持WAV、MP3、M4A、AAC、OGG和FLAC。",
        "fallback": "无法使用麦克风？请改选上传已有录音。",
    },
}

HISTORY_COPY = {
    "English": {
        "today": "Today",
        "trends": "Trends",
        "latest": "Latest 8-feature snapshot",
        "no_scores": "No saved scores yet.",
        "progress": "{count} of 3 sessions completed. Trends begin after session 3.",
        "recent": "Recent sessions",
        "all": "All sessions",
        "change": "Observed change since the first sample",
        "higher": "Higher in latest sample",
        "lower": "Lower in latest sample",
        "similar": "Similar",
        "method": "How the 8 features are calculated",
        "method_intro": (
            "KinaBot calculates descriptive 0–100 feature indexes with its own "
            "Python and multilingual NLP pipeline. They are not percentages, "
            "population rankings, or health scores."
        ),
        "boundary": (
            "These are descriptive sample-to-sample differences only. KinaBot does "
            "not infer health, improvement, decline, cause, or risk."
        ),
    },
    "日本語": {
        "today": "今日",
        "trends": "トレンド",
        "latest": "最新の8項目",
        "no_scores": "保存されたスコアはまだありません。",
        "progress": "3回中{count}回完了しました。3回目からトレンドを表示します。",
        "recent": "最近のセッション",
        "all": "すべてのセッション",
        "change": "最初のサンプルからの変化",
        "higher": "最新サンプルで高い",
        "lower": "最新サンプルで低い",
        "similar": "ほぼ同じ",
        "method": "8項目の計算方法",
        "method_intro": (
            "KinaBot独自のPythonと多言語NLPにより、0〜100の記述的な特徴指数を"
            "計算します。割合、集団順位、健康スコアではありません。"
        ),
        "boundary": (
            "サンプル間の記述的な差だけを示します。健康、改善、低下、原因、"
            "リスクを推定するものではありません。"
        ),
    },
    "中文": {
        "today": "今天",
        "trends": "趋势",
        "latest": "最近一次的8项指标",
        "no_scores": "目前还没有保存的分数。",
        "progress": "已完成3次中的{count}次，第3次开始显示趋势。",
        "recent": "最近记录",
        "all": "全部记录",
        "change": "与第一次样本相比的变化",
        "higher": "最近一次较高",
        "lower": "最近一次较低",
        "similar": "基本相近",
        "method": "8项指标如何计算",
        "method_intro": (
            "KinaBot使用自己的Python与多语言NLP流程计算0–100的描述性特征指数。"
            "它们不是百分比、人群排名或健康评分。"
        ),
        "boundary": (
            "这里只描述不同语音样本之间的差异。KinaBot不推断健康、改善、下降、"
            "原因或风险。"
        ),
    },
}

CHALLENGE_COPY = {
    "English": {
        "title": "30 Days to Know Your Patterns",
        "subtitle": "One 60-second reflection when it works for you. Extra check-ins are optional.",
        "day": "Day {day} of 30",
        "progress_label": "Challenge progress",
        "reflection_days": "Reflection days",
        "today_ready": "Today's reflection is ready when you are.",
        "today_complete": "Today's reflection is complete. Another check-in is optional.",
        "foundation": "Your first 30-day period is complete. Continue whenever reflection is useful.",
        "available": "{remaining} optional check-ins still available today",
        "limit": "You have reached today's optional check-in limit. Your daily reflection is complete.",
    },
    "日本語": {
        "title": "30日間で自分のパターンを知る",
        "subtitle": "都合のよい時に60秒の振り返りを1回。追加の記録は任意です。",
        "day": "30日中 {day} 日目",
        "progress_label": "30日間の進捗",
        "reflection_days": "記録した日数",
        "today_ready": "今日の振り返りは、できる時に行いましょう。",
        "today_complete": "今日の振り返りは完了しました。追加の記録は任意です。",
        "foundation": "最初の30日間が完了しました。必要な時に続けてください。",
        "available": "本日あと{remaining}回、任意で追加できます",
        "limit": "本日の任意追加回数に達しました。今日の振り返りは完了です。",
    },
    "中文": {
        "title": "用30天了解自己的表达模式",
        "subtitle": "方便时完成一次60秒记录；额外记录完全自愿。",
        "day": "30天中的第{day}天",
        "progress_label": "30天进度",
        "reflection_days": "完成记录天数",
        "today_ready": "方便时完成今天的一次记录即可。",
        "today_complete": "今天的一次记录已完成；额外记录完全自愿。",
        "foundation": "第一个30天周期已完成。今后可在需要时继续。",
        "available": "今天还可自愿增加{remaining}次记录",
        "limit": "今天的自愿追加次数已用完；今日记录已经完成。",
    },
}


if "ui_language" not in st.session_state:
    st.session_state.ui_language = "English"

copy = LANDING_COPY[st.session_state.ui_language]
st.markdown(
    f"""
    <section class="kinabot-hero">
      <div class="kinabot-hero__brand">KinaBot</div>
      <div class="kinabot-hero__title">{copy['title']}</div>
      <div class="kinabot-hero__subtitle">{copy['subtitle']}</div>
    </section>
    <div class="kinabot-language-label">{copy['language']}</div>
    """,
    unsafe_allow_html=True,
)
st.radio(
    "Language / 言語 / 语言",
    ["English", "日本語", "中文"],
    horizontal=True,
    key="ui_language",
    label_visibility="collapsed",
)
copy = LANDING_COPY[st.session_state.ui_language]

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
    st.subheader(copy["start"])
    st.caption(copy["login_caption"])
    email = st.text_input(copy["email"], value=st.session_state.email)
    if not st.session_state.code_sent:
        send_code = st.button(
            copy["send_code"], type="primary", use_container_width=True
        )
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
            st.error(copy["invalid_email"])
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
                st.error(copy["email_unavailable"])
            st.rerun()

    if st.session_state.code_sent:
        if st.session_state.staging_code:
            st.info(f"Private staging code: {st.session_state.staging_code}")
        code = st.text_input(copy["code"], max_chars=6)
        if st.button(copy["continue"], type="primary", use_container_width=True):
            email_hash = verify_code(st.session_state.email, code)
            if not email_hash:
                st.error(copy["invalid_code"])
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

    st.caption(copy["disclaimer"])
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

history_copy = HISTORY_COPY[st.session_state.ui_language]
challenge_copy = CHALLENGE_COPY[st.session_state.ui_language]
primary_view = st.radio(
    "KinaBot navigation",
    ["today", "trends"],
    format_func=lambda option: history_copy[option],
    horizontal=True,
    label_visibility="collapsed",
    key="primary_view",
)

if primary_view == "trends":
    assign_timezone_to_legacy_sessions(st.session_state.user_id, browser_timezone)
    rows = get_user_scores(st.session_state.user_id)
    st.subheader(history_copy["trends"])
    if not rows:
        st.caption(history_copy["no_scores"])
        st.stop()

    history = pd.DataFrame([dict(row) for row in rows])
    session_count = int(history["session_id"].nunique())
    st.markdown(f"### {history_copy['latest']}")
    st.markdown(
        metric_grid_html(
            latest_session_scores(history),
            st.session_state.ui_language,
        ),
        unsafe_allow_html=True,
    )

    with st.expander(history_copy["method"]):
        st.write(history_copy["method_intro"])
        for feature_name in history["feature_name"].drop_duplicates():
            label = display_feature_name(feature_name, st.session_state.ui_language)
            explanation = feature_explanation(
                feature_name, st.session_state.ui_language
            )
            st.markdown(f"**{label}** — {explanation}")
        st.link_button(
            "Open scoring methodology",
            "https://github.com/usekina/kina/blob/main/aoi_kinabot_app/SCORING-METHODOLOGY.md",
            use_container_width=True,
        )

    if session_count < 3:
        st.info(history_copy["progress"].format(count=session_count))
        st.stop()

    feature_names = list(history["feature_name"].drop_duplicates())
    selected_feature = st.selectbox(
        history_copy["recent"],
        feature_names,
        format_func=lambda name: display_feature_name(
            name, st.session_state.ui_language
        ),
    )
    history_scope = st.radio(
        "History range",
        ["recent", "all"],
        format_func=lambda option: history_copy[option],
        horizontal=True,
        label_visibility="collapsed",
    )
    feature_history = history[history["feature_name"] == selected_feature].sort_values(
        "session_id"
    )
    if history_scope == "recent":
        feature_history = feature_history.tail(3)
    feature_history = feature_history.copy()
    feature_history["session_label"] = (
        feature_history["session_date"].astype(str)
        + " #"
        + feature_history["session_number"].astype(str)
    )
    chart_df = feature_history[["session_label", "score"]].set_index("session_label")
    chart_df = chart_df.rename(
        columns={
            "score": display_feature_name(
                selected_feature, st.session_state.ui_language
            )
        }
    )
    st.line_chart(chart_df, height=260)

    ordered = history.sort_values("session_id")
    first_score = float(
        ordered[ordered["feature_name"] == selected_feature].iloc[0]["score"]
    )
    latest_score = float(
        ordered[ordered["feature_name"] == selected_feature].iloc[-1]["score"]
    )
    observed_change = latest_score - first_score
    pattern = (
        history_copy["higher"]
        if observed_change > 2
        else history_copy["lower"]
        if observed_change < -2
        else history_copy["similar"]
    )
    st.markdown(f"**{history_copy['change']}**")
    st.write(f"{pattern} ({observed_change:+.1f})")
    st.caption(history_copy["boundary"])

    insight = generate_wellness_insight([dict(row) for row in rows], st.session_state.ui_language)
    st.markdown("#### One small action")
    if insight.get("encouragement"):
        st.write(insight["encouragement"])
    st.info(insight["action"])
    st.caption(f"{insight['why']} [Research source]({insight['source']})")
    st.caption(insight["boundary"])
    st.stop()

assign_timezone_to_legacy_sessions(st.session_state.user_id, browser_timezone)
challenge_rows = get_user_scores(st.session_state.user_id)
challenge_session_dates = list(
    {
        int(row["session_id"]): str(row["session_date"])
        for row in challenge_rows
    }.values()
)
challenge = challenge_status(challenge_session_dates, date.fromisoformat(today))
st.markdown(f"### {challenge_copy['title']}")
st.caption(challenge_copy["subtitle"])
if challenge["challenge_complete"]:
    st.success(challenge_copy["foundation"])
else:
    st.progress(challenge["day"] / CHALLENGE_DAYS)
    challenge_col_1, challenge_col_2 = st.columns(2)
    challenge_col_1.metric(
        challenge_copy["progress_label"],
        challenge_copy["day"].format(day=challenge["day"]),
    )
    challenge_col_2.metric(
        challenge_copy["reflection_days"],
        challenge["reflection_days"],
    )
if challenge["complete_today"]:
    st.success(challenge_copy["today_complete"])
else:
    st.info(challenge_copy["today_ready"])

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

assign_timezone_to_legacy_sessions(st.session_state.user_id, browser_timezone)
tests_today = count_tests_today(st.session_state.user_id, today)
remaining = MAX_TESTS_PER_DAY - tests_today
if remaining <= 0:
    st.info(challenge_copy["limit"])
    st.stop()

capture_copy = AUDIO_CAPTURE_COPY[st.session_state.ui_language]
st.subheader(capture_copy["new"])

if "recording_language" not in st.session_state:
    st.session_state.recording_language = st.session_state.ui_language

language = st.radio(
    capture_copy["language"],
    ["English", "日本語", "中文"],
    horizontal=True,
    help=capture_copy["language_help"],
    key="recording_language",
)

session_type = "Daily reflection"
st.markdown(f"**{capture_copy['method']}**")
audio_method = st.radio(
    capture_copy["method"],
    ["upload", "record"],
    format_func=lambda option: capture_copy[option],
    horizontal=True,
    label_visibility="collapsed",
)
if audio_method == "record":
    selected_audio = st.audio_input(
        capture_copy["record_label"],
        help=capture_copy["record_help"],
    )
    st.caption(capture_copy["fallback"])
else:
    selected_audio = st.file_uploader(
        capture_copy["upload_label"],
        type=SUPPORTED_AUDIO_TYPES,
        help=capture_copy["upload_help"],
    )

if selected_audio is not None:
    st.audio(selected_audio)
    st.caption(
        f"Ready: {selected_audio.name} "
        f"({selected_audio.size / 1024:.1f} KB). Raw audio will not be stored."
    )
    audio_extension = selected_audio.name.rsplit(".", 1)[-1].lower()
    can_transcribe = audio_extension in LOCAL_TRANSCRIPTION_TYPES
    if not can_transcribe:
        st.info(
            "Automatic transcription supports MP3, MP4, MPEG, MPGA, M4A, WAV, and WEBM."
        )
if tests_today == 0:
    st.caption(challenge_copy["today_ready"])
else:
    st.caption(challenge_copy["available"].format(remaining=max(0, remaining)))
if st.button("3 · Analyze my reflection", type="primary", use_container_width=True):
    if selected_audio is None:
        st.warning("Upload a speech sample first.")
    elif selected_audio.size > MAX_AUDIO_BYTES:
        st.warning(f"Audio must be {MAX_AUDIO_BYTES // (1024 * 1024)} MB or smaller.")
    elif selected_audio.name.rsplit(".", 1)[-1].lower() not in LOCAL_TRANSCRIPTION_TYPES:
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
                selected_audio,
                selected_audio.name,
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
            audio_metadata = accept_audio_upload(selected_audio, selected_audio.name)
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
                timezone_name=browser_timezone,
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
        snapshot = build_reflection_profile(scores, language)
        st.markdown(f"### {snapshot['title']}")
        st.caption(snapshot["subtitle"])
        snapshot_columns = st.columns(2)
        for index, dimension in enumerate(snapshot["dimensions"]):
            score = int(dimension["score"])
            with snapshot_columns[index % 2]:
                st.markdown(
                    f"""
                    <div class="snapshot-card">
                      <div class="snapshot-card__top">
                        <span class="snapshot-card__label">{dimension["label"]}</span>
                        <span class="snapshot-card__value">{score}</span>
                      </div>
                      <div class="snapshot-card__track">
                        <div class="snapshot-card__fill" style="width:{score}%"></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown(f"#### {snapshot['takeaway_title']}")
        st.write(snapshot["takeaway"])
        st.markdown(f"#### {snapshot['action_title']}")
        st.info(snapshot["action"])
        st.caption(result_copy["scale"])
        st.markdown(
            metric_grid_html(scores, language),
            unsafe_allow_html=True,
        )
        with st.expander(snapshot["detail_label"]):
            for item in scores:
                label = display_feature_name(item["feature_name"], language)
                st.markdown(f"**{label}** — {item['explanation']}")
        st.caption(result_copy["boundary"])

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
