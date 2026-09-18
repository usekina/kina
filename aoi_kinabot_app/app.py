"""Aoi-maintained KinaBot V1 local skeleton app."""

from __future__ import annotations

from datetime import date
import json
import uuid
import pandas as pd
import streamlit as st

from audio_processing import SUPPORTED_AUDIO_TYPES, accept_audio_upload
from auth import create_local_verification_code, verify_code
from challenge_progress import CHALLENGE_DAYS, challenge_status
from config import (
    ADMIN_KEY,
    ADMIN_EMAIL,
    ALLOW_LOCAL_VERIFICATION_CODES,
    APP_VERSION,
    ANALYSIS_PIPELINE_ID,
    CONSENT_VERSION,
    MAX_AUDIO_BYTES,
    MAX_TESTS_PER_DAY,
    OFFLINE_RESEARCH_MODE,
    PARTICIPANT_KEY_SECRET,
    SCORING_MODEL_VERSION,
)
from database import (
    assign_timezone_to_legacy_sessions,
    count_tests_today,
    complete_test_session,
    DailyLimitReached,
    create_test_session,
    get_admin_metrics,
    get_user_habit_checkins,
    get_user_profile,
    get_user_scores,
    has_active_consent,
    delete_user_research_data,
    export_user_data,
    init_db,
    list_admin_test_records,
    list_admin_users,
    list_research_records,
    record_consent,
    save_feature_scores,
    save_habit_checkins,
    update_user_profile,
    withdraw_research_consent,
    upsert_user,
)
from email_delivery import send_verification_email
from history_view import (
    latest_session_scores,
    metric_grid_html,
    select_latest_comparable_history,
)
from insight_service import generate_wellness_insight
from language_analysis import LANGUAGE_CODES, analyze_transcript
from local_time import local_date_iso
from offline_identity import normalize_participant_id, participant_key, valid_participant_id
from pilot_report import build_personal_pdf_report
from speech_to_text import (
    LOCAL_TRANSCRIPTION_TYPES,
    speech_to_text_configured,
    transcribe_audio_upload,
)
from scoring import display_feature_name, feature_explanation
from clarity_ui import inject_theme, ui_copy, render_result, recording_prompt
from radar_view import measured_score
from admin_view import is_owner, render_admin, verify_owner_key
from wellness_guidance import wellness_suggestions


st.set_page_config(page_title="KinaBot", page_icon="🎙️", layout="wide")
init_db()
browser_timezone = st.context.timezone or "UTC"
today = local_date_iso(browser_timezone)

inject_theme()

LANDING_COPY = {
    "English": {
        "eyebrow": "A private moment to reflect",
        "title": "Your Voice, Your Patterns, Over Time",
        "subtitle": (
            "Record a short reflection. KinaBot turns speech patterns into clear, "
            "personal trends—without diagnosing, ranking, or comparing you with anyone else."
        ),
        "trust_privacy": "✓ Privacy-first",
        "trust_history": "✓ Your own history only",
        "trust_wellness": "✓ Wellness, not diagnosis",
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
        "eyebrow": "自分と向き合う、プライベートなひととき",
        "title": "あなたの声、あなたのパターン、時間とともに",
        "subtitle": (
            "短い振り返りを録音すると、KinaBotが話し方の特徴を分かりやすい個人の"
            "傾向として示します。診断・順位付け・他者との比較は行いません。"
        ),
        "trust_privacy": "✓ プライバシーを優先",
        "trust_history": "✓ 自分自身の履歴のみ",
        "trust_wellness": "✓ 診断ではなくウェルネス",
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
        "eyebrow": "留给自己的一段私密反思时间",
        "title": "你的声音，你的模式，随时间变化",
        "subtitle": (
            "录制一段简短反思。KinaBot把语言特征转化为清晰的个人趋势，"
            "不进行诊断、排名，也不与他人比较。"
        ),
        "trust_privacy": "✓ 隐私优先",
        "trust_history": "✓ 只比较自己的历史",
        "trust_wellness": "✓ 关注身心状态，而非诊断",
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

LANDING_STEPS = {
    "English": [
        ("Speak naturally", "Share a short reflection in English, 日本語, or 中文."),
        ("See clear signals", "Review eight understandable speech and language features."),
        ("Follow your pattern", "Compare only with your own compatible past sessions."),
    ],
    "日本語": [
        ("自然に話す", "英語・日本語・中国語で短い振り返りを話します。"),
        ("特徴を分かりやすく見る", "8つの発話と言語の特徴を確認します。"),
        ("自分のパターンを追う", "互換性のある自分自身の過去記録とのみ比較します。"),
    ],
    "中文": [
        ("自然表达", "使用英语、日语或中文完成一段简短反思。"),
        ("查看清晰指标", "了解八项易于理解的语言与语音特征。"),
        ("关注自己的变化", "只与评分兼容的个人历史记录进行比较。"),
    ],
}

OFFLINE_LOGIN_COPY = {
    "English": {
        "caption": "Offline research mode: enter the participant ID assigned by the study administrator.",
        "label": "Participant ID",
        "continue": "Continue offline",
        "invalid": "Use 3-32 letters, numbers, underscores, or hyphens.",
    },
    "日本語": {
        "caption": "オフライン研究モード：研究担当者から割り当てられた参加者IDを入力してください。",
        "label": "参加者ID",
        "continue": "オフラインで続ける",
        "invalid": "3〜32文字の英数字、アンダースコア、またはハイフンを使用してください。",
    },
    "中文": {
        "caption": "离线研究模式：请输入研究管理员分配的参与者编号。",
        "label": "参与者编号",
        "continue": "离线继续",
        "invalid": "请使用3至32位字母、数字、下划线或连字符。",
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
        "comparability_note": (
            "Earlier recordings remain visible, but trends compare only sessions with the same "
            "language and scoring method."
        ),
        "change": "Observed change since the first sample",
        "higher": "Higher in latest sample",
        "lower": "Lower in latest sample",
        "similar": "Similar",
        "small_action": "One small action",
        "research_summary": "What the research says",
        "research_source": "Read the research source",
        "insight_role": (
            "KinaBot helps you understand the insight and turn it into one simple action. "
            "It does not provide medical advice."
        ),
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
        "comparability_note": (
            "以前の記録も表示しますが、トレンドでは同じ言語・採点方式の記録だけを比較します。"
        ),
        "change": "最初のサンプルからの変化",
        "higher": "最新サンプルで高い",
        "lower": "最新サンプルで低い",
        "similar": "ほぼ同じ",
        "small_action": "小さな行動を一つ",
        "research_summary": "研究の要点",
        "research_source": "研究資料を読む",
        "insight_role": (
            "KinaBotは、気づきを理解し、無理のない一つの行動につなげるお手伝いをします。"
            "医療上の助言ではありません。"
        ),
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
        "comparability_note": (
            "历史记录仍会保留；趋势只比较语言和评分方式相同的记录。"
        ),
        "change": "与第一次样本相比的变化",
        "higher": "最近一次较高",
        "lower": "最近一次较低",
        "similar": "基本相近",
        "small_action": "一个简单行动",
        "research_summary": "研究简介",
        "research_source": "查看研究来源",
        "insight_role": (
            "KinaBot帮助你理解这些信息，并把它转化为一个容易实践的小行动；这不是医疗建议。"
        ),
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
st.markdown('<header class="kinabot-topbar"><div class="kinabot-topbar__brand"><span class="kinabot-topbar__mark">◉</span>KinaBot</div></header>', unsafe_allow_html=True)
if st.session_state.get("verified", False):
    with st.sidebar:
        st.markdown("## KinaBot")
        st.selectbox("Language / 言語 / 语言", ["English", "日本語", "中文"], key="ui_language")
else:
    st.radio("Language / 言語 / 语言", ["English", "日本語", "中文"], horizontal=True, key="ui_language")
    st.title(ui_copy("start", st.session_state.ui_language))
    st.caption(ui_copy("intro", st.session_state.ui_language))
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
    if OFFLINE_RESEARCH_MODE:
        offline_copy = OFFLINE_LOGIN_COPY[st.session_state.ui_language]
        st.info("🔒 Offline research mode · no email · no cloud AI")
        st.caption(offline_copy["caption"])
        participant_id = st.text_input(offline_copy["label"])
        if st.button(offline_copy["continue"], type="primary", use_container_width=True):
            if not valid_participant_id(participant_id):
                st.error(offline_copy["invalid"])
            else:
                normalized_id = normalize_participant_id(participant_id)
                if len(PARTICIPANT_KEY_SECRET) < 32:
                    st.error(
                        "Offline participant-key secret is missing. Ask the study "
                        "administrator to run install-offline.ps1."
                    )
                    st.stop()
                pseudonymous_key = participant_key(
                    normalized_id, PARTICIPANT_KEY_SECRET
                )
                user_id = upsert_user(pseudonymous_key, email=None)
                profile = get_user_profile(user_id)
                if not profile or not profile["display_name"]:
                    update_user_profile(
                        user_id,
                        "Offline participant",
                        "Prefer not to say",
                        "Prefer not to say",
                        None,
                        None,
                    )
                st.session_state.email = f"Participant {normalized_id}"
                st.session_state.email_hash = pseudonymous_key
                st.session_state.user_id = user_id
                st.session_state.profile = dict(get_user_profile(user_id))
                st.session_state.verified = True
                st.rerun()
        st.caption(copy["disclaimer"])
        st.stop()

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
        owner_login = bool(ADMIN_EMAIL and st.session_state.email.strip().casefold() == ADMIN_EMAIL)
        owner_login_key = st.text_input("Admin key", type="password", key="owner_login_key") if owner_login else ""
        code = st.text_input(copy["code"], max_chars=6)
        if st.button(copy["continue"], type="primary", use_container_width=True):
            if owner_login and not verify_owner_key(st.session_state.email, ADMIN_EMAIL, owner_login_key, ADMIN_KEY):
                st.error("An administrator key is required for this account.")
                st.stop()
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
                st.session_state.admin_identity_email = ADMIN_EMAIL if owner_login else ""
                st.session_state.pop("owner_login_key", None)
                st.session_state.staging_code = ""
                st.rerun()

    st.caption(copy["disclaimer"])
    st.stop()

if "pending_primary_view" in st.session_state:
    st.session_state.primary_view = st.session_state.pop("pending_primary_view")
navigation_labels = {
    "today": ui_copy("record", st.session_state.ui_language),
    "results": ui_copy("results", st.session_state.ui_language),
    "trends": ui_copy("history", st.session_state.ui_language),
}
owner = is_owner(st.session_state.verified, st.session_state.email, ADMIN_EMAIL,
                 offline=OFFLINE_RESEARCH_MODE,
                 owner_key_verified=st.session_state.get("admin_identity_email") == ADMIN_EMAIL)
if owner and ADMIN_KEY:
    navigation_labels["admin"] = "Admin"
if st.session_state.get("primary_view") not in navigation_labels:
    st.session_state.primary_view = "today"
primary_view = st.sidebar.radio(
    "KinaBot", list(navigation_labels),
    format_func=lambda key: navigation_labels[key],
    key="primary_view", label_visibility="collapsed",
)

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

with st.sidebar.expander(
    ui_copy("account", st.session_state.ui_language),
    expanded=False,
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
        update_user_profile(
            st.session_state.user_id,
            display_name.strip() or None,
            None if age_range in (None, "Prefer not to say") else age_range,
            None if gender in (None, "Prefer not to say") else gender,
            None if primary_language == "Prefer not to say" else primary_language,
            country_region.strip() or None,
        )
        refreshed_profile = get_user_profile(st.session_state.user_id)
        st.session_state.profile = dict(refreshed_profile) if refreshed_profile else {}
        st.success("Account saved.")
        st.rerun()

with st.expander("Manage my data"):
    st.caption("View, export, correct, withdraw, or delete your KinaBot data.")
    export_payload = export_user_data(st.session_state.user_id)
    st.download_button(
        "Download my data",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="kinabot_my_data.json",
        mime="application/json",
        use_container_width=True,
    )
    personal_report = build_personal_pdf_report(
        st.session_state.get("profile") or {},
        get_user_scores(st.session_state.user_id),
    )
    st.download_button(
        "Download my PDF report",
        data=personal_report,
        file_name="kinabot_research_pilot_report.pdf",
        mime="application/pdf",
        help="A basic report of your own scores. It excludes raw audio and internal model data.",
        use_container_width=True,
    )
    if st.button("Withdraw from research"):
        withdraw_research_consent(st.session_state.user_id)
        st.session_state["research_pilot_consent"] = False
        st.warning("Future research collection is stopped. You must re-consent to rejoin.")
        st.stop()
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.caption("Account deletion permanently removes your profile, sessions, scores, habits, consent records, and verification records.")
    delete_confirm = st.checkbox("I understand this cannot be undone.")
    if st.button("Delete account and history", disabled=not delete_confirm):
        delete_user_research_data(st.session_state.user_id)
        st.session_state.clear()
        st.success("Your account and stored data have been deleted.")
        st.stop()

history_copy = HISTORY_COPY[st.session_state.ui_language]
challenge_copy = CHALLENGE_COPY[st.session_state.ui_language]
if primary_view == "admin":
    render_admin(owner=owner, admin_key=ADMIN_KEY)
    st.stop()

if primary_view == "results":
    assign_timezone_to_legacy_sessions(st.session_state.user_id, browser_timezone)
    render_result([dict(row) for row in get_user_scores(st.session_state.user_id)], st.session_state.ui_language)
    st.stop()

if primary_view == "trends":
    assign_timezone_to_legacy_sessions(st.session_state.user_id, browser_timezone)
    rows = get_user_scores(st.session_state.user_id)
    st.title(ui_copy("history", st.session_state.ui_language))
    st.caption(ui_copy("history_intro", st.session_state.ui_language))
    if not rows:
        st.caption(history_copy["no_scores"])
        st.stop()

    history = pd.DataFrame([
        dict(row) for row in rows
        if measured_score(dict(row)) is not None
    ])
    if history.empty:
        st.info("No measured features are available for comparable trends yet.")
        st.stop()
    session_count = int(history["session_id"].nunique())
    has_mixed_languages = history["language"].dropna().nunique() > 1
    has_mixed_versions = history["scoring_model_version"].dropna().nunique() > 1
    if has_mixed_languages or has_mixed_versions:
        st.info(history_copy["comparability_note"])
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

    comparable_history, comparison_key = select_latest_comparable_history(history)
    comparable_count = int(comparable_history["session_id"].nunique())
    if comparison_key is None:
        st.info(
            "Not enough comparable sessions for a trend yet. Complete 3 sessions "
            "using the same spoken language and analysis version."
        )
        st.stop()
    st.caption(
        "Comparison set: "
        f"{comparison_key[0]} · scoring {comparison_key[1]} · pipeline {comparison_key[2]} "
        f"({comparable_count} sessions). Other history is preserved but not compared."
    )

    feature_names = list(comparable_history["feature_name"].drop_duplicates())
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
    feature_history = comparable_history[
        comparable_history["feature_name"] == selected_feature
    ].sort_values("session_id")
    if history_scope == "recent":
        feature_history = feature_history.tail(3)
    feature_history = feature_history.copy()
    feature_history["session_label"] = (
        feature_history["session_date"].astype(str)
        + " · "
        + feature_history["language"].fillna("Unknown").astype(str)
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

    ordered = comparable_history.sort_values("session_id")
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

    insight = generate_wellness_insight(
        comparable_history.to_dict("records"), st.session_state.ui_language
    )
    st.markdown(f"#### {history_copy['small_action']}")
    st.info(insight["action"])
    st.caption(insight["why"])
    st.markdown(f"**{history_copy['research_summary']}**")
    st.caption(insight["research_summary"])
    st.markdown(f"[{history_copy['research_source']}]({insight['source']})")
    st.caption(history_copy["insight_role"])
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
recording_prompt(st.session_state.ui_language)
st.caption(challenge_copy["today_complete"] if challenge["complete_today"] else challenge_copy["today_ready"])

with st.expander("Read Research Notice", expanded=False):
    st.markdown(
        """
        **AImoji LLC** provides KinaBot as a long-term research pilot. You may continue
        using the pilot until the pilot ends or you withdraw your consent.

        KinaBot helps you view and understand your own voice-derived cognitive-wellness
        results. With your consent, AImoji LLC may analyze pseudonymized voice-derived
        scores, usage trends, and demographic information that you voluntarily provide
        for the research purposes described here.

        Gender, age range, location, and first language are optional. Data is not used
        for commercial sales or targeted advertising. AImoji LLC may publish aggregated,
        non-identifying findings in academic papers, research reports, or presentations.
        Publications will not include your name, email address, raw voice recordings,
        or directly identifying information.

        Joining is voluntary. You may decline or withdraw without penalty; the standard
        non-research version is not currently available.

        Withdrawal stops future research collection. Data already included in completed
        or published aggregate results may not be removable. Contact the study
        administrator if you have questions or want to withdraw.
        """
    )

consent = st.checkbox(
    "I have had an opportunity to review the Research Notice and agree to join the KinaBot Research Pilot.",
    value=has_active_consent(st.session_state.user_id, CONSENT_VERSION),
    key="research_pilot_consent",
)

if not consent:
    st.caption("This free version is only available to research-pilot participants. Decline and exit to leave.")
    st.stop()

record_consent(st.session_state.user_id, CONSENT_VERSION)

assign_timezone_to_legacy_sessions(st.session_state.user_id, browser_timezone)
tests_today = count_tests_today(st.session_state.user_id, today)
remaining = MAX_TESTS_PER_DAY - tests_today
if remaining <= 0:
    st.info(challenge_copy["limit"])
    st.stop()

capture_copy = AUDIO_CAPTURE_COPY[st.session_state.ui_language]
st.markdown(
    f"""
    <div class="reflection-panel-head">
      <div class="reflection-panel-head__title">{capture_copy['new']}</div>
      <div class="reflection-panel-head__private">🔒 Private processing</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    ["record", "upload"],
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
if st.button(ui_copy("analyze", st.session_state.ui_language), type="primary", use_container_width=True, disabled=selected_audio is None):
    if selected_audio is None:
        st.warning("Upload a speech sample first.")
    elif selected_audio.size > MAX_AUDIO_BYTES:
        st.warning(f"Audio must be {MAX_AUDIO_BYTES // (1024 * 1024)} MB or smaller.")
    elif selected_audio.name.rsplit(".", 1)[-1].lower() not in LOCAL_TRANSCRIPTION_TYPES:
        st.warning("Use MP3, MP4, MPEG, MPGA, M4A, WAV, or WEBM for automatic analysis.")
    else:
        request_key = st.session_state.setdefault(
            "pending_analysis_request_id", uuid.uuid4().hex
        )
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
            try:
                test_session_id, session_number, already_saved = complete_test_session(
                    user_id=st.session_state.user_id,
                    session_date=today,
                    app_version=APP_VERSION,
                    consent_version=CONSENT_VERSION,
                    scoring_model_version=SCORING_MODEL_VERSION,
                    analysis_pipeline_id=ANALYSIS_PIPELINE_ID,
                    scores=scores,
                    max_tests_per_day=MAX_TESTS_PER_DAY,
                    session_type=session_type,
                    language=language,
                    duration_seconds=detected_duration or audio_metadata["duration_seconds"],
                    timezone_name=browser_timezone,
                    idempotency_key=request_key,
                )
            except DailyLimitReached as exc:
                analysis_status.update(label="Daily limit reached", state="error")
                st.warning(str(exc))
                st.stop()
            if already_saved:
                st.info("This analysis was already saved; showing the existing result.")
            st.session_state.pop("pending_analysis_request_id", None)
            analysis_status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state.pending_primary_view = "results"
        st.rerun()

with st.expander(ui_copy("habit", st.session_state.ui_language)):
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
