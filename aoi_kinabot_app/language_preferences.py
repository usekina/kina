"""Independent display and speaking languages, with session-stable choices."""
from __future__ import annotations

import streamlit as st

LANGUAGES = ('English', '日本語', '中文')
DISPLAY_LABEL = 'Display language / 表示言語 / 界面语言'
COPY = {
    'English': {
        'intro': 'Read in one language, speak in another. Choose each separately.',
        'speaking': 'Speaking language',
        'help': 'Choose the language spoken in your recording or uploaded audio. Supported: English, Japanese and Chinese.',
        'record_hint': 'Your screen language stays the same. You can change the speaking language before each recording.',
        'example': 'For example: read in English and record in Japanese.',
        'summary': 'Screen: {display} · Recording: {speaking}',
        'history': 'Trends compare recordings in the same speaking language and analysis method.',
    },
    '日本語': {
        'intro': '読む言語と話す言語は、別々に選べます。',
        'speaking': '録音で話す言語',
        'help': '録音・アップロード音声で話す言語を選んでください。英語・日本語・中国語に対応しています。',
        'record_hint': '表示言語は変わりません。話す言語は録音ごとに選べます。',
        'example': 'たとえば、日本語の画面を見ながら英語で録音できます。',
        'summary': '表示：{display} · 録音：{speaking}',
        'history': '傾向は、同じ発話言語・分析方式の録音同士で比較します。',
    },
    '中文': {
        'intro': '看得顺手，说得自然。界面和录音语言可以不同。',
        'speaking': '录音说什么语言？',
        'help': '请选择录音或上传音频中实际说的语言。目前支持英语、日语和中文。',
        'record_hint': '切换录音语言，不会改变界面语言。每次录音前都可以单独选择。',
        'example': '例如：用中文看界面，用日语说话。',
        'summary': '界面：{display} · 录音：{speaking}',
        'history': '历史趋势仅比较同一录音语言、同一分析方式的记录。',
    },
}


def language_copy(display_language: str) -> dict:
    return COPY.get(display_language, COPY['English'])


def initialize_languages() -> None:
    if st.session_state.get('ui_language') not in LANGUAGES:
        st.session_state.ui_language = 'English'
    # This durable key is never owned by a widget. Leaving the recording page
    # or changing display language must not reset the speaking preference.
    if st.session_state.get('recording_language') not in LANGUAGES:
        st.session_state.recording_language = st.session_state.ui_language


def _remember_speaking(widget_key: str) -> None:
    st.session_state.recording_language = st.session_state[widget_key]


def render_speaking_language(display_language: str, *, location: str) -> str:
    initialize_languages()
    c = language_copy(display_language)
    widget_key = f'{location}_speaking_language'
    # Synchronize when moving between landing and recording widgets, without
    # using the display language as a new default on each render.
    st.session_state[widget_key] = st.session_state.recording_language
    st.radio(c['speaking'], LANGUAGES, horizontal=True,
             help=c['help']+' '+c['history'], key=widget_key,
             on_change=_remember_speaking, args=(widget_key,))
    return st.session_state.recording_language


def render_language_preferences(*, verified: bool) -> None:
    initialize_languages()
    if verified:
        with st.sidebar:
            st.selectbox(DISPLAY_LABEL, LANGUAGES, key='ui_language')
            st.caption(language_copy(st.session_state.ui_language)['record_hint'])
    else:
        st.radio(DISPLAY_LABEL, LANGUAGES, horizontal=True, key='ui_language')
        c = language_copy(st.session_state.ui_language)
        st.caption(c['intro'])
        render_speaking_language(st.session_state.ui_language, location='landing')
        st.caption(c['example'])


def render_language_summary(display_language: str, speaking_language: str) -> None:
    st.caption(language_copy(display_language)['summary'].format(
        display=display_language, speaking=speaking_language))
