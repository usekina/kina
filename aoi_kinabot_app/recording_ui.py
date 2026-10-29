"""A discoverable recording control around the existing native audio widget."""
from __future__ import annotations

import streamlit as st

COPY = {
    'English': {
        'recording': 'Recording… Click Finish recording when you are done.',
        'analyze': '3 · Analyze my reflection',
        'key': 'en', 'start': 'Start recording', 'stop': 'Finish recording',
        'label': 'Record your voice',
        'instruction': 'Click Start recording below. Allow microphone access if asked, then speak naturally for 30–90 seconds. When you finish speaking, click Finish recording.',
        'ready': 'Recording ready. Listen to check it, then choose Analyze my reflection below.',
    },
    '日本語': {
        'recording': '録音中です。話し終えたら「録音を終了」を押してください。',
        'analyze': '3 · 録音を解析する',
        'key': 'ja', 'start': '録音を開始', 'stop': '録音を終了',
        'label': '声を録音する',
        'instruction': '下の「録音を開始」を押してください。求められたらマイクを許可し、30〜90秒ほど自然に話します。終わったら「録音を終了」を押してください。',
        'ready': '録音できました。再生して確認し、下の解析ボタンに進んでください。',
    },
    '中文': {
        'recording': '正在录音… 讲完后，点击「结束录音」。',
        'analyze': '3 · 分析我的录音',
        'key': 'zh', 'start': '开始录音', 'stop': '结束录音',
        'label': '录下你的声音',
        'instruction': '点击下方「开始录音」。如浏览器询问，请允许使用麦克风，自然讲述30–90秒；完成后点击「结束录音」。',
        'ready': '录音已准备好。可先回放检查，再点击下方的分析按钮。',
    },
}


def recording_copy(language: str) -> dict:
    return COPY.get(language, COPY['English'])


def recorder_css(language: str) -> str:
    copy = recording_copy(language)
    scope = '.st-key-kinabot_recorder_' + copy['key']
    # Style the actual native buttons, not a separate button or click overlay.
    # aria-label values are the native Streamlit audio widget's control names.
    return f'''<style>
{scope} [data-testid="stAudioInput"] {{ padding: 1rem; border: 2px solid #e85d2a; border-radius: .8rem; }}
{scope} [data-testid="stAudioInput"] button {{ min-width: 44px; min-height: 44px; color: #c2411b; }}
{scope} [data-testid="stAudioInput"]:has(button[aria-label="Stop recording"])::before {{
    content: "{copy['recording']}"; display: block; color: #b93818;
    font-weight: 650; margin-bottom: .75rem; line-height: 1.5;
}}
{scope} [data-testid="stAudioInput"]:has(button[aria-label="Stop recording"]) {{ background: #fff4ee; }}
{scope} [data-testid="stAudioInputWaveformTimeCode"] {{ color: #b93818; font-weight: 650; }}
{scope} [data-testid="stAlert"] {{ background: #fff4ee; color: #943015; border-color: #e85d2a; }}
{scope} [data-testid="stAudioInput"] button:is([aria-label="Record"], [aria-label="Stop recording"]) {{
    display: inline-flex; align-items: center; justify-content: center; gap: .65rem;
    width: auto !important; min-width: 180px; min-height: 56px;
    padding: .75rem 1rem; border-radius: .7rem; border: 2px solid transparent;
    background: #c2411b; color: #fff; font-size: 1.05rem; font-weight: 650;
    cursor: pointer; white-space: normal; line-height: 1.3;
}}
{scope} [data-testid="stAudioInput"] button[aria-label="Record"]::after {{ content: "{copy['start']}"; }}
{scope} [data-testid="stAudioInput"] button[aria-label="Stop recording"] {{ background: #b93818; }}
{scope} [data-testid="stAudioInput"] button[aria-label="Stop recording"]::after {{ content: "{copy['stop']}"; }}
{scope} [data-testid="stAudioInput"] button:is([aria-label="Record"], [aria-label="Stop recording"]) svg {{ width: 24px; height: 24px; flex-shrink: 0; }}
{scope} [data-testid="stAudioInput"] button:hover:not(:disabled) {{ filter: brightness(.9); }}
{scope} [data-testid="stAudioInput"] button:focus-visible {{ outline: 3px solid #e85d2a; outline-offset: 3px; }}
{scope} [data-testid="stAudioInput"] button:disabled {{ opacity: .6; cursor: not-allowed; }}
@media (max-width: 480px) {{
    {scope} [data-testid="stAudioInput"] {{ padding: .6rem; }}
    {scope} [data-testid="stAudioInput"] button:is([aria-label="Record"], [aria-label="Stop recording"]) {{ min-width: 0; padding: .75rem .7rem; font-size: 1rem; }}
}}
</style>'''


def render_recorder(language: str, help_text: str):
    copy = recording_copy(language)
    st.markdown(recorder_css(language), unsafe_allow_html=True)
    with st.container(key='kinabot_recorder_' + copy['key']):
        st.markdown('#### ' + copy['label'])
        st.write(copy['instruction'])
        audio = st.audio_input(copy['label'], help=help_text, key='reflection_audio')
        if audio is not None:
            st.success(copy['ready'])
    return audio
