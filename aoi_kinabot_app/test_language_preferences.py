import pytest
from streamlit.testing.v1 import AppTest
from language_preferences import LANGUAGES, language_copy
from language_analysis import LANGUAGE_CODES

PREVIEW = '''
import streamlit as st
from language_preferences import render_language_preferences, render_speaking_language, render_language_summary
from language_analysis import LANGUAGE_CODES
verified=st.session_state.get('preview_verified',False)
render_language_preferences(verified=verified)
if not verified:
    if st.button('Enter'):
        st.session_state.preview_verified=True
        st.rerun()
elif not st.session_state.get('away',False):
    language=render_speaking_language(st.session_state.ui_language,location='recording')
    render_language_summary(st.session_state.ui_language,language)
    st.text('transcription='+LANGUAGE_CODES[language])
    if st.button('Leave'):
        st.session_state.away=True
        st.rerun()
else:
    if st.button('Return'):
        st.session_state.away=False
        st.rerun()
'''


@pytest.mark.parametrize('display',LANGUAGES)
@pytest.mark.parametrize('speaking',LANGUAGES)
def test_display_and_speech_are_independent_across_navigation(display,speaking):
    app=AppTest.from_string(PREVIEW).run()
    app.radio[0].set_value(display).run()
    app.radio[1].set_value(speaking).run()
    assert not app.exception
    assert app.session_state['ui_language']==display
    assert app.session_state['recording_language']==speaking
    next(b for b in app.button if b.label=='Enter').click().run()
    assert not app.exception
    assert app.radio[0].value==speaking
    assert 'transcription='+LANGUAGE_CODES[speaking] in [x.value for x in app.text]
    different=next(x for x in LANGUAGES if x!=display)
    app.selectbox[0].set_value(different).run()
    assert app.radio[0].value==speaking
    # The recording widget is destroyed while away; its durable preference stays.
    next(b for b in app.button if b.label=='Leave').click().run()
    app.run()
    next(b for b in app.button if b.label=='Return').click().run()
    assert not app.exception
    assert app.radio[0].value==speaking
    assert app.session_state['ui_language']==different
    updated=next(x for x in LANGUAGES if x!=speaking)
    app.radio[0].set_value(updated).run()
    assert app.session_state['recording_language']==updated
    assert app.session_state['ui_language']==different
    assert 'transcription='+LANGUAGE_CODES[updated] in [x.value for x in app.text]
    assert language_copy(different)['summary'].format(display=different,speaking=updated) in [x.value for x in app.caption]


def test_invalid_preferences_have_supported_defaults():
    app=AppTest.from_string('''
import streamlit as st
from language_preferences import initialize_languages
st.session_state.ui_language='unsupported'
st.session_state.recording_language='unsupported'
initialize_languages()
''').run()
    assert not app.exception
    assert app.session_state['ui_language']=='English'
    assert app.session_state['recording_language']=='English'
