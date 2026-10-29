from pathlib import Path
import pytest
import database
from insight_service import generate_wellness_insight
from wellness_guidance import wellness_suggestions, COPY
from wellness_view import recent_observation
from streamlit.testing.v1 import AppTest


def row(sid, score, **kw):
    return dict(session_id=sid, score=score, feature_name='Repetition Pattern',
                language=kw.pop('language','English'), scoring_model_version='v1',
                analysis_pipeline_id=kw.pop('analysis_pipeline_id','p1'), **kw)


def setup_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database,'DATABASE_PATH',tmp_path/'test.sqlite3')
    database.init_db()
    return database.upsert_user('test-user')


def test_actions_do_not_follow_score_changes_or_call_api(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','must-not-be-used')
    import openai
    monkeypatch.setattr(openai,'OpenAI',lambda *a,**k: pytest.fail('Unexpected API request'))
    assert generate_wellness_insight([row(1,10),row(2,90)],'English') == generate_wellness_insight([row(1,90),row(2,10)],'English')
    for language in COPY:
        assert wellness_suggestions(language,[row(1,10)]) == wellness_suggestions(language,[row(1,90)])


def test_comparison_uses_latest_and_previous_compatible_recordings():
    result=recent_observation([row(1,20),row(2,30),row(3,95,language='日本語'),row(4,32)])
    assert result[0]['previous_id']==2
    assert result[0]['delta']==2
    assert result[0]['latest_id']==4


def test_no_old_group_substitution_or_unknown_measurements():
    assert recent_observation([row(1,20),row(2,30),row(3,40,language='日本語')])==[]
    assert recent_observation([row(1,20,analysis_pipeline_id='legacy-unknown'),row(2,30,analysis_pipeline_id='legacy-unknown')])==[]
    assert recent_observation([row(1,20),row(2,float('nan'))])==[]
    assert recent_observation([row(1,20),row(2,30,availability_status='unavailable')])==[]


def test_plan_is_not_completion_and_skip_cannot_be_completed(tmp_path,monkeypatch):
    uid=setup_db(tmp_path,monkeypatch)
    database.save_wellness_plan(uid,'2026-10-29','social_connection')
    assert database.get_wellness_plan(uid,'2026-10-29')['status']=='planned'
    database.complete_wellness_plan(uid,'2026-10-29')
    assert database.get_wellness_plan(uid,'2026-10-29')['status']=='completed'
    database.save_wellness_plan(uid,'2026-10-29','skip','discard this')
    database.complete_wellness_plan(uid,'2026-10-29')
    plan=database.get_wellness_plan(uid,'2026-10-29')
    assert plan['status']=='skipped' and plan['custom_text']==''
    assert database.get_user_habit_checkins(uid)==[]


def test_plan_scope_export_deletion_and_custom_validation(tmp_path,monkeypatch):
    uid=setup_db(tmp_path,monkeypatch)
    other=database.upsert_user('other-user')
    for text in ['', 'x'*161]:
        with pytest.raises(ValueError): database.save_wellness_plan(uid,'2026-10-29','custom',text)
    database.save_wellness_plan(uid,'2026-10-29','custom','Read a favorite story')
    assert database.get_wellness_plan(other,'2026-10-29') is None
    assert database.get_wellness_plan(uid,'2026-10-30') is None
    assert database.export_user_data(uid)['wellness_plans'][0]['custom_text']=='Read a favorite story'
    database.delete_user_research_data(uid)
    assert database.get_wellness_plan(uid,'2026-10-29') is None
    with pytest.raises(ValueError): database.save_wellness_plan(uid,'2026-10-29','skip')


@pytest.mark.parametrize('language', list(COPY))
def test_localized_plan_form_save_and_complete(tmp_path,monkeypatch,language):
    uid=setup_db(tmp_path,monkeypatch)
    script=f"from wellness_view import render_wellness_choice\nrender_wellness_choice({uid},'2026-10-29',{language!r},show_observation=True)"
    app=AppTest.from_string(script).run()
    assert not app.exception
    app.radio[0].set_value('custom')
    app.text_input[0].set_value('Read something enjoyable')
    next(b for b in app.button if b.label==COPY[language]['save']).click().run()
    assert not app.exception
    assert database.get_wellness_plan(uid,'2026-10-29')['status']=='planned'
    next(b for b in app.button if b.label==COPY[language]['mark']).click().run()
    assert not app.exception
    assert database.get_wellness_plan(uid,'2026-10-29')['status']=='completed'
