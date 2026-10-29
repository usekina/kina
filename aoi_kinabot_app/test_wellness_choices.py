from pathlib import Path
import pytest
import database
from insight_service import generate_wellness_insight
from wellness_guidance import wellness_suggestions, COPY
from wellness_view import recent_observation, repetition_change
from wellness_compact import compact_copy
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
    app.selectbox[0].set_value('custom')
    app.text_input[0].set_value('Read something enjoyable')
    next(b for b in app.button if b.key.endswith('FormSubmitter:wellness_'+str(uid)+'_2026-10-29_custom_form-'+compact_copy(language)['save'])).click().run()
    assert not app.exception
    assert database.get_wellness_plan(uid,'2026-10-29')['status']=='planned'
    next(b for b in app.button if b.label==compact_copy(language)['mark']).click().run()
    assert not app.exception
    assert database.get_wellness_plan(uid,'2026-10-29')['status']=='completed'


def test_raw_ratio_changes_do_not_relabel_index_scores():
    result = repetition_change([row(1,90,raw_metric='repeated_instances=10; ratio=0.100'),
                                row(2,88,raw_metric='repeated_instances=12; ratio=0.120')])
    assert result['current'] == 12
    assert result['points'] == pytest.approx(2)
    assert result['relative'] == pytest.approx(20)
    assert repetition_change([row(1,90),row(2,88)]) is None
    assert repetition_change([row(1,90,raw_metric='ratio=0.1'),
                              row(2,88,raw_metric='ratio=NaN')]) is None


def test_raw_ratio_zero_and_comparability():
    before = row(1,100,raw_metric='ratio=0.000',session_type='reflection')
    after = row(2,90,raw_metric='ratio=0.100',session_type='reflection')
    assert repetition_change([before, after])['relative'] is None
    assert repetition_change([before,{**after,'session_type':'test'}]) is None
    assert repetition_change([before,{**after,'availability_status':'unavailable'}]) is None
    assert repetition_change([before,{**after,'raw_metric':'ratio=1.100'}]) is None
    # Do not skip a missing prior measurement and substitute an older session.
    assert repetition_change([before,row(2,90,session_type='reflection'),
                              {**after,'session_id':3}]) is None


@pytest.mark.parametrize('language', list(COPY))
def test_compact_main_journey_and_skip(tmp_path, monkeypatch, language):
    uid=setup_db(tmp_path,monkeypatch)
    script=f"from wellness_view import render_wellness_choice\nrender_wellness_choice({uid},'2026-10-29',{language!r})"
    app=AppTest.from_string(script).run()
    assert len(app.radio[0].options)==3
    app.radio[0].set_value('physical_activity').run()
    assert compact_copy(language)['move_why'] in [x.value for x in app.caption]
    next(b for b in app.button if b.key.endswith('_save')).click().run()
    assert database.get_wellness_plan(uid,'2026-10-29')['status']=='planned'
    next(b for b in app.button if b.key.endswith('_done')).click().run()
    assert database.wellness_completion_summary(uid,'2026-10-29')['count']==1
    app.run()
    assert database.wellness_completion_summary(uid,'2026-10-29')['count']==1
    next(b for b in app.button if b.key.endswith('_edit')).click().run()
    next(b for b in app.button if b.key.endswith('_skip')).click().run()
    assert not app.exception
    assert database.wellness_completion_summary(uid,'2026-10-29')['count']==0


def test_completion_history_excludes_other_accounts_and_future_dates(tmp_path, monkeypatch):
    uid=setup_db(tmp_path,monkeypatch)
    other=database.upsert_user('another')
    for owner, day in [(uid,'2026-10-29'),(uid,'2026-10-30'),(other,'2026-10-29')]:
        database.save_wellness_plan(owner,day,'social_connection')
        database.complete_wellness_plan(owner,day)
    assert database.wellness_completion_summary(uid,'2026-10-29')['count']==1
