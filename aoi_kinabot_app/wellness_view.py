"""Local observations followed by self-directed, evidence-linked action choices."""
from __future__ import annotations

import streamlit as st
from database import (get_user_scores, get_wellness_plan, save_wellness_plan,
                      complete_wellness_plan)
from history_view import comparison_key
from radar_view import measured_score
from scoring import display_feature_name
from wellness_guidance import wellness_suggestions


def recent_observation(records: list[dict]) -> list[dict]:
    """Compare latest session only with its previous compatible session.

    All values are feature-index points; never infer disease or action needs.
    Missing and default-imputed features are excluded by measured_score.
    """
    sessions = {}
    for row in records:
        if row.get('session_id') is not None:
            sessions.setdefault(int(row['session_id']), []).append(row)
    if not sessions:
        return []
    latest_id = max(sessions)
    key = comparison_key(sessions[latest_id][0])
    if any(x in {'unknown', 'legacy-unknown', ''} for x in key):
        return []
    candidates = [sid for sid, rows in sessions.items()
                  if sid < latest_id and comparison_key(rows[0]) == key]
    if not candidates:
        return []
    previous_id = max(candidates)
    old = {r['feature_name']: measured_score(r) for r in sessions[previous_id]}
    results = []
    for row in sessions[latest_id]:
        name = row['feature_name']
        value, prior = measured_score(row), old.get(name)
        if value is not None and prior is not None:
            results.append({'feature_name': name, 'current': value, 'previous': prior,
                            'delta': round(value - prior, 1),
                            'latest_id': latest_id, 'previous_id': previous_id})
    return results


def render_wellness_choice(user_id: int, today: str, language: str, *, show_observation=False):
    c = wellness_suggestions(language, [])
    if show_observation:
        st.subheader(c['observation'])
        changes = recent_observation([dict(r) for r in get_user_scores(user_id)])
        if changes:
            st.caption(c['observation_boundary'])
            with st.expander(c['observation']):
                for item in changes:
                    st.write(f"{display_feature_name(item['feature_name'], language)}: "
                             f"{item['previous']:g} → {item['current']:g} ({item['delta']:+.1f})")
        else:
            st.caption(c['no_comparison'])

    st.subheader(c['heading'])
    st.write(c['intro'])
    with st.expander(c['source_heading']):
        st.caption(c['boundary'])
        st.caption(c['provenance'])
        for idea in c['suggestions']:
            st.markdown('**' + idea['text'] + '**')
            st.caption(idea['evidence'])
            st.link_button(c['source_link'] + ' · ' + idea['source_title'], idea['source'])

    key = f'wellness_{user_id}_{today}'
    labels = dict(c['habit_labels'], custom=c['custom'], skip=c['skip'])
    saved = get_wellness_plan(user_id, today)
    current = saved['action_id'] if saved else None
    if saved:
        title = saved['custom_text'] if current == 'custom' else labels.get(current, current)
        st.write(c['current'] + ': ' + title + ' · ' + c[saved['status']])
        if saved['status'] == 'planned' and st.button(c['mark'], key=key+'_done'):
            complete_wellness_plan(user_id, today)
            st.session_state[key+'_done_notice'] = True
            st.rerun()
    if st.session_state.pop(key+'_done_notice', False):
        st.success(c['done'])
    ids = list(labels)
    with st.form(key+'_form'):
        selected = st.radio(c['choose'], ids,
                            index=ids.index(current) if current in ids else None,
                            format_func=lambda value: labels[value], key=key+'_choice')
        # Always available in the form: selecting custom does not require a
        # hidden widget to rerun before the user can enter their own action.
        text = st.text_input(c['custom_label'], max_chars=160,
                             value=saved['custom_text'] if saved else '',
                             help=c['custom_help'], key=key+'_custom')
        submit = st.form_submit_button(c['save'], type='primary')
    if submit:
        if selected is None:
            st.warning(c['choose_first'])
        elif selected == 'custom' and not text.strip():
            st.warning(c['custom_required'])
        else:
            save_wellness_plan(user_id, today, selected, text)
            st.session_state[key+'_saved_notice'] = True
            st.rerun()
    if st.session_state.pop(key+'_saved_notice', False):
        st.success(c['saved'])
