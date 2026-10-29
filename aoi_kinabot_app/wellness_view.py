"""Local observations followed by self-directed, evidence-linked action choices."""
from __future__ import annotations

import streamlit as st
import math
import re
from html import escape
from database import (get_user_scores, get_wellness_plan, save_wellness_plan,
                      complete_wellness_plan, wellness_completion_summary)
from history_view import comparison_key
from radar_view import measured_score
from scoring import display_feature_name
from wellness_guidance import wellness_suggestions
from wellness_compact import compact_copy, SOURCES


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


def repetition_change(records: list[dict]) -> dict | None:
    """Use recorded raw ratios, never reinterpret feature-index points as %."""
    sessions = {}
    for row in records:
        if row.get('session_id') is not None:
            sessions.setdefault(int(row['session_id']), []).append(row)
    if not sessions:
        return None
    latest = max(sessions)
    metadata = sessions[latest][0]
    key = comparison_key(metadata)
    if any(part in {'unknown', 'legacy-unknown', ''} for part in key):
        return None
    def ratio(sid):
        row = next((r for r in sessions[sid] if r['feature_name'] == 'Repetition Pattern'), None)
        if not row or measured_score(row) is None:
            return None
        match = re.search(r'(?:^|;\s*)ratio=([0-9]+(?:\.[0-9]+)?)(?:;|$)', str(row.get('raw_metric') or ''))
        value = float(match[1]) if match else None
        return value if value is not None and math.isfinite(value) and 0 <= value <= 1 else None
    earlier = [sid for sid, rows in sessions.items() if sid < latest
               and comparison_key(rows[0]) == key
               and rows[0].get('session_type') == metadata.get('session_type')]
    if not earlier:
        return None
    previous = max(earlier)
    now, before = ratio(latest), ratio(previous)
    if now is None or before is None:
        return None
    return {'current': now * 100, 'previous': before * 100,
            'points': (now - before) * 100,
            'relative': (now / before - 1) * 100 if before else None,
            'latest_id': latest, 'previous_id': previous}


def render_wellness_choice(user_id: int, today: str, language: str, *, show_observation=False):
    c = {**wellness_suggestions(language, []), **compact_copy(language)}
    key = f'wellness_{user_id}_{today}'
    labels = dict(c['habit_labels'], **{k: c[k] for k in SOURCES}, custom=c['custom'])
    saved = get_wellness_plan(user_id, today)
    current = saved['action_id'] if saved else None
    with st.container(key='wellness_care'):
        if show_observation:
            change = repetition_change([dict(r) for r in get_user_scores(user_id)])
            if change:
                relative = (f"{change['relative']:+.1f}% {c['change']}"
                            if change['relative'] is not None else c['zero'])
                text = (f"{c['ratio']} **{change['current']:.1f}%** · "
                        f"{c['previous']} {change['previous']:.1f}% · {relative}")
                st.markdown(text)
                with st.expander(c['method']):
                    st.caption(c['method_note'])
                    st.caption(f"{change['points']:+.1f} {c['points']}")
            else:
                st.caption(c['no_ratio'])
        st.subheader(c['heading'])
        st.caption(c['intro'])
        editing = not saved or saved['status'] == 'skipped' or st.session_state.get(key+'_editing', False)
        if not editing:
            title = saved['custom_text'] if current == 'custom' else labels.get(current, current)
            st.markdown('**' + escape(title) + '**')
            st.caption(c['planned'] if saved['status'] == 'planned' else c['done'])
            if saved['status'] == 'planned' and st.button(c['mark'], key=key+'_done', type='primary', use_container_width=True):
                complete_wellness_plan(user_id, today)
                st.rerun()
            if st.button(c['more'], key=key+'_edit'):
                st.session_state[key+'_editing'] = True
                st.rerun()
            selected = current
        else:
            ids = list(SOURCES)
            hints = dict(zip(ids, (c['social_hint'], c['move_hint'], c['food_hint'])))
            selected = st.radio(c['choose'], ids, index=ids.index(current) if current in ids else 0,
                                format_func=lambda value: labels[value]+' — '+hints[value],
                                label_visibility='collapsed', key=key+'_choice')
        if selected in SOURCES:
            source_title, source_url, prefix = SOURCES[selected]
            st.caption(c[prefix+'_why'])
            with st.expander(c['source_heading']):
                st.caption(c['choice_note'])
                st.write(c[prefix+'_evidence'])
                st.link_button(c['source_link']+' · '+source_title, source_url)
        elif selected == 'learning_interest':
            with st.expander(c['source_heading']):
                st.caption(c['evidence'])
                st.link_button(c['source_link'], 'https://www.nia.nih.gov/health/cognitive-health-and-older-adults')
        elif selected == 'custom':
            st.caption(c['custom_evidence'])
        if editing:
            if st.button(c['save'], key=key+'_save', type='primary', use_container_width=True):
                save_wellness_plan(user_id, today, selected)
                st.session_state[key+'_editing'] = False
                st.rerun()
            if st.button(c['skip'], key=key+'_skip', use_container_width=True):
                save_wellness_plan(user_id, today, 'skip')
                st.session_state[key+'_editing'] = False
                st.rerun()
            with st.expander(c['more']):
                with st.form(key+'_custom_form'):
                    extra = st.selectbox(c['choose'], ['custom', 'learning_interest'], format_func=lambda x: labels[x])
                    text = st.text_input(c['custom_label'], max_chars=160,
                                         value=saved['custom_text'] if saved else '',
                                         help=c['custom_help'], key=key+'_custom')
                    submit = st.form_submit_button(c['save'])
                if submit:
                    if extra == 'custom' and not text.strip():
                        st.warning(c['custom_required'])
                    else:
                        save_wellness_plan(user_id, today, extra, text)
                        st.session_state[key+'_editing'] = False
                        st.rerun()
        summary = wellness_completion_summary(user_id, today)
        st.divider()
        st.caption(c['history'].format(count=summary['count']))
        if summary['latest']:
            last = summary['latest']
            title = last['custom_text'] if last['action_id'] == 'custom' else labels.get(last['action_id'], last['action_id'])
            st.caption(f"{c['last']}: {last['plan_date']} · {title}")
