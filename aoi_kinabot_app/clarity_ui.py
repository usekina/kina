"""Clarity UI: plain-language results and a descriptive eight-axis radar."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from radar_view import radar_comparison
from scoring import display_feature_name, feature_explanation


COPY = {
    "record": ("Record", "録音", "录音"),
    "results": ("Results", "結果", "结果"),
    "history": ("History", "履歴", "历史"),
    "account": ("Account settings", "アカウント設定", "账户设置"),
    "start": ("Start with a short recording.", "短い録音から、はじめよう。", "从一段录音开始。"),
    "intro": ("Speak naturally. Notice your patterns over time.", "自然に話して、自分の変化を振り返る。", "自然表达，慢慢看见自己的变化。"),
    "prompt": ("What small moment stood out today?", "今日、心に残った小さな出来事は？", "今天，哪件小事让你印象深刻？"),
    "prompt_hint": ("Or talk about anything you like. Aim for 30–90 seconds.", "好きな話題でも大丈夫。30〜90秒を目安に。", "也可以聊任何想说的事，建议录制 30–90 秒。"),
    "title": ("This recording, compared with you.", "今回と、これまでの自分。", "这一次，与过去的自己。"),
    "empty": ("Your first recording starts your history.", "最初の録音から、履歴が始まります。", "从第一段录音，开始留下自己的记录。"),
    "choose": ("Choose a recording", "録音を選択", "选择记录"),
    "current": ("This recording", "今回の録音", "本次记录"),
    "reference": ("Recent reference", "最近の参考値", "近期参考"),
    "compare": ("Show recent reference", "最近の参考値を重ねる", "叠加近期参考"),
    "reference_note": (
        "Reference: the three preceding recordings with the same language and analysis method. This is a recent mean, not an established personal baseline.",
        "参考値は、同じ言語・分析方式による直前3回の平均です。確立された個人ベースラインではありません。",
        "参考来自此前三次同语言、同分析方式的记录均值，尚不是稳定的个人基线。"),
    "building": ("Recent reference needs 3 earlier comparable recordings. Available: {count}/3.", "参考値には比較可能な過去の録音が3回必要です。現在 {count}/3 回。", "近期参考需要此前三次可比较记录，目前 {count}/3 次。"),
    "boundary": ("Feature indexes, 0–100. A larger shape is not better health. No overall score.", "0〜100の特徴指数です。図形の大きさは健康の良し悪しを示しません。総合点はありません。", "0–100 特征指数。图形越大不代表越健康，不设综合总分。"),
    "missing": ("Gaps mean no valid measurement. Missing values are not zero; emotion-word defaults are not measured mood.", "空白は有効な測定がない項目です。0点ではなく、感情語の既定値は気分の測定ではありません。", "留空表示没有有效测量，不是零分；情绪词默认值不代表真实情绪。"),
    "detail": ("All eight features and their meaning", "8つの特徴とその意味", "八项指标与含义"),
    "feature": ("Feature", "特徴", "指标"),
    "unavailable": ("Not measured", "測定なし", "无有效测量"),
    "change": ("Difference", "差", "差异"),
    "explain": ("Meaning", "意味", "含义"),
    "method": ("Comparison details", "比較条件", "比较条件"),
    "next": ("See changes over time", "時間の変化を見る", "查看历史变化"),
    "analyze": ("Analyze this recording", "この録音を分析", "分析这段录音"),
    "habit": ("Optional: today's habit", "任意：今日の習慣", "可选：今天的习惯"),
    "history_intro": ("Compare your own compatible recordings. One change does not establish a lasting pattern.", "同じ条件の自分の録音を比較します。1回の差だけで長期的な変化は判断できません。", "只比较条件一致的个人记录。一次差异，不代表持续变化。"),
    "duration": ("Duration", "録音時間", "录音时长"),
    "unknown": ("Unknown", "不明", "未知"),
    "seconds": ("seconds", "秒", "秒"),
}


def ui_copy(key: str, language: str) -> str:
    return COPY[key][{"日本語": 1, "中文": 2}.get(language, 0)]


def inject_theme() -> None:
    st.markdown("<style>" + Path(__file__).with_name("clarity.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


def radar_document(data: dict, language: str, show_reference: bool) -> str:
    """Self-contained responsive SVG with no requests or stored user content."""
    payload = {
        "names": [display_feature_name(n, language) for n in data["features"]],
        "current": data["current"], "reference": data["reference"] if show_reference else [],
        "currentLabel": ui_copy("current", language), "referenceLabel": ui_copy("reference", language),
    }
    safe_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return """<!doctype html><html><head><meta charset="utf-8"><style>
    :root{color-scheme:light dark;--ink:light-dark(#172d48,#e7eef9);--line:light-dark(#dce3ed,#404d60);--accent:light-dark(#245b95,#8cbfff);--muted:light-dark(#617086,#b8c3d4)}
    body{margin:0;font:14px system-ui,sans-serif;color:var(--ink)}svg{display:block;width:100%;height:390px}text{fill:var(--ink);font:12px system-ui,sans-serif}.grid{fill:none;stroke:var(--line);stroke-width:1}.now{stroke:var(--accent);stroke-width:2.5;fill:color-mix(in srgb,var(--accent) 10%,transparent)}.ref{stroke:var(--muted);stroke-width:2;stroke-dasharray:5 5;fill:none}.dot{fill:var(--accent)}.tick{fill:var(--muted);font-size:11px}.legend{display:flex;justify-content:center;gap:22px;flex-wrap:wrap;font-size:12px}.swatch{display:inline-block;width:23px;margin-right:7px;border-top:3px solid var(--accent)}.past{border-top:2px dashed var(--muted)}</style></head><body>
    <svg role="img" aria-label="Eight descriptive feature indexes; full values in the table below"></svg><div class="legend"></div><script>
    const d=PAYLOAD;const svg=document.querySelector('svg');
    document.querySelector('.legend').innerHTML='<span><span class="swatch"></span>'+d.currentLabel+'</span>'+(d.reference.length?'<span><span class="swatch past"></span>'+d.referenceLabel+'</span>':'');
    function draw(){const w=svg.getBoundingClientRect().width,h=390,cx=w/2,cy=h/2,r=Math.max(40,Math.min(120,(w-160)/2));const pt=(i,v)=>[cx+Math.cos(-Math.PI/2+i*Math.PI/4)*r*v/100,cy+Math.sin(-Math.PI/2+i*Math.PI/4)*r*v/100];
    const path=(vals,cls)=>{if(!vals.length)return '';const complete=vals.every(v=>v!==null);if(complete)return '<polygon class="'+cls+'" points="'+vals.map((v,i)=>pt(i,v).join(',')).join(' ')+'"/>';
    let lines='';vals.forEach((v,i)=>{const j=(i+1)%8;if(v!==null&&vals[j]!==null){const a=pt(i,v),b=pt(j,vals[j]);lines+='<line class="'+cls+'" x1="'+a[0]+'" y1="'+a[1]+'" x2="'+b[0]+'" y2="'+b[1]+'"/>';}});return lines;};
    svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
    svg.innerHTML=[25,50,75,100].map(v=>'<polygon class="grid" points="'+d.names.map((_,i)=>pt(i,v).join(',')).join(' ')+'"/>').join('')+d.names.map((_,i)=>{const p=pt(i,100);return `<line class="grid" x1="${cx}" y1="${cy}" x2="${p[0]}" y2="${p[1]}"/>`;}).join('')+path(d.reference,'ref')+path(d.current,'now')+d.current.map((v,i)=>{if(v===null)return '';const p=pt(i,v);return `<circle class="dot" cx="${p[0]}" cy="${p[1]}" r="3.5"/>`;}).join('')+d.names.map((n,i)=>{const a=-Math.PI/2+i*Math.PI/4,x=cx+Math.cos(a)*(r+17),y=cy+Math.sin(a)*(r+23),anchor=(i===0||i===4)?'middle':i<4?'start':'end';const words=n.includes(' ')?n.split(' '):n.length>5?[n.slice(0,3),n.slice(3)]:[n];return `<text x="${x}" y="${y+4-(words.length-1)*7}" text-anchor="${anchor}">`+words.map((word,j)=>`<tspan x="${x}" dy="${j?14:0}">${word}</tspan>`).join('')+'</text>';}).join('')+[0,50,100].map(v=>`<text class="tick" x="${cx+5}" y="${cy-r*v/100+12}">${v}</text>`).join('');}
    new ResizeObserver(draw).observe(svg);draw();</script></body></html>""".replace("PAYLOAD", safe_json)


def render_result(records: list[dict], language: str) -> None:
    c = lambda key: ui_copy(key, language)
    st.title(c("title"))
    if not records:
        st.info(c("empty"))
        if st.button(c("record"), type="primary"):
            st.session_state.pending_primary_view = "today"
            st.rerun()
        return
    sessions = {int(r["session_id"]): dict(r) for r in records}
    selected = st.selectbox(c("choose"), sorted(sessions, reverse=True), format_func=lambda sid: f"{sessions[sid]['session_date']} · {sessions[sid]['language']} · #{sessions[sid]['session_number']}")
    data = radar_comparison(records, selected)
    seconds = data["metadata"].get("duration_seconds")
    st.caption(f"{c('duration')}: {round(seconds)} {c('seconds')}" if seconds is not None else f"{c('duration')}: {c('unknown')}")
    has_reference = any(v is not None for v in data["reference"])
    show_reference = st.checkbox(c("compare"), value=True, disabled=not has_reference)
    components.html(radar_document(data, language, show_reference and has_reference), height=430, scrolling=False)
    st.caption(c("boundary"))
    if data["reference_count"] < 3:
        st.info(c("building").format(count=data["reference_count"]))
    else:
        st.caption(c("reference_note"))
    if any(v is None for v in data["current"]) or (data["reference_count"] == 3 and any(v is None for v in data["reference"])):
        st.caption(c("missing"))
    with st.expander(c("detail")):
        rows = []
        for name, value, reference in zip(data["features"], data["current"], data["reference"]):
            rows.append({c("feature"): display_feature_name(name, language), c("current"): c("unavailable") if value is None else f"{value:g}", c("reference"): c("unavailable") if reference is None else f"{reference:g}", c("change"): "—" if value is None or reference is None else f"{value-reference:+.1f}", c("explain"): feature_explanation(name, language)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander(c("method")):
        st.caption(" · ".join(data["key"]))
        st.caption(c("reference_note"))
    if st.button(c("next"), type="primary", use_container_width=True):
        st.session_state.pending_primary_view = "trends"
        st.rerun()


def recording_prompt(language: str) -> None:
    st.title(ui_copy("start", language))
    st.caption(ui_copy("intro", language))
    st.markdown(f'<div class="clarity-prompt"><p>{escape(ui_copy("prompt", language))}</p><small>{escape(ui_copy("prompt_hint", language))}</small></div>', unsafe_allow_html=True)
