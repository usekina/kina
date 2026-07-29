"""Friendly, local-only presentation of KinaBot sample feature scores."""

from __future__ import annotations

from typing import Any


PROFILE_COMPONENTS = {
    "Expression": ("Vocabulary Variety", "Response Length", "Sentence Complexity"),
    "Flow": ("Speech Pace", "Pause Pattern", "Repetition Pattern"),
    "Clarity": ("Transcription Clarity", "Sentence Complexity"),
    "Energy": ("Response Length", "Speech Pace", "Emotional Tone"),
}

COPY = {
    "English": {
        "title": "Your expression snapshot",
        "subtitle": "A simple view of this recording—not a health or personality rating.",
        "labels": {
            "Expression": "Expression",
            "Flow": "Flow",
            "Clarity": "Clarity",
            "Energy": "Vocal energy",
        },
        "takeaway": "Core takeaway",
        "summary": "Your clearest feature in this sample was {strong}. A useful area to explore next is {focus}.",
        "actions": {
            "Expression": "Tomorrow, tell a short story using one person, one place, and one feeling.",
            "Flow": "Tomorrow, speak for one minute and allow a comfortable pause between ideas.",
            "Clarity": "Tomorrow, record in a quiet place and speak one complete thought at a time.",
            "Energy": "Tomorrow, describe one enjoyable moment for one minute in your natural voice.",
        },
        "action_title": "Try tomorrow",
        "detail": "View the 8 measured features",
    },
    "日本語": {
        "title": "今回の表現スナップショット",
        "subtitle": "今回の録音をわかりやすく示したもので、健康や性格の評価ではありません。",
        "labels": {
            "Expression": "表現",
            "Flow": "流れ",
            "Clarity": "明瞭さ",
            "Energy": "声のエネルギー",
        },
        "takeaway": "今回のポイント",
        "summary": "今回もっとも明確だった特徴は「{strong}」です。次は「{focus}」も試してみましょう。",
        "actions": {
            "Expression": "明日は、人物・場所・気持ちを一つずつ入れて短い話をしてみましょう。",
            "Flow": "明日は1分間、考えの間に自然な間を取りながら話してみましょう。",
            "Clarity": "明日は静かな場所で、一つの考えを一文ずつ話してみましょう。",
            "Energy": "明日は楽しかった出来事を一つ、自然な声で1分間話してみましょう。",
        },
        "action_title": "明日やってみること",
        "detail": "8つの測定特徴を見る",
    },
    "中文": {
        "title": "本次表达画像",
        "subtitle": "这是本次录音的简明展示，不是健康或性格评价。",
        "labels": {
            "Expression": "表达",
            "Flow": "流畅",
            "Clarity": "清晰",
            "Energy": "声音活力",
        },
        "takeaway": "核心特点",
        "summary": "本次最明显的特点是“{strong}”；下次可以重点体验“{focus}”。",
        "actions": {
            "Expression": "明天讲一个小故事，加入一个人物、一个地点和一种感受。",
            "Flow": "明天连续说一分钟，每个想法之间留一个自然停顿。",
            "Clarity": "明天在安静环境中录音，每次完整表达一个想法。",
            "Energy": "明天用自然的声音讲一分钟令你开心的小事。",
        },
        "action_title": "明天试一试",
        "detail": "查看 8 项测量特征",
    },
}


def build_reflection_profile(scores: list[dict[str, Any]], language: str) -> dict:
    """Aggregate existing local features into four transparent UX dimensions."""
    values = {str(item["feature_name"]): float(item["score"]) for item in scores}
    dimensions = {
        name: round(
            sum(values.get(feature, 0.0) for feature in features) / len(features)
        )
        for name, features in PROFILE_COMPONENTS.items()
    }
    copy = COPY.get(language, COPY["English"])
    strongest = max(dimensions, key=dimensions.get)
    focus = min(dimensions, key=dimensions.get)
    labels = copy["labels"]
    return {
        "title": copy["title"],
        "subtitle": copy["subtitle"],
        "dimensions": [
            {"key": key, "label": labels[key], "score": dimensions[key]}
            for key in PROFILE_COMPONENTS
        ],
        "takeaway_title": copy["takeaway"],
        "takeaway": copy["summary"].format(
            strong=labels[strongest],
            focus=labels[focus],
        ),
        "action_title": copy["action_title"],
        "action": copy["actions"][focus],
        "detail_label": copy["detail"],
    }
