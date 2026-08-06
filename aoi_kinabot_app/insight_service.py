"""Data-minimized, evidence-bounded wellness insight generation."""

from __future__ import annotations

import json
import os
from typing import Any

from config import OFFLINE_RESEARCH_MODE, OPENAI_INSIGHT_MODEL
from wellness_guidance import (
    MEDITERRANEAN_TRIAL,
    SOCIAL_ENGAGEMENT_STUDY,
    WHO_GUIDELINE,
)


ACTIONS = {
    "English": [
        {
            "action": (
                "Tomorrow, talk with a friend or family member for 20 minutes. "
                "Tell one story about your day and ask one follow-up question."
            ),
            "why": "Social connection is a general cognitive-wellness habit.",
            "source": SOCIAL_ENGAGEMENT_STUDY,
        },
        {
            "action": (
                "Tomorrow, if it is safe for you, take a comfortable 20-minute "
                "walk and describe three things you noticed afterward."
            ),
            "why": "Regular physical activity is supported as a general wellness habit.",
            "source": WHO_GUIDELINE,
        },
        {
            "action": (
                "At one meal tomorrow, include vegetables, legumes, whole grains, "
                "fish, nuts, or olive oil in a way that fits your dietary needs."
            ),
            "why": "Mediterranean-style eating patterns have been studied for cognitive wellness.",
            "source": MEDITERRANEAN_TRIAL,
        },
    ],
    "日本語": [
        {
            "action": "明日、家族や友人と20分話しましょう。今日の出来事を一つ話し、相手にも質問を一つしてみてください。",
            "why": "人との交流は、一般的な認知ウェルネス習慣の一つです。",
            "source": SOCIAL_ENGAGEMENT_STUDY,
        },
        {
            "action": "安全に行える場合、明日20分ほど無理のない散歩をし、後で気づいたことを三つ話してみましょう。",
            "why": "定期的な身体活動は、一般的な健康習慣として支持されています。",
            "source": WHO_GUIDELINE,
        },
        {
            "action": "明日の一食に、体調や食事制限に合わせて、野菜、豆類、全粒穀物、魚、ナッツ、オリーブ油などを取り入れてみましょう。",
            "why": "地中海食に近い食習慣は、認知ウェルネスとの関連が研究されています。",
            "source": MEDITERRANEAN_TRIAL,
        },
    ],
    "中文": [
        {
            "action": "从明天开始，和家人或朋友聊20分钟。讲一件今天发生的事，再问对方一个问题。",
            "why": "保持社交联系是一种通用的认知健康生活习惯。",
            "source": SOCIAL_ENGAGEMENT_STUDY,
        },
        {
            "action": "如果身体情况允许，明天舒适地散步20分钟，之后说出途中注意到的三件事。",
            "why": "规律的身体活动是一种有研究支持的通用健康习惯。",
            "source": WHO_GUIDELINE,
        },
        {
            "action": "明天选一餐，在符合自身饮食需要的前提下加入蔬菜、豆类、全谷物、鱼、坚果或橄榄油。",
            "why": "偏地中海式的饮食模式已被用于认知健康相关研究。",
            "source": MEDITERRANEAN_TRIAL,
        },
    ],
}

BOUNDARY = {
    "English": "This is a general wellness action, not a medical assessment or treatment.",
    "日本語": "これは一般的なウェルネス行動であり、医療評価や治療ではありません。",
    "中文": "这是一项通用健康行动，不是医疗评估或治疗。",
}


def anonymous_trend_payload(history: list[dict[str, Any]], language: str) -> dict:
    """Build the complete and exclusive payload allowed to leave KinaBot."""
    feature_series: dict[str, list[float]] = {}
    for row in history:
        name = str(row["feature_name"])
        feature_series.setdefault(name, []).append(round(float(row["score"]), 1))
    return {
        "language": language,
        "sessions_compared": len(
            {(str(row.get("created_at", "")), int(row.get("session_number", 0))) for row in history}
        ),
        "feature_scores": feature_series,
    }


def _fallback_action(language: str, payload: dict) -> dict:
    series = payload.get("feature_scores", {})
    largest_drop = min(
        ((values[-1] - values[0], name) for name, values in series.items() if len(values) >= 2),
        default=(0, ""),
    )[1]
    index = sum(ord(character) for character in largest_drop) % len(ACTIONS["English"])
    result = dict(ACTIONS.get(language, ACTIONS["English"])[index])
    result["boundary"] = BOUNDARY.get(language, BOUNDARY["English"])
    result["generated_by"] = "KinaBot research action library"
    return result


def generate_wellness_insight(history: list[dict[str, Any]], language: str) -> dict:
    """Generate one action; OpenAI sees anonymous score series and allowed actions only."""
    payload = anonymous_trend_payload(history, language)
    if (
        OFFLINE_RESEARCH_MODE
        or payload["sessions_compared"] < 3
        or not os.getenv("OPENAI_API_KEY", "").strip()
    ):
        return _fallback_action(language, payload)

    from openai import OpenAI

    allowed = ACTIONS.get(language, ACTIONS["English"])
    prompt = {
        "task": (
            "Choose exactly one allowed action that is most useful for the repeated "
            "longitudinal pattern. Return its zero-based action_index and one short, "
            "plain-language encouragement sentence in the requested language."
        ),
        "anonymous_kina_scores": payload,
        "allowed_actions": allowed,
    }
    try:
        response = OpenAI().responses.create(
            model=OPENAI_INSIGHT_MODEL,
            instructions=(
                "You write general cognitive-wellness habit information for KinaBot. "
                "Use only the supplied anonymous sample-score history and choose only "
                "from the supplied action list. Never diagnose, infer cognitive decline, "
                "claim that a habit will repair a score, claim causation, recommend "
                "medical treatment, or introduce a new action. State the useful action "
                "directly without filler."
            ),
            input=json.dumps(prompt, ensure_ascii=False),
            reasoning={"effort": "low"},
            text={
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "kinabot_wellness_insight",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action_index": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": len(allowed) - 1,
                            },
                            "encouragement": {
                                "type": "string",
                                "maxLength": 300,
                            },
                        },
                        "required": ["action_index", "encouragement"],
                        "additionalProperties": False,
                    },
                },
            },
            store=False,
        )
        parsed = json.loads(response.output_text)
        index = max(0, min(len(allowed) - 1, int(parsed.get("action_index", 0))))
        result = dict(allowed[index])
        result["encouragement"] = str(parsed.get("encouragement", "")).strip()[:300]
        result["boundary"] = BOUNDARY.get(language, BOUNDARY["English"])
        result["generated_by"] = "OpenAI from anonymous scores and curated actions"
        return result
    except Exception:
        return _fallback_action(language, payload)
