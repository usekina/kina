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
            "research_summary": (
                "A 3-year observational study followed 217 cognitively normal older adults. "
                "Greater social engagement was associated with relatively better cognitive trajectories "
                "in one higher-risk subgroup; the study does not prove that conversation prevents decline."
            ),
            "source": SOCIAL_ENGAGEMENT_STUDY,
        },
        {
            "action": (
                "Tomorrow, if it is safe for you, take a comfortable 20-minute "
                "walk and describe three things you noticed afterward."
            ),
            "why": "Regular physical activity is supported as a general wellness habit.",
            "research_summary": (
                "WHO reviewed evidence on modifiable lifestyle factors and recommends regular physical "
                "activity as part of risk-reduction guidance. This is population guidance, not an "
                "individual treatment promise."
            ),
            "source": WHO_GUIDELINE,
        },
        {
            "action": (
                "At one meal tomorrow, include vegetables, legumes, whole grains, "
                "fish, nuts, or olive oil in a way that fits your dietary needs."
            ),
            "why": "Mediterranean-style eating patterns have been studied for cognitive wellness.",
            "research_summary": (
                "A randomized PREDIMED-NAVARRA trial reported better cognitive performance with "
                "Mediterranean-style diets than with its control diet in older adults at high vascular risk. "
                "It does not mean that one meal changes cognition."
            ),
            "source": MEDITERRANEAN_TRIAL,
        },
    ],
    "日本語": [
        {
            "action": "明日、家族や友人と20分話しましょう。今日の出来事を一つ話し、相手にも質問を一つしてみてください。",
            "why": "人との交流は、一般的な認知ウェルネス習慣の一つです。",
            "research_summary": (
                "認知機能が正常な高齢者217人を3年間追跡した観察研究です。特定の高リスク群では、"
                "交流の多さが認知機能の相対的な維持と関連しましたが、会話が低下を防ぐと証明した研究ではありません。"
            ),
            "source": SOCIAL_ENGAGEMENT_STUDY,
        },
        {
            "action": "安全に行える場合、明日20分ほど無理のない散歩をし、後で気づいたことを三つ話してみましょう。",
            "why": "定期的な身体活動は、一般的な健康習慣として支持されています。",
            "research_summary": (
                "WHOは修正可能な生活習慣に関する研究を検討し、リスク低減の一環として定期的な身体活動を"
                "推奨しています。これは集団向けの指針であり、個人への治療効果を保証するものではありません。"
            ),
            "source": WHO_GUIDELINE,
        },
        {
            "action": "明日の一食に、体調や食事制限に合わせて、野菜、豆類、全粒穀物、魚、ナッツ、オリーブ油などを取り入れてみましょう。",
            "why": "地中海食に近い食習慣は、認知ウェルネスとの関連が研究されています。",
            "research_summary": (
                "PREDIMED-NAVARRA無作為化試験では、血管リスクの高い高齢者において、地中海食群の認知成績が"
                "対照食群より良好でした。一回の食事で認知機能が変わることを示すものではありません。"
            ),
            "source": MEDITERRANEAN_TRIAL,
        },
    ],
    "中文": [
        {
            "action": "从明天开始，和家人或朋友聊20分钟。讲一件今天发生的事，再问对方一个问题。",
            "why": "保持社交联系是一种通用的认知健康生活习惯。",
            "research_summary": (
                "这项观察性研究对217名认知功能正常的老年人随访了3年。在一个较高风险亚组中，较多社交参与"
                "与认知表现相对保持有关；研究并未证明一次谈话能够预防认知下降。"
            ),
            "source": SOCIAL_ENGAGEMENT_STUDY,
        },
        {
            "action": "如果身体情况允许，明天舒适地散步20分钟，之后说出途中注意到的三件事。",
            "why": "规律的身体活动是一种有研究支持的通用健康习惯。",
            "research_summary": (
                "WHO审查了可调整生活方式因素的相关证据，并把规律身体活动列为风险降低建议的一部分。"
                "这是面向人群的一般指南，不代表对个人的治疗保证。"
            ),
            "source": WHO_GUIDELINE,
        },
        {
            "action": "明天选一餐，在符合自身饮食需要的前提下加入蔬菜、豆类、全谷物、鱼、坚果或橄榄油。",
            "why": "偏地中海式的饮食模式已被用于认知健康相关研究。",
            "research_summary": (
                "PREDIMED-NAVARRA随机试验发现，在心血管风险较高的老年人中，地中海式饮食组的认知表现"
                "优于对照饮食组；这并不表示一顿饭就能改变认知功能。"
            ),
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
        if row.get("score") is None or row.get("availability_status") == "unavailable":
            continue
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
            "longitudinal pattern. Return only its zero-based action_index."
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
                        },
                        "required": ["action_index"],
                        "additionalProperties": False,
                    },
                },
            },
            store=False,
        )
        parsed = json.loads(response.output_text)
        index = max(0, min(len(allowed) - 1, int(parsed.get("action_index", 0))))
        result = dict(allowed[index])
        result["boundary"] = BOUNDARY.get(language, BOUNDARY["English"])
        result["generated_by"] = "OpenAI from anonymous scores and curated actions"
        return result
    except Exception:
        return _fallback_action(language, payload)
