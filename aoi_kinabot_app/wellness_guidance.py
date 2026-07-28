"""Evidence-aligned, non-diagnostic wellness suggestions for KinaBot."""

from __future__ import annotations


WHO_GUIDELINE = "https://www.who.int/publications/i/item/9789241550543"
MEDITERRANEAN_TRIAL = "https://pubmed.ncbi.nlm.nih.gov/23670794/"
SOCIAL_ENGAGEMENT_STUDY = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6778491/"


COPY = {
    "English": {
        "heading": "Optional wellness ideas",
        "boundary": (
            "These are general wellness ideas, not treatment and not conclusions "
            "from your speech score. Choose what fits your health, abilities, and clinician's advice."
        ),
        "social": "Try a 10-minute conversation, shared story, or call with a friend or family member.",
        "walk": "If it is safe for you, consider a comfortable walk or another physical activity you enjoy.",
        "food": (
            "Consider Mediterranean-style choices such as vegetables, legumes, whole grains, "
            "fish, nuts, and olive oil. Check dietary changes with a clinician when appropriate."
        ),
    },
    "日本語": {
        "heading": "任意のウェルネス提案",
        "boundary": (
            "これは一般的な健康習慣の提案であり、治療や音声スコアからの診断ではありません。"
            "体調・能力・医療専門家の助言に合うものを選んでください。"
        ),
        "social": "友人や家族と10分ほど会話したり、出来事を話したり、電話したりしてみましょう。",
        "walk": "安全に行える場合は、無理のない散歩や好きな身体活動を検討してみましょう。",
        "food": (
            "野菜、豆類、全粒穀物、魚、ナッツ、オリーブ油などを取り入れた"
            "地中海食に近い食事を検討できます。必要に応じて医療専門家に相談してください。"
        ),
    },
    "中文": {
        "heading": "可选的健康生活建议",
        "boundary": (
            "这些是一般健康生活建议，不是治疗，也不是根据语音分数作出的诊断。"
            "请选择适合自己身体状况、能力及专业人员建议的做法。"
        ),
        "social": "可以尝试与朋友或家人交谈十分钟、分享一件往事，或者打一个电话。",
        "walk": "如果对你来说安全，可以考虑舒适地散步，或进行自己喜欢的身体活动。",
        "food": (
            "可以考虑偏地中海式的饮食选择，例如蔬菜、豆类、全谷物、鱼、坚果和橄榄油。"
            "如有特殊健康或饮食需求，请先咨询专业人员。"
        ),
    },
}

HABIT_LABELS = {
    "English": {
        "social_connection": "Connected with a friend or family member",
        "physical_activity": "Did comfortable physical activity",
        "mediterranean_style_meal": "Chose a Mediterranean-style meal",
    },
    "日本語": {
        "social_connection": "友人や家族と交流した",
        "physical_activity": "無理のない身体活動をした",
        "mediterranean_style_meal": "地中海食に近い食事を選んだ",
    },
    "中文": {
        "social_connection": "与朋友或家人进行了交流",
        "physical_activity": "进行了舒适的身体活动",
        "mediterranean_style_meal": "选择了偏地中海式的一餐",
    },
}


def wellness_suggestions(language: str, scores: list[dict]) -> dict:
    # Scores are intentionally not used to select or prioritize suggestions.
    # KinaBot does not infer wellness state, cognitive change, risk, or cause
    # from a speech sample. The ideas below are an independent optional menu.
    del scores
    copy = COPY.get(language, COPY["English"])
    suggestions = [
        {"text": copy["social"], "source": SOCIAL_ENGAGEMENT_STUDY},
        {"text": copy["walk"], "source": WHO_GUIDELINE},
        {"text": copy["food"], "source": MEDITERRANEAN_TRIAL},
    ]
    return {
        "heading": copy["heading"],
        "boundary": copy["boundary"],
        "suggestions": suggestions[:3],
        "habit_labels": HABIT_LABELS.get(language, HABIT_LABELS["English"]),
    }
