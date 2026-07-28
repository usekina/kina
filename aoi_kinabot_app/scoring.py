"""V1 speech and language feature scoring.

These scores are for cognitive wellness reflection and trend awareness.
They are not medical diagnosis, cognitive age, or disease risk prediction.
"""

from __future__ import annotations

import re
from collections import Counter

from config import SCORING_MODEL_VERSION


FEATURE_EXPLANATIONS = {
    "Vocabulary Variety": "This reflects word variety in this sample. It is not a diagnosis.",
    "Response Length": "This reflects the amount of speech captured in this session.",
    "Sentence Complexity": "This reflects sentence structure and expression style in this sample.",
    "Speech Pace": "This reflects speaking pace in this sample. Faster or slower is not automatically better or worse.",
    "Pause Pattern": "This reflects pause patterns during speech. In V1, this may be limited by audio processing quality.",
    "Repetition Pattern": "This reflects repetition patterns in this sample. It is a communication feature, not a medical conclusion.",
    "Emotional Tone": "This reflects emotional tone in the language sample. It may be affected by topic, mood, and context.",
    "Transcription Clarity": "This reflects whether the recording was clear enough for analysis.",
}

LANGUAGE_ALIASES = {
    "English": "en",
    "日本語": "ja",
    "Japanese": "ja",
    "中文": "zh",
    "Chinese": "zh",
}

CONNECTORS = {
    "en": {"and", "but", "because", "when", "while", "although", "if", "so"},
    "ja": {"そして", "しかし", "だから", "ので", "から", "けれど", "もし", "また"},
    "zh": {"而且", "但是", "因为", "所以", "如果", "然后", "虽然", "不过"},
}

EMOTION_WORDS = {
    "en": {
        "positive": {"good", "happy", "calm", "love", "great", "fine", "hope", "well"},
        "negative": {"bad", "sad", "angry", "tired", "worry", "worried", "fear", "pain"},
    },
    "ja": {
        "positive": {"良い", "嬉しい", "楽しい", "穏やか", "安心", "好き", "希望", "元気"},
        "negative": {"悪い", "悲しい", "怒り", "疲れた", "心配", "不安", "怖い", "痛い"},
    },
    "zh": {
        "positive": {"好", "开心", "高兴", "平静", "安心", "喜欢", "希望", "舒服"},
        "negative": {"不好", "难过", "生气", "疲惫", "担心", "焦虑", "害怕", "疼痛"},
    },
}


def language_code(language: str) -> str:
    return LANGUAGE_ALIASES.get(language, "en")


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def tokenize(text: str, language: str = "English") -> list[str]:
    code = language_code(language)
    if code == "ja":
        try:
            from janome.tokenizer import Tokenizer

            return [
                token.surface.lower()
                for token in Tokenizer().tokenize(text)
                if token.part_of_speech.split(",")[0] not in {"記号", "助詞", "助動詞"}
            ]
        except ImportError:
            return re.findall(
                r"[\u3400-\u9fff]+|[\u3040-\u309f]+|[\u30a0-\u30ff]+|[A-Za-z0-9]+",
                text,
            )
    if code == "zh":
        try:
            import jieba

            return [
                word.strip().lower()
                for word in jieba.cut(text)
                if re.search(r"[\w\u3400-\u9fff]", word)
            ]
        except ImportError:
            return re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+", text)
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def split_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[.!?。！？]+", text) if item.strip()]


def score_vocabulary_variety(words: list[str]) -> tuple[float, str]:
    total = len(words)
    unique = len(set(words))
    ratio = unique / total if total else 0.0
    return clamp_score(ratio * 100), f"unique_words={unique}; total_words={total}; ratio={ratio:.3f}"


def score_response_length(words: list[str], language: str) -> tuple[float, str]:
    total = len(words)
    target = {"en": 120, "ja": 100, "zh": 100}[language_code(language)]
    return clamp_score((total / target) * 100), f"units={total}; target={target}"


def score_sentence_complexity(
    text: str, words: list[str], sentences: list[str], language: str
) -> tuple[float, str]:
    sentence_count = len(sentences)
    avg_len = len(words) / sentence_count if sentence_count else 0.0
    connectors = CONNECTORS[language_code(language)]
    connector_count = sum(text.lower().count(connector) for connector in connectors)
    score = (min(avg_len, 20) / 20) * 70 + min(connector_count, 6) / 6 * 30
    return clamp_score(score), f"avg_sentence_length={avg_len:.2f}; connectors={connector_count}"


def score_speech_pace(
    words: list[str], duration_seconds: float | None, language: str
) -> tuple[float, str]:
    if not duration_seconds or duration_seconds <= 0:
        return 50.0, "duration_seconds=unknown; neutral_score_used=true"
    units_per_minute = len(words) / duration_seconds * 60
    center = {"en": 130, "ja": 110, "zh": 110}[language_code(language)]
    score = 100 - abs(units_per_minute - center) * 0.5
    return clamp_score(score), f"units_per_minute={units_per_minute:.2f}; center={center}"


def score_pause_pattern(acoustic_metrics: dict | None) -> tuple[float, str]:
    if not acoustic_metrics:
        return 50.0, "pause_analysis=unavailable; neutral_score_used=true"
    pause_ratio = float(acoustic_metrics.get("pause_ratio", 0.0))
    mean_pause = float(acoustic_metrics.get("mean_pause_seconds", 0.0))
    # A broad descriptive center avoids treating one exact speaking style as ideal.
    ratio_component = 100 - abs(pause_ratio - 0.25) * 140
    duration_component = 100 - max(0.0, mean_pause - 1.5) * 18
    score = ratio_component * 0.65 + duration_component * 0.35
    raw = "; ".join(
        f"{key}={value}"
        for key, value in acoustic_metrics.items()
        if key
        in {
            "voiced_seconds",
            "pause_seconds",
            "pause_count",
            "mean_pause_seconds",
            "max_pause_seconds",
            "pause_ratio",
        }
    )
    return clamp_score(score), raw


def score_repetition_pattern(words: list[str]) -> tuple[float, str]:
    if not words:
        return 0.0, "repeated_words=0; total_words=0"
    counts = Counter(words)
    repeated_instances = sum(count - 1 for count in counts.values() if count > 1)
    ratio = repeated_instances / len(words)
    score = 100 - ratio * 100
    return clamp_score(score), f"repeated_instances={repeated_instances}; ratio={ratio:.3f}"


def score_emotional_tone(text: str, language: str = "English") -> tuple[float, str]:
    lexicon = EMOTION_WORDS[language_code(language)]
    words = tokenize(text, language)
    positive = sum(1 for word in words if word in lexicon["positive"])
    negative = sum(1 for word in words if word in lexicon["negative"])
    total = positive + negative
    if total == 0:
        return 50.0, "positive_words=0; negative_words=0; neutral_score_used=true"
    score = ((positive - negative) / total + 1) * 50
    return clamp_score(score), f"positive_words={positive}; negative_words={negative}"


def score_transcription_clarity(text: str, language: str = "English") -> tuple[float, str]:
    words = tokenize(text, language)
    if len(words) >= 20:
        return 90.0, f"recognized_words={len(words)}"
    if len(words) >= 8:
        return 70.0, f"recognized_words={len(words)}"
    if len(words) > 0:
        return 45.0, f"recognized_words={len(words)}; short_transcript=true"
    return 0.0, "recognized_words=0; transcript_empty=true"


def calculate_feature_scores(
    text: str,
    duration_seconds: float | None = None,
    language: str = "English",
    acoustic_metrics: dict | None = None,
) -> list[dict]:
    words = tokenize(text, language)
    sentences = split_sentences(text)
    score_builders = [
        ("Vocabulary Variety", lambda: score_vocabulary_variety(words)),
        ("Response Length", lambda: score_response_length(words, language)),
        (
            "Sentence Complexity",
            lambda: score_sentence_complexity(text, words, sentences, language),
        ),
        ("Speech Pace", lambda: score_speech_pace(words, duration_seconds, language)),
        ("Pause Pattern", lambda: score_pause_pattern(acoustic_metrics)),
        ("Repetition Pattern", lambda: score_repetition_pattern(words)),
        ("Emotional Tone", lambda: score_emotional_tone(text, language)),
        ("Transcription Clarity", lambda: score_transcription_clarity(text, language)),
    ]
    results = []
    for feature_name, builder in score_builders:
        score, raw_metric = builder()
        results.append(
            {
                "feature_name": feature_name,
                "score": score,
                "raw_metric": raw_metric,
                "explanation": FEATURE_EXPLANATIONS[feature_name],
                "scoring_model_version": SCORING_MODEL_VERSION,
            }
        )
    return results
