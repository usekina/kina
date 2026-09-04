"""V1 speech and language feature scoring.

These scores are for cognitive wellness reflection and trend awareness.
They are not medical diagnosis, cognitive age, or disease risk prediction.
"""

from __future__ import annotations

import re
from collections import Counter

from config import SCORING_MODEL_VERSION


FEATURE_LABELS = {
    "en": {
        "Vocabulary Variety": "Vocabulary variety",
        "Response Length": "Response length",
        "Sentence Complexity": "Sentence structure",
        "Speech Pace": "Speech pace",
        "Pause Pattern": "Pause pattern",
        "Repetition Pattern": "Repetition pattern",
        "Emotional Tone": "Emotional tone",
        "Transcription Clarity": "Recording clarity",
    },
    "ja": {
        "Vocabulary Variety": "語彙の多様性",
        "Response Length": "発話量",
        "Sentence Complexity": "文の構成",
        "Speech Pace": "話す速さ",
        "Pause Pattern": "間の取り方",
        "Repetition Pattern": "繰り返し",
        "Emotional Tone": "感情表現",
        "Transcription Clarity": "録音の明瞭さ",
    },
    "zh": {
        "Vocabulary Variety": "词汇多样性",
        "Response Length": "表达长度",
        "Sentence Complexity": "句子结构",
        "Speech Pace": "说话速度",
        "Pause Pattern": "停顿模式",
        "Repetition Pattern": "重复模式",
        "Emotional Tone": "情绪表达",
        "Transcription Clarity": "录音清晰度",
    },
}

FEATURE_EXPLANATIONS = {
    "en": {
        "Vocabulary Variety": "Range of different words used in this recording.",
        "Response Length": "Amount of speech captured in this recording.",
        "Sentence Complexity": "Variation and connection within sentence structure.",
        "Speech Pace": "Speaking speed in this recording; faster or slower is not inherently better.",
        "Pause Pattern": "Timing and duration of pauses detected in the audio.",
        "Repetition Pattern": "Frequency of repeated words in this recording.",
        "Emotional Tone": "Positive and negative wording detected; topic and context can affect this result.",
        "Transcription Clarity": "How much speech was clear enough for reliable transcription.",
    },
    "ja": {
        "Vocabulary Variety": "今回の録音で使われた言葉の種類の広さです。",
        "Response Length": "今回の録音で分析できた発話量です。",
        "Sentence Complexity": "文の長さや接続表現から見た構成の豊かさです。",
        "Speech Pace": "今回の話す速さです。速い・遅いだけで良し悪しは決まりません。",
        "Pause Pattern": "音声から検出した間の回数と長さです。",
        "Repetition Pattern": "今回の録音で同じ言葉が繰り返された頻度です。",
        "Emotional Tone": "肯定的・否定的な言葉の傾向です。話題や状況に影響されます。",
        "Transcription Clarity": "音声を安定して文字にできた程度です。",
    },
    "zh": {
        "Vocabulary Variety": "这次录音中使用了多少种不同词语。",
        "Response Length": "这次录音中可用于分析的表达量。",
        "Sentence Complexity": "句子长度和连接方式所体现的结构丰富度。",
        "Speech Pace": "这次录音的说话速度；快慢本身不代表好坏。",
        "Pause Pattern": "音频中检测到的停顿次数和持续时间。",
        "Repetition Pattern": "这次录音中相同词语重复出现的频率。",
        "Emotional Tone": "用词中呈现的积极或消极倾向；话题和情境会影响结果。",
        "Transcription Clarity": "语音能够被稳定转写和分析的清晰程度。",
    },
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


def display_feature_name(feature_name: str, language: str) -> str:
    code = language_code(language)
    return FEATURE_LABELS.get(code, FEATURE_LABELS["en"]).get(
        feature_name, feature_name
    )


def feature_explanation(feature_name: str, language: str) -> str:
    code = language_code(language)
    return FEATURE_EXPLANATIONS.get(code, FEATURE_EXPLANATIONS["en"]).get(
        feature_name, ""
    )


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


def count_english_connectors(words: list[str], connectors: set[str]) -> int:
    """Count connector token sequences without matching inside other words."""
    normalized_words = [word.lower() for word in words]
    connector_count = 0
    for connector in connectors:
        connector_words = tokenize(connector, "English")
        width = len(connector_words)
        if not width:
            continue
        connector_count += sum(
            normalized_words[index : index + width] == connector_words
            for index in range(len(normalized_words) - width + 1)
        )
    return connector_count


def score_sentence_complexity(
    text: str, words: list[str], sentences: list[str], language: str
) -> tuple[float, str]:
    sentence_count = len(sentences)
    avg_len = len(words) / sentence_count if sentence_count else 0.0
    code = language_code(language)
    connectors = CONNECTORS[code]
    if code == "en":
        connector_count = count_english_connectors(words, connectors)
    else:
        # Japanese and Chinese connectors do not follow English word boundaries.
        connector_count = sum(text.lower().count(connector) for connector in connectors)
    score = (min(avg_len, 20) / 20) * 70 + min(connector_count, 6) / 6 * 30
    return clamp_score(score), f"avg_sentence_length={avg_len:.2f}; connectors={connector_count}"


def score_speech_pace(
    words: list[str], duration_seconds: float | None, language: str
) -> tuple[float | None, str, str | None]:
    if not duration_seconds or duration_seconds <= 0:
        return None, "duration_seconds=unknown", "duration_unknown"
    units_per_minute = len(words) / duration_seconds * 60
    center = {"en": 130, "ja": 110, "zh": 110}[language_code(language)]
    score = 100 - abs(units_per_minute - center) * 0.5
    return clamp_score(score), f"units_per_minute={units_per_minute:.2f}; center={center}", None


def score_pause_pattern(acoustic_metrics: dict | None) -> tuple[float | None, str, str | None]:
    if not acoustic_metrics:
        return None, "pause_analysis=unavailable", "acoustic_analysis_unavailable"
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
            "internal_pause_seconds",
            "speech_span_seconds",
            "leading_silence_seconds",
            "trailing_silence_seconds",
            "pause_count",
            "mean_pause_seconds",
            "max_pause_seconds",
            "pause_ratio",
        }
    )
    return clamp_score(score), raw, None


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
        built = builder()
        if len(built) == 2:
            score, raw_metric = built
            failure_reason = None
        else:
            score, raw_metric, failure_reason = built
        results.append(
            {
                "feature_name": feature_name,
                "score": score,
                "availability_status": "unavailable" if score is None else "available",
                "failure_reason": failure_reason,
                "raw_metric": raw_metric,
                "explanation": feature_explanation(feature_name, language),
                "scoring_model_version": SCORING_MODEL_VERSION,
            }
        )
    return results
