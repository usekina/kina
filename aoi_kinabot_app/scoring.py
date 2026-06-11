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


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def split_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[.!?。！？]+", text) if item.strip()]


def score_vocabulary_variety(words: list[str]) -> tuple[float, str]:
    total = len(words)
    unique = len(set(words))
    ratio = unique / total if total else 0.0
    return clamp_score(ratio * 100), f"unique_words={unique}; total_words={total}; ratio={ratio:.3f}"


def score_response_length(words: list[str]) -> tuple[float, str]:
    total = len(words)
    # 120 words is treated as a strong short reflection sample for V1.
    return clamp_score((total / 120) * 100), f"total_words={total}"


def score_sentence_complexity(words: list[str], sentences: list[str]) -> tuple[float, str]:
    sentence_count = len(sentences)
    avg_len = len(words) / sentence_count if sentence_count else 0.0
    connectors = {"and", "but", "because", "when", "while", "although", "if", "so"}
    connector_count = sum(1 for word in words if word in connectors)
    score = (min(avg_len, 20) / 20) * 70 + min(connector_count, 6) / 6 * 30
    return clamp_score(score), f"avg_sentence_length={avg_len:.2f}; connectors={connector_count}"


def score_speech_pace(words: list[str], duration_seconds: float | None) -> tuple[float, str]:
    if not duration_seconds or duration_seconds <= 0:
        return 50.0, "duration_seconds=unknown; neutral_score_used=true"
    words_per_minute = len(words) / duration_seconds * 60
    # Center the V1 display around a broad conversational range.
    score = 100 - abs(words_per_minute - 130) * 0.5
    return clamp_score(score), f"words_per_minute={words_per_minute:.2f}"


def score_pause_pattern() -> tuple[float, str]:
    # Placeholder until audio pause extraction is implemented.
    return 50.0, "pause_analysis=v1_placeholder"


def score_repetition_pattern(words: list[str]) -> tuple[float, str]:
    if not words:
        return 0.0, "repeated_words=0; total_words=0"
    counts = Counter(words)
    repeated_instances = sum(count - 1 for count in counts.values() if count > 1)
    ratio = repeated_instances / len(words)
    score = 100 - ratio * 100
    return clamp_score(score), f"repeated_instances={repeated_instances}; ratio={ratio:.3f}"


def score_emotional_tone(text: str) -> tuple[float, str]:
    positive_words = {"good", "happy", "calm", "love", "great", "fine", "hope", "well"}
    negative_words = {"bad", "sad", "angry", "tired", "worry", "worried", "fear", "pain"}
    words = tokenize(text)
    positive = sum(1 for word in words if word in positive_words)
    negative = sum(1 for word in words if word in negative_words)
    total = positive + negative
    if total == 0:
        return 50.0, "positive_words=0; negative_words=0; neutral_score_used=true"
    score = ((positive - negative) / total + 1) * 50
    return clamp_score(score), f"positive_words={positive}; negative_words={negative}"


def score_transcription_clarity(text: str) -> tuple[float, str]:
    words = tokenize(text)
    if len(words) >= 20:
        return 90.0, f"recognized_words={len(words)}"
    if len(words) >= 8:
        return 70.0, f"recognized_words={len(words)}"
    if len(words) > 0:
        return 45.0, f"recognized_words={len(words)}; short_transcript=true"
    return 0.0, "recognized_words=0; transcript_empty=true"


def calculate_feature_scores(text: str, duration_seconds: float | None = None) -> list[dict]:
    words = tokenize(text)
    sentences = split_sentences(text)
    score_builders = [
        ("Vocabulary Variety", lambda: score_vocabulary_variety(words)),
        ("Response Length", lambda: score_response_length(words)),
        ("Sentence Complexity", lambda: score_sentence_complexity(words, sentences)),
        ("Speech Pace", lambda: score_speech_pace(words, duration_seconds)),
        ("Pause Pattern", score_pause_pattern),
        ("Repetition Pattern", lambda: score_repetition_pattern(words)),
        ("Emotional Tone", lambda: score_emotional_tone(text)),
        ("Transcription Clarity", lambda: score_transcription_clarity(text)),
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
