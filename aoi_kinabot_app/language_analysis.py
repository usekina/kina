"""Local, language-aware analysis for KinaBot speech samples."""

from __future__ import annotations

from scoring import calculate_feature_scores


LANGUAGE_CODES = {
    "English": "en",
    "日本語": "ja",
    "中文": "zh",
}


def analyze_transcript(
    transcript: str,
    language: str,
    duration_seconds: float | None,
) -> tuple[list[dict], str]:
    """Calculate sample features locally; transcript text never leaves the server."""
    scores = calculate_feature_scores(transcript, duration_seconds, language)
    summaries = {
        "English": "This summary describes observable patterns in this recording only.",
        "日本語": "この結果は、今回の録音で観察された特徴だけを表します。",
        "中文": "本结果仅描述这一次录音中可观察到的表达特点。",
    }
    return scores, summaries.get(language, summaries["English"])
