"""Descriptive, provenance-matched radar data; never a cognitive score."""

from __future__ import annotations

import math
from statistics import mean

from history_view import comparison_key


FEATURES = (
    "Vocabulary Variety", "Response Length", "Sentence Complexity", "Speech Pace",
    "Pause Pattern", "Repetition Pattern", "Transcription Clarity", "Emotional Tone",
)


def measured_score(row: dict) -> float | None:
    """Do not plot unavailable, nonfinite, out-of-range or fallback values."""
    if row.get("availability_status") == "unavailable":
        return None
    if row.get("feature_name") == "Emotional Tone":
        raw = str(row.get("raw_metric") or "").lower()
        if "neutral_score_used=true" in raw:
            return None
        # An unproven legacy 50 cannot be distinguished from the default.
        if not raw and row.get("score") == 50:
            return None
    try:
        value = float(row["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 100 else None


def radar_comparison(records: list[dict], session_id: int | None = None) -> dict:
    """Compare a selected session with the last three *earlier* compatible ones.

    A reference is only shown for a feature measured in all three sessions.
    References are recent averages, not validated personal baselines. Unknown
    provenance is never matched. The selected session is never in its reference.
    """
    sessions: dict[int, dict[str, dict]] = {}
    for source in records:
        row = dict(source)
        if row.get("session_id") is None:
            continue
        sessions.setdefault(int(row["session_id"]), {})[row["feature_name"]] = row
    if not sessions:
        return {}
    selected_id = max(sessions) if session_id is None else session_id
    if selected_id not in sessions:
        return {}
    selected = sessions[selected_id]
    metadata = next(iter(selected.values()))
    key = comparison_key(metadata)
    trusted_key = all(part and part not in {"unknown", "legacy-unknown"} for part in key)
    earlier = sorted(
        sid for sid, scores in sessions.items()
        if sid < selected_id and trusted_key
        and comparison_key(next(iter(scores.values()))) == key
    )[-3:]
    current = [measured_score(selected.get(name, {})) for name in FEATURES]
    reference = []
    for name in FEATURES:
        values = [measured_score(sessions[sid].get(name, {})) for sid in earlier]
        reference.append(
            round(mean(values), 1)
            if len(values) == 3 and all(v is not None for v in values) else None
        )
    return {
        "session_id": selected_id, "metadata": metadata, "features": FEATURES,
        "current": current, "reference": reference, "reference_ids": earlier,
        "reference_count": len(earlier), "key": key,
    }
