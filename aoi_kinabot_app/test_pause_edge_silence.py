from scoring import score_pause_pattern
from speech_to_text import calculate_pause_metrics


class Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def test_edge_padding_does_not_change_internal_pause_score():
    cropped = calculate_pause_metrics(
        [Segment(0.0, 2.0), Segment(3.0, 5.0)], 5.0
    )
    padded = calculate_pause_metrics(
        [Segment(2.0, 4.0), Segment(5.0, 7.0)], 10.0
    )

    for key in (
        "internal_pause_seconds",
        "speech_span_seconds",
        "pause_count",
        "mean_pause_seconds",
        "max_pause_seconds",
        "pause_ratio",
    ):
        assert padded[key] == cropped[key]
    assert score_pause_pattern(padded)[0] == score_pause_pattern(cropped)[0]
    assert padded["leading_silence_seconds"] == 2.0
    assert padded["trailing_silence_seconds"] == 3.0


def test_a_real_internal_gap_changes_pause_metrics_and_score():
    short_gap = calculate_pause_metrics(
        [Segment(0.0, 2.0), Segment(3.0, 5.0)], 5.0
    )
    long_gap = calculate_pause_metrics(
        [Segment(0.0, 2.0), Segment(4.0, 6.0)], 6.0
    )

    assert short_gap["internal_pause_seconds"] == 1.0
    assert long_gap["internal_pause_seconds"] == 2.0
    assert short_gap["pause_ratio"] == 0.2
    assert long_gap["pause_ratio"] == 0.3333
    assert score_pause_pattern(short_gap)[0] != score_pause_pattern(long_gap)[0]


def test_single_segment_reports_edge_silence_without_internal_pause():
    metrics = calculate_pause_metrics([Segment(2.0, 5.0)], 8.0)

    assert metrics["voiced_seconds"] == 3.0
    assert metrics["internal_pause_seconds"] == 0
    assert metrics["pause_ratio"] == 0.0
    assert metrics["leading_silence_seconds"] == 2.0
    assert metrics["trailing_silence_seconds"] == 3.0


def test_overlapping_out_of_order_segments_are_merged():
    metrics = calculate_pause_metrics(
        [Segment(4.0, 6.0), Segment(1.0, 3.0), Segment(2.0, 5.0)], 8.0
    )

    assert metrics["voiced_seconds"] == 5.0
    assert metrics["speech_span_seconds"] == 5.0
    assert metrics["internal_pause_seconds"] == 0
    assert metrics["pause_count"] == 0


def test_invalid_and_out_of_range_segments_are_handled_deterministically():
    metrics = calculate_pause_metrics(
        [
            Segment("invalid", 2.0),
            Segment(float("inf"), float("inf")),
            Segment(-2.0, 1.0),
            Segment(9.0, 12.0),
            Segment(4.0, 4.0),
        ],
        10.0,
    )

    assert metrics["voiced_seconds"] == 2.0
    assert metrics["internal_pause_seconds"] == 8.0
    assert metrics["speech_span_seconds"] == 10.0
    assert metrics["pause_ratio"] == 0.8


def test_no_valid_segments_returns_no_pause_analysis():
    assert calculate_pause_metrics([], 10.0) == {}
    assert calculate_pause_metrics([Segment(2.0, 2.0)], 10.0) == {}
