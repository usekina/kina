from scoring import (
    count_english_connectors,
    score_sentence_complexity,
    split_sentences,
    tokenize,
)


def sentence_score(text: str) -> tuple[float, str]:
    words = tokenize(text, "English")
    return score_sentence_complexity(
        text, words, split_sentences(text), "English"
    )


def test_english_connector_substrings_do_not_increase_complexity():
    score, raw_metric = sentence_score("Candy is a gift.")

    assert score == 14.0
    assert raw_metric == "avg_sentence_length=4.00; connectors=0"

    _, softly_metric = sentence_score("They moved softly.")
    assert softly_metric.endswith("connectors=0")


def test_real_english_connectors_ignore_case_and_punctuation():
    _, raw_metric = sentence_score("And, if it rains, stay because it is wet.")

    assert raw_metric.endswith("connectors=3")


def test_repeated_english_connectors_are_counted_repeatedly():
    _, raw_metric = sentence_score("If it rains and if it snows, wait.")

    assert raw_metric.endswith("connectors=3")


def test_multiword_connectors_are_counted_as_token_sequences():
    words = tokenize("Even though we left, even thoughts remained.", "English")

    assert count_english_connectors(words, {"even though"}) == 1
