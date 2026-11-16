from intelliquery.answering.citations import parse_sentences, render
from intelliquery.retrieval.fusion import reciprocal_rank_fusion


def test_sentences_are_split_and_sources_resolved() -> None:
    sentences = parse_sentences(
        "The limit is 8 C [2]. Alarms are sent by SMS [1][3]. It is very reliable.", 3
    )
    assert [(s.text, s.sources) for s in sentences] == [
        ("The limit is 8 C.", [2]),
        ("Alarms are sent by SMS.", [1, 3]),
        ("It is very reliable.", []),
    ]


def test_marker_before_the_full_stop_and_grouped_markers() -> None:
    sentences = parse_sentences("The limit is 8 C [2, 3]. Probes are wired [1].", 3)
    assert [(s.text, s.sources) for s in sentences] == [
        ("The limit is 8 C.", [2, 3]),
        ("Probes are wired.", [1]),
    ]


def test_markers_outside_the_provided_sources_are_ignored() -> None:
    (sentence,) = parse_sentences("The device is waterproof [7][0][2].", 3)
    assert sentence.sources == [2]
    (invented,) = parse_sentences("The device is waterproof [9].", 3)
    assert invented.sources == []


def test_decimal_numbers_and_abbreviations_do_not_split_sentences() -> None:
    sentences = parse_sentences("Accuracy is 0.5 C at 25.0 C [1]. Range is wide [2].", 2)
    assert [s.text for s in sentences] == ["Accuracy is 0.5 C at 25.0 C.", "Range is wide."]


def test_list_items_keep_their_bullets() -> None:
    sentences = parse_sentences("Steps:\n- Mount the base [1].\n2. Connect power [2].", 2)
    assert [(s.prefix, s.text, s.starts_line) for s in sentences] == [
        ("", "Steps:", True),
        ("- ", "Mount the base.", True),
        ("2. ", "Connect power.", True),
    ]


def test_render_renumbers_markers_in_order_of_first_use() -> None:
    sentences = parse_sentences("B is true [3]. A is true [1][3]. C is true [2].", 3)
    text, order = render(sentences)
    assert text == "B is true. [1] A is true. [2][1] C is true. [3]"
    assert order == [3, 1, 2]


def test_render_preserves_line_structure() -> None:
    sentences = [
        s for s in parse_sentences("Intro.\n- One [2].\n- Two [1]. More [1].", 2) if s.sources
    ]
    text, order = render(sentences)
    assert text == "- One. [1]\n- Two. [2] More. [2]"
    assert order == [2, 1]


def test_empty_and_marker_only_text_yield_nothing() -> None:
    assert parse_sentences("", 3) == []
    assert parse_sentences("[1] [2]", 3) == []


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    assert [item for item, _ in fused] == ["a", "c", "b", "d"]
    scores = dict(fused)
    assert scores["a"] == 1 / 61 + 1 / 62
    assert scores["d"] == 1 / 63


def test_rrf_handles_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], ["x"]]) == [("x", 1 / 61)]
