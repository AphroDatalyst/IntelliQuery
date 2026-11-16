from intelliquery.db.models import ChunkModality
from intelliquery.ingestion.chunking import build_chunks, estimate_tokens
from intelliquery.ingestion.types import BBox, Element, ElementKind


def text(value: str, page: int = 1, box: BBox | None = None) -> Element:
    return Element(ElementKind.TEXT, value, page, box)


def heading(value: str, level: int = 1) -> Element:
    return Element(ElementKind.HEADING, value, 1, None, level=level)


def test_text_is_grouped_under_its_heading_and_not_across_sections() -> None:
    chunks = build_chunks(
        [heading("Setup"), text("Step one."), text("Step two."), heading("Usage"), text("Run it.")]
    )
    assert [(c.section_path, c.content) for c in chunks] == [
        ("Setup", "Step one.\nStep two."),
        ("Usage", "Run it."),
    ]


def test_section_path_tracks_heading_depth() -> None:
    chunks = build_chunks(
        [
            heading("Guide", 1),
            heading("Install", 2),
            text("a"),
            heading("Wiring", 3),
            text("b"),
            heading("Operate", 2),
            text("c"),
            heading("Appendix", 1),
            text("d"),
        ]
    )
    assert [c.section_path for c in chunks] == [
        "Guide > Install",
        "Guide > Install > Wiring",
        "Guide > Operate",
        "Appendix",
    ]


def test_buffer_is_flushed_when_token_budget_is_reached() -> None:
    sentence = "alpha beta gamma delta epsilon."  # 6 tokens
    chunks = build_chunks([text(sentence) for _ in range(5)], max_tokens=12)
    assert [c.content.count(sentence) for c in chunks] == [2, 2, 1]
    assert all(c.token_count <= 12 for c in chunks)


def test_oversized_block_is_split_at_sentence_boundaries() -> None:
    block = " ".join(f"Sentence number {i} ends here." for i in range(20))
    chunks = build_chunks([text(block)], max_tokens=30)
    assert len(chunks) > 1
    assert all(c.token_count <= 30 for c in chunks)
    assert all(c.content.endswith(".") for c in chunks)
    assert " ".join(c.content for c in chunks) == block


def test_block_without_sentence_boundaries_is_still_bounded() -> None:
    chunks = build_chunks([text(" ".join(["word"] * 500))], max_tokens=50)
    assert all(c.token_count <= 50 for c in chunks)
    assert sum(c.token_count for c in chunks) == 500


def test_tables_and_figures_are_their_own_chunks_in_reading_order() -> None:
    chunks = build_chunks(
        [
            heading("Results"),
            text("Before."),
            Element(ElementKind.TABLE, "| a | b |", 2, None, caption="Table 1"),
            Element(ElementKind.FIGURE, "", 2, None, caption="Figure 1", image=b"png"),
            text("After."),
        ]
    )
    assert [c.modality for c in chunks] == [
        ChunkModality.TEXT,
        ChunkModality.TABLE,
        ChunkModality.FIGURE,
        ChunkModality.TEXT,
    ]
    assert chunks[1].content == "Table 1\n\n| a | b |"
    assert chunks[2].image == b"png"
    assert {c.section_path for c in chunks} == {"Results"}


def test_regions_cover_every_page_a_chunk_spans() -> None:
    chunks = build_chunks(
        [
            text("one", 1, BBox(10, 10, 50, 20)),
            text("two", 1, BBox(5, 30, 80, 40)),
            text("three", 2, BBox(1, 2, 3, 4)),
        ]
    )
    (chunk,) = chunks
    assert chunk.page_number == 1
    assert chunk.bbox == BBox(5, 10, 80, 40)
    assert chunk.regions == [
        {"page": 1, "bbox": {"x0": 5, "y0": 10, "x1": 80, "y1": 40}},
        {"page": 2, "bbox": {"x0": 1, "y0": 2, "x1": 3, "y1": 4}},
    ]


def test_retrieval_text_is_prefixed_with_the_section_path() -> None:
    (chunk,) = build_chunks([heading("Alarms"), text("Limit is 8 C.")])
    assert chunk.retrieval_text == "Alarms\n\nLimit is 8 C."
    assert chunk.content == "Limit is 8 C."


def test_empty_input_and_blank_elements_produce_nothing() -> None:
    assert build_chunks([]) == []
    assert build_chunks([heading("  "), text("   "), Element(ElementKind.TABLE, "", 1)]) == []


def test_estimate_tokens_counts_words_and_punctuation() -> None:
    assert estimate_tokens("Hello, world!") == 4
