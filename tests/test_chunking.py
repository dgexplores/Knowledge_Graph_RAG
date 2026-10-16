"""Tests for chunking.

The properties pinned here are what make hybrid retrieval work: ids are stable across runs,
ids change when content changes, ids are unique within a document, and overlapping chunks really
overlap. If any of these regress, citations silently resolve to the wrong passage.
"""

from __future__ import annotations

import pytest

from kg_rag.chunking import (
    ChunkingError,
    chunk_document,
    clean_text,
    form_prefix,
    make_chunk_id,
)


@pytest.fixture
def body() -> str:
    return " ".join(f"token{i}" for i in range(1000))


class TestFormPrefix:
    @pytest.mark.parametrize(
        ("form", "expected"),
        [("10-K", "10k"), ("10-Q", "10q"), ("8-K", "8k"), ("10-K/A", "10ka")],
    )
    def test_known_forms(self, form, expected):
        assert form_prefix(form) == expected

    def test_unknown_form_is_slugified_not_rejected(self):
        """A new form type should not stop ingestion."""
        assert form_prefix("S-1") == "s1"
        assert form_prefix("20-F") == "20f"

    def test_form_with_no_usable_characters_raises(self):
        with pytest.raises(ChunkingError, match="empty prefix"):
            form_prefix("---")


class TestCleanText:
    def test_collapses_runs_of_spaces(self):
        assert clean_text("a    b\t\tc") == "a b c"

    def test_preserves_paragraph_boundaries(self):
        """Paragraph structure carries list semantics the extractor needs."""
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_inline_newlines_collapse_to_spaces(self):
        """An inline newline in HTML-extracted text carries no meaning, and preserving it would
        make chunk ids depend on source formatting rather than content."""
        assert clean_text("line one\nline two\n\npara two") == "line one line two\n\npara two"

    def test_non_breaking_space_becomes_a_space(self):
        assert clean_text("a\xa0b") == "a b"

    @pytest.mark.parametrize("raw", ["", "   ", "\n\n\n"])
    def test_blank_input(self, raw):
        assert clean_text(raw) == ""


class TestChunkId:
    def test_positional_id_without_content_hash(self):
        assert make_chunk_id("10k", 2024, 12, 3) == "10k-2024-012#0003"

    def test_content_hash_appended_when_text_supplied(self):
        a = make_chunk_id("10k", 2024, 12, 3, text="hello", section_path="Item 1")
        assert a.startswith("10k-2024-012#0003-")
        assert len(a) == len("10k-2024-012#0003-") + 8

    def test_id_changes_when_text_changes(self):
        """A stale citation must never silently resolve to edited text."""
        a = make_chunk_id("10k", 2024, 12, 3, text="original", section_path="Item 1")
        b = make_chunk_id("10k", 2024, 12, 3, text="edited", section_path="Item 1")
        assert a != b

    def test_id_changes_when_section_changes(self):
        a = make_chunk_id("10k", 2024, 12, 3, text="same", section_path="Item 1A")
        b = make_chunk_id("10k", 2024, 12, 3, text="same", section_path="Item 7")
        assert a != b

    def test_anchor_is_zero_padded_to_three_digits(self):
        assert make_chunk_id("10k", 2024, 7, 0).startswith("10k-2024-007#")

    def test_ordinal_is_zero_padded_to_four_digits(self):
        assert "#0000" in make_chunk_id("10k", 2024, 7, 0)
        assert "#0042" in make_chunk_id("10k", 2024, 7, 42)


class TestChunkDocument:
    def test_empty_input_returns_no_chunks_rather_than_raising(self):
        """A blank filing section is normal; ingestion should skip it, not crash."""
        assert chunk_document("", form="10-K", fiscal_year=2024) == []
        assert chunk_document("   \n\n ", form="10-K", fiscal_year=2024) == []

    def test_short_document_yields_one_chunk(self):
        chunks = chunk_document("a short filing section", form="10-K", fiscal_year=2024)
        assert len(chunks) == 1
        assert chunks[0].token_count == 4

    def test_long_document_is_split(self):
        chunks = chunk_document(
            " ".join(f"t{i}" for i in range(1000)), form="10-K", fiscal_year=2024,
            target_tokens=100, overlap_tokens=10,
        )
        assert len(chunks) > 1
        assert all(c.token_count <= 100 for c in chunks)

    def test_ids_are_unique_within_a_document(self):
        """Duplicate ids would make two passages share one vector row."""
        chunks = chunk_document(
            " ".join(["item"] * 500), form="10-K", fiscal_year=2024,
            target_tokens=50, overlap_tokens=5,
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "duplicate chunk ids in one document"

    def test_ids_are_stable_across_runs(self):
        """Idempotent ingestion depends on this."""
        text = " ".join(f"t{i}" for i in range(500))
        first = [c.chunk_id for c in chunk_document(text, form="10-K", fiscal_year=2024)]
        second = [c.chunk_id for c in chunk_document(text, form="10-K", fiscal_year=2024)]
        assert first == second

    def test_ids_are_unique_across_documents_with_the_same_content(self):
        """Two filings with identical text must not collide on chunk id."""
        text = "identical boilerplate"
        a = chunk_document(text, form="10-K", fiscal_year=2024, accession="acc-a")[0]
        b = chunk_document(text, form="10-K", fiscal_year=2024, accession="acc-b")[0]
        assert a.chunk_id != b.chunk_id

    def test_overlap_repeats_boundary_tokens(self):
        """Overlap is what stops a relationship spanning a boundary from being lost."""
        text = " ".join(f"t{i}" for i in range(200))
        chunks = chunk_document(
            text, form="10-K", fiscal_year=2024, target_tokens=100, overlap_tokens=20
        )
        first_tail = chunks[0].text.split()[-20:]
        second_head = chunks[1].text.split()[:20]
        assert first_tail == second_head

    def test_no_tokens_are_lost(self):
        text = " ".join(f"t{i}" for i in range(250))
        chunks = chunk_document(
            text, form="10-K", fiscal_year=2024, target_tokens=100, overlap_tokens=20
        )
        recovered = {tok for c in chunks for tok in c.text.split()}
        assert recovered == set(text.split())

    def test_sections_are_chunked_independently(self):
        sections = {
            "Item 1A/Risk Factors": " ".join(f"risk{i}" for i in range(200)),
            "Item 7/MD&A": " ".join(f"mda{i}" for i in range(200)),
        }
        chunks = chunk_document(
            "", form="10-K", fiscal_year=2024, sections=sections,
            target_tokens=100, overlap_tokens=10,
        )
        paths = {c.section_path for c in chunks}
        assert paths == {"Item 1A/Risk Factors", "Item 7/MD&A"}

    def test_identical_text_in_two_sections_gets_distinct_ids(self):
        """Same boilerplate in Item 1 and Item 7 must not collide."""
        sections = {"Item 1": "identical text", "Item 7": "identical text"}
        chunks = chunk_document("", form="10-K", fiscal_year=2024, sections=sections)
        assert len({c.chunk_id for c in chunks}) == 2

    def test_document_id_defaults_to_accession(self):
        chunks = chunk_document("some text", form="10-K", fiscal_year=2024, accession="acc-1")
        assert chunks[0].document_id == "acc-1"

    def test_metadata_round_trips_the_join_key(self):
        """`chunk_id` must survive into the pgvector metadata or the hybrid join breaks."""
        chunks = chunk_document("some text", form="10-K", fiscal_year=2024, accession="acc-1")
        meta = chunks[0].as_metadata()
        assert meta["chunk_id"] == chunks[0].chunk_id
        assert meta["accession"] == "acc-1"
        assert meta["entities"] == []

    def test_line_anchor_is_monotonic(self):
        """Anchors must increase through the document for ids to sort meaningfully."""
        text = "\n".join(" ".join(f"t{i}" for i in range(20)) for _ in range(20))
        chunks = chunk_document(text, form="10-K", fiscal_year=2024, target_tokens=50, overlap_tokens=5)
        anchors = [int(c.chunk_id.split("-")[2].split("#")[0]) for c in chunks]
        assert anchors == sorted(anchors)

    def test_line_anchor_is_correct_with_repeated_tokens(self):
        """Regression: resolving a token back to its position by text search returned the wrong
        offset whenever a token repeated, which filings do constantly."""
        line = "the company the company the company the company"
        text = "\n".join([line] * 30)
        chunks = chunk_document(text, form="10-K", fiscal_year=2024, target_tokens=40, overlap_tokens=4)
        # Every chunk's anchor must point at a line that actually begins its text.
        lines = text.split("\n")
        for c in chunks:
            anchor = int(c.chunk_id.split("-")[2].split("#")[0])
            assert c.text.split()[0] in lines[anchor - 1]

    def test_zero_target_is_rejected(self):
        with pytest.raises(ChunkingError, match="target_tokens must be positive"):
            chunk_document("text", form="10-K", fiscal_year=2024, target_tokens=0)

    def test_overlap_not_smaller_than_target_is_rejected(self):
        """Silently looping forever on overlap >= target is worse than raising."""
        with pytest.raises(ChunkingError, match="must be less than target_tokens"):
            chunk_document("text", form="10-K", fiscal_year=2024, target_tokens=50, overlap_tokens=50)

    def test_negative_overlap_is_rejected(self):
        with pytest.raises(ChunkingError, match="must be non-negative"):
            chunk_document("text", form="10-K", fiscal_year=2024, overlap_tokens=-1)

    def test_fiscal_year_is_part_of_the_id(self):
        a = chunk_document("text", form="10-K", fiscal_year=2023)[0]
        b = chunk_document("text", form="10-K", fiscal_year=2024)[0]
        assert a.chunk_id != b.chunk_id
