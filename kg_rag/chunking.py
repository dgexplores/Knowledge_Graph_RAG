"""Document chunking with stable, content-derived ids.

Chunk ids are the join key between Neo4j `source_chunk_id` and pgvector `chunks.chunk_id`.
Two properties are load-bearing (docs/02-data-model.md):

1. **Stable** — the same input always produces the same id, so re-ingestion is idempotent and a
   citation from Phase 4 resolves in both stores.
2. **Content-addressed** — the id changes when the text changes, so a stale citation can never
   silently resolve to edited text.

Id format: `{prefix}-{year}-{anchor:03d}#{ordinal:04d}`
    prefix  short form type, e.g. `10k`, `10q`, `8k`
    year    the filing's fiscal year, not the current year
    anchor  zero-padded line number where the chunk starts, for human debugging
    ordinal position of the chunk within its section
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

DEFAULT_TARGET_TOKENS = 350
DEFAULT_OVERLAP_TOKENS = 60

_HORIZONTAL = re.compile(r"[ \t\r\f\v]+")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_PARA = "\x00PARA\x00"  # sentinel: a real newline is a paragraph boundary, not whitespace

FORM_PREFIXES = {
    "10-K": "10k",
    "10-K/A": "10ka",
    "10-Q": "10q",
    "10-Q/A": "10qa",
    "8-K": "8k",
    "8-K/A": "8ka",
}


class ChunkingError(ValueError):
    """Chunking input cannot produce chunks — surfaced instead of silently returning none."""


@dataclass(frozen=True)
class Chunk:
    """One indexed passage."""

    chunk_id: str
    document_id: str
    accession: str | None
    section_path: str
    text: str
    token_count: int
    filed_at: str | None = None
    entities: tuple[str, ...] = field(default=())

    def as_metadata(self) -> dict:
        """Metadata dict for the pgvector row. `text` and `embedding` are added by the indexer."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "accession": self.accession,
            "section_path": self.section_path,
            "token_count": self.token_count,
            "filed_at": self.filed_at,
            "entities": list(self.entities),
        }


def form_prefix(form: str) -> str:
    """Short, filesystem-safe prefix for a form type. Unknown forms are slugified, not rejected."""
    if form in FORM_PREFIXES:
        return FORM_PREFIXES[form]
    slug = re.sub(r"[^a-z0-9]+", "", form.lower())
    if not slug:
        raise ChunkingError(f"form type produced an empty prefix: {form!r}")
    return slug[:8]


def clean_text(text: str) -> str:
    """Collapse whitespace while preserving paragraph boundaries.

    A run of two or more newlines is a paragraph boundary and survives as `\n\n` — the structure
    that tells the extractor which sentences are siblings rather than one run-on passage. Every
    other newline, and every run of horizontal whitespace, collapses to a single space: an inline
    newline in HTML-extracted text carries no meaning the tokenizer would not lose, and preserving
    it would make chunk ids depend on source formatting rather than content.
    """
    if not text:
        return ""
    out = text.replace("\xa0", " ")
    out = _HORIZONTAL.sub(" ", out)
    out = _PARAGRAPH_BREAK.sub(_PARA, out)
    out = out.replace("\n", " ")
    out = _HORIZONTAL.sub(" ", out)
    return out.replace(_PARA, "\n\n").strip()


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenization with paragraph markers preserved as boundary signals.

    A dependency-free tokenizer is deliberate. Chunk boundaries must be identical on every
    machine and in CI, because chunk ids are content-addressed — a tokenizer whose output depends
    on an installed model revision would change ids between runs and break idempotency.
    """
    return [t for t in text.split() if t]


def _pack(token_count: int, target: int, overlap: int) -> list[tuple[int, int]]:
    """Greedy token ranges of ~`target` tokens with `overlap` trailing tokens repeated.

    Returns half-open `(start, end)` index ranges rather than token lists. Ranges are what makes
    line anchors correct: resolving a token back to its position by searching for its text would
    return the wrong offset whenever a token repeats in the document, which in a filing of
    boilerplate it always does.
    """
    if target <= 0:
        raise ChunkingError(f"target_tokens must be positive, got {target}")
    if overlap < 0:
        raise ChunkingError(f"overlap_tokens must be non-negative, got {overlap}")
    if overlap >= target:
        raise ChunkingError(
            f"overlap_tokens ({overlap}) must be less than target_tokens ({target}); "
            "otherwise packing cannot advance"
        )
    if token_count <= 0:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < token_count:
        end = min(start + target, token_count)
        ranges.append((start, end))
        if end == token_count:
            break
        start = end - overlap
    return ranges


def make_chunk_id(
    prefix: str,
    fiscal_year: int | str,
    anchor_line: int,
    ordinal: int,
    *,
    text: str | None = None,
    section_path: str | None = None,
    accession: str | None = None,
) -> str:
    """Build a chunk id.

    `text`, `section_path` and `accession` are mixed into the hash when supplied, which makes the
    id change if any of them change. Without them the id is positional only — stable across
    re-runs of unchanged input, but blind to edits that shift content without shifting position.
    All three should be passed.
    """
    base = f"{prefix}-{fiscal_year}-{anchor_line:03d}#{ordinal:04d}"
    if text is None and section_path is None and accession is None:
        return base
    digest = hashlib.sha256(
        f"{base}|{accession or ''}|{section_path or ''}|{text or ''}".encode()
    ).hexdigest()[:8]
    return f"{base}-{digest}"


def _line_anchor(text: str, char_offset: int) -> int:
    """1-based line number where `char_offset` falls, for human-readable ids."""
    return text.count("\n", 0, char_offset) + 1


def chunk_document(
    text: str,
    *,
    form: str,
    fiscal_year: int | str,
    accession: str | None = None,
    document_id: str | None = None,
    sections: dict[str, str] | None = None,
    filed_at: str | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split a filing into citable chunks.

    Pass `sections` to preserve the filing's own structure — a chunk's `section_path` becomes
    part of its id, so `Item 1A/Risk Factors` chunks stay distinct from `Item 7/MD&A` chunks even
    when the text is identical. `sections` may be supplied with empty `text`, since the sections
    are the content in that case.

    Returns an empty list for empty or whitespace-only input rather than raising: a filing
    section that is genuinely blank is normal, and ingestion should skip it, not crash.
    """
    sections = dict(sections) if sections else None
    cleaned = clean_text(text)
    if not cleaned and not sections:
        return []

    prefix = form_prefix(form)
    doc_id = document_id or accession or f"{prefix}-{fiscal_year}"

    # (section_path, section_text), preserving the order sections were supplied.
    section_items = list(sections.items()) if sections else [("", cleaned)]
    if not sections and "\n\n" in cleaned:
        # No explicit structure: treat blank-line paragraphs as sections so ids stay stable when
        # unrelated text elsewhere in the document changes.
        parts = [p for p in cleaned.split("\n\n") if p.strip()]
        if len(parts) > 1:
            section_items = [(f"p{i:04d}", p) for i, p in enumerate(parts)]

    chunks: list[Chunk] = []
    consumed = 0
    for section_path, section_text in section_items:
        section = clean_text(section_text)
        if not section:
            continue

        # Global character offset of this section, for line anchors that match the original file.
        section_offset = cleaned.find(section, consumed)
        if section_offset < 0:
            section_offset = consumed
        else:
            consumed = section_offset + len(section)

        tokens = _tokenize(section)
        if not tokens:
            continue

        # Recover each window's character offset so line anchors are real, not sequential guesses.
        offsets: list[int] = []
        cursor = 0
        for token in tokens:
            found = section.find(token, cursor)
            if found < 0:  # unreachable for whitespace tokens, but never emit a wrong anchor
                found = cursor
            offsets.append(found)
            cursor = found + len(token)

        for ordinal, (start_idx, end_idx) in enumerate(
            _pack(len(tokens), target_tokens, overlap_tokens)
        ):
            window = tokens[start_idx:end_idx]
            char_start = section_offset + offsets[start_idx]
            anchor = _line_anchor(cleaned, char_start)

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(
                        prefix,
                        fiscal_year,
                        anchor,
                        ordinal,
                        text=" ".join(window),
                        section_path=section_path,
                        accession=accession,
                    ),
                    document_id=doc_id,
                    accession=accession,
                    section_path=section_path,
                    text=" ".join(window),
                    token_count=len(window),
                    filed_at=filed_at,
                )
            )

    return chunks
