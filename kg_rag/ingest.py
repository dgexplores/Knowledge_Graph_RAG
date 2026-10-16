"""Ingestion orchestration.

Runs extraction -> resolution -> graph write, and is the layer that enforces the two rules that
protect a build from wasting an afternoon:

- **Cost ceiling.** `MAX_INGESTION_USD` is checked before extraction starts, from a token estimate
  over the actual corpus. Extraction is by far the dominant cost and caching is by document hash,
  so a re-run costs nothing.
- **Idempotency.** Everything downstream of chunking is content-addressed, so re-running the whole
  ingestion produces the same graph rather than a second copy of it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path

from kg_rag.chunking import Chunk, chunk_document
from kg_rag.corpus import SECTIONS_OF_INTEREST
from kg_rag.extract import ExtractionResult, Extractor, Gazetteer
from kg_rag.resolve import EntityResolver
from kg_rag.store import GraphStore

# Rough token estimate for extraction cost planning. Deliberately an over-estimate: a budget guard
# that under-predicts protects nothing.
CHARS_PER_TOKEN = 3.5
PROMPT_OVERHEAD_TOKENS = 700
OUTPUT_TOKENS_PER_CHUNK = 450

# USD per 1M tokens. Overwritten from live pricing before the README's cost table is filled in;
# these are placeholders that make the guard behave, not a pricing claim.
USD_PER_MTOK_INPUT = 0.80
USD_PER_MTOK_OUTPUT = 4.00


@dataclass
class FilingRecord:
    """One filing ready for ingestion."""

    ticker: str
    company: str | None
    cik: str | None
    form: str
    fiscal_year: int
    filed_at: str | None
    accession: str
    sections: dict[str, str]

    @property
    def document_id(self) -> str:
        return self.accession

    def content_hash(self) -> str:
        """Hash of the content that matters. Cache key for extraction results."""
        payload = json.dumps(self.sections, sort_keys=True)
        return hashlib.sha256(
            f"{self.form}|{self.fiscal_year}|{self.accession}|{payload}".encode()
        ).hexdigest()


@dataclass
class IngestionReport:
    documents: int = 0
    chunks: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_usd: float = 0.0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    rejected: int = 0
    nodes_written: int = 0
    edges_written: int = 0
    merges: int = 0
    extractor: str = "unknown"
    per_document: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "estimated_usd": round(self.estimated_usd, 4),
            "entities_extracted": self.entities_extracted,
            "relationships_extracted": self.relationships_extracted,
            "rejected": self.rejected,
            "nodes_written": self.nodes_written,
            "edges_written": self.edges_written,
            "merges": self.merges,
            "extractor": self.extractor,
        }


class BudgetExceeded(RuntimeError):
    """Refused to start: the corpus would cost more than MAX_INGESTION_USD."""


def estimate_tokens(chunks: list[Chunk]) -> tuple[int, int]:
    """(input_tokens, output_tokens) for extracting every chunk."""
    total_in = 0
    for chunk in chunks:
        total_in += int(len(chunk.text) / CHARS_PER_TOKEN) + PROMPT_OVERHEAD_TOKENS
    total_out = len(chunks) * OUTPUT_TOKENS_PER_CHUNK
    return total_in, total_out


def estimate_usd(tokens_in: int, tokens_out: int) -> float:
    return (
        tokens_in / 1_000_000 * USD_PER_MTOK_INPUT
        + tokens_out / 1_000_000 * USD_PER_MTOK_OUTPUT
    )


def plan_budget(records: list[FilingRecord], ceiling: float | None) -> IngestionReport:
    """Chunk every filing and cost the extraction, without calling a model.

    Run this before a real ingestion. If the estimate exceeds the ceiling, raise rather than
    discovering it halfway through.
    """
    report = IngestionReport()
    for record in records:
        chunks = chunk_filing(record)
        report.chunks += len(chunks)
        tokens_in, tokens_out = estimate_tokens(chunks)
        report.tokens_in += tokens_in
        report.tokens_out += tokens_out
    report.documents = len(records)
    report.estimated_usd = estimate_usd(report.tokens_in, report.tokens_out)

    if ceiling is not None and report.estimated_usd > ceiling:
        raise BudgetExceeded(
            f"estimated extraction cost ${report.estimated_usd:.2f} exceeds "
            f"MAX_INGESTION_USD=${ceiling:.2f} for {report.documents} documents / "
            f"{report.chunks} chunks. Raise the ceiling, or shrink the corpus."
        )
    return report


def chunk_filing(record: FilingRecord) -> list[Chunk]:
    """Chunk a filing, keeping only the sections the corpus decision nominates."""
    sections = record.sections
    wanted = {s for s in SECTIONS_OF_INTEREST}
    filtered = {k: v for k, v in sections.items() if k in wanted} or sections
    year = record.fiscal_year
    if not year and record.filed_at:
        year = int(str(record.filed_at)[:4])
    return chunk_document(
        "",
        form=record.form,
        fiscal_year=year or 0,
        accession=record.accession,
        document_id=record.document_id,
        sections=filtered,
        filed_at=record.filed_at,
    )


class ExtractionCache:
    """Content-addressed cache of extraction results.

    Re-running ingestion must not re-pay for extraction. Keyed on the filing's content hash, so a
    changed document re-extracts and an unchanged one does not.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # A corrupt cache is not a reason to fail an ingestion; it costs money to redo.
                self._data = {}

    def get(self, key: str) -> ExtractionResult | None:
        blob = self._data.get(key)
        if not blob:
            return None
        from kg_rag.config import EntityType, RelationshipType
        from kg_rag.extract import (
            ExtractedEntity,
            ExtractedRelationship,
        )

        return ExtractionResult(
            entities=[
                ExtractedEntity(
                    local_id=e["local_id"], type=EntityType(e["type"]), name=e["name"],
                    aliases=e.get("aliases", []), cik=e.get("cik"),
                    confidence=e["confidence"], source_chunk_id=e.get("source_chunk_id"),
                )
                for e in blob.get("entities", [])
            ],
            relationships=[
                ExtractedRelationship(
                    from_local_id=r["from_local_id"], to_local_id=r["to_local_id"],
                    type=RelationshipType(r["type"]), attributes=r.get("attributes", {}),
                    confidence=r["confidence"], source_chunk_id=r.get("source_chunk_id"),
                )
                for r in blob.get("relationships", [])
            ],
            rejected=blob.get("rejected", []),
            extractor=blob.get("extractor", "cache"),
        )

    def put(self, key: str, result: ExtractionResult) -> None:
        self._data[key] = {
            "entities": [
                {
                    "local_id": e.local_id, "type": e.type.value, "name": e.name,
                    "aliases": e.aliases, "cik": e.cik, "confidence": e.confidence,
                    "source_chunk_id": e.source_chunk_id,
                }
                for e in result.entities
            ],
            "relationships": [
                {
                    "from_local_id": r.from_local_id, "to_local_id": r.to_local_id,
                    "type": r.type.value, "attributes": r.attributes,
                    "confidence": r.confidence, "source_chunk_id": r.source_chunk_id,
                }
                for r in result.relationships
            ],
            "rejected": result.rejected,
            "extractor": result.extractor,
        }

    def flush(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=1), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class Ingester:
    """Runs a corpus through extraction, resolution, and the graph."""

    extractor: Extractor
    store: GraphStore
    gazetteer: Gazetteer
    cache: ExtractionCache | None = None
    report: IngestionReport = field(default_factory=IngestionReport)

    def run(self, records: list[FilingRecord]) -> IngestionReport:
        resolver = EntityResolver()
        report = self.report
        report.extractor = getattr(self.extractor, "name", "unknown")

        for record in records:
            chunks = chunk_filing(record)
            report.documents += 1
            report.chunks += len(chunks)

            results: list[ExtractionResult] = []
            for chunk in chunks:
                cached = self.cache.get(record.content_hash() + chunk.chunk_id) \
                    if self.cache else None
                if cached is not None:
                    result = cached
                else:
                    result = self.extractor.extract(
                        chunk.text,
                        chunk.chunk_id,
                        subject=record.company or record.ticker,
                        filed_at=record.filed_at,
                    )
                    if self.cache:
                        self.cache.put(record.content_hash() + chunk.chunk_id, result)

                # Each chunk is its own ExtractionResult so local_ids cannot collide, then merged.
                results.append(result)

            merged = ExtractionResult(extractor=report.extractor)
            for result in results:
                merged.merge(result)

            report.entities_extracted += len(merged.entities)
            report.relationships_extracted += len(merged.relationships)
            report.rejected += len(merged.rejected)

            entities, edges = resolver.resolve_all([merged], filer_cik=record.cik)
            document_of_chunk = {c.chunk_id: c.document_id for c in chunks}

            report.nodes_written += self.store.upsert_entities(
                entities, model_version=report.extractor
            )
            report.edges_written += self.store.upsert_relationships(
                edges, document_id_of_chunk=document_of_chunk, model_version=report.extractor
            )
            report.per_document.append(
                {
                    "accession": record.accession,
                    "ticker": record.ticker,
                    "chunks": len(chunks),
                    "entities": len(merged.entities),
                    "relationships": len(merged.relationships),
                    "rejected": len(merged.rejected),
                }
            )

        report.merges = resolver.index.merge_count()
        if self.cache:
            self.cache.flush()
        return report


# --------------------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------------------

def load_sample_corpus(path: Path | None = None) -> list[FilingRecord]:
    """Load the synthetic offline corpus from `data/sample_corpus.json`."""
    path = path or Path(__file__).resolve().parent.parent / "data" / "sample_corpus.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[FilingRecord] = []
    for entry in payload.get("filings", []):
        records.append(
            FilingRecord(
                ticker=entry["ticker"],
                company=entry.get("company"),
                cik=entry.get("cik"),  # intentionally absent in the sample corpus
                form=entry.get("form", "10-K"),
                fiscal_year=int(entry.get("fiscal_year", 0)),
                filed_at=entry.get("filed_at"),
                accession=entry["accession"],
                sections=entry.get("sections", {}),
            )
        )
    return records


def load_corpus_from_edgar(
    sec_client, tickers: tuple[str, ...], forms: tuple[str, ...] = ("10-K",), limit: int = 1
) -> list[FilingRecord]:
    """Fetch filing text from EDGAR. Network path; requires SEC_USER_AGENT.

    Section extraction from filing HTML is deliberately minimal here. Item boundaries are marked in
    the document with anchors; a fuller parser is Phase 1 work and the honest state of this
    function is "fetches filings, keeps whole-document text", which is enough to start extraction
    and not enough to be the final shape.
    """
    records: list[FilingRecord] = []
    for registrant in sec_client.resolve(tickers):
        for filing in sec_client.filings(registrant, forms=forms, limit=limit):
            html = sec_client.document(filing)
            text = _html_to_text(html)
            records.append(
                FilingRecord(
                    ticker=registrant.ticker,
                    company=registrant.name,
                    cik=registrant.cik,
                    form=filing.form,
                    fiscal_year=int(filing.filed_at[:4]),
                    filed_at=filing.filed_at,
                    accession=filing.accession,
                    sections={_guess_section(text): text},
                )
            )
    return records


def _html_to_text(html: str) -> str:
    stripped = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    stripped = re.sub(r"(?i)<br\s*/?>", "\n", stripped)
    stripped = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n\n", stripped)
    stripped = re.sub(r"(?s)<[^>]+>", " ", stripped)
    return unescape(re.sub(r"[ \t]+", " ", stripped))


def _guess_section(text: str) -> str:
    """Label whole-document text so it still flows through the section-aware chunker."""
    head = text[:2000].lower()
    if "item 1a" in head or "risk factors" in head:
        return "Item 1A. Risk Factors"
    if "item 7" in head or "management's discussion" in head:
        return "Item 7. Management's Discussion and Analysis"
    return "Item 1. Business"
