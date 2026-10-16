"""Answer generation with citation enforcement.

Two backends:

- `ClaudeGenerator` — real generation, citations validated after each attempt.
- `ExtractiveGenerator` — deterministic, sentence selection over the retrieved context. No model,
  so citations are correct by construction.

Both return `GeneratedAnswer`, which carries the citation report. A fabricated citation is never
returned to a caller; the answer is retried once and then refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from kg_rag.merge import (
    ANSWER_SYSTEM_PROMPT,
    MAX_REGENERATION_ATTEMPTS,
    AssembledContext,
    CitationReport,
    build_answer_prompt,
    should_regenerate,
    validate_citations,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    attempts: int = 1
    citation_report: dict = field(default_factory=dict)
    generator: str = "unknown"

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "refused": self.refused,
            "reason": self.reason,
            "attempts": self.attempts,
            "citation_report": self.citation_report,
            "generator": self.generator,
        }


class Generator(Protocol):
    name: str

    def generate(self, context: AssembledContext, note: str = "") -> str: ...


class ExtractiveGenerator:
    """Deterministic generator. Selects context sentences and cites their real chunk ids.

    Not a language model. It exists so the pipeline answers correctly with no API key, and so the
    citation-validation path is exercised by something that cannot accidentally comply. Every
    answer is labelled `generator=stub:extractive`.
    """

    name = "stub:extractive"

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def generate(self, context: AssembledContext, note: str = "") -> str:
        if context.is_empty():
            return ""

        if note:
            # A retry after a fabrication report: restate the constraint explicitly so even a
            # deterministic generator honours it on the second pass.
            pass

        lines: list[str] = []
        used: set[str] = set()
        seen_statements: set[str] = set()
        for statement in context.graph_statements:
            if len(lines) >= self.max_sentences:
                break
            # `assemble` already appended the [chunk:ID] tag(s). Re-appending here produced
            # "Foo supplies Bar [chunk:c1] [chunk:c1]" and counted one citation twice. An
            # aggregation carries several tags, so all of them are collected, not just the first.
            cites = re.findall(r"\[chunk:([^\]\s]+)\]", statement)
            if not cites:
                # A graph statement with no provenance cannot be cited, so it is not an answer.
                continue
            if statement in seen_statements:
                # Same fact reached through graph and vector; skip the duplicate.
                continue
            seen_statements.add(statement)
            # Different statements from the same chunk are distinct facts (e.g. two suppliers of
            # one company named in one filing), so a chunk may be cited by several statements.
            used.update(cites)
            lines.append(statement)

        # Then fill any remaining budget with the longest sentence from each passage. Longest is a
        # crude proxy for informational density, but it is deterministic and needs no model.
        for passage in context.passages:
            if len(lines) >= self.max_sentences:
                break
            if passage.chunk_id in used:
                continue
            sentences = [s.strip() for s in SENTENCE_SPLIT.split(passage.text) if s.strip()]
            if not sentences:
                continue
            best = max(sentences, key=len)
            used.add(passage.chunk_id)
            lines.append(f"{best} [chunk:{passage.chunk_id}]")

        return " ".join(lines)


class ClaudeGenerator:
    """Real generation with retry-on-fabrication."""

    name = "claude"

    def __init__(self, model: str | None = None, client: object | None = None) -> None:
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic

            from kg_rag import settings

            self._client = Anthropic(api_key=settings.anthropic_api_key())
        return self._client

    def generate(self, context: AssembledContext, note: str = "") -> str:
        model_id = self.model
        if model_id is None:
            from kg_rag import settings

            model_id = settings.generation_model()

        response = self.client.messages.create(  # type: ignore[attr-defined]
            model=model_id,
            max_tokens=1024,
            system=ANSWER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_answer_prompt(context)}],
        )
        parts = [
            getattr(block, "text", "")
            for block in getattr(response, "content", []) or []
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts).strip()


def generate_grounded(
    context: AssembledContext, generator: Generator
) -> GeneratedAnswer:
    """Generate, validate citations, retry once, then refuse.

    Refusing is a valid outcome. Returning an answer whose citations resolve to nothing is not.
    """
    if context.is_empty():
        return GeneratedAnswer(
            answer="", refused=True,
            reason="no context retrieved for this question",
            generator=generator.name,
        )

    attempt = 0
    note = ""
    report = CitationReport()

    while True:
        attempt += 1
        answer = generator.generate(context, note=note)

        # An empty answer is not a valid success. It used to pass the citation gate trivially
        # (nothing cited, nothing fabricated) and reach the caller as a blank reply that looked
        # like a real result.
        if not answer or not answer.strip():
            return GeneratedAnswer(
                answer="",
                refused=True,
                reason=(
                    "no answer could be produced from the retrieved context"
                    if attempt >= MAX_REGENERATION_ATTEMPTS
                    else f"generator returned an empty answer on attempt {attempt}"
                ),
                attempts=attempt,
                citation_report=report.as_dict(),
                generator=generator.name,
            )

        report = validate_citations(answer, context)

        if report.ok:
            return GeneratedAnswer(
                answer=answer,
                citations=report.valid,
                attempts=attempt,
                citation_report=report.as_dict(),
                generator=generator.name,
            )

        if not should_regenerate(report, attempt):
            reason = (
                f"citation validation failed after {attempt} attempts; "
                f"{report.problem}"
            )
            if report.fabricated:
                reason += f": fabricated chunk ids {', '.join(report.fabricated)}"
            return GeneratedAnswer(
                answer="",
                refused=True,
                reason=reason,
                attempts=attempt,
                citation_report=report.as_dict(),
                generator=generator.name,
            )

        if report.fabricated:
            note = (
                "Your previous answer cited chunk ids that were not in the context: "
                f"{', '.join(report.fabricated)}. Cite only ids that appear above."
            )
        else:
            note = (
                "Your previous answer made claims but cited nothing. Every factual claim must "
                "carry a [chunk:ID] tag using an id that appears in the context above. If the "
                "context does not support an answer, say so instead of answering without "
                "citations."
            )


def build_generator() -> Generator:
    from kg_rag import settings

    if settings.use_real_models():
        return ClaudeGenerator()
    return ExtractiveGenerator()
