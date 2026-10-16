"""Context assembly and citation validation.

Phase 4. Two jobs:

1. Merge graph-derived facts with retrieved passages into one labelled context block, so every
   claim is attributable at answer time. That attributability is what makes step 2 possible.
2. Validate that every citation in the generated answer resolves to a chunk that was actually
   retrieved. Reject and regenerate once, then refuse rather than return an uncited answer.

This costs nothing and eliminates invented sources. It is the highest value-per-line correctness
control in the project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE_PATTERN = re.compile(r"\[chunk:([^\]\s]+)\]")
MAX_REGENERATION_ATTEMPTS = 2


@dataclass
class RetrievedPassage:
    chunk_id: str
    text: str
    section_path: str = ""
    document_id: str = ""
    score: float = 0.0

    def as_block(self) -> str:
        return f"[chunk:{self.chunk_id}] ({self.section_path or 'unknown section'})\n{self.text}"


@dataclass
class AssembledContext:
    """Labelled context. Graph facts and passages are separated so citations stay attributable."""

    graph_statements: list[str] = field(default_factory=list)
    passages: list[RetrievedPassage] = field(default_factory=list)
    question: str = ""
    route: str = ""
    query_type: str | None = None

    def allowed_chunk_ids(self) -> set[str]:
        """Exactly the chunk ids that were retrieved. A citation outside this set is fabricated."""
        return {p.chunk_id for p in self.passages} | set(self.graph_chunk_ids())

    def graph_chunk_ids(self) -> set[str]:
        found: set[str] = set()
        for statement in self.graph_statements:
            found.update(re.findall(r"\[chunk:([^\]\s]+)\]", statement))
        return found

    def render(self) -> str:
        sections: list[str] = []
        if self.graph_statements:
            labelled = "\n".join(f"- {s}" for s in self.graph_statements)
            sections.append(f"## Graph-derived facts\n{labelled}")
        if self.passages:
            blocks = "\n\n".join(p.as_block() for p in self.passages)
            sections.append(f"## Retrieved passages\n{blocks}")
        return "\n\n".join(sections) if sections else "(no context retrieved)"

    def is_empty(self) -> bool:
        return not self.graph_statements and not self.passages


def assemble(
    question: str,
    route: str,
    graph_result: dict | None = None,
    passages: list[RetrievedPassage] | None = None,
    query_type: str | None = None,
) -> AssembledContext:
    """Build labelled context from both retrieval paths.

    `graph_result` is the dict returned by `GraphRetriever.retrieve`. Graph statements carry the
    `source_chunk_id` of the edge that justified them, appended as a citation marker, so a graph
    fact is as citable as a passage.
    """
    statements: list[str] = []
    if graph_result and graph_result.get("records"):
        for statement, record in zip(
            graph_result.get("statement_rows", []), graph_result["records"]
        ):
            # A traversal cites one chunk; an aggregation is supported by every edge it counted,
            # so it cites all of them. Citing only the first would attribute a computed total to
            # a single passage that does not state it.
            chunk_ids = [
                c for c in (record.get("chunk_ids") or []) if c
            ] or ([record["chunk_id"]] if record.get("chunk_id") else [])
            tags = "".join(f" [chunk:{c}]" for c in chunk_ids)
            statements.append(f"{statement}{tags}")

    # Deduplicate across the two sources: the same fact from graph and vector is one fact.
    seen: set[str] = set()
    deduped: list[str] = []
    for statement in statements:
        key = re.sub(r"\s+", " ", statement).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(statement)

    return AssembledContext(
        graph_statements=deduped,
        passages=list(passages or []),
        question=question,
        route=route,
        query_type=query_type,
    )


@dataclass
class CitationReport:
    cited_chunk_ids: list[str] = field(default_factory=list)
    valid: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)
    uncited_claims: int = 0

    @property
    def ok(self) -> bool:
        # An answer is acceptable only if every citation resolves AND it cites at least one
        # thing. `not self.fabricated` alone would pass an answer with zero citations, which is
        # the same failure as a fabricated one: an unsupported claim with nothing to check.
        return not self.fabricated and not self.uncited_claims

    @property
    def problem(self) -> str:
        if self.fabricated:
            return "fabricated citations"
        if self.uncited_claims:
            return "claims without any citation"
        return ""

    def as_dict(self) -> dict:
        return {
            "cited": len(self.cited_chunk_ids),
            "valid": len(self.valid),
            "fabricated": self.fabricated,
            "uncited_claims": self.uncited_claims,
            "problem": self.problem,
            "ok": self.ok,
        }


def validate_citations(answer: str, context: AssembledContext) -> CitationReport:
    """Every citation must resolve to a chunk that was actually retrieved."""
    allowed = context.allowed_chunk_ids()
    cited = CITE_PATTERN.findall(answer or "")
    report = CitationReport(cited_chunk_ids=cited)

    for chunk_id in cited:
        if chunk_id in allowed:
            report.valid.append(chunk_id)
        else:
            report.fabricated.append(chunk_id)

    # Deduplicate while preserving order, so a repeated citation is not reported as two fabrications.
    report.fabricated = list(dict.fromkeys(report.fabricated))

    # An answer with claims but no citation at all is uncited, not fabricated. Counted separately
    # because the fix is a prompt problem, not a hallucination.
    if answer and answer.strip() and not cited:
        report.uncited_claims = 1
    return report


ANSWER_SYSTEM_PROMPT = """You answer questions about SEC filings using only the supplied context.

Rules:
1. Use only facts present in the context. If the context does not answer the question, say so.
2. Cite every claim with the chunk id it came from, in the form [chunk:ID].
3. A citation must be an id that appears in the context. Never invent one.
4. Prefer graph-derived facts for relationship questions; prefer retrieved passages for definitions.
5. Be concise. State the answer, then the evidence."""


def build_answer_prompt(context: AssembledContext) -> str:
    """Generation prompt. The schema and chunk ids come from the context, so the model cannot be
    asked to cite something outside it."""
    return (
        f"Question: {context.question}\n\n"
        f"Retrieval route: {context.route}"
        + (f" (template: {context.query_type})" if context.query_type else "")
        + "\n\nContext:\n"
        + context.render()
        + "\n\nAnswer, citing every claim."
    )


def should_regenerate(report: CitationReport, attempt: int) -> bool:
    """Whether to retry. Bounded, because an unbounded retry loop is how one query costs $12."""
    return not report.ok and attempt < MAX_REGENERATION_ATTEMPTS


def refusal(reason: str) -> dict:
    """A structured refusal rather than an uncited guess.

    Returning a refusal is a valid outcome. The out-of-scope class in the benchmark depends on the
    system being willing to say it does not know.
    """
    return {"answer": "", "refused": True, "reason": reason, "citations": []}
