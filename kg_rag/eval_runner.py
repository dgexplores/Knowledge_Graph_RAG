"""Benchmark harness.

Runs the same question set through two systems:

  vector_only   pgvector search + extractive generation, no graph
  hybrid        the routed pipeline

and reports accuracy **broken out by question class**, because a single aggregate number hides the
entire claim. The expected shape is parity on 1-hop and a widening gap as hops increase; if the
hybrid system does not show that, the report says so rather than dressing up the aggregate.

Scoring is string-based and deliberately strict. A judge model would score more favourably and
would also be uncalibrated — see the sibling project for the kappa work. Every report carries the
scorer name so the numbers are never ambiguous.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_CLASSES = ("1_hop", "2_hop", "3_hop", "aggregation", "out_of_scope")


@dataclass
class ScoredResult:
    question_id: str
    question: str
    question_class: str
    system: str
    correct: bool
    answered: bool
    expected_terms: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    latency_ms: float = 0.0
    route: str = ""
    citation_ok: bool = True
    citation_problem: str = ""


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def score_answer(expected_terms: list[str], answer: str, refused: bool) -> tuple[bool, list[str]]:
    """Strict term matching.

    An answer must contain every expected term. Loose scoring would reward an answer that names
    every company in the corpus, which is the failure mode this benchmark exists to catch.

    A question with no expected terms scores incorrect rather than correct. With an empty list
    `len(matched) == len(expected)` is vacuously true, so every answer would pass — a question
    set that forgot its `expected_terms` would report 100% accuracy instead of an error.
    """
    if refused:
        return False, []
    if not expected_terms:
        return False, []
    haystack = _normalize(answer)
    matched = [term for term in expected_terms if _normalize(term) in haystack]
    return len(matched) == len(expected_terms), matched


def should_refuse(question_class: str) -> bool:
    return question_class == "out_of_scope"


def load_question_set(path: Path | None = None) -> list[dict]:
    path = path or Path(__file__).resolve().parent.parent / "eval" / "question_set.jsonl"
    if not path.exists():
        return []
    questions: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        questions.append(json.loads(line))
    return questions


def run_vector_only(pipeline, question: dict, k: int = 5) -> ScoredResult:
    """Baseline: vector retrieval only, no routing and no graph.

    This is the system the hybrid system has to beat, so it is deliberately given the same
    generator and the same passage budget — the only difference is the retrieval path.
    """
    from kg_rag.answer import ExtractiveGenerator, generate_grounded
    from kg_rag.merge import assemble

    started = time.perf_counter()
    hits = pipeline.index.search(question["question"], k=k)
    context = assemble(
        question=question["question"],
        route="vector",
        graph_result=None,
        passages=[hit.as_passage() for hit in hits],
        query_type=None,
    )
    generated = generate_grounded(context, ExtractiveGenerator())
    latency_ms = (time.perf_counter() - started) * 1000

    expected = list(question.get("expected_terms", []))
    if should_refuse(question["class"]):
        correct = generated.refused
        matched: list[str] = []
    else:
        correct, matched = score_answer(expected, generated.answer, generated.refused)

    return ScoredResult(
        question_id=question["id"], question=question["question"],
        question_class=question["class"], system="vector_only",
        correct=correct, answered=not generated.refused, expected_terms=expected,
        matched_terms=matched, citations=generated.citations, refused=generated.refused,
        latency_ms=latency_ms, route="vector",
        citation_ok=bool(generated.citation_report.get("ok", True)),
        citation_problem=str(generated.citation_report.get("problem", "") or ""),
    )


def run_hybrid(pipeline, question: dict, k: int = 5) -> ScoredResult:
    started = time.perf_counter()
    result = pipeline.ask(question["question"], k=k)
    latency_ms = (time.perf_counter() - started) * 1000

    expected = list(question.get("expected_terms", []))
    if should_refuse(question["class"]):
        correct = result.refused
        matched: list[str] = []
    else:
        correct, matched = score_answer(expected, result.answer, result.refused)

    return ScoredResult(
        question_id=question["id"], question=question["question"],
        question_class=question["class"], system="hybrid",
        correct=correct, answered=not result.refused, expected_terms=expected,
        matched_terms=matched, citations=result.citations, refused=result.refused,
        latency_ms=latency_ms, route=result.route,
        citation_ok=bool(result.citation_report.get("ok", True)),
        citation_problem=str(result.citation_report.get("problem", "") or ""),
    )


def run_benchmark(
    pipeline, split: str = "all", limit: int = 0, question_set: list[dict] | None = None
) -> dict:
    questions = question_set if question_set is not None else load_question_set()
    if split != "all":
        questions = [q for q in questions if q.get("split") == split]
    if limit:
        questions = questions[:limit]

    if not questions:
        return {
            "summary": {"error": "no questions; run `python -m kg_rag.cli ingest --sample` first"},
            "results": [],
        }

    results: list[ScoredResult] = []
    started = time.perf_counter()
    for question in questions:
        results.append(run_vector_only(pipeline, question))
        results.append(run_hybrid(pipeline, question))
    elapsed = time.perf_counter() - started

    return {
        "summary": summarize(results, elapsed),
        "by_class": by_class(results),
        "routing": pipeline.stats.as_dict(),
        "results": [
            {
                "id": r.question_id, "class": r.question_class, "system": r.system,
                "correct": r.correct, "answered": r.answered, "refused": r.refused,
                "route": r.route, "citations": r.citations, "latency_ms": round(r.latency_ms, 2),
                "expected_terms": r.expected_terms, "matched_terms": r.matched_terms,
                "citation_ok": r.citation_ok, "citation_problem": r.citation_problem,
                "question": r.question,
            }
            for r in results
        ],
        "scorer": "strict-substring-all-terms",
        "note": (
            "Term matching, not a judge model. Strict on purpose: partial credit would reward an "
            "answer that names every entity in the graph."
        ),
    }


def summarize(results: list[ScoredResult], elapsed: float) -> dict:
    def accuracy(system: str, question_class: str | None = None) -> float:
        subset = [
            r for r in results
            if r.system == system and (question_class is None or r.question_class == question_class)
        ]
        if not subset:
            return 0.0
        return sum(1 for r in subset if r.correct) / len(subset)

    def mean_latency(system: str) -> float:
        subset = [r for r in results if r.system == system]
        if not subset:
            return 0.0
        return round(sum(r.latency_ms for r in subset) / len(subset), 2)

    defects = citation_defects(results)
    return {
        "questions": len({r.question_id for r in results}),
        "total_questions_run": len(results) // 2,
        "wall_clock_s": round(elapsed, 2),
        "vector_only_accuracy": round(accuracy("vector_only"), 4),
        "hybrid_accuracy": round(accuracy("hybrid"), 4),
        "delta": round(accuracy("hybrid") - accuracy("vector_only"), 4),
        "vector_only_mean_latency_ms": mean_latency("vector_only"),
        "hybrid_mean_latency_ms": mean_latency("hybrid"),
        "hybrid_refusals": sum(1 for r in results if r.system == "hybrid" and r.refused),
        "citation_defects": len(defects),
        "citation_defect_detail": defects,
    }


def citation_defects(results: list[ScoredResult]) -> list[str]:
    """Questions whose answer failed citation validation.

    A refusal is not a defect: refusing is a correct outcome. An *answered* question with a
    citation problem is. This must be zero, and reporting zero unconditionally would make the
    number meaningless, so the failures are listed by question id.
    """
    return [
        f"{r.question_id} ({r.system}): {r.citation_problem or 'citation report failed'}"
        for r in results
        if not r.citation_ok and not r.refused
    ]


def by_class(results: list[ScoredResult]) -> dict:
    out: dict[str, dict] = {}
    for question_class in QUESTION_CLASSES:
        vector = [r for r in results if r.system == "vector_only" and r.question_class == question_class]
        hybrid = [r for r in results if r.system == "hybrid" and r.question_class == question_class]
        if not vector and not hybrid:
            continue
        vector_acc = sum(1 for r in vector if r.correct) / len(vector) if vector else 0.0
        hybrid_acc = sum(1 for r in hybrid if r.correct) / len(hybrid) if hybrid else 0.0
        out[question_class] = {
            "n": len(vector),
            "vector_only": round(vector_acc, 4),
            "hybrid": round(hybrid_acc, 4),
            "delta": round(hybrid_acc - vector_acc, 4),
        }
    return out


def render_table(report: dict) -> str:
    """The markdown table that belongs at the top of the README."""
    lines = [
        "| Question class | n | Vector-only | Hybrid | Delta |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for question_class, row in report.get("by_class", {}).items():
        lines.append(
            f"| {question_class} | {row['n']} | {row['vector_only']:.4f} | "
            f"{row['hybrid']:.4f} | {row['delta']:+.4f} |"
        )
    summary = report.get("summary", {})
    lines.append(
        f"| **overall** | {summary.get('questions', 0)} | "
        f"{summary.get('vector_only_accuracy', 0):.4f} | "
        f"{summary.get('hybrid_accuracy', 0):.4f} | "
        f"{summary.get('delta', 0):+.4f} |"
    )
    return "\n".join(lines)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
