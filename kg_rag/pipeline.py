"""End-to-end pipeline wiring.

One object holds the graph store, the vector index, the router and the generator, so the API, the
benchmark and the CLI all exercise the same code path. A benchmark that runs different code from
the server measures nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from kg_rag import settings
from kg_rag.answer import Generator, build_generator, generate_grounded
from kg_rag.cypher_templates import DEFAULT_LIMIT, UnknownQueryType, bind, get_template
from kg_rag.extract import Gazetteer, build_extractor, build_gazetteer_from_registrants
from kg_rag.index import HashingEmbedder, VectorIndex
from kg_rag.ingest import (
    ExtractionCache,
    FilingRecord,
    Ingester,
    load_corpus_from_edgar,
    load_sample_corpus,
)
from kg_rag.merge import assemble, validate_citations
from kg_rag.router import (
    GraphRetriever,
    KeywordRouter,
    RouteDecision,
    build_corpus_vocabulary,
)
from kg_rag.store import GraphStore

CACHE_FILENAME = "extraction_cache.json"


@dataclass
class PipelineStats:
    questions: int = 0
    routed_graph: int = 0
    routed_vector: int = 0
    routed_definition: int = 0
    routed_out_of_scope: int = 0
    fallbacks: int = 0
    refusals: int = 0
    citation_failures: int = 0
    total_latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "questions": self.questions,
            "routed_graph": self.routed_graph,
            "routed_vector": self.routed_vector,
            "routed_definition": self.routed_definition,
            "routed_out_of_scope": self.routed_out_of_scope,
            "fallbacks": self.fallbacks,
            "refusals": self.refusals,
            "citation_failures": self.citation_failures,
            "mean_latency_ms": round(
                self.total_latency_ms / self.questions, 2
            ) if self.questions else 0.0,
        }


@dataclass
class AskResult:
    question: str
    answer: str
    route: str
    query_type: str | None
    citations: list[str]
    refused: bool
    reason: str
    latency_ms: float
    graph_records: list[dict] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)
    cypher: str | None = None
    params: dict = field(default_factory=dict)
    citation_report: dict = field(default_factory=dict)
    generator: str = "unknown"
    attempts: int = 1

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "query_type": self.query_type,
            "citations": self.citations,
            "refused": self.refused,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
            "graph_records": self.graph_records,
            "passages": self.passages,
            "cypher": self.cypher,
            "params": self.params,
            "citation_report": self.citation_report,
            "generator": self.generator,
            "attempts": self.attempts,
        }


class Pipeline:
    """Graph + vector retrieval, routing, grounded generation, citation enforcement."""

    def __init__(
        self,
        store: GraphStore,
        index: VectorIndex,
        router: KeywordRouter,
        generator: Generator,
        known_names: list[str],
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        self.store = store
        self.index = index
        self.router = router
        self.generator = generator
        self.known_names = known_names
        self.graph = GraphRetriever(store, known_names, limit=limit)
        self.stats = PipelineStats()
        # Routing decisions are logged because Phase 5 cannot distinguish a bad route from a bad
        # traversal without them, and they cannot be reconstructed after the fact.
        self.decision_log: list[dict] = []

    # -- ingest -------------------------------------------------------------------------------

    def ingest(self, records: list[FilingRecord], cache: bool = True) -> dict:
        gazetteer = build_gazetteer_from_registrants(_registrants_from_records(records))
        extractor = build_extractor(gazetteer)
        extraction_cache = (
            ExtractionCache(Path(settings.corpus_root()) / CACHE_FILENAME) if cache else None
        )

        ingester = Ingester(
            extractor=extractor, store=self.store, gazetteer=gazetteer, cache=extraction_cache
        )
        report = ingester.run(records)

        # Write passages to the vector index, linking each chunk to the entities it mentions.
        # Entity identity is already settled by the ingester (Entity.surface in the graph), so the
        # chunk -> entity back-link is read back from `e.mentions` rather than re-resolved here.
        # One query for every chunk: querying per chunk is an N+1 against the graph.
        from kg_rag.ingest import chunk_filing

        chunk_rows = []
        for record in records:
            chunk_rows.extend(chunk_filing(record))

        entity_of_chunk: dict[str, list[str]] = {}
        if chunk_rows:
            mentions = self.store.run_query(
                "MATCH (e:Entity) "
                "UNWIND coalesce(e.mentions, []) AS chunk_id "
                "WITH chunk_id, collect(DISTINCT e.entity_id) AS entity_ids "
                "RETURN chunk_id, entity_ids",
                {},
            )
            for row in mentions:
                entity_of_chunk[row["chunk_id"]] = row["entity_ids"]

        indexed = self.index.upsert_chunks(chunk_rows, entity_ids_of_chunk=entity_of_chunk)

        self.known_names = self.refresh_known_names(gazetteer)
        self.router.known_names = self.known_names
        self.graph.known_names = self.known_names
        # The corpus vocabulary is the evidence for out-of-scope refusals, so it is rebuilt from
        # the chunks that were just indexed rather than left unset.
        self.router.corpus_vocabulary = build_corpus_vocabulary(
            chunk.text for chunk in chunk_rows
        )

        report_dict = report.as_dict()
        report_dict["chunks_indexed"] = indexed
        report_dict["graph"] = self.store.counts()
        report_dict["relationships_by_type"] = self.store.relationship_type_counts()
        return report_dict

    def refresh_known_names(self, gazetteer: Gazetteer | None = None) -> list[str]:
        """Entity names the router may anchor on, read back from the graph."""
        records = self.store.run_query(
            "MATCH (e:Entity) WHERE e.entity_type = 'Company' RETURN DISTINCT e.name AS name"
        )
        names = sorted({r["name"] for r in records if r.get("name")})
        if gazetteer:
            for canonical in gazetteer.known_surfaces():
                names.append(canonical)
        return sorted(set(names))

    # -- ask ----------------------------------------------------------------------------------

    def ask(self, question: str, as_of: str | None = None, k: int = 5) -> AskResult:
        started = time.perf_counter()
        self.stats.questions += 1

        decision = self.router.route(question)
        graph_result: dict | None = None
        passages = []
        route_label = decision.route

        wants_graph = decision.route in ("graph", "both") or decision.runs_both
        wants_vector = decision.route in ("vector", "definition", "both") or decision.runs_both

        if decision.fallback and route_label not in ("both",):
            route_label = "both"

        if wants_graph and decision.query_type:
            try:
                graph_result = self.graph.retrieve(
                    decision.query_type, decision.params, as_of=as_of
                )
            except (UnknownQueryType, ValueError) as exc:
                # No usable template. Fall back to vector rather than returning nothing, and
                # record why — a misconfigured template must not read as "no relationship exists".
                graph_result = None
                self.decision_log.append(
                    {"question": question, "error": str(exc), "query_type": decision.query_type}
                )
                wants_vector = True

        if wants_vector:
            hits = self.index.search(question, k=k)
            passages = [hit.as_passage() for hit in hits]

        context = assemble(
            question=question,
            route=route_label,
            graph_result=graph_result,
            passages=passages,
            query_type=decision.query_type,
        )

        if decision.route == "out_of_scope":
            generated = generate_grounded(
                context, _RefusingGenerator("router classified the question as out of scope")
            )
        else:
            generated = generate_grounded(context, self.generator)

        latency_ms = (time.perf_counter() - started) * 1000
        self._record_stats(decision, route_label, generated, latency_ms)
        self.decision_log.append(
            {
                "question": question,
                "route": route_label,
                "query_type": decision.query_type,
                "confidence": decision.confidence,
                "decided_by": decision.decided_by,
                "fallback": decision.fallback,
                "params": decision.params,
                "graph_hit": bool(graph_result and graph_result.get("hit")),
                "passage_count": len(passages),
                "refused": generated.refused,
                "latency_ms": round(latency_ms, 2),
            }
        )

        return AskResult(
            question=question,
            answer=generated.answer,
            route=route_label,
            query_type=decision.query_type,
            citations=generated.citations,
            refused=generated.refused,
            reason=generated.reason,
            latency_ms=latency_ms,
            graph_records=(graph_result or {}).get("records", []),
            passages=[
                {"chunk_id": p.chunk_id, "section_path": p.section_path, "score": p.score}
                for p in passages
            ],
            cypher=(graph_result or {}).get("cypher"),
            params=(graph_result or {}).get("params", {}),
            citation_report=generated.citation_report,
            generator=generated.generator,
            attempts=generated.attempts,
        )

    def _record_stats(
        self, decision: RouteDecision, route_label: str, generated, latency_ms: float
    ) -> None:
        if decision.route == "graph":
            self.stats.routed_graph += 1
        elif decision.route == "vector":
            self.stats.routed_vector += 1
        elif decision.route == "definition":
            self.stats.routed_definition += 1
        elif decision.route == "out_of_scope":
            self.stats.routed_out_of_scope += 1
        if decision.fallback:
            self.stats.fallbacks += 1
        if generated.refused:
            self.stats.refusals += 1
        if generated.citation_report and not generated.citation_report.get("ok", True):
            self.stats.citation_failures += 1
        self.stats.total_latency_ms += latency_ms


class _RefusingGenerator:
    """Generator that declines. Used when the router says the question is out of scope."""

    name = "stub:refusing"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def generate(self, context, note: str = "") -> str:
        return ""


def _registrants_from_records(records: list[FilingRecord]):
    from kg_rag.corpus import Registrant

    seen: set[str] = set()
    registrants = []
    for record in records:
        if record.ticker in seen:
            continue
        seen.add(record.ticker)
        registrants.append(
            Registrant(
                ticker=record.ticker,
                name=record.company or record.ticker,
                cik=record.cik,
            )
        )
    return registrants


def build_pipeline(
    gazetteer: Gazetteer | None = None, limit: int = DEFAULT_LIMIT
) -> Pipeline:
    """Wire the whole system from environment settings."""
    store = GraphStore.connect(settings.neo4j_uri(), settings.neo4j_auth())
    index = VectorIndex(
        dsn=settings.postgres_dsn(),
        embedder=HashingEmbedder(dimension=settings.embedding_dimension()),
    )
    index.ensure_schema()

    names = gazetteer.known_surfaces() if gazetteer else []
    router = KeywordRouter(known_names=names)
    generator = build_generator()

    pipeline = Pipeline(
        store=store, index=index, router=router, generator=generator, known_names=names,
        limit=limit,
    )
    pipeline.known_names = pipeline.refresh_known_names()
    pipeline.router.known_names = pipeline.known_names
    pipeline.graph.known_names = pipeline.known_names
    return pipeline


def default_corpus() -> list[FilingRecord]:
    """The synthetic offline corpus. Real EDGAR requires SEC_USER_AGENT and a network round trip."""
    return load_sample_corpus()


def build_sample_gazetteer() -> Gazetteer:
    return build_gazetteer_from_registrants(_registrants_from_records(load_sample_corpus()))


__all__ = [
    "AskResult",
    "Pipeline",
    "PipelineStats",
    "bind",
    "build_pipeline",
    "build_sample_gazetteer",
    "default_corpus",
    "get_template",
    "load_corpus_from_edgar",
    "validate_citations",
]
