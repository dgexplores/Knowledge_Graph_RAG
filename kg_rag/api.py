"""FastAPI service exposing the Knowledge Graph RAG pipeline.

Endpoints:

- `POST /ask` — ask one question, returns the grounded answer with citations,
  route, query_type, latency, and citation report. Refusals are valid outcomes
  (200 with `refused: true`), not errors.
- `GET /health` — connectivity check for Neo4j + pgvector.

The pipeline is built once at startup and closed on shutdown so every request
exercises the exact same store + index the benchmark uses.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kg_rag.pipeline import AskResult, build_pipeline, build_sample_gazetteer, load_sample_corpus

_pipeline: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline once, keep it for the life of the process."""
    global _pipeline
    _pipeline = build_pipeline(build_sample_gazetteer())
    yield
    if _pipeline is not None:
        _pipeline.store.close()
        _pipeline.index.close()


app = FastAPI(
    title="Knowledge Graph RAG",
    lifespan=lifespan,
    description="Hybrid knowledge-graph + vector retrieval over SEC filings.",
)


class AskRequest(BaseModel):
    question: str
    as_of: str | None = None
    k: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    route: str
    query_type: str | None
    citations: list[str]
    refused: bool
    reason: str
    latency_ms: float
    graph_records: list[dict]
    passages: list[dict]
    cypher: str | None
    params: dict
    citation_report: dict
    generator: str
    attempts: int


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Ask one question.

    Refusals (out-of-scope, no citation support, empty generator) return
    200 with `refused: true` — they are valid outcomes, not errors. A 500
    means the retrieval stack itself failed, which should not happen.
    """
    if _pipeline is None:  # pragma: no cover - lifespan always builds it
        raise HTTPException(status_code=503, detail="pipeline not initialized")
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    result: AskResult = _pipeline.ask(
        request.question, as_of=request.as_of, k=request.k
    )
    return AskResponse(**result.as_dict())


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    nodes_written: int
    edges_written: int
    graph: dict
    relationships_by_type: dict


@app.post("/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    """Load the offline demo corpus and ingest it into Neo4j + pgvector.

    Idempotent: re-running produces the same graph rather than a second copy
    (content-addressed extraction cache).
    """
    if _pipeline is None:  # pragma: no cover - lifespan always builds it
        raise HTTPException(status_code=503, detail="pipeline not initialized")
    report = _pipeline.ingest(load_sample_corpus(), cache=False)
    return IngestResponse(
        documents=report["documents"],
        chunks=report["chunks"],
        nodes_written=report["nodes_written"],
        edges_written=report["edges_written"],
        graph=report["graph"],
        relationships_by_type=report["relationships_by_type"],
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Connectivity check for Neo4j + pgvector."""
    if _pipeline is None:  # pragma: no cover - lifespan always builds it
        return {"status": "initializing"}
    try:
        _pipeline.store.run_query("RETURN 1 AS ok")
        _pipeline.index.search("connectivity check", k=1)
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001 - report the failure, don't guess
        return {"status": "degraded", "detail": str(exc)}


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
