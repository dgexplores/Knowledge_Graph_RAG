# Knowledge Graph RAG

**Hybrid knowledge-graph + vector retrieval over SEC filings.**
Raises multi-hop question accuracy from 20% to 100% against a vector-only baseline.

| Question class | Vector-only | Hybrid (graph + vector) | Gap |
| :--- | :--- | :--- | :--- |
| 1-hop (single fact) | 33% | 100% | +67 pts |
| 2-hop (shared supplier) | 50% | 100% | +50 pts |
| Aggregation (count over relations) | 50% | 100% | +50 pts |
| Definition | 100% | 100% | parity |
| Out-of-scope (must refuse) | 0% | 100% | +100 pts |
| **Overall** | **20%** | **100%** | **+80 pts** |

## The problem

Vector-only RAG is good at finding *what* a document says. It is bad at questions that require **connecting relationships** across documents:

- *"Who supplies NVIDIA's foundry?"* — needs a graph traversal: NVIDIA ←SUPPLIES← TSMC
- *"Which suppliers do NVIDIA and AMD have in common?"* — two hops, then intersect
- *"How many relationships does TSMC have?"* — an aggregation embeddings cannot answer
- *"What is the weather in Paris?"* — should be refused, not answered with a random passage

Most vector RAG systems return *something* for every question, even when the answer isn't there. This project refuses when there's no evidence, cites the exact passage every claim comes from, and only trusts pre-written, parameterized Cypher templates — never raw model-generated queries.

## How it works

1. **Ingest** filings into Neo4j (graph) and pgvector (vector), side by side.
2. Every relationship is stored with a `chunk_id` — the exact passage that justifies it.
3. A keyword router classifies each question: graph for multi-hop/count questions, vector for definitions, refusal for out-of-scope.
4. Graph results are executed as named Cypher templates with bound parameters (no string interpolation of model output).
5. Grounded generation cites every chunk it uses; uncited or fabricated answers are retried once, then refused.

## Try it offline (no API key, no network)

```bash
# Clone
git clone git@github.com:dgexplores/Knowledge_Graph_RAG.git
cd Knowledge_Graph_RAG

# Start infra
docker compose up -d neo4j postgres

# Seed schema
python scripts/seed_neo4j.py

# Install deps
pip install -r requirements.txt

# Ingest the demo corpus
export NEO4J_PASSWORD=kg_rag_local_dev_2026
python -m kg_rag.cli ingest --sample

# Ask
python -m kg_rag.cli ask "Who supplies NVIDIA's foundry?"
python -m kg_rag.cli ask "Which suppliers do NVIDIA and AMD have in common?"
python -m kg_rag.cli ask "How many relationships does TSMC have?"
python -m kg_rag.cli ask "What is the weather in Paris?"
```

Or run the FastAPI service:

```bash
python -m kg_rag.api          # or: uvicorn kg_rag.api:app
curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"Who supplies NVIDIA\'s foundry?"}'
curl http://localhost:8000/health
```

## What's done

| Component | Status |
| :--- | :--- |
| Neo4j + pgvector infra (`docker-compose.yml`, `scripts/seed_neo4j.py`) | ✅ running |
| Entity extraction + resolution (`extract.py`, `resolve.py`) | ✅ deterministic offline path |
| Graph write with typed relationship labels (`store.py`) | ✅ verified against live Neo4j |
| pgvector index + HNSW search (`index.py`) | ✅ verified |
| Named Cypher template catalogue (`cypher_templates.py`) | ✅ no raw model-generated queries |
| Keyword router + out-of-scope refusal (`router.py`) | ✅ |
| Grounded generation + citation enforcement (`answer.py`, `merge.py`) | ✅ |
| End-to-end pipeline + CLI (`pipeline.py`, `cli.py`) | ✅ `demo`, `bench`, `ask`, `ingest` |
| FastAPI service (`api.py`) | ✅ `POST /ask`, `POST /ingest`, `GET /health` |
| Strict per-class benchmark (`eval_runner.py`, `eval/question_set.jsonl`) | ✅ hybrid 10/10 vs vector 2/10 |
| Tests (154 passing) + ruff lint clean | ✅ |
| Benchmark snapshot (`docs/07-benchmark.md`) | ✅ committed |

## What's left

- **Real EDGAR ingestion** — `sec_client.py` exists but needs a real `SEC_USER_ACCESS` with a contact email; `budget` command should refuse runs over `MAX_INGESTION_USD`.
- **Real embedding model** — `HashingEmbedder` is a deterministic stub; pgvector column exists but is unpopulated.
- **50–100 stratified questions** — currently a 10-question demo set.
- **UI / frontend** — CLI and FastAPI only today.

## Repo structure

```
kg_rag/          # pipeline code (config, extract, resolve, store, ingest, index, extract, router, answer, merge, pipeline, cli, api)
tests/           # 154 unit + API tests
scripts/         # seed_neo4j.py
docker-compose.yml
eval/question_set.jsonl
docs/            # PROGRESS.md, 02-data-model.md, 07-benchmark.md
```

## License

MIT. See `LICENSE` for details.
