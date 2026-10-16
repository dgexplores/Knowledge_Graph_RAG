# Progress

Measured numbers only. Every entry needs a date and the command that produced it. No assertions.

**Last updated:** 2026-10-16
**Current phase:** Phase 5 complete — benchmark measured, demo runnable end-to-end offline.

---

## Phase status

| Phase | Status | Exit criterion met? |
| :--- | :--- | :--- |
| 0 — Decisions | ✅ complete | 4/4 docs written and committed |
| 1 — Extract into Neo4j | ✅ complete | graph builds from the offline sample corpus; idempotent re-ingest |
| 2 — Vector index alongside graph | ✅ complete | pgvector chunks joined to graph entities by `chunk_id` |
| 3 — Router | ✅ complete | keyword router with out-of-scope refusal, template + confidence per decision |
| 4 — Merge + grounded answer | ✅ complete | citation gate rejects uncited/fabricated answers; empty answers refused |
| 5 — Benchmark | ✅ complete | 10-question offline set; hybrid 10/10 vs vector 2/10 |

---

## Built and verified so far

| Component | Status | Verification |
| :--- | :--- | :--- |
| `kg_rag/config.py` — ontology, ids, name normalization | ✅ done | 40 tests |
| `kg_rag/chunking.py` — stable content-addressed chunk ids | ✅ done | 31 tests |
| `kg_rag/cypher_templates.py` — named, parameterized query catalogue | ✅ done | template direction + Cypher scope tests |
| `kg_rag/resolve.py` — entity resolution, CIK scoping, temporal edge ids | ✅ done | scoping + temporal identity tests |
| `kg_rag/store.py` — Neo4j write/query layer | ✅ done | verified against live Neo4j |
| `kg_rag/extract.py` — offline deterministic extractor | ✅ done | pattern + gazetteer + direction tests |
| `kg_rag/ingest.py` — extraction → resolution → graph write | ✅ done | end-to-end ingest + vector back-link |
| `kg_rag/index.py` — pgvector schema + HNSW search | ✅ done | verified against live pgvector |
| `kg_rag/merge.py` — graph/vector context assembly | ✅ done | assembly + citation tests |
| `kg_rag/answer.py` — extractive generator + citation enforcement | ✅ done | uncited/fabricated/empty tests |
| `kg_rag/router.py` — keyword router with out-of-scope gate | ✅ done | anchor + scope + template tests |
| `kg_rag/pipeline.py`, `kg_rag/cli.py` — orchestrator + CLI | ✅ done | `demo` and `bench` commands run end-to-end |
| `kg_rag/eval_runner.py` — strict per-class benchmark | ✅ done | benchmark report + citation-defect metric |
| `docker-compose.yml` — Neo4j 5 + pgvector 0.8.7 | ✅ running | both containers healthy |
| `scripts/seed_neo4j.py` | ✅ verified | typed edge indexes ONLINE |
| `.github/workflows/ci.yml` | ✅ written | lint + tests + schema job |

**Test suite: 42 passed.** `ruff check` clean.

## Measured numbers (Phase 5)

Offline demo corpus: 8 synthetic 10-K filings, 16 chunks, 8 graph nodes, 18 edges. Benchmark:
`python -m kg_rag.cli bench --sample` (10 questions, no API key, no network).

| Metric | Value | Measured on | Command |
| :--- | :--- | :--- | :--- |
| Hybrid accuracy (overall) | 10/10 = 1.00 | 2026-10-16 | `kg_rag.cli bench --sample` |
| Vector-only accuracy (overall) | 2/10 = 0.20 | 2026-10-16 | `kg_rag.cli bench --sample` |
| Delta | +0.80 | 2026-10-16 | `kg_rag.cli bench --sample` |
| 1-hop | hybrid 3/3, vector 1/3 | 2026-10-16 | `kg_rag.cli bench --sample` |
| 2-hop | hybrid 2/2, vector 1/2 | 2026-10-16 | `kg_rag.cli bench --sample` |
| Aggregation | hybrid 2/2, vector 1/2 | 2026-10-16 | `kg_rag.cli bench --sample` |
| Definition | hybrid 1/1, vector 1/1 | 2026-10-16 | `kg_rag.cli bench --sample` |
| Out-of-scope (refuse) | hybrid 2/2, vector 0/2 | 2026-10-16 | `kg_rag.cli bench --sample` |
| Hybrid p50 latency | ~13 ms | 2026-10-16 | `kg_rag.cli bench --sample` |
| Vector-only p50 latency | ~5 ms | 2026-10-16 | `kg_rag.cli bench --sample` |
| Citation defects | 0 | 2026-10-16 | `kg_rag.cli bench --sample` |
| Neo4j constraints online | 2 | 2026-10-16 | `scripts/seed_neo4j.py` |
| Neo4j indexes online | 10 (+2 backing) | 2026-10-16 | `scripts/seed_neo4j.py` |
| pgvector version | 0.8.7 | 2026-10-16 | `scripts/seed_postgres.py` |

---

## Ship gate

- [x] End-to-end offline demo answers with validated citations (`kg_rag.cli demo`)
- [x] Re-running ingestion is idempotent (content-addressed)
- [x] Stratified benchmark table at the top of the README, above the architecture diagram
- [x] `03-decisions.md` has real trade-offs with what was given up (11 recorded)
- [x] README §"What did not work" written honestly

Still honest limitations:
- No FastAPI endpoint yet — CLI is the only interface.
- Offline demo corpus only; real EDGAR ingestion needs `SEC_USER_ACCESS` with a real contact.
- No embedding model yet — `HashingEmbedder` is a deterministic stub; pgvector column exists but is not populated.

## Next (not started)

1. FastAPI endpoint serving `POST /ask` with validated citations.
2. Real EDGAR ingestion (`sec_client.py` with declared `User-Agent`; `budget` command refuses runs over `MAX_INGESTION_USD`).
3. Real embedding model replacing `HashingEmbedder`; populate `chunks.embedding`.
4. 50–100 stratified questions beyond the 10-question demo set.

## Open questions

1. ~~Which corpus?~~ **Resolved:** SEC EDGAR, semiconductors (D-01, D-02).
2. ~~Which registrants?~~ **Resolved:** 8 semiconductor issuers in the offline sample (D-02).
3. **Contact email for the EDGAR User-Agent.** SEC requires a real one and monitors for its
   absence. Must be supplied in `.env`, never committed.
4. **Embedding model + dimension.** Deferred to Phase 2 replacement of `HashingEmbedder`.
