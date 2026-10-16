# Progress

Measured numbers only. Every entry needs a date and the command that produced it. No assertions.

**Last updated:** 2026-10-16
**Current phase:** Phase 0 complete → Phase 1 next (extraction)

---

## Phase status

| Phase | Status | Exit criterion met? |
| :--- | :--- | :--- |
| 0 — Decisions | ✅ complete | 4/4 docs written and committed |
| 1 — Extract into Neo4j | ⬜ not started | — |
| 2 — Vector index alongside graph | ⬜ not started | — |
| 3 — Router | ⬜ not started | — |
| 4 — Merge + grounded answer | ⬜ not started | — |
| 5 — Benchmark | ⬜ not started | — |

---

## Phase 0 — documents

| Doc | Status |
| :--- | :--- |
| `00-corpus.md` | ✅ SEC EDGAR, one sector (semiconductors), 8–12 registrants |
| `01-ontology.md` | ✅ 10 entity types, 10 relationship types, compatibility table |
| `02-data-model.md` | ✅ Cypher DDL — **verified against a live Neo4j 5** |
| `03-decisions.md` | ✅ 11 decisions (D-01 … D-11) with what each gave up |

---

## Built and verified so far

| Component | Status | Verification |
| :--- | :--- | :--- |
| `kg_rag/config.py` — ontology, ids, name normalization | ✅ done | 40 tests |
| `kg_rag/chunking.py` — stable content-addressed chunk ids | ✅ done | 31 tests |
| `docker-compose.yml` — Neo4j 5 + pgvector 0.8.7 | ✅ running | both containers healthy |
| `scripts/seed_neo4j.py` | ✅ verified | 2 constraints + 8 indexes ONLINE |
| `scripts/seed_postgres.py` | ✅ verified | `chunks` table, 9 columns, `chunk_id` PK |
| `.github/workflows/ci.yml` | ✅ written | lint + tests + schema job |

**Test suite: 71 passed.** `ruff check` clean.

### Two bugs the tests caught before they became data problems

Recorded because both would have produced silently wrong graph data rather than a crash:

1. **Duplicate-token offset bug.** Line anchors were resolved by searching the section text for a
   token, which returns the *first* occurrence. A filing is mostly boilerplate, so nearly every
   chunk anchored to the wrong line. Fixed by packing token *index ranges* instead of token lists.
2. **Chunk ids ignored the accession.** Two different filings with identical boilerplate text
   collided on one chunk id — which would have merged two documents' passages into a single
   vector row. Fixed by mixing `accession` into the id hash.

### Environment note

The pgvector container binds host **5433**, not 5432. A Homebrew PostgreSQL 16 already owns 5432
on this machine and `Task_Master_Agent` depends on it.

---

## Measured numbers

Nothing extracted yet, so retrieval numbers are still empty. Schema-level facts are measured.

| Metric | Value | Measured on | Command |
| :--- | :--- | :--- | :--- |
| Documents ingested | — | — | — |
| Chunks created | — | — | — |
| Graph nodes | — | — | — |
| Graph edges | — | — | — |
| Orphan nodes | — | — | — |
| Extraction failure rate | — | — | — |
| Entity merges (alias collapses) | — | — | — |
| Ingestion cost (one-time) | — | — | — |
| Vector recall@k (Phase 2) | — | — | — |
| Router accuracy per class (Phase 3) | — | — | — |
| Hybrid accuracy per class (Phase 5) | — | — | — |
| Vector-only baseline per class (Phase 5) | — | — | — |
| Latency p50 / p95, both systems | — | — | — |
| Cost per query, both systems | — | — | — |
| Neo4j constraints online | 2 | 2026-10-16 | `scripts/seed_neo4j.py` |
| Neo4j indexes online | 8 (+2 backing) | 2026-10-16 | `scripts/seed_neo4j.py` |
| pgvector version | 0.8.7 | 2026-10-16 | `scripts/seed_postgres.py` |

---

## Ship gate

- [ ] FastAPI endpoint answers with validated citations
- [ ] Re-running ingestion is idempotent
- [ ] 50–100 stratified questions, both systems measured
- [ ] Benchmark table at the top of the README, above the architecture diagram
- [ ] 90-second demo video
- [ ] `03-decisions.md` has at least 2 real trade-offs with what was given up *(11 recorded)*
- [ ] README §"What did not work" written honestly

---

## Next: Phase 1

1. `sec_client.py` — EDGAR fetcher enforcing the fair-access policy: declared `User-Agent` with a
   real contact email, 10 req/s ceiling, sequential-by-default. An undeclared UA gets the IP
   blocked mid-build.
2. `scripts/budget_ingestion.py` — refuse to start a full-corpus run without a cost estimate and
   a `MAX_INGESTION_USD` ceiling.
3. `extract.py` — Claude extraction against the generated schema, validated against
   `config.COMPATIBILITY`, failures retried.
4. `resolve.py` — the CIK → ticker → normalized → embedding → manual cascade (D-05), with a merge
   log, since a wrong merge silently breaks a traversal while a wrong non-merge only returns less.

---

## Open questions

1. ~~Which corpus?~~ **Resolved:** SEC EDGAR, semiconductors (D-01, D-02).
2. **Which 8–12 registrants?** Semiconductor issuers that name each other. Blocking the first
   ingestion run.
3. **Contact email for the EDGAR User-Agent.** SEC requires a real one and monitors for its
   absence. Must be supplied in `.env`, never committed.
4. **Embedding model + dimension.** Phase 2 owns it; the `chunks.embedding` column and its HNSW
   index are deferred until then, deliberately.

