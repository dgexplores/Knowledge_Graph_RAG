# Progress

Measured numbers only. Every entry needs a date and the command that produced it. No assertions.

**Last updated:** 2026-10-16
**Current phase:** Phase 0 — decisions before code

---

## Phase status

| Phase | Status | Exit criterion met? |
| :--- | :--- | :--- |
| 0 — Decisions | 🔄 in progress | — |
| 1 — Extract into Neo4j | ⬜ not started | — |
| 2 — Vector index alongside graph | ⬜ not started | — |
| 3 — Router | ⬜ not started | — |
| 4 — Merge + grounded answer | ⬜ not started | — |
| 5 — Benchmark | ⬜ not started | — |

---

## Phase 0 — required documents

| Doc | Status |
| :--- | :--- |
| `00-corpus.md` | ⬜ |
| `01-ontology.md` | ⬜ |
| `02-data-model.md` | ⬜ |
| `03-decisions.md` | ⬜ |

**Blocked on one decision: which corpus?** SEC filings is the default recommendation — it has real
relationships and a free identifying key (CIK) for entity resolution. Confirm before writing code;
this choice is the most common reason the project dies.

---

## Measured numbers

Nothing measured yet. As each phase lands, record here and mirror into the README header.

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

---

## Ship gate

- [ ] FastAPI endpoint answers with validated citations
- [ ] Re-running ingestion is idempotent
- [ ] 50–100 stratified questions, both systems measured
- [ ] Benchmark table at the top of the README, above the architecture diagram
- [ ] 90-second demo video
- [ ] `03-decisions.md` has at least 2 real trade-offs with what was given up
- [ ] README §"What did not work" written honestly

---

## Open questions

1. **Corpus?** SEC filings / internal wiki / papers / product docs. Blocks Phase 0.
2. **Graph and vector in one Postgres?** Neo4j is external; pgvector can reuse an existing Postgres
   if one is already running. Or run pgvector in its own container.
3. **Model for the router?** A cheap fast model, and generation with Claude. Keeps the router cost
   from dominating per-query cost.
