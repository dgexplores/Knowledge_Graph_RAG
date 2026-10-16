# 00 — Corpus

**Decision: SEC EDGAR filings.** Written 2026-10-16, before any extraction code.

---

## Why this corpus

The project dies on a corpus with no relationships. A knowledge graph over unrelated blog posts is
a graph of nothing. This corpus was chosen because it has three properties the graph needs:

1. **Real multi-hop structure.** Subsidiaries, suppliers, customers, competitors, officers and
   auditors are *named inside other companies' filings*. A question like "which suppliers to
   Subsidiary A also appear in a risk factor for Subsidiary B in the same fiscal year" has no
   single containing passage — it requires joining across documents.
2. **A free authoritative identifying key.** CIK is a zero-padded 10-digit registrant id, unique and
   stable. This is what makes entity resolution tractable: CIK match is authoritative, so the
   ambiguous embedding-similarity step only ever runs on entities with no CIK.
3. **Machine-readable structure for the non-graph half.** XBRL company facts give numeric financial
   data without parsing prose, so the graph handles *relationships* and XBRL handles *numbers*.

---

## Acquisition

SEC EDGAR is a documented public API. No key required. Three endpoints do the work:

| Endpoint | Yields | Use |
| :--- | :--- | :--- |
| `https://www.sec.gov/files/company_tickers.json` | ticker → CIK map | company selection |
| `https://data.sec.gov/submissions/CIK##########.json` | filing history: form type, date, accession number, document URLs | which filings to ingest |
| `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | structured XBRL facts over time | financial metrics, segments, revenue |

Filing documents themselves come from
`https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_document}`.

### Access policy — this is not optional

The SEC publishes a fair-access policy and violating it gets your IP blocked mid-build:

- **A declared `User-Agent` on every request**, containing a real contact email. SEC explicitly
  requires this and monitors for its absence.
- **10 requests per second maximum.**
- Politeness over throughput: sequential requests with a small delay is fine for a personal build.

A scraper that ignores this is the reason someone's build died at 40% ingestion. The client in
`kg_rag/sec_client.py` must enforce all three, not just the UA.

---

## Scope

Start small and deliberately. A 10-K for one company is enough to prove the pipeline.

| | Value | Why |
| :--- | :--- | :--- |
| Registrants | 8–12 | enough for cross-company 2-hop and 3-hop joins |
| Form types | 10-K, 10-Q, 8-K | K = annual narrative, Q = interim, 8-K = events |
| Sections from K | Item 1A (risk factors), Item 1 (business), Item 7 (MD&A) | where named relationships actually appear |
| Target corpus size | ~150–400 documents | ~3–6 weeks part time |
| Target chunks | 3k–8k | enough for recall@k to be meaningful, small enough to iterate |

### Entity concentration matters more than document count

Pick registrants that **name each other**. An airline, a supplier, and a competitor all filing in
the same window produces edges that a set of unrelated issuers never will. Concentration is the
difference between a connected graph and a forest of isolated stars.

A public-company universe with heavy cross-naming by construction: **airlines, semiconductors,
automakers, pharma**. Choose one sector first, then take 8–12 issuers from it. Cross-sector
selection produces isolated components.

---

## Licensing and public nature

SEC filings are **US government works in the public domain**. No licence encumbrance, no
attribution requirement, no commercial restriction. This is a genuine advantage over a scraped
corpus and should be stated in the README.

---

## The three questions that must be answerable

Written now, before the system exists. If the build finishes and these three fail, the ontology or
the extraction is wrong — and these become the seed of the Phase 5 benchmark set.

**Q1 — 1-hop / vector path.**
> What does Company A's 10-K for fiscal 2024 say about supply chain concentration risk?

Sourced from a single passage. Should route to vector and land on vector retrieval parity.

**Q2 — 2-hop / graph path.**
> Which companies are named as both a competitor and a risk-factor subject in Company A's filings?

Requires: Company A →[MENTIONS_RISK]→ RiskFactor ←[MENTIONS_RISK]← Company B, then a second
relationship edge between A and B. No single passage contains this.

**Q3 — 3-hop / graph path.**
> Which suppliers to Subsidiary A of Company A also appear in a risk factor filed by Subsidiary B
> in the same fiscal year?

Requires joining across *two different registrants' documents*. This is the demo question and the
headline of the demo video — if a reviewer watches 90 seconds, it should be this.

---

## Out of scope

Recording these now prevents scope creep later:

- Full-text search across all of EDGAR (petabytes; needs EDGAR FTS and heavy filtering)
- PDFs and scanned exhibits — EDGAR HTML documents are the target
- Non-US filings — EDGAR is US-only, no IFRS/foreign equivalents
- Insider transaction forms (4) — too high volume, low relationship density for this project

---

## Open questions

1. **Sector pick.** Semiconductors or airlines? Semiconductors have richer supply-chain language;
   airlines have heavier subsidiary structures. Semiconductors recommended — more
   supplier/customer edges, which is exactly the 3-hop shape the demo needs.
2. **XBRL depth.** Pull `companyfacts` in Phase 1 or defer to a later phase? Recommended: defer.
   The graph is the deliverable; financial numbers can come from a second system later.
