# 03 — Decision log

Every decision, the options considered, what we chose, and **what we gave up**. This file is
append-only. The README's "Design decisions" section is a distillation of the top entries, not a
separate write-up.

Format: `D-nn` · date · status.

---

## D-01 · 2026-10-16 · SEC EDGAR as the corpus

**Decision.** SEC EDGAR filings, one sector, 8–12 registrants.

**Options.** (a) SEC filings — real relationships, public domain, documented API. (b) Internal
wiki — private, needs access, relationships are implicit. (c) Papers with citations — citation
graph is free, but domain language is dense and entity types sprawl. (d) Product docs — good
dependency chains, smaller scale.

**Chose** (a). Cross-registrant naming gives multi-hop joins that no other option guarantees, CIK
gives an authoritative entity key, and public-domain status removes a licensing question from the
README.

**Gave up.** A single-sector corpus means no cross-sector generalization claim. XBRL financial
numbers are deferred, so this system answers relationship questions, not financial ones. Both are
acceptable and will be stated as limitations rather than hidden.

**Revisit if.** Extraction produces a graph where cross-registrant edges are sparse — then
concentration was not achieved and a denser sector is needed before anything else.

---

## D-02 · 2026-10-16 · One sector, not "leading companies"

**Decision.** Take 8–12 issuers from **one** sector, not 12 large companies across sectors.

**Options.** (a) One sector. (b) Cross-sector mega-caps. (c) Whole index.

**Chose** (a). The demo question is "which suppliers to Subsidiary A also appear in a risk factor
filed by Subsidiary B". That requires A and B to name each other, which happens by construction
inside one sector and largely does not happen across sectors.

**Gave up.** Sector-specific vocabulary in the extraction prompt, and no claim that the approach
generalizes across industries. This is the single decision most likely to silently produce a
disconnected forest, and it was made deliberately.

**Revisit if.** The degree check in `docs/04-ingestion-run.md` shows a graph with one component per
company.

---

## D-03 · 2026-10-16 · `PARENT_OF` and `SUBSIDIARY_OF` both stored

**Decision.** Store both directions as separate edges.

**Options.** (a) Store one and traverse bidirectionally in Cypher. (b) Store both.

**Chose** (b). A 3-hop query that changes direction mid-pattern is where bugs live, and Phase 5
correctness matters more than edge count at a corpus of this size.

**Gave up.** Roughly 2× the edges for subsidiary relationships, and the invariant that both
directions must be written atomically — a half-written pair is a silent bug. Mitigated by writing
both in the same Cypher statement.

**Revisit if.** Edge count reaches a scale where traversal latency visibly suffers.

---

## D-04 · 2026-10-16 · `COMPETES_WITH` stored symmetrically

**Decision.** Write A→B and B→A.

**Options.** (a) One edge, bidirectionally traversed. (b) Both directions.

**Chose** (b), same reasoning as D-03: "who competes with X" becomes a single-hop lookup.

**Gave up.** Duplicate edges that must never drift apart. Mitigated by writing both directions in
one statement with a deterministic edge id per direction.

---

## D-05 · 2026-10-16 · CIK first, embedding similarity last

**Decision.** Entity resolution cascades: CIK → ticker → normalized name → embedding similarity
above a tuned threshold → manual alias table.

**Options.** (a) Pure embedding similarity. (b) Normalized-name exact match only. (c) The cascade.

**Chose** (c). CIK match is authoritative and ends the cascade for most registrants, so the
ambiguous step — the one that produces wrong merges — only ever runs on entities with no CIK, which
is a small minority.

**Gave up.** A tuning burden: the similarity threshold must be measured against a hand-built set of
known-same and known-different pairs before it is trusted, and the merge log must be reviewed
because a wrong merge silently breaks a traversal while a wrong non-merge just returns less.

**Revisit if.** The merge log shows systematic over-merging — threshold and name-normalization are
the two knobs, in that order.

---

## D-06 · 2026-10-16 · Ontology generated from code, prompt cannot drift

**Decision.** The entity/relationship enums and the relationship compatibility table live in
`kg_rag/config.py`. The extraction prompt is **generated** from them.

**Options.** (a) Hand-write the prompt and keep the schema separate. (b) Generate the prompt from
the schema.

**Chose** (b). A hand-written prompt and a separately-maintained schema inevitably disagree, and the
disagreement is discovered as bad graph data rather than as a bug. One source of truth makes drift
impossible by construction.

**Gave up.** Slightly more indirection — the prompt is not readable in isolation without opening
`config.py`.

---

## D-07 · 2026-10-16 · Model fills parameters; never emits Cypher

**Decision.** A template library keyed by query type. The router returns
`{"query_type": "...", "params": {...}}` and code substitutes bound parameters into a
pre-written Cypher template.

**Options.** (a) Model emits Cypher text. (b) Template library with parameters.

**Chose** (b), on two independent grounds. It is a security control — model output never reaches
the query engine as syntax. And it makes results reproducible: the same question with the same
params produces the same query, so a Phase 5 result can be traced to an exact statement.

**Gave up.** Expressiveness. Only question shapes that match a template are answerable, so the
router must map free-form questions onto a fixed set, and an unmatched shape has no graceful path
other than falling back to vector retrieval.

**Revisit if.** Template coverage proves to be the accuracy ceiling — then the trade is real and
the ceiling is worth reporting as a finding.

---

## D-08 · 2026-10-16 · Measure vector recall@k before building the router

**Decision.** Phase 2 ends with a measured recall@k. Phase 3 does not start until it exists.

**Options.** (a) Router first, evaluate the whole system at the end. (b) Measure retrieval first.

**Chose** (b). Without a per-path measurement, a bad hybrid result cannot be attributed — a wrong
answer could be a bad route, a bad traversal, or bad retrieval underneath both. Measuring first
makes every later failure attributable.

**Gave up.** Time before the first interesting demo. Roughly a day of work on a labelled query set
that produces no visible feature.

---

## D-09 · 2026-10-16 · Low-confidence router runs both paths and merges

**Decision.** The router returns an enum **and** a confidence. Below threshold → run graph **and**
vector, merge.

**Options.** (a) Enum only. (b) Enum plus confidence with a both-path fallback. (c) Always run both.

**Chose** (b). Option (c) doubles cost for the easy 1-hop majority where the graph adds nothing.
Option (a) makes every borderline question a coin flip. The fallback rate becomes a measured number
in its own right — it is the system's own statement of how often it is unsure.

**Gave up.** The fallback path is a second code path that needs its own dedupe and merge logic in
Phase 4, and its latency is the sum of both paths rather than the faster one.

---

## D-10 · 2026-10-16 · Reuse `Groundtruth`; do not write a second eval harness

**Decision.** Phase 5 reports through the existing `Groundtruth` harness and its regression-gate
pattern.

**Options.** (a) Extend `Groundtruth` to score this system. (b) New standalone benchmark script.

**Chose** (a). Two eval harnesses in one portfolio is the "six half-finished repos" failure mode in a
different costume. `Groundtruth` already measures recall@k, already gates CI on a metric drop, and
already documents its own failure modes honestly.

**Gave up.** A hard coupling between this project and another repo's release cycle — a change to
`Groundtruth` can break this project's CI. Accepted deliberately; the alternative is duplicated
measurement logic that drifts.

**Revisit if.** `Groundtruth`'s schema cannot represent per-hop-class breakdown without being
generalized, and generalizing it endangers the other project's clean interface. In that case a
thin adapter is added **here**, and `Groundtruth` keeps its own interface.

---

## D-11 · 2026-10-16 · Confidence accumulates by maximum, not last-write

**Decision.** `e.confidence` takes the max observed value; aliases and mentions accumulate as
sets.

**Options.** (a) Last write wins. (b) Max confidence, accumulating arrays.

**Chose** (b). Re-ingestion must not lower a well-evidenced fact's confidence because a later filing
mentions it more weakly. Set semantics also make `MERGE` naturally idempotent — no read-modify-write
race on arrays.

**Gave up.** Max-confidence is not a calibrated probability and must never be reported as one. It is
a ranking hint for resolution and merge ordering, nothing more.
