# 02 — Data model

Concrete Cypher. Written 2026-10-16, before any extraction code.

---

## Neo4j schema

```cypher
// ---- constraints: one row per logical entity, enforced by the database ----
CREATE CONSTRAINT entity_id IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

CREATE CONSTRAINT filing_accession IF NOT EXISTS
  FOR (f:Filing) REQUIRE f.accession IS UNIQUE;

// ---- indexes: every hot traversal path is indexed ----
CREATE INDEX entity_type      IF NOT EXISTS FOR (e:Entity) ON (e.entity_type);
CREATE INDEX entity_cik        IF NOT EXISTS FOR (e:Entity) ON (e.cik);
CREATE INDEX entity_name_norm  IF NOT EXISTS FOR (e:Entity) ON (e.name_norm);
CREATE INDEX filing_form_date  IF NOT EXISTS FOR (f:Filing) ON (f.form, f.filed_at);
CREATE INDEX edge_type_asof    IF NOT EXISTS FOR ()-[r:REL]-() ON (r.type, r.as_of);
CREATE INDEX risk_heading      IF NOT EXISTS FOR (r:RiskFactor) ON (r.heading_norm);
```

`entity_id` is a **deterministic hash**, not a UUID. This is what makes `MERGE` idempotent: the
same logical entity always produces the same id, so re-running ingestion merges instead of
duplicating.

```
entity_id = "ent_" + sha256(f"{entity_type}|{natural_key}")[:24]

natural_key:
  Company      CIK                        → "cik:0000320193"
  Filing       accession                  → "acc:0000320193-24-000012"
  Subsidiary   f"{filer_cik}|{name_norm}" → "0000320193|example holdings llc"
  Officer      f"{filer_cik}|{name_norm}"
  RiskFactor   f"{accession}|{heading_norm}"
  Product      f"{filer_cik}|{name_norm}"
  Auditor      name_norm
  Supplier     name_norm
  Customer     name_norm
  Competitor   name_norm
```

Node labels are the **specific** type (`Company`, `RiskFactor`, …). `Entity` is an additional label
on every node so the uniqueness constraint and the global queries work, but never query `Entity`
alone for domain answers — that is the mistake that makes a graph unusable.

---

## Node properties

```cypher
// Every node, all types
:Entity {
  entity_id     : STRING   // deterministic, unique
  entity_type   : STRING   // enum from config.py
  name          : STRING   // canonical display form
  name_norm     : STRING   // normalized: lowercase, stripped suffixes/punctuation
  aliases       : STRING[] // every accepted alias
  cik           : STRING   // null unless a CIK was found
  confidence    : FLOAT
  mentions      : STRING[] // source chunk ids — provenance for reverse lookup
  mention_count : INTEGER
  model_version : STRING   // extraction model that last touched this node
  first_seen    : DATE
  last_updated  : DATE
}
```

```cypher
// Additional per type
:Filing    { accession, form, filed_at, period_of_report, primary_document, url }
:Subsidiary{ parent_cik, jurisdiction }
:Officer   { role, role_since, role_until }
:RiskFactor{ heading, heading_norm, risk_category }
:Product   { segment }
```

---

## Relationship properties

```cypher
:REL {
  entity_id     : STRING   // deterministic — MERGE key for edges
  type          : STRING   // enum from config.py
  as_of         : DATE     // filing date; every relationship is time-scoped
  confidence    : FLOAT
  source_chunk_id : STRING // the chunk that justified this edge — Phase 4 cites this
  src_document_id : STRING
  model_version : STRING
  extracted_at  : DATETIME
}
```

`source_chunk_id` is the most important property in the schema. It is what lets Phase 4 turn a
graph path into a citation, and it is why the hybrid trick works: the path ends in a chunk id, and
that chunk id is the join key into pgvector.

**Edge id** is `sha256(f"{from_id}|{rel_type}|{to_id}")[:24]` — note it does **not** include `as_of`.
Re-running the same filing must merge; a genuinely new relationship from a later filing is a new
node name (different `as_of`), so both behaviours are correct.

---

## pgvector schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    chunk_id      TEXT PRIMARY KEY,     -- SAME id used in Neo4j source_chunk_id
    document_id   TEXT NOT NULL,
    accession     TEXT,                 -- joins back to :Filing
    section_path  TEXT NOT NULL,        -- e.g. "Item 1A/Risk Factors"
    text          TEXT NOT NULL,
    token_count   INTEGER NOT NULL,
    entity_ids    TEXT[] NOT NULL,      -- entities this chunk mentions → graph neighbourhood
    filed_at      DATE,
    metadata      JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX chunks_entity_ids_idx ON chunks USING GIN (entity_ids);

-- HNSW, tuned in Phase 2 (ef_search is set at query time, not here)
CREATE INDEX chunks_embedding_idx ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

### Why `chunk_id` is the join key

```
Neo4j  (:Filing)-[:MENTIONS_RISK]->(:RiskFactor)-[:MENTIONS_RISK]->(:Filing)
                                                     |
                                        source_chunk_id
                                                     v
pgvector  chunks(chunk_id = '10k-2024#412')  →  the original text
```

A graph path resolves to source text. A retrieved passage resolves to graph entities via
`entity_ids`. Both directions work **only** because the ids are identical. This is the entire
mechanism of hybrid retrieval; breaking the id convention breaks the project.

---

## Write rules

1. **`MERGE` never `CREATE`.** Re-running ingestion must be idempotent.
2. **Every write carries provenance.** `source_chunk_id`, `model_version`, `extracted_at`. An edge
   without provenance cannot be cited and therefore did not need to exist.
3. **One transaction per chunk.** A chunk's entities and relationships land atomically, so a crash
   never leaves a relationship pointing at a node that was not created.
4. **Batch, but bound it.** Chunk entities in batches of ~500 relationships. Unbounded batches
   exhaust the driver's parameter buffer; over-tiny batches make ingestion slow for no reason.

```cypher
// the only shape of write the code may emit — parameterized, no string interpolation
UNWIND $rows AS row
MERGE (e:Entity {entity_id: row.entity_id})
  ON CREATE SET e.entity_id = row.entity_id, e.first_seen = date()
SET e.name = coalesce(row.name, e.name),
    e.name_norm = coalesce(row.name_norm, e.name_norm),
    e.aliases = CASE
                   WHEN row.name IN e.aliases THEN e.aliases
                   ELSE e.aliases + [row.name]
                 END,
    e.mentions = CASE
                    WHEN row.source_chunk_id IN e.mentions THEN e.mentions
                    ELSE e.mentions + [row.source_chunk_id]
                  END,
    e.mention_count = size(e.mentions),
    e.confidence = CASE WHEN row.confidence > e.confidence THEN row.confidence ELSE e.confidence END,
    e.model_version = row.model_version,
    e.last_updated = date()
RETURN count(e) AS merged;
```

`e.confidence` takes the **maximum** observed confidence rather than being overwritten. A
relationship asserted once with 0.99 should not be diluted by being mentioned again at 0.6.

---

## Deletion

Nothing in the graph is deleted by re-ingestion. If a filing is superseded, its relationships get
`superseded_by` set — they are not removed, because a 2022 filing still describes 2022 reality and
a query with a time filter needs it.

Full reset for development only, and it is destructive:

```cypher
MATCH (n) DETACH DELETE n;
```

---

## Known costs, stated up front

- **Extraction dominates cost**, by orders of magnitude. `scripts/budget_ingestion.py` estimates
  before any run; results are cached by document content hash so re-runs cost nothing.
- **Entity resolution is CPU-bound** on embedding comparison, not network-bound. It is the slowest
  phase per document and the easiest to parallelize locally.
- **pgvector HNSW build time** grows faster than linearly. Measure recall@k before assuming a
  larger index helps — recall saturates well before the index gets slow.
