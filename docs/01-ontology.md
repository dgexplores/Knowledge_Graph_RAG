# 01 — Ontology

**10 entity types, 10 relationship types.** Constrained on purpose. Written 2026-10-16, before any
extraction code.

An open-ended "extract all entities" prompt produces a graph nobody can query: thousands of
`OTHER` nodes, relationship types that vary per document, and no way to write a parameterized
query that is worth anything. The constraint *is* the feature.

---

## Entity types

| Type | Identifying key | Canonical form | Resolution strategy |
| :--- | :--- | :--- | :--- |
| `Company` | CIK when present | legal name as filed | 1. CIK 2. ticker 3. normalized name 4. embedding |
| `Subsidiary` | parent's CIK + local name | as written in filing | normalized name, scoped to parent CIK |
| `Officer` | person name + filer CIK | `Firstname Lastname` | normalized name, scoped to filer CIK |
| `Auditor` | PCAOB firm id when available | firm name as filed | normalized name |
| `Supplier` | — (usually unnamed) | as written | normalized + alias |
| `Customer` | — (usually unnamed) | as written | normalized + alias |
| `Competitor` | CIK when named as a registrant | as written | normalized + alias |
| `RiskFactor` | risk heading + filer CIK | heading text | heading, scoped to filing |
| `Product` | segment/product name as filed | as written | normalized, scoped to filer CIK |
| `Filing` | accession number | form type + date | exact accession |

### The three keys that make this tractable

- **CIK** is authoritative for `Company`. It ends the resolution cascade early for most registrants.
- **`(filer_cik, local_name)`** is the composite key for `Subsidiary`, `Officer`, `RiskFactor`,
  `Product`. Same person or same risk heading in two different filings is a *different* node
  unless they name each other, which is correct: consolidation across filings is a merge decision
  the data should make, not the schema.
- **`accession`** is the natural key for `Filing` and is already unique.

### Why `Subsidiary` is not just `Company`

A named subsidiary of Apple is a separate legal entity with its own CIK, but in a parent's 10-K it
appears as bare text with no CIK. Modelling it as `Company` invites CIK-matching failures on a node
that has no CIK. Modelling it separately keeps "does this have an authoritative key?" a property
of the type rather than a per-node surprise.

---

## Relationship types

| Type | From → To | Cardinality | Symmetric | Time-scoped |
| :--- | :--- | :--- | :--- | :--- |
| `FILES` | Company → Filing | 1:N | no | yes (filing date) |
| `PARENT_OF` | Company → Subsidiary | 1:N | no | yes (as-of date) |
| `SUBSIDIARY_OF` | Subsidiary → Company | N:1 | no | yes |
| `OFFICER_OF` | Officer → Company | N:M | no | yes (role period) |
| `AUDITED_BY` | Company → Auditor | N:M | no | yes (fiscal year) |
| `SUPPLIES` | Company → Company | N:M | no | yes |
| `CUSTOMER_OF` | Company → Company | N:M | no | yes |
| `COMPETES_WITH` | Company ↔ Company | N:M | **yes** | yes |
| `MENTIONS_RISK` | Filing → RiskFactor | 1:N | no | no |
| `REPORTS_SEGMENT` | Filing → Product | 1:N | no | yes |

### Decisions worth defending

**`PARENT_OF` and `SUBSIDIARY_OF` are both stored.** Normally you store one and traverse
backwards. Both are stored because the direction you need is not always the direction you will ask
in, and a 3-hop Cypher query that flips direction mid-string is where bugs live. Cost is one extra
edge per subsidiary; correctness is worth more. *(Alternative: store one, expose both via Cypher
patterns. Revisit if edge count becomes a problem — unlikely at this scale.)*

**`COMPETES_WITH` is symmetric.** Storing it twice (A→B and B→A) makes "who competes with X" a
single-hop query with no `WHERE` gymnastics, at the cost of a duplicate that must be written
atomically. At this scale, symmetry is worth more than storage.

**Everything is time-scoped.** A supply relationship in 2021 is not evidence about 2024. Every
relationship carries `as_of` (filing date) and the queries filter on it. Skipping this means the
graph confidently answers with stale relationships and nobody notices.

**There is no `MENTIONS` relationship.** Overlap between the same entity and two different filing
texts is stored as `mentions[]` on the node — an array property, not an edge. This would be
quadratic in edges for no query benefit, since nothing traverses "who mentioned" as a graph step.

---

## Extraction output schema

The model returns this and nothing else. Validated against the enums; failures retried.

```json
{
  "entities": [
    {
      "local_id": "e1",
      "type": "Subsidiary",
      "name": "Example Holdings LLC",
      "aliases": ["Example Holdings"],
      "cik": null,
      "confidence": 0.95,
      "source_chunk_id": "10k-2024#412"
    }
  ],
  "relationships": [
    {
      "from_local_id": "e1",
      "to_local_id": "e2",
      "type": "PARENT_OF",
      "attributes": {"as_of": "2024-11-01"},
      "confidence": 0.91,
      "source_chunk_id": "10k-2024#412"
    }
  ]
}
```

`local_id` is scoped to the extraction call. The resolver maps `local_id` → global entity id.
The model never emits a global id, a Cypher statement, or a node label it invented.

**Validation rules, enforced in code — the model cannot bypass them:**

1. Every `type` is in the enum, or the whole response is rejected.
2. Every `from_local_id`/`to_local_id` resolves to an entity in the same response.
3. Every `source_chunk_id` is a chunk that was actually sent for extraction.
4. `confidence` is in [0, 1].
5. `relationships[].type` is legal for the given endpoint types — `PARENT_OF` from a
   `RiskFactor` is rejected, not stored.

Rule 5 is the one that keeps the graph queryable. It is enforced from a static compatibility table,
not from the model's judgment.

---

## Compatibility table (enforced, not promptable)

| Relationship | Legal from | Legal to |
| :--- | :--- | :--- |
| `FILES` | Company | Filing |
| `PARENT_OF` | Company | Subsidiary |
| `SUBSIDIARY_OF` | Subsidiary | Company |
| `OFFICER_OF` | Officer | Company |
| `AUDITED_BY` | Company | Auditor |
| `SUPPLIES` | Company | Company |
| `CUSTOMER_OF` | Company | Company |
| `COMPETES_WITH` | Company | Company |
| `MENTIONS_RISK` | Filing | RiskFactor |
| `REPORTS_SEGMENT` | Filing | Product |

Anything outside this table is rejected at validation. This table is the single source of truth in
code (`kg_rag/config.py`), and the extraction prompt is generated from it — so the prompt cannot
drift from the schema.

---

## Extension rule

Adding a type is allowed **only** with a stated query it serves. An entity type that no Cypher
query ever touches is a cost with no return: more extraction tokens, more resolution mistakes, more
graph noise. Write the query first, then add the type.

At 10 entity and 10 relationship types the ontology is at the upper bound of what stays
queryable. The next addition should replace something, not stack on top.
