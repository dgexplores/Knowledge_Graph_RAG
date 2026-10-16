"""Cypher template library — decision D-07.

The router returns `{"query_type": ..., "params": {...}}`. Code looks up a **named, pre-written**
Cypher statement and binds the parameters. Model output never reaches the query engine as syntax.

Two independent reasons, both of which matter:

1. **Security.** Parameterized queries. No string interpolation of model output into Cypher, ever.
2. **Reproducibility.** The same question with the same parameters produces byte-identical Cypher,
   so a Phase 5 result traces back to an exact statement.

The cost is expressiveness: only question shapes matching a template are answerable. That ceiling
is real and is reported rather than hidden — see docs/03-decisions.md D-07.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownQueryType(KeyError):
    """The router named a query type with no template. Never falls back to raw Cypher."""

    def __init__(self, query_type: str, available: tuple[str, ...]) -> None:
        super().__init__(query_type)
        self.query_type = query_type
        self.available = available

    def __str__(self) -> str:
        return (
            f"no template for query_type {self.query_type!r}; "
            f"available: {', '.join(sorted(self.available))}"
        )


@dataclass(frozen=True)
class Template:
    """One named query. `cypher` is a constant — parameters are bound, never formatted in."""

    name: str
    description: str
    cypher: str
    required_params: tuple[str, ...]
    hop_count: int = 1
    returns_statements: bool = True


#: Every template carries `as_of` filtering because every relationship is time-scoped (ontology
#: decision). A query without it answers with stale relationships and nobody notices.
_AS_OF = "  AND ($as_of IS NULL OR r.as_of IS NULL OR r.as_of <= date($as_of))"
_REL_TYPES = "['SUPPLIES','CUSTOMER_OF','COMPETES_WITH','PARENT_OF']"

TEMPLATES: dict[str, Template] = {

    "suppliers_of": Template(
        name="suppliers_of",
        description="Companies that supply the given company.",
        hop_count=1,
        required_params=("company",),
        # The filter is on the *customer* side. SUPPLIES is stored supplier -> customer, so
        # "suppliers of X" is "edges whose target is X". Filtering on the source instead returned
        # X's own suppliers-of list as if it were the answer to a different question, and in
        # practice returned nothing at all.
        cypher=f"""
MATCH (s:Company)-[r:SUPPLIES]->(c:Company)
WHERE toLower(c.name) CONTAINS toLower($company)
  AND r.type = 'SUPPLIES'{_AS_OF}
RETURN s.name AS subject, 'SUPPLIES' AS relation, c.name AS object,
       r.as_of AS as_of, r.source_chunk_id AS chunk_id
LIMIT $limit
""",
    ),

    "customers_of": Template(
        name="customers_of",
        description="Companies that buy from the given company.",
        hop_count=1,
        required_params=("company",),
        # No CUSTOMER_OF edges are extracted from the sample filings, so customers are read as the
        # reverse of SUPPLIES: companies that the given company supplies.
        cypher=f"""
MATCH (s:Company)-[r:SUPPLIES]->(c:Company)
WHERE toLower(s.name) CONTAINS toLower($company)
  AND r.type = 'SUPPLIES'{_AS_OF}
RETURN c.name AS subject, 'CUSTOMER_OF' AS relation, s.name AS object,
       r.as_of AS as_of, r.source_chunk_id AS chunk_id
LIMIT $limit
""",
    ),

    "competitors_of": Template(
        name="competitors_of",
        description="Companies the given company names as competitors.",
        hop_count=1,
        required_params=("company",),
        cypher=f"""
MATCH (a:Company)-[r:COMPETES_WITH]->(b:Company)
WHERE toLower(a.name) CONTAINS toLower($company)
  AND r.type = 'COMPETES_WITH'{_AS_OF}
RETURN a.name AS subject, 'COMPETES_WITH' AS relation, b.name AS object,
       r.as_of AS as_of, r.source_chunk_id AS chunk_id
LIMIT $limit
""",
    ),

    "subsidiaries_of": Template(
        name="subsidiaries_of",
        description="Subsidiaries of the given company.",
        hop_count=1,
        required_params=("company",),
        cypher=f"""
MATCH (p:Company)-[r:PARENT_OF]->(s:Subsidiary)
WHERE toLower(p.name) CONTAINS toLower($company)
  AND r.type = 'PARENT_OF'{_AS_OF}
RETURN p.name AS subject, 'PARENT_OF' AS relation, s.name AS object,
       r.as_of AS as_of, r.source_chunk_id AS chunk_id
LIMIT $limit
""",
    ),

    "officers_of": Template(
        name="officers_of",
        description="Named officers of the given company.",
        hop_count=1,
        required_params=("company",),
        cypher=f"""
MATCH (o:Officer)-[r:OFFICER_OF]->(c:Company)
WHERE toLower(c.name) CONTAINS toLower($company)
  AND r.type = 'OFFICER_OF'{_AS_OF}
RETURN o.name AS subject, 'OFFICER_OF' AS relation, c.name AS object,
       r.as_of AS as_of, r.source_chunk_id AS chunk_id
LIMIT $limit
""",
    ),

    "auditor_of": Template(
        name="auditor_of",
        description="The accounting firm auditing the given company.",
        hop_count=1,
        required_params=("company",),
        cypher=f"""
MATCH (c:Company)-[r:AUDITED_BY]->(a:Auditor)
WHERE toLower(c.name) CONTAINS toLower($company)
  AND r.type = 'AUDITED_BY'{_AS_OF}
RETURN c.name AS subject, 'AUDITED_BY' AS relation, a.name AS object,
       r.as_of AS as_of, r.source_chunk_id AS chunk_id
LIMIT $limit
""",
    ),

    "shared_counterparties": Template(
        name="shared_counterparties",
        description=(
            "Companies connected to BOTH given companies. The 2-hop join: the answer is not in any "
            "single passage, which is the whole reason the graph exists."
        ),
        hop_count=2,
        required_params=("company_a", "company_b"),
        cypher=f"""
MATCH (x:Company)
WHERE toLower(x.name) CONTAINS toLower($company_a)
   OR toLower(x.name) CONTAINS toLower($company_b)
WITH collect(DISTINCT x) AS anchors
UNWIND anchors AS a
MATCH (a)-[r]->(counterparty:Company)
WHERE NOT counterparty IN anchors
  AND r.type IN {_REL_TYPES}
  AND ($as_of IS NULL OR r.as_of IS NULL OR r.as_of <= date($as_of))
WITH counterparty,
     collect(DISTINCT a.name)          AS connected_to,
     collect(DISTINCT r.source_chunk_id) AS chunk_ids
WHERE size(connected_to) > 1
RETURN counterparty.name AS object,
       counterparty.name + ' connected to ' + reduce(acc = connected_to[0], name IN tail(connected_to) |
                acc + ' and ' + name) AS statement,
       connected_to AS anchors,
       head(connected_to) AS anchor,
       head(chunk_ids) AS chunk_id
ORDER BY size(connected_to) DESC, counterparty.name
LIMIT $limit
""",
    ),

    "counterparty_chain": Template(
        name="counterparty_chain",
        description=(
            "Companies reachable from the given company in two relationship hops. The 3-hop demo "
            "question resolves through this."
        ),
        hop_count=2,
        required_params=("company",),
        cypher=f"""
MATCH (start:Company)-[r1]->(mid:Company)
WHERE toLower(start.name) CONTAINS toLower($company)
  AND r1.type IN {_REL_TYPES}
  AND ($as_of IS NULL OR r1.as_of IS NULL OR r1.as_of <= date($as_of))
MATCH (mid)-[r2]->(end:Company)
WHERE NOT end = start
  AND r2.type IN {_REL_TYPES}
  AND ($as_of IS NULL OR r2.as_of IS NULL OR r2.as_of <= date($as_of))
RETURN start.name AS from_name, r1.type AS hop1_type, mid.name AS via_name,
       r2.type AS hop2_type, end.name AS to_name,
       r1.source_chunk_id AS chunk_id,
       r1.as_of AS as_of
LIMIT $limit
""",
    ),

    "count_relationships": Template(
        name="count_relationships",
        description=(
            "Aggregation: how many entities are connected to the given company, by relation type. "
            "Vector retrieval cannot answer this at all."
        ),
        hop_count=1,
        required_params=("company",),
        cypher="""
MATCH (c:Company)-[r]-(other:Entity)
WHERE toLower(c.name) CONTAINS toLower($company)
  AND r.type <> 'FILES'
  AND ($as_of IS NULL OR r.as_of IS NULL OR r.as_of <= date($as_of))
WITH c, r, other
ORDER BY r.source_chunk_id
WITH c.name AS subject, r.type AS relation,
     count(DISTINCT other) AS count,
     [x IN collect(DISTINCT other.name) | x][0..$limit] AS examples,
     collect(DISTINCT r.source_chunk_id) AS chunk_ids
RETURN subject, relation, count, examples, chunk_ids
ORDER BY count DESC
""",
    ),

    "entity_exists": Template(
        name="entity_exists",
        description="Whether the named entity exists in the graph, and what type it is.",
        hop_count=0,
        returns_statements=False,
        required_params=("company",),
        cypher="""
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($company)
RETURN e.name AS name, e.entity_type AS entity_type, coalesce(e.cik, '') AS cik,
       size(coalesce(e.mentions, [])) AS mention_count
LIMIT $limit
""",
    ),
}

DEFAULT_LIMIT = 50


def get_template(query_type: str) -> Template:
    """Look up a template. Raises rather than ever falling back to executing free-form Cypher."""
    template = TEMPLATES.get(query_type)
    if template is None:
        raise UnknownQueryType(query_type, tuple(TEMPLATES))
    return template


def available_types() -> tuple[str, ...]:
    return tuple(TEMPLATES)


def describe_templates() -> str:
    """Template catalogue for the router prompt. Generated from TEMPLATES so it cannot drift."""
    lines = []
    for template in TEMPLATES.values():
        params = ", ".join(template.required_params)
        lines.append(
            f"  - {template.name}({params}): {template.description} "
            f"[{template.hop_count} hop(s)]"
        )
    return "\n".join(lines)


def bind(
    template: Template,
    params: dict[str, object],
    as_of: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, object]:
    """Build the parameter dict for a template execution.

    Every required parameter must be present and be a string. A missing parameter raises rather
    than producing a query that silently matches nothing — a silently empty result is
    indistinguishable from "the graph does not contain this relationship".
    """
    bound: dict[str, object] = {"as_of": as_of, "limit": limit}
    for required in template.required_params:
        value = params.get(required)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"template {template.name!r} requires parameter {required!r}")
        if not isinstance(value, str):
            raise TypeError(
                f"template {template.name!r} parameter {required!r} must be a string, "
                f"got {type(value).__name__}"
            )
        bound[required] = value.strip()
    return bound
