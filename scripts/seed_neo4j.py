"""Apply the Neo4j constraints and indexes from docs/02-data-model.md.

The DDL is verified against a running database rather than trusted. Run after
`docker compose up -d neo4j`.

    python scripts/seed_neo4j.py

Refuses to run when NEO4J_PASSWORD is unset or still at its default: seeding a developer's
machine with a throwaway password and then shipping that password into .env is how a real
credential reaches a public repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# `python scripts/seed_neo4j.py` makes `scripts/` the import root. Add the repository root so the
# documented command works from a fresh clone without first installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neo4j import GraphDatabase

from kg_rag.config import RelationshipType

DEFAULT_PASSWORD = "kg_rag_dev"

CONSTRAINTS = [
    (
        "CREATE CONSTRAINT entity_id IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT filing_accession IF NOT EXISTS "
        "FOR (f:Filing) REQUIRE f.accession IS UNIQUE"
    ),
]

# Edges are written with concrete relationship types (see GraphStore.upsert_relationships), so the
# index has to name them. An index on a `:REL` label that no longer exists is dead DDL: Neo4j
# accepts it, it is never used, and it looks like coverage in a schema review.
#
# One index per type rather than a single `-[r:A|B|C]-()` alternation: Neo4j 5.26 Community
# rejects alternation in CREATE INDEX, so the compact form is a syntax error here.
EDGE_INDEXES = [
    f"CREATE INDEX edge_type_asof_{rel_type.value.lower()} IF NOT EXISTS "
    f"FOR ()-[r:{rel_type.value}]-() ON (r.type, r.as_of)"
    for rel_type in sorted(RelationshipType, key=lambda r: r.value)
]

#: The old generic index. Neo4j has no "IF NOT EXISTS DROP", and dropping a missing index is an
#: error, so this is attempted and its failure ignored — it is a migration, not a requirement.
LEGACY_INDEX_DROPS = ["DROP INDEX edge_type_asof IF EXISTS"]

INDEXES = [
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX entity_cik IF NOT EXISTS FOR (e:Entity) ON (e.cik)",
    "CREATE INDEX entity_name_norm IF NOT EXISTS FOR (e:Entity) ON (e.name_norm)",
    "CREATE INDEX filing_form_date IF NOT EXISTS FOR (f:Filing) ON (f.form, f.filed_at)",
    *EDGE_INDEXES,
    "CREATE INDEX risk_heading IF NOT EXISTS FOR (r:RiskFactor) ON (r.heading_norm)",
]


def main() -> int:
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password or password == DEFAULT_PASSWORD:
        print(
            f"Refusing to seed. Set NEO4J_PASSWORD in your environment "
            f"(the docker-compose default '{DEFAULT_PASSWORD}' is development-only and must not "
            f"reach .env or a commit).",
            file=sys.stderr,
        )
        return 1

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            # Drop the superseded generic index before creating the per-type ones, so a reseed
            # does not leave both the dead and the live index in place.
            for statement in LEGACY_INDEX_DROPS:
                try:
                    session.run(statement).consume()
                    print(f"  dropped {statement}")
                except Exception as exc:  # noqa: BLE001 - absence of a legacy index is expected
                    print(f"  skip    {statement} ({type(exc).__name__})")

            for statement in CONSTRAINTS + INDEXES:
                session.run(statement).consume()
                print(f"  ok  {statement}")

            constraints = session.run("SHOW CONSTRAINTS YIELD name RETURN name").values()
            indexes = session.run("SHOW INDEXES YIELD name, state RETURN name, state").values()

    stale = [
        name
        for name, *_ in indexes
        if name == "edge_type_asof"
    ]
    if stale:
        print(
            "\nWARNING: generic :REL index still present — it is dead DDL now that edges are "
            "typed.",
            file=sys.stderr,
        )

    print(f"\n{len(constraints)} constraints, {len(indexes)} indexes present.")
    for name, *rest in indexes:
        state = rest[0] if rest else ""
        print(f"  index {name} [{state}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
