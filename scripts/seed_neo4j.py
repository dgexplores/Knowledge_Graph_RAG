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

from neo4j import GraphDatabase

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

INDEXES = [
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX entity_cik IF NOT EXISTS FOR (e:Entity) ON (e.cik)",
    "CREATE INDEX entity_name_norm IF NOT EXISTS FOR (e:Entity) ON (e.name_norm)",
    "CREATE INDEX filing_form_date IF NOT EXISTS FOR (f:Filing) ON (f.form, f.filed_at)",
    "CREATE INDEX edge_type_asof IF NOT EXISTS FOR ()-[r:REL]-() ON (r.type, r.as_of)",
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
            for statement in CONSTRAINTS + INDEXES:
                session.run(statement).consume()
                print(f"  ok  {statement}")

            constraints = session.run("SHOW CONSTRAINTS YIELD name RETURN name").values()
            indexes = session.run("SHOW INDEXES YIELD name, state RETURN name, state").values()

    print(f"\n{len(constraints)} constraints, {len(indexes)} indexes present.")
    for name, *rest in indexes:
        state = rest[0] if rest else ""
        print(f"  index {name} [{state}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
