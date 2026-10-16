"""Neo4j writer.

Every statement is a `MERGE` keyed on a deterministic id, so re-running ingestion is idempotent
rather than duplicating. Every node and edge carries provenance (`source_chunk_id`,
`model_version`), because an edge that cannot be cited did not need to exist — that is what makes
Phase 4's citation validation possible.

No Cypher string is ever built by interpolating model output. All parameters are bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from kg_rag.config import EntityType, RelationshipType, normalize_name
from kg_rag.resolve import ResolvedEntity, ResolvedRelationship

# Nodes are written in one batched UNWIND per entity type. The type label is substituted from the
# `EntityType` enum, which is code — never model output — so building the statement per type is
# safe. The alternative, `apoc.create.addLabels`, would need the APOC plugin, which the
# neo4j-community image does not ship.
_UPSERT_NODE_TEMPLATE = """
UNWIND $rows AS row
MERGE (e:Entity:{label} {{entity_id: row.entity_id}})
  ON CREATE SET e.entity_id = row.entity_id, e.first_seen = date()
SET e.entity_type    = row.entity_type,
    e.name           = row.name,
    e.name_norm      = row.name_norm,
    e.cik            = row.cik,
    e.filer_cik      = row.filer_cik,
    e.confidence     = CASE WHEN row.confidence > coalesce(e.confidence, 0.0)
                            THEN row.confidence ELSE e.confidence END,
    e.aliases        = reduce(acc = coalesce(e.aliases, []), a IN row.surfaces |
                             CASE WHEN a IN acc THEN acc ELSE acc + [a] END),
    e.mentions       = reduce(acc = coalesce(e.mentions, []), m IN row.mentions |
                             CASE WHEN m IN acc THEN acc ELSE acc + [m] END),
    e.mention_count  = size(coalesce(e.mentions, [])),
    e.model_version  = row.model_version,
    e.last_updated   = date()
RETURN count(e) AS upserted
"""

UPSERT_EDGE_TEMPLATE = """
UNWIND $rows AS row
MATCH (a:Entity {{entity_id: row.from_entity_id}})
MATCH (b:Entity {{entity_id: row.to_entity_id}})
MERGE (a)-[r:{rel_type} {{edge_id: row.edge_id}}]->(b)
  ON CREATE SET r.edge_id = row.edge_id, r.created_at = datetime()
SET r.type            = row.type,
    r.as_of           = row.as_of,
    r.confidence      = CASE WHEN row.confidence > coalesce(r.confidence, 0.0)
                             THEN row.confidence ELSE r.confidence END,
    r.source_chunk_id = row.source_chunk_id,
    r.src_document_id = row.src_document_id,
    r.model_version   = row.model_version,
    r.extracted_at    = datetime()
RETURN count(r) AS upserted
"""

BATCH_SIZE = 500


@dataclass
class WriteStats:
    nodes_upserted: int = 0
    edges_upserted: int = 0
    batches: int = 0

    def as_dict(self) -> dict:
        return {
            "nodes_upserted": self.nodes_upserted,
            "edges_upserted": self.edges_upserted,
            "batches": self.batches,
        }


@dataclass
class GraphStore:
    """Thin write/query layer over Neo4j. All Cypher lives here as named constants."""

    driver: object
    stats: WriteStats = field(default_factory=WriteStats)

    @classmethod
    def connect(cls, uri: str, auth: tuple[str, str]) -> GraphStore:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
        return cls(driver=driver)

    def close(self) -> None:
        """Close the driver. A driver that is already closed, or was never connected, is not an
        error worth propagating — closing is cleanup and cleanup must not mask a real failure."""
        driver = self.driver
        if driver is None:
            return
        try:
            driver.close()
        except (AttributeError, RuntimeError, OSError):
            pass

    # -- writes -------------------------------------------------------------------------------

    def upsert_entities(
        self, entities: dict[str, ResolvedEntity], model_version: str = "unknown"
    ) -> int:
        """Write nodes, grouped by entity type so each batch targets one concrete label."""
        by_type: dict[str, list[dict]] = {}
        for entity in entities.values():
            surfaces = [s for s in (entity.name, *entity.aliases) if s]
            by_type.setdefault(entity.type.value, []).append(
                {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.type.value,
                    "name": entity.name,
                    "name_norm": entity.name_norm or normalize_name(entity.name),
                    "cik": entity.cik,
                    "filer_cik": entity.filer_cik,
                    "confidence": entity.confidence,
                    # Every surface form, and every chunk that mentioned this entity. The previous
                    # version kept only the first mention, so the chunk -> entity back-link used
                    # to build pgvector provenance silently dropped all but one chunk per entity.
                    "surfaces": surfaces,
                    "mentions": [m for m in (entity.mentions or []) if m],
                    "model_version": model_version,
                }
            )

        written = 0
        for entity_type, rows in by_type.items():
            # Safe interpolation: `entity_type` is an EntityType enum value from this module, not
            # anything the model produced.
            statement = _UPSERT_NODE_TEMPLATE.format(label=entity_type)
            for batch in _batched(rows, BATCH_SIZE):
                with self._session() as session:
                    written += self._run(session, statement, {"rows": batch})
                self.stats.batches += 1
        self.stats.nodes_upserted += written
        return written

    def upsert_relationships(
        self,
        edges: list[ResolvedRelationship],
        document_id_of_chunk: dict[str, str] | None = None,
        model_version: str = "unknown",
    ) -> int:
        # Group by relationship type so each query targets one concrete edge type. A single
        # generic `:REL` label cannot be matched as `:SUPPLIES` in a template, so writing one
        # generic label would make every typed template in cypher_templates.py return zero rows.
        by_type: dict[str, list[dict]] = {}
        for edge in edges:
            by_type.setdefault(edge.type.value, []).append(
                {
                    "edge_id": edge.edge_id,
                    "from_entity_id": edge.from_entity_id,
                    "to_entity_id": edge.to_entity_id,
                    "type": edge.type.value,
                    "as_of": _as_date(edge.as_of),
                    "confidence": edge.confidence,
                    "source_chunk_id": edge.source_chunk_id or "",
                    "src_document_id": (document_id_of_chunk or {}).get(
                        edge.source_chunk_id or "", ""
                    ),
                    "model_version": model_version,
                }
            )

        written = 0
        for rel_type, type_rows in by_type.items():
            # Safe interpolation: `rel_type` is a RelationshipType enum value from this module,
            # never model output. Same reasoning as the node label above.
            statement = UPSERT_EDGE_TEMPLATE.format(rel_type=rel_type)
            for batch in _batched(type_rows, BATCH_SIZE):
                with self._session() as session:
                    written += self._run(session, statement, {"rows": batch})
                self.stats.batches += 1
        self.stats.edges_upserted += written
        return written

    # -- read-back ----------------------------------------------------------------------------

    def counts(self) -> dict:
        with self._session() as session:
            nodes = self._run(
                session, "MATCH (e:Entity) RETURN count(e) AS n", scalar=True
            )
            edges = self._run(
                session, "MATCH ()-[r]->() RETURN count(r) AS n", scalar=True
            )
            orphans = self._run(
                session,
                "MATCH (e:Entity) WHERE size([(e)--() | 1]) = 0 RETURN count(e) AS n",
                scalar=True,
            )
        return {"nodes": nodes or 0, "edges": edges or 0, "orphan_nodes": orphans or 0}

    def degree_distribution(self) -> dict[str, int]:
        """Nodes per degree bucket. A flat distribution means the graph is not connected, which is
        the failure that silently makes every multi-hop query return nothing."""
        with self._session() as session:
            records = session.run(
                "MATCH (e:Entity)-[r]-(m:Entity) "
                "RETURN e.entity_type AS t, count(r) AS c ORDER BY c DESC"
            ).data()
        distribution: dict[str, int] = {}
        for record in records:
            distribution[record["t"]] = record["c"]
        return distribution

    def relationship_type_counts(self) -> dict[str, int]:
        # Untyped match plus the `type` property. Reading `type(r)` would work too, but the
        # property is what the ontology documents, so it stays the single source of truth.
        with self._session() as session:
            records = session.run(
                "MATCH ()-[r]->() RETURN r.type AS t, count(r) AS c ORDER BY c DESC"
            ).data()
        return {record["t"]: record["c"] for record in records}

    def run_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a read query. Named Cypher only — never a string built from model output."""
        with self._session() as session:
            return [dict(r) for r in session.run(cypher, params or {}).data()]

    # -- internals ----------------------------------------------------------------------------

    def _session(self):
        return self.driver.session()  # type: ignore[attr-defined]

    @staticmethod
    def _run(session, statement: str, params: dict | None = None, scalar: bool = False):
        """Execute and return the first column of the single result row.

        The upsert statements end in `RETURN count(...) AS upserted`, so this returns how many
        rows were processed. It used to return `counters.properties_set`, which counts every
        property on every node — so a 3-node write reported 39 "nodes upserted" and an ingest
        report claimed 634 nodes for a graph that held 8. The number looked like data quality.
        """
        record = session.run(statement, params or {}).single()
        if record is None:
            return 0
        return record[0]


def _batched(rows: list, size: int):
    for index in range(0, len(rows), size):
        yield rows[index:index + size]


def _as_date(value: str | None) -> str | None:
    """Normalize a date string for Neo4j, or drop it rather than store a broken value.

    A malformed `as_of` silently breaks every time-filtered query, which is most of them, so an
    unparseable value is discarded instead of written.
    """
    if not value:
        return None
    text = str(value)[:10]
    try:
        date.fromisoformat(text)
    except ValueError:
        return None
    return text


def node_label(entity_type: EntityType) -> str:
    return entity_type.value


def relationship_type_value(rel: RelationshipType) -> str:
    return rel.value
