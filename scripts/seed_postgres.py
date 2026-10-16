"""Create the pgvector extension and the `chunks` table from docs/02-data-model.md.

Run after `docker compose up -d postgres`.

    python scripts/seed_postgres.py

The `chunks.chunk_id` primary key is the join key into Neo4j's `source_chunk_id`. If the two
disagree, hybrid retrieval silently degrades to two disconnected systems, so this script prints
the column types it created rather than just "ok".
"""

from __future__ import annotations

import os

import psycopg

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    accession     TEXT,
    section_path  TEXT NOT NULL,
    text          TEXT NOT NULL,
    token_count   INTEGER NOT NULL,
    entity_ids    TEXT[] NOT NULL DEFAULT '{}',
    filed_at      DATE,
    metadata      JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS chunks_entity_ids_idx ON chunks USING GIN (entity_ids);

-- Embedding column added separately: the vector dimension is a deployment decision (the
-- extraction and indexing phases own it), and pinning it here would force a migration the
-- moment the model changes.
"""

INDEX_DDL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
"""


def main() -> int:
    dsn = os.environ.get(
        "POSTGRES_DSN", "postgresql://kg_rag:kg_rag_dev@localhost:5433/kg_rag"
    )
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(DDL)
        # The embedding column is absent until Phase 2 sets a dimension, so the HNSW index is
        # applied only when the column exists. Reported rather than assumed.
        has_embedding = conn.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'embedding'"
        ).fetchone()[0]
        if has_embedding:
            conn.execute(INDEX_DDL)
            print("  ok  HNSW index on chunks.embedding")
        else:
            print("  --  chunks.embedding absent; HNSW index deferred to Phase 2")

        columns = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'chunks' ORDER BY ordinal_position"
        ).fetchall()

    print(f"\nchunks table: {len(columns)} columns")
    for name, dtype in columns:
        print(f"  {name:<14} {dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
