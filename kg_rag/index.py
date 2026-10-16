"""Retrieval index over pgvector.

Phase 2. Two responsibilities kept separate on purpose:

- `VectorIndex.upsert_chunks` — write passages with metadata, including `entity_ids`, which is the
  hop from a retrieved passage into the graph neighbourhood.
- `VectorIndex.search` — cosine similarity search with the `chunk_id`s that come back. Those ids
  are the join key, and the only thing citation validation checks.

The embedding backend is pluggable. The default is a deterministic hashing embedder so the system
runs with no model download; it is clearly weaker than a real sentence encoder and every report
says which one produced the numbers.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

from kg_rag.chunking import Chunk

DEFAULT_DIMENSION = 384


class IndexError_(RuntimeError):
    """Index write or search failed."""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class HashingEmbedder:
    """Deterministic bag-of-words embedder with no model dependency.

    A hashing trick over token hashes, L2-normalized. Two properties matter here: it needs no
    download, and the same text always produces the same vector, so a benchmark is reproducible.

    It is genuinely weaker than a learned encoder on paraphrases — which is the exact failure this
    project exists to measure — so it is a placeholder for the real embedder, never a substitute.
    """

    name = "stub:hashing"

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    score: float
    section_path: str = ""
    document_id: str = ""

    def as_passage(self):
        from kg_rag.merge import RetrievedPassage

        return RetrievedPassage(
            chunk_id=self.chunk_id,
            text=self.text,
            section_path=self.section_path,
            document_id=self.document_id,
            score=self.score,
        )


@dataclass
class VectorIndex:
    """pgvector-backed passage index."""

    dsn: str
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)
    _conn: object | None = field(default=None, init=False)

    def connection(self):
        if self._conn is None:
            import psycopg

            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except (AttributeError, RuntimeError, OSError):
                pass
            self._conn = None

    # -- schema --------------------------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Add the embedding column and its HNSW index at the configured dimension.

        Deferred from `seed_postgres.py` deliberately: the dimension belongs to the embedding
        model, and pinning it in the seed script would force a migration the moment the model
        changes.
        """
        conn = self.connection()
        dim = self.embedder.dimension
        conn.execute(
            f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector({dim})"
        )
        has_index = conn.execute(
            "SELECT count(*) FROM pg_indexes WHERE tablename = 'chunks' "
            "AND indexname = 'chunks_embedding_idx'"
        ).fetchone()[0]
        if not has_index:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks "
                "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
            )

    # -- writes --------------------------------------------------------------------------------

    def upsert_chunks(
        self, chunks: list[Chunk], entity_ids_of_chunk: dict[str, list[str]] | None = None
    ) -> int:
        """Write chunks and their vectors. Re-running is idempotent on `chunk_id`."""
        if not chunks:
            return 0
        self.ensure_schema()
        conn = self.connection()

        rows = []
        for chunk in chunks:
            vector = self.embedder.embed(chunk.text)
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.accession,
                    chunk.section_path,
                    chunk.text,
                    chunk.token_count,
                    list((entity_ids_of_chunk or {}).get(chunk.chunk_id, [])),
                    chunk.filed_at,
                    vector,
                )
            )

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, document_id, accession, section_path, text,
                     token_count, entity_ids, filed_at, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    entity_ids = EXCLUDED.entity_ids,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
        return len(rows)

    # -- reads ---------------------------------------------------------------------------------

    def search(self, query: str, k: int = 5, ef_search: int = 100) -> list[SearchHit]:
        """Cosine similarity search. Returns chunk ids, which are the join key."""
        if not query.strip():
            return []
        self.ensure_schema()
        conn = self.connection()

        # ef_search is a per-query HNSW setting, not a table setting.
        # It must go through set_config(): `SET LOCAL hnsw.ef_search = %s` is a syntax error,
        # because SET does not accept a bound parameter.
        conn.execute("BEGIN")
        try:
            conn.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(max(ef_search, k)),),
            )
            with conn.cursor() as cur:
                # `%s::vector` is required. psycopg adapts a Python list to double precision[],
                # and `<=>` has no such overload, so the operator lookup fails without the cast.
                cur.execute(
                    """
                    SELECT chunk_id, text, section_path, document_id,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (self.embedder.embed(query), self.embedder.embed(query), k),
                )
                rows = cur.fetchall()
        finally:
            conn.execute("COMMIT")

        return [
            SearchHit(
                chunk_id=r[0], text=r[1], section_path=r[2] or "", document_id=r[3] or "",
                score=float(r[4]),
            )
            for r in rows
        ]

    def chunk_ids(self) -> set[str]:
        conn = self.connection()
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM chunks")
            return {r[0] for r in cur.fetchall()}

    def count(self) -> int:
        conn = self.connection()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            return int(cur.fetchone()[0])

    def entity_ids_for(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        """chunk_id -> entity ids mentioned. The passage -> graph hop."""
        if not chunk_ids:
            return {}
        conn = self.connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, entity_ids FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)
            )
            return {r[0]: list(r[1] or []) for r in cur.fetchall()}


def build_index() -> VectorIndex:
    from kg_rag import settings

    return VectorIndex(
        dsn=settings.postgres_dsn(),
        embedder=HashingEmbedder(dimension=settings.embedding_dimension()),
    )
