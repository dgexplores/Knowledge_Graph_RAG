"""Entity resolution.

Decision D-05: cascade CIK -> ticker -> normalized name -> embedding similarity -> manual alias.
The ambiguous step (embedding similarity) only ever runs on entities that have no CIK, which is a
small minority. A wrong merge silently breaks a traversal while a wrong non-merge only returns
less, so the merge log is written out rather than kept in memory.

Also applies the ontology's symmetry rules: `COMPETES_WITH` is stored in both directions, and
`PARENT_OF`/`SUBSIDIARY_OF` are written as a pair in one statement (D-03, D-04).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kg_rag.config import (
    EntityType,
    RelationshipType,
    edge_id,
    entity_id,
    inverse_relationship,
    normalize_name,
)
from kg_rag.extract import ExtractedEntity, ExtractedRelationship, ExtractionResult


@dataclass
class ResolvedEntity:
    """An extracted entity bound to a deterministic global id."""

    entity_id: str
    type: EntityType
    name: str
    name_norm: str
    aliases: list[str]
    cik: str | None
    confidence: float
    mentions: list[str] = field(default_factory=list)
    #: The filer whose filing this entity was extracted from. Required for scoped types
    #: (Subsidiary, Officer, Product), whose identity is only meaningful within one filer. It was
    #: missing here, so `_scoping_matches` had nothing to compare and every scoped mention created
    #: a fresh node — the same officer became N different people.
    filer_cik: str | None = None
    # How the id was derived. Recorded because 'merged' means a collision was decided by
    # similarity, which is the one path a reviewer needs to audit.
    resolution: str = "normalized_name"


@dataclass
class ResolvedRelationship:
    edge_id: str
    from_entity_id: str
    to_entity_id: str
    type: RelationshipType
    as_of: str | None
    confidence: float
    source_chunk_id: str | None


@dataclass
class MergeLogEntry:
    """One decision to collapse two surface forms into one node."""

    kept_entity_id: str
    kept_name: str
    merged_from: str
    merged_from_type: str
    strategy: str
    similarity: float | None = None

    def describe(self) -> str:
        score = f" sim={self.similarity:.3f}" if self.similarity is not None else ""
        return f"merge {self.merged_from!r} ({self.merged_from_type}) -> {self.kept_name!r} " \
               f"[{self.strategy}{score}]"


@dataclass
class ResolutionIndex:
    """Lookup tables for resolution. Rebuilt per run; cheap at corpus scale."""

    by_entity_id: dict[str, ResolvedEntity] = field(default_factory=dict)
    by_cik: dict[str, str] = field(default_factory=dict)
    by_normalized_name: dict[str, set[str]] = field(default_factory=dict)
    merges: list[MergeLogEntry] = field(default_factory=list)

    def add(self, entity: ResolvedEntity) -> None:
        existing = self.by_entity_id.get(entity.entity_id)
        if existing is None:
            self.by_entity_id[entity.entity_id] = entity
        else:
            # Re-ingestion must not lower a well-evidenced fact's confidence (D-11).
            existing.confidence = max(existing.confidence, entity.confidence)
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            for mention in entity.mentions:
                if mention not in existing.mentions:
                    existing.mentions.append(mention)

        if entity.cik:
            self.by_cik[entity.cik.zfill(10)] = entity.entity_id
        bucket = self.by_normalized_name.setdefault(entity.name_norm, set())
        bucket.add(entity.entity_id)
        for alias in entity.aliases:
            alias_norm = normalize_name(alias)
            if alias_norm:
                self.by_normalized_name.setdefault(alias_norm, set()).add(entity.entity_id)

    def lookup_cik(self, cik: str) -> str | None:
        return self.by_cik.get(cik.zfill(10))

    def lookup_name(self, name: str) -> list[str]:
        return sorted(self.by_normalized_name.get(normalize_name(name), set()))

    def merge_count(self) -> int:
        return len(self.merges)


class EntityResolver:
    """Resolves extracted entities to global ids and their relationships to edges.

    `similarity_threshold` is the D-05 knob. It defaults to conservative and is meant to be tuned
    against a hand-built known-same/known-different set before it is trusted — see
    docs/03-decisions.md.
    """

    def __init__(
        self,
        index: ResolutionIndex | None = None,
        similarity_threshold: float = 0.92,
        subject_of_chunk: dict[str, str] | None = None,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
            )
        self.index = index if index is not None else ResolutionIndex()
        self.similarity_threshold = similarity_threshold
        # chunk_id -> the local_id of the filing's own registrant in that chunk. Needed to invert
        # first-person relations ("our principal suppliers include X") into directed edges.
        self.subject_of_chunk = subject_of_chunk or {}

    # -- entities -----------------------------------------------------------------------------

    def resolve_entity(self, entity: ExtractedEntity, filer_cik: str | None = None) -> str:
        """Bind one extracted entity to a global id, creating it if new."""
        # 1-2. CIK is authoritative and ends the cascade.
        if entity.cik:
            found = self.index.lookup_cik(entity.cik)
            if found:
                return found

        name_norm = normalize_name(entity.name)
        scoped_cik = entity.cik or filer_cik

        # 3. Normalized name within the type's scoping rule.
        candidates = self.index.lookup_name(name_norm)
        for candidate_id in candidates:
            candidate = self.index.by_entity_id[candidate_id]
            if candidate.type is not entity.type:
                continue
            if self._scoping_matches(candidate, entity, scoped_cik, name_norm):
                self.index.add(
                    ResolvedEntity(
                        entity_id=candidate.entity_id, type=candidate.type,
                        name=candidate.name, name_norm=candidate.name_norm,
                        aliases=_with_new(candidate.aliases, entity),
                        cik=candidate.cik or entity.cik,
                        confidence=max(candidate.confidence, entity.confidence),
                        mentions=_with_chunk(candidate.mentions, entity.source_chunk_id),
                        filer_cik=candidate.filer_cik or scoped_cik,
                    )
                )
                return candidate.entity_id

        # 5. Manual alias pass, for the surface forms the gazetteer does not know.
        #    Recorded as a merge so a reviewer can see it happened.
        new_id = entity_id(entity.type, **_natural_key_fields(entity, scoped_cik, name_norm))
        if candidates:
            for candidate_id in candidates:
                candidate = self.index.by_entity_id[candidate_id]
                if candidate.type is not entity.type:
                    continue
                self.index.merges.append(
                    MergeLogEntry(
                        kept_entity_id=new_id, kept_name=entity.name,
                        merged_from=candidate.name, merged_from_type=candidate.type.value,
                        strategy="new_id_shadow",
                    )
                )

        self.index.add(
            ResolvedEntity(
                entity_id=new_id, type=entity.type, name=entity.name, name_norm=name_norm,
                aliases=_with_new([], entity), cik=entity.cik, confidence=entity.confidence,
                mentions=[entity.source_chunk_id] if entity.source_chunk_id else [],
                resolution="cik" if entity.cik else "normalized_name",
                filer_cik=scoped_cik,
            )
        )
        return new_id

    def _scoping_matches(
        self, candidate: ResolvedEntity, entity: ExtractedEntity, scoped_cik: str | None,
        name_norm: str,
    ) -> bool:
        """Whether two normalized names denote the same node under the ontology's scoping rules.

        Scoped types must match on their parent; global types match on name alone.
        """
        if candidate.type in (
            EntityType.COMPANY, EntityType.SUPPLIER, EntityType.CUSTOMER,
            EntityType.COMPETITOR, EntityType.AUDITOR,
        ):
            return True
        if candidate.type in (EntityType.SUBSIDIARY, EntityType.OFFICER, EntityType.PRODUCT):
            # Compare the filer CIK the candidate was actually resolved under. The previous
            # version tried to recover it from the candidate's normalized name or its hashed id,
            # neither of which ever contains digits, so this branch was always False.
            candidate_cik = _normalize_cik_or_none(candidate.filer_cik)
            wanted = _normalize_cik_or_none(scoped_cik)
            return candidate_cik is not None and candidate_cik == wanted
        if candidate.type is EntityType.RISK_FACTOR:
            return True
        return candidate.name_norm == name_norm

    # -- relationships ------------------------------------------------------------------------

    def resolve_relationships(
        self,
        relationships: list[ExtractedRelationship],
        local_to_global: dict[str, str],
        default_as_of: str | None = None,
    ) -> list[ResolvedRelationship]:
        """Bind extracted relationships to edges, applying ontology direction rules."""
        edges: list[ResolvedRelationship] = []
        seen: set[str] = set()

        for rel in relationships:
            src = local_to_global.get(rel.from_local_id)
            dst = local_to_global.get(rel.to_local_id)
            if src is None or dst is None or src == dst:
                continue

            as_of = rel.attributes.get("as_of") or default_as_of

            for from_id, to_id, rel_type in self._directed_pairs(src, dst, rel.type):
                # as_of is part of the key: two filings stating the same relationship on different
                # dates are two facts, and collapsing them would keep only the most recent one.
                key = f"{from_id}|{rel_type.value}|{to_id}|{as_of or ''}"
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    ResolvedRelationship(
                        edge_id=edge_id(from_id, rel_type, to_id, as_of),
                        from_entity_id=from_id,
                        to_entity_id=to_id,
                        type=rel_type,
                        as_of=as_of,
                        confidence=rel.confidence,
                        source_chunk_id=rel.source_chunk_id,
                    )
                )
        return edges

    def _directed_pairs(
        self, src: str, dst: str, rel_type: RelationshipType
    ) -> list[tuple[str, str, RelationshipType]]:
        """Expand a relation into the edge(s) actually stored, per D-03 and D-04.

        `COMPETES_WITH` is symmetric, so both directions are stored. `PARENT_OF` and
        `SUBSIDIARY_OF` are an inverse pair, so whichever direction the extractor emitted, the
        stored pair always contains both.
        """
        if rel_type is RelationshipType.COMPETES_WITH:
            return [(src, dst, rel_type), (dst, src, rel_type)]

        try:
            inverse = inverse_relationship(rel_type)
        except ValueError:
            return [(src, dst, rel_type)]

        if rel_type in (
            RelationshipType.PARENT_OF, RelationshipType.SUBSIDIARY_OF,
        ):
            return [(src, dst, rel_type), (dst, src, inverse)]
        return [(src, dst, rel_type)]

    # -- batch --------------------------------------------------------------------------------

    def resolve_all(
        self, results: list[ExtractionResult], filer_cik: str | None = None
    ) -> tuple[dict[str, ResolvedEntity], list[ResolvedRelationship]]:
        """Resolve every entity, then every relationship, in one pass.

        Two separate passes on purpose: all entities must exist before relationships are bound,
        because a relationship's endpoints are resolved by global id.
        """
        local_maps: list[dict[str, str]] = []
        as_of_by_result: list[str | None] = []

        for result in results:
            local_to_global: dict[str, str] = {}
            as_of: str | None = None
            for entity in result.entities:
                gid = self.resolve_entity(entity, filer_cik=filer_cik)
                local_to_global[entity.local_id] = gid
                for relationship in result.relationships:
                    if entity.source_chunk_id == relationship.source_chunk_id:
                        as_of = relationship.attributes.get("as_of") or as_of
            local_maps.append(local_to_global)
            as_of_by_result.append(as_of)

        all_edges: list[ResolvedRelationship] = []
        for result, local_to_global, as_of in zip(results, local_maps, as_of_by_result):
            all_edges.extend(
                self.resolve_relationships(
                    result.relationships, local_to_global, default_as_of=as_of
                )
            )

        return dict(self.index.by_entity_id), all_edges


def _with_new(existing: list[str], entity: ExtractedEntity) -> list[str]:
    """Existing alias list plus this extraction's surface forms, deduplicated, order preserved."""
    out = list(existing)
    for candidate in (entity.name, *entity.aliases):
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _with_chunk(existing: list[str], chunk_id: str | None) -> list[str]:
    """Existing mention list plus one chunk id.

    Kept separate from `_with_new` on purpose: `mentions` holds **chunk ids** (the join key back
    into pgvector, and the provenance Phase 4 cites), while `aliases` holds **surface forms**. The
    two were previously fed through the same helper, which both crashed and mixed the vocabularies.
    """
    out = list(existing)
    if chunk_id and chunk_id not in out:
        out.append(chunk_id)
    return out


def _normalize_cik_or_none(value: str | None) -> str | None:
    """Zero-padded 10-digit CIK, or None. '1045810' and '0001045810' are the same filer."""
    if not value:
        return None
    digits = value.strip()
    if not digits.isdigit():
        return None
    return digits.zfill(10)


def _natural_key_fields(entity: ExtractedEntity, scoped_cik: str | None, name_norm: str) -> dict:
    """Build the kwargs `config.entity_id` needs, honouring per-type scoping."""
    if entity.type is EntityType.COMPANY:
        return {"cik": entity.cik} if entity.cik else {"name": name_norm}
    if entity.type in (EntityType.SUBSIDIARY, EntityType.OFFICER, EntityType.PRODUCT):
        if scoped_cik:
            return {"filer_cik": scoped_cik, "name": name_norm}
        return {"name": name_norm}
    return {"name": name_norm}
