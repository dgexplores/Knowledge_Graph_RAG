"""Entity and relationship extraction.

Two backends behind one interface:

- `ClaudeExtractor` — real structured extraction against the generated schema.
- `OfflineExtractor` — deterministic, gazetteer-driven, no API key.

Both return the same `ExtractionResult`, and both go through `validate_extraction`, which is
where the ontology is actually enforced. The prompt asks for compliance; this module is what
guarantees it. A model cannot widen the schema by wishing.

Every offline artifact is stamped `extractor=stub:offline` so a demo can never be mistaken for a
measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Protocol

from kg_rag.config import (
    COMPATIBILITY,
    EntityType,
    RelationshipType,
    build_extraction_instructions,
    normalize_name,
)

MAX_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.0

# Surface forms that signal a relationship, mapped to the relationship they imply. Ordered
# longest-first at match time so "supplies" is not beaten by a shorter overlapping pattern.
#
# The third element is the perspective: which endpoint of the stored edge is the *subject*
# (the filer). It is not cosmetic. English states the same fact from different sides — "TSMC is
# our principal supplier" and "we purchase from TSMC" mean the same edge — and the emitter
# previously hardcoded counterparty->subject for every pattern, which silently inverted
# CUSTOMER_OF, PARENT_OF and AUDITED_BY. resolve.py does not correct this: SUPPLIES and
# CUSTOMER_OF are inverses, and it stores only the direction it was handed.
#
#   FROM_COUNTERPARTY  X supplies us            X SUPPLIES subject
#   FROM_SUBJECT       we purchase from X      subject CUSTOMER_OF X
#   SYMMETRIC          we compete with X       either direction is true
FROM_COUNTERPARTY = "from_counterparty"
FROM_SUBJECT = "from_subject"
SYMMETRIC = "symmetric"

# A captured entity name. The tempered token forbids ". " inside the span, so a name can never
# run across a sentence boundary — without it, "…advanced logic dies. Micron Technology and SK
# Hynix" was captured as one name and the whole sentence was dropped.
_NAME = r"(?:(?!\.\s)[\w.&' -]){3,60}?"
_TAIL = r"(?=[,.;]|\s+(?:and|or|which|that)\b|$)"

_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], RelationshipType, str], ...] = (
    (re.compile(rf"\b(?:names?|identifies?)\s+(?P<other>{_NAME})\s+as\s+"
                r"(?:a\s+|its\s+|one\s+of\s+its\s+)?(?:principal\s+|primary\s+|key\s+|"
                r"major\s+)?competitors?\b", re.IGNORECASE), RelationshipType.COMPETES_WITH,
     SYMMETRIC),
    (re.compile(rf"\bcompetes?\s+with\s+(?P<other>{_NAME}){_TAIL}",
                re.IGNORECASE), RelationshipType.COMPETES_WITH, SYMMETRIC),
    # "our principal supplier is X" — subject-first phrasing.
    (re.compile(rf"\b(?:our\s+|the\s+)?(?:principal|primary|major|key|sole)\s+suppliers?\s+"
                rf"(?:is|are|include[s]?)\s+(?P<other>{_NAME}){_TAIL}",
                re.IGNORECASE), RelationshipType.SUPPLIES, FROM_COUNTERPARTY),
    # "X is our principal foundry supplier" — counterparty-first phrasing, which is how filings
    # actually read. The old table only had the subject-first form, so no supplier was ever found.
    (re.compile(rf"\b(?P<other>{_NAME})\s+is\s+(?:our|the|its)\s+"
                r"(?:\w+[\s-]+){0,2}suppliers?\b", re.IGNORECASE),
     RelationshipType.SUPPLIES, FROM_COUNTERPARTY),
    (re.compile(rf"\b(?P<other>{_NAME})\s+suppl(?:y|ies|ied)\s+"
                r"(?:us\b|our\b|the\s+high-|\w+\s+used\s+in\s+our)", re.IGNORECASE),
     RelationshipType.SUPPLIES, FROM_COUNTERPARTY),
    (re.compile(rf"\b(?:supplies|supplied|fabricates?\s+for)\s+(?P<other>{_NAME}){_TAIL}",
                re.IGNORECASE), RelationshipType.SUPPLIES, FROM_COUNTERPARTY),
    (re.compile(rf"\bpurchases?\s+from\s+(?P<other>{_NAME}){_TAIL}",
                re.IGNORECASE), RelationshipType.CUSTOMER_OF, FROM_SUBJECT),
    (re.compile(rf"\b(?:our\s+|the\s+)?subsidiar(?:y|ies)\s+(?:include[s]?|are|comprise[s]?)\s+"
                rf"(?P<other>{_NAME}){_TAIL}", re.IGNORECASE),
     RelationshipType.PARENT_OF, FROM_SUBJECT),
    (re.compile(r"\bwholly[-\s]owned\s+subsidiar(?:y|ies)\b[^.]{0,80}?"
                rf"\b(?P<other>{_NAME}){_TAIL}", re.IGNORECASE),
     RelationshipType.PARENT_OF, FROM_SUBJECT),
    (re.compile(rf"\baudited\s+by\s+(?P<other>{_NAME}){_TAIL}",
                re.IGNORECASE), RelationshipType.AUDITED_BY, FROM_SUBJECT),
)

# An "our"/"we" filing names the registrant itself. Kept explicit so the offline extractor can
# resolve a first-person pronoun to the right node instead of emitting a junk entity.
_SELF_REFERENCE = re.compile(r"\b(?:our|we|the\s+company)\b", re.IGNORECASE)


@dataclass
class ExtractedEntity:
    local_id: str
    type: EntityType
    name: str
    aliases: list[str] = field(default_factory=list)
    cik: str | None = None
    confidence: float = 0.8
    source_chunk_id: str | None = None


@dataclass
class ExtractedRelationship:
    from_local_id: str
    to_local_id: str
    type: RelationshipType
    attributes: dict = field(default_factory=dict)
    confidence: float = 0.8
    source_chunk_id: str | None = None


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    # Discards that validation removed. Recorded rather than dropped: an extraction pass that
    # silently rejects 40% of its output is a finding about the prompt, not a detail.
    rejected: list[dict] = field(default_factory=list)
    extractor: str = "unknown"

    @property
    def total_output(self) -> int:
        return len(self.entities) + len(self.relationships)

    def rejection_rate(self) -> float:
        if self.total_output == 0:
            return 0.0
        return len(self.rejected) / self.total_output

    def merge(self, other: ExtractionResult) -> ExtractionResult:
        """Combine two results. local_ids are namespaced so they cannot collide across chunks.

        Every chunk numbers its entities `e_subject`, `e1`, `e2`… from scratch, so two chunks of
        the same filing would otherwise overwrite each other's endpoints.

        The remap has to be applied to the *entities* as well as the relationships. It previously
        rewrote only the relationship endpoints, leaving the merged entities with their original
        ids — so after merging, every relationship pointed at an id no entity had, resolution
        silently dropped all of them, and the graph came out with zero edges.
        """
        offset = len(self.entities)
        remap = {
            e.local_id: f"x{offset + i}" for i, e in enumerate(other.entities)
        }

        # Copy rather than mutate: `other` may be a cached extraction reused on a later run.
        for entity in other.entities:
            self.entities.append(
                replace(entity, local_id=remap[entity.local_id])
            )

        for rel in other.relationships:
            src = remap.get(rel.from_local_id)
            dst = remap.get(rel.to_local_id)
            if src and dst:
                self.relationships.append(
                    ExtractedRelationship(
                        from_local_id=src,
                        to_local_id=dst,
                        type=rel.type,
                        attributes=rel.attributes,
                        confidence=rel.confidence,
                        source_chunk_id=rel.source_chunk_id,
                    )
                )
            else:
                # An endpoint that was never emitted as an entity is a real defect in the
                # extractor, so it is recorded rather than dropped in silence.
                self.rejected.append({
                    "kind": "relationship", "reason": "dangling_endpoint_after_merge",
                    "type": rel.type.value,
                    "value": f"{rel.from_local_id}->{rel.to_local_id}",
                })
        self.rejected.extend(other.rejected)
        return self


class ExtractionError(ValueError):
    """The extractor produced something that is not usable at all."""


def validate_extraction(
    result: ExtractionResult, valid_chunk_ids: set[str] | None = None
) -> ExtractionResult:
    """Enforce the ontology against an extraction result.

    Rules, all in code because the prompt cannot be trusted to hold them:
      1. every type is in the enum
      2. every relationship endpoint resolves to an entity in the same result
      3. every `source_chunk_id` was actually supplied for extraction
      4. confidence is within [0, 1]
      5. the relationship's endpoints match the compatibility table

    Returns a new result containing only what passed, with failures recorded in `rejected`.
    """
    kept_entities: list[ExtractedEntity] = []
    seen_local_ids: set[str] = set()

    for entity in result.entities:
        if not isinstance(entity.type, EntityType):
            result.rejected.append({"kind": "entity", "reason": "unknown_entity_type",
                                    "value": str(entity.type), "name": entity.name})
            continue
        if not entity.name or not entity.name.strip():
            result.rejected.append({"kind": "entity", "reason": "empty_name",
                                    "name": entity.name})
            continue
        if not MIN_CONFIDENCE <= entity.confidence <= MAX_CONFIDENCE:
            result.rejected.append({"kind": "entity", "reason": "confidence_out_of_range",
                                    "name": entity.name, "value": entity.confidence})
            continue
        if valid_chunk_ids is not None and entity.source_chunk_id is not None \
                and entity.source_chunk_id not in valid_chunk_ids:
            # An invented source id would let a citation resolve to nothing, which is the exact
            # failure Phase 4 exists to prevent — caught here at the earliest point.
            result.rejected.append({
                "kind": "entity", "reason": "unknown_source_chunk_id",
                "name": entity.name, "value": entity.source_chunk_id,
            })
            continue
        if entity.local_id in seen_local_ids:
            result.rejected.append({"kind": "entity", "reason": "duplicate_local_id",
                                    "value": entity.local_id, "name": entity.name})
            continue
        seen_local_ids.add(entity.local_id)
        kept_entities.append(entity)

    types_by_local_id = {e.local_id: e.type for e in kept_entities}

    kept_relationships: list[ExtractedRelationship] = []
    for rel in result.relationships:
        if not isinstance(rel.type, RelationshipType):
            result.rejected.append({"kind": "relationship", "reason": "unknown_relationship_type",
                                    "value": str(rel.type)})
            continue
        if rel.from_local_id not in types_by_local_id or rel.to_local_id not in types_by_local_id:
            result.rejected.append({"kind": "relationship", "reason": "dangling_endpoint",
                                    "value": f"{rel.from_local_id}->{rel.to_local_id}",
                                    "type": rel.type.value})
            continue
        if not MIN_CONFIDENCE <= rel.confidence <= MAX_CONFIDENCE:
            result.rejected.append({"kind": "relationship", "reason": "confidence_out_of_range",
                                    "value": rel.confidence, "type": rel.type.value})
            continue
        if valid_chunk_ids is not None and rel.source_chunk_id is not None \
                and rel.source_chunk_id not in valid_chunk_ids:
            result.rejected.append({"kind": "relationship",
                                    "reason": "unknown_source_chunk_id",
                                    "value": rel.source_chunk_id,
                                    "type": rel.type.value})
            continue
        from_type = types_by_local_id[rel.from_local_id]
        to_type = types_by_local_id[rel.to_local_id]
        if (from_type, to_type) not in COMPATIBILITY.get(rel.type, frozenset()):
            # The rule that keeps the graph queryable: PARENT_OF a RiskFactor is nonsense, and an
            # edge that no query can use is a cost with no return.
            result.rejected.append({
                "kind": "relationship", "reason": "incompatible_endpoints",
                "type": rel.type.value,
                "value": f"{from_type.value}->{to_type.value}",
            })
            continue
        kept_relationships.append(rel)

    return ExtractionResult(
        entities=kept_entities,
        relationships=kept_relationships,
        rejected=result.rejected,
        extractor=result.extractor,
    )


class Extractor(Protocol):
    def extract(self, text: str, chunk_id: str, subject: str | None = None) -> ExtractionResult: ...


# --------------------------------------------------------------------------------------
# Offline backend
# --------------------------------------------------------------------------------------

class Gazetteer:
    """Known surface forms -> (entity type, canonical name, cik).

    Built from the resolved registrants plus whatever subsidiary, officer and auditor names the
    corpus mentions. Deterministic by construction, which is what makes the offline path testable.
    """

    def __init__(self) -> None:
        self._by_surface: dict[str, tuple[EntityType, str, str | None]] = {}
        self._by_normalized: dict[str, tuple[EntityType, str, str | None]] = {}

    def add(
        self,
        surface: str,
        entity_type: EntityType,
        canonical: str,
        cik: str | None = None,
        aliases: tuple[str, ...] = (),
    ) -> None:
        for form in (canonical, surface, *aliases):
            key = re.sub(r"\s+", " ", form.lower()).strip()
            if key and key not in self._by_surface:
                self._by_surface[key] = (entity_type, canonical, cik)
            # Second index keyed on the suffix-stripped name. Without it, a filing that writes
            # "Taiwan Semiconductor Manufacturing Company" could never resolve to the registered
            # "Taiwan Semiconductor Manufacturing Company Limited", so every such mention was
            # silently skipped and the graph came out with zero supplier edges.
            norm = normalize_name(form) if form else ""
            if norm and norm not in self._by_normalized:
                self._by_normalized[norm] = (entity_type, canonical, cik)

    def lookup(self, surface: str) -> tuple[EntityType, str, str | None] | None:
        """Exact surface first, then suffix-stripped. Never a substring match: 'Broadcom' must
        not resolve a mention of 'Broadcom Fiber'."""
        key = re.sub(r"\s+", " ", surface.lower()).strip()
        hit = self._by_surface.get(key)
        if hit is not None:
            return hit
        norm = normalize_name(surface) if surface else ""
        return self._by_normalized.get(norm) if norm else None

    def known_surfaces(self) -> list[str]:
        return sorted(self._by_surface, key=len, reverse=True)

    def __len__(self) -> int:
        return len(self._by_surface)

    def scan(self, text: str) -> list[tuple[str, str, tuple[EntityType, str, str | None]]]:
        """Find every known surface form in `text`, longest-first so 'Taiwan Semiconductor' wins
        over 'Semiconductor'. Overlapping matches are dropped."""
        found: list[tuple[str, str, tuple[EntityType, str, str | None]]] = []
        claimed: list[tuple[int, int]] = []
        lowered = text.lower()
        for surface in self.known_surfaces():
            entity_type, canonical, cik = self._by_surface[surface]
            for match in re.finditer(re.escape(surface), lowered):
                span = match.span()
                if any(span[0] < e and s < span[1] for s, e in claimed):
                    continue
                claimed.append(span)
                found.append((surface, match.group(0), (entity_type, canonical, cik)))
        return found


class OfflineExtractor:
    """Deterministic extractor over a gazetteer. Not an LLM.

    Exists so the pipeline is demonstrable and testable without an API key. Every result is
    labelled `stub:offline` so a downstream report cannot be mistaken for a real measurement.
    """

    name = "stub:offline"

    #: How many trailing words to drop while backing off to a resolvable name. Enough for
    #: "Taiwan Semiconductor Manufacturing Company for wafers" without letting the loop wander.
    _MAX_BACKOFF_WORDS = 6

    def _resolve_surface(self, surface: str) -> tuple[EntityType, str, str | None] | None:
        """Longest prefix of `surface` the gazetteer knows.

        Regex captures overshoot — "We purchase from Taiwan Semiconductor Manufacturing Company
        for wafers" captures the whole clause. Enumerating boundary words never covers every
        case, so instead the trailing words are dropped until a known name is left. Backing off
        only from the right keeps the longest-match preference intact.
        """
        hit = self.gazetteer.lookup(surface)
        if hit is not None:
            return hit

        words = surface.split()
        for drop in range(1, min(self._MAX_BACKOFF_WORDS, len(words) - 1) + 1):
            candidate = " ".join(words[:-drop]).strip(" .,")
            if not candidate:
                break
            hit = self.gazetteer.lookup(candidate)
            if hit is not None:
                return hit
        return None

    def __init__(self, gazetteer: Gazetteer) -> None:
        self.gazetteer = gazetteer

    def extract(
        self, text: str, chunk_id: str, subject: str | None = None, filed_at: str | None = None
    ) -> ExtractionResult:
        result = ExtractionResult(extractor=self.name)
        if not text.strip():
            return result

        entities: dict[str, ExtractedEntity] = {}

        def upsert(local_id: str, entity_type: EntityType, name: str, cik: str | None,
                   confidence: float) -> str:
            if local_id in entities:
                existing = entities[local_id]
                existing.confidence = max(existing.confidence, confidence)
                return local_id
            entities[local_id] = ExtractedEntity(
                local_id=local_id, type=entity_type, name=name, cik=cik,
                confidence=confidence, source_chunk_id=chunk_id,
            )
            return local_id

        # The filing's own registrant. A first-person reference resolves to this node.
        subject_id: str | None = None
        if subject:
            hit = self.gazetteer.lookup(subject)
            entity_type, canonical, cik = hit or (EntityType.COMPANY, subject, None)
            subject_id = upsert("e_subject", entity_type, canonical, cik, 1.0)

        # Canonical name -> local id, so a counterparty found by a relation pattern reuses the
        # node the gazetteer scan already created. Without this, Broadcom scanned as `e4` was
        # created a second time as `r_0_4`, and the graph held two copies of every counterparty.
        id_of_name: dict[str, str] = {}

        for index, (_surface, matched, (entity_type, canonical, cik)) in enumerate(
            self.gazetteer.scan(text)
        ):
            local_id = upsert(f"e{index + 1}", entity_type, canonical, cik, 0.9)
            id_of_name[normalize_name(canonical)] = local_id

        if subject_id:
            for pattern, rel_type, perspective in _RELATION_PATTERNS:
                for match in pattern.finditer(text):
                    other_surface = (match.group("other") or "").strip(" .,")
                    if not other_surface or len(other_surface) < 3:
                        continue
                    hit = self._resolve_surface(other_surface)
                    if hit is None:
                        continue
                    other_type, other_canonical, other_cik = hit

                    key = normalize_name(other_canonical)
                    if key in id_of_name:
                        other_id = id_of_name[key]
                    else:
                        other_id = upsert(
                            f"r{len(entities) + 1}", other_type, other_canonical, other_cik, 0.85
                        )
                        id_of_name[key] = other_id

                    # Compare entity ids. This previously compared a CIK against the literal
                    # "e_subject", which is never equal, so the self-relation guard never fired.
                    if other_id == subject_id:
                        continue

                    # Direction comes from the pattern's perspective, not from a fixed convention.
                    if perspective == FROM_SUBJECT:
                        from_id, to_id = subject_id, other_id
                    else:
                        # FROM_COUNTERPARTY and SYMMETRIC both emit counterparty -> subject;
                        # resolve.py expands SYMMETRIC into both stored directions.
                        from_id, to_id = other_id, subject_id

                    attributes = {"as_of": filed_at} if filed_at else {}
                    result.relationships.append(
                        ExtractedRelationship(
                            from_local_id=from_id,
                            to_local_id=to_id,
                            type=rel_type,
                            attributes=attributes,
                            confidence=0.8,
                            source_chunk_id=chunk_id,
                        )
                    )

        # Assigned last. Previously it ran before the relation patterns, so any entity created
        # there was invisible to validate_extraction and every such edge was rejected as a
        # dangling endpoint.
        result.entities = list(entities.values())

        return validate_extraction(result, valid_chunk_ids={chunk_id})


# --------------------------------------------------------------------------------------
# Claude backend
# --------------------------------------------------------------------------------------

class ClaudeExtractor:
    """Real structured extraction.

    Uses tool-use for schema enforcement rather than asking for JSON in prose — the model cannot
    then return a shape the code has to guess at. `validate_extraction` still runs afterwards,
    because a schema-conformant response can still violate the ontology.
    """

    name = "claude"

    def __init__(self, model: str | None = None, client: object | None = None) -> None:
        self.model = model
        self._client = client

    @property
    def client(self) -> object:
        if self._client is None:
            from anthropic import Anthropic

            from kg_rag import settings

            self._client = Anthropic(api_key=settings.anthropic_api_key())
        return self._client

    @property
    def _model_id(self) -> str:
        return self.model or __import__("kg_rag.settings", fromlist=["x"]).extraction_model()

    def extract(
        self, text: str, chunk_id: str, subject: str | None = None,
        filed_at: str | None = None,
    ) -> ExtractionResult:
        result = ExtractionResult(extractor=self.name)
        if not text.strip():
            return result

        instructions = build_extraction_instructions()
        subject_note = f"\n\nThis filing was submitted by: {subject}" if subject else ""
        prompt = (
            f"{instructions}{subject_note}\n\n"
            f"Source chunk id: {chunk_id}\n"
            f"Filing date: {filed_at or 'unknown'}\n\n"
            f"--- TEXT ---\n{text}\n--- END TEXT ---"
        )

        tool = {
            "name": "record_extraction",
            "description": "Record the entities and relationships found in the text.",
            "input_schema": _extraction_tool_schema(),
        }

        try:
            response = self.client.messages.create(  # type: ignore[attr-defined]
                model=self._model_id,
                max_tokens=4096,
                system="You extract structured facts from SEC filings. You call "
                       "record_extraction exactly once with what you found. Prefer omission "
                       "over speculation.",
                tools=[tool],
                tool_choice={"type": "tool", "name": "record_extraction"},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # provider/transport errors are not extraction errors
            raise ExtractionError(f"model call failed for chunk {chunk_id}: {exc}") from exc

        payload = _first_tool_input(response)
        if payload is None:
            return result

        result.entities = [
            ExtractedEntity(
                local_id=str(item.get("local_id", f"e{i}")),
                type=_coerce_enum(EntityType, item.get("type")),
                name=str(item.get("name", "")),
                aliases=[str(a) for a in item.get("aliases", []) or []],
                cik=str(item["cik"]) if item.get("cik") else None,
                confidence=_coerce_float(item.get("confidence"), 0.7),
                source_chunk_id=str(item.get("source_chunk_id") or chunk_id),
            )
            for i, item in enumerate(payload.get("entities", []) or [])
        ]
        result.relationships = [
            ExtractedRelationship(
                from_local_id=str(item.get("from_local_id", "")),
                to_local_id=str(item.get("to_local_id", "")),
                type=_coerce_enum(RelationshipType, item.get("type")),
                attributes=item.get("attributes", {}) or {},
                confidence=_coerce_float(item.get("confidence"), 0.7),
                source_chunk_id=str(item.get("source_chunk_id") or chunk_id),
            )
            for item in payload.get("relationships", []) or []
        ]
        return validate_extraction(result, valid_chunk_ids={chunk_id})


def _extraction_tool_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "local_id": {"type": "string"},
                        "type": {"type": "string", "enum": [t.value for t in EntityType]},
                        "name": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "cik": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "source_chunk_id": {"type": "string"},
                    },
                    "required": ["local_id", "type", "name", "confidence", "source_chunk_id"],
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_local_id": {"type": "string"},
                        "to_local_id": {"type": "string"},
                        "type": {"type": "string", "enum": [r.value for r in RelationshipType]},
                        "attributes": {"type": "object"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "source_chunk_id": {"type": "string"},
                    },
                    "required": [
                        "from_local_id", "to_local_id", "type", "confidence", "source_chunk_id"
                    ],
                },
            },
        },
        "required": ["entities", "relationships"],
    }


def _first_tool_input(response: object) -> dict | None:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            return getattr(block, "input", None)
    return None


def _coerce_enum(enum_cls, value: object):
    """Unknown enum values stay unknown, so `validate_extraction` rejects them explicitly."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except (ValueError, TypeError):
        return value  # type: ignore[return-value]


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def build_extractor(gazetteer: Gazetteer) -> Extractor:
    """Pick a backend. `USE_REAL_MODELS` decides; see `settings.use_real_models`."""
    from kg_rag import settings

    if settings.use_real_models():
        return ClaudeExtractor()
    return OfflineExtractor(gazetteer)


def build_gazetteer_from_registrants(registrants: list) -> Gazetteer:
    """Seed the gazetteer from resolved registrants.

    Company aliases include the short form, because filings say "NVIDIA" as often as they say
    "NVIDIA Corporation" and a gazetteer that only knows the long form misses the short one.
    """
    from kg_rag.config import normalize_name

    gazetteer = Gazetteer()
    for registrant in registrants:
        canonical = registrant.name
        short = canonical.split(",")[0].strip()
        aliases = {short, short.replace(" Incorporated", ""), registrant.ticker}
        gazetteer.add(
            registrant.ticker, EntityType.COMPANY, canonical,
            cik=registrant.cik, aliases=tuple(a for a in aliases if a),
        )
        # Register the normalized form too, so a filing naming the entity in a different case
        # still resolves.
        gazetteer.add(normalize_name(canonical), EntityType.COMPANY, canonical, cik=registrant.cik)
    return gazetteer
