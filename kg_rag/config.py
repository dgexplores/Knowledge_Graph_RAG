"""Ontology and schema. Single source of truth for entity/relationship vocabulary.

The extraction prompt is *generated* from this module (decision D-06), so the prompt cannot
drift from the schema it is meant to satisfy.

Ontology design rationale: docs/01-ontology.md
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum

# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------


class EntityType(StrEnum):
    COMPANY = "Company"
    SUBSIDIARY = "Subsidiary"
    OFFICER = "Officer"
    AUDITOR = "Auditor"
    SUPPLIER = "Supplier"
    CUSTOMER = "Customer"
    COMPETITOR = "Competitor"
    RISK_FACTOR = "RiskFactor"
    PRODUCT = "Product"
    FILING = "Filing"


class RelationshipType(StrEnum):
    FILES = "FILES"
    PARENT_OF = "PARENT_OF"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    OFFICER_OF = "OFFICER_OF"
    AUDITED_BY = "AUDITED_BY"
    SUPPLIES = "SUPPLIES"
    CUSTOMER_OF = "CUSTOMER_OF"
    COMPETES_WITH = "COMPETES_WITH"
    MENTIONS_RISK = "MENTIONS_RISK"
    REPORTS_SEGMENT = "REPORTS_SEGMENT"


# (from_type, to_type) pairs each relationship is legal for. Enforced in validation, never
# promptable — a model cannot widen its own schema.
COMPATIBILITY: dict[RelationshipType, frozenset[tuple[EntityType, EntityType]]] = {
    RelationshipType.FILES: frozenset({(EntityType.COMPANY, EntityType.FILING)}),
    RelationshipType.PARENT_OF: frozenset({(EntityType.COMPANY, EntityType.SUBSIDIARY)}),
    RelationshipType.SUBSIDIARY_OF: frozenset({(EntityType.SUBSIDIARY, EntityType.COMPANY)}),
    RelationshipType.OFFICER_OF: frozenset({(EntityType.OFFICER, EntityType.COMPANY)}),
    RelationshipType.AUDITED_BY: frozenset({(EntityType.COMPANY, EntityType.AUDITOR)}),
    RelationshipType.SUPPLIES: frozenset({(EntityType.COMPANY, EntityType.COMPANY)}),
    RelationshipType.CUSTOMER_OF: frozenset({(EntityType.COMPANY, EntityType.COMPANY)}),
    RelationshipType.COMPETES_WITH: frozenset({(EntityType.COMPANY, EntityType.COMPANY)}),
    RelationshipType.MENTIONS_RISK: frozenset({(EntityType.FILING, EntityType.RISK_FACTOR)}),
    RelationshipType.REPORTS_SEGMENT: frozenset({(EntityType.FILING, EntityType.PRODUCT)}),
}

# Relationships written in both directions. One Cypher statement writes the pair, so the
# invariant cannot be violated by a partial write (decisions D-03, D-04).
SYMMETRIC_PAIRS: tuple[tuple[RelationshipType, RelationshipType], ...] = (
    (RelationshipType.PARENT_OF, RelationshipType.SUBSIDIARY_OF),
)


def inverse_relationship(rel: RelationshipType) -> RelationshipType:
    """Inverse of a relationship. Raises for relationships with no stored inverse."""
    for forward, backward in SYMMETRIC_PAIRS:
        if rel == forward:
            return backward
        if rel == backward:
            return forward
    raise ValueError(f"{rel} has no stored inverse")


def is_legal(rel: RelationshipType, from_type: EntityType, to_type: EntityType) -> bool:
    """Whether `rel` may connect these endpoint types. Enforced from code, not from the prompt."""
    return (from_type, to_type) in COMPATIBILITY.get(rel, frozenset())


# --------------------------------------------------------------------------------------
# Natural keys and deterministic ids
# --------------------------------------------------------------------------------------

# Trailing corporate suffixes stripped during name normalization. Anchored to the end of the
# name so "Apple Computer" is not normalized to "apple computer" -> "computer".
_SUFFIXES = (
    "incorporated", "inc", "corporation", "corp", "company", "co",
    "limited", "ltd", "llc", "l l c", "plc", "lp", "llp", "nv", "sa", "sas", "ag", "gmbh",
    "holdings", "holding", "group",
)
_SUFFIX_RE = re.compile(
    r"(?:[,\s.]+\s*(?:" + "|".join(re.escape(s) for s in _SUFFIXES) + r")\s*)+$",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Canonical normalization for entity resolution.

    Lowercase, drop punctuation, collapse whitespace, strip trailing corporate suffixes.
    `Acme Corp.` and `Acme Corporation` both normalize to `acme`.
    """
    if not name:
        return ""
    out = _NON_ALNUM.sub(" ", name.lower())
    out = _WS.sub(" ", out).strip()
    previous = None
    while previous != out:  # "Acme Holdings Inc." -> strip "inc", then "holdings"
        previous = out
        out = _SUFFIX_RE.sub("", out).strip()
    return out or _WS.sub(" ", _NON_ALNUM.sub(" ", name.lower())).strip()


def normalize_cik(cik: str | int) -> str:
    """Zero-pad a CIK to the SEC's 10 digits. `320193` and `0000320193` are the same company."""
    digits = re.sub(r"\D", "", str(cik))
    if not digits:
        raise ValueError(f"not a CIK: {cik!r}")
    return digits.zfill(10)


def normalize_accession(accession: str) -> str:
    """Strip dashes from an EDGAR accession number. `0000320193-24-000012` -> `000032019324000012`."""
    return re.sub(r"\D", "", accession)


def natural_key(
    entity_type: EntityType,
    *,
    name: str | None = None,
    cik: str | int | None = None,
    accession: str | None = None,
    filer_cik: str | int | None = None,
    heading: str | None = None,
) -> str:
    """The string that uniquely identifies a logical entity within its type.

    Identity differs by type on purpose (docs/01-ontology.md): a Company is identified by its
    CIK, a Subsidiary only in the context of its parent's filing, a RiskFactor only within one
    accession. Scoping is what stops unrelated filings from collapsing into one node.
    """
    if entity_type is EntityType.COMPANY:
        if cik:
            return f"cik:{normalize_cik(cik)}"
        if name:
            return f"name:{normalize_name(name)}"
    elif entity_type is EntityType.FILING:
        if accession:
            return f"acc:{normalize_accession(accession)}"
    elif entity_type in (EntityType.SUBSIDIARY, EntityType.OFFICER, EntityType.PRODUCT):
        if filer_cik and name:
            return f"{normalize_cik(filer_cik)}|{normalize_name(name)}"
    elif entity_type is EntityType.RISK_FACTOR:
        if accession and heading:
            return f"{normalize_accession(accession)}|{normalize_name(heading)}"
        if heading:
            return f"heading:{normalize_name(heading)}"

    if name:
        return f"name:{normalize_name(name)}"
    raise ValueError(f"cannot build a natural key for {entity_type} from the given fields")


def entity_id(entity_type: EntityType, **key_fields) -> str:
    """Deterministic entity id.

    Deterministic ids are what make `MERGE` idempotent: the same logical entity always produces
    the same id, so re-running ingestion merges instead of duplicating (docs/02-data-model.md).
    """
    key = natural_key(entity_type, **key_fields)
    digest = hashlib.sha256(f"{entity_type}|{key}".encode()).hexdigest()[:24]
    return f"ent_{digest}"


def edge_id(from_entity_id: str, rel: RelationshipType, to_entity_id: str) -> str:
    """Deterministic edge id. Deliberately excludes `as_of` — see docs/02-data-model.md."""
    digest = hashlib.sha256(f"{from_entity_id}|{rel}|{to_entity_id}".encode()).hexdigest()[:24]
    return f"rel_{digest}"


# --------------------------------------------------------------------------------------
# Prompt generation — generated from the schema so it cannot drift (decision D-06)
# --------------------------------------------------------------------------------------

_ENTITY_GUIDANCE: dict[EntityType, str] = {
    EntityType.COMPANY: "A registrant that files with the SEC. Include the CIK when the filing "
                        "states it.",
    EntityType.SUBSIDIARY: "A legal entity named as owned or controlled by a parent company.",
    EntityType.OFFICER: "A named director, executive officer, or principal accounting officer.",
    EntityType.AUDITOR: "An accounting firm that audits a registrant's financial statements.",
    EntityType.SUPPLIER: "An entity named as providing goods or services to a company.",
    EntityType.CUSTOMER: "An entity named as receiving goods or services from a company.",
    EntityType.COMPETITOR: "An entity the filing explicitly names as a competitor.",
    EntityType.RISK_FACTOR: "A named risk-factor heading. Use the heading, not a summary of it.",
    EntityType.PRODUCT: "A named product, service, or reportable business segment.",
    EntityType.FILING: "The filing document itself, identified by accession number.",
}

_RELATIONSHIP_GUIDANCE: dict[RelationshipType, str] = {
    RelationshipType.FILES: "The company filed this document.",
    RelationshipType.PARENT_OF: "The company owns or controls the subsidiary.",
    RelationshipType.SUBSIDIARY_OF: "The subsidiary is owned by the company.",
    RelationshipType.OFFICER_OF: "The person serves in the stated role at the company.",
    RelationshipType.AUDITED_BY: "The firm audits the company's financial statements.",
    RelationshipType.SUPPLIES: "The first company supplies the second.",
    RelationshipType.CUSTOMER_OF: "The first company buys from the second.",
    RelationshipType.COMPETES_WITH: "The two entities are named as competitors. Write both "
                                    "directions.",
    RelationshipType.MENTIONS_RISK: "The filing raises this risk factor.",
    RelationshipType.REPORTS_SEGMENT: "The filing reports results for this product or segment.",
}


def build_extraction_instructions() -> str:
    """The extraction instructions, generated from the schema above.

    Returned as a string rather than a template with hand-written type names, so a change to an
    enum or to the compatibility table automatically reaches the prompt.
    """
    entity_lines = "\n".join(
        f"  - {t.value}: {_ENTITY_GUIDANCE[t]}" for t in EntityType
    )
    rel_lines = "\n".join(
        f"  - {t.value} ({from_t.value} -> {to_t.value}): {guidance}"
        for t, pairs in COMPATIBILITY.items()
        for from_t, to_t in sorted(pairs, key=lambda p: (p[0].value, p[1].value))
        for guidance in [_RELATIONSHIP_GUIDANCE[t]]
    )
    return f"""Extract entities and relationships from the supplied text chunks.

Entity types — use only these, exactly as spelled:
{entity_lines}

Relationship types, with the endpoint types each one is legal for:
{rel_lines}

Rules:
1. Emit only relationships whose endpoints are both in the emitted entity list.
2. Use local_id values to link relationships to entities. Never invent a global id.
3. source_chunk_id must be one of the supplied chunk ids. Never invent one.
4. Every relationship carries an `as_of` date taken from the filing, not today's date.
5. Write COMPETES_WITH in both directions.
6. If a relationship's endpoints are not in the legal list above, omit the relationship.
7. Prefer omission over speculation. An entity you are not confident exists should not be emitted.

Return JSON matching this schema and nothing else:
{{
  "entities": [
    {{"local_id": "e1", "type": "<EntityType>", "name": "<canonical name>",
      "aliases": [], "cik": null, "confidence": 0.0,
      "source_chunk_id": "<supplied chunk id>"}}
  ],
  "relationships": [
    {{"from_local_id": "e1", "to_local_id": "e2", "type": "<RelationshipType>",
      "attributes": {{"as_of": "YYYY-MM-DD"}}, "confidence": 0.0,
      "source_chunk_id": "<supplied chunk id>"}}
  ]
}}"""
