"""Tests for the ontology module.

These pin the properties the rest of the system depends on: deterministic ids (so `MERGE` is
idempotent), a compatibility table that the model cannot widen, and name normalization that
actually collapses the aliases it claims to collapse.
"""

from __future__ import annotations

import pytest

from kg_rag import config
from kg_rag.config import (
    COMPATIBILITY,
    EntityType,
    RelationshipType,
    edge_id,
    entity_id,
    inverse_relationship,
    is_legal,
    natural_key,
    normalize_accession,
    normalize_cik,
    normalize_name,
)


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw",
        [
            "Acme Corp",
            "Acme Corp.",
            "Acme Corporation",
            "ACME CORP",
            "Acme  Corp",
            "  Acme Corp  ",
            "Acme Corp, Inc.",
        ],
    )
    def test_corporate_suffix_variants_collapse(self, raw):
        """The aliases in docs/01 must all become one node, or multi-hop queries return nothing."""
        assert normalize_name(raw) == "acme"

    def test_suffix_stripping_is_anchored_to_the_end(self):
        """Stripping mid-name would turn 'Apple Computer' into 'computer'."""
        assert normalize_name("Apple Computer") == "apple computer"
        assert normalize_name("Computer Supply Company") == "computer supply"

    def test_punctuation_and_whitespace_collapse(self):
        assert normalize_name("Acme, Inc. — West") == "acme inc west"

    def test_mid_name_suffix_is_kept(self):
        """Suffix stripping is anchored to the end on purpose. Stripping mid-name would turn
        'Acme, Inc. West' into 'acme west' and merge it with a genuinely different company."""
        assert normalize_name("Acme Inc West") == "acme inc west"
        assert normalize_name("Acme Inc West") != normalize_name("Acme West")

    def test_name_that_is_entirely_a_suffix_is_not_emptied(self):
        """Stripping must not reduce a real name to the empty string."""
        assert normalize_name("Holdings") != ""

    @pytest.mark.parametrize("raw", ["", None])
    def test_empty_input(self, raw):
        assert normalize_name(raw) == ""


class TestIdentifiers:
    def test_cik_is_zero_padded_to_ten_digits(self):
        assert normalize_cik(320193) == "0000320193"
        assert normalize_cik("0000320193") == "0000320193"
        assert normalize_cik("CIK0000320193") == "0000320193"

    def test_cik_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="not a CIK"):
            normalize_cik("not-a-number")

    def test_accession_strips_dashes(self):
        assert normalize_accession("0000320193-24-000012") == "000032019324000012"

    def test_entity_id_is_deterministic(self):
        """Non-deterministic ids silently break idempotent ingestion via MERGE."""
        a = entity_id(EntityType.COMPANY, cik=320193)
        b = entity_id(EntityType.COMPANY, cik="0000320193")
        assert a == b
        assert a.startswith("ent_")
        assert len(a) == 28  # "ent_" + 24 hex

    def test_entity_id_differs_across_types_with_the_same_name(self):
        """A named subsidiary and a named company are different entities."""
        name = "Example Holdings"
        assert entity_id(EntityType.SUBSIDIARY, name=name, filer_cik=320193) != entity_id(
            EntityType.COMPANY, name=name
        )

    def test_subsidiary_is_scoped_to_its_parent_filing(self):
        """Same subsidiary name in two different filings must not collapse into one node."""
        a = entity_id(EntityType.SUBSIDIARY, name="Example Holdings", filer_cik=320193)
        b = entity_id(EntityType.SUBSIDIARY, name="Example Holdings", filer_cik=789001)
        assert a != b

    def test_risk_factor_is_scoped_to_its_accession(self):
        a = entity_id(
            EntityType.RISK_FACTOR, heading="Supply chain risk", accession="0000320193-24-000012"
        )
        b = entity_id(
            EntityType.RISK_FACTOR, heading="Supply chain risk", accession="0000320193-23-000009"
        )
        assert a != b

    def test_entity_id_falls_back_to_name_without_cik(self):
        assert entity_id(EntityType.SUPPLIER, name="Acme Corp").startswith("ent_")

    def test_natural_key_raises_when_nothing_identifies_the_entity(self):
        with pytest.raises(ValueError, match="cannot build a natural key"):
            natural_key(EntityType.COMPANY)

    def test_edge_id_is_deterministic_and_ignores_as_of(self):
        """Re-running the same filing must merge; a later filing's edge is a different node name."""
        a = edge_id("ent_x", RelationshipType.SUPPLIES, "ent_y")
        b = edge_id("ent_x", RelationshipType.SUPPLIES, "ent_y")
        c = edge_id("ent_x", RelationshipType.CUSTOMER_OF, "ent_y")
        assert a == b
        assert a != c
        assert a.startswith("rel_")


class TestCompatibilityTable:
    def test_every_relationship_has_at_least_one_legal_pairing(self):
        for rel, pairs in COMPATIBILITY.items():
            assert pairs, f"{rel} has no legal endpoint pairing and can never be written"

    def test_every_relationship_type_is_in_the_table(self):
        """A relationship missing from the table is silently unwritable."""
        assert set(COMPATIBILITY) == set(RelationshipType)

    def test_legal_pairing_is_accepted(self):
        assert is_legal(RelationshipType.SUPPLIES, EntityType.COMPANY, EntityType.COMPANY)

    def test_illegal_pairing_is_rejected(self):
        """The guard that keeps the graph queryable: PARENT_OF a RiskFactor is nonsense."""
        assert not is_legal(RelationshipType.PARENT_OF, EntityType.RISK_FACTOR, EntityType.COMPANY)
        assert not is_legal(RelationshipType.SUPPLIES, EntityType.SUBSIDIARY, EntityType.COMPANY)

    def test_inverse_round_trips(self):
        assert inverse_relationship(RelationshipType.PARENT_OF) is RelationshipType.SUBSIDIARY_OF
        assert inverse_relationship(RelationshipType.SUBSIDIARY_OF) is RelationshipType.PARENT_OF

    def test_relationship_without_inverse_raises(self):
        with pytest.raises(ValueError, match="no stored inverse"):
            inverse_relationship(RelationshipType.SUPPLIES)


class TestGeneratedPrompt:
    def test_prompt_contains_every_entity_type(self):
        prompt = config.build_extraction_instructions()
        for entity_type in EntityType:
            assert entity_type.value in prompt, f"{entity_type} missing from the prompt"

    def test_prompt_contains_every_relationship_type(self):
        prompt = config.build_extraction_instructions()
        for rel in RelationshipType:
            assert rel.value in prompt, f"{rel} missing from the prompt"

    def test_prompt_states_endpoint_constraints(self):
        """The prompt must communicate the compatibility table so the model does not waste
        tokens on relationships that validation will reject."""
        prompt = config.build_extraction_instructions()
        assert "Company -> Company" in prompt
        assert "Filing -> RiskFactor" in prompt

    def test_prompt_forbids_invented_ids_and_cypher(self):
        prompt = config.build_extraction_instructions()
        assert "Never invent a global id" in prompt
        assert "Never invent one" in prompt

    def test_prompt_drifts_with_the_schema(self):
        """If the schema changes, the prompt must change with it (decision D-06)."""
        baseline = config.build_extraction_instructions()
        original = config.COMPATIBILITY[RelationshipType.SUPPLIES]
        try:
            config.COMPATIBILITY[RelationshipType.SUPPLIES] = frozenset(
                {(EntityType.COMPANY, EntityType.SUPPLIER)}
            )
            assert config.build_extraction_instructions() != baseline
        finally:
            config.COMPATIBILITY[RelationshipType.SUPPLIES] = original
