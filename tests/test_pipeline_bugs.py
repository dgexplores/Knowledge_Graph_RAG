"""Tests for bugs found by actually running the pipeline.

Each case here corresponds to a defect that a green test suite did not catch, because all of them
only misbehaved against real data or a live database:

- `ExtractionResult.merge` remapped relationship endpoints but not the entities, so after merging
  chunks of a filing every edge dangled and the graph was written with zero relationships;
- `suppliers_of` filtered on the supplier side instead of the customer, so "suppliers of NVIDIA"
  returned nothing even though TSMC -> NVIDIA existed;
- the router had no out-of-scope rule beyond an empty question, so "What is the weather in Paris?"
  was answered with sentences about lithography equipment;
- an empty generator answer passed the citation gate and reached the caller as a blank result;
- the aggregation template returned no chunk id, so aggregate answers were uncitable and dropped;
- the write counter returned `properties_set`, so a 3-node write reported 39 nodes.

These are pure-unit where possible; the ones that need a database are marked and run by the demo.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from kg_rag.config import EntityType, RelationshipType
from kg_rag.cypher_templates import TEMPLATES, get_template
from kg_rag.extract import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from kg_rag.merge import AssembledContext, assemble, validate_citations
from kg_rag.router import (
    KeywordRouter,
    build_corpus_vocabulary,
    content_terms,
)


def _entity(local_id: str, name: str) -> ExtractedEntity:
    return ExtractedEntity(
        local_id=local_id, type=EntityType.COMPANY, name=name, confidence=0.9,
        source_chunk_id="c1",
    )


def _chunk_result(chunk_id: str, subject: str, other: str) -> ExtractionResult:
    return ExtractionResult(
        entities=[_entity("e_subject", subject), _entity("e1", other)],
        relationships=[
            ExtractedRelationship(
                from_local_id="e1",
                to_local_id="e_subject",
                type=RelationshipType.SUPPLIES,
                attributes={"as_of": "2024-01-01"},
                confidence=0.8,
                source_chunk_id=chunk_id,
            )
        ],
        extractor="stub:offline",
    )


class TestMergeRemapsEntities:
    def test_relationship_endpoints_survive_a_merge(self):
        """The bug: remap was applied to relationships only, so after merging, `x2`/`x3` existed
        in no entity and every edge was dropped by resolution."""
        a = _chunk_result("c1", "NVIDIA Corporation", "TSMC")
        b = _chunk_result("c2", "NVIDIA Corporation", "Broadcom")

        merged = ExtractionResult(extractor="stub:offline")
        merged.merge(a)
        merged.merge(b)

        ids = {e.local_id for e in merged.entities}
        assert len(merged.relationships) == 2
        for rel in merged.relationships:
            assert rel.from_local_id in ids, "relationship endpoint is not a merged entity"
            assert rel.to_local_id in ids, "relationship endpoint is not a merged entity"

    def test_local_ids_are_namespaced_across_chunks(self):
        """Both chunks number their entities e_subject/e1, so an un-namespaced merge would have
        two different 'e1' entities."""
        merged = ExtractionResult(extractor="stub:offline")
        merged.merge(_chunk_result("c1", "NVIDIA Corporation", "TSMC"))
        merged.merge(_chunk_result("c2", "NVIDIA Corporation", "Broadcom"))
        ids = [e.local_id for e in merged.entities]
        assert len(ids) == len(set(ids)), f"local ids collided: {ids}"

    def test_merge_does_not_mutate_the_source_result(self):
        """The source may be a cached extraction reused on a later run."""
        source = _chunk_result("c1", "NVIDIA Corporation", "TSMC")
        before = [e.local_id for e in source.entities]
        merged = ExtractionResult(extractor="stub:offline")
        merged.merge(source)
        assert [e.local_id for e in source.entities] == before

    def test_dangling_endpoint_after_merge_is_recorded(self):
        result = _chunk_result("c1", "NVIDIA Corporation", "TSMC")
        result.relationships[0].to_local_id = "does_not_exist"
        merged = ExtractionResult(extractor="stub:offline")
        merged.merge(result)
        assert any(r["reason"] == "dangling_endpoint_after_merge" for r in merged.rejected)


class TestTemplateDirection:
    """SUPPLIES is stored supplier -> customer, so 'suppliers of X' filters on the customer."""

    def test_suppliers_of_filters_on_the_customer_side(self):
        cypher = get_template("suppliers_of").cypher
        match_line = next(
            line for line in cypher.splitlines() if "CONTAINS toLower($company)" in line
        )
        assert "c.name" in match_line, (
            "suppliers_of must match the *customer* name; matching the supplier returns X's own "
            "supplier list rather than the answer to the question asked"
        )

    def test_customers_of_filters_on_the_supplier_side(self):
        cypher = get_template("customers_of").cypher
        match_line = next(
            line for line in cypher.splitlines() if "CONTAINS toLower($company)" in line
        )
        assert "s.name" in match_line

    def test_every_one_hop_template_returns_a_chunk_id(self):
        """A statement with no provenance is uncitable and therefore cannot be an answer."""
        for name, template in TEMPLATES.items():
            if not template.returns_statements:
                continue
            assert "chunk" in template.cypher.lower(), (
                f"template {name} returns statements but no chunk id"
            )


class TestAggregationCitations:
    def test_aggregate_carries_every_contributing_chunk(self):
        result = {
            "records": [
                {"chunk_ids": ["c1", "c2", "c3"]},
                {"chunk_ids": ["c4"]},
            ],
            "statement_rows": ["NVIDIA has 3 suppliers", "NVIDIA has 1 competitor"],
        }
        context = assemble("q", "graph", graph_result=result, passages=[])
        assert "[chunk:c1] [chunk:c2] [chunk:c3]" in context.graph_statements[0]

    def test_aggregate_answer_passes_the_citation_gate(self):
        result = {
            "records": [{"chunk_ids": ["c1", "c2"]}],
            "statement_rows": ["NVIDIA has 2 suppliers"],
        }
        context = assemble("q", "graph", graph_result=result, passages=[])
        from kg_rag.answer import ExtractiveGenerator

        answer = ExtractiveGenerator().generate(context)
        report = validate_citations(answer, context)
        assert report.ok, report.problem
        assert set(report.valid) == {"c1", "c2"}

    def test_count_relationships_returns_chunk_ids(self):
        assert "chunk_ids" in get_template("count_relationships").cypher


class TestOutOfScopeRouting:
    VOCAB = build_corpus_vocabulary(
        ["TSMC manufactures advanced logic dies and competes with Intel on foundry capacity."]
    )

    def router(self) -> KeywordRouter:
        return KeywordRouter(
            known_names=["Taiwan Semiconductor Manufacturing Company Limited", "Intel Corporation"],
            corpus_vocabulary=self.VOCAB,
        )

    def test_unrelated_question_is_refused(self):
        """The bug: this was answered with sentences about lithography equipment."""
        decision = self.router().route("What is the weather in Paris?")
        assert decision.route == "out_of_scope"

    def test_on_topic_question_is_not_refused(self):
        decision = self.router().route("Which companies does Intel compete with?")
        assert decision.route != "out_of_scope"

    def test_entity_question_is_never_refused_for_being_unusual(self):
        """An anchor entity in the graph is evidence enough on its own."""
        decision = self.router().route("What is the weather at Intel Corporation?")
        assert decision.route != "out_of_scope"

    def test_empty_vocabulary_disables_the_check(self):
        """A router built without a vocabulary must not refuse everything."""
        router = KeywordRouter(known_names=["Intel Corporation"])
        assert router.route("What is the weather in Paris?").route != "out_of_scope"

    def test_stopwords_cannot_satisfy_the_scope_check(self):
        """Otherwise 'what' and 'is' would match the corpus and every question would look on-topic."""
        assert "what" not in content_terms("What is the weather in Paris?")
        assert "weather" in content_terms("What is the weather in Paris?")


class TestEmptyAnswerRefused:
    def test_empty_answer_is_refused_not_returned_blank(self):
        """The bug: an empty answer passed the citation gate and reached the caller as if it
        were a result."""
        from kg_rag.answer import ExtractiveGenerator, generate_grounded

        context = AssembledContext(
            graph_statements=["a fact with no provenance"], passages=[], question="q"
        )
        result = generate_grounded(context, ExtractiveGenerator())
        assert result.refused
        assert result.answer == ""


class TestCitationGateOnEmptyAnswers:
    def test_empty_answer_is_neither_ok_nor_a_defect(self):
        """Refusing produces no answer. That is valid, so validate_citations must not call it a
        failure — generate_grounded handles emptiness before it gets here."""
        context = AssembledContext(graph_statements=[], passages=[], question="q")
        assert validate_citations("", context).ok


class TestGazetteerLookup:
    def test_suffix_variant_resolves_to_the_registered_name(self):
        """The bug: 'Taiwan Semiconductor Manufacturing Company' never matched '...Limited', so
        no supplier edge was ever extracted from that filing."""
        from kg_rag.extract import Gazetteer

        gaz = Gazetteer()
        gaz.add("TSM", EntityType.COMPANY, "Taiwan Semiconductor Manufacturing Company Limited")
        assert gaz.lookup("Taiwan Semiconductor Manufacturing Company") is not None

    def test_lookup_is_not_a_substring_match(self):
        from kg_rag.extract import Gazetteer

        gaz = Gazetteer()
        gaz.add("AVGO", EntityType.COMPANY, "Broadcom Inc.")
        assert gaz.lookup("Broadcom Fiber") is None


class TestRelationNamePatterns:
    @pytest.mark.parametrize(
        "text,expected_type",
        [
            ("Taiwan Semiconductor Manufacturing Company is our principal foundry supplier.",
             RelationshipType.SUPPLIES),
            ("Our principal supplier is TSMC.", RelationshipType.SUPPLIES),
            ("We compete with Intel and AMD in data center accelerators.",
             RelationshipType.COMPETES_WITH),
            ("We purchase from TSMC for wafers.", RelationshipType.CUSTOMER_OF),
            ("Our consolidated subsidiaries include Foo Industries and Bar Labs.",
             RelationshipType.PARENT_OF),
            ("Our financial statements are audited by Ernst & Young.",
             RelationshipType.AUDITED_BY),
        ],
    )
    def test_pattern_matches(self, text, expected_type):
        from kg_rag.extract import _RELATION_PATTERNS

        found = [rt for _p, rt, _pers in _RELATION_PATTERNS for _m in _p.finditer(text)]
        assert expected_type in found, f"{text!r} produced {found}"

    def test_name_capture_does_not_cross_a_sentence_boundary(self):
        """The bug: the capture ran across '. ' and swallowed a whole sentence, so the real
        counterparty was never isolated and the extraction was discarded."""
        from kg_rag.extract import _RELATION_PATTERNS

        text = ("TSMC manufactures our dies. Micron Technology and SK Hynix supply the "
                "high-bandwidth memory used in our products.")
        captured = [
            m.group("other")
            for _p, _rt, _pers in _RELATION_PATTERNS
            for m in _p.finditer(text)
        ]
        for name in captured:
            assert "." not in name, f"name crossed a sentence boundary: {name!r}"


class TestRelationDirection:
    """Direction comes from the pattern's perspective, not a fixed convention."""

    def _extract(self, text: str, subject: str = "NVIDIA Corporation"):
        """Extract against a gazetteer holding exactly the counterparties each test names.

        The sample gazetteer only knows the eight semiconductor registrants, so a test naming
        "TSMC" or "Ernst & Young" would measure gazetteer coverage, not direction.
        """
        from kg_rag.extract import Gazetteer, OfflineExtractor

        gaz = Gazetteer()
        gaz.add("NVDA", EntityType.COMPANY, "NVIDIA Corporation", cik="0001045810")
        gaz.add("TSM", EntityType.COMPANY, "Taiwan Semiconductor Manufacturing Company Limited")
        gaz.add("INTC", EntityType.COMPANY, "Intel Corporation")
        gaz.add("EY", EntityType.AUDITOR, "Ernst & Young")
        gaz.add("Foo", EntityType.SUBSIDIARY, "Foo Industries")
        result = OfflineExtractor(gaz).extract(
            text, "chunk-1", subject=subject, filed_at="2024-02-21"
        )
        names = {e.local_id: e.name for e in result.entities}
        return [
            (names[r.from_local_id], r.type.value, names[r.to_local_id])
            for r in result.relationships
        ]

    def test_counterparty_supplier_points_at_the_subject(self):
        """'TSMC is our supplier' means TSMC SUPPLIES NVIDIA."""
        edges = self._extract(
            "Taiwan Semiconductor Manufacturing Company is our principal foundry supplier."
        )
        assert edges, "no supplier edge extracted"
        frm, rel, to = edges[0]
        assert rel == "SUPPLIES"
        assert "Taiwan Semiconductor" in frm
        assert to == "NVIDIA Corporation"

    def test_purchase_from_points_from_the_subject(self):
        """"We purchase from X" means subject CUSTOMER_OF X. The old fixed convention emitted
        X CUSTOMER_OF subject, which is the reverse fact."""
        edges = self._extract(
            "We purchase from Taiwan Semiconductor Manufacturing Company for wafers."
        )
        assert edges, "no customer edge extracted"
        subject, rel, other = edges[0]
        assert rel == "CUSTOMER_OF"
        assert subject == "NVIDIA Corporation"
        assert "Taiwan Semiconductor" in other

    def test_audited_by_points_from_the_subject(self):
        edges = self._extract("Our financial statements are audited by Ernst & Young.")
        assert edges, "no auditor edge extracted"
        subject, rel, other = edges[0]
        assert rel == "AUDITED_BY"
        assert subject == "NVIDIA Corporation"
        assert other == "Ernst & Young"

    def test_parent_of_points_from_the_subject(self):
        edges = self._extract("Our subsidiaries include Foo Industries.")
        assert edges, "no parent_of edge extracted"
        subject, rel, other = edges[0]
        assert rel == "PARENT_OF"
        assert subject == "NVIDIA Corporation"
        assert other == "Foo Industries"


class TestScopedEntityResolution:
    def test_same_officer_name_under_two_filers_stays_two_nodes(self):
        from kg_rag.extract import ExtractedEntity
        from kg_rag.resolve import EntityResolver

        resolver = EntityResolver()
        for filer in ("0000000001", "0000000002"):
            resolver.resolve_entity(
                ExtractedEntity(
                    local_id="e1", type=EntityType.OFFICER, name="Jane Smith",
                    source_chunk_id="c1",
                ),
                filer_cik=filer,
            )
        assert len(resolver.index.by_entity_id) == 2

    def test_same_officer_name_under_one_filer_is_one_node(self):
        from kg_rag.extract import ExtractedEntity
        from kg_rag.resolve import EntityResolver

        resolver = EntityResolver()
        for _ in range(3):
            resolver.resolve_entity(
                ExtractedEntity(
                    local_id="e1", type=EntityType.OFFICER, name="Jane Smith",
                    source_chunk_id="c1",
                ),
                filer_cik="0000000001",
            )
        assert len(resolver.index.by_entity_id) == 1

    def test_cik_padding_does_not_defeat_scoping(self):
        """"1045810" and "0001045810" are the same filer."""
        from kg_rag.extract import ExtractedEntity
        from kg_rag.resolve import EntityResolver

        resolver = EntityResolver()
        resolver.resolve_entity(
            ExtractedEntity(local_id="e1", type=EntityType.OFFICER, name="Jane Smith"),
            filer_cik="1045810",
        )
        resolver.resolve_entity(
            ExtractedEntity(local_id="e1", type=EntityType.OFFICER, name="Jane Smith"),
            filer_cik="0001045810",
        )
        assert len(resolver.index.by_entity_id) == 1


class TestTemporalEdgeIdentity:
    def test_later_filing_does_not_overwrite_the_earlier_fact(self):
        from kg_rag.config import edge_id

        early = edge_id("a", RelationshipType.SUPPLIES, "b", "2023-01-01")
        late = edge_id("a", RelationshipType.SUPPLIES, "b", "2024-01-01")
        assert early != late

    def test_resolver_keeps_both_dates(self):
        from kg_rag.extract import ExtractedRelationship
        from kg_rag.resolve import EntityResolver

        resolver = EntityResolver()
        edges = resolver.resolve_relationships(
            [
                ExtractedRelationship("x", "y", RelationshipType.SUPPLIES, {"as_of": "2023-01-01"}),
                ExtractedRelationship("x", "y", RelationshipType.SUPPLIES, {"as_of": "2024-01-01"}),
            ],
            {"x": "ea", "y": "eb"},
        )
        assert len({e.edge_id for e in edges}) == 2


class TestMentionsAreChunkIds:
    def test_merging_mentions_does_not_append_surface_forms(self):
        """mentions holds chunk ids (the join key into pgvector); aliases hold surface forms.
        They were fed through one helper, which mixed the two vocabularies."""
        from kg_rag.extract import ExtractedEntity
        from kg_rag.resolve import EntityResolver

        resolver = EntityResolver()
        first = resolver.resolve_entity(
            ExtractedEntity(local_id="e1", type=EntityType.COMPANY, name="NVIDIA Corporation",
                            source_chunk_id="chunk-a")
        )
        resolver.resolve_entity(
            ExtractedEntity(local_id="e1", type=EntityType.COMPANY, name="NVIDIA Corporation",
                            source_chunk_id="chunk-b")
        )
        entity = resolver.index.by_entity_id[first]
        assert set(entity.mentions) == {"chunk-a", "chunk-b"}
        assert "NVIDIA Corporation" not in entity.mentions

class TestRouterAnchorExtraction:
    """The router must anchor questions to registered entities despite short forms
    ("TSMC", "AMD") and possessives ("NVIDIA's")."""

    NAMES: ClassVar[list[str]] = [
        "Taiwan Semiconductor Manufacturing Company Limited",
        "Intel Corporation",
        "Micron Technology, Inc.",
        "NVIDIA Corporation",
        "Advanced Micro Devices, Inc.",
        "Broadcom Inc.",
        "QUALCOMM Incorporated",
        "ASML Holding N.V.",
    ]

    def test_short_forms_resolve(self):
        from kg_rag.router import extract_company, match_names

        assert extract_company("What does TSMC supply?", self.NAMES) == (
            "Taiwan Semiconductor Manufacturing Company Limited"
        )
        assert extract_company("Who does AMD compete with?", self.NAMES) == (
            "Advanced Micro Devices, Inc."
        )
        # Both short forms resolve, longest first.
        assert match_names("TSMC and AMD", self.NAMES) == [
            "Taiwan Semiconductor Manufacturing Company Limited",
            "Advanced Micro Devices, Inc.",
        ]

    def test_possessives_resolve(self):
        from kg_rag.router import extract_company

        assert extract_company("Who supplies NVIDIA's foundry?", self.NAMES) == (
            "NVIDIA Corporation"
        )
        assert extract_company("Micron's equipment supplier?", self.NAMES) == (
            "Micron Technology, Inc."
        )

    def test_substring_full_name_still_works(self):
        from kg_rag.router import extract_company

        assert extract_company(
            "What is Intel Corporation's strategy?", self.NAMES
        ) == "Intel Corporation"

    def test_unregistered_name_returns_none(self):
        from kg_rag.router import extract_company

        assert extract_company("Who is the CEO of Samsung?", self.NAMES) is None

    def test_templates_use_expected_question_patterns(self):
        from kg_rag.router import pick_template

        assert pick_template("Who supplies NVIDIA?", self.NAMES) == "suppliers_of"
        assert pick_template("Who does NVIDIA compete with?", self.NAMES) == "competitors_of"
        assert pick_template("Which companies does TSMC supply?", self.NAMES) == "customers_of"
        assert pick_template(
            "How many relationships does NVIDIA have?", self.NAMES
        ) == "count_relationships"
        assert pick_template(
            "Which suppliers do NVIDIA and AMD have in common?", self.NAMES
        ) == "shared_counterparties"
        assert pick_template("What is a lithography system?", self.NAMES) is None


class TestExtractiveGeneratorDedup:
    """Two different statements citing the same chunk are distinct facts; both must appear.
    The old code deduped by chunk id, so the second statement from a shared chunk was dropped.
    """

    def test_same_chunk_different_statements_both_appear(self):
        from kg_rag.answer import ExtractiveGenerator

        context = AssembledContext(
            graph_statements=[
                "Micron Technology, Inc. --[SUPPLIES]--> NVIDIA Corporation [chunk:c1]",
                (
                    "Taiwan Semiconductor Manufacturing Company Limited --[SUPPLIES]--> "
                    "NVIDIA Corporation [chunk:c1]"
                ),
            ],
            passages=[],
            question="q",
        )
        answer = ExtractiveGenerator().generate(context)
        assert "Micron Technology, Inc." in answer
        assert "Taiwan Semiconductor Manufacturing Company Limited" in answer
        assert answer.count("[chunk:c1]") == 2
