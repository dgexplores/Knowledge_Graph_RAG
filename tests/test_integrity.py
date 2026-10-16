"""Tests for the pieces whose bugs were silent.

Every case here corresponds to a defect that shipped past a green test run:

- the store wrote `:REL` while templates matched concrete types, so every graph query returned
  zero rows while the whole suite stayed green;
- `CitationReport.ok` passed an answer with no citations at all;
- the extractive generator appended a second `[chunk:ID]` to statements that already had one;
- the evaluator reported a citation-failure count that could only ever be zero.

Cypher is asserted on the statement text rather than against a live database, so these run with no
services. The end-to-end run against real Neo4j is `python -m kg_rag.cli demo`.
"""

from __future__ import annotations

import re

import pytest

from kg_rag.config import RelationshipType, edge_id
from kg_rag.cypher_templates import TEMPLATES, UnknownQueryType, bind, get_template
from kg_rag.eval_runner import citation_defects, score_answer
from kg_rag.merge import AssembledContext, CitationReport, assemble, validate_citations


class TestEdgeIdentity:
    def test_same_filing_merges(self):
        """Re-ingesting the same filing must be idempotent."""
        a = edge_id("ent_x", RelationshipType.SUPPLIES, "ent_y", "2024-02-01")
        b = edge_id("ent_x", RelationshipType.SUPPLIES, "ent_y", "2024-02-01")
        assert a == b

    def test_later_filing_is_a_distinct_edge(self):
        """The bug: a later filing restating a relationship overwrote the earlier fact, which made
        every `as_of <= ?` filter a no-op. Same endpoints, different date, different edge."""
        early = edge_id("ent_x", RelationshipType.SUPPLIES, "ent_y", "2023-02-01")
        late = edge_id("ent_x", RelationshipType.SUPPLIES, "ent_y", "2024-02-01")
        assert early != late

    def test_unknown_as_of_is_stable(self):
        assert edge_id("a", RelationshipType.SUPPLIES, "b") == edge_id(
            "a", RelationshipType.SUPPLIES, "b", None
        )

    def test_type_still_separates_edges(self):
        base = ("x", None, "y")
        assert edge_id(*base[:1], RelationshipType.SUPPLIES, *base[2:]) != edge_id(
            *base[:1], RelationshipType.CUSTOMER_OF, *base[2:]
        )


class TestTypedRelationshipLabels:
    """Edges are written with concrete types, so templates must match concrete types."""

    def test_store_writes_concrete_type_not_generic_rel(self):
        from kg_rag.store import UPSERT_EDGE_TEMPLATE

        rendered = UPSERT_EDGE_TEMPLATE.format(rel_type="SUPPLIES")
        assert ":SUPPLIES" in rendered
        assert ":REL " not in rendered and ":REL{" not in rendered

    def test_every_direct_template_matches_a_written_type(self):
        enum_values = {r.value for r in RelationshipType}
        used = set()
        for name, template in TEMPLATES.items():
            for label in re.findall(r"-\[r?:([A-Z_]+)", template.cypher):
                assert label in enum_values, (
                    f"template {name} matches relationship type {label!r}, which the store "
                    f"never writes — this query would silently return zero rows"
                )
                used.add(label)
        assert used, "no template uses a concrete relationship type"

    def test_no_template_queries_a_generic_rel_label(self):
        """`:REL` is not written any more, so a template matching it matches nothing."""
        for name, template in TEMPLATES.items():
            assert ":REL" not in template.cypher, (
                f"template {name} still matches the generic :REL label"
            )

    def test_relationship_type_value_is_no_longer_dead(self):
        """It existed to name the edge type; before it was wired up the store ignored it."""
        from kg_rag.store import relationship_type_value

        assert relationship_type_value(RelationshipType.SUPPLIES) == "SUPPLIES"


class TestTemplateCypherScope:
    def test_shared_counterparties_does_not_use_dropped_variable(self):
        """`r` was referenced in RETURN after a WITH that no longer carried it — a compile error at
        runtime, invisible to every test that did not execute the query."""
        cypher = TEMPLATES["shared_counterparties"].cypher
        return_clause = cypher[cypher.rindex("RETURN"):]
        assert not re.search(r"\br\.", return_clause), (
            f"RETURN references `r.`, but the preceding WITH no longer carries `r` into scope:\n"
            f"{return_clause}"
        )

    def test_shared_counterparties_aggregates_provenance_inside_the_with(self):
        """Provenance must be collected before the WITH drops the row variable."""
        cypher = TEMPLATES["shared_counterparties"].cypher
        assert "collect(DISTINCT r.source_chunk_id)" in cypher

    def test_every_template_filters_by_as_of(self):
        for name, template in TEMPLATES.items():
            if not template.returns_statements:
                continue
            assert "$as_of" in template.cypher, (
                f"template {name} returns time-scoped facts but never filters on as_of"
            )


class TestTemplateBinding:
    def test_unknown_type_raises_rather_than_returning_free_form(self):
        with pytest.raises(UnknownQueryType):
            get_template("DROP_DATABASE")

    def test_missing_parameter_raises(self):
        """A silently empty result is indistinguishable from 'the graph does not know'."""
        template = get_template("suppliers_of")
        with pytest.raises(ValueError):
            bind(template, {}, as_of=None)

    def test_blank_parameter_raises(self):
        template = get_template("suppliers_of")
        with pytest.raises(ValueError):
            bind(template, {"company": "   "}, as_of=None)

    def test_non_string_parameter_raises(self):
        """Type is checked: a number or a list would otherwise reach the driver and behave oddly."""
        template = get_template("suppliers_of")
        with pytest.raises(TypeError):
            bind(template, {"company": 42}, as_of=None)

    def test_valid_parameters_bind(self):
        bound = bind(get_template("suppliers_of"), {"company": "NVIDIA"}, as_of="2024-01-01")
        assert bound["company"] == "NVIDIA"
        assert bound["as_of"] == "2024-01-01"


class TestCitationGate:
    def test_uncited_answer_is_not_ok(self):
        """The bug: `ok` only checked for fabrication, so an answer with no citations passed."""
        context = AssembledContext(
            graph_statements=["NVIDIA supplies TSMC [chunk:c1]"], passages=[], question="q"
        )
        report = validate_citations("NVIDIA supplies TSMC.", context)
        assert report.uncited_claims == 1
        assert not report.ok
        assert report.problem == "claims without any citation"

    def test_fabricated_citation_is_not_ok(self):
        context = AssembledContext(
            graph_statements=["NVIDIA supplies TSMC [chunk:c1]"], passages=[], question="q"
        )
        report = validate_citations("NVIDIA supplies TSMC [chunk:made_up]", context)
        assert report.fabricated == ["made_up"]
        assert not report.ok

    def test_valid_citation_is_ok(self):
        context = AssembledContext(
            graph_statements=["NVIDIA supplies TSMC [chunk:c1]"], passages=[], question="q"
        )
        report = validate_citations("NVIDIA supplies TSMC [chunk:c1]", context)
        assert report.ok
        assert report.valid == ["c1"]

    def test_empty_answer_is_not_a_citation_failure(self):
        """Refusing produces no answer; that is a valid outcome, not a citation defect."""
        context = AssembledContext(graph_statements=[], passages=[], question="q")
        report = validate_citations("", context)
        assert report.ok
        assert report.uncited_claims == 0

    def test_problem_reports_fabrication_first(self):
        report = CitationReport(fabricated=["x"], uncited_claims=1)
        assert report.problem == "fabricated citations"


class TestExtractiveGenerator:
    def test_does_not_duplicate_the_chunk_tag(self):
        """The bug: `assemble` appends [chunk:ID]; the generator appended it again, so one
        citation was counted twice and the answer read 'X [chunk:c1] [chunk:c1]'."""
        from kg_rag.answer import ExtractiveGenerator

        context = AssembledContext(
            graph_statements=["NVIDIA supplies TSMC [chunk:c1]"], passages=[], question="q"
        )
        answer = ExtractiveGenerator().generate(context)
        assert answer.count("[chunk:c1]") == 1

    def test_graph_statements_carry_a_citation(self):
        from kg_rag.answer import ExtractiveGenerator

        context = AssembledContext(
            graph_statements=["NVIDIA supplies TSMC [chunk:c1]"], passages=[], question="q"
        )
        report = validate_citations(ExtractiveGenerator().generate(context), context)
        assert report.ok

    def test_provenance_free_statement_is_dropped(self):
        """A graph statement with no source chunk cannot be cited, so it cannot be an answer."""
        from kg_rag.answer import ExtractiveGenerator

        context = AssembledContext(
            graph_statements=["NVIDIA supplies TSMC"], passages=[], question="q"
        )
        assert ExtractiveGenerator().generate(context) == ""

    def test_respects_the_item_budget(self):
        from kg_rag.answer import ExtractiveGenerator

        statements = [f"Fact number {i} [chunk:c{i}]" for i in range(10)]
        context = AssembledContext(
            graph_statements=statements, passages=[], question="q"
        )
        answer = ExtractiveGenerator(max_sentences=3).generate(context)
        assert answer.count("[chunk:") == 3


class TestAssembleProvenance:
    def test_appends_exactly_one_chunk_tag(self):
        result = {
            "records": [{"chunk_id": "c1"}],
            "statement_rows": ["NVIDIA supplies TSMC"],
        }
        context = assemble("q", "graph", graph_result=result, passages=[])
        assert context.graph_statements == ["NVIDIA supplies TSMC [chunk:c1]"]

    def test_deduplicates_the_same_fact_from_both_paths(self):
        result = {
            "records": [{"chunk_id": "c1"}],
            "statement_rows": ["NVIDIA supplies TSMC", "NVIDIA supplies TSMC"],
        }
        context = assemble(
            "q", "both", graph_result=result, passages=[], query_type="suppliers_of"
        )
        assert context.graph_statements == ["NVIDIA supplies TSMC [chunk:c1]"]

    def test_statement_without_provenance_is_still_carried_but_uncitable(self):
        """It stays in the context (the model may still see it) but carries no [chunk:] tag, so the
        citation gate treats any answer leaning on it as uncited."""
        result = {"records": [{}], "statement_rows": ["NVIDIA supplies TSMC"]}
        context = assemble("q", "graph", graph_result=result, passages=[])
        assert context.graph_statements == ["NVIDIA supplies TSMC"]


class TestScorer:
    def test_requires_every_expected_term(self):
        """Strict on purpose: partial credit would reward naming every company in the graph."""
        ok, matched = score_answer(["NVIDIA", "TSMC"], "NVIDIA supplies TSMC.", False)
        assert ok and set(matched) == {"NVIDIA", "TSMC"}

        ok, matched = score_answer(["NVIDIA", "ASML"], "NVIDIA supplies TSMC.", False)
        assert not ok and matched == ["NVIDIA"]

    def test_refusal_is_never_correct_for_a_factual_question(self):
        ok, _ = score_answer(["NVIDIA"], "", True)
        assert not ok

    def test_matching_is_case_and_whitespace_insensitive(self):
        ok, _ = score_answer(["Taiwan Semiconductor"], "TSMC  and  TAIWAN SEMICONDUCTOR", False)
        assert ok

    def test_empty_expected_terms_do_not_auto_pass(self):
        """Guard against a vacuous `0 == 0`: a question set that forgot `expected_terms` would
        otherwise report 100% accuracy instead of an error."""
        assert score_answer([], "any answer at all", False) == (False, [])


class TestCitationDefectReporting:
    def test_refusals_are_not_defects(self):
        from kg_rag.eval_runner import ScoredResult

        results = [
            ScoredResult(
                question_id="q1", question="?", question_class="out_of_scope", system="hybrid",
                correct=True, answered=False, refused=True, citation_ok=False,
                citation_problem="claims without any citation",
            )
        ]
        assert citation_defects(results) == []

    def test_answered_failure_is_a_defect(self):
        from kg_rag.eval_runner import ScoredResult

        results = [
            ScoredResult(
                question_id="q2", question="?", question_class="1_hop", system="hybrid",
                correct=True, answered=True, refused=False, citation_ok=False,
                citation_problem="fabricated citations",
            )
        ]
        assert citation_defects(results) == ["q2 (hybrid): fabricated citations"]

    def test_clean_run_reports_zero(self):
        from kg_rag.eval_runner import ScoredResult

        results = [
            ScoredResult(
                question_id="q3", question="?", question_class="1_hop", system="hybrid",
                correct=True, answered=True, citation_ok=True,
            )
        ]
        assert citation_defects(results) == []