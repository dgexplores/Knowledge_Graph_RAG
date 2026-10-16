"""Query router and hybrid retrieval.

Decision D-09: the router returns an enum **and** a confidence. Below threshold it runs both the
graph and vector paths and merges them. The fallback rate is measured and reported — it is the
system's own statement of how often it is unsure.

Retrieval never executes model-authored Cypher. The router names a query type; `cypher_templates`
supplies the statement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kg_rag.cypher_templates import DEFAULT_LIMIT, Template, bind, get_template

ROUTES = ("graph", "vector", "definition", "out_of_scope")

# Question shapes that a vector index genuinely cannot answer. Matched before the model is asked,
# because an aggregation is not a judgement call — it is a structural fact about the question.
_MULTIHOP_MARKERS = (
    r"\bboth\b.*\band\b", r"\bshared\b", r"\bcommon\b", r"\bin common\b",
    r"\bwhich (?:companies|suppliers|vendors|competitors|firms)\b",
    r"\bhow many\b", r"\bcount\b", r"\bnumber of\b", r"\btotal\b",
    r"\bconnect(?:ed|ion|s)?\b", r"\bbetween\b.*\band\b",
    r"\balso\b.*\b(?:in|filed|mentioned)\b",
    r"\bsame\b.*\b(?:supplier|counterparty|vendor)\b",
)
_DEFINITION_MARKERS = (r"\bwhat is\b", r"\bwhat does .* mean\b", r"\bdefine\b", r"\bmeaning of\b")

_MULTIHOP_RE = re.compile("|".join(_MULTIHOP_MARKERS), re.IGNORECASE)
_DEFINITION_RE = re.compile("|".join(_DEFINITION_MARKERS), re.IGNORECASE)


@dataclass
class RouteDecision:
    route: str
    query_type: str | None
    confidence: float
    params: dict[str, str] = field(default_factory=dict)
    # Which rule decided. Recorded so a misroute is attributable rather than mysterious.
    decided_by: str = "unknown"
    fallback: bool = False

    @property
    def runs_both(self) -> bool:
        return self.route == "both" or self.fallback


#: Function words carry no topical signal, so they must not be allowed to satisfy the corpus
#: check — otherwise "what is the weather in Paris" looks on-topic because "what" is in the corpus.
_STOPWORDS = frozenset(
    ["a", "an", "the", "this", "that", "these", "those", "and", "or", "but", "if", "then", "than", "so", "such", "as", "of", "in", "on", "at", "by", "for", "with", "from", "into", "over", "under", "about", "above", "below", "between", "among", "during", "before", "after", "while", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "have", "has", "had", "having", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "not", "no", "nor", "any", "some", "all", "each", "every", "both", "few", "more", "most", "other", "another", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "does", "it", "its", "their", "there", "here", "they", "them", "we", "us", "our", "your", "his", "her", "him", "he", "she", "i", "me", "my", "mine", "ours", "yours", "tell", "explain", "describe", "list", "give", "show"]
)

_MIN_TERM_LENGTH = 4

_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}")


def content_terms(text: str) -> set[str]:
    """Topical tokens of a question, with function words and very short tokens removed."""
    return {
        token
        for token in _TOKEN_RE.findall((text or "").lower())
        if len(token) >= _MIN_TERM_LENGTH and token not in _STOPWORDS
    }


def build_corpus_vocabulary(texts) -> set[str]:
    """Vocabulary of the ingested corpus, used to decide whether a question is answerable at all.

    This is the evidence for an out-of-scope refusal. Without it the router had no such rule —
    only an empty question was refused — so "What is the weather in Paris?" was answered with
    sentences about lithography equipment, which is worse than refusing: it looks authoritative.
    """
    vocabulary: set[str] = set()
    for text in texts:
        vocabulary |= content_terms(text)
    return vocabulary


def heuristic_route(question: str) -> RouteDecision | None:
    """Structural pre-classifier. Returns None when the question is not obviously shaped.

    Deliberately conservative: it only claims a question when a marker is unambiguous. When it
    abstains, the model decides.
    """
    if not question or not question.strip():
        return RouteDecision(
            route="out_of_scope", query_type=None, confidence=1.0, decided_by="heuristic:empty"
        )

    if _MULTIHOP_RE.search(question):
        return RouteDecision(
            route="graph", query_type=None, confidence=0.6,
            decided_by="heuristic:multihop_marker",
        )
    if _DEFINITION_RE.search(question):
        return RouteDecision(
            route="definition", query_type=None, confidence=0.6,
            decided_by="heuristic:definition_marker",
        )
    return None


#: Short forms people actually type ("TSMC", "AMD") mapped to the registered name the graph uses.
#: Only the forms that substring/first-word matching cannot recover on their own.
SHORT_FORMS: dict[str, str] = {
    "tsmc": "Taiwan Semiconductor Manufacturing Company Limited",
    "amd": "Advanced Micro Devices, Inc.",
}

_POSSESSIVE_RE = re.compile(r"\b(\w+)'s\b")
_WORD_RE = re.compile(r"[a-z0-9]+")

def _normalize_question(question: str) -> str:
    """Lowercase and strip possessive markers so 'NVIDIA's' matches 'NVIDIA Corporation'."""
    return _POSSESSIVE_RE.sub(r"\1", (question or "").lower())

def match_names(question: str, known_names: list[str]) -> list[str]:
    """All registered names the question refers to, longest first.

    Tries, in order: short forms (TSMC -> Taiwan Semiconductor...), exact substring of the
    registered name, and finally the name's first distinctive word ("micron" -> Micron Technology,
    Inc.). Deduped and longest-first so 'Advanced Micro Devices' beats 'Micro'.
    """
    lowered = _normalize_question(question)
    tokens = set(_WORD_RE.findall(lowered))
    matched: list[str] = []

    for short, full in SHORT_FORMS.items():
        if short in tokens and full not in matched:
            matched.append(full)

    for name in sorted(known_names, key=len, reverse=True):
        name_l = name.lower()
        if name_l in lowered and name not in matched:
            matched.append(name)
            continue
        first = name_l.split()[0]
        if len(first) >= 4 and first in tokens and name not in matched:
            matched.append(name)

    return matched


def extract_company(question: str, known_names: list[str]) -> str | None:
    """Find the entity the question is about.

    Longest-first matters: 'Taiwan Semiconductor Manufacturing Company' must beat 'Taiwan'.
    Returns None when the question names no registered entity.
    """
    names = match_names(question, known_names)
    return names[0] if names else None



def pick_template(question: str, known_names: list[str]) -> str | None:
    """Choose a template from the question's shape and the company it names."""
    company = extract_company(question, known_names)
    lowered = question.lower()

    if company is None:
        return None

    if re.search(r"\bshared\b|\bcommon\b|\bin common\b", lowered):
        # The two-company template needs a second anchor; match_names resolves short forms and
        # possessives, so "NVIDIA and AMD" resolves just like the registered names.
        names = match_names(question, known_names)
        if len(names) >= 2:
            return "shared_counterparties"

    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal\b", lowered):
        return "count_relationships"
    if re.search(r"\bwho supplies\b|\bwho are the suppliers of\b|\bwho is the supplier of\b", lowered):
        return "suppliers_of"
    if re.search(r"\bwhich companies? (?:does|do) \w+ (?:supply|supplies|manufacture|sell|provide)\b", lowered):
        return "customers_of"
    if re.search(r"\bcompetitor|competes?\b", lowered):
        return "competitors_of"
    if re.search(r"\bsubsidiar|wholly[- ]owned|owns\b", lowered):
        return "subsidiaries_of"
    if re.search(r"\bauditor|audited by\b", lowered):
        return "auditor_of"
    if re.search(r"\bofficer|director|ceo|cfo\b", lowered):
        return "officers_of"
    if re.search(r"\bconnect|reachab|two[- ]hop|chain\b", lowered):
        return "counterparty_chain"
    return "entity_exists"


class KeywordRouter:
    """Deterministic router. No model call.

    The model-backed router is the intended production path; this exists so the pipeline is
    testable and demonstrable without a key, and so routing decisions are reproducible in CI. Every
    decision it makes is labelled `router=stub:keyword`.
    """

    name = "stub:keyword"

    def __init__(
        self,
        known_names: list[str],
        fallback_threshold: float = 0.55,
        corpus_vocabulary: set[str] | None = None,
    ) -> None:
        if not 0.0 < fallback_threshold <= 1.0:
            raise ValueError(f"fallback_threshold must be in (0, 1], got {fallback_threshold}")
        self.known_names = known_names
        self.fallback_threshold = fallback_threshold
        # Empty means "scope checking disabled", so a router built without one behaves as before
        # rather than refusing every question.
        self.corpus_vocabulary = corpus_vocabulary or set()

    def _is_out_of_scope(self, question: str, company: str | None) -> bool:
        """Whether nothing in the question refers to the corpus.

        Only consulted when no graph template applies. With an empty vocabulary the check is
        disabled, so tests and callers that do not build one keep the old behaviour.
        """
        if not self.corpus_vocabulary:
            return False
        terms = content_terms(question) - content_terms(company or "")
        if not terms:
            return False
        return not (terms & self.corpus_vocabulary)

    def route(self, question: str) -> RouteDecision:
        if not question or not question.strip():
            return RouteDecision("out_of_scope", None, 1.0, decided_by=f"{self.name}:empty")

        heuristic = heuristic_route(question)
        company = extract_company(question, self.known_names)
        query_type = pick_template(question, self.known_names)

        if heuristic and heuristic.route == "out_of_scope":
            return RouteDecision(
                "out_of_scope", None, 1.0, decided_by=f"{self.name}:{heuristic.decided_by}"
            )

        # Scope check runs before the vector fallback, because answering a question the corpus
        # cannot possibly address is worse than refusing it.
        if query_type is None and self._is_out_of_scope(question, company):
            return RouteDecision(
                "out_of_scope", None, 0.9,
                decided_by=f"{self.name}:no_term_in_corpus",
            )

        route = heuristic.route if heuristic else None
        confidence = heuristic.confidence if heuristic else 0.0
        decided_by = heuristic.decided_by if heuristic else f"{self.name}:no_marker"

        if query_type is None:
            # No template matched. Vector retrieval is the safe default: a wrong graph answer is
            # worse than a generic passage, because it looks authoritative.
            if route != "definition":
                return RouteDecision(
                    "vector", None, 0.3, decided_by=f"{self.name}:no_template"
                )
            return RouteDecision(
                "definition", None, 0.5, decided_by=f"{self.name}:definition_no_template"
            )

        if route is None:
            # Single-hop company question with a matched template: the graph answers it directly.
            route = "graph"
            confidence = 0.7
            decided_by = f"{self.name}:template_match"

        if route == "definition":
            return RouteDecision("definition", None, confidence, decided_by=decided_by)

        if company is None:
            # A multi-hop shape with no resolvable anchor entity. The graph cannot run it.
            return RouteDecision(
                "vector", None, 0.3, params={}, decided_by=f"{self.name}:no_anchor_entity"
            )

        decision = RouteDecision(
            route=route,
            query_type=query_type,
            confidence=confidence,
            params={"company": company},
            decided_by=decided_by,
        )
        if query_type == "shared_counterparties":
            # Both anchors must come from the question; short forms ("AMD") and possessives
            # ("Intel's") are resolved by match_names, not by brittle substring matching.
            names = match_names(question, self.known_names)
            if len(names) >= 2:
                decision.params = {
                    "company_a": names[0],
                    "company_b": names[1],
                }

        # D-09: low confidence runs both paths and merges.
        if decision.confidence < self.fallback_threshold:
            decision.fallback = True
        return decision


class GraphRetriever:
    """Executes templates against Neo4j. Holds no Cypher of its own."""

    def __init__(self, store, known_names: list[str], limit: int = DEFAULT_LIMIT) -> None:
        self.store = store
        self.known_names = known_names
        self.limit = limit

    def retrieve(
        self, query_type: str, params: dict[str, str], as_of: str | None = None
    ) -> dict:
        template: Template = get_template(query_type)
        bound = bind(template, params, as_of=as_of, limit=self.limit)
        records = self.store.run_query(template.cypher, bound)
        return {
            "query_type": query_type,
            "cypher": _redact(template.cypher),
            "params": bound,
            "records": records,
            "statement_rows": _to_statements(records) if template.returns_statements else [],
            "hit": bool(records),
        }


def _to_statements(records: list[dict]) -> list[str]:
    """Convert path rows to readable statements.

    Deterministic string formatting, never a model call. Raw triples generate awkward prose and
    an extra model call here would be both slower and less reproducible.
    """
    statements: list[str] = []
    for record in records:
        if "subject" in record and "relation" in record and "object" in record:
            statements.append(
                f"{record['subject']} --[{record['relation']}]--> {record['object']}"
            )
        elif "from_name" in record:
            statements.append(
                f"{record['from_name']} --[{record['hop1_type']}]--> {record['via_name']} "
                f"--[{record['hop2_type']}]--> {record['to_name']}"
            )
        elif "statement" in record:
            statements.append(str(record["statement"]))
        elif "count" in record:
            statements.append(
                f"{record.get('subject', '')} has {record['count']} "
                f"{record['relation']} relationship(s)"
            )
    return statements


def _redact(cypher: str) -> str:
    """Whitespace-normalized Cypher for logging. Contains no user data — parameters are separate."""
    return " ".join(cypher.split())
