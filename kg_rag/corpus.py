"""Corpus definition and resolution.

Registrants are selected by **ticker**, and CIKs are resolved at runtime from SEC's official
`company_tickers.json`. Hardcoded CIKs are avoided deliberately: a wrong identifier silently
resolves to a different registrant and corrupts the graph in a way that looks like a data problem.

Sector choice is decision D-02: one sector, because cross-registrant naming is what makes a
multi-hop join possible. These 12 are semiconductor issuers that name each other as suppliers,
customers, and competitors in their own filings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Sector: semiconductors + semiconductor equipment. Deliberately one sector (D-02).
# Chosen because supplier/customer/competitor edges exist *between* them, which is the shape the
# 3-hop demo question requires.
REGISTRANTS: tuple[str, ...] = (
    "TSM",    # foundry — names Apple, Nvidia, AMD, Qualcomm
    "ASML",   # lithography — names every leading-edge fab
    "NVDA",   # fabless — names TSMC, Samsung, SK Hynix
    "AMD",    # fabless — names TSMC, GlobalFoundries
    "AVGO",   # fabless/custom silicon — names TSMC
    "INTC",   # integrated — names TSMC, ASML
    "QCOM",   # fabless — names TSMC, Samsung
    "MU",     # memory — names ASML, Applied Materials
    "AMAT",   # equipment — names every fab
    "LRCX",   # equipment — names every fab
    "TXN",    # analog — names TSMC
    "GFS",    # foundry — names ASML, AMD
)

# Forms ingested. K carries the narrative where relationships are named (D-01).
DEFAULT_FORMS: tuple[str, ...] = ("10-K",)

# Sections pulled from a 10-K, per docs/00-corpus.md. Item 1A is where supply-chain and
# concentration risk language lives, which is the point of the 3-hop question.
SECTIONS_OF_INTEREST: tuple[str, ...] = (
    "Item 1. Business",
    "Item 1A. Risk Factors",
    "Item 7. Management's Discussion and Analysis",
)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass(frozen=True)
class Registrant:
    """One SEC registrant in the corpus."""

    ticker: str
    name: str
    cik: str | None = None

    @property
    def cik_padded(self) -> str:
        if not self.cik:
            raise ValueError(f"{self.ticker} has no CIK; resolve it before use")
        return self.cik.zfill(10)


@dataclass
class FilingRef:
    """A filing to ingest."""

    registrant: Registrant
    accession: str
    form: str
    filed_at: str
    period_of_report: str
    primary_document: str

    @property
    def url(self) -> str:
        cik = self.registrant.cik_padded.lstrip("0")
        acc = self.accession.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{self.primary_document}"
        )


@dataclass
class CorpusSelection:
    """Registrants with CIKs resolved, plus whatever failed to resolve."""

    registrants: list[Registrant] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def resolved_count(self) -> int:
        return len(self.registrants)


def parse_ticker_map(payload: dict) -> dict[str, tuple[str, str]]:
    """SEC's `company_tickers.json` is `{id: {cik_str, ticker, title}}`. Flatten it."""
    out: dict[str, tuple[str, str]] = {}
    for entry in payload.values():
        ticker = str(entry.get("ticker", "")).upper().strip()
        cik = entry.get("cik_str")
        if not ticker or cik is None:
            continue
        out[ticker] = (str(cik).zfill(10), str(entry.get("title", "")))
    return out


def resolve_registrants(
    tickers: tuple[str, ...], ticker_map: dict[str, tuple[str, str]]
) -> CorpusSelection:
    """Resolve tickers to registrants.

    An unresolved ticker is reported rather than skipped silently: a corpus quietly one issuer
    short changes which multi-hop questions are even answerable.
    """
    selection = CorpusSelection()
    for ticker in tickers:
        hit = ticker_map.get(ticker.upper())
        if hit is None:
            selection.unresolved.append(ticker)
            continue
        cik, name = hit
        selection.registrants.append(Registrant(ticker=ticker.upper(), name=name, cik=cik))
    return selection


def load_ticker_map(raw: str) -> dict[str, tuple[str, str]]:
    """Parse a `company_tickers.json` document. Raises on malformed input rather than
    returning a partial map, which would look like missing tickers."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"company_tickers.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(
            f"expected a JSON object keyed by id, got {type(payload).__name__}"
        )
    return parse_ticker_map(payload)
