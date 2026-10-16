"""SEC EDGAR client.

The fair-access policy is enforced here, not left to the caller: a declared `User-Agent` with a
contact address, a request-rate ceiling, and polite sequential fetching. The SEC blocks
non-compliant clients mid-run, which surfaces as an intermittent failure that reads like a bug in
your own code.

Caching is by URL with an on-disk copy, so re-running ingestion costs no requests and costs no
money.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

from kg_rag import settings
from kg_rag.corpus import (
    DEFAULT_FORMS,
    SECTIONS_OF_INTEREST,
    TICKER_MAP_URL,
    FilingRef,
    Registrant,
    load_ticker_map,
    resolve_registrants,
)

JSON_HEADERS = {"Accept": "application/json"}
HTML_HEADERS = {"Accept": "text/html,application/xhtml+xml"}


class SecClientError(RuntimeError):
    """EDGAR request failed. Carries the status so the caller can distinguish blocks from typos."""


class RateLimiter:
    """Token-free fixed-window limiter. Simple and sufficient for a 10 req/s ceiling."""

    def __init__(self, max_per_second: float) -> None:
        if max_per_second <= 0:
            raise ValueError(f"max_per_second must be positive, got {max_per_second}")
        self._min_interval = 1.0 / max_per_second
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()


@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    from_cache: bool


@dataclass
class SecClient:
    """Minimal EDGAR client: ticker map, filing index, document fetch.

    Deliberately not an SDK. The three endpoints used are stable and public, and an SDK would add
    a dependency that obscures exactly which requests are being made.
    """

    user_agent: str
    max_rps: float = 5.0
    cache_dir: Path | None = None
    retries: int = 3
    _limiter: RateLimiter = field(init=False)
    _memory_cache: dict[str, str] = field(default_factory=dict, init=False)
    stats: dict[str, int] = field(default_factory=lambda: {"hits": 0, "misses": 0, "bytes": 0})

    @classmethod
    def from_env(cls, cache_dir: Path | None = None) -> SecClient:
        return cls(
            user_agent=settings.sec_user_agent(),
            max_rps=settings.sec_max_requests_per_second(),
            cache_dir=cache_dir if cache_dir is not None else Path(settings.corpus_root()),
        )

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.max_rps)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- transport ----------------------------------------------------------------------------

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{digest}.cache"

    def fetch(self, url: str, *, headers: dict[str, str] | None = None) -> FetchResult:
        """GET a URL, honouring the rate limit, the User-Agent, and the on-disk cache."""
        if url in self._memory_cache:
            self.stats["hits"] += 1
            return FetchResult(url, 200, self._memory_cache[url], from_cache=True)

        path = self._cache_path(url)
        if path and path.exists():
            text = path.read_text(encoding="utf-8")
            self._memory_cache[url] = text
            self.stats["hits"] += 1
            return FetchResult(url, 200, text, from_cache=True)

        merged = {"User-Agent": self.user_agent, **(headers or {})}
        last_error: Exception | None = None

        for attempt in range(self.retries):
            self._limiter.wait()
            request = urllib.request.Request(url, headers=merged)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    text = response.read().decode("utf-8", errors="replace")
                self.stats["misses"] += 1
                self.stats["bytes"] += len(text)
                self._memory_cache[url] = text
                if path:
                    path.write_text(text, encoding="utf-8")
                return FetchResult(url, 200, text, from_cache=False)
            except urllib.error.HTTPError as exc:
                # 403/429 are the SEC's non-compliant-client responses: backing off and retrying
                # wastes their time and yours. Fail loudly with the status instead.
                if exc.code in (403, 429):
                    raise SecClientError(
                        f"EDGAR refused the request ({exc.code}) for {url}. This is almost always "
                        f"a missing or unacceptable User-Agent. Set SEC_USER_AGENT to something "
                        f"like 'Your Name your@email.com' and retry more slowly."
                    ) from exc
                last_error = exc
                if exc.code < 500:
                    break
            except urllib.error.URLError as exc:
                last_error = exc
            time.sleep(2**attempt * 0.5)  # exponential backoff

        raise SecClientError(f"failed to fetch {url} after {self.retries} attempts: {last_error}")

    # -- endpoints ----------------------------------------------------------------------------

    def ticker_map(self) -> dict[str, tuple[str, str]]:
        """ticker -> (cik, title), from SEC's official mapping."""
        result = self.fetch(TICKER_MAP_URL, headers=JSON_HEADERS)
        return load_ticker_map(result.text)

    def resolve(self, tickers: tuple[str, ...]) -> list[Registrant]:
        return resolve_registrants(tickers, self.ticker_map()).registrants

    def submissions(self, registrant: Registrant) -> dict:
        """Filing history for one registrant."""
        url = f"https://data.sec.gov/submissions/CIK{registrant.cik_padded}.json"
        result = self.fetch(url, headers=JSON_HEADERS)
        try:
            return json.loads(result.text)
        except json.JSONDecodeError as exc:
            raise SecClientError(f"submissions for {registrant.ticker} was not JSON") from exc

    def filings(
        self,
        registrant: Registrant,
        forms: tuple[str, ...] = DEFAULT_FORMS,
        limit: int = 2,
    ) -> list[FilingRef]:
        """Most recent `limit` filings of each requested form type."""
        payload = self.submissions(registrant)
        recent = payload.get("filings", {}).get("recent", {})
        forms_list = recent.get("form", [])
        refs: list[FilingRef] = []

        per_form: dict[str, int] = {}
        for index, form in enumerate(forms_list):
            if form not in forms:
                continue
            if per_form.get(form, 0) >= limit:
                continue
            try:
                refs.append(
                    FilingRef(
                        registrant=registrant,
                        accession=str(recent["accessionNumber"][index]),
                        form=form,
                        filed_at=str(recent["filingDate"][index]),
                        period_of_report=str(recent["reportDate"][index]),
                        primary_document=str(recent["primaryDocument"][index]),
                    )
                )
                per_form[form] = per_form.get(form, 0) + 1
            except (KeyError, IndexError):
                # A short or ragged `recent` block should not abort the whole corpus.
                continue
        return refs

    def document(self, filing: FilingRef) -> str:
        """Filing HTML."""
        result = self.fetch(filing.url, headers=HTML_HEADERS)
        return result.text

    def company_facts(self, registrant: Registrant) -> dict:
        """XBRL facts. Deferred from the critical path (docs/00-corpus.md)."""
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{registrant.cik_padded}.json"
        result = self.fetch(url, headers=JSON_HEADERS)
        try:
            return json.loads(result.text)
        except json.JSONDecodeError as exc:
            raise SecClientError(f"companyfacts for {registrant.ticker} was not JSON") from exc


def sec_search_url(query: str, **params: str) -> str:
    """EDGAR full-text search URL, for finding filings by content rather than by issuer."""
    base = "https://efts.sec.gov/LATEST/search-index"
    return f"{base}?{urlencode({'q': query, **params})}"


def sections_of_interest() -> tuple[str, ...]:
    return SECTIONS_OF_INTEREST
