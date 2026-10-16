"""Runtime settings. Every secret comes from the environment; nothing is defaulted to a real value.

A missing required setting raises at first use with the name of the variable to set, rather than
failing later inside an HTTP call with an opaque error.
"""

from __future__ import annotations

import os
from functools import lru_cache


class SettingsError(RuntimeError):
    """A required setting is missing or still holds its placeholder."""


def _required(name: str, *, secret: bool = False) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SettingsError(
            f"{name} is not set. Add it to your .env (see .env.example). "
            f"Export it before running: export {name}=..."
        )
    if secret and value.lower().startswith(("your-", "changeme", "<", "example")):
        raise SettingsError(
            f"{name} still holds its placeholder. Set a real value — a placeholder credential "
            f"reaching EDGAR is how an IP gets blocked."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip() or default


# --- Neo4j ---------------------------------------------------------------------------------

def neo4j_uri() -> str:
    return _optional("NEO4J_URI", "bolt://localhost:7687")


def neo4j_auth() -> tuple[str, str]:
    return _optional("NEO4J_USER", "neo4j"), _required("NEO4J_PASSWORD", secret=True)


# --- Postgres ------------------------------------------------------------------------------

def postgres_dsn() -> str:
    # 5433, not 5432: a Homebrew PostgreSQL owns 5432 on the development machine.
    return _optional("POSTGRES_DSN", "postgresql://kg_rag:kg_rag_dev@localhost:5433/kg_rag")


# --- SEC fair-access policy ------------------------------------------------------------------

def sec_user_agent() -> str:
    """The `User-Agent` SEC requires, including a real contact address.

    The SEC monitors for its absence and blocks non-compliant clients mid-scrape, which turns a
    build into an intermittent failure that looks like a bug in your code.
    """
    return _required("SEC_USER_AGENT", secret=True)


def sec_max_requests_per_second() -> float:
    """SEC fair-access ceiling is 10 requests/second. Default sits under it on purpose."""
    raw = _optional("SEC_MAX_RPS", "5")
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"SEC_MAX_RPS must be a number, got {raw!r}") from exc
    if value <= 0:
        raise SettingsError(f"SEC_MAX_RPS must be positive, got {value}")
    if value > 10:
        # Clamping rather than raising: the ceiling is a hard limit from the SEC, so honouring a
        # higher request is never what the caller meant.
        return 10.0
    return value


# --- Models ----------------------------------------------------------------------------------

def anthropic_api_key() -> str:
    return _required("ANTHROPIC_API_KEY", secret=True)


def extraction_model() -> str:
    return _optional("EXTRACTION_MODEL", "claude-haiku-4-5-20251001")


def router_model() -> str:
    return _optional("ROUTER_MODEL", "claude-haiku-4-5-20251001")


def generation_model() -> str:
    return _optional("GENERATION_MODEL", "claude-sonnet-4-5-20250929")


def use_real_models() -> bool:
    """False selects the deterministic offline extractor.

    The offline path exists so the pipeline is demonstrable and testable with no API key. It is
    not an LLM and every artifact it produces is labelled `stub:offline-extractor`, so a report
    can never be mistaken for a real measurement.
    """
    return _optional("USE_REAL_MODELS", "0") in {"1", "true", "yes", "on"}


def max_ingestion_usd() -> float | None:
    """Hard ceiling for a full-corpus extraction run. `None` disables the check."""
    raw = _optional("MAX_INGESTION_USD", "")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"MAX_INGESTION_USD must be a number, got {raw!r}") from exc
    if value <= 0:
        raise SettingsError(f"MAX_INGESTION_USD must be positive, got {value}")
    return value


# --- Serving ---------------------------------------------------------------------------------

def api_key() -> str:
    """Optional. Empty means the API is unauthenticated, which is fine for localhost only."""
    return _optional("API_KEY", "")


def embedding_dimension() -> int:
    raw = _optional("EMBEDDING_DIM", "384")
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"EMBEDDING_DIM must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise SettingsError(f"EMBEDDING_DIM must be positive, got {value}")
    return value


@lru_cache(maxsize=1)
def corpus_root() -> str:
    """Where downloaded filings are cached on disk."""
    path = os.environ.get("CORPUS_DIR", "").strip() or os.path.join(
        os.path.expanduser("~"), ".cache", "kg_rag", "corpus"
    )
    os.makedirs(path, exist_ok=True)
    return path
