"""Ingestion — per-source adapters producing normalized `Item`s (DESIGN §3/§7).

The public entry point is `ingest_all`: give it a validated `SourcesConfig` and
it returns every item Phase 0's sources yielded plus a structured error record
for each source that failed. It does not touch the database — persisting items
(`db.upsert_item`) and closing the run (`db.finish_run(errors=...)`) is the
CLI's job, which keeps this package free of DB state and trivially testable.

Because persistence happens outside this package, the returned `IngestRun`
carries conditional-GET validators that are **staged, not durable**. The caller
calls `run.commit_validators()` once its writes have succeeded; until then the
next run refetches rather than 304s. This is what keeps ingestion at-least-once
across a crash without `ingest/` ever importing `db.py` (CLAUDE.md §2).

Phase 0 sources only: RSS, GitHub releases, Hacker News. arXiv and awesome-list
diffing are Phase 1; YouTube and newsletters are Phase 3 (NEVER rule 15).

Nothing here imports `llm.py` or calls an LLM (CLAUDE.md §2).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from signalforge.config import SourcesConfig, get_secret
from signalforge.ingest.base import (
    DEFAULT_MAX_CONCURRENCY,
    FetchError,
    FetchResponse,
    HttpFetcher,
    IngestError,
    Ingestor,
    IngestResult,
    IngestRun,
    ValidatorStore,
    filter_by_age,
    run_ingestors,
)
from signalforge.ingest.github import GithubReleasesIngestor, build_github_ingestors
from signalforge.ingest.hackernews import HackerNewsIngestor, build_hackernews_ingestors
from signalforge.ingest.rss import RssIngestor, build_rss_ingestors, parse_feed

__all__ = [
    "FetchError",
    "FetchResponse",
    "GithubReleasesIngestor",
    "HackerNewsIngestor",
    "HttpFetcher",
    "IngestError",
    "IngestResult",
    "IngestRun",
    "Ingestor",
    "RssIngestor",
    "ValidatorStore",
    "build_ingestors",
    "ingest_all",
    "parse_feed",
    "run_ingestors",
]

logger = logging.getLogger(__name__)


def build_ingestors(config: SourcesConfig) -> list[Ingestor]:
    """Construct every Phase 0 ingestor from `sources.yaml`.

    The GitHub token is resolved here, once, from the env var *named* by the
    config (`github.token_env`) — never from YAML and never logged (NEVER rule
    16). A missing token is a warning, not an error: GitHub serves 60 req/hr
    unauthenticated, which is enough for a daily run over a handful of repos.
    """
    ingestors: list[Ingestor] = []
    ingestors.extend(build_rss_ingestors(config))

    if config.github is not None and config.github.releases:
        secret = get_secret(config.github.token_env)
        if secret is None:
            logger.warning(
                "github token not set; falling back to unauthenticated rate limits",
                extra={"env_var": config.github.token_env},
            )
        token = secret.get_secret_value() if secret is not None else None
        ingestors.extend(build_github_ingestors(config, token))

    ingestors.extend(build_hackernews_ingestors(config))
    return ingestors


async def ingest_all(
    config: SourcesConfig,
    *,
    cache_dir: Path,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    now: datetime | None = None,
) -> IngestRun:
    """Run every configured Phase 0 ingestor; return all items and all errors.

    Never raises on a source failure — a dead feed becomes an `IngestError` on
    the run while every other source's items come back normally (CLAUDE.md §7).

    Items older than `defaults.max_item_age_days` (measured from `now`, which
    defaults to the current time and exists as a parameter only so tests can
    freeze it) are dropped here — before the caller can upsert them — so a first
    run or a new source never backfills feed history into the DB or triage.
    Items with no parseable published date pass through untouched; see
    `filter_by_age`.

    **The caller must call `run.commit_validators()` after persisting**, or every
    source refetches unconditionally next run. Nothing is lost either way; the
    unconfirmed path just costs bandwidth. Typical use::

        run = await ingest_all(config, cache_dir=Path("data/http_cache"))
        items_new = sum(upsert_item(conn, item)[1] for item in run.items)
        run.commit_validators()          # only now are the 304s earned
        finish_run(conn, run_id, status=..., finished_at=...,
                   items_new=items_new, errors=run.error_records())
    """
    ingestors = build_ingestors(config)
    logger.info("starting ingest", extra={"ingestor_count": len(ingestors)})
    async with HttpFetcher(
        cache_dir=cache_dir,
        timeout=config.defaults.fetch_timeout,
        max_concurrency=max_concurrency,
    ) as fetcher:
        result = await run_ingestors(ingestors, fetcher)
        validators = fetcher.validators
    items = filter_by_age(
        result.items,
        max_age_days=config.defaults.max_item_age_days,
        now=now,
    )
    logger.info(
        "ingest complete",
        extra={
            "item_count": len(items),
            "skipped_too_old": len(result.items) - len(items),
            "error_count": len(result.errors),
            "pending_validators": validators.pending_count(),
        },
    )
    return IngestRun(items=items, errors=result.errors, validators=validators)
