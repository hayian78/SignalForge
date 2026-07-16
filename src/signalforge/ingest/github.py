"""GitHub releases ingestion — REST `/releases` with a `/tags` fallback (DESIGN §7).

Release notes become `Item.summary`, which is what triage reads. Phase 0 covers
`github.releases` only: awesome-list diffing is Phase 1 and star velocity /
issues are Phase 2, so neither is implemented here (NEVER rule 15).

**Token handling** (NEVER rule 16): the PAT is read via `config.get_secret` from
the env var *named* by `sources.yaml`, held as a `SecretStr`, and only ever
unwrapped into an `Authorization` header. It is never logged, never put in an
error message, and never written to the payload archive path. Unauthenticated
still works (60 req/hr vs 5000), so a missing token is a warning, not a failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from signalforge.config import SourcesConfig
from signalforge.ingest.base import (
    FetchError,
    HttpFetcher,
    IngestError,
    IngestResult,
    truncate_summary,
)
from signalforge.models import Item, SourceType

__all__ = ["GITHUB_API_ROOT", "GithubReleasesIngestor", "build_github_ingestors"]

logger = logging.getLogger(__name__)

GITHUB_API_ROOT: Final = "https://api.github.com"
_PER_PAGE: Final = 10
"""Ingestors look back, not just at the last item (DESIGN §14: a missed run
self-heals on the next one). Ten releases covers a week for even the busiest
repo, and dedup makes the overlap free."""

_ACCEPT: Final = "application/vnd.github+json"
_API_VERSION: Final = "2022-11-28"


@dataclass(frozen=True, slots=True)
class _ReleasesOutcome:
    """What the `/releases` probe learned.

    `publishes_releases` is deliberately separate from `items` being non-empty:
    a payload of nothing but drafts is still a repo that publishes releases, and
    must not be re-routed to the `/tags` fallback.
    """

    publishes_releases: bool
    items: list[Item]


_RELEASES_UNCHANGED: Final = _ReleasesOutcome(publishes_releases=True, items=[])
"""Sentinel: the `/releases` payload 304'd, so there is nothing new to ingest."""


def _parse_timestamp(value: object) -> datetime | None:
    """Parse GitHub's ISO 8601 `...Z` timestamps."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _log_rate_limit(source_id: str, headers: dict[str, str]) -> None:
    """Surface rate-limit state without ever touching the token itself."""
    remaining = headers.get("x-ratelimit-remaining")
    if remaining is None:
        return
    try:
        left = int(remaining)
    except ValueError:
        return
    if left <= 10:
        logger.warning(
            "github rate limit nearly exhausted",
            extra={
                "source_id": source_id,
                "remaining": left,
                "reset_at": headers.get("x-ratelimit-reset"),
            },
        )


class GithubReleasesIngestor:
    """Ingests releases for one `owner/repo` slug from `sources.yaml`."""

    source_type = SourceType.GITHUB

    def __init__(
        self,
        repo: str,
        token: str | None = None,
        *,
        max_summary_chars: int,
        api_root: str = GITHUB_API_ROOT,
    ) -> None:
        self.repo = repo
        self.source_id = repo
        """The `owner/repo` slug is literally the key in `sources.yaml`."""
        self.max_summary_chars = max_summary_chars
        self._token = token
        self._api_root = api_root.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"accept": _ACCEPT, "x-github-api-version": _API_VERSION}
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        return headers

    async def ingest(self, fetcher: HttpFetcher) -> IngestResult:
        """Fetch `/releases`, falling back to `/tags` for repos that only tag.

        Any failure is returned as an `IngestError`, never raised past here.
        """
        try:
            outcome = await self._fetch_releases(fetcher)
            if outcome is _RELEASES_UNCHANGED:
                # A 304 is only reachable when the repo publishes releases; an
                # empty payload is never cached (see `_fetch_releases`), so this
                # cannot mean "the tags fallback got skipped".
                return IngestResult()
            if outcome.publishes_releases:
                return IngestResult(items=outcome.items)
            return await self._fetch_tags(fetcher)
        except FetchError as exc:
            _log_rate_limit(self.source_id, dict(exc.headers))
            logger.warning(
                "github fetch failed",
                extra={
                    "source_id": self.source_id,
                    "status_code": exc.status_code,
                    "error": str(exc),
                },
            )
            return IngestResult(
                errors=[
                    IngestError.from_exception(
                        exc, source_id=self.source_id, source_type=self.source_type
                    )
                ]
            )
        except Exception as exc:  # noqa: BLE001 - one repo never kills the run (§7)
            logger.warning(
                "github ingest failed",
                extra={"source_id": self.source_id, "error": str(exc)},
            )
            return IngestResult(
                errors=[
                    IngestError.from_exception(
                        exc, source_id=self.source_id, source_type=self.source_type
                    )
                ]
            )

    async def _fetch_json(
        self, fetcher: HttpFetcher, url: str
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Return `(payload, raw_path)`; payload is None on a 304."""
        response = await fetcher.get(url, source_id=self.source_id, headers=self._headers())
        if response is None:
            return None, None
        _log_rate_limit(self.source_id, dict(response.headers))
        try:
            decoded = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise FetchError(f"invalid JSON from GitHub: {exc}", url=url) from exc
        if not isinstance(decoded, list):
            raise FetchError(
                f"expected a JSON array from GitHub, got {type(decoded).__name__}", url=url
            )
        return [entry for entry in decoded if isinstance(entry, dict)], response.raw_path

    async def _fetch_releases(self, fetcher: HttpFetcher) -> _ReleasesOutcome:
        """Probe `/releases`, distinguishing "unchanged" from "this repo has none".

        `publishes_releases=False` routes the caller to `/tags`. Critically, when
        the payload is empty we *invalidate* the cached validators first: an
        empty `/releases` array carries a stable ETag, so caching it would 304
        on every subsequent run, short-circuit the fallback, and silently strand
        a tags-only repo (llama.cpp is exactly this shape) with no error to show
        for it. Refetching one empty array a day is the correct price.
        """
        url = f"{self._api_root}/repos/{self.repo}/releases?per_page={_PER_PAGE}"
        try:
            payload, raw_path = await self._fetch_json(fetcher, url)
        except FetchError as exc:
            if exc.status_code == 404:
                # Repos with releases disabled 404 here; tags still exist.
                logger.debug(
                    "no releases endpoint; trying tags", extra={"source_id": self.source_id}
                )
                return _ReleasesOutcome(publishes_releases=False, items=[])
            raise
        if payload is None:
            logger.debug("releases unchanged since last run", extra={"source_id": self.source_id})
            return _RELEASES_UNCHANGED
        if not payload:
            fetcher.invalidate(url, source_id=self.source_id)
            logger.debug(
                "repo publishes no releases; trying tags", extra={"source_id": self.source_id}
            )
            return _ReleasesOutcome(publishes_releases=False, items=[])
        items = self._releases_to_items(payload, raw_path)
        logger.info(
            "github releases ingested",
            extra={"source_id": self.source_id, "item_count": len(items)},
        )
        return _ReleasesOutcome(publishes_releases=True, items=items)

    def _releases_to_items(self, payload: list[dict[str, Any]], raw_path: str | None) -> list[Item]:
        items: list[Item] = []
        seen: set[str] = set()
        for release in payload:
            if release.get("draft"):
                continue
            tag = release.get("tag_name")
            url = release.get("html_url")
            if not isinstance(tag, str) or not tag or not isinstance(url, str) or not url:
                logger.warning("skipping malformed release", extra={"source_id": self.source_id})
                continue
            external_id = f"{self.repo}@{tag}"
            if external_id in seen:
                continue
            seen.add(external_id)

            name = release.get("name")
            title = name.strip() if isinstance(name, str) and name.strip() else tag
            author = release.get("author")
            items.append(
                Item(
                    source_id=self.source_id,
                    source_type=SourceType.GITHUB,
                    external_id=external_id,
                    url=url,
                    title=f"{self.repo} {title}",
                    author=author.get("login") if isinstance(author, dict) else None,
                    published_at=_parse_timestamp(release.get("published_at"))
                    or _parse_timestamp(release.get("created_at")),
                    # Release notes are markdown: keep the line structure that
                    # makes a changelog readable rather than flattening it.
                    summary=truncate_summary(
                        release.get("body"),
                        max_chars=self.max_summary_chars,
                        collapse_whitespace=False,
                    ),
                    raw_path=raw_path,
                )
            )
        return items

    async def _fetch_tags(self, fetcher: HttpFetcher) -> IngestResult:
        """Fallback for repos that tag but never cut a release (DESIGN §7).

        A tag carries no notes and no date, so these items are title-only. That
        is honest: the item says "this tag exists", and scoring can judge it on
        that basis rather than on a summary we invented.
        """
        url = f"{self._api_root}/repos/{self.repo}/tags?per_page={_PER_PAGE}"
        payload, raw_path = await self._fetch_json(fetcher, url)
        if payload is None:
            return IngestResult()

        items: list[Item] = []
        seen: set[str] = set()
        for tag in payload:
            name = tag.get("name")
            if not isinstance(name, str) or not name:
                continue
            external_id = f"{self.repo}@{name}"
            if external_id in seen:
                continue
            seen.add(external_id)
            items.append(
                Item(
                    source_id=self.source_id,
                    source_type=SourceType.GITHUB,
                    external_id=external_id,
                    url=f"https://github.com/{self.repo}/releases/tag/{name}",
                    title=f"{self.repo} {name}",
                    raw_path=raw_path,
                )
            )
        logger.info(
            "github tags ingested (releases fallback)",
            extra={"source_id": self.source_id, "item_count": len(items)},
        )
        return IngestResult(items=items)


def build_github_ingestors(
    config: SourcesConfig, token: str | None = None
) -> list[GithubReleasesIngestor]:
    """One ingestor per slug in `github.releases`.

    `awesome_lists` is deliberately ignored — that is Phase 1 (DESIGN §7/§16).
    """
    if config.github is None:
        return []
    return [
        GithubReleasesIngestor(repo, token, max_summary_chars=config.defaults.max_summary_chars)
        for repo in config.github.releases
    ]
