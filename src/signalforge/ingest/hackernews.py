"""Hacker News ingestion via the Algolia API — free, no auth (DESIGN §7).

Two query families, both driven by config (CLAUDE.md §4):

* the **front page**, filtered to `defaults.min_hn_points`;
* one **keyword search** per entry in `hackernews.keywords`, over a 7-day
  window so a missed cron run self-heals on the next one (DESIGN §14).

**Comments are not fetched.** DESIGN §7 scopes comment fetching to top items
only, and that is not Phase 0 (NEVER rule 15).

All queries share `source_id = "hn"` — one key into `sources.yaml`. They
overlap heavily by design (a popular story matches the front page *and* three
keywords); duplicates collapse on `objectID` before returning, so the overlap
costs nothing downstream.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, Final
from urllib.parse import urlencode

from signalforge.config import SourcesConfig
from signalforge.ingest.base import (
    FetchError,
    HttpFetcher,
    IngestError,
    IngestResult,
    truncate_summary,
)
from signalforge.models import Item, SourceType

__all__ = ["HN_API_ROOT", "HN_SOURCE_ID", "HackerNewsIngestor", "build_hackernews_ingestors"]

logger = logging.getLogger(__name__)

HN_API_ROOT: Final = "https://hn.algolia.com/api/v1"
HN_SOURCE_ID: Final = "hn"
_HITS_PER_PAGE: Final = 50
_LOOKBACK_DAYS: Final = 7


def _item_permalink(object_id: str) -> str:
    return f"https://news.ycombinator.com/item?id={object_id}"


def _parse_created_at(hit: dict[str, Any]) -> datetime | None:
    timestamp = hit.get("created_at_i")
    if isinstance(timestamp, int):
        return datetime.fromtimestamp(timestamp, tz=UTC)
    raw = hit.get("created_at")
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


class HackerNewsIngestor:
    """Ingests HN stories matching the config's front-page floor and keywords."""

    source_type = SourceType.HN
    source_id = HN_SOURCE_ID

    def __init__(
        self,
        *,
        keywords: list[str],
        min_points: int,
        max_summary_chars: int,
        api_root: str = HN_API_ROOT,
        now: datetime | None = None,
    ) -> None:
        self.keywords = keywords
        self.min_points = min_points
        self.max_summary_chars = max_summary_chars
        self._api_root = api_root.rstrip("/")
        self._now = now

    def _queries(self) -> list[tuple[str, str]]:
        """Return `(label, url)` pairs. Labels key per-query conditional-GET state."""
        since = int(((self._now or datetime.now(UTC)) - timedelta(days=_LOOKBACK_DAYS)).timestamp())
        queries: list[tuple[str, str]] = [
            (
                "front_page",
                f"{self._api_root}/search?"
                + urlencode({"tags": "front_page", "hitsPerPage": _HITS_PER_PAGE}),
            )
        ]
        for keyword in self.keywords:
            queries.append(
                (
                    f"keyword:{keyword}",
                    f"{self._api_root}/search?"
                    + urlencode(
                        {
                            "query": keyword,
                            "tags": "story",
                            "numericFilters": f"points>={self.min_points},created_at_i>{since}",
                            "hitsPerPage": _HITS_PER_PAGE,
                        }
                    ),
                )
            )
        return queries

    async def ingest(self, fetcher: HttpFetcher) -> IngestResult:
        """Run every query, collecting items and per-query errors.

        A failing keyword query does not discard the front page's items: each
        query contributes independently (CLAUDE.md §7).
        """
        items: list[Item] = []
        errors: list[IngestError] = []
        seen: set[str] = set()

        for label, url in self._queries():
            try:
                hits, raw_path = await self._fetch_hits(fetcher, url, label)
            except Exception as exc:  # noqa: BLE001 - one query never kills the run
                logger.warning(
                    "hn query failed",
                    extra={"source_id": self.source_id, "query": label, "error": str(exc)},
                )
                errors.append(
                    IngestError.from_exception(
                        exc, source_id=self.source_id, source_type=self.source_type
                    )
                )
                continue
            if hits is None:
                logger.debug(
                    "hn query unchanged since last run",
                    extra={"source_id": self.source_id, "query": label},
                )
                continue
            items.extend(self._hits_to_items(hits, raw_path=raw_path, seen=seen))

        logger.info(
            "hn ingested",
            extra={
                "source_id": self.source_id,
                "item_count": len(items),
                "error_count": len(errors),
            },
        )
        return IngestResult(items=items, errors=errors)

    async def _fetch_hits(
        self, fetcher: HttpFetcher, url: str, label: str
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        response = await fetcher.get(
            url, source_id=self.source_id, cache_key=f"{self.source_id}:{label}"
        )
        if response is None:
            return None, None
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise FetchError(f"invalid JSON from HN Algolia: {exc}", url=url) from exc
        if not isinstance(payload, dict):
            raise FetchError("expected a JSON object from HN Algolia", url=url)
        hits = payload.get("hits")
        if not isinstance(hits, list):
            raise FetchError("HN Algolia response has no 'hits' array", url=url)
        return [hit for hit in hits if isinstance(hit, dict)], response.raw_path

    def _hits_to_items(
        self,
        hits: Iterable[dict[str, Any]],
        *,
        raw_path: str | None,
        seen: set[str],
    ) -> list[Item]:
        """Normalize hits, applying the points floor and collapsing duplicates.

        The floor is applied here as well as in the keyword query's
        `numericFilters` because the `front_page` tag does not accept one — the
        API returns the whole front page and we filter it ourselves.
        """
        items: list[Item] = []
        for hit in hits:
            object_id = hit.get("objectID")
            title = hit.get("title")
            if not isinstance(object_id, str) or not object_id:
                continue
            if not isinstance(title, str) or not title.strip():
                # Comments have no title; a titleless hit is not a story.
                continue
            if object_id in seen:
                continue

            points = hit.get("points")
            if not isinstance(points, int) or points < self.min_points:
                continue
            seen.add(object_id)

            story_url = hit.get("url")
            # Ask HN / Show HN text posts carry no external URL; the HN
            # permalink *is* the document in that case.
            url = (
                story_url.strip()
                if isinstance(story_url, str) and story_url.strip()
                else _item_permalink(object_id)
            )
            author = hit.get("author")
            items.append(
                Item(
                    source_id=self.source_id,
                    source_type=SourceType.HN,
                    external_id=object_id,
                    url=url,
                    title=title.strip(),
                    author=author if isinstance(author, str) and author else None,
                    published_at=_parse_created_at(hit),
                    summary=truncate_summary(
                        hit.get("story_text"), max_chars=self.max_summary_chars
                    ),
                    raw_path=raw_path,
                )
            )
        return items


def build_hackernews_ingestors(config: SourcesConfig) -> list[HackerNewsIngestor]:
    """Zero or one HN ingestor, depending on whether `hackernews:` is configured."""
    if config.hackernews is None:
        return []
    return [
        HackerNewsIngestor(
            keywords=list(config.hackernews.keywords),
            min_points=config.defaults.min_hn_points,
            max_summary_chars=config.defaults.max_summary_chars,
        )
    ]
