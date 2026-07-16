"""RSS/Atom ingestion via `feedparser` — blogs and personal sites (DESIGN §7).

Parsing is deterministic (CLAUDE.md §2): feeds are messy, but a messy feed is a
parsing problem, and parsing problems are never solved by asking an LLM
(NEVER rule 3).

Robustness is the whole job here. `feedparser` sets `bozo` on malformed XML but
still returns whatever entries it recovered, so a partial feed yields its good
entries and a warning rather than an exception. Entries missing a link or title
are skipped individually — one bad entry never costs the other thirty.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

import feedparser

from signalforge.config import RssSource, SourcesConfig
from signalforge.ingest.base import HttpFetcher, IngestError, IngestResult, truncate_summary
from signalforge.models import Item, SourceType

__all__ = ["RssIngestor", "build_rss_ingestors", "parse_feed"]

logger = logging.getLogger(__name__)


class _TextExtractor(HTMLParser):
    """Strips tags from a feed summary, keeping the text.

    Feed summaries are HTML. Storing the markup would spend triage tokens on
    `<div class=...>` (CLAUDE.md §6) and make `content_hash` sensitive to
    cosmetic template churn rather than to the text.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def _strip_html(value: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(value)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup must never fail an item
        return " ".join(unescape(value).split())
    return " ".join(parser.text().split())


def _clean_summary(raw: object, *, max_chars: int) -> str | None:
    """Feed summaries are HTML: strip the markup, then apply the configured ceiling."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    return truncate_summary(_strip_html(raw), max_chars=max_chars)


def _struct_time_to_datetime(value: object) -> datetime | None:
    """Convert feedparser's `*_parsed` struct_time (always UTC) to a datetime."""
    if not isinstance(value, time.struct_time):
        return None
    try:
        return datetime(*value[:6], tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _entry_published(entry: Any) -> datetime | None:
    """Prefer the publication date; fall back to the update date.

    A feed that only carries `updated` is common enough that dropping the date
    entirely would lose ordering for those sources.
    """
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        when = _struct_time_to_datetime(entry.get(key))
        if when is not None:
            return when
    return None


def _entry_author(entry: Any, feed: Any) -> str | None:
    for candidate in (entry.get("author"), feed.get("author")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _usable_link(entry: Any) -> str | None:
    """Return the entry's web link, or None if it hasn't got a real one.

    The scheme check is not paranoia. When an entry carries no `<link>`,
    feedparser falls back to its `<id>` — and Atom ids are conventionally
    `tag:` URIs, not URLs. Such a value would sail through canonicalization and
    occupy a UNIQUE `canonical_url` slot with a key that resolves to nothing,
    permanently shadowing the real post if it later appears with a proper link.
    Requiring http(s) also drops `javascript:`/`data:` links.
    """
    link = entry.get("link")
    if not isinstance(link, str) or not link.strip():
        return None
    candidate = link.strip()
    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    return candidate


def _entry_summary(entry: Any, *, max_chars: int) -> str | None:
    for key in ("summary", "description"):
        cleaned = _clean_summary(entry.get(key), max_chars=max_chars)
        if cleaned is not None:
            return cleaned
    contents = entry.get("content")
    if isinstance(contents, list) and contents:
        first = contents[0]
        if isinstance(first, dict):
            return _clean_summary(first.get("value"), max_chars=max_chars)
    return None


def parse_feed(
    content: bytes,
    *,
    source_id: str,
    max_summary_chars: int,
    raw_path: str | None = None,
    fetched_at: datetime | None = None,
) -> list[Item]:
    """Parse feed bytes into normalized `Item`s. Pure — no I/O, no network.

    Split out from the ingestor so the normalizer is testable against captured
    payloads on its own (CLAUDE.md §8) and so an archived payload in
    `data/http_cache/` can be replayed through a fixed parser without refetching.

    Entries lacking a usable link or title are skipped with a warning; duplicate
    guids within one feed collapse to the first occurrence, so a feed that
    repeats an entry cannot produce two rows racing for one unique key.
    """
    parsed = feedparser.parse(content)
    feed = parsed.get("feed", {})

    if parsed.get("bozo"):
        # Not fatal: feedparser returns the entries it recovered. A feed that is
        # *entirely* unparseable simply yields zero entries, which the caller
        # reports as an empty fetch rather than a crash.
        logger.warning(
            "malformed feed; parsing recovered entries only",
            extra={
                "source_id": source_id,
                "error": str(parsed.get("bozo_exception", "")),
                "entry_count": len(parsed.get("entries", [])),
            },
        )

    stamp = fetched_at or datetime.now(UTC)
    items: list[Item] = []
    seen: set[str] = set()

    for entry in parsed.get("entries", []):
        title = entry.get("title")
        link = _usable_link(entry)
        if link is None:
            logger.warning(
                "skipping feed entry with no usable http(s) link",
                extra={"source_id": source_id, "title": entry.get("title")},
            )
            continue
        if not isinstance(title, str) or not title.strip():
            logger.warning(
                "skipping feed entry with no title",
                extra={"source_id": source_id, "url": link},
            )
            continue

        guid = entry.get("id")
        external_id = guid.strip() if isinstance(guid, str) and guid.strip() else link
        if external_id in seen:
            logger.debug(
                "skipping duplicate entry within feed",
                extra={"source_id": source_id, "external_id": external_id},
            )
            continue
        seen.add(external_id)

        try:
            items.append(
                Item(
                    source_id=source_id,
                    source_type=SourceType.RSS,
                    external_id=external_id,
                    url=link,
                    title=title.strip(),
                    author=_entry_author(entry, feed),
                    published_at=_entry_published(entry),
                    fetched_at=stamp,
                    summary=_entry_summary(entry, max_chars=max_summary_chars),
                    raw_path=raw_path,
                )
            )
        except ValueError as exc:
            # A single unnormalizable entry (e.g. an empty/relative URL that
            # canonicalization rejects) is skipped, not fatal.
            logger.warning(
                "skipping unnormalizable feed entry",
                extra={"source_id": source_id, "url": link, "error": str(exc)},
            )

    return items


class RssIngestor:
    """Ingests one feed listed under `rss:` in `sources.yaml`."""

    source_type = SourceType.RSS

    def __init__(self, source: RssSource, *, max_summary_chars: int) -> None:
        self.source = source
        self.source_id = source.id
        self.max_summary_chars = max_summary_chars

    async def ingest(self, fetcher: HttpFetcher) -> IngestResult:
        """Conditionally GET the feed and normalize it.

        Returns an empty result on 304 — the common daily case (DESIGN §7).
        """
        try:
            response = await fetcher.get(self.source.url, source_id=self.source_id)
        except Exception as exc:  # noqa: BLE001 - one feed never kills the run (§7)
            logger.warning(
                "rss fetch failed",
                extra={"source_id": self.source_id, "url": self.source.url, "error": str(exc)},
            )
            return IngestResult(
                errors=[
                    IngestError.from_exception(
                        exc, source_id=self.source_id, source_type=self.source_type
                    )
                ]
            )

        if response is None:
            logger.debug("feed unchanged since last run", extra={"source_id": self.source_id})
            return IngestResult()

        items = parse_feed(
            response.content,
            source_id=self.source_id,
            max_summary_chars=self.max_summary_chars,
            raw_path=response.raw_path,
        )
        logger.info(
            "rss ingested",
            extra={"source_id": self.source_id, "item_count": len(items)},
        )
        return IngestResult(items=items)


def build_rss_ingestors(config: SourcesConfig) -> list[RssIngestor]:
    """One ingestor per feed in `sources.yaml`. Sources are config, not code."""
    return [
        RssIngestor(source, max_summary_chars=config.defaults.max_summary_chars)
        for source in config.rss
    ]
