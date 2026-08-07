"""arXiv ingestion via the public Atom API (DESIGN §7, Phase 1).

`arxiv.categories` and `arxiv.require_keywords` fold into a single
`search_query` — `(cat:A OR cat:B) AND (abs:kw1 OR abs:"multi word kw")` — so
one run's own logic issues exactly **one** HTTP request, sorted by submission
date descending. arXiv's own guidance caps polite programmatic use at one
request every 3 seconds; this ingestor's request loop never approaches that,
because it never issues a second request itself. `ingest/base.py`'s shared
`HttpFetcher` can still retry *this* request once on a 429/5xx/transport
error, ~1s later (`_wait_honoring_retry_after`'s exponential floor) — a real
second request this module does not control, and the reason there is still no
explicit 3s floor coded here: the shared retry path is bounded (3 attempts)
and only fires when arXiv itself signalled trouble.

The response is Atom — the same wire format `ingest/rss.py` already parses —
so this reuses `feedparser` rather than a bespoke XML client.

Only titles and abstracts are stored (DESIGN §7's "abstracts only at triage" —
CLAUDE.md §6, NEVER rule 9); full paper text is never fetched.

**A malformed query is not an HTTP error.** The arXiv API answers a bad
`search_query` with `200 OK` and a single synthetic entry whose `id` starts
`http://arxiv.org/api/errors#...` instead of `.../abs/...`. `parse_arxiv_feed`
detects that prefix explicitly and reports it via `ArxivParseResult.error_message`;
`ArxivIngestor.ingest` turns that into an `IngestError` and invalidates the
response's conditional-GET validator so the same bad config keeps erroring
loudly every run instead of 304ing into silence. A misconfigured
`require_keywords` — including one a curation proposal wrote, unreviewed
until the human tick — must show up in `runs.errors` and the next digest's
error footer, not degrade to a permanently quiet zero-item source (CLAUDE.md
§7: "the reports are the monitoring channel").
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urlencode

import feedparser

from signalforge.config import SourcesConfig
from signalforge.ingest.base import HttpFetcher, IngestError, IngestResult, truncate_summary
from signalforge.models import Item, SourceType

__all__ = [
    "ARXIV_API_ROOT",
    "ARXIV_SOURCE_ID",
    "ArxivIngestor",
    "ArxivParseResult",
    "build_arxiv_ingestors",
    "build_search_query",
    "parse_arxiv_feed",
]

logger = logging.getLogger(__name__)

ARXIV_API_ROOT: Final = "https://export.arxiv.org/api/query"
ARXIV_SOURCE_ID: Final = "arxiv"
_MAX_RESULTS: Final = 50
"""Ingestors look back, not just at the newest entry (DESIGN §14: a missed run
self-heals on the next one). 50 results sorted by submission date comfortably
covers a week across a handful of categories, and dedup makes the overlap free."""

_VERSION_SUFFIX_RE: Final = re.compile(r"v\d+$")


def _quote_term(keyword: str) -> str:
    """A multi-word keyword must search as a phrase, not an implicit AND of
    its words — `abs:tool use` would match "use" anywhere in the abstract."""
    stripped = keyword.strip()
    return f'"{stripped}"' if " " in stripped else stripped


def build_search_query(categories: list[str], require_keywords: list[str]) -> str:
    """The arXiv API `search_query` value for `categories` AND (any keyword).

    Public so a test can see exactly what will be queried without duplicating
    this join logic. Keywords are optional — an `arxiv:` block with categories
    but no `require_keywords` queries the categories alone.
    """
    cat_clause = " OR ".join(f"cat:{category}" for category in categories)
    if not require_keywords:
        return cat_clause
    keyword_clause = " OR ".join(f"abs:{_quote_term(keyword)}" for keyword in require_keywords)
    if not cat_clause:
        return keyword_clause
    return f"({cat_clause}) AND ({keyword_clause})"


def _parse_arxiv_id(entry_id: object) -> str | None:
    """`.../abs/2507.12345v1` -> `2507.12345`.

    The version suffix is stripped so a metadata-only revision upserts onto
    the same row (`UNIQUE(source_id, external_id)`) instead of duplicating it,
    and the item's `url` always points at the latest version. Returns None for
    anything without an `/abs/` segment — including arXiv's own synthetic
    error entries (see module docstring).
    """
    if not isinstance(entry_id, str) or not entry_id:
        return None
    _, _, raw_id = entry_id.partition("/abs/")
    raw_id = raw_id.strip()
    if not raw_id:
        return None
    return _VERSION_SUFFIX_RE.sub("", raw_id)


def _struct_time_to_datetime(value: object) -> datetime | None:
    if not isinstance(value, time.struct_time):
        return None
    try:
        return datetime(*value[:6], tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _entry_published(entry: Any) -> datetime | None:
    """Prefer the original submission date; fall back to the last-updated date
    for the rare entry missing `published`."""
    for key in ("published_parsed", "updated_parsed"):
        when = _struct_time_to_datetime(entry.get(key))
        if when is not None:
            return when
    return None


def _entry_authors(entry: Any) -> str | None:
    """First author, or `"First et al."` for multi-author papers.

    arXiv papers routinely carry a dozen-plus authors; `Item.author` is a
    one-line attribution for a digest row, not a citation block.
    """
    authors = entry.get("authors")
    names: list[str] = []
    if isinstance(authors, list):
        names = [
            author["name"].strip()
            for author in authors
            if isinstance(author, dict)
            and isinstance(author.get("name"), str)
            and author["name"].strip()
        ]
    if not names:
        single = entry.get("author")
        return single.strip() if isinstance(single, str) and single.strip() else None
    if len(names) == 1:
        return names[0]
    return f"{names[0]} et al."


_ARXIV_ERROR_ID_PREFIX: Final = "http://arxiv.org/api/errors#"
"""How arXiv marks its synthetic "your search_query was malformed" entry — a
`200 OK` carrying exactly one entry with this id prefix, not an HTTP error."""


@dataclass(frozen=True, slots=True)
class ArxivParseResult:
    """What parsing one response yielded: normal items, or arXiv's own
    complaint about the query that produced this response."""

    items: list[Item]
    error_message: str | None = None
    """Set only for arXiv's synthetic error entry (see `_ARXIV_ERROR_ID_PREFIX`).
    `items` is always empty when this is set — that entry is the whole response."""


def parse_arxiv_feed(
    content: bytes,
    *,
    source_id: str,
    max_summary_chars: int,
    raw_path: str | None = None,
) -> ArxivParseResult:
    """Parse the Atom response into normalized `Item`s. Pure — no I/O.

    Entries with an unparseable id or no title are skipped individually — one
    malformed record never costs the rest of the response. Duplicate ids
    within one response collapse to the first occurrence, the same discipline
    `ingest/rss.py::parse_feed` applies to feed guids.
    """
    parsed = feedparser.parse(content)
    if parsed.get("bozo"):
        logger.warning(
            "malformed arxiv response; parsing recovered entries only",
            extra={
                "source_id": source_id,
                "error": str(parsed.get("bozo_exception", "")),
                "entry_count": len(parsed.get("entries", [])),
            },
        )

    items: list[Item] = []
    seen: set[str] = set()
    for entry in parsed.get("entries", []):
        entry_id = entry.get("id")
        if isinstance(entry_id, str) and entry_id.startswith(_ARXIV_ERROR_ID_PREFIX):
            message = entry.get("summary") or entry.get("title") or entry_id
            logger.warning(
                "arxiv rejected the search_query",
                extra={"source_id": source_id, "arxiv_message": message},
            )
            return ArxivParseResult(items=[], error_message=str(message).strip())

        arxiv_id = _parse_arxiv_id(entry_id)
        if arxiv_id is None:
            logger.warning(
                "skipping arxiv entry with no parseable id", extra={"source_id": source_id}
            )
            continue
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            logger.warning(
                "skipping arxiv entry with no title",
                extra={"source_id": source_id, "external_id": arxiv_id},
            )
            continue
        if arxiv_id in seen:
            logger.debug(
                "skipping duplicate entry within response",
                extra={"source_id": source_id, "external_id": arxiv_id},
            )
            continue
        seen.add(arxiv_id)

        try:
            items.append(
                Item(
                    source_id=source_id,
                    source_type=SourceType.ARXIV,
                    external_id=arxiv_id,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    title=title.strip(),
                    author=_entry_authors(entry),
                    published_at=_entry_published(entry),
                    summary=truncate_summary(entry.get("summary"), max_chars=max_summary_chars),
                    raw_path=raw_path,
                )
            )
        except ValueError as exc:
            logger.warning(
                "skipping unnormalizable arxiv entry",
                extra={"source_id": source_id, "external_id": arxiv_id, "error": str(exc)},
            )

    return ArxivParseResult(items=items)


class ArxivIngestor:
    """Ingests one `search_query` built from `arxiv.categories` + `arxiv.require_keywords`.

    One instance covers the whole `arxiv:` block — categories and keywords
    fold into a single API call (see module docstring) rather than one
    ingestor per category.
    """

    source_type = SourceType.ARXIV
    source_id = ARXIV_SOURCE_ID

    def __init__(
        self,
        *,
        categories: list[str],
        require_keywords: list[str],
        max_summary_chars: int,
        max_results: int = _MAX_RESULTS,
        api_root: str = ARXIV_API_ROOT,
    ) -> None:
        self.categories = categories
        self.require_keywords = require_keywords
        self.max_summary_chars = max_summary_chars
        self.max_results = max_results
        self._api_root = api_root.rstrip("/")

    def _url(self) -> str:
        params = {
            "search_query": build_search_query(self.categories, self.require_keywords),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(self.max_results),
        }
        return f"{self._api_root}?{urlencode(params)}"

    async def ingest(self, fetcher: HttpFetcher) -> IngestResult:
        """Fetch and normalize. Any failure is returned as an `IngestError`,
        never raised past here (CLAUDE.md §7)."""
        try:
            response = await fetcher.get(self._url(), source_id=self.source_id)
        except Exception as exc:  # noqa: BLE001 - one source never kills the run (§7)
            logger.warning(
                "arxiv fetch failed", extra={"source_id": self.source_id, "error": str(exc)}
            )
            return IngestResult(
                errors=[
                    IngestError.from_exception(
                        exc, source_id=self.source_id, source_type=self.source_type
                    )
                ]
            )

        if response is None:
            logger.debug(
                "arxiv query unchanged since last run", extra={"source_id": self.source_id}
            )
            return IngestResult()

        parsed = parse_arxiv_feed(
            response.content,
            source_id=self.source_id,
            max_summary_chars=self.max_summary_chars,
            raw_path=response.raw_path,
        )
        if parsed.error_message is not None:
            # A 200 carrying arXiv's own "malformed search_query" complaint
            # still has a perfectly cacheable ETag — invalidated so a bad
            # `categories`/`require_keywords` value keeps erroring loudly
            # every run instead of 304ing into permanent silence (same
            # reasoning as `github.py`'s empty-`/releases` invalidation).
            fetcher.invalidate(self._url(), source_id=self.source_id)
            return IngestResult(
                errors=[
                    IngestError(
                        source_id=self.source_id,
                        source_type=self.source_type,
                        message=parsed.error_message,
                        error_type="ArxivQueryError",
                        occurred_at=datetime.now(UTC),
                    )
                ]
            )

        logger.info(
            "arxiv ingested",
            extra={"source_id": self.source_id, "item_count": len(parsed.items)},
        )
        return IngestResult(items=parsed.items)


def build_arxiv_ingestors(config: SourcesConfig) -> list[ArxivIngestor]:
    """Zero or one arXiv ingestor: nothing to query without at least one category."""
    if config.arxiv is None or not config.arxiv.categories:
        return []
    return [
        ArxivIngestor(
            categories=list(config.arxiv.categories),
            require_keywords=list(config.arxiv.require_keywords),
            max_summary_chars=config.defaults.max_summary_chars,
        )
    ]
