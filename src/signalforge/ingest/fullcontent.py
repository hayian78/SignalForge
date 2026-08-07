"""Deep-read fetch — lazily fetches full article text for top-N survivors (CLAUDE.md §6).

Deterministic, no LLM (CLAUDE.md §2): fetches through `ingest/base.py`'s
`HttpFetcher` and extracts body text with `trafilatura`. `content` sent to
triage stays titles + summaries only (NEVER rule 9) — this module only ever
runs on a small, already-ranked slice a caller selected, never on every item.

Unlike the rest of `ingest/`, this module touches the database directly.
`ingest/__init__.py` and `ingest/base.py` stay free of `db.py` while
*discovering* items — that convention protects the conditional-GET validator
staging/commit dance, not a rule CLAUDE.md §2 states (that rule is only "never
calls an LLM, never imports `llm.py`", which this module also honors). Here
there is nothing to stage: the caller already knows the item's id, and the
write is a single `UPDATE ... WHERE content IS NULL` that is naturally
idempotent on its own (`db.update_item_content`) — there is no atomicity gap
for a validator-staging dance to close.

Per-item failures never abort the run (CLAUDE.md §7): a fetch or extraction
failure simply leaves `content` NULL, and the item degrades to title+summary
in whatever the caller builds next (the podcast script, DESIGN §13.3).

`item.url` is the first URL this function fetches that the operator did not
choose — a feed publisher did, in `sources.yaml`'s entries. Every fetch here
is checked against `ingest.base.is_disallowed_fetch_host` first and made
with `follow_redirects=False`, the same guard `curate/scout.py` and
`ingest/probe.py` use for web-search- and scout-proposed URLs, so a
malicious or compromised feed cannot point an entry at an internal address
and have it fetched, extracted, and rendered into the vault.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urlsplit

import trafilatura

from signalforge.db import update_item_content
from signalforge.ingest.base import FetchError, HttpFetcher, IngestError, is_disallowed_fetch_host
from signalforge.models import Item

__all__ = ["MAX_FULL_CONTENT_CHARS", "FullContentResult", "fetch_full_content"]

logger = logging.getLogger(__name__)

MAX_FULL_CONTENT_CHARS: Final = 25_000
"""Per-item stored `content` ceiling (2026-08-06 measurement: zero real items
clipped at this length). A safety rail, not a tuning knob — the same
reasoning as `deliver.MAX_DELIVERY_AGE_DAYS` (CLAUDE.md §4): nobody should
want to widen it, since every character here is Opus input on the podcast
script call downstream (DESIGN §8)."""


@dataclass(frozen=True, slots=True)
class FullContentResult:
    """What one `fetch_full_content` call achieved."""

    fetched: int = 0
    """Items whose `content` was newly written this run."""

    errors: list[IngestError] = field(default_factory=list)
    """One entry per fetch or extraction *exception* — never per item that
    merely yielded no extractable text, which is a normal outcome (CLAUDE.md
    §7: a broken source must not vanish, but a paywall is not broken)."""


def _extract(content: bytes, url: str, *, floor: int) -> str | None:
    """Body text from a fetched page, truncated at `MAX_FULL_CONTENT_CHARS`.

    None covers every "nothing usable here" case alike — a paywall stub, a
    non-HTML payload, an empty extraction, a teaser shorter than `floor` — so
    the caller cannot tell them apart and does not need to: all four degrade
    the item to title+summary.

    `include_comments=False`: comment threads are unbounded, attacker-authored
    text (the most attacker-controlled text on the open web) that would
    otherwise ride along as if it were the article — paid for as Opus input
    on the Stage 3 script call for no benefit, and a target for the same
    forgery class `models.flatten_to_single_line` guards the vault line
    against, just arriving a stage earlier.

    `floor` rejects an extraction *shorter* than the item's existing
    `summary`: a common paywall shape (a two-paragraph teaser, then a wall) is
    not empty, so it survives `trafilatura.extract` as real-looking text and
    would otherwise permanently shadow a `summary` that already said more —
    `update_item_content`'s `WHERE content IS NULL` guard means a thin extract
    stored once is never revisited.
    """
    text = trafilatura.extract(content, url=url, include_comments=False)
    if text is None:
        return None
    text = text.strip()
    if not text or len(text) < floor:
        return None
    if len(text) > MAX_FULL_CONTENT_CHARS:
        text = text[:MAX_FULL_CONTENT_CHARS].rstrip() + "…"
    return text


async def fetch_full_content(
    fetcher: HttpFetcher,
    conn: sqlite3.Connection,
    items: Sequence[Item],
) -> FullContentResult:
    """Backfill `items.content` for every item in `items` that still lacks it.

    `items` is the caller's already-ranked top-N slice
    (`report.daily.select_digest_items`); this function never selects or
    ranks on its own, and it never fetches an item whose `content` is already
    set — re-running it on a fully-fetched slice performs zero HTTP calls and
    zero writes (CLAUDE.md §3, NEVER rule 4).
    """
    fetched = 0
    errors: list[IngestError] = []
    for item in items:
        if item.content is not None or item.id is None:
            continue

        # `item.url` is the feed's own entry link — a publisher chooses it,
        # not the operator, unlike every other URL `HttpFetcher` fetches
        # elsewhere in `ingest/` (CLAUDE.md §7). This is the same untrusted-
        # URL shape `curate/scout.py` and `ingest/probe.py` guard against for
        # scout-proposed feeds; a malicious or compromised feed pointing an
        # entry at a cloud metadata endpoint or an internal service would
        # otherwise have it fetched, extracted, sent to Opus, and rendered
        # into the vault — a reconnaissance oracle, one stage later than the
        # one `is_disallowed_fetch_host` already exists to stop.
        if is_disallowed_fetch_host(urlsplit(item.url).hostname or ""):
            logger.warning(
                "deep-read fetch skipped: item url resolves to a disallowed host",
                extra={"item_id": item.id, "url": item.url},
            )
            errors.append(
                IngestError(
                    source_id=item.source_id,
                    source_type=item.source_type,
                    message=f"deep-read fetch of item {item.id} ({item.url}): disallowed host",
                    error_type="DisallowedFetchHost",
                    occurred_at=datetime.now(UTC),
                )
            )
            continue

        try:
            # conditional=False: this function only ever fetches a URL whose
            # content is NULL, i.e. one we hold no usable prior body for, so
            # there is nothing a validator could usefully 304 against. Worse,
            # `conditional=True` would *stage* one on every 200 (`base.py`'s
            # `ValidatorStore.stage`) — and if this fetcher is later reused
            # and committed by a caller (e.g. the same run's ingest fetcher),
            # an article whose extraction failed once could only ever 304 on
            # retry, permanently degrading it with no error to show for it.
            #
            # follow_redirects=False for the same reason `ingest/probe.py`
            # disables it for a scout-proposed feed: `is_disallowed_fetch_host`
            # only checked `item.url` as given, and a public, allowed host
            # that 302s to a private one would otherwise reach exactly the
            # destination that check exists to stop. A redirect response
            # comes back as-is here rather than followed; `_extract` finds
            # nothing extractable in a redirect body and the item simply
            # degrades to title+summary, same as any other unfetchable page.
            response = await fetcher.get(
                item.url, source_id=item.source_id, conditional=False, follow_redirects=False
            )
        except FetchError as exc:
            logger.warning(
                "full-content fetch failed",
                extra={"item_id": item.id, "url": item.url, "error": str(exc)},
            )
            errors.append(
                IngestError(
                    source_id=item.source_id,
                    source_type=item.source_type,
                    message=f"deep-read fetch of item {item.id} ({item.url}): {exc}",
                    error_type=exc.__class__.__name__,
                    occurred_at=datetime.now(UTC),
                )
            )
            continue
        if response is None:
            # `conditional=False` means we never sent validators, so a
            # spec-conforming server has no basis to answer 304 — but nothing
            # stops a nonconforming one. There is nothing to extract either
            # way; leave content NULL and move on.
            continue

        try:
            text = _extract(response.content, response.url, floor=len(item.summary or ""))
        except Exception as exc:  # noqa: BLE001 - trafilatura's failure modes aren't part of its contract
            logger.warning(
                "full-content extraction raised; leaving content unset",
                extra={"item_id": item.id, "url": item.url, "error": str(exc)},
            )
            errors.append(
                IngestError(
                    source_id=item.source_id,
                    source_type=item.source_type,
                    message=f"deep-read extraction of item {item.id} ({item.url}): {exc}",
                    error_type=exc.__class__.__name__,
                    occurred_at=datetime.now(UTC),
                )
            )
            continue

        if text is None:
            logger.debug("no extractable content", extra={"item_id": item.id, "url": item.url})
            continue

        if update_item_content(conn, item.id, text):
            fetched += 1

    return FullContentResult(fetched=fetched, errors=errors)
