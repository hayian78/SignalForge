"""Tests for `ingest/fullcontent.py` — the deep-read backfill (CLAUDE.md §6).

The two gates Stage 2 of `docs/plans/podcast-channel.md` rests on: a second run
over already-fetched items performs zero HTTP calls and zero row deltas
(CLAUDE.md §3, NEVER rule 4), and a fixture HTML page extracts to a pinned,
known text (the golden-file discipline CLAUDE.md §8 asks for).

The golden string below is deliberately a literal, not a second call to
`trafilatura.extract` on the same bytes — comparing the library's output to
itself would prove nothing. Extraction output is genuinely sensitive to both
`trafilatura`'s version and this module's own options (`include_comments`
among them), and since every character here is paid Opus input downstream
(DESIGN §8), a break on a dependency bump is a signal worth seeing, not noise
to silence. If a `trafilatura` upgrade breaks this test, re-review the new
extraction before re-pasting it as the golden value.
"""

from __future__ import annotations

import sqlite3

import httpx
import respx

from signalforge.db import get_item, upsert_item
from signalforge.ingest.base import HttpFetcher
from signalforge.ingest.fullcontent import (
    MAX_FULL_CONTENT_CHARS,
    FullContentResult,
    fetch_full_content,
)
from signalforge.models import Item
from tests.conftest import dump_table, make_item
from tests.ingest.conftest import fixture_bytes

ARTICLE_URL = "https://example.com/agents-that-ship"

EXPECTED_ARTICLE_TEXT = (
    "Agents that ship: what changed in production AI this week\n"
    "Three teams shipped agent-driven pipelines to production this week, and the "
    "pattern across all three was the same: narrow scope, deterministic guardrails, "
    "and a human approving anything that spends money.\n"
    "The first, a logistics startup, wired an agent to reroute shipments when a "
    "carrier misses an SLA. It never touches billing directly — it drafts a "
    "reroute plan and a human clicks approve. That one guardrail is why the team "
    "trusts it enough to run unattended overnight.\n"
    "The second team took the opposite lesson from a near-miss: an agent that "
    "drafted refund emails once produced a plausible-sounding apology for an "
    "order that did not exist. The fix was not a smarter prompt. It was a "
    "database lookup the agent could not bypass.\n"
    "The third case is the most boring and the most instructive: an internal "
    "tool that classifies support tickets. No news there, except that it has run "
    "unattended for four months without a single incident, because the scope is "
    "narrow enough that nobody had to think hard about failure modes.\n"
    "None of this is about bigger models. It is about smaller blast radii."
)
"""`article_full.html` carries a `<section class="comments">` this text does
NOT include — the golden test below is what pins `include_comments=False`."""


def _stored_item(conn: sqlite3.Connection, **overrides: object) -> tuple[int, Item]:
    """Persist a fresh item with no content, and hand back its id and row."""
    fields: dict[str, object] = {"url": ARTICLE_URL, "external_id": "guid-fullcontent"}
    fields.update(overrides)
    item = make_item(**fields)
    item_id, _ = upsert_item(conn, item)
    stored = get_item(conn, item_id)
    assert stored is not None
    return item_id, stored


# --------------------------------------------------------------------------- #
# Golden extraction
# --------------------------------------------------------------------------- #


@respx.mock
async def test_fetch_full_content_extracts_and_stores_article_text(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    respx.get(ARTICLE_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("article_full.html"))
    )
    item_id, stored = _stored_item(conn)

    result = await fetch_full_content(fetcher, conn, [stored])

    assert result.fetched == 1
    assert result.errors == []
    row = get_item(conn, item_id)
    assert row is not None
    assert row.content == EXPECTED_ARTICLE_TEXT


@respx.mock
async def test_fetch_full_content_truncates_at_the_char_cap(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    long_paragraphs = "".join(
        f"<p>Paragraph number {n} repeats to push this article past the cap. "
        "Nothing here is meant to be read, only counted.</p>\n"
        for n in range(1000)
    )
    long_html = f"<html><body><article>{long_paragraphs}</article></body></html>".encode()
    respx.get(ARTICLE_URL).mock(return_value=httpx.Response(200, content=long_html))
    item_id, stored = _stored_item(conn)

    result = await fetch_full_content(fetcher, conn, [stored])

    assert result.fetched == 1
    row = get_item(conn, item_id)
    assert row is not None
    assert row.content is not None
    # <= rather than ==: the cut can land on whitespace and rstrip a
    # character or two off; the cap itself and the ellipsis are what matter.
    assert len(row.content) <= MAX_FULL_CONTENT_CHARS + 1  # +1 for the truncation ellipsis
    assert row.content.endswith("…")


# --------------------------------------------------------------------------- #
# Degradation: no extractable content, never a crash
# --------------------------------------------------------------------------- #


@respx.mock
async def test_fetch_full_content_leaves_content_null_on_a_paywall_stub(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    respx.get(ARTICLE_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("paywall_stub.html"))
    )
    item_id, stored = _stored_item(conn)

    result = await fetch_full_content(fetcher, conn, [stored])

    assert result.fetched == 0
    assert result.errors == []
    row = get_item(conn, item_id)
    assert row is not None
    assert row.content is None


@respx.mock
async def test_fetch_full_content_rejects_a_teaser_shorter_than_the_summary(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    """A common paywall shape: a short teaser, then a wall. Not empty — it
    survives `trafilatura.extract` as real-looking text — so without the
    length floor it would permanently shadow a `summary` that already said
    more (`WHERE content IS NULL` means a thin extract is never revisited).
    """
    teaser_html = (
        b"<html><body><article><p>Subscribe to read the rest of this story.</p>"
        b"</article></body></html>"
    )
    respx.get(ARTICLE_URL).mock(return_value=httpx.Response(200, content=teaser_html))
    long_summary = "A feed summary that is deliberately much longer than the teaser text. " * 3
    item_id, stored = _stored_item(conn, summary=long_summary)

    result = await fetch_full_content(fetcher, conn, [stored])

    assert result.fetched == 0
    assert result.errors == []
    row = get_item(conn, item_id)
    assert row is not None
    assert row.content is None


@respx.mock
async def test_fetch_full_content_leaves_content_null_on_a_non_html_payload(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    respx.get(ARTICLE_URL).mock(
        return_value=httpx.Response(
            200,
            content=fixture_bytes("non_html_payload.json"),
            headers={"content-type": "application/json"},
        )
    )
    item_id, stored = _stored_item(conn)

    result = await fetch_full_content(fetcher, conn, [stored])

    assert result.fetched == 0
    assert result.errors == []
    row = get_item(conn, item_id)
    assert row is not None
    assert row.content is None


@respx.mock
async def test_fetch_full_content_records_a_404_as_a_source_error_and_moves_on(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    respx.get(ARTICLE_URL).mock(return_value=httpx.Response(404))
    item_id, stored = _stored_item(conn)

    result = await fetch_full_content(fetcher, conn, [stored])

    assert result.fetched == 0
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error.source_id == stored.source_id
    assert error.error_type == "FetchError"
    # Distinguishable from a whole-feed 404 (CLAUDE.md §7: the reports are the
    # monitoring channel) — the item id and URL must actually be in the message,
    # not just the source id every other error on that source would also carry.
    assert str(item_id) in error.message
    assert ARTICLE_URL in error.message
    row = get_item(conn, item_id)
    assert row is not None
    assert row.content is None


@respx.mock
async def test_fetch_full_content_does_not_send_conditional_headers(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    """Locks `conditional=False`: this function only ever targets a URL whose
    `content` is NULL, so there is no prior body to validate against, and
    staging a validator here risks a later commit turning a retriable
    extraction failure into a permanent 304 (see the docstring in
    `fetch_full_content`)."""
    route = respx.get(ARTICLE_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("article_full.html"))
    )
    _item_id, stored = _stored_item(conn)

    await fetch_full_content(fetcher, conn, [stored])

    sent_headers = route.calls[0].request.headers
    assert "if-none-match" not in sent_headers
    assert "if-modified-since" not in sent_headers
    # The header check alone can't fail on this fixture regardless of
    # `conditional=`, since the mocked 200 carries no ETag/Last-Modified to
    # send back next time — `conditional=True` would still send no headers on
    # this *first* fetch. What `conditional=True` actually does differently is
    # *stage* a validator for next time (`ValidatorStore.stage`), which is the
    # real hazard the docstring in `fetch_full_content` describes; this is
    # what actually locks `conditional=False`.
    assert fetcher.validators.pending_count() == 0


# --------------------------------------------------------------------------- #
# Idempotency (CLAUDE.md §3, NEVER rule 4)
# --------------------------------------------------------------------------- #


@respx.mock
async def test_fetch_full_content_skips_an_item_that_already_has_content(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    """No HTTP call at all for an item whose `content` is already set — the
    core of the "second run costs nothing" gate."""
    route = respx.get(ARTICLE_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("article_full.html"))
    )
    _item_id, stored = _stored_item(conn, content="Already fetched by an earlier run.")

    result = await fetch_full_content(fetcher, conn, [stored])

    assert result == FullContentResult(fetched=0, errors=[])
    assert route.call_count == 0


# --------------------------------------------------------------------------- #
# Untrusted URL: item.url is a publisher's choice, not the operator's
# --------------------------------------------------------------------------- #


@respx.mock
async def test_fetch_full_content_skips_an_item_whose_url_is_a_disallowed_host(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    """`item.url` is a feed entry's own link — publisher-controlled, not
    operator-controlled, unlike every other URL this module's neighbors
    fetch. A malicious or compromised feed pointing an entry at an internal
    address must never be fetched, extracted, and rendered into the vault.
    No route call at all is the real assertion, mirroring
    `test_probe_feed_does_not_follow_a_redirect`'s reasoning.
    """
    internal_url = "http://169.254.169.254/latest/meta-data/"
    route = respx.get(internal_url).mock(
        return_value=httpx.Response(200, content=fixture_bytes("article_full.html"))
    )
    item_id, stored = _stored_item(conn, url=internal_url)

    result = await fetch_full_content(fetcher, conn, [stored])

    assert route.call_count == 0
    assert result.fetched == 0
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error.error_type == "DisallowedFetchHost"
    assert str(item_id) in error.message
    row = get_item(conn, item_id)
    assert row is not None
    assert row.content is None


@respx.mock
async def test_fetch_full_content_does_not_follow_a_redirect_to_a_disallowed_host(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    """The redirect-based half of the same guard: `item.url` itself can pass
    `is_disallowed_fetch_host` (a public host) while 302ing to a private one.
    `follow_redirects=False` means that hop is never taken — the redirect
    target is mocked to a *success* response so a passing test proves the
    hop genuinely wasn't followed, not that the redirect merely failed for
    some other reason.
    """
    redirect_target = "http://169.254.169.254/latest/meta-data/"
    respx.get(ARTICLE_URL).mock(
        return_value=httpx.Response(302, headers={"location": redirect_target})
    )
    target_route = respx.get(redirect_target).mock(
        return_value=httpx.Response(200, content=fixture_bytes("article_full.html"))
    )
    item_id, stored = _stored_item(conn)

    result = await fetch_full_content(fetcher, conn, [stored])

    assert target_route.call_count == 0
    assert result.fetched == 0
    row = get_item(conn, item_id)
    assert row is not None
    assert row.content is None


@respx.mock
async def test_a_second_run_over_the_same_slice_performs_zero_http_calls_and_zero_writes(
    fetcher: HttpFetcher, conn: sqlite3.Connection
) -> None:
    """The Stage 2 double-run gate: re-running on a freshly re-read slice (the
    shape a real caller — re-querying the DB each invocation — actually takes)
    makes no HTTP call and writes no row the second time."""
    route = respx.get(ARTICLE_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("article_full.html"))
    )
    _item_id, stored = _stored_item(conn)

    first = await fetch_full_content(fetcher, conn, [stored])
    assert first.fetched == 1
    assert route.call_count == 1
    before = dump_table(conn, "items")

    refreshed = get_item(conn, stored.id) if stored.id is not None else None
    assert refreshed is not None
    second = await fetch_full_content(fetcher, conn, [refreshed])

    assert second.fetched == 0
    assert second.errors == []
    assert route.call_count == 1  # unchanged
    assert dump_table(conn, "items") == before
