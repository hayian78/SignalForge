"""Tests for `ingest/probe.py` — the health facts behind a curation proposal.

Every request is a recorded payload served by `respx`; the suite never touches a
live network (CLAUDE.md §8, NEVER rule 13).

The three cases that matter are the three already recorded by hand in
`config/sources.yaml`, because those are the decisions this probe is automating:

* a healthy feed (the shipped `simonwillison` capture);
* `the-batch` — no feed exists at any plausible path, so the fetch 404s;
* `stratechery` — the feed parses fine but its entries are teaser stubs.

The third is the interesting one: it must come back `ok=True` with a small
`median_summary_chars`, not rejected. Nothing about it is broken, so a mechanical
verdict would be inventing a threshold and hiding a judgment that belongs to the
operator (see the module docstring).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from signalforge.ingest.base import HttpFetcher
from signalforge.ingest.probe import PROBE_SOURCE_ID, probe_feed, probe_repo
from tests.ingest.conftest import MAX_SUMMARY_CHARS, fixture_bytes, fixture_text

FEED_URL = "https://simonwillison.net/atom/everything/"
TEASER_URL = "https://members-only.example.com/feed"
RICH_URL = "https://applied-inference.example.com/feed"
RELEASES_URL = "https://api.github.com/repos/anthropics/claude-code/releases"

# The recorded feeds carry real capture dates, so age-sensitive assertions pin
# `now` rather than letting the suite rot as the fixtures age.
FROZEN_NOW = datetime(2026, 7, 28, 6, 0, 0, tzinfo=UTC)
WIDE_WINDOW = 3650


# --------------------------------------------------------------------------- #
# probe_feed
# --------------------------------------------------------------------------- #


@respx.mock
async def test_probe_feed_reports_a_healthy_feed(fetcher: HttpFetcher) -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )

    probe = await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    assert probe.ok is True
    assert probe.error is None
    assert probe.status_code == 200
    assert probe.items_total > 0
    assert probe.median_summary_chars > 0
    # "Has this gone quiet?" is the most decision-relevant fact in the block, so
    # it must actually be populated rather than silently None.
    assert probe.newest_published_at is not None
    assert probe.newest_published_at.tzinfo is not None
    assert probe.label is not None


@respx.mock
async def test_probe_feed_reports_a_missing_feed_as_not_ok(fetcher: HttpFetcher) -> None:
    # The `the-batch` case: every plausible feed path 404s, so there is nothing
    # to evaluate and nothing to put in front of the operator.
    respx.get(FEED_URL).mock(return_value=httpx.Response(404, text="Not Found"))

    probe = await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    assert probe.ok is False
    assert probe.status_code == 404
    assert probe.error is not None
    assert probe.items_total == 0


@respx.mock
async def test_probe_feed_reports_an_html_page_as_not_ok(fetcher: HttpFetcher) -> None:
    # The other half of the `the-batch` case: the path returns 200, but it is a
    # JS-rendered page, not a feed. A 200 is not evidence of a feed.
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, html="<html><body><h1>The Batch</h1></body></html>")
    )

    probe = await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    assert probe.ok is False
    assert probe.error == "no parseable entries"


@respx.mock
async def test_probe_feed_surfaces_teaser_stubs_as_a_number_not_a_rejection(
    fetcher: HttpFetcher,
) -> None:
    """The `stratechery` case — the whole reason probes report rather than reject.

    This feed is valid, fresh, and correctly dated. The only thing wrong with it
    is that its entries say "Today's Article is for subscribers only", which is a
    judgment call about worth, not a mechanical defect. So it must come back
    usable, carrying the number that makes the judgment obvious.
    """
    respx.get(TEASER_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("teaser_stub_feed.xml"))
    )

    probe = await probe_feed(
        fetcher,
        TEASER_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    assert probe.ok is True, "a paywalled-but-valid feed is a judgment call, not a probe failure"
    assert probe.items_total == 4
    # Three of four entries are one-line teasers, so the median lands in stub
    # territory while the weekly roundup keeps the *maximum* respectable. That is
    # exactly why the median is the reported statistic.
    assert probe.median_summary_chars < 200
    assert max(probe.median_summary_chars, 0) > 0


OUT_OF_ORDER_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Two Authors</title><link>https://two.example.com</link><description>d</description>
  <item>
    <title>Older post, listed first</title>
    <link>https://two.example.com/older/</link>
    <guid>https://two.example.com/older/</guid>
    <pubDate>Wed, 01 Jul 2026 09:00:00 GMT</pubDate>
    <author>Former Writer</author>
    <description>An older post that the feed happens to list first.</description>
  </item>
  <item>
    <title>Newer post, listed second</title>
    <link>https://two.example.com/newer/</link>
    <guid>https://two.example.com/newer/</guid>
    <pubDate>Mon, 27 Jul 2026 09:00:00 GMT</pubDate>
    <author>Current Writer</author>
    <description>The newest post, which the feed lists after the older one.</description>
  </item>
</channel></rss>
"""
"""A feed whose entries are NOT in reverse-chronological order, with a different
author on each. Inline rather than a fixture file because its only job is to be
mis-ordered — there is nothing realistic to capture.

RSS ordering is a convention, not a guarantee, and plenty of generators get it
wrong. Without this payload, `label=items[0].author` and `label=<newest>.author`
are indistinguishable on every other fixture in the suite (verified: that
mutation survived the whole file)."""


@respx.mock
async def test_probe_feed_labels_the_newest_entry_not_the_first_listed(
    fetcher: HttpFetcher,
) -> None:
    # `label` answers "who writes this *now*". A feed that lists an old post first
    # would otherwise attribute the source to whoever used to write it.
    respx.get(TEASER_URL).mock(
        return_value=httpx.Response(200, content=OUT_OF_ORDER_FEED.encode("utf-8"))
    )

    probe = await probe_feed(
        fetcher,
        TEASER_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    assert probe.label == "Current Writer"
    assert probe.newest_published_at == datetime(2026, 7, 27, 9, 0, tzinfo=UTC)


@respx.mock
async def test_median_summary_chars_separates_rich_feeds_from_teaser_stubs(
    fetcher: HttpFetcher,
) -> None:
    """The statistic must actually do the job the design claims it does.

    `probe_feed` reports `median_summary_chars` on the argument that it lets an
    operator tell a publication worth a digest slot from a members-only feed of
    one-line teasers. That argument is only worth anything if the number
    separates the two by a wide, obvious margin — so measure both and assert the
    gap, rather than asserting a threshold on one fixture and hoping.

    This test exists because the first version of this suite checked only
    `teaser_median < 200`, which the two-entry `simonwillison` capture also
    satisfies (its median is ~99 chars — realistic for a link blog). The claim
    looked verified and wasn't.
    """
    respx.get(TEASER_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("teaser_stub_feed.xml"))
    )
    respx.get(RICH_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("content_rich_feed.xml"))
    )

    async def median(url: str) -> int:
        probe = await probe_feed(
            fetcher,
            url,
            max_item_age_days=WIDE_WINDOW,
            max_summary_chars=MAX_SUMMARY_CHARS,
            now=FROZEN_NOW,
        )
        assert probe.ok is True
        return probe.median_summary_chars

    teaser_median = await median(TEASER_URL)
    rich_median = await median(RICH_URL)

    assert teaser_median < 200
    assert rich_median > 1000
    assert rich_median > teaser_median * 10, (
        f"the median must separate these by an obvious margin to be worth reporting; "
        f"got {teaser_median} vs {rich_median}"
    )


@respx.mock
async def test_probe_feed_counts_the_freshness_window_separately_from_the_total(
    fetcher: HttpFetcher,
) -> None:
    # A slow publisher legitimately has 0 items in a 7-day window. That must be
    # visible as a number without making the feed `ok=False` — otherwise every
    # monthly newsletter is unproposable.
    respx.get(TEASER_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("teaser_stub_feed.xml"))
    )

    probe = await probe_feed(
        fetcher,
        TEASER_URL,
        max_item_age_days=1,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=datetime(2026, 9, 1, 6, 0, tzinfo=UTC),
    )

    assert probe.ok is True
    assert probe.items_total == 4
    assert probe.items_in_window == 0


@respx.mock
async def test_probe_feed_recovers_what_it_can_from_a_malformed_feed(
    fetcher: HttpFetcher,
) -> None:
    # `parse_feed` returns the entries feedparser recovered; a partially broken
    # feed is a reportable state, not a failure. Whether the recovered count is
    # good enough is the operator's call.
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("malformed_feed.xml"))
    )

    probe = await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    # Pin the measurement. `assert probe.ok is (probe.items_total > 0)` reads like
    # coverage of partial recovery but holds for every possible input — `_failed`
    # always sets items_total=0 and the success path always has entries.
    assert probe.ok is True
    assert probe.items_total == 1, "the capture damages the feed after its first entry"


@respx.mock
async def test_probe_feed_does_not_send_conditional_headers_or_store_validators(
    fetcher: HttpFetcher,
) -> None:
    """Probes must always get a body, and must leave no conditional-GET state.

    Two failures this prevents. A 304 has nothing to measure, so a probe that
    inherited validators from an earlier run would report an unparseable feed for
    a healthy source. And staging validators under a `source_id` that does not
    exist yet would leave cache state for a candidate the operator may reject.
    """
    route = respx.get(FEED_URL).mock(
        return_value=httpx.Response(
            200,
            content=fixture_bytes("simonwillison_atom.xml"),
            headers={"etag": '"abc123"', "last-modified": "Mon, 27 Jul 2026 11:00:00 GMT"},
        )
    )

    await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    request = route.calls.last.request
    assert "if-none-match" not in request.headers
    assert "if-modified-since" not in request.headers
    # The server offered an ETag and the probe declined to remember it.
    assert fetcher.validators.commit() == 0


@respx.mock
async def test_probe_feed_truncates_a_long_error(fetcher: HttpFetcher) -> None:
    """Probe errors render in a digest block; a long one would swamp it.

    Driven through a transport error, because that is the only path whose message
    is actually long: a 500 is retryable, so `HttpFetcher` reports the short
    `"HTTP 500 after 2 attempts"` and the response body never reaches the error
    string. An earlier version of this test used a 5000-char 500 body and passed
    with the truncation removed entirely.
    """
    respx.get(FEED_URL).mock(side_effect=httpx.ConnectError("x" * 5000))

    probe = await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    assert probe.ok is False
    assert probe.error is not None
    assert len(probe.error) == 200, "the message is long enough that truncation must engage"


@pytest.mark.parametrize("kind", ["feed", "repo"])
@respx.mock
async def test_a_redirect_loop_is_reported_not_raised(fetcher: HttpFetcher, kind: str) -> None:
    """A redirect loop must not abort a run probing several candidates.

    `HttpFetcher.get` wraps transport errors and bad statuses, but `httpx` raises
    `TooManyRedirects` — an `HTTPError`, not a `TransportError` — which it neither
    retries nor wraps. Probe URLs and repo slugs both come from the scout, the
    least trustworthy source of either in the system, and consent walls and
    paywalls redirect-loop routinely. Before this was handled, one such candidate
    killed the whole curate run (CLAUDE.md §7, NEVER rule 12).

    Parametrized over both probes rather than testing the feed alone: they have
    separate `except` blocks, and an earlier version of this test left the repo
    path's entirely unverified.
    """
    url = FEED_URL if kind == "feed" else RELEASES_URL
    respx.get(url).mock(return_value=httpx.Response(302, headers={"location": url}))

    if kind == "feed":
        probe = await probe_feed(
            fetcher,
            url,
            max_item_age_days=WIDE_WINDOW,
            max_summary_chars=MAX_SUMMARY_CHARS,
            now=FROZEN_NOW,
        )
    else:
        probe = await probe_repo(
            fetcher, "anthropics/claude-code", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
        )

    assert probe.ok is False
    assert probe.error is not None
    assert "edirect" in probe.error


@respx.mock
async def test_probe_feed_archives_under_a_reserved_source_id(fetcher: HttpFetcher) -> None:
    # A candidate has no sources.yaml id yet, so probe payloads must not land in
    # the cache directory of the source it would become.
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )

    await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    assert PROBE_SOURCE_ID.startswith("_"), "reserved ids stay outside the real id namespace"
    assert (fetcher.cache_dir / PROBE_SOURCE_ID).exists()


# --------------------------------------------------------------------------- #
# probe_repo
# --------------------------------------------------------------------------- #


@respx.mock
async def test_probe_repo_reports_releases(fetcher: HttpFetcher) -> None:
    respx.get(RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )

    probe = await probe_repo(
        fetcher, "anthropics/claude-code", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
    )

    assert probe.ok is True
    assert probe.items_total > 0
    assert probe.label is not None


@respx.mock
async def test_probe_repo_reports_a_repo_with_no_releases_as_not_ok(
    fetcher: HttpFetcher,
) -> None:
    # A release watch on a repo that publishes none would ingest nothing. Unlike
    # `github.py`, the probe does not fall back to /tags: proposing a repo on the
    # strength of bare version tags is how the noise this feature removes gets
    # back in (see the pruned llama.cpp CI tags in sources.yaml).
    respx.get(RELEASES_URL).mock(return_value=httpx.Response(200, json=[]))

    probe = await probe_repo(
        fetcher, "anthropics/claude-code", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
    )

    assert probe.ok is False
    assert probe.error == "no published releases"


@respx.mock
async def test_probe_repo_reads_the_newest_release_by_date_not_payload_order(
    fetcher: HttpFetcher,
) -> None:
    """`label` and `newest_published_at` come from the newest release by date.

    Deliberately listed out of order, because GitHub's ordering is a convention
    rather than a guarantee and `releases[0]` quietly encodes trust in it. Also
    covers the two timestamp behaviours `_release_timestamp` adds on top of
    `github.parse_github_timestamp`: the `created_at` fallback for a release with
    no publish time, and naive→UTC coercion (a naive value would otherwise raise
    when compared against an aware `now`).
    """
    respx.get(RELEASES_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"tag_name": "v1.0.0", "body": "old", "published_at": "2026-06-01T10:00:00Z"},
                # Newest, and only dated via created_at.
                {"tag_name": "v1.2.0", "body": "newest", "created_at": "2026-07-27T10:00:00Z"},
                # Naive timestamp: read as UTC rather than blowing up the compare.
                {"tag_name": "v1.1.0", "body": "middle", "published_at": "2026-07-01T10:00:00"},
            ],
        )
    )

    probe = await probe_repo(
        fetcher, "anthropics/claude-code", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
    )

    assert probe.ok is True
    assert probe.items_total == 3
    assert probe.label == "v1.2.0"
    assert probe.newest_published_at == datetime(2026, 7, 27, 10, 0, tzinfo=UTC)


@respx.mock
async def test_probe_repo_skips_draft_releases(fetcher: HttpFetcher) -> None:
    # Parity with `github.py::_releases_to_items`, which also skips drafts. Without
    # it an authenticated probe measures releases a watch would never ingest — the
    # probe describing something other than the thing it validates.
    respx.get(RELEASES_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"tag_name": "v2.0.0-rc", "body": "x" * 500, "draft": True},
                {"tag_name": "v1.0.0", "body": "real", "published_at": "2026-07-01T10:00:00Z"},
            ],
        )
    )

    probe = await probe_repo(
        fetcher, "anthropics/claude-code", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
    )

    assert probe.items_total == 1
    assert probe.label == "v1.0.0"


@respx.mock
async def test_probe_repo_reports_a_repo_whose_only_release_is_a_draft_as_not_ok(
    fetcher: HttpFetcher,
) -> None:
    respx.get(RELEASES_URL).mock(
        return_value=httpx.Response(200, json=[{"tag_name": "v2.0.0-rc", "draft": True}])
    )

    probe = await probe_repo(
        fetcher, "anthropics/claude-code", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
    )

    assert probe.ok is False
    assert probe.error == "no published releases"


@respx.mock
async def test_probe_repo_reports_the_median_body_not_the_mean(fetcher: HttpFetcher) -> None:
    """The statistic must be the median, and the two must be far apart here.

    A repo of bare CI tags with one substantial release is the exact shape where
    mean and median disagree — and the shape `sources.yaml` records having pruned
    by hand. With every fixture body empty, mean == median == 0 and a
    median→mean mutation survives, so one long body is what gives this teeth.
    """
    respx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"tag_name": "b10068", "body": "", "published_at": "2026-07-27T10:00:00Z"},
                {"tag_name": "b10067", "body": "", "published_at": "2026-07-26T10:00:00Z"},
                {"tag_name": "b10066", "body": "", "published_at": "2026-07-25T10:00:00Z"},
                {"tag_name": "b10065", "body": "y" * 4000, "published_at": "2026-07-24T10:00:00Z"},
            ],
        )
    )

    probe = await probe_repo(
        fetcher, "ggml-org/llama.cpp", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
    )

    assert probe.ok is True, "bare CI tags are a judgment call, not a probe failure"
    assert probe.median_summary_chars == 0, "three of four bodies are empty"
    assert probe.items_total == 4


@respx.mock
async def test_probe_repo_counts_the_freshness_window_and_keeps_undated_releases(
    fetcher: HttpFetcher,
) -> None:
    """Pins the repo window filter, which every other repo test leaves unbound.

    Two mutations survived the whole file before this existed: dropping undated
    releases from the count (contradicting the parity rule the field docstring
    promises) and removing the cutoff entirely so `items_in_window` just equals
    `items_total`. Every other repo test passes a 3650-day window, so the cutoff
    is never binding in any of them.

    Undated counts as in-window on purpose, matching `filter_by_age`'s rule for
    feed items: the conservative failure is showing the operator one release too
    many, not hiding a source that publishes without timestamps.
    """
    respx.get(RELEASES_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"tag_name": "v3.0.0", "body": "recent", "published_at": "2026-07-28T01:00:00Z"},
                {"tag_name": "v2.0.0", "body": "ancient", "published_at": "2026-01-01T10:00:00Z"},
                {"tag_name": "v1.0.0", "body": "no date at all"},
            ],
        )
    )

    probe = await probe_repo(fetcher, "anthropics/claude-code", max_item_age_days=1, now=FROZEN_NOW)

    assert probe.items_total == 3
    assert probe.items_in_window == 2, "the recent release plus the undated one"


@respx.mock
async def test_probe_repo_reports_a_missing_repo_as_not_ok(fetcher: HttpFetcher) -> None:
    respx.get(RELEASES_URL).mock(return_value=httpx.Response(404, json={"message": "Not Found"}))

    probe = await probe_repo(
        fetcher, "anthropics/claude-code", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
    )

    assert probe.ok is False
    assert probe.status_code == 404


@respx.mock
async def test_probe_repo_sends_the_token_in_a_header_and_never_stores_it(
    fetcher: HttpFetcher,
) -> None:
    # NEVER rule 16: the token reaches the request and nothing else — not the
    # probe facts, which are written to the DB and rendered in a digest.
    route = respx.get(RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )

    probe = await probe_repo(
        fetcher,
        "anthropics/claude-code",
        max_item_age_days=WIDE_WINDOW,
        token="ghp_secret_value",  # noqa: S106 - a fake token for a mocked request
        now=FROZEN_NOW,
    )

    assert route.calls.last.request.headers["authorization"] == "Bearer ghp_secret_value"
    assert "ghp_secret_value" not in json.dumps(probe.as_facts())


@respx.mock
async def test_probe_repo_works_unauthenticated_and_stores_no_validators(
    fetcher: HttpFetcher,
) -> None:
    # 60 req/hr is ample for a handful of weekly candidates, so a missing token
    # is not an error — same stance as `github.py`.
    #
    # The conditional-GET assertions matter as much here as on the feed path, and
    # more on this endpoint specifically: `/releases` returns a famously stable
    # ETag on an empty array (the reason `github.py`'s /tags fallback exists), so
    # a probe that remembered validators would 304 itself into reporting a
    # perfectly healthy repo as unparseable on the next run.
    route = respx.get(RELEASES_URL).mock(
        return_value=httpx.Response(
            200,
            text=fixture_text("github_releases.json"),
            headers={"etag": '"releases-etag"'},
        )
    )

    probe = await probe_repo(
        fetcher, "anthropics/claude-code", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
    )

    request = route.calls.last.request
    assert probe.ok is True
    assert "authorization" not in request.headers
    assert "if-none-match" not in request.headers
    assert "if-modified-since" not in request.headers
    assert fetcher.validators.commit() == 0


@respx.mock
async def test_probe_repo_rejects_a_non_list_payload(fetcher: HttpFetcher) -> None:
    respx.get(RELEASES_URL).mock(return_value=httpx.Response(200, json={"message": "nope"}))

    probe = await probe_repo(
        fetcher, "anthropics/claude-code", max_item_age_days=WIDE_WINDOW, now=FROZEN_NOW
    )

    assert probe.ok is False
    assert probe.error == "releases response was not a list"


# --------------------------------------------------------------------------- #
# as_facts — the shape stored in proposals.probe
# --------------------------------------------------------------------------- #


@respx.mock
async def test_as_facts_is_json_serializable_and_omits_absent_fields(
    fetcher: HttpFetcher,
) -> None:
    # `db.insert_proposal` and `db.update_proposal_probe` both `json.dumps` this,
    # so an unserializable value would fail at write time, after the money is
    # already spent on the scout call.
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )

    probe = await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )
    facts = probe.as_facts()

    assert json.loads(json.dumps(facts, sort_keys=True)) == facts
    assert "error" not in facts, "a successful probe carries no error key"
    assert facts["ok"] is True


@respx.mock
async def test_as_facts_carries_the_reason_a_probe_failed(fetcher: HttpFetcher) -> None:
    # This is what `curate` stores on an `invalid` proposal, and what tells the
    # operator (and the next re-probe) whether the failure looked transient.
    respx.get(FEED_URL).mock(return_value=httpx.Response(503, text="Service Unavailable"))

    probe = await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )
    facts = probe.as_facts()

    assert facts["ok"] is False
    assert "503" in str(facts["error"])
    assert facts["status_code"] == 503


def test_probe_module_imports_no_llm_machinery() -> None:
    """`ingest/` never calls an LLM (CLAUDE.md §2, NEVER rule 2).

    Worth asserting rather than trusting, because `probe.py` is the module in
    `ingest/` that a curation feature reaches into — the one place where someone
    might reasonably think "the model could judge this feed for me". Checked
    against the module's real imports rather than its text, so a mention in a
    docstring cannot trip it and an aliased import cannot slip past it.
    """
    module = importlib.import_module("signalforge.ingest.probe")
    tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `from signalforge import llm` puts the module in `names`, not in
            # `module`, so recording only `node.module` would miss the most
            # idiomatic way to reach `llm.py` from inside the package. A relative
            # import carries `level > 0` and a partial (or absent) module, so
            # resolve it against this package before comparing.
            base = node.module or ""
            if node.level:
                # Resolve against this module's package so `..llm` becomes
                # `signalforge.llm` rather than `signalforge.ingest.llm`. Ruff's
                # TID252 already rejects parent-relative imports, so this is
                # defence in depth — but a guard with a known hole is worse than
                # no guard, because it reads as covered.
                base = importlib.util.resolve_name("." * node.level + base, "signalforge.ingest")
            imported.add(base)
            imported.update(f"{base}.{alias.name}".strip(".") for alias in node.names)

    assert not any(name.startswith("anthropic") for name in imported), (
        f"probe.py must not reach the Anthropic SDK; imports were {sorted(imported)}"
    )
    assert "signalforge.llm" not in imported
    # The module docstring also promises "No writes" — a probe produces a
    # SourceProbe and nothing else, so it has no business holding a DB handle.
    assert "signalforge.db" not in imported


@pytest.mark.parametrize("status", [500, 502, 503])
@respx.mock
async def test_probe_feed_retries_then_reports_a_server_error(
    fetcher: HttpFetcher, status: int
) -> None:
    # Retryable statuses are retried by `HttpFetcher` (max_attempts=2 in the
    # fixture) and then reported, not raised — a probe failure must never abort
    # the curate run that is probing several candidates (CLAUDE.md §7).
    route = respx.get(FEED_URL).mock(return_value=httpx.Response(status, text="upstream sad"))

    probe = await probe_feed(
        fetcher,
        FEED_URL,
        max_item_age_days=WIDE_WINDOW,
        max_summary_chars=MAX_SUMMARY_CHARS,
        now=FROZEN_NOW,
    )

    assert probe.ok is False
    assert route.call_count == 2
    assert probe.status_code == status
