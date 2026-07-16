"""RSS ingestor tests against captured feed payloads (CLAUDE.md §8)."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import respx

from signalforge.config import RssSource
from signalforge.ingest.base import HttpFetcher
from signalforge.ingest.rss import RssIngestor, build_rss_ingestors, parse_feed
from signalforge.models import Item, SourceType
from tests.ingest.conftest import MAX_SUMMARY_CHARS, fixture_bytes, make_sources_config

FEED_URL = "https://simonwillison.net/atom/everything/"


def _source() -> RssSource:
    return RssSource(id="simonwillison", url=FEED_URL, weight=1.3)


def _ingestor(**overrides: object) -> RssIngestor:
    kwargs: dict[str, object] = {"max_summary_chars": MAX_SUMMARY_CHARS}
    kwargs.update(overrides)
    return RssIngestor(_source(), **kwargs)  # type: ignore[arg-type]


def _parse(content: bytes, **overrides: object) -> list[Item]:
    """`parse_feed` with the cost knob supplied, as a real caller must."""
    kwargs: dict[str, object] = {"max_summary_chars": MAX_SUMMARY_CHARS}
    kwargs.update(overrides)
    return parse_feed(content, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Parsing / normalization
# --------------------------------------------------------------------------- #


def test_parses_captured_atom_feed() -> None:
    items = _parse(fixture_bytes("simonwillison_atom.xml"), source_id="simonwillison")

    # 4 entries in, 2 out: one duplicate guid and one linkless entry are dropped.
    assert len(items) == 2

    first = items[0]
    assert first.source_id == "simonwillison"
    assert first.source_type is SourceType.RSS
    assert first.title == "MCP sampling shifts agent orchestration server-side"
    assert first.external_id == "tag:simonwillison.net,2026:/2026/Jul/15/mcp-sampling/"
    assert first.author == "Simon Willison"
    assert first.published_at == datetime(2026, 7, 15, 18, 4, 21, tzinfo=UTC)


def test_summary_html_is_stripped_to_text() -> None:
    items = _parse(fixture_bytes("simonwillison_atom.xml"), source_id="simonwillison")

    summary = items[0].summary
    assert summary is not None
    assert "<p>" not in summary
    assert "<strong>" not in summary
    assert summary.startswith("The new sampling primitive lets an MCP server")


def test_tracking_params_are_stripped_by_canonicalization() -> None:
    """The feed serves `?utm_source=rss`; the dedup key must not carry it."""
    items = _parse(fixture_bytes("simonwillison_atom.xml"), source_id="simonwillison")

    assert "utm_source" in items[0].url  # the URL we actually fetched is preserved
    assert items[0].canonical_url == "https://simonwillison.net/2026/Jul/15/mcp-sampling"


def test_duplicate_guid_within_one_feed_collapses() -> None:
    """A feed repeating a guid must not produce two rows racing for one unique key."""
    items = _parse(fixture_bytes("simonwillison_atom.xml"), source_id="simonwillison")

    external_ids = [item.external_id for item in items]
    assert len(external_ids) == len(set(external_ids))
    assert external_ids.count("tag:simonwillison.net,2026:/2026/Jul/15/mcp-sampling/") == 1


def test_entry_without_link_is_skipped_not_fatal() -> None:
    items = _parse(fixture_bytes("simonwillison_atom.xml"), source_id="simonwillison")
    assert all("no-link" not in item.url for item in items)


def test_malformed_feed_yields_recovered_entries() -> None:
    """feedparser sets `bozo`; we keep what it recovered (DESIGN §7)."""
    items = _parse(fixture_bytes("malformed_feed.xml"), source_id="broken-blog")

    assert len(items) >= 1
    assert items[0].title == "A recoverable item before the damage"
    assert items[0].external_id == "example-recoverable-1"


def test_totally_unparseable_content_yields_no_items_and_no_exception() -> None:
    items = _parse(b"this is not xml at all, it's a 404 page", source_id="broken-blog")
    assert items == []


def test_empty_body_yields_no_items() -> None:
    assert _parse(b"", source_id="broken-blog") == []


def test_raw_path_is_recorded_on_every_item() -> None:
    items = _parse(
        fixture_bytes("simonwillison_atom.xml"),
        source_id="simonwillison",
        raw_path="simonwillison/20260716/abc.raw",
    )
    assert all(item.raw_path == "simonwillison/20260716/abc.raw" for item in items)


def test_content_is_never_populated_at_ingest() -> None:
    """Full content is a lazy top-N deep read, not an ingest concern (CLAUDE.md §6)."""
    items = _parse(fixture_bytes("simonwillison_atom.xml"), source_id="simonwillison")
    assert all(item.content is None for item in items)


# --------------------------------------------------------------------------- #
# The ingestor
# --------------------------------------------------------------------------- #


@respx.mock
async def test_ingest_fetches_and_normalizes(fetcher: HttpFetcher) -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )

    result = await _ingestor().ingest(fetcher)

    assert result.ok
    assert len(result.items) == 2
    assert result.items[0].raw_path is not None


@respx.mock
async def test_ingest_304_yields_no_items(fetcher: HttpFetcher) -> None:
    """The common daily case: unchanged feed, zero items, zero errors."""
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(
            200, content=fixture_bytes("simonwillison_atom.xml"), headers={"etag": '"v1"'}
        )
    )
    first = await _ingestor().ingest(fetcher)
    assert len(first.items) == 2
    fetcher.validators.commit()  # the caller persisted; the 304 is now earned

    respx.get(FEED_URL).mock(return_value=httpx.Response(304))
    second = await _ingestor().ingest(fetcher)

    assert second.items == []
    assert second.ok


@respx.mock
async def test_ingest_error_is_returned_not_raised(fetcher: HttpFetcher) -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(404))

    result = await _ingestor().ingest(fetcher)

    assert result.items == []
    assert len(result.errors) == 1
    assert result.errors[0].source_id == "simonwillison"
    assert result.errors[0].source_type is SourceType.RSS


@respx.mock
async def test_ingest_malformed_feed_over_http_does_not_crash(fetcher: HttpFetcher) -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("malformed_feed.xml"))
    )

    result = await _ingestor().ingest(fetcher)

    assert result.ok
    assert len(result.items) >= 1


def test_build_rss_ingestors_from_config() -> None:
    config = make_sources_config(
        rss=[
            {"id": "simonwillison", "url": FEED_URL, "weight": 1.3},
            {"id": "interconnects", "url": "https://www.interconnects.ai/feed"},
        ]
    )

    ingestors = build_rss_ingestors(config)

    assert [i.source_id for i in ingestors] == ["simonwillison", "interconnects"]


def test_build_rss_ingestors_with_no_feeds() -> None:
    assert build_rss_ingestors(make_sources_config()) == []
