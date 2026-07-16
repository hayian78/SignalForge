"""Hacker News (Algolia) ingestor tests against captured payloads."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import respx

from signalforge.ingest.base import HttpFetcher
from signalforge.ingest.hackernews import HackerNewsIngestor, build_hackernews_ingestors
from signalforge.models import SourceType
from tests.ingest.conftest import (
    MAX_ITEM_AGE_DAYS,
    MAX_SUMMARY_CHARS,
    fixture_text,
    make_sources_config,
)

SEARCH_URL = "https://hn.algolia.com/api/v1/search"
NOW = datetime(2026, 7, 16, 6, 0, 0, tzinfo=UTC)


def _ingestor(**overrides: object) -> HackerNewsIngestor:
    kwargs: dict[str, object] = {
        "keywords": ["mcp"],
        "min_points": 80,
        "max_summary_chars": MAX_SUMMARY_CHARS,
        "now": NOW,
    }
    kwargs.update(overrides)
    return HackerNewsIngestor(**kwargs)  # type: ignore[arg-type]


def _mock_front_page() -> respx.Route:
    return respx.get(SEARCH_URL, params__contains={"tags": "front_page"}).mock(
        return_value=httpx.Response(200, text=fixture_text("hn_front_page.json"))
    )


def _mock_keyword() -> respx.Route:
    return respx.get(SEARCH_URL, params__contains={"query": "mcp"}).mock(
        return_value=httpx.Response(200, text=fixture_text("hn_keyword_mcp.json"))
    )


@respx.mock
async def test_front_page_and_keyword_hits_become_items(fetcher: HttpFetcher) -> None:
    _mock_front_page()
    _mock_keyword()

    result = await _ingestor().ingest(fetcher)

    assert result.ok
    titles = [item.title for item in result.items]
    assert "Show HN: A local-first RAG pipeline in 400 lines of SQLite" in titles
    assert "The MCP specification adds streaming sampling" in titles
    assert all(item.source_id == "hn" for item in result.items)
    assert all(item.source_type is SourceType.HN for item in result.items)


@respx.mock
async def test_item_fields_are_normalized(fetcher: HttpFetcher) -> None:
    _mock_front_page()
    _mock_keyword()

    result = await _ingestor().ingest(fetcher)
    story = next(item for item in result.items if item.external_id == "44921100")

    assert story.url == "https://github.com/example/sqlite-rag"
    assert story.author == "dang_not_really"
    assert story.published_at == datetime(2026, 7, 15, 14, 22, 3, tzinfo=UTC)


@respx.mock
async def test_below_threshold_stories_are_dropped(fetcher: HttpFetcher) -> None:
    """The front-page tag has no points filter; we apply `min_hn_points` ourselves."""
    _mock_front_page()
    _mock_keyword()

    result = await _ingestor().ingest(fetcher)

    assert all("straggler" not in item.title for item in result.items)
    assert all(item.external_id != "44922801" for item in result.items)


@respx.mock
async def test_comment_hits_are_ignored(fetcher: HttpFetcher) -> None:
    _mock_front_page()
    _mock_keyword()

    result = await _ingestor().ingest(fetcher)

    assert all(item.external_id != "44919999" for item in result.items)


@respx.mock
async def test_comments_are_never_fetched(fetcher: HttpFetcher) -> None:
    """DESIGN §7 scopes comment fetching to top items — not Phase 0."""
    _mock_front_page()
    _mock_keyword()

    await _ingestor().ingest(fetcher)

    assert all("/items/" not in str(call.request.url) for call in respx.calls)
    assert all("comment" not in str(call.request.url) for call in respx.calls)


@respx.mock
async def test_ask_hn_without_url_uses_the_permalink(fetcher: HttpFetcher) -> None:
    _mock_front_page()
    _mock_keyword()

    result = await _ingestor().ingest(fetcher)
    ask = next(item for item in result.items if item.external_id == "44918744")

    assert ask.url == "https://news.ycombinator.com/item?id=44918744"
    assert ask.summary is not None
    assert "war stories" in ask.summary


@respx.mock
async def test_duplicate_story_across_queries_collapses(fetcher: HttpFetcher) -> None:
    """The same story appears on the front page and in the `mcp` search."""
    _mock_front_page()
    _mock_keyword()

    result = await _ingestor().ingest(fetcher)

    external_ids = [item.external_id for item in result.items]
    assert external_ids.count("44921100") == 1
    assert len(external_ids) == len(set(external_ids))


@respx.mock
async def test_keyword_query_carries_points_and_lookback_filters(fetcher: HttpFetcher) -> None:
    _mock_front_page()
    route = _mock_keyword()

    await _ingestor().ingest(fetcher)

    filters = route.calls.last.request.url.params["numericFilters"]
    assert "points>=80" in filters
    # 7-day lookback so a missed run self-heals (DESIGN §14).
    # 1783576800 == 2026-07-09T06:00:00Z, exactly 7 days before the fixed NOW.
    assert "created_at_i>1783576800" in filters


@respx.mock
async def test_one_failing_query_keeps_the_others(fetcher: HttpFetcher) -> None:
    _mock_front_page()
    respx.get(SEARCH_URL, params__contains={"query": "mcp"}).mock(return_value=httpx.Response(404))

    result = await _ingestor().ingest(fetcher)

    assert len(result.items) == 2  # the front page still landed
    assert len(result.errors) == 1
    assert result.errors[0].source_type is SourceType.HN


@respx.mock
async def test_304_yields_no_items(fetcher: HttpFetcher) -> None:
    respx.get(SEARCH_URL, params__contains={"tags": "front_page"}).mock(
        return_value=httpx.Response(
            200, text=fixture_text("hn_front_page.json"), headers={"etag": '"fp1"'}
        )
    )
    _mock_keyword()
    first = await _ingestor().ingest(fetcher)
    assert len(first.items) == 3
    fetcher.validators.commit()

    respx.get(SEARCH_URL, params__contains={"tags": "front_page"}).mock(
        return_value=httpx.Response(304)
    )
    respx.get(SEARCH_URL, params__contains={"query": "mcp"}).mock(return_value=httpx.Response(304))
    second = await _ingestor().ingest(fetcher)

    assert second.items == []
    assert second.ok


@respx.mock
async def test_malformed_json_is_an_error_record(fetcher: HttpFetcher) -> None:
    respx.get(SEARCH_URL, params__contains={"tags": "front_page"}).mock(
        return_value=httpx.Response(200, text="{ truncated")
    )
    _mock_keyword()

    result = await _ingestor().ingest(fetcher)

    assert len(result.errors) == 1
    assert len(result.items) == 2  # the keyword query still produced items


@respx.mock
async def test_response_without_hits_array_is_an_error_record(fetcher: HttpFetcher) -> None:
    respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json={"message": "nope"}))

    result = await _ingestor().ingest(fetcher)

    assert result.items == []
    assert len(result.errors) == 2  # front page + one keyword


@respx.mock
async def test_no_keywords_still_fetches_the_front_page(fetcher: HttpFetcher) -> None:
    _mock_front_page()

    result = await _ingestor(keywords=[]).ingest(fetcher)

    assert len(result.items) == 2
    assert result.ok


def test_build_hackernews_ingestor_reads_threshold_from_defaults() -> None:
    config = make_sources_config(
        defaults={
            "fetch_timeout": 20,
            "min_hn_points": 150,
            "max_summary_chars": MAX_SUMMARY_CHARS,
            "max_item_age_days": MAX_ITEM_AGE_DAYS,
        },
        hackernews={"keywords": ["llm", "mcp"]},
    )

    ingestors = build_hackernews_ingestors(config)

    assert len(ingestors) == 1
    assert ingestors[0].min_points == 150
    assert ingestors[0].keywords == ["llm", "mcp"]
    assert ingestors[0].source_id == "hn"


def test_build_hackernews_ingestors_without_block() -> None:
    assert build_hackernews_ingestors(make_sources_config()) == []
