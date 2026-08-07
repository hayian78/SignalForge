"""arXiv (Atom API) ingestor tests against captured payloads."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import respx

from signalforge.ingest.arxiv import (
    ARXIV_API_ROOT,
    ArxivIngestor,
    build_arxiv_ingestors,
    build_search_query,
)
from signalforge.ingest.base import HttpFetcher
from signalforge.models import SourceType
from tests.ingest.conftest import MAX_SUMMARY_CHARS, fixture_text, make_sources_config


def _ingestor(**overrides: object) -> ArxivIngestor:
    kwargs: dict[str, object] = {
        "categories": ["cs.AI", "cs.CL"],
        "require_keywords": ["agents", "tool use"],
        "max_summary_chars": MAX_SUMMARY_CHARS,
    }
    kwargs.update(overrides)
    return ArxivIngestor(**kwargs)  # type: ignore[arg-type]


def _mock_search(fixture: str) -> respx.Route:
    return respx.get(ARXIV_API_ROOT).mock(
        return_value=httpx.Response(200, text=fixture_text(fixture))
    )


@respx.mock
async def test_entries_become_items(fetcher: HttpFetcher) -> None:
    _mock_search("arxiv_search.xml")

    result = await _ingestor().ingest(fetcher)

    assert result.ok
    titles = [item.title for item in result.items]
    assert "Planning with Long-Horizon Memory in Tool-Using Agents" in titles
    assert "A Survey of Evaluation Protocols for Autonomous Coding Agents" in titles
    assert all(item.source_id == "arxiv" for item in result.items)
    assert all(item.source_type is SourceType.ARXIV for item in result.items)


@respx.mock
async def test_version_suffix_is_stripped_from_external_id(fetcher: HttpFetcher) -> None:
    _mock_search("arxiv_search.xml")

    result = await _ingestor().ingest(fetcher)
    survey = next(item for item in result.items if item.external_id == "2506.09999")

    assert survey.url == "https://arxiv.org/abs/2506.09999"


@respx.mock
async def test_multi_line_title_and_abstract_are_flattened(fetcher: HttpFetcher) -> None:
    """arXiv wraps long titles/abstracts across lines in the raw XML."""
    _mock_search("arxiv_search.xml")

    result = await _ingestor().ingest(fetcher)
    paper = next(item for item in result.items if item.external_id == "2507.12345")

    assert "\n" not in paper.title
    assert paper.summary is not None
    assert "\n" not in paper.summary
    assert "episodic memory" in paper.summary


@respx.mock
async def test_multi_author_paper_gets_et_al(fetcher: HttpFetcher) -> None:
    _mock_search("arxiv_search.xml")

    result = await _ingestor().ingest(fetcher)
    survey = next(item for item in result.items if item.external_id == "2506.09999")

    assert survey.author == "Priya Natarajan et al."


@respx.mock
async def test_published_date_is_parsed(fetcher: HttpFetcher) -> None:
    _mock_search("arxiv_search.xml")

    result = await _ingestor().ingest(fetcher)
    paper = next(item for item in result.items if item.external_id == "2507.12345")

    assert paper.published_at == datetime(2026, 7, 15, 17, 32, 11, tzinfo=UTC)


@respx.mock
async def test_duplicate_id_within_response_collapses(fetcher: HttpFetcher) -> None:
    _mock_search("arxiv_search.xml")

    result = await _ingestor().ingest(fetcher)

    external_ids = [item.external_id for item in result.items]
    assert external_ids.count("2507.12345") == 1
    assert len(external_ids) == len(set(external_ids))


@respx.mock
async def test_titleless_entry_is_skipped(fetcher: HttpFetcher) -> None:
    _mock_search("arxiv_search.xml")

    result = await _ingestor().ingest(fetcher)

    assert all(item.external_id != "2507.00042" for item in result.items)
    # The other two distinct papers still landed (the duplicate collapses too).
    assert len(result.items) == 2


@respx.mock
async def test_truncated_response_recovers_the_well_formed_entries(fetcher: HttpFetcher) -> None:
    """`feedparser` sets `bozo` on malformed XML but still returns whatever
    entries it recovered (CLAUDE.md §8) — the same discipline `ingest/rss.py`
    is tested against for a malformed feed."""
    _mock_search("arxiv_malformed.xml")

    result = await _ingestor().ingest(fetcher)

    assert result.ok
    assert len(result.items) == 1
    assert result.items[0].external_id == "2507.55555"


@respx.mock
async def test_old_style_id_scheme_is_handled(fetcher: HttpFetcher) -> None:
    """Pre-2007 ids (`archive/YYMMNNN`) carry a slash inside the id itself, the
    one shape where `partition("/abs/")` + a trailing-`vN` strip could plausibly
    misfire. arXiv still serves these, and the API still returns them."""
    _mock_search("arxiv_old_style_id.xml")

    result = await _ingestor().ingest(fetcher)

    assert len(result.items) == 1
    assert result.items[0].external_id == "hep-th/9901001"
    assert result.items[0].url == "https://arxiv.org/abs/hep-th/9901001"


@respx.mock
async def test_empty_response_yields_no_items(fetcher: HttpFetcher) -> None:
    _mock_search("arxiv_no_results.xml")

    result = await _ingestor().ingest(fetcher)

    assert result.items == []
    assert result.ok


@respx.mock
async def test_synthetic_error_entry_becomes_an_error_record(fetcher: HttpFetcher) -> None:
    """A malformed `search_query` is a 200 OK with an `api/errors#...` entry,
    not an HTTP error — the ingestor must not mistake it for a real paper, and
    must not raise (CLAUDE.md §7). It must also not degrade to silent zero
    items: `runs.errors` is the monitoring channel, and a bad `require_keywords`
    value (including one an unreviewed curation proposal wrote) must show up
    in the next digest's error footer rather than reading as "no papers today"
    forever."""
    _mock_search("arxiv_error.xml")

    result = await _ingestor().ingest(fetcher)

    assert result.items == []
    assert not result.ok
    assert len(result.errors) == 1
    assert result.errors[0].source_type is SourceType.ARXIV
    assert result.errors[0].error_type == "ArxivQueryError"
    assert "incorrect id format" in result.errors[0].message


@respx.mock
async def test_synthetic_error_entry_never_caches_a_304(fetcher: HttpFetcher) -> None:
    """An error response carries a perfectly cacheable ETag; caching it would
    304 forever and turn one bad config edit into a permanently silent source
    (same reasoning as `github.py`'s empty-`/releases` invalidation). Verified
    directly against the conditional-GET headers the *next* request would
    carry, rather than against a canned 304 — respx would return whatever a
    mock says regardless of whether the ingestor actually sent a validator."""
    route = respx.get(ARXIV_API_ROOT).mock(
        return_value=httpx.Response(
            200, text=fixture_text("arxiv_error.xml"), headers={"etag": '"err1"'}
        )
    )
    first = await _ingestor().ingest(fetcher)
    assert len(first.errors) == 1
    fetcher.validators.commit()

    await _ingestor().ingest(fetcher)

    assert "if-none-match" not in route.calls.last.request.headers
    assert "if-modified-since" not in route.calls.last.request.headers


@respx.mock
async def test_304_yields_no_items(fetcher: HttpFetcher) -> None:
    respx.get(ARXIV_API_ROOT).mock(
        return_value=httpx.Response(
            200, text=fixture_text("arxiv_search.xml"), headers={"etag": '"arxiv1"'}
        )
    )
    first = await _ingestor().ingest(fetcher)
    assert len(first.items) == 2
    fetcher.validators.commit()

    respx.get(ARXIV_API_ROOT).mock(return_value=httpx.Response(304))
    second = await _ingestor().ingest(fetcher)

    assert second.items == []
    assert second.ok


@respx.mock
async def test_fetch_failure_is_an_error_record(fetcher: HttpFetcher) -> None:
    respx.get(ARXIV_API_ROOT).mock(return_value=httpx.Response(500))

    result = await _ingestor().ingest(fetcher)

    assert result.items == []
    assert len(result.errors) == 1
    assert result.errors[0].source_type is SourceType.ARXIV


@respx.mock
async def test_query_carries_categories_and_keywords(fetcher: HttpFetcher) -> None:
    route = _mock_search("arxiv_search.xml")

    await _ingestor().ingest(fetcher)

    query = route.calls.last.request.url.params["search_query"]
    assert query == '(cat:cs.AI OR cat:cs.CL) AND (abs:agents OR abs:"tool use")'
    assert route.calls.last.request.url.params["sortBy"] == "submittedDate"
    assert route.calls.last.request.url.params["sortOrder"] == "descending"


def test_build_search_query_categories_only() -> None:
    assert build_search_query(["cs.AI", "cs.SE"], []) == "cat:cs.AI OR cat:cs.SE"


def test_build_search_query_keywords_only() -> None:
    assert build_search_query([], ["agents", "planning"]) == "abs:agents OR abs:planning"


def test_build_search_query_quotes_multi_word_keywords() -> None:
    assert build_search_query(["cs.AI"], ["tool use"]) == '(cat:cs.AI) AND (abs:"tool use")'


def test_build_arxiv_ingestor_reads_from_config() -> None:
    config = make_sources_config(
        arxiv={"categories": ["cs.AI"], "require_keywords": ["reasoning"]},
    )

    ingestors = build_arxiv_ingestors(config)

    assert len(ingestors) == 1
    assert ingestors[0].source_id == "arxiv"
    assert ingestors[0].categories == ["cs.AI"]
    assert ingestors[0].require_keywords == ["reasoning"]


def test_build_arxiv_ingestors_without_block() -> None:
    assert build_arxiv_ingestors(make_sources_config()) == []


def test_build_arxiv_ingestors_with_no_categories() -> None:
    config = make_sources_config(arxiv={"categories": [], "require_keywords": ["reasoning"]})
    assert build_arxiv_ingestors(config) == []
