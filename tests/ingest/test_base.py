"""Tests for the shared HTTP fetcher and the failure-isolation runner.

The two load-bearing behaviours proved here are the ones DESIGN §7 and
CLAUDE.md §7 rest on: a 304 costs nothing and yields nothing, and a source that
raises cannot take the run down with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from signalforge.ingest.base import (
    FetchError,
    HttpFetcher,
    IngestError,
    Ingestor,
    IngestResult,
    run_ingestors,
)
from signalforge.models import Item, SourceType

FEED_URL = "https://example.com/feed.xml"


# --------------------------------------------------------------------------- #
# Conditional GET
# --------------------------------------------------------------------------- #


@respx.mock
async def test_first_fetch_stores_etag_and_archives_payload(
    fetcher: HttpFetcher, cache_dir: Path
) -> None:
    route = respx.get(FEED_URL).mock(
        return_value=httpx.Response(
            200,
            content=b"<feed>hello</feed>",
            headers={"etag": '"abc123"', "last-modified": "Wed, 15 Jul 2026 10:00:00 GMT"},
        )
    )

    response = await fetcher.get(FEED_URL, source_id="example")
    fetcher.validators.commit()  # stands in for the caller's successful persist

    assert response is not None
    assert response.content == b"<feed>hello</feed>"
    assert route.call_count == 1

    # The archive path is relative to the cache root and actually holds the bytes.
    assert response.raw_path is not None
    assert not Path(response.raw_path).is_absolute()
    assert (cache_dir / response.raw_path).read_bytes() == b"<feed>hello</feed>"

    # ETag state is a sidecar under the cache root, not a DB table.
    meta_files = list(cache_dir.glob("example/_meta/*.json"))
    assert len(meta_files) == 1
    stored = json.loads(meta_files[0].read_text())
    assert stored["etag"] == '"abc123"'
    assert stored["last_modified"] == "Wed, 15 Jul 2026 10:00:00 GMT"


@respx.mock
async def test_second_fetch_sends_validators_and_304_returns_none(fetcher: HttpFetcher) -> None:
    """A 304 must cost nothing and yield no items (DESIGN §7)."""
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"body", headers={"etag": '"v1"'})
    )
    await fetcher.get(FEED_URL, source_id="example")
    fetcher.validators.commit()

    conditional = respx.get(FEED_URL).mock(return_value=httpx.Response(304))
    result = await fetcher.get(FEED_URL, source_id="example")

    assert result is None
    request = conditional.calls.last.request
    assert request.headers["if-none-match"] == '"v1"'


@respx.mock
async def test_304_archives_nothing(fetcher: HttpFetcher, cache_dir: Path) -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(304))

    assert await fetcher.get(FEED_URL, source_id="example") is None
    assert list(cache_dir.glob("example/*/*.raw")) == []


@respx.mock
async def test_conditional_false_skips_validators(fetcher: HttpFetcher) -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"body", headers={"etag": '"v1"'})
    )
    await fetcher.get(FEED_URL, source_id="example")
    fetcher.validators.commit()

    route = respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=b"body"))
    await fetcher.get(FEED_URL, source_id="example", conditional=False)

    assert "if-none-match" not in route.calls.last.request.headers


@respx.mock
async def test_cache_key_separates_state_for_one_source(
    fetcher: HttpFetcher, cache_dir: Path
) -> None:
    """HN runs many queries under one source_id; their validators must not collide."""
    respx.get("https://example.com/a").mock(
        return_value=httpx.Response(200, content=b"a", headers={"etag": '"a"'})
    )
    respx.get("https://example.com/b").mock(
        return_value=httpx.Response(200, content=b"b", headers={"etag": '"b"'})
    )

    await fetcher.get("https://example.com/a", source_id="hn", cache_key="hn:one")
    await fetcher.get("https://example.com/b", source_id="hn", cache_key="hn:two")
    fetcher.validators.commit()

    assert len(list(cache_dir.glob("hn/_meta/*.json"))) == 2


@respx.mock
async def test_corrupt_metadata_is_discarded_not_fatal(
    fetcher: HttpFetcher, cache_dir: Path
) -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"body", headers={"etag": '"v1"'})
    )
    await fetcher.get(FEED_URL, source_id="example")
    fetcher.validators.commit()

    meta = next(iter(cache_dir.glob("example/_meta/*.json")))
    meta.write_text("{ this is not json")

    route = respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=b"body2"))
    response = await fetcher.get(FEED_URL, source_id="example")

    assert response is not None
    assert "if-none-match" not in route.calls.last.request.headers


# --------------------------------------------------------------------------- #
# Validator staging — nothing durable until the caller confirms
# --------------------------------------------------------------------------- #


@respx.mock
async def test_fetch_stages_validators_without_writing_them(
    fetcher: HttpFetcher, cache_dir: Path
) -> None:
    """A validator written at fetch time would promise something we can't yet."""
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"body", headers={"etag": '"v1"'})
    )

    await fetcher.get(FEED_URL, source_id="example")

    assert fetcher.validators.pending_count() == 1
    assert fetcher.validators.pending_sources() == {"example"}
    assert list(cache_dir.glob("example/_meta/*.json")) == []  # not yet durable


@respx.mock
async def test_uncommitted_validators_mean_an_unconditional_refetch(fetcher: HttpFetcher) -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"body", headers={"etag": '"v1"'})
    )
    await fetcher.get(FEED_URL, source_id="example")

    route = respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=b"body"))
    await fetcher.get(FEED_URL, source_id="example")

    assert "if-none-match" not in route.calls.last.request.headers


@respx.mock
async def test_commit_makes_validators_durable(fetcher: HttpFetcher, cache_dir: Path) -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"body", headers={"etag": '"v1"'})
    )
    await fetcher.get(FEED_URL, source_id="example")

    assert fetcher.validators.commit() == 1
    assert len(list(cache_dir.glob("example/_meta/*.json"))) == 1
    assert fetcher.validators.pending_count() == 0


@respx.mock
async def test_commit_selects_by_source_id(fetcher: HttpFetcher, cache_dir: Path) -> None:
    respx.get("https://a.example/feed").mock(
        return_value=httpx.Response(200, content=b"a", headers={"etag": '"a"'})
    )
    respx.get("https://b.example/feed").mock(
        return_value=httpx.Response(200, content=b"b", headers={"etag": '"b"'})
    )
    await fetcher.get("https://a.example/feed", source_id="source-a")
    await fetcher.get("https://b.example/feed", source_id="source-b")

    assert fetcher.validators.commit(["source-a"]) == 1

    assert len(list(cache_dir.glob("source-a/_meta/*.json"))) == 1
    assert list(cache_dir.glob("source-b/_meta/*.json")) == []
    # The unconfirmed source stays staged rather than being dropped.
    assert fetcher.validators.pending_sources() == {"source-b"}


@respx.mock
async def test_unwritable_sidecar_does_not_fail_the_commit(
    fetcher: HttpFetcher, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A validator that can't be stored costs a refetch, never the run."""
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"body", headers={"etag": '"v1"'})
    )
    await fetcher.get(FEED_URL, source_id="example")

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", explode)

    assert fetcher.validators.commit() == 0  # reported as not-written, not raised


async def _commit_a_real_sidecar(fetcher: HttpFetcher, cache_dir: Path) -> Path:
    """Leave one genuinely durable validator on disk, and return its path."""
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"body", headers={"etag": '"v1"'})
    )
    await fetcher.get(FEED_URL, source_id="example")
    fetcher.validators.commit()
    sidecar = next(iter(cache_dir.glob("example/_meta/*.json")))
    assert sidecar.is_file()
    return sidecar


@respx.mock
async def test_response_without_validators_stages_the_deletion(
    fetcher: HttpFetcher, cache_dir: Path
) -> None:
    """Deletions stage like writes: nothing touches disk before commit.

    This is what makes a dry run non-mutating — asserted against a real
    committed sidecar, because an in-memory-only check would miss the point.
    """
    sidecar = await _commit_a_real_sidecar(fetcher, cache_dir)

    # The server stops sending validators: the sidecar is now wrong...
    respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=b"body2"))
    await fetcher.get(FEED_URL, source_id="example")

    # ...but it survives until the caller confirms.
    assert sidecar.is_file()
    assert fetcher.validators.pending_count() == 1
    # The staged deletion is honored in-run, so we stop offering the dead ETag.
    assert fetcher.validators.read("example", sidecar.stem) == {}

    fetcher.validators.commit()
    assert not sidecar.exists()


@respx.mock
async def test_uncommitted_deletion_leaves_the_cache_untouched(
    fetcher: HttpFetcher, cache_dir: Path
) -> None:
    """A dry run must not mutate the validator cache it promised not to touch."""
    sidecar = await _commit_a_real_sidecar(fetcher, cache_dir)
    before = sidecar.read_text()

    respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=b"body2"))
    await fetcher.get(FEED_URL, source_id="example")
    fetcher.invalidate(FEED_URL, source_id="example")
    # No commit — the caller walked away.

    assert sidecar.is_file()
    assert sidecar.read_text() == before


@respx.mock
async def test_invalidate_stages_removal_of_a_committed_validator(
    fetcher: HttpFetcher, cache_dir: Path
) -> None:
    sidecar = await _commit_a_real_sidecar(fetcher, cache_dir)
    await fetcher.get(FEED_URL, source_id="example")

    fetcher.invalidate(FEED_URL, source_id="example")

    assert fetcher.validators.pending_count() == 1  # staged, not applied
    assert sidecar.is_file()

    fetcher.validators.commit()
    assert not sidecar.exists()
    assert fetcher.validators.pending_count() == 0


# --------------------------------------------------------------------------- #
# Retries and errors
# --------------------------------------------------------------------------- #


@respx.mock
async def test_retries_then_succeeds(fetcher: HttpFetcher) -> None:
    route = respx.get(FEED_URL).mock(
        side_effect=[
            httpx.Response(503, headers={"retry-after": "0"}),
            httpx.Response(200, content=b"recovered"),
        ]
    )

    response = await fetcher.get(FEED_URL, source_id="example")

    assert response is not None
    assert response.content == b"recovered"
    assert route.call_count == 2


@respx.mock
async def test_retry_exhaustion_raises_fetch_error(fetcher: HttpFetcher) -> None:
    route = respx.get(FEED_URL).mock(return_value=httpx.Response(429, headers={"retry-after": "0"}))

    with pytest.raises(FetchError) as excinfo:
        await fetcher.get(FEED_URL, source_id="example")

    assert excinfo.value.status_code == 429
    assert route.call_count == 2  # max_attempts from the fixture


@respx.mock
async def test_non_retryable_status_fails_immediately(fetcher: HttpFetcher) -> None:
    route = respx.get(FEED_URL).mock(return_value=httpx.Response(404))

    with pytest.raises(FetchError) as excinfo:
        await fetcher.get(FEED_URL, source_id="example")

    assert excinfo.value.status_code == 404
    assert route.call_count == 1  # a 404 is never retried


@respx.mock
async def test_transport_error_becomes_fetch_error(fetcher: HttpFetcher) -> None:
    respx.get(FEED_URL).mock(side_effect=httpx.ConnectError("dns failure"))

    with pytest.raises(FetchError):
        await fetcher.get(FEED_URL, source_id="example")


@respx.mock
async def test_source_id_with_slash_is_path_safe(fetcher: HttpFetcher, cache_dir: Path) -> None:
    """GitHub source ids are `owner/repo`; they must not escape the cache root."""
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"x", headers={"etag": '"e"'})
    )

    response = await fetcher.get(FEED_URL, source_id="Aider-AI/aider")

    assert response is not None
    assert response.raw_path is not None
    assert (cache_dir / response.raw_path).is_file()
    assert (cache_dir / response.raw_path).resolve().is_relative_to(cache_dir.resolve())


# --------------------------------------------------------------------------- #
# Failure isolation
# --------------------------------------------------------------------------- #


class _StubIngestor:
    """Minimal Ingestor implementation for runner tests."""

    def __init__(
        self,
        source_id: str,
        *,
        items: list[Item] | None = None,
        raises: Exception | None = None,
        errors: list[IngestError] | None = None,
    ) -> None:
        self.source_id = source_id
        self.source_type = SourceType.RSS
        self._items = items or []
        self._raises = raises
        self._errors = errors or []

    async def ingest(self, fetcher: HttpFetcher) -> IngestResult:
        if self._raises is not None:
            raise self._raises
        return IngestResult(items=self._items, errors=self._errors)


def _item(source_id: str) -> Item:
    return Item(
        source_id=source_id,
        source_type=SourceType.RSS,
        url=f"https://example.com/{source_id}",
        title=f"Post from {source_id}",
    )


async def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(_StubIngestor("a"), Ingestor)


async def test_one_raising_source_does_not_kill_the_others(fetcher: HttpFetcher) -> None:
    """NEVER rule 12, proved structurally."""
    ingestors: list[Ingestor] = [
        _StubIngestor("good-1", items=[_item("good-1")]),
        _StubIngestor("broken", raises=RuntimeError("feed exploded")),
        _StubIngestor("good-2", items=[_item("good-2")]),
    ]

    result = await run_ingestors(ingestors, fetcher)

    assert [item.source_id for item in result.items] == ["good-1", "good-2"]
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error.source_id == "broken"
    assert error.error_type == "RuntimeError"
    assert "feed exploded" in error.message
    assert not result.ok


async def test_errors_survive_as_json_records(fetcher: HttpFetcher) -> None:
    """`runs.errors` is JSON — the record must serialize without a custom encoder."""
    result = await run_ingestors([_StubIngestor("broken", raises=ValueError("bad"))], fetcher)

    records = [error.as_record() for error in result.errors]
    decoded = json.loads(json.dumps(records))

    assert decoded[0]["source_id"] == "broken"
    assert decoded[0]["source_type"] == "rss"
    assert decoded[0]["error_type"] == "ValueError"


async def test_partial_result_keeps_both_items_and_errors(fetcher: HttpFetcher) -> None:
    partial = _StubIngestor(
        "hn",
        items=[_item("hn")],
        errors=[
            IngestError.from_exception(
                RuntimeError("one query failed"), source_id="hn", source_type=SourceType.HN
            )
        ],
    )

    result = await run_ingestors([partial], fetcher)

    assert len(result.items) == 1
    assert len(result.errors) == 1


async def test_no_ingestors_is_an_empty_result(fetcher: HttpFetcher) -> None:
    result = await run_ingestors([], fetcher)
    assert result.items == []
    assert result.ok
