"""GitHub releases ingestor tests against captured REST payloads."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from signalforge.ingest.base import HttpFetcher
from signalforge.ingest.github import GithubReleasesIngestor, build_github_ingestors
from signalforge.models import SourceType
from tests.ingest.conftest import MAX_SUMMARY_CHARS, fixture_text, make_sources_config

REPO = "Aider-AI/aider"
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases"
TAGS_URL = f"https://api.github.com/repos/{REPO}/tags"


def _ingestor(*, token: str | None = None, **overrides: object) -> GithubReleasesIngestor:
    """The ingestor under test. `max_summary_chars` is config in production, so
    tests state it explicitly rather than relying on a Python default."""
    kwargs: dict[str, object] = {"max_summary_chars": MAX_SUMMARY_CHARS}
    kwargs.update(overrides)
    return GithubReleasesIngestor(REPO, token, **kwargs)  # type: ignore[arg-type]


@respx.mock
async def test_releases_become_items(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )

    result = await _ingestor().ingest(fetcher)

    assert result.ok
    # 4 releases in, 2 out: one duplicate tag and one draft are dropped.
    assert len(result.items) == 2

    first = result.items[0]
    assert first.source_id == REPO
    assert first.source_type is SourceType.GITHUB
    assert first.external_id == f"{REPO}@v0.86.0"
    assert first.title == f"{REPO} Aider v0.86.0"
    assert first.author == "paul-gauthier"
    assert first.published_at == datetime(2026, 7, 15, 8, 41, 2, tzinfo=UTC)
    assert first.url == "https://github.com/Aider-AI/aider/releases/tag/v0.86.0"


@respx.mock
async def test_release_notes_become_the_summary(fetcher: HttpFetcher) -> None:
    """DESIGN §7: release notes are the summary — what triage reads."""
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )

    result = await _ingestor().ingest(fetcher)

    summary = result.items[0].summary
    assert summary is not None
    assert "--watch-files" in summary


@respx.mock
async def test_release_without_a_name_falls_back_to_the_tag(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )

    result = await _ingestor().ingest(fetcher)

    assert result.items[1].title == f"{REPO} v0.85.2"


@respx.mock
async def test_duplicate_tag_names_collapse(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )

    result = await _ingestor().ingest(fetcher)

    external_ids = [item.external_id for item in result.items]
    assert len(external_ids) == len(set(external_ids))


@respx.mock
async def test_drafts_are_never_ingested(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )

    result = await _ingestor().ingest(fetcher)

    assert all("rc1" not in str(item.external_id) for item in result.items)


# --------------------------------------------------------------------------- #
# Auth — never logged, never hardcoded
# --------------------------------------------------------------------------- #


@respx.mock
async def test_token_is_sent_as_a_bearer_header(fetcher: HttpFetcher) -> None:
    route = respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text="[]")
    )
    respx.get(url__startswith=TAGS_URL).mock(return_value=httpx.Response(200, text="[]"))

    await _ingestor(token="ghp_secret_value").ingest(fetcher)

    assert route.calls.last.request.headers["authorization"] == "Bearer ghp_secret_value"


@respx.mock
async def test_token_never_appears_in_logs(
    fetcher: HttpFetcher, caplog: pytest.LogCaptureFixture
) -> None:
    """NEVER rule 16: a failing request must not leak the credential."""
    caplog.set_level("DEBUG")
    respx.get(url__startswith=RELEASES_URL).mock(return_value=httpx.Response(500))

    result = await _ingestor(token="ghp_super_secret").ingest(fetcher)

    assert not result.ok
    assert "ghp_super_secret" not in caplog.text
    assert all("ghp_super_secret" not in error.message for error in result.errors)


@respx.mock
async def test_works_unauthenticated(fetcher: HttpFetcher) -> None:
    route = respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )

    result = await _ingestor(token=None).ingest(fetcher)

    assert result.ok
    assert "authorization" not in route.calls.last.request.headers


# --------------------------------------------------------------------------- #
# Fallbacks, rate limits, failures
# --------------------------------------------------------------------------- #


@respx.mock
async def test_falls_back_to_tags_when_no_releases(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(return_value=httpx.Response(200, text="[]"))
    respx.get(url__startswith=TAGS_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_tags.json"))
    )

    result = await _ingestor().ingest(fetcher)

    assert result.ok
    assert len(result.items) == 2
    assert result.items[0].external_id == f"{REPO}@v1.4.2"
    assert result.items[0].url == f"https://github.com/{REPO}/releases/tag/v1.4.2"
    assert result.items[0].summary is None  # a tag carries no notes; we invent none


@respx.mock
async def test_falls_back_to_tags_on_404_releases(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(return_value=httpx.Response(404))
    respx.get(url__startswith=TAGS_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_tags.json"))
    )

    result = await _ingestor().ingest(fetcher)

    assert result.ok
    assert len(result.items) == 2


@respx.mock
async def test_304_yields_no_items(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(
            200, text=fixture_text("github_releases.json"), headers={"etag": '"v1"'}
        )
    )
    first = await _ingestor().ingest(fetcher)
    assert len(first.items) == 2
    fetcher.validators.commit()

    respx.get(url__startswith=RELEASES_URL).mock(return_value=httpx.Response(304))
    result = await _ingestor().ingest(fetcher)

    assert result.items == []
    assert result.ok


@respx.mock
async def test_tags_only_repo_keeps_working_on_later_runs(fetcher: HttpFetcher) -> None:
    """Regression: an empty `/releases` payload must not cache away the fallback.

    An empty `/releases` array has a perfectly stable ETag. Caching it meant run
    2 got a 304, returned early, and never probed `/tags` again — a tags-only
    repo went silently dark forever, with no error recorded. Dedup made the loss
    invisible, which is exactly why this needs a two-run test rather than a
    single-run one.
    """
    releases = respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text="[]", headers={"etag": '"stable-empty"'})
    )
    respx.get(url__startswith=TAGS_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_tags.json"))
    )

    first = await _ingestor().ingest(fetcher)
    # Commit as a successful caller would: without the empty-payload
    # invalidation, this is exactly what would arm the fatal 304 on run 2.
    fetcher.validators.commit()
    second = await _ingestor().ingest(fetcher)

    assert [item.external_id for item in first.items] == [f"{REPO}@v1.4.2", f"{REPO}@v1.4.1"]
    # The second run must see the tags again, not an empty 304 short-circuit.
    assert [item.external_id for item in second.items] == [f"{REPO}@v1.4.2", f"{REPO}@v1.4.1"]
    assert second.ok
    # The empty payload's validators were dropped, so run 2 re-probed releases
    # unconditionally rather than sending If-None-Match.
    assert "if-none-match" not in releases.calls.last.request.headers


@respx.mock
async def test_a_repo_with_releases_still_304s_and_skips_tags(fetcher: HttpFetcher) -> None:
    """The fix must not cost the 304 benefit for repos that do publish releases."""
    respx.get(url__startswith=RELEASES_URL).mock(
        side_effect=[
            httpx.Response(
                200, text=fixture_text("github_releases.json"), headers={"etag": '"v1"'}
            ),
            httpx.Response(304),
        ]
    )
    tags = respx.get(url__startswith=TAGS_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_tags.json"))
    )

    await _ingestor().ingest(fetcher)
    fetcher.validators.commit()
    second = await _ingestor().ingest(fetcher)

    assert second.items == []
    assert second.ok
    assert tags.call_count == 0  # never fell back for a repo that has releases


@respx.mock
async def test_all_draft_releases_do_not_trigger_the_tags_fallback(fetcher: HttpFetcher) -> None:
    """A payload of nothing but drafts is still a repo that publishes releases."""
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "tag_name": "v1.0.0-rc1",
                    "draft": True,
                    "html_url": "https://github.com/Aider-AI/aider/releases/tag/v1.0.0-rc1",
                }
            ],
        )
    )
    tags = respx.get(url__startswith=TAGS_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_tags.json"))
    )

    result = await _ingestor().ingest(fetcher)

    assert result.items == []
    assert tags.call_count == 0


@respx.mock
async def test_rate_limit_403_is_an_error_record_not_a_crash(
    fetcher: HttpFetcher, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING")
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1784136000"},
            json={"message": "API rate limit exceeded"},
        )
    )

    result = await _ingestor().ingest(fetcher)

    assert result.items == []
    assert len(result.errors) == 1
    assert result.errors[0].source_id == REPO
    assert "rate limit nearly exhausted" in caplog.text


@respx.mock
async def test_malformed_json_is_an_error_record(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(return_value=httpx.Response(200, text="{not json"))

    result = await _ingestor().ingest(fetcher)

    assert result.items == []
    assert len(result.errors) == 1


@respx.mock
async def test_unexpected_json_shape_is_an_error_record(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, json={"message": "Not Found"})
    )

    result = await _ingestor().ingest(fetcher)

    assert len(result.errors) == 1


@respx.mock
async def test_malformed_release_entries_are_skipped(fetcher: HttpFetcher) -> None:
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"tag_name": None, "html_url": "https://github.com/x/y/releases/tag/v1"},
                {"tag_name": "v2", "html_url": None},
                {
                    "tag_name": "v3",
                    "html_url": "https://github.com/Aider-AI/aider/releases/tag/v3",
                    "name": "Good one",
                },
            ],
        )
    )

    result = await _ingestor().ingest(fetcher)

    assert len(result.items) == 1
    assert result.items[0].external_id == f"{REPO}@v3"


# --------------------------------------------------------------------------- #
# Construction from config
# --------------------------------------------------------------------------- #


def test_build_github_ingestors_from_config() -> None:
    config = make_sources_config(
        github={
            "token_env": "GITHUB_TOKEN",
            "releases": ["Aider-AI/aider", "ollama/ollama"],
            "awesome_lists": ["e2b-dev/awesome-ai-agents"],
        }
    )

    ingestors = build_github_ingestors(config, None)

    # awesome_lists is Phase 1 — configured, but not ingested (NEVER rule 15).
    assert [i.source_id for i in ingestors] == ["Aider-AI/aider", "ollama/ollama"]


def test_build_github_ingestors_without_github_block() -> None:
    assert build_github_ingestors(make_sources_config(), None) == []
