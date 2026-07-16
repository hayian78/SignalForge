"""End-to-end tests for `ingest_all` — the entry point `cli.py` calls.

Also validates the *shipped* `config/sources.yaml` and `config/interests.yaml`:
those files are data the pipeline depends on (CLAUDE.md §4), so a typo in them
is a test failure, not a 6am cron surprise.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from signalforge.config import load_interests, load_sources
from signalforge.ingest import build_ingestors, ingest_all
from tests.ingest.conftest import fixture_bytes, fixture_text, make_sources_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

FEED_URL = "https://example.com/feed.xml"
RELEASES_URL = "https://api.github.com/repos/Aider-AI/aider/releases"
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"


def _full_config() -> object:
    return make_sources_config(
        rss=[{"id": "example-blog", "url": FEED_URL}],
        github={"token_env": "SIGNALFORGE_TEST_TOKEN", "releases": ["Aider-AI/aider"]},
        hackernews={"keywords": ["mcp"]},
    )


@respx.mock
async def test_ingest_all_collects_from_every_source(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SIGNALFORGE_TEST_TOKEN", raising=False)
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )
    respx.get(HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("hn_front_page.json"))
    )

    result = await ingest_all(_full_config(), cache_dir=cache_dir)  # type: ignore[arg-type]

    source_types = {item.source_type.value for item in result.items}
    assert source_types == {"rss", "github", "hn"}
    assert result.ok


@respx.mock
async def test_a_dead_source_does_not_stop_the_run(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of CLAUDE.md §7, at the level the CLI sees it."""
    monkeypatch.delenv("SIGNALFORGE_TEST_TOKEN", raising=False)
    respx.get(FEED_URL).mock(side_effect=httpx.ConnectError("dns is down"))
    respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )
    respx.get(HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("hn_front_page.json"))
    )

    result = await ingest_all(_full_config(), cache_dir=cache_dir)  # type: ignore[arg-type]

    assert not result.ok
    assert [error.source_id for error in result.errors] == ["example-blog"]
    # Everything else still landed.
    assert {item.source_type.value for item in result.items} == {"github", "hn"}

    # And the errors serialize straight into `runs.errors`.
    json.dumps([error.as_record() for error in result.errors])


@respx.mock
async def test_token_is_read_from_the_env_var_named_by_config(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGNALFORGE_TEST_TOKEN", "ghp_from_env")
    route = respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(200, text=fixture_text("github_releases.json"))
    )
    respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=b"<feed></feed>"))
    respx.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))

    await ingest_all(_full_config(), cache_dir=cache_dir)  # type: ignore[arg-type]

    assert route.calls.last.request.headers["authorization"] == "Bearer ghp_from_env"


async def test_missing_token_is_a_warning_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("SIGNALFORGE_TEST_TOKEN", raising=False)
    caplog.set_level("WARNING")

    ingestors = build_ingestors(_full_config())  # type: ignore[arg-type]

    assert len(ingestors) == 3
    assert "unauthenticated" in caplog.text


async def test_empty_config_ingests_nothing(cache_dir: Path) -> None:
    result = await ingest_all(make_sources_config(), cache_dir=cache_dir)
    assert result.items == []
    assert result.ok


# --------------------------------------------------------------------------- #
# The validator seam: at-least-once across a crash
# --------------------------------------------------------------------------- #


def _rss_only_config() -> object:
    return make_sources_config(rss=[{"id": "example-blog", "url": FEED_URL}])


@respx.mock
async def test_persistence_failure_refetches_and_recovers_items(cache_dir: Path) -> None:
    """The seam this whole mechanism exists for.

    Simulates the process dying (or a DB write failing) after a successful fetch
    but before the items are persisted: `commit_validators()` is never reached.
    The next run must refetch unconditionally — a 304 here would return nothing
    and lose the items permanently, defeating DESIGN §14's self-healing.
    """
    route = respx.get(FEED_URL).mock(
        return_value=httpx.Response(
            200, content=fixture_bytes("simonwillison_atom.xml"), headers={"etag": '"v1"'}
        )
    )

    first = await ingest_all(_rss_only_config(), cache_dir=cache_dir)  # type: ignore[arg-type]
    assert len(first.items) == 2
    # ... and here the process dies. No commit_validators() call.

    second = await ingest_all(_rss_only_config(), cache_dir=cache_dir)  # type: ignore[arg-type]

    assert "if-none-match" not in route.calls.last.request.headers
    assert len(second.items) == 2  # the items are recovered, not lost
    assert route.call_count == 2


@respx.mock
async def test_committed_validators_earn_the_304(cache_dir: Path) -> None:
    """The happy path keeps DESIGN §7's free 304s once persistence is confirmed."""
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(
            200, content=fixture_bytes("simonwillison_atom.xml"), headers={"etag": '"v1"'}
        )
    )
    first = await ingest_all(_rss_only_config(), cache_dir=cache_dir)  # type: ignore[arg-type]

    assert first.commit_validators() == 1

    route = respx.get(FEED_URL).mock(return_value=httpx.Response(304))
    second = await ingest_all(_rss_only_config(), cache_dir=cache_dir)  # type: ignore[arg-type]

    assert route.calls.last.request.headers["if-none-match"] == '"v1"'
    assert second.items == []
    assert second.ok


@respx.mock
async def test_partial_commit_isolates_one_failed_source(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """8 sources persisting must not be punished for a 9th that didn't.

    CLAUDE.md §7 failure isolation, applied to the persistence seam.
    """
    monkeypatch.delenv("SIGNALFORGE_TEST_TOKEN", raising=False)
    feed = respx.get(FEED_URL).mock(
        return_value=httpx.Response(
            200, content=fixture_bytes("simonwillison_atom.xml"), headers={"etag": '"feed-v1"'}
        )
    )
    releases = respx.get(url__startswith=RELEASES_URL).mock(
        return_value=httpx.Response(
            200, text=fixture_text("github_releases.json"), headers={"etag": '"gh-v1"'}
        )
    )
    respx.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))

    run = await ingest_all(_full_config(), cache_dir=cache_dir)  # type: ignore[arg-type]
    # "hn" is present as a staged *deletion*: its response carried no ETag, so
    # there is nothing to validate against next run.
    assert run.validators.pending_sources() == {"example-blog", "Aider-AI/aider", "hn"}

    # The blog's rows failed to persist; GitHub's landed. Confirm only GitHub.
    committed = run.commit_validators(["Aider-AI/aider"])
    assert committed == 1

    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )
    respx.get(url__startswith=RELEASES_URL).mock(return_value=httpx.Response(304))
    second = await ingest_all(_full_config(), cache_dir=cache_dir)  # type: ignore[arg-type]

    # The uncommitted source refetches unconditionally and recovers its items...
    assert "if-none-match" not in feed.calls.last.request.headers
    assert any(item.source_id == "example-blog" for item in second.items)
    # ...while the committed one keeps its cheap 304 and yields nothing new.
    assert releases.calls.last.request.headers["if-none-match"] == '"gh-v1"'
    assert not any(item.source_id == "Aider-AI/aider" for item in second.items)


@respx.mock
async def test_commit_is_idempotent(cache_dir: Path) -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(
            200, content=fixture_bytes("simonwillison_atom.xml"), headers={"etag": '"v1"'}
        )
    )
    run = await ingest_all(_rss_only_config(), cache_dir=cache_dir)  # type: ignore[arg-type]

    assert run.commit_validators() == 1
    assert run.commit_validators() == 0  # already durable; not written twice
    assert run.validators.pending_count() == 0


@respx.mock
async def test_error_records_and_source_ids_helpers(cache_dir: Path) -> None:
    respx.get(FEED_URL).mock(side_effect=httpx.ConnectError("down"))

    run = await ingest_all(_rss_only_config(), cache_dir=cache_dir)  # type: ignore[arg-type]

    assert run.source_ids == set()
    assert json.loads(json.dumps(run.error_records()))[0]["source_id"] == "example-blog"


# --------------------------------------------------------------------------- #
# The shipped config files
# --------------------------------------------------------------------------- #


def test_shipped_sources_yaml_validates() -> None:
    config = load_sources(CONFIG_DIR)

    assert config.defaults.fetch_timeout > 0
    assert config.defaults.min_hn_points >= 0
    assert config.rss
    assert config.github is not None
    assert config.github.token_env == "GITHUB_TOKEN"
    assert "Aider-AI/aider" in config.github.releases
    assert config.hackernews is not None
    assert "mcp" in config.hackernews.keywords


def test_shipped_sources_yaml_carries_no_inline_secret() -> None:
    """NEVER rule 16: `token_env` names a variable; it never holds a token."""
    raw = (CONFIG_DIR / "sources.yaml").read_text(encoding="utf-8")
    assert "ghp_" not in raw
    assert "github_pat_" not in raw


def test_shipped_interests_yaml_validates() -> None:
    interests = load_interests(CONFIG_DIR)

    assert interests.thresholds.weekly_min_signal == 3
    assert interests.thresholds.weekly_min_relevance == 3
    assert interests.thresholds.weekly_min_total == 10
    assert "agents.mcp" in interests.priority_topics
    assert "crypto" in interests.ignore.topics
    assert interests.architecture_philosophy


def test_shipped_config_builds_every_phase0_ingestor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    config = load_sources(CONFIG_DIR)

    ingestors = build_ingestors(config)

    # one per feed + one per release repo + one HN
    expected = len(config.rss) + len(config.github.releases if config.github else []) + 1
    assert len(ingestors) == expected
    assert len({i.source_id for i in ingestors}) == len(ingestors)
