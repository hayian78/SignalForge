"""Tests for `curate/probe.py` — driving `ingest.probe` over a batch.

The batch is the point. One candidate must never be able to cost the run its other
candidates, because by the time probing starts the Opus call and its searches are
already paid for (CLAUDE.md §7, NEVER rule 12).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from signalforge.curate.probe import (
    ProbeRequest,
    github_probe_token,
    is_probeable,
    probe_requests,
)
from signalforge.ingest.probe import SourceProbe
from signalforge.models import ProposalKind
from tests.curate.conftest import fixture_bytes, make_sources

FEED_URL = "https://simonwillison.net/atom/everything/"
DEAD_URL = "https://the-batch.example.com/rss.xml"
RELEASES_URL = "https://api.github.com/repos/anthropics/claude-code/releases"
FROZEN_NOW = datetime(2026, 7, 28, 6, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ProposalKind.ADD_RSS, True),
        (ProposalKind.RETIRE_RSS, True),
        (ProposalKind.ADD_GITHUB_REPO, True),
        (ProposalKind.RETIRE_GITHUB_REPO, True),
        (ProposalKind.ADD_HN_KEYWORD, False),
        (ProposalKind.REMOVE_HN_KEYWORD, False),
        (ProposalKind.ADD_ARXIV_KEYWORD, False),
        (ProposalKind.REMOVE_ARXIV_KEYWORD, False),
    ],
)
def test_only_kinds_with_something_to_fetch_are_probeable(
    kind: ProposalKind, expected: bool
) -> None:
    """Pinned per kind so adding a ninth kind cannot silently default to unprobed."""
    assert is_probeable(kind) is expected


@respx.mock
async def test_retirements_are_probed_too(cache_dir: Path) -> None:
    """ "Has this gone quiet?" is what a retirement claims, so it gets measured.

    Probing only additions is the obvious reading of "validate before proposing"
    and the wrong one: our own ingest history can look empty for reasons that have
    nothing to do with the publisher.
    """
    route = respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )

    probes = await probe_requests(
        [ProbeRequest(kind=ProposalKind.RETIRE_RSS, locator=FEED_URL)],
        cache_dir=cache_dir,
        defaults=make_sources().defaults,
        now=FROZEN_NOW,
    )

    assert route.called
    assert probes[0] is not None
    assert probes[0].ok is True


async def test_a_batch_of_keyword_kinds_opens_no_client(cache_dir: Path, tmp_path: Path) -> None:
    """No fetchable request means no HTTP client and no cache directory touched.

    Asserted through `respx` being un-mocked: any real request here would fail the
    test by attempting a live connection, which is the hermeticity NEVER rule 13
    requires.
    """
    probes = await probe_requests(
        [
            ProbeRequest(kind=ProposalKind.ADD_HN_KEYWORD, locator="agent"),
            ProbeRequest(kind=ProposalKind.REMOVE_ARXIV_KEYWORD, locator="memory"),
        ],
        cache_dir=tmp_path / "never-created",
        defaults=make_sources().defaults,
    )

    assert probes == [None, None]
    assert not (tmp_path / "never-created").exists()


@respx.mock
async def test_results_line_up_with_requests_including_the_unprobed_ones(
    cache_dir: Path,
) -> None:
    """Index alignment is the contract; the caller zips.

    A dict keyed by locator would silently collapse two proposals that legitimately
    name the same URL — a rejected addition and a re-proposal — into one result.
    """
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )
    respx.get(DEAD_URL).mock(return_value=httpx.Response(404))
    respx.get(RELEASES_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "tag_name": "v2.0.0",
                    "published_at": "2026-07-20T00:00:00Z",
                    "body": "Release notes with real content.",
                    "draft": False,
                }
            ],
        )
    )

    requests = [
        ProbeRequest(kind=ProposalKind.ADD_HN_KEYWORD, locator="agent"),
        ProbeRequest(kind=ProposalKind.ADD_RSS, locator=FEED_URL),
        ProbeRequest(kind=ProposalKind.ADD_RSS, locator=DEAD_URL),
        ProbeRequest(kind=ProposalKind.ADD_GITHUB_REPO, locator="anthropics/claude-code"),
    ]
    probes = await probe_requests(
        requests, cache_dir=cache_dir, defaults=make_sources().defaults, now=FROZEN_NOW
    )

    assert len(probes) == len(requests)
    assert probes[0] is None
    assert probes[1] is not None and probes[1].ok is True
    assert probes[2] is not None and probes[2].ok is False
    assert probes[3] is not None and probes[3].label == "v2.0.0"


@respx.mock
async def test_one_dead_candidate_does_not_stop_the_others(cache_dir: Path) -> None:
    respx.get(DEAD_URL).mock(return_value=httpx.Response(404))
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )

    probes = await probe_requests(
        [
            ProbeRequest(kind=ProposalKind.ADD_RSS, locator=DEAD_URL),
            ProbeRequest(kind=ProposalKind.ADD_RSS, locator=FEED_URL),
        ],
        cache_dir=cache_dir,
        defaults=make_sources().defaults,
        now=FROZEN_NOW,
    )

    assert [probe.ok for probe in probes if probe is not None] == [False, True]


async def test_a_probe_that_raises_past_ingest_becomes_a_failed_probe(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one failure `ingest/probe.py` cannot catch: a bug in its own measurement.

    Simulated by making `probe_feed` raise. Without the wrapper this aborts the
    whole batch — and with it, a run that has already paid for its searches.
    """

    async def explode(*_args: object, **_kwargs: object) -> SourceProbe:
        raise ZeroDivisionError("median of an empty list")

    monkeypatch.setattr("signalforge.curate.probe.probe_feed", explode)

    probes = await probe_requests(
        [
            ProbeRequest(kind=ProposalKind.ADD_RSS, locator=FEED_URL),
            ProbeRequest(kind=ProposalKind.ADD_HN_KEYWORD, locator="agent"),
        ],
        cache_dir=cache_dir,
        defaults=make_sources().defaults,
    )

    assert probes[0] is not None
    assert probes[0].ok is False
    assert probes[0].error is not None
    assert "ZeroDivisionError" in probes[0].error
    assert probes[1] is None


def test_the_github_token_is_read_from_the_env_var_the_config_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SF_TEST_PAT", "ghp_notarealtoken")
    sources = make_sources(github={"token_env": "SF_TEST_PAT", "releases": ["owner/repo"]})

    assert github_probe_token(sources) == "ghp_notarealtoken"


def test_a_missing_github_token_probes_unauthenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    """60 req/hr is ample for a weekly handful, so this is not an error."""
    monkeypatch.delenv("SF_TEST_PAT", raising=False)
    sources = make_sources(github={"token_env": "SF_TEST_PAT", "releases": ["owner/repo"]})

    assert github_probe_token(sources) is None


def test_no_github_block_needs_no_token() -> None:
    assert github_probe_token(make_sources()) is None
