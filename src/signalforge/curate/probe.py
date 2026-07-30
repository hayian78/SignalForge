"""Driving `ingest.probe` over a batch of candidates (DESIGN §7.1).

Thin by design. Everything about *how* to fetch and measure a feed or a repo lives
in `ingest/probe.py`; this module only decides which probe a proposal kind needs,
runs the batch concurrently, and guarantees one bad candidate cannot take the run
down with it (CLAUDE.md §7, NEVER rule 12).

### Why retirements are probed too

The obvious reading of "validate before proposing" is that only *additions* need
checking. Retirements need it more: "has this gone quiet?" is precisely the
question a retirement asks, and `items_in_window` and `newest_published_at` answer
it from the live source rather than from our own ingest history — which can look
empty for reasons that have nothing to do with the publisher (a rename, a redirect,
a fetch that has been failing silently for a fortnight).

What a failed probe *means* differs by direction, and that judgment is the
caller's: a feed that 404s is a reason not to show an addition, and a reason to
show a retirement. This module reports either way and decides nothing.

### The fetcher is created here, not passed in

Both callers — the candidate pass and the weekly re-probe of `invalid` rows — want
a short-lived client for a handful of requests, and `HttpFetcher` owns the timeout,
retry, politeness and payload-archiving rules that make probe traffic behave like
every other fetch in the system. Constructing it in one place means neither caller
can get those rules subtly wrong, and a batch with nothing fetchable in it (all
keyword proposals) opens no client at all.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from signalforge.config import SourceDefaults, SourcesConfig, get_secret
from signalforge.ingest.base import HttpFetcher
from signalforge.ingest.probe import SourceProbe, failed_probe, probe_feed, probe_repo
from signalforge.models import ProposalKind

__all__ = [
    "FEED_KINDS",
    "REPO_KINDS",
    "ProbeRequest",
    "github_probe_token",
    "is_probeable",
    "probe_requests",
]

logger = logging.getLogger(__name__)

FEED_KINDS: Final = frozenset({ProposalKind.ADD_RSS, ProposalKind.RETIRE_RSS})
"""Kinds whose locator is a feed URL."""

REPO_KINDS: Final = frozenset({ProposalKind.ADD_GITHUB_REPO, ProposalKind.RETIRE_GITHUB_REPO})
"""Kinds whose locator is an `owner/repo` slug."""


def is_probeable(kind: ProposalKind) -> bool:
    """Whether this kind names something fetchable.

    False for every keyword kind. A keyword has no health to measure — there is
    no URL behind `hackernews.keywords: [agent]` — so those proposals reach the
    digest with `probe = NULL` and are judged on the scout's rationale alone.
    """
    return kind in FEED_KINDS or kind in REPO_KINDS


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """One thing to fetch and measure."""

    kind: ProposalKind
    """Chooses the probe. A kind `is_probeable` says no to yields None."""

    locator: str
    """The feed URL or `owner/repo` slug to fetch.

    Resolved by the caller, not derived here, because it is not always the
    proposal's `dedup_key`: a retirement's key is the source *id* from
    `sources.yaml`, and only the caller holding the parsed config can turn that
    back into the URL to fetch.
    """


def github_probe_token(sources: SourcesConfig) -> str | None:
    """The GitHub PAT for repo probes, or None to probe unauthenticated.

    Read from the environment variable *named* by `github.token_env`, exactly as
    `ingest.build_ingestors` does — never from YAML, never logged, never stored in
    a probe fact (NEVER rule 16). Unauthenticated probing works at 60 req/hr,
    which is far more than a weekly handful of candidates needs, so a missing
    token is not an error and not a warning here.
    """
    if sources.github is None:
        return None
    secret = get_secret(sources.github.token_env)
    return secret.get_secret_value() if secret is not None else None


async def _probe_one(
    fetcher: HttpFetcher,
    request: ProbeRequest,
    *,
    defaults: SourceDefaults,
    token: str | None,
    now: datetime | None,
) -> SourceProbe | None:
    """Run the probe this request needs, converting any escape into a failure.

    `probe_feed` and `probe_repo` already catch everything their fetch and parse
    can raise, so this `except` covers only the case they cannot: a bug in the
    measurement code that runs after the fetch. It is here because the alternative
    — one arithmetic mistake on candidate three aborting a run that has already
    paid for its Opus call and its searches — is the failure CLAUDE.md §7 exists
    to prevent. `asyncio.CancelledError` is a `BaseException` and still propagates,
    so a cancelled run is not silently turned into five failed probes.
    """
    if request.kind in FEED_KINDS:
        probe = probe_feed(
            fetcher,
            request.locator,
            max_item_age_days=defaults.max_item_age_days,
            max_summary_chars=defaults.max_summary_chars,
            now=now,
        )
    elif request.kind in REPO_KINDS:
        probe = probe_repo(
            fetcher,
            request.locator,
            max_item_age_days=defaults.max_item_age_days,
            token=token,
            now=now,
        )
    else:
        return None

    try:
        return await probe
    except Exception as exc:  # noqa: BLE001 - one candidate never kills a curate run
        logger.warning(
            "a probe raised past ingest.probe's own handling",
            extra={"kind": request.kind.value, "locator": request.locator, "error": str(exc)},
        )
        return failed_probe(f"{type(exc).__name__}: {exc}")


async def probe_requests(
    requests: list[ProbeRequest],
    *,
    cache_dir: Path,
    defaults: SourceDefaults,
    token: str | None = None,
    now: datetime | None = None,
) -> list[SourceProbe | None]:
    """Probe every request concurrently, returning results **aligned by index**.

    Positional alignment rather than a dict keyed by locator: two proposals can
    legitimately name the same URL (an addition the operator rejected last month
    and a re-proposal this week), and a dict would silently collapse them into one
    result. The caller zips.

    Concurrency is bounded by `HttpFetcher`'s own semaphore, so this cannot become
    an accidental burst against one host however many candidates a run produces.
    """
    if not any(is_probeable(request.kind) for request in requests):
        # No client, no cache directory created, no politeness budget spent.
        return [None] * len(requests)

    async with HttpFetcher(cache_dir=cache_dir, timeout=defaults.fetch_timeout) as fetcher:
        gathered = await asyncio.gather(
            *(
                _probe_one(fetcher, request, defaults=defaults, token=token, now=now)
                for request in requests
            )
        )
    probes: list[SourceProbe | None] = list(gathered)
    logger.info(
        "probed candidate sources",
        extra={
            "requested": len(requests),
            "fetched": sum(1 for probe in probes if probe is not None),
            "failed": sum(1 for probe in probes if probe is not None and not probe.ok),
        },
    )
    return probes
