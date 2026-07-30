"""Health probes for candidate sources — facts only, no judgment (DESIGN §7.1).

Adaptive source curation proposes feeds and repos the operator has never fetched.
Before any candidate reaches a digest for approval, this module answers the
mechanical questions: does it resolve, does it parse, how much has it published
lately, and how much text does an entry actually carry.

It lives in `ingest/` because fetching and parsing are this package's job and
nothing else's — `curate/` calls in here rather than growing a second HTTP path
with its own timeout and retry rules. Two boundaries hold:

* **No LLM.** Same as every other module here (CLAUDE.md §2). Measuring is
  deterministic; deciding what the measurements *mean* happens elsewhere.
* **No writes.** A probe produces a `SourceProbe` and nothing else. It creates no
  `items` row, so probing a candidate cannot smuggle content into the pipeline
  ahead of the operator approving its source.

### Why probes report rather than reject

`ok` is False only for the unambiguous mechanical failures: the fetch failed, or
nothing parseable came back. Everything else — a thin feed, a slow feed, a feed
of teaser stubs — is reported as a number and left to the human at the gate.

That is a deliberate narrowing of "validate before proposing". The two failures
already recorded by hand in `sources.yaml` split cleanly along this line:
`the-batch` (no feed exists at any plausible path) is mechanical and is caught
here; `stratechery` (feed parses, but entries are teasers because it is
members-only) is a judgment about whether that content is worth a slot. Encoding
the second as a threshold would mean inventing a magic number for "enough text"
and hiding a real decision behind it — and under NEVER rule 6 that number would
have to be a `curation:` knob, which `CurationConfig` deliberately does not have.

**What the narrowing costs, precisely:** a stub candidate now consumes one of
`max_proposals_per_run` and a few lines of digest space. It does not save or spend
scout tokens either way — those are sunk before the probe runs. The trade is
worth it because the opposite failure is worse: a machine-invented threshold
silently dropping a publication the operator would have wanted, with no record
that it was ever considered.

### Conditional GET is deliberately off

`HttpFetcher.get(conditional=False)`. A 304 returns no body, and a probe with no
body has nothing to measure. It also keeps probe traffic out of the validator
store: a candidate that is never approved must not leave conditional-GET state
behind for a `source_id` that does not exist.

### `probe_feed` does not follow redirects

`curate/scout.py::_is_disallowed_fetch_host` checks a candidate feed URL's host
once, before the first fetch — the host the scout proposed, not necessarily the
one the fetch would end up talking to. A public, allowed hostname that 302s to a
private one would otherwise reach exactly the destination that check exists to
stop. `probe_repo` keeps the default (`follow_redirects=True`): its request always
targets `GITHUB_API_ROOT` regardless of what the candidate slug names, so there is
no scout-controlled host in that request to redirect away from.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from signalforge.ingest.base import FetchError, HttpFetcher, filter_by_age
from signalforge.ingest.github import (
    GITHUB_ACCEPT,
    GITHUB_API_ROOT,
    GITHUB_API_VERSION,
    parse_github_timestamp,
)
from signalforge.ingest.rss import parse_feed
from signalforge.models import flatten_to_single_line

__all__ = [
    "PROBE_SOURCE_ID",
    "SourceProbe",
    "failed_probe",
    "probe_feed",
    "probe_repo",
]

logger = logging.getLogger(__name__)

PROBE_SOURCE_ID: Final = "_probe"
"""The `source_id` probe traffic is archived under.

Candidates have no `sources.yaml` id yet — that is the thing being proposed — and
borrowing the id they *would* get would mix probe payloads into a real source's
cache directory before it exists. The leading underscore keeps it outside the
namespace a real source id could occupy."""

_MAX_ERROR_CHARS: Final = 200
"""Ceiling on a stored error string. Probe failures land in `proposals.probe` and
render in a digest; an HTML error page pasted verbatim would swamp the block."""


@dataclass(frozen=True, slots=True)
class SourceProbe:
    """What one probe attempt found. Facts, not a verdict.

    `ok` distinguishes "we could not evaluate this at all" from "we evaluated it";
    it is not a recommendation. A source with `ok=True` and one thin entry is
    perfectly probeable and probably not worth adding — that judgment belongs to
    the reader of the digest, not to this dataclass.
    """

    ok: bool
    error: str | None
    """Why the probe failed, short enough to render. None when `ok`."""

    status_code: int | None
    items_total: int
    """Entries the parser recovered. 0 with `ok=False` means nothing usable came
    back — the `the-batch` case, where every plausible feed path 404s or returns
    a page rather than a feed."""

    items_in_window: int
    """Entries published inside the freshness window. Informational only: a good
    monthly newsletter legitimately has 0 in a 7-day window, so this must never
    be read as a pass/fail signal.

    An entry with no usable date counts as in-window, for both feeds and repos —
    the rule `filter_by_age` already applies to feed items, where the conservative
    failure is a few extra triage tokens rather than a lost item."""

    median_summary_chars: int
    """Median entry summary length — the teaser-stub tell.

    Measured against the suite's fixture pair: a members-only feed of one-line
    stubs lands around 50, a feed carrying full article text above 1500. The
    *median* rather than the mean or maximum, because such a publication usually
    posts one substantial public roundup among the stubs, which pulls both of
    those up and hides the pattern.

    Note this does not separate stubs from a legitimately terse **link blog**,
    whose summaries are short by design (the shipped `simonwillison` capture
    medians around 99). Read it alongside `items_total` and the source's own URL,
    which is what the digest block shows — it is one fact for a human, not a
    score.

    Saturates at `defaults.max_summary_chars` (4000 as shipped), because
    `parse_feed` truncates there: a reported 4000 means "at or above the cap",
    not exactly 4000."""

    newest_published_at: datetime | None

    label: str | None
    """A human-recognizable string lifted from the payload, for the digest block:
    the author of the newest entry for a feed, the newest release's tag for a repo.

    "Newest" means by date, not by payload order — feed and API ordering are
    conventions, not guarantees. Not the feed's `<title>`, which would mean
    parsing the payload a second time here; "who writes this" answers the
    operator's actual question ("is this the person I think it is?") better than a
    site name does."""

    def as_facts(self) -> dict[str, object]:
        """The JSON-serializable form stored in `proposals.probe`.

        Kept narrow and flat on purpose: these values are rendered into a digest
        block a human skims in seconds, and `db.update_proposal_probe` replaces
        them wholesale, so nothing here may depend on a previous probe's shape.
        """
        facts: dict[str, object] = {
            "ok": self.ok,
            "items_total": self.items_total,
            "items_in_window": self.items_in_window,
            "median_summary_chars": self.median_summary_chars,
        }
        if self.status_code is not None:
            facts["status_code"] = self.status_code
        if self.error is not None:
            facts["error"] = self.error
        if self.label is not None:
            facts["label"] = self.label
        if self.newest_published_at is not None:
            facts["newest_published_at"] = self.newest_published_at.isoformat()
        return facts


def failed_probe(message: str, *, status_code: int | None = None) -> SourceProbe:
    """A probe that could not evaluate its candidate.

    Public because the caller driving a batch of probes needs the same shape for
    the one failure this module cannot produce: a bug escaping `probe_feed` or
    `probe_repo` themselves. Constructing `SourceProbe`'s eight fields by hand at
    that call site would be a second definition of "failed" waiting to drift.

    Note what is *not* here: a retry, a fallback URL, or a guess. A probe failure
    is a fact to record and re-check next week (`db.reopen_proposal` covers the
    transient case), not something to work around — working around it is how a
    dead feed ends up in the config.
    """
    return SourceProbe(
        ok=False,
        # Flattened like `label` below: `message` can carry exception text this
        # module does not control the shape of (`httpx`'s own `TransportError`
        # formatting, in particular), and `error` renders into the digest the same
        # way `label` does. Flattened before truncating so the character cap means
        # what it says about the text actually shown.
        error=flatten_to_single_line(message)[:_MAX_ERROR_CHARS],
        status_code=status_code,
        items_total=0,
        items_in_window=0,
        median_summary_chars=0,
        newest_published_at=None,
        label=None,
    )


async def probe_feed(
    fetcher: HttpFetcher,
    url: str,
    *,
    max_item_age_days: int,
    max_summary_chars: int,
    now: datetime | None = None,
) -> SourceProbe:
    """Fetch and parse a candidate feed, reporting what it contains.

    Reuses `parse_feed` rather than inspecting the XML here, so a candidate is
    measured through exactly the parser that would ingest it. A feed that
    `feedparser` can only partially recover yields its good entries and a warning
    (that is `parse_feed`'s contract), which is the right answer for a probe too:
    partially parseable is a real, reportable state, not a failure.
    """
    stamp = now or datetime.now(UTC)
    try:
        # `follow_redirects=False`: this URL's host came from the scout, not the
        # operator, and `curate/scout.py::_is_disallowed_fetch_host` only checked
        # the URL as proposed. A redirect is exactly how that check would
        # otherwise be walked around — a public, allowed hostname that 302s to an
        # internal address. Reported as a failed probe rather than followed.
        response = await fetcher.get(
            url, source_id=PROBE_SOURCE_ID, conditional=False, follow_redirects=False
        )
    except FetchError as exc:
        logger.info("feed probe failed", extra={"url": url, "error": str(exc)})
        return failed_probe(str(exc), status_code=exc.status_code)
    except Exception as exc:  # noqa: BLE001 - one candidate never kills a curate run
        # `HttpFetcher.get` wraps transport errors and bad statuses, but not every
        # `httpx.HTTPError`: a redirect loop raises `TooManyRedirects`, which is
        # neither retried nor wrapped. Probe URLs come from the scout — the least
        # trustworthy URL source in the system — and consent walls and paywalls
        # redirect-loop routinely, so anything escaping here would abort a run
        # partway through a batch of candidates (CLAUDE.md §7, NEVER rule 12).
        logger.warning("feed probe raised", extra={"url": url, "error": str(exc)})
        return failed_probe(f"{type(exc).__name__}: {exc}")

    if response is None:  # pragma: no cover - only a non-compliant server does this
        return failed_probe("server returned 304 to an unconditional request")

    items = parse_feed(
        response.content,
        source_id=PROBE_SOURCE_ID,
        max_summary_chars=max_summary_chars,
        fetched_at=stamp,
    )
    if not items:
        # Parsed cleanly to nothing, or was never a feed. Either way there is
        # nothing here to ingest, which is the one content-shaped call this
        # module does make — it needs no threshold.
        return failed_probe("no parseable entries", status_code=response.status_code)

    fresh = filter_by_age(items, max_age_days=max_item_age_days, now=stamp)
    published = [item.published_at for item in items if item.published_at is not None]
    summary_lengths = [len(item.summary or "") for item in items]
    # Newest by date, not feed order: RSS ordering is a convention, not a
    # guarantee, and this is the field the operator reads as "who writes this".
    newest = max(items, key=lambda item: item.published_at or datetime.min.replace(tzinfo=UTC))

    return SourceProbe(
        ok=True,
        error=None,
        status_code=response.status_code,
        items_total=len(items),
        items_in_window=len(fresh),
        median_summary_chars=int(statistics.median(summary_lengths)),
        newest_published_at=max(published) if published else None,
        label=newest.author or None,
    )


def _github_headers(token: str | None) -> dict[str, str]:
    """Mirror `github.py`'s request headers, including the token when present.

    Unauthenticated probing works at 60 req/hr, which is ample for a handful of
    weekly candidates, so a missing token is not an error here either. The token
    is only ever placed in a header — never logged, never stored in a probe fact
    (NEVER rule 16).
    """
    headers = {"accept": GITHUB_ACCEPT, "x-github-api-version": GITHUB_API_VERSION}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return headers


async def probe_repo(
    fetcher: HttpFetcher,
    slug: str,
    *,
    max_item_age_days: int,
    token: str | None = None,
    now: datetime | None = None,
) -> SourceProbe:
    """Check that a candidate `owner/repo` publishes releases worth watching.

    Answers the question the existing hand-written `sources.yaml` comments were
    answering by hand: does this repo cut releases with notes, or only bare
    version-bump tags a digest cannot act on? `median_summary_chars` is the tell —
    the pruned `llama.cpp` per-commit CI tags carried no body at all.

    A repo with zero releases is `ok=False`: the release watch would have nothing
    to ingest. `github.py` falls back to `/tags` when ingesting, but a probe
    deliberately does not — proposing a repo on the strength of tags that carry no
    notes is how the noise this feature exists to remove gets back in.
    """
    stamp = now or datetime.now(UTC)
    url = f"{GITHUB_API_ROOT}/repos/{slug}/releases"
    try:
        response = await fetcher.get(
            url,
            source_id=PROBE_SOURCE_ID,
            headers=_github_headers(token),
            conditional=False,
        )
    except FetchError as exc:
        logger.info("repo probe failed", extra={"slug": slug, "error": str(exc)})
        return failed_probe(str(exc), status_code=exc.status_code)
    except Exception as exc:  # noqa: BLE001 - one candidate never kills a curate run
        # Same reasoning as `probe_feed`: an unwrapped `httpx.HTTPError` escaping
        # here would abort a run midway through its candidates.
        logger.warning("repo probe raised", extra={"slug": slug, "error": str(exc)})
        return failed_probe(f"{type(exc).__name__}: {exc}")

    if response is None:  # pragma: no cover - only a non-compliant server does this
        return failed_probe("server returned 304 to an unconditional request")

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        return failed_probe(f"unparseable JSON: {exc}", status_code=response.status_code)
    if not isinstance(payload, list):
        return failed_probe("releases response was not a list", status_code=response.status_code)

    # Drafts are skipped for parity with `github.py::_releases_to_items`, which
    # also skips them. Without this, an authenticated probe would measure releases
    # a watch would never ingest — the probe describing something other than the
    # thing it is validating.
    releases = [entry for entry in payload if isinstance(entry, dict) and not entry.get("draft")]
    if not releases:
        return failed_probe("no published releases", status_code=response.status_code)

    # Timestamp each release once and carry it alongside its entry: it is needed
    # three times below, and re-parsing inside a `max` key was easy to read as
    # cheap when it is not.
    dated = [(entry, _release_timestamp(entry)) for entry in releases]
    published = [when for _, when in dated if when is not None]
    # Undated releases count as in-window, matching `filter_by_age`'s rule for
    # feed items (`base.py`: a missing date is kept, never dropped). One number
    # rendered in one digest column must not mean opposite things for the two
    # source types.
    cutoff = stamp - timedelta(days=max_item_age_days)
    in_window = [when for _, when in dated if when is None or when >= cutoff]
    body_lengths = [len(str(entry.get("body") or "")) for entry in releases]
    newest, _ = max(dated, key=lambda pair: pair[1] or datetime.min.replace(tzinfo=UTC))

    return SourceProbe(
        ok=True,
        error=None,
        status_code=response.status_code,
        items_total=len(releases),
        items_in_window=len(in_window),
        median_summary_chars=int(statistics.median(body_lengths)),
        newest_published_at=max(published, default=None),
        # Flattened: `tag_name` is entirely controlled by whoever owns the candidate
        # repo, and it renders into the digest before any human approves the
        # candidate — the same forgery class `Item.author`'s validator closes for
        # `probe_feed`, but a release tag never passes through `Item` at all.
        label=flatten_to_single_line(str(newest.get("tag_name") or "")) or None,
    )


def _release_timestamp(entry: dict[str, Any]) -> datetime | None:
    """Read a release's publish time, preferring `published_at` over `created_at`.

    Delegates the parse to `github.parse_github_timestamp` — the same function the
    ingestor uses, so a probe and an ingest never disagree about what a GitHub
    timestamp means — and adds the two behaviours a probe needs on top:

    * the `created_at` fallback, for a release with no publish time of its own;
    * naive→UTC coercion, matching the convention `Item._require_utc` and
      `db._window_start` both hold, so a comparison against `now` cannot raise.
    """
    for key in ("published_at", "created_at"):
        parsed = parse_github_timestamp(entry.get(key))
        if parsed is None:
            continue
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None
