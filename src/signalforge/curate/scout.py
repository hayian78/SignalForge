"""The weekly scout run: evidence in, validated proposals out (DESIGN §7.1).

This is the module the boundary exception in `curate/__init__.py` exists for. It
calls `llm.py` for judgment and then an `ingest/` fetch helper to check what that
judgment produced, strictly in that order, and it writes no `items` row.

### Model output becomes a config edit here, so this is where it is checked

`llm.ScoutProposal` validates the *shape* of what came back. This module validates
its *meaning* against the config the edit will land in: that a feed URL is a URL,
that a retirement names a source that actually exists, that an addition is not
something the operator already has, that a keyword cannot corrupt the inline YAML
list it will be spliced into. A proposal that fails any of those is dropped with a
recorded reason rather than stored, because the alternative is a checkbox in a
digest that either does nothing when ticked or does something unintended.

The `dedup_key` each kind gets is decided here too, for the reason
`ScoutProposal.target` gives: the normalization differs per kind, and only code
holding the parsed `SourcesConfig` can turn "retire deepmind" into the exact
spelling `sources.yaml` uses.

### Three passes, and why they run in this order

`reprobe_invalid_proposals` → `scout_for_proposals` → `persist_candidates`.

The re-probe runs **first**, before the paid call. Its job is to refresh the
`invalid` rows from previous weeks and reopen the ones that now fetch, and doing it
first means a candidate the scout re-proposes this week has already been refreshed:
the re-proposal hits `ux_proposals_kind_key`, does nothing, and the fresh facts are
already on the row rather than being discarded by the conflict. Running it after
would fetch the same URL twice in one run and keep the later result anyway.

### One failure never costs the run

Everything after the API call is per-candidate: a bad proposal, a probe that
raises, an insert that refuses. The Opus call and its searches are already paid for
by then, so aborting on candidate three would waste the money for candidates four
and five as well (CLAUDE.md §7). Every such failure lands in `runs.errors` and
surfaces in the next digest.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from signalforge import db, llm
from signalforge.config import CurationConfig, InterestsConfig, SourcesConfig
from signalforge.curate.gather import gather_evidence
from signalforge.curate.probe import ProbeRequest, github_probe_token, is_probeable, probe_requests
from signalforge.curate.prompts import (
    build_scout_system_prompt,
    build_scout_user_prompt,
    proposal_payload,
)
from signalforge.ingest.probe import SourceProbe
from signalforge.models import (
    ProposalKind,
    ProposalStatus,
    ProposalTier,
    canonicalize_url,
    is_safe_url,
)

__all__ = [
    "Candidate",
    "PersistOutcome",
    "ReprobeOutcome",
    "ScoutOutcome",
    "persist_candidates",
    "reprobe_invalid_proposals",
    "scout_for_proposals",
    "search_spend_usd",
]

logger = logging.getLogger(__name__)

_PROBE_GATES_APPROVAL: Final = frozenset(
    {ProposalKind.ADD_RSS, ProposalKind.ADD_GITHUB_REPO},
)
"""The kinds a failed probe disqualifies.

Only additions. A retirement whose probe fails is not invalid — it is *better
evidenced*: a feed that 404s or has published nothing in a month is exactly what a
retirement claims, and hiding that proposal would suppress the one it most makes
sense to show. So a failed probe marks an addition `invalid` and never renders it,
and leaves a retirement pending with the failure recorded in its facts."""

_KEYWORD_SAFE: Final = re.compile(r"[a-z0-9][a-z0-9 .+#/-]*")
r"""What a keyword may contain.

A syntax guard on model output, not a tuning knob: HN keywords are spliced into a
flow-style list (`keywords: [llm, claude]`), where a `,`, `[`, `]`, `#` at the
start, or a quote character would produce a different list than intended or invalid
YAML — and `apply.py`'s safety net would then revert the whole edit. Bounded to the
characters real keywords use, so a malformed one is refused at proposal time where
the reason can be recorded, rather than at apply time where it costs the operator's
approval."""

_MAX_KEYWORD_CHARS: Final = 40
"""Sanity bound on a "keyword". Not a threshold on anything the operator tunes —
`hackernews.keywords` is theirs to set to any length in YAML. This only stops a
model returning a sentence where a search term belongs."""

_GENERIC_HOST_LABELS: Final = frozenset({"www", "blog", "feed", "feeds", "rss", "news", "mail"})
"""Host labels that identify no publication, skipped when deriving a source id."""

_MAX_ID_SUFFIX: Final = 9
"""How far to count when disambiguating a derived source id."""

_WEIGHT_BAND: Final = (0.8, 1.5)
"""The range a *scout-suggested* weight is clamped into.

Not a tuning knob, and not a limit on the operator: `RssSource.weight` accepts any
positive number and `sources.yaml` is theirs to set (NEVER rule 6). This bounds only
what a model may suggest, and it exists because of how the suggestion is reviewed —
a human skimming a digest over coffee. A visible 1.3 gets judged on its merits;
nothing in the loop would catch a quietly-proposed 9.0 before it reweighted a source
against everything else in the feed.

The band is drawn around the identity element, a little wider than the 1.0-1.3 the
operator's own config uses, so a genuine "trust this author" or "read but discount"
suggestion still survives the clamp intact."""


def search_spend_usd(requests: int) -> float:
    """What `requests` web searches cost, in dollars.

    A pass-through to `llm.WEB_SEARCH_USD_PER_REQUEST`, and the indirection is the
    point: `cli.py` states that it never imports `llm.py` (so the Anthropic SDK stays
    out of the CLI's direct dependencies), but `status` has to turn a search count
    into a dollar figure or the $30 alarm cannot see the whole bill (DESIGN §8).
    `curate/` already imports `llm.py` legitimately, so it carries the number across
    rather than the price being copied into a second module where it would drift.
    """
    return requests * llm.WEB_SEARCH_USD_PER_REQUEST


def _error(target: str, message: str) -> dict[str, str]:
    """One `runs.errors` record, shaped as `apply.py` shapes them."""
    return {"source_id": target, "message": message}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One scout proposal, normalized against the config and ready to store."""

    kind: ProposalKind
    dedup_key: str
    payload: dict[str, object]
    rationale: str
    evidence: list[dict[str, str]]
    tier: ProposalTier
    locator: str | None = None
    """What to fetch when probing, or None for a kind with nothing to fetch."""

    probe: SourceProbe | None = None
    status: ProposalStatus = ProposalStatus.PENDING


@dataclass(slots=True)
class ScoutOutcome:
    """What one scout pass produced, including everything it spent.

    Spend travels with the output for the same reason `llm.ScoutResult` does it:
    persisting proposals without recording what they cost has to take an extra line
    of code, not zero (NEVER rule 11).
    """

    candidates: list[Candidate] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    web_search_requests: int = 0

    @property
    def invalid_count(self) -> int:
        """Candidates a failed probe disqualified — stored, never shown."""
        return sum(1 for candidate in self.candidates if candidate.status is ProposalStatus.INVALID)


@dataclass(slots=True)
class PersistOutcome:
    """What storing a pass's candidates changed."""

    stored: list[int] = field(default_factory=list)
    duplicates: int = 0
    """Candidates the unique index rejected: already proposed, already rejected, or
    already applied. The expected majority on a steady-state run, not a problem."""

    errors: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ReprobeOutcome:
    """What the weekly re-check of `invalid` rows changed."""

    probed: int = 0
    reopened: list[int] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


_is_http_url = is_safe_url
"""`models.is_safe_url`, under the name every call site here already uses.

The scout's URLs come from web-search results and model output — the least
trustworthy input in the system — so this is checked here, and again at the write
site in `apply.py` and again in `db.insert_proposal`. Kept in `models.py` rather
than duplicated here because `db.py` needs the identical check on the read and
write paths for evidence citations, and a second copy of an RFC 3986 allowlist is
exactly the kind of drift that produced the arithmetic bugs this project's cost
constants had to stop restating."""


_DISALLOWED_FETCH_SUFFIXES: Final = (".localhost", ".local", ".internal")
"""Hostname suffixes that never name a public feed, checked case-insensitively."""

_CGNAT_RANGE: Final = ipaddress.ip_network("100.64.0.0/10")
"""RFC 6598 shared address space, used internally by several cloud providers.

`ipaddress.IPv4Address.is_private`/`.is_reserved` do not cover this range (checked
against 3.12: neither is `True` for an address in it), so it needs an explicit
check alongside them."""


def _ipv4_literal(hostname: str) -> ipaddress.IPv4Address | None:
    """An IPv4 address `hostname` names, in any form a resolver would accept.

    `ipaddress.ip_address` only parses strict dotted-quad notation and raises on
    the legacy forms glibc's resolver (and `socket.inet_aton`) still treat as
    literals — `127.1`, `0x7f000001`, `2130706433`, and `0177.0.0.1` all resolve to
    `127.0.0.1` with no DNS lookup at all. Falling back to "not an IP literal, a
    hostname DNS will resolve" for any of those would let every one of them past
    `_is_disallowed_fetch_host` for a target the fetch never actually looks up by
    name.
    """
    try:
        return ipaddress.IPv4Address(hostname)
    except ValueError:
        pass
    try:
        packed = socket.inet_aton(hostname)
    except OSError:
        return None
    return ipaddress.IPv4Address(packed)


def _is_disallowed_fetch_host(hostname: str) -> bool:
    """Whether `hostname` names a destination the probe must never reach.

    Checked for every URL the run will actually *fetch* — `_normalize_add_rss`'s
    candidate feed URL before its first probe, and `_reprobe_locator`'s stored URL
    before every later re-check. Evidence citations and retirement targets are
    read by a human, never fetched, so they have nothing this check protects.

    Every other URL `HttpFetcher` has ever been given came from the operator's own
    hand-edited `sources.yaml` (CLAUDE.md §7). This is the first fetch whose
    destination host is chosen by web-search output — a model steered by
    adversarial content it encountered while searching could otherwise point a
    probe at a cloud metadata endpoint or an internal service and have the result
    rendered back into the digest as a reconnaissance oracle.

    An allowlist of "safe" hosts is not possible here — a legitimate feed can be
    hosted anywhere — so this is a blocklist of the ranges that are never a public
    RSS feed. Two things it does not cover, for different reasons:

    * **A redirect** to a private address is not this function's job — the probe
      fetch itself refuses to follow one at all (`ingest/probe.py`), so there is no
      second hop for this check to see.
    * **DNS rebinding** — a public hostname that resolves to a private address only
      at the moment of the fetch, after passing this check — has no defense here.
      Closing it needs the check to run against the resolved connection, not the
      proposed URL, which this single-operator pipeline's threat model does not
      currently justify building.
    """
    # A trailing "." is the DNS root label — glibc's resolver (and `httpx`, through
    # it) treats `127.0.0.1.` and `localhost.` identically to the form without the
    # dot, with no lookup for the IP-literal case. Stripped once, up front, so
    # every check below sees the same string the resolver would actually act on.
    folded = hostname.casefold().rstrip(".")
    if folded == "localhost" or folded.endswith(_DISALLOWED_FETCH_SUFFIXES):
        return True
    try:
        address: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(folded)
    except ValueError:
        literal = _ipv4_literal(folded)
        if literal is None:
            return False  # Not an IP literal in any form — a hostname DNS resolves.
        address = literal
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        or (isinstance(address, ipaddress.IPv4Address) and address in _CGNAT_RANGE)
    )


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().casefold()).strip("-")


def _derive_source_id(proposed: str | None, url: str) -> str:
    """A `sources.yaml` id for an added feed.

    Prefers the scout's own suggestion — the operator will read this id in every
    future digest, and a human-chosen name is better than a derived one. Falls back
    to the first meaningful label of the host, because the alternative is discarding
    an otherwise good proposal over a cosmetic, hand-editable field.
    """
    if proposed and (slug := _slugify(proposed)):
        return slug
    host = urlsplit(url).hostname or ""
    labels = [label for label in host.split(".") if label and label not in _GENERIC_HOST_LABELS]
    if labels and (slug := _slugify(labels[0])):
        return slug
    return _slugify(host) or "new-source"


def _available_source_id(proposed: str | None, url: str, sources: SourcesConfig) -> str:
    """`_derive_source_id`, suffixed until it collides with no configured id.

    Collisions are renamed rather than refused because the id is a label, not the
    proposal: the operator asked for the *feed*. `apply.py` raises on a collision it
    meets at apply time, so this keeps that error for the genuinely rare case — the
    config changing between the proposal and the approval.
    """
    taken = {source.id.casefold() for source in sources.rss}
    base = _derive_source_id(proposed, url)
    if base.casefold() not in taken:
        return base
    for suffix in range(2, _MAX_ID_SUFFIX + 1):
        candidate = f"{base}-{suffix}"
        if candidate.casefold() not in taken:
            return candidate
    raise ValueError(f"could not find a free source id near {base!r}")


def _clamp_weight(weight: float | None, errors: list[dict[str, str]], target: str) -> float | None:
    """Bring a suggested weight into `_WEIGHT_BAND`, recording any clamp.

    Clamped rather than refused: the proposal is "add this source", and throwing away
    a well-argued addition over an over-enthusiastic multiplier would waste a slot the
    run has already paid for. Recorded rather than silent, because the number the
    operator reads in the digest has to be the number that would be written.
    """
    if weight is None:
        return None
    low, high = _WEIGHT_BAND
    clamped = min(max(weight, low), high)
    if clamped != weight:
        errors.append(
            _error(
                target,
                f"suggested weight {weight} is outside the {low}-{high} band a proposal "
                f"may use; showing {clamped} instead",
            )
        )
    return clamped


def _clean_keyword(value: str) -> str:
    """Validate a keyword and return its canonical (lowercase) form."""
    keyword = " ".join(value.split()).casefold()
    if not keyword:
        raise ValueError("keyword is empty")
    if len(keyword) > _MAX_KEYWORD_CHARS:
        raise ValueError(f"keyword {keyword[:20]!r}… is too long to be a search term")
    if not _KEYWORD_SAFE.fullmatch(keyword):
        raise ValueError(f"keyword {keyword!r} contains characters that would break the YAML list")
    return keyword


def _configured_keywords(kind: ProposalKind, sources: SourcesConfig) -> list[str]:
    """The keyword list a proposal of this kind edits, or None-safe empty."""
    if kind in (ProposalKind.ADD_HN_KEYWORD, ProposalKind.REMOVE_HN_KEYWORD):
        if sources.hackernews is None:
            raise ValueError("sources.yaml has no hackernews block to edit")
        return sources.hackernews.keywords
    if sources.arxiv is None:
        raise ValueError("sources.yaml has no arxiv block to edit")
    return sources.arxiv.require_keywords


def _normalize_add_rss(
    proposal: llm.ScoutProposal, sources: SourcesConfig, errors: list[dict[str, str]]
) -> tuple[str, dict[str, object], str]:
    url = (proposal.url or proposal.target).strip()
    if not _is_http_url(url):
        raise ValueError(f"{url!r} is not an http(s) feed URL")
    if _is_disallowed_fetch_host(urlsplit(url).hostname or ""):
        raise ValueError(f"{url!r} points at a local or internal address and cannot be probed")
    canonical = canonicalize_url(url)
    if canonical in {canonicalize_url(source.url) for source in sources.rss}:
        raise ValueError(f"{url} is already configured")
    source_id = _available_source_id(proposal.source_id, url, sources)
    weight = _clamp_weight(proposal.weight, errors, canonical)
    return (
        canonical,
        proposal_payload(source_id=source_id, url=url, target=url, weight=weight),
        url,
    )


def _normalize_retire_rss(
    proposal: llm.ScoutProposal, sources: SourcesConfig
) -> tuple[str, dict[str, object], str]:
    """Resolve a retirement to the config's own id and the URL to probe.

    Accepts either an id or a feed URL as the target, because a model reasoning
    about "the DeepMind blog" may reach for either, and refusing one of them
    discards a proposal over a naming convention.
    """
    target = proposal.target.strip()
    folded = target.casefold()
    canonical = canonicalize_url(target) if _is_http_url(target) else None
    for source in sources.rss:
        if source.id.casefold() == folded or (
            canonical is not None and canonicalize_url(source.url) == canonical
        ):
            return (
                source.id,
                proposal_payload(source_id=source.id, url=source.url, target=source.id),
                source.url,
            )
    raise ValueError(f"no configured rss source matches {target!r}")


def _normalize_github(
    proposal: llm.ScoutProposal, sources: SourcesConfig
) -> tuple[str, dict[str, object], str]:
    slug = proposal.target.strip().strip("/")
    owner, _, name = slug.partition("/")
    if not owner or not name or "/" in name or any(char.isspace() for char in slug):
        raise ValueError(f"{proposal.target!r} is not an 'owner/repo' slug")
    if sources.github is None:
        raise ValueError("sources.yaml has no github block to edit")
    configured = next(
        (entry for entry in sources.github.releases if entry.casefold() == slug.casefold()), None
    )
    if proposal.kind is ProposalKind.ADD_GITHUB_REPO:
        if configured is not None:
            raise ValueError(f"{configured} is already watched")
        return slug, proposal_payload(source_id=None, url=None, target=slug), slug
    if configured is None:
        raise ValueError(f"{slug} is not a configured release watch")
    # The config's own spelling: `apply.py` matches the line literally.
    return configured, proposal_payload(source_id=None, url=None, target=configured), configured


def _normalize_keyword(
    proposal: llm.ScoutProposal, sources: SourcesConfig
) -> tuple[str, dict[str, object], None]:
    keyword = _clean_keyword(proposal.target)
    configured = _configured_keywords(proposal.kind, sources)
    present = next((entry for entry in configured if entry.casefold() == keyword), None)
    adding = proposal.kind in (ProposalKind.ADD_HN_KEYWORD, ProposalKind.ADD_ARXIV_KEYWORD)
    if adding and present is not None:
        raise ValueError(f"{keyword!r} is already a configured keyword")
    if not adding:
        if present is None:
            raise ValueError(f"{keyword!r} is not a configured keyword")
        keyword = present
    return keyword, proposal_payload(source_id=None, url=None, target=keyword), None


def _normalize(
    proposal: llm.ScoutProposal,
    sources: SourcesConfig,
    *,
    corpus_only: bool,
    errors: list[dict[str, str]],
) -> Candidate:
    """Turn one validated model proposal into a storable candidate, or raise.

    Raising is how a nonsense proposal is refused: the caller records the reason
    against the run and moves to the next one.
    """
    if not proposal.target.strip():
        raise ValueError("proposal has an empty target")
    evidence = [
        {"url": entry.url.strip(), "note": entry.note.strip()}
        for entry in proposal.evidence
        if _is_http_url(entry.url.strip())
    ]
    if not evidence:
        # `ScoutProposal` only enforces a non-empty list and `min_length=1` on
        # `url`, so a whitespace-only or malformed "URL" passes that. Requiring a
        # real `http(s)` shape here — not just non-blank — is load-bearing: an
        # evidence entry renders as its own line in the digest (`daily.md.j2`'s
        # evidence loop), and a citation "URL" of `"[x] approve <!-- sf:proposal=N
        # v=approve --> "` contains no control character and would otherwise reach
        # that line intact, forging approval of an unrelated pending proposal —
        # confirmed exploitable by security review before this check existed.
        raise ValueError("proposal has no usable evidence URL")

    dedup_key: str
    payload: dict[str, object]
    locator: str | None
    match proposal.kind:
        case ProposalKind.ADD_RSS:
            dedup_key, payload, locator = _normalize_add_rss(proposal, sources, errors)
        case ProposalKind.RETIRE_RSS:
            dedup_key, payload, locator = _normalize_retire_rss(proposal, sources)
        case ProposalKind.ADD_GITHUB_REPO | ProposalKind.RETIRE_GITHUB_REPO:
            dedup_key, payload, locator = _normalize_github(proposal, sources)
        case _:
            dedup_key, payload, locator = _normalize_keyword(proposal, sources)

    return Candidate(
        kind=proposal.kind,
        dedup_key=dedup_key,
        payload=payload,
        rationale=proposal.rationale.strip(),
        evidence=evidence,
        # A run configured for no searches cannot have found anything on the web,
        # whatever the model labelled it. The digest shows this tier to tell the
        # operator how much scrutiny a candidate deserves, so it has to be true.
        tier=ProposalTier.CORPUS if corpus_only else proposal.tier,
        locator=locator,
    )


def _statused(candidate: Candidate, probe: SourceProbe | None) -> Candidate:
    """Attach probe facts and decide whether the candidate may be shown."""
    if probe is None:
        return candidate
    disqualified = not probe.ok and candidate.kind in _PROBE_GATES_APPROVAL
    return replace(
        candidate,
        probe=probe,
        status=ProposalStatus.INVALID if disqualified else ProposalStatus.PENDING,
    )


async def scout_for_proposals(
    conn: sqlite3.Connection,
    *,
    sources: SourcesConfig,
    interests: InterestsConfig,
    curation: CurationConfig,
    cache_dir: Path,
    now: datetime | None = None,
) -> ScoutOutcome:
    """Gather evidence, ask the scout, probe what it proposed. No DB writes.

    Read-only against the database so `--dry-run` is the same code path as a real
    run minus `persist_candidates` — a dry run that exercised different code would
    not be the quality gate DESIGN §7.1 relies on. It still costs money: the Opus
    call and its searches happen either way.

    `curation` is passed in rather than read off `sources.curation` so the type
    system carries the fact that an absent `curation:` block means the feature is
    off. There is no client seam: `llm.py` is faked at its boundary in tests
    (CLAUDE.md §8), which is also what keeps the `anthropic` SDK out of this module
    (NEVER rule 1).

    **Never raises once the call has been billed.** `llm.run_source_scout` already
    promises not to raise for an API failure, so that its cost is never lost; this
    function extends the same promise across everything it does *afterwards*
    (normalizing, probing), because the caller reads the spend off the returned
    value. Raising there would report a paid run as having spent nothing, which is
    precisely the accounting NEVER rule 11 exists to protect.
    """
    stamp = now or datetime.now(UTC)
    evidence = gather_evidence(conn, sources, window_days=curation.yield_window_days, now=stamp)

    # Blocking, on purpose. Nothing else is scheduled on this loop: the probes
    # cannot start until there is something to probe, which is the ordering the
    # `curate/` boundary exception is predicated on.
    result = llm.run_source_scout(
        system_prompt=build_scout_system_prompt(interests),
        user_prompt=build_scout_user_prompt(
            sources=sources,
            yield_rows=evidence.yield_rows,
            feedback_by_source=evidence.feedback_by_source,
            outbound_domains=evidence.outbound_domains,
            kept_titles=evidence.kept_titles,
            rejected=evidence.rejected,
            window_days=evidence.window_days,
            max_proposals=curation.max_proposals_per_run,
        ),
        max_searches=curation.max_searches_per_run,
        max_proposals=curation.max_proposals_per_run,
    )

    outcome = ScoutOutcome(
        errors=[_error("scout", message) for message in result.errors],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        web_search_requests=result.web_search_requests,
    )

    try:
        return await _shape_candidates(
            result, outcome, sources=sources, curation=curation, cache_dir=cache_dir, stamp=stamp
        )
    except Exception as exc:  # noqa: BLE001 - see the docstring: the money is already spent
        # Everything from here on is free — normalizing, probing, counting. A bug in
        # any of it must not take the *record* of what the call cost with it: the
        # caller reads spend off the returned value, so raising would report a paid
        # run as having spent nothing (NEVER rule 11). Both reviews of this branch
        # found that gap independently, which is why the guarantee lives here rather
        # than in the CLI's `finally` — the function holding the numbers is the one
        # that has to promise to hand them back.
        logger.exception("shaping scout candidates failed after the call was billed")
        outcome.candidates = []
        outcome.errors.append(
            _error("scout", f"proposals were lost after the call was billed: {exc}")
        )
        return outcome


async def _shape_candidates(
    result: llm.ScoutResult,
    outcome: ScoutOutcome,
    *,
    sources: SourcesConfig,
    curation: CurationConfig,
    cache_dir: Path,
    stamp: datetime,
) -> ScoutOutcome:
    """Validate, cap, and probe what the scout returned. No further spend."""
    proposals = result.proposals
    if len(proposals) > curation.max_proposals_per_run:
        # The tool schema's `maxItems` asks for this; the cap is enforced here
        # because a schema is a request and `max_proposals_per_run` is a promise.
        outcome.errors.append(
            _error(
                "scout",
                f"scout returned {len(proposals)} proposals, above the configured cap of "
                f"{curation.max_proposals_per_run}; the excess was dropped",
            )
        )
        proposals = proposals[: curation.max_proposals_per_run]

    corpus_only = curation.max_searches_per_run == 0
    candidates: list[Candidate] = []
    for proposal in proposals:
        try:
            candidates.append(
                _normalize(proposal, sources, corpus_only=corpus_only, errors=outcome.errors)
            )
        except ValueError as exc:
            outcome.errors.append(
                _error(f"{proposal.kind.value}:{proposal.target[:60]}", f"dropped: {exc}")
            )

    probes = await probe_requests(
        [
            ProbeRequest(kind=candidate.kind, locator=candidate.locator or candidate.dedup_key)
            for candidate in candidates
        ],
        cache_dir=cache_dir,
        defaults=sources.defaults,
        token=github_probe_token(sources),
        now=stamp,
    )
    outcome.candidates = [
        _statused(candidate, probe) for candidate, probe in zip(candidates, probes, strict=True)
    ]
    logger.info(
        "scout pass complete",
        extra={
            "candidates": len(outcome.candidates),
            "invalid": outcome.invalid_count,
            "errors": len(outcome.errors),
            "web_search_requests": outcome.web_search_requests,
        },
    )
    return outcome


def persist_candidates(
    conn: sqlite3.Connection,
    candidates: list[Candidate],
    *,
    run_id: int,
    surface_date: Date,
    created_at: datetime,
) -> PersistOutcome:
    """Store each candidate, one at a time, never failing the batch for one row.

    `surface_date` is the digest date a proposal first renders on, so a Sunday scout
    run surfaces in Monday's digest — the caller decides which day that is, because
    the scout does not know the report schedule.
    """
    outcome = PersistOutcome()
    for candidate in candidates:
        try:
            proposal_id = db.insert_proposal(
                conn,
                run_id=run_id,
                kind=candidate.kind,
                dedup_key=candidate.dedup_key,
                payload=candidate.payload,
                rationale=candidate.rationale,
                evidence=candidate.evidence,
                probe=candidate.probe.as_facts() if candidate.probe is not None else None,
                tier=candidate.tier,
                status=candidate.status,
                surface_date=surface_date,
                created_at=created_at,
            )
        except ValueError as exc:
            # A refused insert is one lost proposal, not a lost run: the searches
            # behind the other four are already paid for.
            outcome.errors.append(
                _error(f"{candidate.kind.value}:{candidate.dedup_key[:60]}", f"not stored: {exc}")
            )
            continue
        if proposal_id is None:
            outcome.duplicates += 1
            continue
        outcome.stored.append(proposal_id)
    logger.info(
        "stored scout proposals",
        extra={
            "stored": len(outcome.stored),
            "duplicates": outcome.duplicates,
            "errors": len(outcome.errors),
        },
    )
    return outcome


def _reprobe_locator(proposal: db.Proposal) -> str | None:
    """What to fetch to re-check an `invalid` row, or None if it cannot be re-checked.

    Only the kinds `_PROBE_GATES_APPROVAL` can disqualify are re-probed, because
    only those can be `invalid` in the first place. Anything else in that state was
    hand-edited, and guessing what to fetch for it would be inventing a lifecycle.

    An `ADD_RSS` row's locator is a URL whose host came from the scout, re-checked
    against `_is_disallowed_fetch_host` here for the same reason
    `_normalize_add_rss` checks it before the first probe: this function runs
    **every week, forever**, against whatever is already stored, so a row that
    predates this guard — or reached this state through some future insert path —
    must not get a standing weekly fetch of an internal address just because it
    was never re-validated. An `ADD_GITHUB_REPO` row's locator is an `owner/repo`
    slug, not a URL; `probe_repo` builds the request against `api.github.com`
    itself, so there is no scout-controlled host to check.
    """
    if proposal.kind not in _PROBE_GATES_APPROVAL:
        return None
    if proposal.kind is ProposalKind.ADD_GITHUB_REPO:
        return proposal.dedup_key
    url = proposal.payload.get("url")
    locator = str(url) if isinstance(url, str) and url.strip() else proposal.dedup_key
    if _is_disallowed_fetch_host(urlsplit(locator).hostname or ""):
        logger.warning(
            "an invalid proposal's stored URL names a disallowed fetch host; skipping its re-probe",
            extra={"proposal_id": proposal.id, "url": locator},
        )
        return None
    return locator


async def reprobe_invalid_proposals(
    conn: sqlite3.Connection,
    *,
    sources: SourcesConfig,
    cache_dir: Path,
    now: datetime | None = None,
) -> ReprobeOutcome:
    """Re-check every `invalid` proposal, reopening the ones that now fetch.

    **Every** invalid row, every run, with no notion of a durable failure. That
    looks wasteful — a feed that 404s is likely to 404 again — and it is one request
    per row per week, which is nothing against being wrong in the other direction:
    `no parseable entries` is what a consent interstitial or a Cloudflare
    challenge returns, `403` is what a rate limit returns, and treating either as
    permanent removes a good source from the scout's reachable set forever, because
    `ux_proposals_kind_key` means it can never be re-proposed.

    Costs no tokens: HTTP and DB only, no LLM. That is why it can run before the
    paid call rather than being folded into it.
    """
    stamp = now or datetime.now(UTC)
    invalid = db.get_proposals(conn, statuses=[ProposalStatus.INVALID])
    targets = [(proposal, _reprobe_locator(proposal)) for proposal in invalid]
    fetchable = [
        (proposal, locator)
        for proposal, locator in targets
        if locator is not None and is_probeable(proposal.kind)
    ]
    outcome = ReprobeOutcome()
    if not fetchable:
        return outcome

    probes = await probe_requests(
        [ProbeRequest(kind=proposal.kind, locator=locator) for proposal, locator in fetchable],
        cache_dir=cache_dir,
        defaults=sources.defaults,
        token=github_probe_token(sources),
        now=stamp,
    )
    for (proposal, _locator), probe in zip(fetchable, probes, strict=True):
        if probe is None:  # pragma: no cover - filtered by `is_probeable` above
            continue
        outcome.probed += 1
        db.update_proposal_probe(conn, proposal_id=proposal.id, probe=probe.as_facts())
        if probe.ok and db.reopen_proposal(conn, proposal_id=proposal.id):
            outcome.reopened.append(proposal.id)
    logger.info(
        "re-probed invalid proposals",
        extra={"probed": outcome.probed, "reopened": len(outcome.reopened)},
    )
    return outcome
