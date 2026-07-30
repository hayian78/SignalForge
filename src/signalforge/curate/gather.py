"""Deterministic evidence for the scout — counting, never judging (DESIGN §7.1).

Everything here is arithmetic over rows the pipeline already has: what each source
delivered, what the operator made of it, and what their valued reading points at.
No LLM, no HTTP, no writes. That split is the whole point — an LLM asked to count
will approximate, and the numbers are the part of the prompt that has to be right
(CLAUDE.md §2, NEVER rule 3).

### The feedback ladder is reduced here, not in SQL

An item can hold several verdicts at once (`UNIQUE(item_id, verdict)`), so summing
them per source double-counts and deflates exactly the way DESIGN §11 warns about.
The reduction to each item's *highest* rung therefore happens in Python, next to
`feedback.LADDER`, which is the single definition of that order — `db.py` cannot
import `feedback.py` without a cycle, and a second copy of the ladder in SQL is the
drift hazard `LADDER`'s own docstring warns about.

`missed` sits off the ladder and is counted separately rather than dropped. It
means "you should have shown me this", which is *positive* signal about a source,
and feeding it to the scout as if it were absent would be a small lie.

### A known gap: outbound links from feeds

The strongest available add-signal is "domains the operator's kept reading keeps
pointing at". This module extracts it from `items.summary`, which means it works
for HN and **not** for RSS: `rss._clean_summary` strips HTML at ingest, so a feed
item's anchor text survives and its hrefs do not. Simon Willison's link blog — the
single richest source of outbound attention in the shipped config — therefore
contributes no domains at all.

That is accepted rather than worked around. Recovering the hrefs means re-reading
the archived payload from `data/http_cache/`, re-parsing the feed, and matching
entries back to items — real machinery for a signal the scout can largely infer
from the kept *titles* this module already sends it. If the first live runs show it
genuinely needs the domains, that is a felt gap and the moment to build it, which
is the bar this project sets for every other component.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from typing import Final
from urllib.parse import urlsplit

from signalforge import db
from signalforge.config import SourcesConfig
from signalforge.feedback import LADDER
from signalforge.models import canonicalize_url

__all__ = [
    "MISSED_VERDICT",
    "CurationEvidence",
    "gather_evidence",
]

logger = logging.getLogger(__name__)

MISSED_VERDICT: Final = "missed"
"""The off-ladder verdict, counted alongside the rungs rather than reduced into
them. Named here so the tally keys stay in step with `feedback.VERDICTS`."""

_KEPT_ITEM_PROMPT_LIMIT: Final = 25
"""How many kept items' titles reach the prompt.

A token-budget decision, not a display one: these titles are input tokens on a
call billed at Opus rates, and 25 titles is ~400 tokens — enough for the scout to
see what the operator actually reads, small enough to be noise against the search
results that dominate the bill."""

_OUTBOUND_DOMAIN_LIMIT: Final = 15
"""How many outbound domains reach the prompt, most-referenced first. Same
reasoning as the item limit: a long tail of once-seen domains is noise the scout
would have to read past."""

_MIN_DOMAIN_REFERENCES: Final = 2
"""A domain must be pointed at more than once to count as attention.

One link is a citation; two or more is a pattern. Without this floor the list is
mostly incidental references — a docs page, a GitHub repo linked in passing — and
the scout's strongest signal reads as noise."""


class _LinkExtractor(HTMLParser):
    """Collects `href` values. Used only where markup survived ingest (HN).

    An `HTMLParser` rather than a regex because a regex over attacker-adjacent
    HTML is a footgun, and this is already the pattern `rss._TextExtractor` uses.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


def _extract_hrefs(html: str) -> list[str]:
    """Every link in `html`. Malformed markup yields what was recovered, never raises."""
    parser = _LinkExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - one bad summary must not fail a gather
        logger.debug("could not fully parse a summary for links")
    return parser.hrefs


def _registrable_domain(url: str) -> str | None:
    """The host of `url`, `www.` stripped, or None if it has no usable host.

    Deliberately not a public-suffix lookup: that needs a dependency and a list
    that goes stale, and the question here is only "have I seen this site before",
    where host-level comparison is exactly right. `canonicalize_url` already
    normalizes case and `www.`, so this reuses it rather than re-deriving the rule.
    """
    try:
        host = urlsplit(canonicalize_url(url)).hostname
    except ValueError:
        return None
    return host or None


@dataclass(frozen=True, slots=True)
class CurationEvidence:
    """Everything deterministic the scout is given. Facts only."""

    window_days: int
    yield_rows: list[db.SourceYield] = field(default_factory=list)
    feedback_by_source: dict[str, dict[str, int]] = field(default_factory=dict)
    """`source_id -> {rung_or_missed: count}`, each item reduced to its highest
    ladder rung so one item cannot inflate two columns."""

    outbound_domains: list[tuple[str, int]] = field(default_factory=list)
    """`(domain, references)`, most-referenced first, excluding anything already
    covered by a configured source. Partial by construction — see the module
    docstring on RSS summaries."""

    kept_titles: list[str] = field(default_factory=list)
    """Titles of the highest-ranked kept items, best first. What the operator is
    actually reading, which is the signal that survives the href gap."""


def _highest_rungs(
    verdicts: list[tuple[str, int, str]],
) -> dict[str, dict[str, int]]:
    """Reduce raw verdict rows to per-source counts of each item's highest rung.

    `verdicts` is `(source_id, item_id, verdict)` — one row per stored mark, so an
    item marked both `useful` and `exceptional` appears twice and must be counted
    once, at `exceptional`. `missed` is tallied on its own because it is not on the
    ladder and `LADDER.index` would raise on it.
    """
    ladder_by_item: dict[tuple[str, int], str] = {}
    missed_by_source: Counter[str] = Counter()

    for source_id, item_id, verdict in verdicts:
        if verdict == MISSED_VERDICT:
            missed_by_source[source_id] += 1
            continue
        if verdict not in LADDER:
            logger.warning(
                "ignoring an unknown feedback verdict while gathering evidence",
                extra={"verdict": verdict, "source_id": source_id},
            )
            continue
        key = (source_id, item_id)
        current = ladder_by_item.get(key)
        if current is None or LADDER.index(verdict) > LADDER.index(current):
            ladder_by_item[key] = verdict

    tallies: dict[str, dict[str, int]] = {}
    for (source_id, _item_id), rung in ladder_by_item.items():
        tallies.setdefault(source_id, {})[rung] = tallies.setdefault(source_id, {}).get(rung, 0) + 1
    for source_id, count in missed_by_source.items():
        tallies.setdefault(source_id, {})[MISSED_VERDICT] = count
    return tallies


def _covered_domains(sources: SourcesConfig, kept: list[db.KeptItem]) -> set[str]:
    """Domains the operator already receives, so they are not proposed back to them.

    Two sources of "already covered": the configured feed URLs, and the domains of
    items already ingested — an RSS item's own URL *is* its source's domain, so a
    blog we subscribe to can never appear as a discovery.
    """
    covered: set[str] = set()
    for source in sources.rss:
        if domain := _registrable_domain(source.url):
            covered.add(domain)
    for item in kept:
        if domain := _registrable_domain(item.url):
            covered.add(domain)
    # Release watches and HN are handled by their own proposal kinds, and both
    # would otherwise flood the list with incidental links.
    covered.update({"github.com", "news.ycombinator.com"})
    return covered


def _outbound_domains(sources: SourcesConfig, kept: list[db.KeptItem]) -> list[tuple[str, int]]:
    """Count domains linked *from* kept items, excluding ones already covered."""
    covered = _covered_domains(sources, kept)
    counts: Counter[str] = Counter()
    for item in kept:
        if "href" not in item.summary.lower():
            # Cheap skip for the common case: an RSS summary had its markup
            # stripped at ingest, so there is nothing here to find.
            continue
        for href in _extract_hrefs(item.summary):
            domain = _registrable_domain(href)
            if domain and domain not in covered:
                counts[domain] += 1
    return [
        (domain, count)
        for domain, count in counts.most_common(_OUTBOUND_DOMAIN_LIMIT)
        if count >= _MIN_DOMAIN_REFERENCES
    ]


def gather_evidence(
    conn: sqlite3.Connection,
    sources: SourcesConfig,
    *,
    window_days: int,
    now: datetime | None = None,
) -> CurationEvidence:
    """Assemble the scout's evidence from stored rows. Reads only.

    `now` defaults to the current time and is a parameter so tests freeze it. The
    window bound is computed in UTC deliberately: `db._window_start` would coerce a
    local-zone value anyway, but keeping the caller in UTC means that coercion
    stays a safety net rather than the thing load-bearing correctness depends on.
    """
    stamp = now or datetime.now(UTC)
    since = stamp - timedelta(days=window_days)

    yield_rows = db.source_yield_stats(conn, since=since)
    verdicts = db.feedback_verdicts_since(conn, since=since)
    kept = db.kept_items(conn, since=since, limit=_KEPT_ITEM_PROMPT_LIMIT)

    evidence = CurationEvidence(
        window_days=window_days,
        yield_rows=yield_rows,
        feedback_by_source=_highest_rungs(verdicts),
        outbound_domains=_outbound_domains(sources, kept),
        kept_titles=[item.title for item in kept],
    )
    logger.info(
        "gathered curation evidence",
        extra={
            "window_days": window_days,
            "source_count": len(yield_rows),
            "marked_sources": len(evidence.feedback_by_source),
            "outbound_domains": len(evidence.outbound_domains),
            "kept_titles": len(evidence.kept_titles),
        },
    )
    return evidence
