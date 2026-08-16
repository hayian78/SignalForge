"""Awesome-list diffing — new entries in a curated list become items (DESIGN §7).

A curated list is a slow-moving, human-maintained index. What is worth reading
is not the list, it is **what someone added to it this week** — so this is a
diff, not a scrape.

**Fetched over HTTP, not shallow-cloned.** DESIGN §7 specced `git clone --depth
1` plus a README diff. Fetching `raw.githubusercontent.com/<repo>/HEAD/README.md`
through the shared `HttpFetcher` gets the same bytes while reusing everything
that already exists here: conditional GET (an unchanged list costs one 304 and
nothing else), `Retry-After`-aware retries, the global politeness cap, and raw
payload archiving. A clone would need `git` on PATH, a clone cache to manage,
and its own failure handling outside all of it. The deviation is recorded in
DESIGN §7.

**The baseline is the whole design.** A large awesome list is 500+ entries. On a
first run there is nothing to diff against, so emitting them all would push
hundreds of items into a triage batch — real money for a list of links the
operator has never asked to read. So a first run **writes the baseline and emits
nothing**, and only entries appearing *after* it become items. The baseline goes
through `ValidatorStore.stage_state`, which means it becomes durable only once
the items actually reached the database: written eagerly, a crash between fetch
and persist would mark unseen entries as seen and lose them for good.

**Deterministic throughout** (NEVER rules 2, 3). Parsing markdown list entries
and subtracting a set is not judgment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from signalforge.config import SourcesConfig
from signalforge.ingest.base import (
    FetchError,
    FetchResponse,
    HttpFetcher,
    IngestError,
    IngestResult,
    truncate_summary,
)
from signalforge.models import Item, SourceType, flatten_to_single_line, is_safe_url

__all__ = [
    "AWESOME_SOURCE_PREFIX",
    "AwesomeEntry",
    "AwesomeListIngestor",
    "BASELINE_KEY",
    "build_awesome_ingestors",
    "parse_entries",
    "readme_url",
]

logger = logging.getLogger(__name__)

AWESOME_SOURCE_PREFIX: Final = "awesome:"
"""`items.source_id` prefix, e.g. `awesome:e2b-dev/awesome-ai-agents`.

`source_type` stays `github` — it is a GitHub README — but the prefix keeps
per-source yield stats (the Phase 2 pruning data) separable from the same repo's
releases, which are a completely different kind of item."""

BASELINE_KEY: Final = "awesome-entries"
"""Sidecar key for the seen-entry baseline, under the source's `_meta/`."""

_README_CANDIDATES: Final = ("README.md", "readme.md", "Readme.md")
"""Tried in order. GitHub's raw host is case-sensitive, and awesome lists are
not consistent about it."""

_ENTRY_PATTERN: Final = re.compile(
    r"""^[ \t]*             # leading indent: nested lists count too
        [-*+][ \t]+         # the bullet
        \[(?P<name>[^\]]+)\]   # [name]
        \((?P<url>[^)\s]+)     # (url
        (?:[ \t]+"[^"]*")?  # optional markdown link title
        \)
        (?P<rest>.*)$       # the description, usually after an em dash
    """,
    re.VERBOSE,
)

_DESCRIPTION_LEAD: Final = re.compile(r"^[\s\-–—:·|]+")
"""Awesome lists separate name from description with any of these."""

_CODE_FENCE: Final = re.compile(r"^[ \t]*(```|~~~)")


@dataclass(frozen=True, slots=True)
class AwesomeEntry:
    """One parsed list entry: a link, its label, and whatever followed it."""

    name: str
    url: str
    description: str


def readme_url(repo: str, filename: str) -> str:
    """Raw URL for a repo's README at its default branch.

    `HEAD` rather than `main`/`master` so a list that never renamed its default
    branch still resolves — the raw host understands it.
    """
    return f"https://raw.githubusercontent.com/{repo}/HEAD/{filename}"


def parse_entries(markdown: str) -> list[AwesomeEntry]:
    """Every `- [name](url) — description` bullet, in document order.

    Skips fenced code blocks: a README's usage example is full of bullets and
    links that are not entries. Everything surviving that is filtered on
    `is_safe_url`, which drops in-document anchors (`#contents`), relative links
    to a `CONTRIBUTING.md`, and `javascript:`/`data:` links in one test.

    Text is flattened at parse time, not at render time (NEVER rule 17): a list
    entry is world-authored, an entry name lands in a vault line, and a name
    containing a checkbox marker would otherwise forge a decision.
    """
    entries: list[AwesomeEntry] = []
    seen_urls: set[str] = set()
    in_fence = False

    for line in markdown.splitlines():
        if _CODE_FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        match = _ENTRY_PATTERN.match(line)
        if match is None:
            continue

        url = match.group("url").strip()
        if not is_safe_url(url):
            continue
        # First occurrence wins. Awesome lists routinely repeat a link across
        # sections, and a duplicate would otherwise fight itself for the same
        # UNIQUE (source_id, external_id) slot.
        if url in seen_urls:
            continue
        seen_urls.add(url)

        name = flatten_to_single_line(match.group("name"))
        description = flatten_to_single_line(_DESCRIPTION_LEAD.sub("", match.group("rest")))
        if not name:
            continue
        entries.append(AwesomeEntry(name=name, url=url, description=description))

    return entries


class AwesomeListIngestor:
    """Diffs one repo listed under `github.awesome_lists` in `sources.yaml`."""

    source_type = SourceType.GITHUB

    def __init__(self, repo: str, *, max_summary_chars: int, max_new_entries: int) -> None:
        self.repo = repo
        self.source_id = f"{AWESOME_SOURCE_PREFIX}{repo}"
        self.max_summary_chars = max_summary_chars
        self.max_new_entries = max_new_entries

    async def ingest(self, fetcher: HttpFetcher) -> IngestResult:
        """Fetch the README, diff it against the baseline, emit what is new."""
        try:
            response, filename = await self._fetch_readme(fetcher)
        except Exception as exc:  # noqa: BLE001 - one list never kills the run (§7)
            logger.warning(
                "awesome list fetch failed",
                extra={"source_id": self.source_id, "repo": self.repo, "error": str(exc)},
            )
            return IngestResult(
                errors=[
                    IngestError.from_exception(
                        exc, source_id=self.source_id, source_type=self.source_type
                    )
                ]
            )

        if response is None:
            logger.debug(
                "awesome list unchanged since last run", extra={"source_id": self.source_id}
            )
            return IngestResult()

        entries = parse_entries(response.text)
        if not entries:
            # Not an error: a 200 that parses to nothing is far more likely to be
            # a moved README than a genuinely empty list, and treating it as an
            # empty list would wipe the baseline and re-emit everything next run.
            logger.warning(
                "awesome list README parsed to zero entries; leaving the baseline alone",
                extra={"source_id": self.source_id, "repo": self.repo, "readme": filename},
            )
            return IngestResult()

        state = fetcher.validators.read_state(self.source_id, BASELINE_KEY)
        known = state.get("urls")
        seeded = isinstance(known, list)
        baseline: set[str] = {str(u) for u in known} if isinstance(known, list) else set()

        # The baseline covers every entry seen this run, capped or not — the cap
        # bounds a single run's triage bill, it does not defer entries to
        # tomorrow. Deferring them would make a long list dribble in forever and
        # make "new" mean "new to us", not "new to the list".
        self._stage_baseline(fetcher, entries)

        if not seeded:
            logger.info(
                "seeding awesome list baseline; emitting no items this run",
                extra={"source_id": self.source_id, "entry_count": len(entries)},
            )
            return IngestResult()

        new_entries = [entry for entry in entries if entry.url not in baseline]
        if len(new_entries) > self.max_new_entries:
            # No silent caps (DESIGN §7.1's rule, and the reason `runs.errors` is
            # the monitoring channel): say exactly what was dropped.
            logger.warning(
                "awesome list produced more new entries than the per-run cap; dropping the excess",
                extra={
                    "source_id": self.source_id,
                    "new_entries": len(new_entries),
                    "cap": self.max_new_entries,
                    "dropped": [entry.url for entry in new_entries[self.max_new_entries :]],
                },
            )
            new_entries = new_entries[: self.max_new_entries]

        items = [self._to_item(entry, raw_path=response.raw_path) for entry in new_entries]
        logger.info(
            "awesome list diffed",
            extra={
                "source_id": self.source_id,
                "entry_count": len(entries),
                "item_count": len(items),
            },
        )
        return IngestResult(items=items)

    async def _fetch_readme(self, fetcher: HttpFetcher) -> tuple[FetchResponse | None, str]:
        """The first README candidate that answers, and which spelling it was.

        A 304 counts as answering and short-circuits: an unchanged list is the
        common daily case, and trying the other spellings past it would turn a
        free run into two wasted requests. A 404 raises `FetchError`, which is
        the signal to try the next spelling; the last one's error is what
        surfaces if none of them exist.
        """
        last_error: Exception | None = None
        for filename in _README_CANDIDATES:
            try:
                response = await fetcher.get(
                    readme_url(self.repo, filename), source_id=self.source_id
                )
            except Exception as exc:  # noqa: BLE001 - try the next spelling
                last_error = exc
                continue
            return response, filename

        if last_error is not None:
            raise last_error
        raise FetchError(f"no README found for {self.repo}", url=readme_url(self.repo, "README.md"))

    def _stage_baseline(self, fetcher: HttpFetcher, entries: list[AwesomeEntry]) -> None:
        """Stage every URL seen this run, durable only once the items persist."""
        fetcher.validators.stage_state(
            self.source_id,
            BASELINE_KEY,
            {"repo": self.repo, "urls": sorted(entry.url for entry in entries)},
        )

    def _to_item(self, entry: AwesomeEntry, *, raw_path: str | None) -> Item:
        """One list entry as an `Item`.

        `external_id` is the entry's URL rather than a hash of the line: the
        identity of an entry is what it points at, so a maintainer rewording a
        description must not resurrect it as a new item.

        `published_at` is left None — a list entry has no publication date, and
        inventing "now" would let the digest's day bucketing claim the linked
        project was published today.
        """
        return Item(
            source_id=self.source_id,
            source_type=self.source_type,
            external_id=entry.url,
            url=entry.url,
            title=entry.name,
            published_at=None,
            fetched_at=datetime.now(UTC),
            summary=truncate_summary(entry.description, max_chars=self.max_summary_chars)
            if entry.description
            else None,
            raw_path=raw_path,
        )


def build_awesome_ingestors(config: SourcesConfig) -> list[AwesomeListIngestor]:
    """One ingestor per repo under `github.awesome_lists`. Sources are config."""
    if config.github is None or not config.github.awesome_lists:
        return []
    # `GithubConfig` refuses to validate with lists but no cap, so this cannot
    # be None here — the assertion is for mypy, not for runtime doubt.
    cap = config.github.awesome_max_new_per_run
    assert cap is not None  # noqa: S101 - enforced by GithubConfig's model validator
    return [
        AwesomeListIngestor(
            repo,
            max_summary_chars=config.defaults.max_summary_chars,
            max_new_entries=cap,
        )
        for repo in config.github.awesome_lists
    ]
