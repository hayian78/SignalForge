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
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urlsplit

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

_BULLET: Final = re.compile(r"^[ \t]*[-*+][ \t]+")
"""The bullet and its indent. Nested lists count — plenty of lists group by
sub-heading and indent the entries under it."""

_LEADING_DECORATION: Final = re.compile(
    r"""^(?:
          <[^>]+>            # an <img>/<sub> tag some lists prefix
        | \*{1,2} | _{1,2}   # bold/italic wrapping the link
        # A leading badge/image, optionally wrapped in a link. The `!` is
        # required: without it this alternative eats the entry's own link.
        | \[?!\[[^\]]*\]\([^)]*\)(?:\]\([^)]*\))?
        | [^\w\[<]           # emoji, arrows, separators, stray punctuation
    )+""",
    re.VERBOSE,
)
"""What sits between the bullet and the link. Awesome lists put emoji, badges,
bold markers and `<img>` tags here as house style — `punkpeye/awesome-mcp-servers`
and `e2b-dev/awesome-ai-agents` between them use most of it. Consuming this
rather than demanding `[` right after the bullet is the difference between
reading a real list and reading none of it."""

_LINK: Final = re.compile(r"\[(?P<name>[^\]]*)\]\((?P<url>[^()\s]*(?:\([^()]*\)[^()\s]*)*)")
"""One markdown link. The URL group allows *one* level of balanced parens so
`.../Agent_(AI)` survives — a naive `[^)]+` truncates it into a broken citation,
and the citation is the whole point (NEVER rule 7)."""

_INLINE_MEDIA: Final = re.compile(r"\[?!\[[^\]]*\]\([^)]*\)(?:\]\([^)]*\))?")
"""A badge or image, optionally wrapped in a link. Stripped from a description
before it reaches triage: `punkpeye/awesome-mcp-servers` puts a glama.ai score
badge after most entries, and its markdown is pure token cost in a summary."""

_DESCRIPTION_LEAD: Final = re.compile(r"^[\s\-–—:·|>]+")
"""Awesome lists separate name from description with any of these."""

_CODE_FENCE: Final = re.compile(r"^[ \t]*(```|~~~)")

_BADGE_HOSTS: Final = frozenset(
    {
        "img.shields.io",
        "shields.io",
        "badge.fury.io",
        "badgen.net",
        "camo.githubusercontent.com",
        "travis-ci.org",
        "travis-ci.com",
        "circleci.com",
        "codecov.io",
        "app.codecov.io",
    }
)
"""Hosts that only ever serve a badge image. Not a taste filter — a badge URL is
not a document, so an item citing one is an uncitable claim wearing a link. This
is a property of the hosts, not of the operator's interests, so it belongs here
rather than in `sources.yaml` (contrast the keyword lists, which are config)."""

_MAX_LABEL_REPEATS: Final = 2
"""How often one link label may repeat before every copy is treated as
navigation rather than as entries.

`e2b-dev/awesome-ai-agents` describes each project with an `### Heading` and a
bullet list of links beneath it — "GitHub" 92 times, "Web" 81, "Discord" 51. A
bullet-link parser reads those as 621 entries whose titles are `GitHub` and
`Discord`, which is exactly the noise this pipeline exists to remove. A repeated
label is a structural signal, not a keyword blocklist: it needs no vocabulary,
generalizes to lists nobody has seen, and leaves genuinely repeated project
links (2 in `punkpeye/awesome-mcp-servers`) alone."""


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


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _parse_line(line: str) -> AwesomeEntry | None:
    """One bullet line as an entry, or None if it is not one.

    The entry's link is the **first** link on the line after any leading
    decoration, not the last. That ordering matters: `punkpeye/awesome-mcp-servers`
    writes `- [project](repo) [![score](badge)](glama) 📇 — description`, so
    taking the last link would cite a score badge instead of the project.
    Image-only links (`[![…](…)](…)`) are skipped rather than preferred, which
    handles the other house style — a badge *before* the name — by the same rule.
    """
    body = _BULLET.sub("", line, count=1)
    if body == line:  # not a bullet
        return None
    body = _LEADING_DECORATION.sub("", body, count=1)

    match = _LINK.match(body)
    if match is None:
        return None

    url = match.group("url").strip()
    if not is_safe_url(url) or _host_of(url) in _BADGE_HOSTS:
        return None

    name = flatten_to_single_line(match.group("name"))
    if not name or name.startswith("!"):
        return None

    rest = body[match.end() :]
    # Drop the closing paren the link regex leaves behind, plus any markdown
    # link title inside it.
    rest = re.sub(r'^(?:[ \t]+"[^"]*")?\)', "", rest, count=1)
    rest = _INLINE_MEDIA.sub("", rest)
    description = flatten_to_single_line(_DESCRIPTION_LEAD.sub("", rest))
    return AwesomeEntry(name=name, url=url, description=description)


def parse_entries(markdown: str) -> list[AwesomeEntry]:
    """Every list entry in document order, in the shapes real lists actually use.

    Skips fenced code blocks: a README's usage example is full of bullets and
    links that are not entries. Everything surviving is filtered on
    `is_safe_url`, which drops in-document anchors (`#contents`), relative links
    to a `CONTRIBUTING.md`, and `javascript:`/`data:` links in one test.

    **Repeated labels are dropped as navigation**, not kept as entries — see
    `_MAX_LABEL_REPEATS`. This is the pass that stops a link-list-per-project
    README from yielding ninety items called "GitHub".

    A wrapped description keeps only its first line. The parser is deliberately
    line-anchored — that is what makes it structurally impossible to smuggle a
    vault checkbox marker across a line break (NEVER rule 17) — and a truncated
    summary is a much better trade than a forgeable one.

    Text is flattened here, at parse time: a list entry is world-authored, an
    entry name lands in a vault line, and a name carrying a marker would
    otherwise forge a decision.
    """
    parsed: list[AwesomeEntry] = []
    seen_urls: set[str] = set()
    in_fence = False

    for line in markdown.splitlines():
        if _CODE_FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        entry = _parse_line(line)
        if entry is None:
            continue
        # First occurrence wins. Awesome lists routinely repeat a link across
        # sections, and a duplicate would otherwise fight itself for the same
        # UNIQUE (source_id, external_id) slot.
        if entry.url in seen_urls:
            continue
        seen_urls.add(entry.url)
        parsed.append(entry)

    label_counts = Counter(entry.name.casefold() for entry in parsed)
    return [entry for entry in parsed if label_counts[entry.name.casefold()] <= _MAX_LABEL_REPEATS]


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
            # Drop the ETag we just earned. Otherwise the next run 304s before
            # it ever parses, this anomaly is never re-detected, and a list the
            # parser cannot read goes dark on day one after exactly one warning.
            # Same reasoning `HttpFetcher.invalidate` exists for: a 200 that is
            # technically cacheable but useless to us.
            fetcher.invalidate(readme_url(self.repo, filename), source_id=self.source_id)
            logger.warning(
                "awesome list %s: README parsed to zero entries; leaving the baseline alone "
                "and forcing an unconditional refetch next run",
                self.source_id,
                extra={"source_id": self.source_id, "repo": self.repo, "readme": filename},
            )
            return IngestResult()

        state = fetcher.validators.read_state(self.source_id, BASELINE_KEY)
        known = state.get("urls")
        # `repo` is verified, not decorative: `_safe_component` collapses every
        # non-`[A-Za-z0-9._-]` run to `-`, so `a/b-c` and `a-b/c` resolve to the
        # same sidecar path. Without this check one list would silently inherit
        # the other's baseline and emit nothing, forever.
        seeded = isinstance(known, list) and state.get("repo") == self.repo
        baseline: set[str] = {str(u) for u in known} if isinstance(known, list) else set()
        if not seeded:
            baseline = set()

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
            # No silent caps: say exactly what was dropped, *in the message*.
            # The configured log format renders no `extra` fields, so anything
            # only carried there is invisible to the operator reading cron.log.
            # These entries enter the baseline and so are dropped permanently,
            # which makes this the only record that they existed.
            dropped = [entry.url for entry in new_entries[self.max_new_entries :]]
            logger.warning(
                "awesome list %s: %d new entries exceeded the per-run cap of %d; "
                "dropping %d permanently: %s",
                self.source_id,
                len(new_entries),
                self.max_new_entries,
                len(dropped),
                ", ".join(dropped),
                extra={
                    "source_id": self.source_id,
                    "new_entries": len(new_entries),
                    "cap": self.max_new_entries,
                    "dropped": dropped,
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
        free run into two wasted requests.

        **Only a 404 advances to the next spelling.** `HttpFetcher.get` has
        already exhausted its retry ladder before raising, so treating a 503 or
        a timeout as "wrong spelling" would fire two more full ladders — up to
        nine requests for one list — and then report the *last* candidate's
        404 in `runs.errors`, hiding the real failure. `runs.errors` is the
        monitoring channel (CLAUDE.md §7), so a misattributed error is worse
        than a loud one.
        """
        last_404: FetchError | None = None
        for filename in _README_CANDIDATES:
            try:
                response = await fetcher.get(
                    readme_url(self.repo, filename), source_id=self.source_id
                )
            except FetchError as exc:
                if exc.status_code != 404:
                    raise
                last_404 = exc
                continue
            return response, filename

        raise last_404 or FetchError(
            f"no README found for {self.repo}", url=readme_url(self.repo, "README.md")
        )

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
