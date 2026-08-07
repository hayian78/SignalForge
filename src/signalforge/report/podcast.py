"""Podcast script → vault (DESIGN §13.3, Stage 4) — deterministic assembly, no LLM calls.

Reads `items` only — via `db.get_item`, resolving the citations
`synth.podcast.build_script` already validated — and writes one markdown
file per date under `vault/podcast/YYYY-MM-DD.md`. This module never calls
an LLM and never writes dialogue of its own: it renders exactly what
`synth/podcast.py` already produced (CLAUDE.md §2, NEVER rule 1).

### Truncation vs. bad citation

A cited item can lose coverage two structurally different ways, and the
rendered file says which. `BuiltScript.dropped_item_ids` is a **bad
citation** — the model named an id it was never given, so the whole
segment built around it is discarded, never rendered even as a teaser.
`teasered_item_ids` / `truncation_dropped_item_ids` are **length** — the
script ran over `PODCAST_MAX_SCRIPT_CHARS` and a legitimate segment was
shrunk or cut to fit. Both look the same from "this item isn't fully
covered," but only one of them means the model fabricated something, so
`CoverageGap.reason` never merges them into one undifferentiated list —
the operator asked for exactly this split (2026-08-07), expecting it to
matter again once real episodes accumulate. `_coverage_gaps` also filters
both sets against what actually aired: nothing forbids the model citing
the same item in two segments, so a drop recorded for one segment must not
brand an item "not covered" when another surviving segment says otherwise.

### Citation resolution at render time

`synth.podcast.build_script` validates a cited id against the ids it sent
the model — it never touches the DB (CLAUDE.md §2). This module is the
first thing in the pipeline that maps an `item_id` back to a real `Item`,
which means it is also the first thing that can discover a row is gone
(deleted, or never committed) between script generation and vault write.
A segment left with no resolvable citation is dropped entirely rather than
rendered uncited (CLAUDE.md §5, NEVER rule 7) — the same rule
`report/daily.py::_to_line` enforces for the digest, applied here to a
segment instead of a line. A coverage-gap entry gets gentler treatment: an
id in `dropped_item_ids` is *expected* not to resolve (the model invented
it, so it is very unlikely to be a real row) — rendering it as a linkless
`- item N — ...` line, rather than silently omitting the row, is what
keeps a fabricated citation from being the one failure mode that vanishes
from the record instead of the one most worth keeping.

### The `ScriptContext` round trip

`write_script` renders and writes the file; `parse_script` reads one back
into the exact same shape. That symmetry is deliberate (DESIGN §13.2's
same-context contract, applied to a second channel): Stage 5/6 read the
file a TTS pass will voice, not the run that generated it, so the audio
can never disagree with what a human actually sees in the vault. Every
id `parse_script` needs to recover is carried in an HTML comment, the same
`<!-- sf:... -->` convention `feedback.checkbox_marker` already uses for
machine-parseable state embedded in human-readable markdown. A title is
world-authored text rendered inside `[title](url)` link syntax, so `]` is
escaped on the way out and unescaped on the way back — otherwise a title
containing a literal `](` could shift where the parser thinks the link
ends, corrupting the recovered `url`. `is_safe_url` (`models.py`) is
checked again on every URL `parse_script` recovers, refusing rather than
trusting a URL a corrupted parse might have produced.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import jinja2
import yaml

from signalforge.db import get_item
from signalforge.models import flatten_to_single_line, is_safe_url
from signalforge.synth.podcast import BuiltScript

if TYPE_CHECKING:
    from signalforge.llm import PodcastSegment

__all__ = [
    "CoverageGap",
    "ScriptContext",
    "ScriptSegment",
    "ScriptTurn",
    "SourceLink",
    "build_script_context",
    "parse_script",
    "render_script",
    "script_path",
    "write_script",
]

logger = logging.getLogger(__name__)

_TEMPLATE_NAME = "podcast.md.j2"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_INTRO_LABEL: Final = "Intro"
_OUTRO_LABEL: Final = "Outro"
_COVERAGE_GAPS_LABEL: Final = "Coverage gaps"

_TEASED_NOTE: Final = " (shortened — episode ran long)"
"""Appended to a truncated segment's heading — single source of truth
between `render_script` (via the `teased_note` template global) and
`parse_script`, the same "one function, two directions" discipline
`feedback.checkbox_marker` uses for its own wire format."""

_REASON_LABELS: Final[dict[str, str]] = {
    "bad_citation": "dropped: cited but never offered to the model",
    "truncated": "dropped: episode ran long",
}

_TITLE_BRACKET_ESCAPE: Final = "&#93;"
"""Stands in for a literal `]` inside a title rendered as `[title](url)`.
Not a general HTML-entity encoder — this is the one character that can
make the parser's `\\](` boundary land in the wrong place (see this
module's docstring). A title that happens to already contain this exact
six-character sequence unescapes as a stray `]` — a cosmetic, vanishingly
unlikely collision, not a security issue, since it only ever affects
display text, never the separately-validated `url`."""


@dataclass(frozen=True, slots=True)
class ScriptTurn:
    """One line of dialogue, as rendered.

    `speaker` is the presenter's *display name* (`PodcastChannelConfig.presenter_a`/
    `_b`), never the model's internal "A"/"B" letter — recovering a turn's
    speaker from a written file needs no config lookup this way, which is
    what makes `parse_script` a pure function of the file's own bytes.
    """

    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class SourceLink:
    """One citation backing a segment, resolved from `items` at render time.

    Not carried over from `synth.podcast.PodcastSegment.item_ids` unchanged:
    resolving here, rather than trusting the id, is what lets a row deleted
    between script generation and vault write drop out of the citation list
    instead of rendering a dead link (see this module's docstring).
    """

    item_id: int
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class ScriptSegment:
    """One rendered block — `Intro`, `Segment N`, or `Outro`, in document order.

    `sources` is always empty for `Intro`/`Outro`: the v2 prompt shape
    forbids either from citing a story (`synth.podcast.PODCAST_SCRIPT_VERSION`'s
    docstring), so there is nothing here to resolve for them.
    """

    label: str
    turns: tuple[ScriptTurn, ...]
    sources: tuple[SourceLink, ...] = ()
    teased: bool = False
    """True if `_truncate_to_fit` shrank this segment to a teaser
    (`BuiltScript.teasered_item_ids`) — rendered as a visible heading note,
    never silently, so a listener who reads the script before listening
    knows this story got less time than the others."""


@dataclass(frozen=True, slots=True)
class CoverageGap:
    """One item that lost all coverage, and why.

    The two reasons are never merged (see this module's docstring): a
    `bad_citation` means the model fabricated something; `truncated` means
    a legitimate story just didn't fit today's runtime. Collapsing them
    would let a real confabulation read as routine cost trimming.

    `title`/`url` are `None` when the item could not be resolved from the
    DB — expected for `bad_citation` (the model invented the id, so it is
    unlikely to name a real row) and rendered as a linkless line rather
    than being dropped from the file, so the one reason most worth keeping
    a record of is never the one that silently disappears.
    """

    item_id: int
    title: str | None
    url: str | None
    reason: Literal["bad_citation", "truncated"]


@dataclass(frozen=True, slots=True)
class ScriptContext:
    """Everything one rendered episode needs — assembled once from a
    `BuiltScript` and the DB, rendered once, and the exact shape
    `parse_script` recovers from the file `write_script` wrote. `segments`
    is empty on a day with no usable episode (no items, a refused call, or
    every segment losing its last citation) — the template still renders a
    file for that date, same as `report/daily.py`'s "no items today"
    branch, so a quiet day is a visible fact in the vault rather than a
    missing one.
    """

    date: Date
    script_version: str
    model: str
    item_count: int
    """Distinct items actually covered by a rendered segment — not
    `len(built.script.segments)` and not the caller's input count, so a
    segment dropped for an unresolvable citation at render time is
    reflected here too."""
    segments: tuple[ScriptSegment, ...]
    coverage_gaps: tuple[CoverageGap, ...] = ()


def _resolve_source(conn: sqlite3.Connection, item_id: int) -> SourceLink | None:
    item = get_item(conn, item_id)
    if item is None:
        logger.warning(
            "podcast citation could not be resolved; item missing from db",
            extra={"item_id": item_id},
        )
        return None
    return SourceLink(item_id=item_id, title=flatten_to_single_line(item.title), url=item.url)


def _coverage_gap(
    conn: sqlite3.Connection, item_id: int, *, reason: Literal["bad_citation", "truncated"]
) -> CoverageGap:
    item = get_item(conn, item_id)
    if item is None:
        logger.warning(
            "podcast coverage-gap item missing from db; rendering without a citation link",
            extra={"item_id": item_id, "reason": reason},
        )
        return CoverageGap(item_id=item_id, title=None, url=None, reason=reason)
    return CoverageGap(
        item_id=item_id, title=flatten_to_single_line(item.title), url=item.url, reason=reason
    )


def _coverage_gaps(
    conn: sqlite3.Connection, built: BuiltScript, *, aired_ids: frozenset[int]
) -> tuple[CoverageGap, ...]:
    """One gap per item that never aired — `dropped_item_ids` (bad citation)
    before `truncation_dropped_item_ids` (length), each already
    deterministically ordered by the pipeline stage that produced them
    (CLAUDE.md §3).

    `aired_ids` and `seen` both guard the same failure: nothing forbids the
    model citing one item across two segments, so a drop recorded against
    one segment must not contradict a citation that actually survived in
    another, and an id recorded in both drop sets (should not happen per
    `synth.podcast`'s disjointness invariant, but this function does not
    assume it) must not render twice.
    """
    gaps: list[CoverageGap] = []
    seen: set[int] = set()

    def add(item_id: int, reason: Literal["bad_citation", "truncated"]) -> None:
        if item_id in aired_ids or item_id in seen:
            return
        seen.add(item_id)
        gaps.append(_coverage_gap(conn, item_id, reason=reason))

    for item_id in built.dropped_item_ids:
        add(item_id, "bad_citation")
    for item_id in built.truncation_dropped_item_ids:
        add(item_id, "truncated")
    return tuple(gaps)


def _story_segment(
    conn: sqlite3.Connection,
    segment: PodcastSegment,
    *,
    label: str,
    speaker_name: dict[str, str],
    teasered_ids: set[int],
) -> ScriptSegment | None:
    """One `Segment N` block, or `None` if every citation it carried failed
    to resolve — NEVER rule 7 applied at render time, not just build time
    (see this module's docstring)."""
    sources = [
        source
        for item_id in segment.item_ids
        if (source := _resolve_source(conn, item_id)) is not None
    ]
    if not sources:
        logger.warning(
            "podcast segment had no surviving citation at render time; dropping segment",
            extra={"item_ids": segment.item_ids},
        )
        return None
    turns = tuple(
        ScriptTurn(speaker=speaker_name[turn.speaker], text=flatten_to_single_line(turn.text))
        for turn in segment.turns
    )
    teased = any(item_id in teasered_ids for item_id in segment.item_ids)
    return ScriptSegment(label=label, turns=turns, sources=tuple(sources), teased=teased)


def build_script_context(
    conn: sqlite3.Connection,
    built: BuiltScript,
    *,
    date: Date,
    presenter_a: str,
    presenter_b: str,
) -> ScriptContext:
    """Assemble everything `podcast.md.j2` needs for one episode. No writes.

    Intro/outro render only when at least one story segment survives
    citation resolution — a show that opened and closed around zero
    stories is the same "nothing usable" outcome `build_script` already
    returns as an empty `BuiltScript.script`, just discovered one step
    later (a row deleted after the script was built rather than before).
    """
    speaker_name = {"A": presenter_a, "B": presenter_b}

    story_segments: list[ScriptSegment] = []
    if built.script is not None:
        teasered_ids = set(built.teasered_item_ids)
        for index, segment in enumerate(built.script.segments, start=1):
            rendered = _story_segment(
                conn,
                segment,
                label=f"Segment {index}",
                speaker_name=speaker_name,
                teasered_ids=teasered_ids,
            )
            if rendered is not None:
                story_segments.append(rendered)
        if not story_segments:
            logger.warning("podcast script had no segments left with a resolvable citation")

    aired_ids = frozenset(
        source.item_id for segment in story_segments for source in segment.sources
    )
    coverage_gaps = _coverage_gaps(conn, built, aired_ids=aired_ids)

    if not story_segments:
        return ScriptContext(
            date=date,
            script_version=built.script_version,
            model=built.model,
            item_count=0,
            segments=(),
            coverage_gaps=coverage_gaps,
        )

    # `story_segments` is only ever appended to inside the `built.script is
    # not None` branch above, so reaching here with a non-empty list proves
    # `built.script` is not None.
    assert built.script is not None  # noqa: S101 - guaranteed by the branch above
    intro = ScriptSegment(
        label=_INTRO_LABEL,
        turns=tuple(
            ScriptTurn(speaker=speaker_name[turn.speaker], text=flatten_to_single_line(turn.text))
            for turn in built.script.intro_turns
        ),
    )
    outro = ScriptSegment(
        label=_OUTRO_LABEL,
        turns=tuple(
            ScriptTurn(speaker=speaker_name[turn.speaker], text=flatten_to_single_line(turn.text))
            for turn in built.script.outro_turns
        ),
    )

    return ScriptContext(
        date=date,
        script_version=built.script_version,
        model=built.model,
        item_count=len(aired_ids),
        segments=(intro, *story_segments, outro),
        coverage_gaps=coverage_gaps,
    )


def _escape_title(title: str) -> str:
    return title.replace("]", _TITLE_BRACKET_ESCAPE)


def _unescape_title(title: str) -> str:
    return title.replace(_TITLE_BRACKET_ESCAPE, "]")


def _template_env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    # Single source of truth between the template and `parse_script`, the
    # same discipline `report/daily.py` uses for `checkbox_marker`.
    env.globals["teased_note"] = _TEASED_NOTE
    env.globals["reason_labels"] = _REASON_LABELS
    env.globals["escape_title"] = _escape_title
    return env


def render_script(context: ScriptContext) -> str:
    """Fill `podcast.md.j2` with `context`. Pure function — no I/O beyond loading the template."""
    template = _template_env().get_template(_TEMPLATE_NAME)
    return template.render(context=context)


def script_path(vault_dir: Path, *, date: Date) -> Path:
    """Where `date`'s episode script lives — the same path every time (CLAUDE.md §3)."""
    return vault_dir / "podcast" / f"{date.isoformat()}.md"


def write_script(
    conn: sqlite3.Connection,
    built: BuiltScript,
    *,
    date: Date,
    presenter_a: str,
    presenter_b: str,
    vault_dir: Path,
) -> Path:
    """Render and write `date`'s episode script, overwriting any existing file.

    Idempotent by construction, same as `report.daily.write_digest`: same
    date, same `BuiltScript`, same path, `write_text` replaces the file's
    contents rather than appending (CLAUDE.md §3, NEVER rule 4).
    """
    context = build_script_context(
        conn, built, date=date, presenter_a=presenter_a, presenter_b=presenter_b
    )
    rendered = render_script(context)
    path = script_path(vault_dir, date=date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HEADING_RE = re.compile(r"^## (.+)$")
_TURN_RE = re.compile(r"^\*\*(.+?):\*\* (.+)$")
_SOURCE_RE = re.compile(r"^- \[(.*?)\]\((.*?)\) <!-- sf:item=(\d+) -->$")
_GAP_LINKED_RE = re.compile(r"^- \[(.*?)\]\((.*?)\) — .*? <!-- sf:item=(\d+) reason=(\w+) -->$")
_GAP_LINKLESS_RE = re.compile(r"^- item (\d+) — .*? <!-- sf:item=(\d+) reason=(\w+) -->$")


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError("podcast script file has no frontmatter block")
    frontmatter = yaml.safe_load(match.group(1))
    if not isinstance(frontmatter, dict):
        raise ValueError("podcast script frontmatter did not parse to a mapping")
    return frontmatter, text[match.end() :]


def _frontmatter_field(frontmatter: dict[str, object], key: str) -> object:
    try:
        return frontmatter[key]
    except KeyError as exc:
        raise ValueError(f"podcast script frontmatter is missing {key!r}") from exc


def _validated_reason(reason: str, *, item_id: str) -> Literal["bad_citation", "truncated"] | None:
    if reason not in _REASON_LABELS:
        # A marker this parser doesn't recognize — a hand-edited or
        # future-version file, not a bug in this pass. Skip rather than
        # trust an unvalidated reason into a `Literal`.
        logger.warning(
            "podcast coverage-gap marker has an unknown reason; skipping",
            extra={"item_id": item_id, "reason": reason},
        )
        return None
    return cast("Literal['bad_citation', 'truncated']", reason)


def parse_script(text: str) -> ScriptContext:
    """Recover a `ScriptContext` from a file `write_script` wrote — the
    read side of the round trip this module's docstring describes.

    Pure function over `text`; no I/O, no DB access. Anything outside the
    recognized line shapes (blank lines, the "no episode" prose, the
    `# Podcast — ...` title, a bare `**Sources:**` label with no citations
    following it) is silently skipped rather than rejected — this parser
    only needs to recover what `render_script` actually wrote, not
    validate arbitrary markdown.
    """
    frontmatter, body = _parse_frontmatter(text)
    raw_date = _frontmatter_field(frontmatter, "date")
    date = raw_date if isinstance(raw_date, Date) else Date.fromisoformat(str(raw_date))

    segments: list[ScriptSegment] = []
    coverage_gaps: list[CoverageGap] = []

    heading: str | None = None
    teased = False
    turns: list[ScriptTurn] = []
    sources: list[SourceLink] = []

    def flush() -> None:
        if heading is not None and heading != _COVERAGE_GAPS_LABEL:
            segments.append(
                ScriptSegment(
                    label=heading, turns=tuple(turns), sources=tuple(sources), teased=teased
                )
            )

    for line in body.splitlines():
        if (heading_match := _HEADING_RE.match(line)) is not None:
            flush()
            raw_heading = heading_match.group(1)
            if raw_heading.endswith(_TEASED_NOTE):
                heading = raw_heading[: -len(_TEASED_NOTE)]
                teased = True
            else:
                heading = raw_heading
                teased = False
            turns, sources = [], []
            continue

        if heading == _COVERAGE_GAPS_LABEL:
            if (gap_match := _GAP_LINKED_RE.match(line)) is not None:
                title, url, item_id, raw_reason = gap_match.groups()
                if (reason := _validated_reason(raw_reason, item_id=item_id)) is not None:
                    if is_safe_url(url):
                        coverage_gaps.append(
                            CoverageGap(
                                item_id=int(item_id),
                                title=_unescape_title(title),
                                url=url,
                                reason=reason,
                            )
                        )
                    else:
                        logger.warning(
                            "podcast coverage-gap url failed validation on parse; dropping link",
                            extra={"item_id": item_id},
                        )
                        coverage_gaps.append(
                            CoverageGap(
                                item_id=int(item_id),
                                title=None,
                                url=None,
                                reason=reason,
                            )
                        )
            elif (linkless_match := _GAP_LINKLESS_RE.match(line)) is not None:
                _, item_id, raw_reason = linkless_match.groups()
                if (reason := _validated_reason(raw_reason, item_id=item_id)) is not None:
                    coverage_gaps.append(
                        CoverageGap(item_id=int(item_id), title=None, url=None, reason=reason)
                    )
            continue

        if (turn_match := _TURN_RE.match(line)) is not None:
            speaker, turn_text = turn_match.groups()
            turns.append(ScriptTurn(speaker=speaker, text=turn_text))
            continue

        if (source_match := _SOURCE_RE.match(line)) is not None:
            title, url, item_id = source_match.groups()
            if is_safe_url(url):
                sources.append(
                    SourceLink(item_id=int(item_id), title=_unescape_title(title), url=url)
                )
            else:
                logger.warning(
                    "podcast source url failed validation on parse; dropping the citation",
                    extra={"item_id": item_id},
                )
            continue

    flush()

    return ScriptContext(
        date=date,
        script_version=str(_frontmatter_field(frontmatter, "script_version")),
        model=str(_frontmatter_field(frontmatter, "model")),
        item_count=int(str(_frontmatter_field(frontmatter, "item_count"))),
        segments=tuple(segments),
        coverage_gaps=tuple(coverage_gaps),
    )
