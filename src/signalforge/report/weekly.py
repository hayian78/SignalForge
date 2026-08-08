"""Weekly Intelligence Brief (DESIGN §13, Phase 1) — selection and rendering.

Phase 1's acceptance gate is four consecutive Sunday briefs with ≥ 80% of their
items marked `useful` or better, so what this module decides — *which* items the
brief carries — is the thing being measured. The selection half is therefore
plain, deterministic Python: no DB handle, no LLM, no clock read (CLAUDE.md §2,
§8). The caller supplies the already-ranked rows and the operator's marks; this
module only filters them, then renders them through jinja.

The whole module is free of a `sqlite3` handle, including the renderer: a
citation resolves against the selection rather than the database, which is both
stronger (an id outside the selection is a fabrication, not a stale row) and
what makes `build_brief_context` a provably pure function.

### The window is the seven days *before* the target Sunday

`db.get_digest_items` buckets on `scores.scored_at`, the score pass runs in the
evening, and the brief runs Sunday morning (DESIGN §14) — so on Sunday at 07:00
the target Sunday's own bucket is still empty, twelve hours from being filled.
A window of `[target-6, target]` would spend a slot on a day that cannot have
items yet *and* leave the previous Sunday in no window at all: not in last
week's brief, not in this one, and next week's window starts after it. One
seventh of the corpus, stranded permanently. `[target-7, target-1]` instead
tiles exactly, and every day in it has been scored.

### Why the brief is stricter than the digest

The `weekly_min_*` gates have existed in `interests.yaml` since Phase 0, but
only as *prompt text*: `score/rubrics.py` tells Haiku to keep an item when it
"plausibly clears" them. Nothing applied the arithmetic. Applying it here is
the deterministic half of that split (NEVER rule 3), and it bites — an item
scored 3/3/2 sums to 8, is kept, appears in the daily digest, and does not
clear `weekly_min_total: 10`. That is not a bug to paper over; it is the
difference between "worth a look today" and "worth the week's attention", and
the footer says so rather than leaving the reader to notice items missing.

Gate-failers are not discarded. They become the near-miss list, which exists to
make `signalforge mark <id> missed` cheap to give — `missed` is the highest-value
verdict and the one the operator has no other prompt for (DESIGN §11).

### What this module deliberately does not do

It does not re-rank by the operator's marks. `report/podcast.py::order_by_verdict`
lifts `exceptional` and `useful` items to the front, which is right for a
five-slot episode and wrong here: `feedback` rows carry no provenance, so a
`useful` ticked on Tuesday's digest is indistinguishable from one ticked on the
brief. Promoting already-marked items would compose the brief preferentially
from items that already satisfy its own acceptance metric — the gate would be
measuring its own input. Only the `noise` drop carries over, because an explicit
"not this one" is a filter, not a re-rank.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta, tzinfo
from pathlib import Path
from typing import Final

import jinja2

from signalforge.config import Thresholds
from signalforge.db import DigestItem
from signalforge.feedback import CHECKBOX_VERDICTS, checkbox_marker, highest_rung
from signalforge.models import escape_markdown_link_text, flatten_to_single_line
from signalforge.report.daily import select_digest_items, utc_range_window
from signalforge.synth.weekly import BuiltBrief

__all__ = [
    "WEEKLY_WINDOW_DAYS",
    "BriefCluster",
    "BriefContext",
    "BriefItemLine",
    "BriefLead",
    "NearMissLine",
    "SourceLink",
    "WeeklySelection",
    "brief_path",
    "build_brief_context",
    "drop_noise_marked",
    "passes_weekly_gate",
    "render_brief",
    "select_weekly_items",
    "weekly_window",
    "write_brief",
]

logger = logging.getLogger(__name__)

_TEMPLATE_NAME = "weekly.md.j2"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

WEEKLY_WINDOW_DAYS: Final = 7
"""How many days a brief covers. A definition, not a tuning knob (NEVER rule 6):
"weekly" means seven days, and a config value here would let one week overlap or
skip another, breaking the tiling property `weekly_window` exists to guarantee."""


def weekly_window(target_sunday: Date, tz: tzinfo) -> tuple[str, str]:
    """The UTC ISO `[start, end)` covering the seven days before `target_sunday`.

    `target_sunday` is the date the brief is *published on*; the window ends the
    day before it. See the module docstring for why the publication day is
    excluded rather than included.

    Delegates the boundary arithmetic to `report.daily.utc_range_window`, so the
    brief and the digest share one DST-correct rule: seven real calendar days in
    `tz`, which is 167 or 169 hours across a transition, never a fixed 168.
    """
    first = target_sunday - timedelta(days=WEEKLY_WINDOW_DAYS)
    last = target_sunday - timedelta(days=1)
    return utc_range_window(first, last, tz)


def passes_weekly_gate(
    scored: DigestItem,
    *,
    min_signal: int,
    min_relevance: int,
    min_total: int,
) -> bool:
    """Whether `scored` clears `interests.yaml`'s weekly inclusion bar (DESIGN §9).

    All three thresholds come from `interests.yaml` — this function holds no
    numbers of its own (NEVER rule 6). It is the deterministic counterpart to the
    same bar quoted into the triage prompt by `score/rubrics.py`; Haiku decides
    "plausibly clears", this decides *clears*.

    A missing dimension counts as 0 and therefore fails: an unscored dimension
    has not cleared anything, and inventing a passing value for it would let a
    partial score row into the report the whole gate exists to keep tight. That
    matches the ranking SQL, which already coalesces nulls to 0.
    """
    signal = scored.signal or 0
    relevance = scored.relevance or 0
    novelty = scored.novelty or 0
    return (
        signal >= min_signal
        and relevance >= min_relevance
        and (signal + relevance + novelty) >= min_total
    )


def drop_noise_marked(
    scored_items: Sequence[DigestItem],
    verdicts: Mapping[int, Sequence[str]],
) -> list[DigestItem]:
    """`scored_items` minus everything the operator marked `noise`. Order untouched.

    A filter, never a re-rank (DESIGN §13's "filter, never reorder"): the result
    is a sub-sequence of the input, so the brief stays a pure function of
    `(window, db state, config)` and re-renders byte-identically.

    `verdicts` maps item id to that item's stored verdicts, unreduced
    (`db.feedback_verdicts_for_items`); each item collapses to its strongest rung
    via `feedback.highest_rung`, so an item marked both `noise` and `useful` — a
    mis-click, or a mind changed — resolves to `useful` and stays. An item with
    no id cannot be looked up and is treated as unmarked rather than dropped;
    citability is what decides whether an item can be carried, and the caller has
    already applied that filter.

    Pure: no DB, no config, no I/O.
    """
    kept: list[DigestItem] = []
    for scored in scored_items:
        item_id = scored.item.id
        marks = verdicts.get(item_id, ()) if item_id is not None else ()
        if highest_rung(marks) == "noise":
            continue
        kept.append(scored)
    return kept


@dataclass(frozen=True, slots=True)
class WeeklySelection:
    """What a week's rows resolve to: the brief's body, and what it left behind.

    `items` and `near_misses` are disjoint by construction — the gate partitions
    the same list, so an item cannot appear in both. That matters because the
    near-miss list prompts for `missed`, a verdict that only makes sense for an
    item the brief did *not* show.

    The two counts exist so the footer reconciles: every citable, non-`noise`
    item in the window is either rendered, counted in `gated_out_count`, or
    counted in `hidden_count`. Nothing is silently dropped (CLAUDE.md §7).
    """

    items: tuple[DigestItem, ...]
    near_misses: tuple[DigestItem, ...]
    gated_out_count: int
    hidden_count: int


def select_weekly_items(
    scored_items: Sequence[DigestItem],
    verdicts: Mapping[int, Sequence[str]],
    *,
    thresholds: Thresholds,
) -> WeeklySelection:
    """Partition a week's kept items into the brief's body and its near-misses.

    `scored_items` is `db.get_digest_items` over `weekly_window` — already ranked
    by score sum, and this function only ever filters that order.

    The step order is load-bearing:

    1. **Uncitable dropped first.** An item with no URL must not take a slot only
       to be dropped at render, which would silently shorten the brief (NEVER
       rule 7 — the same reason `build_digest_context` settles citability before
       selection).
    2. **`noise` dropped next**, so an item the operator rejected cannot occupy a
       slot *or* show up in the near-miss list asking to be marked `missed`.
    3. **The gate partitions** what remains into body candidates and near-misses.
    4. **Crowding caps and top-N truncate last**, so they cut the ranked order
       rather than deciding it.

    `weekly_top_n` bounds an Opus call, which is why it is a separate knob from
    `daily_max_items` — a keep-rate improvement must not silently become a cost
    multiplier. `None` means the brief is off and this returns an empty body,
    the same degradation `podcast_top_n` already has.

    The per-source and per-github-repo caps are the digest's own `daily_*` knobs,
    reused deliberately rather than by oversight: over seven days they are far
    tighter in relative terms than over one, and tight anti-crowding is exactly
    what a brief of the week wants. One source dominating a week is one story,
    not N.
    """
    citable: list[DigestItem] = []
    for scored in scored_items:
        if scored.item.url:
            citable.append(scored)
        else:
            # Should be unreachable — `Item.url` is required — but a corrupted
            # row must leave a trace rather than vanish from both the brief and
            # the log (CLAUDE.md §7). `daily.py::_to_line` logs the same drop.
            logger.warning(
                "dropping an item with no URL from the weekly brief",
                extra={"item_id": scored.item.id},
            )
    marked = drop_noise_marked(citable, verdicts)

    eligible: list[DigestItem] = []
    gated_out: list[DigestItem] = []
    for scored in marked:
        target = eligible if _clears(scored, thresholds) else gated_out
        target.append(scored)

    if thresholds.weekly_top_n is None:
        # The brief is off. Reported by the CLI, which has the run context;
        # saying it here would be a log line with no `run_id` attached.
        selected: list[DigestItem] = []
    else:
        selected = select_digest_items(
            eligible,
            max_items=thresholds.weekly_top_n,
            max_per_source=thresholds.daily_max_per_source,
            max_per_github_repo=thresholds.daily_max_per_github_repo,
        )

    near_miss_n = thresholds.weekly_near_miss_n
    near_misses = gated_out[:near_miss_n] if near_miss_n is not None else []

    return WeeklySelection(
        items=tuple(selected),
        near_misses=tuple(near_misses),
        gated_out_count=len(gated_out),
        hidden_count=len(eligible) - len(selected),
    )


def _clears(scored: DigestItem, thresholds: Thresholds) -> bool:
    return passes_weekly_gate(
        scored,
        min_signal=thresholds.weekly_min_signal,
        min_relevance=thresholds.weekly_min_relevance,
        min_total=thresholds.weekly_min_total,
    )


# --------------------------------------------------------------------------- #
# rendering — the vault file
#
# Citations resolve against the selection, not the database. The brief only
# ever sends the model items the selection chose, so an id outside it is a
# fabrication rather than a stale row, and checking against the selection
# catches that while keeping this whole module free of a DB handle.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SourceLink:
    """One resolved citation behind a lead or a cluster."""

    item_id: int
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class BriefLead:
    """One of "the 3 things that mattered", with its citations resolved."""

    headline: str
    why_it_matters: str
    sources: tuple[SourceLink, ...]


@dataclass(frozen=True, slots=True)
class BriefCluster:
    """One themed group of the week, with its citations resolved."""

    title: str
    narrative: str
    sources: tuple[SourceLink, ...]


@dataclass(frozen=True, slots=True)
class BriefItemLine:
    """One item in the brief's body — the unit the acceptance gate counts.

    Rendered whether or not the model cited it. Attaching the mark checkboxes
    to the model's citations instead would let the model's behaviour decide the
    gate's denominator; attaching them here keeps `item_count` a pure function
    of `(target_sunday, tz, db state, config)`.
    """

    id: int
    title: str
    url: str
    signal: int | None
    relevance: int | None
    novelty: int | None


@dataclass(frozen=True, slots=True)
class NearMissLine:
    """One item the gate turned away, offered for `mark <id> missed`.

    Deliberately carries no checkbox: `missed` sits off the `noise < useful <
    exceptional` ladder and is CLI-only, because a rendered item was by
    definition surfaced (DESIGN §11). The id is rendered so the command is one
    copy-paste.
    """

    id: int
    title: str
    url: str
    signal: int | None
    relevance: int | None
    novelty: int | None


@dataclass(frozen=True, slots=True)
class BriefContext:
    """Everything `weekly.md.j2` needs for one brief. Assembled once, rendered once.

    `leads` and `clusters` are empty whenever synthesis produced nothing usable.
    The template still renders the body, the checkboxes and the near-misses in
    that case — the vault file is useful without the narrative, which is what
    makes a failed or skipped Opus call a degraded brief rather than a missing
    one (DESIGN §13.2's "the vault is written first, and always").
    """

    date: Date
    window_start: Date
    window_end: Date
    brief_version: str
    model: str
    leads: tuple[BriefLead, ...]
    clusters: tuple[BriefCluster, ...]
    items: tuple[BriefItemLine, ...]
    near_misses: tuple[NearMissLine, ...] = ()
    gated_out_count: int = 0
    hidden_count: int = 0
    dropped_item_ids: tuple[int, ...] = ()
    unread_item_count: int = 0
    """Items the brief renders — and offers for marking — that the synthesis call
    never saw, because `llm.WEEKLY_MAX_ITEMS` clamped them away. Zero unless
    `weekly_top_n` is set above that ceiling. Rendered rather than logged: the
    reader would otherwise be marking items nothing had read."""

    @property
    def item_count(self) -> int:
        """The acceptance gate's denominator: items the brief actually offered
        for marking, not blocks the model wrote."""
        return len(self.items)


def _resolve(item_ids: Sequence[int], *, by_id: Mapping[int, DigestItem]) -> tuple[SourceLink, ...]:
    links: list[SourceLink] = []
    # `dict.fromkeys` rather than a set: a model citing the same id twice should
    # render one source line, and the surviving order must stay the model's.
    for item_id in dict.fromkeys(item_ids):
        scored = by_id.get(item_id)
        if scored is None:
            # Defence in depth, and expected to be dead code: `synth/weekly.py`
            # validates against the same set before this runs, so reaching here
            # means the two checks disagree. Logged rather than surfaced in the
            # file, because the footer's fabricated-citation line is fed by the
            # build-time check that should have caught it first.
            logger.warning(
                "weekly brief cited an item that is not in the brief's selection",
                extra={"item_id": item_id},
            )
            continue
        links.append(
            SourceLink(
                item_id=item_id,
                title=flatten_to_single_line(scored.item.title),
                url=scored.item.url,
            )
        )
    return tuple(links)


def _clean(text: str) -> str:
    """Model text, flattened to one line at the storage boundary (NEVER rule 17).

    A newline inside a narrative is not cosmetic here: `feedback.MARK_RE` and
    `curate.approvals` both harvest a decision from any line matching their
    checkbox pattern, so a model that writes one into its prose could forge a
    mark on the very file that carries the acceptance gate.
    """
    return flatten_to_single_line(text)


def build_brief_context(
    built: BuiltBrief,
    selection: WeeklySelection,
    *,
    target_sunday: Date,
) -> BriefContext:
    """Assemble everything the template needs. Pure — no DB, no clock, no writes.

    A lead or cluster whose every citation fails to resolve is dropped whole
    (NEVER rule 7 applied at render time, not only at build time), and so is one
    left with no text after flattening — a model string of `"   "` satisfies
    `min_length=1` and reduces to nothing, which would otherwise render an empty
    heading over real citations.

    The body, the checkboxes and the near-misses come from `selection` and never
    from `built`, so they are identical whether synthesis succeeded, failed, or
    was skipped entirely.
    """
    by_id = {scored.item.id: scored for scored in selection.items if scored.item.id is not None}

    leads: list[BriefLead] = []
    clusters: list[BriefCluster] = []
    if built.brief is not None:
        for lead in built.brief.leads:
            headline = _clean(lead.headline)
            why = _clean(lead.why_it_matters)
            sources = _resolve(lead.item_ids, by_id=by_id)
            if not headline or not why or not sources:
                logger.warning(
                    "dropping a weekly lead with no surviving text or citation",
                    extra={"item_ids": lead.item_ids},
                )
                continue
            leads.append(BriefLead(headline=headline, why_it_matters=why, sources=sources))

        for cluster in built.brief.clusters:
            title = _clean(cluster.title)
            narrative = _clean(cluster.narrative)
            sources = _resolve(cluster.item_ids, by_id=by_id)
            if not title or not narrative or not sources:
                logger.warning(
                    "dropping a weekly cluster with no surviving text or citation",
                    extra={"item_ids": cluster.item_ids},
                )
                continue
            clusters.append(BriefCluster(title=title, narrative=narrative, sources=sources))

    return BriefContext(
        date=target_sunday,
        window_start=target_sunday - timedelta(days=WEEKLY_WINDOW_DAYS),
        window_end=target_sunday - timedelta(days=1),
        brief_version=built.brief_version,
        model=built.model,
        leads=tuple(leads),
        clusters=tuple(clusters),
        items=tuple(_item_line(scored) for scored in selection.items),
        near_misses=tuple(_near_miss_line(scored) for scored in selection.near_misses),
        gated_out_count=selection.gated_out_count,
        hidden_count=selection.hidden_count,
        # De-duplicated here, not in `synth/`, which keeps the faithful raw
        # record: two blocks citing the same fabricated id is one fabrication,
        # and this footer line is the operator's only confabulation signal.
        dropped_item_ids=tuple(dict.fromkeys(built.dropped_item_ids)),
        unread_item_count=built.dropped_item_count,
    )


def _item_line(scored: DigestItem) -> BriefItemLine:
    assert scored.item.id is not None, "selection drops rows without an id before this point"
    return BriefItemLine(
        id=scored.item.id,
        title=flatten_to_single_line(scored.item.title),
        url=scored.item.url,
        signal=scored.signal,
        relevance=scored.relevance,
        novelty=scored.novelty,
    )


def _near_miss_line(scored: DigestItem) -> NearMissLine:
    assert scored.item.id is not None, "selection drops rows without an id before this point"
    return NearMissLine(
        id=scored.item.id,
        title=flatten_to_single_line(scored.item.title),
        url=scored.item.url,
        signal=scored.signal,
        relevance=scored.relevance,
        novelty=scored.novelty,
    )


def _template_env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    # One source of truth between the template and `feedback.parse_marks`, the
    # same discipline `report/daily.py` uses.
    env.globals["checkbox_marker"] = checkbox_marker
    env.globals["checkbox_verdicts"] = CHECKBOX_VERDICTS
    env.globals["escape_title"] = escape_markdown_link_text
    return env


def render_brief(context: BriefContext) -> str:
    """Fill `weekly.md.j2` with `context`. Pure — no I/O beyond loading the template."""
    template = _template_env().get_template(_TEMPLATE_NAME)
    return template.render(context=context)


def brief_path(vault_dir: Path, *, target_sunday: Date) -> Path:
    """Where a brief lives — the same path every time (CLAUDE.md §3)."""
    return vault_dir / "weekly" / f"{target_sunday.isoformat()}.md"


def write_brief(
    built: BuiltBrief,
    selection: WeeklySelection,
    *,
    target_sunday: Date,
    vault_dir: Path,
) -> Path:
    """Render and write the brief, overwriting any existing file for that date.

    Idempotent by construction, same as `report.daily.write_digest`: same date,
    same inputs, same path, and `write_text` replaces rather than appends
    (CLAUDE.md §3, NEVER rule 4).
    """
    context = build_brief_context(built, selection, target_sunday=target_sunday)
    rendered = render_brief(context)
    path = brief_path(vault_dir, target_sunday=target_sunday)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path
