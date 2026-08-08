"""Weekly Intelligence Brief synthesis (DESIGN §13, Phase 1).

`llm.py`'s only weekly-brief entry point, and the seam every test fakes at
(NEVER rule 13). Like `synth/podcast.py` it takes no `client` parameter — the
Anthropic SDK lives behind `llm.py` and nowhere else (NEVER rule 1), so the
only way to reach a real API call from here is through a function this module
imports by name.

`build_brief` never raises once a call has been billed. Every "nothing usable"
outcome — a refusal, an unparseable response, a brief whose every block cited an
item that was never sent — comes back as a `BuiltBrief` carrying its tokens, so
the vault still gets a file and `runs` still gets the spend (NEVER rule 11).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from signalforge.config import InterestsConfig
from signalforge.db import DigestItem
from signalforge.llm import (
    WEEKLY_MAX_CLUSTERS,
    WEEKLY_MAX_LEADS,
    WEEKLY_MODEL,
    WeeklyBrief,
    WeeklyCluster,
    WeeklyLead,
    run_weekly_brief,
)
from signalforge.models import flatten_to_single_line

__all__ = [
    "WEEKLY_BRIEF_VERSION",
    "BuiltBrief",
    "build_brief",
    "build_weekly_stable_prefix",
]

logger = logging.getLogger(__name__)

WEEKLY_BRIEF_VERSION: Final = "weekly-v1"
"""Stamped into every brief's frontmatter, and the `rubric_version` analogue for
synthesis (NEVER rule 5): bump it on any change to the stable prefix or the
output schema, so an old file in the vault still says which prompt shape wrote
it. Without it, a brief written under a superseded prompt is indistinguishable
from one written today, and the four-Sunday acceptance gate loses its meaning
the first time the prompt changes mid-run."""


@dataclass(frozen=True, slots=True)
class BuiltBrief:
    """One brief as it leaves synthesis: cleaned, and carrying what it spent.

    `brief` is `None` for every "nothing usable" outcome — a refused call, an
    unparseable response, or a brief whose every block cited an item that was
    never sent. Those are not interchangeable with "the call never happened":
    the token counts travel on all of them (NEVER rule 11), because a malformed
    response is billed exactly like a good one. A path that returned a bare
    `None` here would silently drop real spend out of `runs`.

    `input_tokens`/`output_tokens` are deliberately **required, with no default**.
    A default of 0 is what lets a failure path construct a `BuiltBrief` without
    saying what it spent, which is how real money leaves `runs` unrecorded; making
    them positional forces every construction site — including the ones that only
    exist to report a failure — to state it, and mypy strict enforces that.

    `brief_version` defaults to `WEEKLY_BRIEF_VERSION` so the constant and the
    frontmatter cannot drift apart by omission; overriding it takes an explicit
    argument.

    `dropped_item_count` is how many items `WEEKLY_MAX_ITEMS` clamped away before
    the model saw them — distinct from `dropped_item_ids`, which is what the model
    then cited wrongly. It has to reach the footer: the brief renders a checkbox
    for every *selected* item, so if `weekly_top_n` were ever raised above the
    hard ceiling, the reader would be marking items the synthesis never read, with
    nothing on the page saying so.

    `dropped_item_ids` records citations the model invented — ids outside the
    set actually sent. They are rendered as a footer line rather than swallowed,
    because a fabricated citation is the single most important thing a synthesis
    step can tell you about itself (CLAUDE.md §5).
    """

    brief: WeeklyBrief | None
    model: str
    input_tokens: int
    output_tokens: int
    brief_version: str = WEEKLY_BRIEF_VERSION
    dropped_item_ids: tuple[int, ...] = ()
    dropped_item_count: int = 0


def build_weekly_stable_prefix(interests: InterestsConfig) -> str:
    """The system prefix: what a brief is, and what the reader cares about.

    Takes only `interests` — no date, window, run id, or item data can reach this
    signature, which is what makes byte-identity across two different weeks
    provable by construction rather than by care (NEVER rule 10). The window
    label and the items are the caller's job, in `run_weekly_brief`'s user turn.

    There is no `cache_control` breakpoint on this text, unlike the podcast's.
    One request a week means an ephemeral cache entry always expires unread, so a
    breakpoint would be a 1.25x surcharge bought for nothing — the scout's
    recorded reasoning at the identical cadence (DESIGN §8). The rule about
    volatile data still applies regardless: this text is what the operator's
    briefs are comparable across, and a date in it would make every week's prompt
    a different prompt.

    Kept deliberately short. It is priced into the worst-case ceiling at its real
    rendered size, and `PODCAST_MONTHLY_CEILING_USD`'s history records a 340-byte
    addition to its prefix eating a third of that ceiling's margin.
    """
    lines = [
        "You write a weekly intelligence brief for one engineer, from the items "
        "that cleared their relevance bar over the past seven days. They have "
        "already skimmed most of these in a daily digest, so your job is not to "
        "summarize them again — it is to say what the week meant.",
        "",
        "Produce two things. First, up to three leads: the things that actually "
        "mattered this week, each with a headline and a short paragraph on why it "
        "matters to this reader specifically. Fewer than three is correct on a "
        "thin week; never invent a third to fill the slot. Second, themes: groups "
        "of items that belong together, each with a title and a paragraph tying "
        "them into one story.",
        "",
        "Every lead and every theme must cite the item_id(s) it draws on, exactly "
        "as given in the user turn. Never cite an id you were not given, and "
        "never make a claim no cited item supports — the brief is read as a "
        "record, and an uncited sentence is indistinguishable from an invented "
        "one.",
        "",
        "Write plain prose in a single paragraph per block. No markdown, no "
        "bullet points, no headings, no line breaks inside a field: this text is "
        "rendered into a file whose lines carry structure.",
        "",
        "What this reader cares about, from interests.yaml — weight the brief accordingly:",
        f"Priority topics: {', '.join(interests.priority_topics) or 'none specified'}.",
        f"Interests: {', '.join(interests.interests) or 'none specified'}.",
        f"Stack: {', '.join(interests.stack) or 'none specified'}.",
        f"Learning goals: {', '.join(interests.learning_goals) or 'none specified'}.",
        f"Architecture philosophy: {interests.architecture_philosophy or 'none specified'}.",
    ]
    if interests.ignore.topics:
        lines.append(f"Topics to de-emphasize or skip: {', '.join(interests.ignore.topics)}.")
    return "\n".join(lines)


def _empty_brief(
    *,
    input_tokens: int,
    output_tokens: int,
    dropped_item_ids: tuple[int, ...] = (),
    dropped_item_count: int = 0,
) -> BuiltBrief:
    """The "nothing usable" outcome, carrying whatever was already spent.

    Every early return in `build_brief` funnels through here rather than a bare
    `None`, so a refusal, a schema failure, or a brief that cleaned down to
    nothing still reports the tokens `run_weekly_brief` already billed
    (NEVER rule 11) — the bug an architectural review caught in the podcast's
    equivalent, recorded in `BuiltScript.script`'s docstring.

    The token arguments are required rather than defaulted for the same reason
    they are required on `BuiltBrief` itself: a default of zero here would
    reintroduce, one level down, exactly the silent-spend-drop that making them
    mandatory was meant to prevent. Only the pre-call path legitimately passes
    zero, and it says so explicitly.

    `dropped_item_ids` matters most in exactly the case `brief` is `None`: a
    brief where *every* block cited a fabricated id is the one where the record
    of which ids were fabricated must not be the thing thrown away.
    """
    return BuiltBrief(
        brief=None,
        model=WEEKLY_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dropped_item_ids=dropped_item_ids,
        dropped_item_count=dropped_item_count,
    )


def _drop_unknown_blocks(
    brief: WeeklyBrief, known_ids: set[int]
) -> tuple[WeeklyBrief, tuple[int, ...]]:
    """Drop any lead or theme citing an id outside `known_ids` — NEVER rule 7,
    applied to ids rather than URLs.

    A block with even one unknown id among several is dropped whole rather than
    partially kept: its paragraph was written about all its cited items together,
    so stripping one citation would leave prose still discussing it, now uncited.

    `known_ids` must be `WeeklyBriefResult.sent_item_ids`, never the caller's own
    (possibly larger) list. An id `llm.py` clamped away before the request was
    built was never shown to the model, so a block citing it is precisely the
    confabulation this exists to catch — not a legitimate citation that missed
    the cut.
    """
    dropped: list[int] = []

    def keep(item_ids: Sequence[int], kind: str) -> bool:
        unknown = [item_id for item_id in item_ids if item_id not in known_ids]
        if not unknown:
            return True
        dropped.extend(unknown)
        logger.warning(
            "weekly brief block cited unknown item id(s); dropping the block",
            extra={"kind": kind, "item_ids": list(item_ids), "unknown_ids": unknown},
        )
        return False

    # Leads first, then clusters, spelled out as two loops. A single
    # `WeeklyBrief(leads=[...], clusters=[...])` with comprehensions produces the
    # same order only because Python evaluates keyword arguments left to right —
    # an implicit dependency for a tuple the vault footer renders verbatim.
    leads = [lead for lead in brief.leads if keep(lead.item_ids, "lead")]
    clusters = [cluster for cluster in brief.clusters if keep(cluster.item_ids, "cluster")]
    return WeeklyBrief(leads=leads, clusters=clusters), tuple(dropped)


def _flatten_and_finalize(brief: WeeklyBrief) -> WeeklyBrief | None:
    """Flatten every model-authored string, and drop whatever collapses to nothing.

    The drop is necessary rather than cosmetic: each field carries `min_length=1`
    on the model's *unflattened* output, so a value of `"   "` passes schema
    validation and then reduces to `""` — reconstructing the block with that text
    would violate the same constraint on the way out. Dropping the block here is
    what keeps that case a `None` return rather than an unhandled
    `ValidationError` at the end of a call that has already been billed.

    Flattening happens here *and* again in `report/weekly.py`. That is deliberate
    duplication, not an oversight: it is idempotent, and NEVER rule 17 is a
    boundary rule, so the boundary is guarded on both sides of it.

    Returns `None` when nothing survives.
    """
    leads: list[WeeklyLead] = []
    for lead in brief.leads:
        headline = flatten_to_single_line(lead.headline)
        why = flatten_to_single_line(lead.why_it_matters)
        if not headline or not why:
            logger.warning(
                "weekly lead had no text left after flattening",
                extra={"item_ids": lead.item_ids},
            )
            continue
        leads.append(WeeklyLead(headline=headline, why_it_matters=why, item_ids=lead.item_ids))

    clusters: list[WeeklyCluster] = []
    for cluster in brief.clusters:
        title = flatten_to_single_line(cluster.title)
        narrative = flatten_to_single_line(cluster.narrative)
        if not title or not narrative:
            logger.warning(
                "weekly theme had no text left after flattening",
                extra={"item_ids": cluster.item_ids},
            )
            continue
        clusters.append(WeeklyCluster(title=title, narrative=narrative, item_ids=cluster.item_ids))

    if not leads and not clusters:
        return None
    return WeeklyBrief(
        leads=_clamp(leads, WEEKLY_MAX_LEADS, "leads"),
        clusters=_clamp(clusters, WEEKLY_MAX_CLUSTERS, "themes"),
    )


def _clamp[BlockT: (WeeklyLead, WeeklyCluster)](
    blocks: list[BlockT], limit: int, kind: str
) -> list[BlockT]:
    """Keep the first `limit` blocks, in the model's own order.

    The count bounds cannot live in the request schema — the API rejects array
    `minItems`/`maxItems` — so they are enforced here. Clamping rather than
    rejecting is deliberate: discarding a whole billed response over one extra
    block loses a good brief and invites the next invocation to pay again.
    """
    if len(blocks) <= limit:
        return blocks
    logger.warning(
        "weekly brief returned more blocks than the cap; keeping the first",
        extra={"kind": kind, "returned": len(blocks), "limit": limit},
    )
    return blocks[:limit]


def build_brief(
    stable_prefix: str,
    *,
    window_label: str,
    items: Sequence[DigestItem],
) -> BuiltBrief:
    """Write, validate and clean one weekly brief.

    Always returns a `BuiltBrief`; never raises once a response has been billed.

    `items` is the deterministic selection from `report/weekly.py` —
    `WeeklySelection.items`, never its `near_misses`, which exist to prompt for a
    `missed` mark and are not material the brief reasons over.

    Only four fields per item reach `llm.py`: id, title, summary, and the stored
    triage reasoning. Passing `DigestItem` in and unpacking here rather than
    accepting tuples keeps that projection in one place, where the fact that
    `Item.content` is deliberately absent is visible.

    Cleaning runs in a fixed order — unknown citations dropped first, then
    flattening — because a block that is about to be dropped for a fabricated
    citation should not have its text flattened, logged, and reasoned about on
    the way to being discarded.
    """
    item_tuples = [
        (scored.item.id, scored.item.title, scored.item.summary, scored.reasoning)
        for scored in items
        if scored.item.id is not None
    ]
    unsaved = len(items) - len(item_tuples)
    if unsaved:
        # Not expected: the selection comes from the DB, which never returns a
        # row with no id. Logged anyway rather than silently shrinking the
        # payload — the same invisible-clamp shape `dropped_item_count` exists
        # to prevent one layer up.
        logger.warning("weekly brief dropped unsaved items with no id", extra={"count": unsaved})
    if not item_tuples:
        # The one path that legitimately spent nothing: no call was made.
        return _empty_brief(input_tokens=0, output_tokens=0)

    result = run_weekly_brief(stable_prefix, window_label=window_label, items=item_tuples)
    input_tokens = result.input_tokens
    output_tokens = result.output_tokens
    clamped = result.dropped_item_count

    if result.brief is None:
        logger.warning("weekly brief call produced nothing usable", extra={"error": result.error})
        return _empty_brief(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_count=clamped,
        )

    cleaned, dropped_item_ids = _drop_unknown_blocks(result.brief, set(result.sent_item_ids))
    finalized = _flatten_and_finalize(cleaned)
    if finalized is None:
        logger.warning("weekly brief had nothing left after cleaning")
        return _empty_brief(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_ids=dropped_item_ids,
            dropped_item_count=clamped,
        )

    logger.info(
        "weekly brief built",
        extra={
            "lead_count": len(finalized.leads),
            "cluster_count": len(finalized.clusters),
            "dropped_citation_count": len(dropped_item_ids),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )
    return BuiltBrief(
        brief=finalized,
        model=WEEKLY_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dropped_item_ids=dropped_item_ids,
        dropped_item_count=clamped,
    )
