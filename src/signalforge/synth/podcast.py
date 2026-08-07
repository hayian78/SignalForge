"""Podcast script generation — the daily two-presenter dialogue (DESIGN §13.3).

Calls `llm.py` only (CLAUDE.md §2, NEVER rule 1) and makes no HTTP call of
its own — this module's whole job is judgment over items an earlier stage
already fetched (`ingest/fullcontent.py`) and scored. It never touches
`db.py`: the caller (`cli.py`, Stage 6) already resolved the top-N item
slice before calling `build_script`, and `build_script` needs nothing more
than that slice — validating a cited `item_id` only requires knowing which
ids `llm.run_podcast_script` actually sent to the model, not looking
anything up.

No client seam here, deliberately: unlike `curate/scout.py`, this module
never imports `anthropic` (NEVER rule 1) — `llm.py` is faked at its
boundary in tests, so `build_script` calls `run_podcast_script` with no
`client` to pass through.

`PODCAST_SCRIPT_VERSION` is the `rubric_version` analogue for scoring
(CLAUDE.md §3, NEVER rule 5): bump it whenever `build_podcast_stable_prefix`
changes, and it will land in the vault frontmatter (`report/podcast.py`,
Stage 4 — not built yet) so a rendered episode always says which prompt
shape wrote it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from signalforge.config import InterestsConfig
from signalforge.llm import (
    PODCAST_MAX_SCRIPT_CHARS,
    PODCAST_MODEL,
    LlmError,
    PodcastScript,
    PodcastSegment,
    PodcastTurn,
    run_podcast_script,
)
from signalforge.models import Item, flatten_to_single_line

__all__ = [
    "PODCAST_SCRIPT_VERSION",
    "BuiltScript",
    "build_podcast_stable_prefix",
    "build_script",
]

logger = logging.getLogger(__name__)

PODCAST_SCRIPT_VERSION: Final = "podcast-v2"
"""Bumped on any change to `build_podcast_stable_prefix` or the schema it
targets — CLAUDE.md §3's NEVER rule 5, applied to a script instead of a
score. A rendered episode's frontmatter records this, so re-reading an old
script always says which prompt shape produced it.

v2: constrained the intro/outro to a plain greeting and sign-off, with no
preview or recap of specific stories. v1 let the model preview upcoming
segments in the intro — a real gap in NEVER rule 7's citation discipline:
`intro_turns`/`outro_turns` carry no `item_ids` and are never validated, so
a segment dropped after writing (a bad citation, or a length cut) could
leave the intro describing content that never aired — a synthesized claim
with no backing item, structurally indistinguishable from the confabulation
citations exist to catch. No episode had ever been generated under v1 (this
stage shipped both together,
before any Stage 6 caller existed to run one for real) — the version still
bumps on principle, because the fix is entirely "what the model was told to
write," the exact thing this constant exists to track."""


@dataclass(frozen=True, slots=True)
class BuiltScript:
    """What one `build_script` call produced, and what it spent — always
    returned, never a bare `None`.

    `script` is `None` for every "nothing usable" outcome (no items, a
    refused call, a script that cleaned down to nothing) — but
    `input_tokens`/`output_tokens` are populated whenever a call actually
    happened, the same "spend travels with the output" discipline
    `curate.scout.ScoutOutcome` and `llm.PodcastScriptResult` both use.
    Returning a bare `None` on those paths would silently report a paid
    Opus call as free (NEVER rule 11) — exactly what an architectural
    review of this stage caught."""

    script: PodcastScript | None
    script_version: str
    model: str
    dropped_item_ids: tuple[int, ...] = ()
    """The *unknown* ids that caused a segment to be dropped (§7) — never a
    real id, since a segment is dropped whole only for citing at least one
    id outside `sent_item_ids`. This does not include a legitimate id that
    happened to share a dropped segment with a fabricated one — that item
    simply lost its coverage in the episode, silently, which is a real gap
    a future revision may want to surface separately. Not the same set as
    `dropped_item_count`, which counts items clamped away before any
    request was ever sent."""
    dropped_item_count: int = 0
    """Items the caller passed beyond `llm.PODCAST_MAX_ITEMS`, clamped
    before the model ever saw them — mirrors `PodcastScriptResult.dropped_item_count`.
    Not summed across the "shorter" retry: the clamp is a property of the
    input item list, not a separate event each call, and both calls see the
    same items."""
    truncated: bool = False
    """True if the script still exceeded `PODCAST_MAX_SCRIPT_CHARS` after the
    one "shorter" retry, and segments were shrunk to teasers, or dropped
    outright, from the end to fit (`_truncate_to_fit`) — most commonly the
    former: shrinking is tried first and usually enough on its own."""
    teasered_item_ids: tuple[int, ...] = ()
    """Ids whose segment survived truncation but was shrunk to
    `_TEASER_TURN_COUNT` turns — partial airtime, distinct from
    `dropped_item_ids` (which is citation drops only) and from
    `truncation_dropped_item_ids` (zero airtime). Only ever non-empty when
    `truncated` is True."""
    truncation_dropped_item_ids: tuple[int, ...] = ()
    """Ids whose segment was removed outright by `_truncate_to_fit`'s phase
    2 — zero airtime, same listener-facing outcome as `dropped_item_ids`,
    but for a length reason rather than a citation-trust one. Kept as a
    separate field, not merged into `dropped_item_ids`, so `report/podcast.py`
    (Stage 4) can render "the show ran long" and "the model cited something
    it wasn't given" as the two distinct facts they are — collapsing them
    would let a real confabulation read as routine cost trimming."""
    input_tokens: int = 0
    output_tokens: int = 0


def build_podcast_stable_prefix(interests: InterestsConfig) -> str:
    """The cached system prefix: show-format brief + interests/taxonomy.

    Takes only `interests` — no date, run id, or item data can reach this
    function's signature, which is what makes byte-identity across two
    calls with different dates provable by construction rather than by
    care (NEVER rule 10; see the Stage 3 gate test). Items, the date, and
    presenter names are the caller's job (`run_podcast_script`'s user turn),
    strictly after the `cache_control` breakpoint this text becomes.
    """
    lines = [
        "You are the writer for a daily two-presenter news podcast about AI "
        "engineering, industry strategy, and frontier research — an audio "
        "companion to a written digest the operator already reads. Two "
        'presenters, referred to only as speaker "A" and speaker "B" in '
        "your structured output, discuss the day's stories conversationally: "
        "back-and-forth, reacting to each other, asking follow-up questions "
        "— never a monologue read-out and never a verbatim reading of the "
        "source material.",
        "",
        "Structure: a short intro, one segment per story or cluster of "
        "related stories, and a short outro. Every segment must cite the "
        "item_id(s) it discusses. Never invent an item_id, and never "
        "discuss an item whose id was not given to you in the user turn. "
        "The intro and outro are a plain greeting and sign-off ONLY — "
        '"good morning, let\'s get into it" / "that\'s the show, see you '
        'tomorrow" — never a preview or recap of specific stories, topics, '
        "or names from today's items. A segment can be dropped after you "
        "write it (a citation error, or the show running long), and an "
        "intro or outro that named it would then be describing content "
        "that never aired.",
        "",
        "Style: conversational spoken language, not written prose read "
        "aloud. Contractions, brief reactions, and the occasional "
        "interruption are good. No headers, no bullet points, no markdown — "
        "this is spoken dialogue that a text-to-speech system will read "
        "verbatim, so every word must be one you'd want spoken aloud.",
        "",
        "What the listener cares about, from interests.yaml — weight the "
        "show's attention accordingly:",
        f"Priority topics: {', '.join(interests.priority_topics) or 'none specified'}.",
        f"Interests: {', '.join(interests.interests) or 'none specified'}.",
        f"Stack: {', '.join(interests.stack) or 'none specified'}.",
        f"Learning goals: {', '.join(interests.learning_goals) or 'none specified'}.",
        f"Architecture philosophy: {interests.architecture_philosophy or 'none specified'}.",
    ]
    if interests.ignore.topics:
        lines.append(f"Topics to de-emphasize or skip: {', '.join(interests.ignore.topics)}.")
    return "\n".join(lines)


def _all_turns(script: PodcastScript) -> list[PodcastTurn]:
    return [
        *script.intro_turns,
        *(turn for segment in script.segments for turn in segment.turns),
        *script.outro_turns,
    ]


def _total_chars(script: PodcastScript) -> int:
    """Spoken-text length — the same measurement `deliver/tts.py` (Stage 5)
    will bill and refuse on, so this is the number `PODCAST_MAX_SCRIPT_CHARS` bounds."""
    return sum(len(turn.text) for turn in _all_turns(script))


def _drop_unknown_segments(
    script: PodcastScript, known_ids: set[int]
) -> tuple[PodcastScript, tuple[int, ...]]:
    """Drop any segment citing an id outside `known_ids` — NEVER rule 7's
    citation discipline, applied to ids instead of URLs. A segment with even
    one unknown id among several is dropped whole rather than partially
    kept: a segment's dialogue is written to flow across all its cited
    items together, so silently stripping one id would leave dialogue that
    still talks about it.

    `known_ids` must be `PodcastScriptResult.sent_item_ids`, never the
    caller's own (possibly larger) input list: an id `llm.py` clamped away
    before the request was built was never shown to the model, so a segment
    citing it is exactly the confabulation this check exists to catch, not
    a legitimate citation that merely missed the cut.
    """
    kept: list[PodcastSegment] = []
    dropped_ids: list[int] = []
    for segment in script.segments:
        unknown = [item_id for item_id in segment.item_ids if item_id not in known_ids]
        if unknown:
            dropped_ids.extend(unknown)
            logger.warning(
                "podcast segment cited unknown item id(s); dropping segment",
                extra={"item_ids": segment.item_ids, "unknown_ids": unknown},
            )
            continue
        kept.append(segment)
    cleaned = PodcastScript(
        intro_turns=script.intro_turns, segments=kept, outro_turns=script.outro_turns
    )
    return cleaned, tuple(dropped_ids)


def _flatten_and_finalize(script: PodcastScript) -> PodcastScript | None:
    """Flatten every turn's text (CLAUDE.md §5, NEVER rule 17) — the model
    wrote this text and it is about to render into a vault file, the exact
    boundary that rule exists to guard — and drop any turn a whitespace- or
    control-character-only original collapses to nothing.

    That drop is necessary, not cosmetic: `PodcastTurn.text` requires
    `min_length=1` on the model's *unflattened* output, so a turn of `"   "`
    or a lone control character passes schema validation and then
    `flatten_to_single_line` reduces it to `""` — reconstructing a
    `PodcastTurn` with that text would violate the same constraint a second
    time, on the way out rather than the way in. Dropping it here, and
    dropping a segment left with zero turns, is what keeps that case a
    `None` return instead of an unhandled `ValidationError`.

    Returns `None` if flattening empties the intro, the outro, or every
    segment — the same "nothing usable" outcome `build_script` already
    treats identically to a model response with no segments at all.
    """

    def flattened(turns: Sequence[PodcastTurn]) -> list[PodcastTurn]:
        kept: list[PodcastTurn] = []
        for turn in turns:
            text = flatten_to_single_line(turn.text)
            if text:
                kept.append(PodcastTurn(speaker=turn.speaker, text=text))
        return kept

    intro = flattened(script.intro_turns)
    outro = flattened(script.outro_turns)
    segments: list[PodcastSegment] = []
    for segment in script.segments:
        turns = flattened(segment.turns)
        if not turns:
            logger.warning(
                "podcast segment had no turns left after flattening",
                extra={"item_ids": segment.item_ids},
            )
            continue
        segments.append(PodcastSegment(item_ids=segment.item_ids, turns=turns))

    if not intro or not outro or not segments:
        return None
    return PodcastScript(intro_turns=intro, segments=segments, outro_turns=outro)


_TEASER_TURN_COUNT: Final = 2
"""How many of a shrunk segment's own turns survive as a teaser, instead of
the segment vanishing outright. Two is a starting guess — one mention plus
one reaction, not a monologue — the operator's own call (2026-08-07, after
reading a live sample where the two lowest-ranked stories of six got no
airtime at all) to be adjusted after a few more real episodes are read
against `_truncate_to_fit`'s actual behavior, not tuned in the abstract."""


def _teaser(segment: PodcastSegment) -> PodcastSegment:
    """`segment`, trimmed to its first `_TEASER_TURN_COUNT` turns.

    Turn-boundary trimming only, never mid-turn — the model's own dialogue
    always ends on a complete line, never a sentence cut off mid-word,
    which is what makes a shrunk segment read as an intentional teaser
    rather than as content that broke. Returns `segment` unchanged (same
    object) if it already has `_TEASER_TURN_COUNT` turns or fewer — a
    segment this short is already a teaser, and re-validating it through
    `PodcastSegment` a second time would be pure overhead.
    """
    if len(segment.turns) <= _TEASER_TURN_COUNT:
        return segment
    return PodcastSegment(item_ids=segment.item_ids, turns=segment.turns[:_TEASER_TURN_COUNT])


@dataclass(frozen=True, slots=True)
class _TruncationResult:
    """`_truncate_to_fit`'s full outcome — the script plus *why* each
    affected item lost coverage, not just that something shrank.

    `report/podcast.py` (Stage 4) needs this split to render a truncated
    item differently from a bad-citation drop (`_drop_unknown_segments`):
    one is "the show ran long," the other is "the model cited something
    that was never offered to it" — collapsing them into one undifferentiated
    `dropped_item_ids` would let a confabulation read as routine cost
    trimming, or vice versa. The operator asked for exactly this split
    (2026-08-07), on the same reasoning `BuiltScript.dropped_item_ids`
    already gives for citation drops."""

    script: PodcastScript
    truncated: bool
    teasered_item_ids: tuple[int, ...]
    """Ids whose segment survived, but shrunk to `_TEASER_TURN_COUNT` turns
    — partial airtime, not silence."""
    dropped_item_ids: tuple[int, ...]
    """Ids whose segment was removed outright in phase 2 — zero airtime,
    same as a citation drop from the listener's perspective, but for a
    length reason rather than a trust one."""


def _truncate_to_fit(script: PodcastScript, *, max_chars: int) -> _TruncationResult:
    """Shrink segments from the end until the script fits `max_chars`.

    Two phases, both walking from the end — the model's own narrative
    order, so the lowest-ranked story (the one `select_digest_items` put
    last among the items actually sent) is always the first one shrunk.
    Trimming from the end, rather than anywhere else, is also what keeps
    this safe post-v2: the intro/outro no longer name specific stories
    (`PODCAST_SCRIPT_VERSION`'s docstring), so there is no risk of leaving
    either referencing a segment that changed shape or vanished.

    **Phase 1 — teaser, not delete.** Each segment, walking from the end,
    becomes a `_teaser` rather than disappearing, until the script fits —
    rank order decides who shrinks, not that segment's own size, the same
    tiebreak `select_digest_items` already made when it ranked the items.
    A story that would previously have gotten zero airtime because it was
    ranked last still gets a mention.

    **Phase 2 — drop, last resort.** Only reached once every segment is
    already a teaser and the script still doesn't fit (a show with an
    unusually long intro/outro, or enough items that even one-exchange
    teasers overflow). Drops teasers from the end exactly as the pre-teaser
    version of this function dropped full segments. If intro + outro alone
    already exceed the cap, this returns a script with zero segments rather
    than mangling either — `build_script` treats that as "nothing usable,"
    same as a script with no segments the model returned validly.

    A segment teasered in phase 1 and then popped in phase 2 counts only as
    dropped, not both: by the time phase 2 runs, all it airs is nothing, so
    reporting it as "teasered" as well would overstate what the listener
    actually got.
    """

    def total_chars(segments: list[PodcastSegment]) -> int:
        candidate = PodcastScript(
            intro_turns=script.intro_turns, segments=segments, outro_turns=script.outro_turns
        )
        return _total_chars(candidate)

    original_segments = list(script.segments)
    segments = list(original_segments)
    if total_chars(segments) <= max_chars:
        return _TruncationResult(script, False, (), ())

    teasered_flags = [False] * len(segments)
    for index in range(len(segments) - 1, -1, -1):
        if total_chars(segments) <= max_chars:
            break
        teased = _teaser(segments[index])
        if teased is not segments[index]:
            teasered_flags[index] = True
        segments[index] = teased

    while segments and total_chars(segments) > max_chars:
        segments.pop()
        teasered_flags.pop()

    kept_count = len(segments)
    dropped_item_ids = tuple(
        item_id for segment in original_segments[kept_count:] for item_id in segment.item_ids
    )
    teasered_item_ids = tuple(
        item_id
        for segment, teasered in zip(segments, teasered_flags, strict=True)
        if teasered
        for item_id in segment.item_ids
    )

    return _TruncationResult(
        PodcastScript(
            intro_turns=script.intro_turns, segments=segments, outro_turns=script.outro_turns
        ),
        True,
        teasered_item_ids,
        dropped_item_ids,
    )


def _empty_script(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    dropped_item_count: int = 0,
    dropped_item_ids: tuple[int, ...] = (),
) -> BuiltScript:
    """The "nothing usable" outcome, carrying whatever was already spent.

    Every early-return path in `build_script` funnels through here rather
    than a bare `None`, so a refusal, a schema failure, or a script that
    cleaned down to nothing still reports the tokens `run_podcast_script`
    already billed for it (NEVER rule 11) — see `BuiltScript.script`'s
    docstring for why a bare `None` return was the bug an architectural
    review caught here. `dropped_item_ids` matters even when `.script` is
    `None`: the one case where every segment cited an unknown id is exactly
    the case where the record of which ids the model fabricated must not
    also be the thing that gets thrown away.
    """
    return BuiltScript(
        script=None,
        script_version=PODCAST_SCRIPT_VERSION,
        model=PODCAST_MODEL,
        dropped_item_ids=dropped_item_ids,
        dropped_item_count=dropped_item_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def build_script(
    stable_prefix: str,
    *,
    date_label: str,
    presenter_a: str,
    presenter_b: str,
    items: Sequence[Item],
) -> BuiltScript:
    """Write, validate, and clean one episode's script.

    Always returns a `BuiltScript` — never raises once the first call has
    been billed, mirroring `curate.scout.scout_for_proposals`'s "never
    raises once the call has been billed" promise and for the identical
    reason: the caller reads spend off the return value, so losing it to an
    exception or a bare `None` return would report a paid Opus call as free
    (NEVER rule 11). `.script` is `None` for every "nothing usable"
    outcome — no items, a refused call, a script that cleaned down to
    nothing — each logged via `logger.warning` with the reason (CLAUDE.md
    §7's monitoring-channel intent), though the log line itself is not yet
    threaded into `runs.errors`; that wiring is Stage 6's job, once a CLI
    caller exists to do it. Otherwise `.script` always has at least one
    segment.

    Cleaning order matters and is deliberate: unknown-id segments are
    dropped **before** the char-cap check, not after. A hallucinated,
    oversized segment citing an id the model was never given would
    otherwise (a) trigger a real second Opus call to "shorten" content that
    was about to be discarded anyway, and (b) very likely get removed by
    `_truncate_to_fit` before `_drop_unknown_segments` ever saw it — losing
    the confabulation record entirely and misreporting the episode as
    merely "too long." Cleaning first can only ever shrink the script, so
    it never creates a false negative on the length check.
    """
    item_tuples = [
        (item.id, item.title, item.summary, item.content) for item in items if item.id is not None
    ]
    unsaved_count = len(items) - len(item_tuples)
    if unsaved_count:
        # Not expected in practice — the caller's item slice comes from the
        # DB (`report.daily.select_digest_items`), which never hands back a
        # row with no id — but silently dropping it here would be the same
        # invisible-clamp shape `dropped_item_count` exists to avoid one
        # layer up, just for a different reason (unsaved, not over-budget).
        logger.warning(
            "podcast build_script dropped unsaved items with no id",
            extra={"unsaved_count": unsaved_count},
        )
    if not item_tuples:
        return _empty_script()

    result = run_podcast_script(
        stable_prefix,
        date_label=date_label,
        presenter_a=presenter_a,
        presenter_b=presenter_b,
        items=item_tuples,
    )
    input_tokens = result.input_tokens
    output_tokens = result.output_tokens
    dropped_item_count = result.dropped_item_count

    if result.script is None:
        logger.warning(
            "podcast script call produced no usable script", extra={"error": result.error}
        )
        return _empty_script(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_count=dropped_item_count,
        )

    known_ids = set(result.sent_item_ids)
    script, dropped_item_ids = _drop_unknown_segments(result.script, known_ids)
    if not script.segments:
        # Checked here, before the char-cap/retry block below, not just once
        # at the end: a script emptied entirely by cleaning is exactly the
        # "nothing usable" case that block exists to *avoid* paying for, not
        # a reason to make the one real, billed API call in this function it
        # doesn't otherwise need to make.
        logger.warning("podcast script had no segments left after cleaning")
        return _empty_script(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_count=dropped_item_count,
            dropped_item_ids=dropped_item_ids,
        )

    truncated = False
    teasered_item_ids: tuple[int, ...] = ()
    truncation_dropped_item_ids: tuple[int, ...] = ()
    if _total_chars(script) > PODCAST_MAX_SCRIPT_CHARS:
        try:
            retry = run_podcast_script(
                stable_prefix,
                date_label=date_label,
                presenter_a=presenter_a,
                presenter_b=presenter_b,
                items=item_tuples,
                shorter=True,
            )
        except LlmError:
            # `run_podcast_script` only raises for a failure *before* any
            # response comes back — nothing was spent on this retry, but the
            # first call's `input_tokens`/`output_tokens` are already in the
            # closure above, and letting this propagate would take them down
            # with it (the exact accounting gap `scout_for_proposals`'s
            # `except Exception` wrapper closes for the scout's own retry-
            # adjacent spend). Falling back to the first attempt's cleaned
            # script — already validated and non-empty at this point — costs
            # nothing further and is strictly better than losing the record
            # of what the first call cost.
            logger.warning("podcast 'shorter' retry failed before a response came back")
        else:
            input_tokens += retry.input_tokens
            output_tokens += retry.output_tokens
            if retry.script is not None:
                # The retry *replaces* the first attempt's script wholesale, and
                # with it `dropped_item_ids`: if the first attempt hallucinated
                # an id and the retry comes back clean, that confabulation goes
                # unrecorded. Deliberate — a retry is a fresh attempt, not a
                # merge, and there is no principled way to decide whether a
                # segment the retry chose not to write at all corresponds to
                # one the first attempt hallucinated or one it simply dropped.
                script, dropped_item_ids = _drop_unknown_segments(retry.script, known_ids)
                if not script.segments:
                    logger.warning("podcast script had no segments left after cleaning the retry")
                    return _empty_script(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        dropped_item_count=dropped_item_count,
                        dropped_item_ids=dropped_item_ids,
                    )
        if _total_chars(script) > PODCAST_MAX_SCRIPT_CHARS:
            truncation = _truncate_to_fit(script, max_chars=PODCAST_MAX_SCRIPT_CHARS)
            script = truncation.script
            truncated = truncation.truncated
            teasered_item_ids = truncation.teasered_item_ids
            truncation_dropped_item_ids = truncation.dropped_item_ids

    final_script = _flatten_and_finalize(script)
    if final_script is None:
        logger.warning("podcast script had no usable content left after flattening")
        return _empty_script(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_count=dropped_item_count,
            dropped_item_ids=dropped_item_ids,
        )

    return BuiltScript(
        script=final_script,
        script_version=PODCAST_SCRIPT_VERSION,
        model=PODCAST_MODEL,
        dropped_item_ids=dropped_item_ids,
        dropped_item_count=dropped_item_count,
        truncated=truncated,
        teasered_item_ids=teasered_item_ids,
        truncation_dropped_item_ids=truncation_dropped_item_ids,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
