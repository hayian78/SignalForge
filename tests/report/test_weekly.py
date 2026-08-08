"""Tests for the Weekly Intelligence Brief's selection half (`report/weekly.py`).

Scope: the window arithmetic and the deterministic partition of a week's kept
items into the brief's body and its near-miss list. Everything here is pure —
no DB, no LLM, no clock — so tests build `DigestItem` values directly rather
than seeding a database (CLAUDE.md §8).

Phase 1's acceptance gate measures *which items the brief carried*, so these
tests are the ones guarding the metric's own denominator.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from signalforge.config import Thresholds
from signalforge.db import DigestItem
from signalforge.feedback import CHECKBOX_VERDICTS, checkbox_marker, parse_marks
from signalforge.llm import WeeklyBrief, WeeklyCluster, WeeklyLead
from signalforge.models import Item, SourceType
from signalforge.report.weekly import (
    WEEKLY_WINDOW_DAYS,
    BriefContext,
    WeeklySelection,
    brief_path,
    build_brief_context,
    drop_noise_marked,
    passes_weekly_gate,
    render_brief,
    select_weekly_items,
    weekly_window,
    write_brief,
)
from signalforge.synth.weekly import WEEKLY_BRIEF_VERSION, BuiltBrief
from tests.conftest import make_item

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FIXTURE = REPO_ROOT / "fixtures" / "weekly_brief_golden.md"

SYDNEY = ZoneInfo("Australia/Sydney")
NEW_YORK = ZoneInfo("America/New_York")

TARGET_SUNDAY = date(2026, 8, 9)


def _thresholds(**overrides: object) -> Thresholds:
    """The shipped shape, so a test that does not care about a knob gets the
    real one rather than an invented value the operator has never run."""
    values: dict[str, object] = {
        "weekly_min_signal": 3,
        "weekly_min_relevance": 3,
        "weekly_min_total": 10,
        "daily_max_items": 15,
        "daily_max_per_source": 2,
        "daily_max_per_github_repo": 1,
        "weekly_top_n": 12,
        "weekly_near_miss_n": 5,
    }
    values.update(overrides)
    return Thresholds(**values)  # type: ignore[arg-type]


def _scored(
    item_id: int,
    *,
    signal: int | None = 4,
    relevance: int | None = 4,
    novelty: int | None = 4,
    source_id: str = "simonwillison",
    url: str | None = None,
    **item_overrides: object,
) -> DigestItem:
    item = make_item(
        id=item_id,
        source_id=source_id,
        external_id=f"guid-{item_id}",
        url=f"https://example.com/{item_id}" if url is None else url,
        **item_overrides,
    )
    return DigestItem(
        item=item,
        signal=signal,
        relevance=relevance,
        novelty=novelty,
        reasoning="Because it matters.",
        model="claude-haiku-4-5",
        rubric_version="triage-v3",
        scored_at=datetime(2026, 8, 5, 9, 0, tzinfo=UTC),
    )


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #


def test_the_window_is_the_seven_days_before_the_target_sunday() -> None:
    """The brief published on the 9th covers the 2nd through the 8th. The 9th
    itself is excluded: the score pass runs in the evening (DESIGN §14) and the
    brief runs Sunday morning, so the 9th's bucket is still twelve hours from
    being filled when the brief is written."""
    start, end = weekly_window(TARGET_SUNDAY, UTC)
    assert start == "2026-08-02T00:00:00+00:00"
    assert end == "2026-08-09T00:00:00+00:00"


def test_the_window_is_seven_calendar_days_wide() -> None:
    start, end = weekly_window(TARGET_SUNDAY, UTC)
    span = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    assert span.days == WEEKLY_WINDOW_DAYS


def test_the_window_tracks_the_operators_zone_not_utc() -> None:
    # Sydney is UTC+10, so the local week opens at 14:00 UTC the day before.
    start, end = weekly_window(TARGET_SUNDAY, SYDNEY)
    assert start == "2026-08-01T14:00:00+00:00"
    assert end == "2026-08-08T14:00:00+00:00"


def test_consecutive_briefs_tile_with_no_gap_and_no_overlap() -> None:
    """A gap here strands a day's items in no brief at all — permanently, since
    nothing re-scores them.

    Tiling alone is not enough to prove correctness: a naive `midnight + 168h`
    implementation also tiles. So this also measures the span of the window that
    *contains* New York's spring-forward, which such an implementation gets
    wrong by an hour."""
    this_week = weekly_window(date(2026, 3, 8), NEW_YORK)
    next_week = weekly_window(date(2026, 3, 15), NEW_YORK)
    assert this_week[1] == next_week[0]

    across_the_transition = datetime.fromisoformat(next_week[1]) - datetime.fromisoformat(
        next_week[0]
    )
    assert across_the_transition == timedelta(hours=167)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def test_the_gate_drops_an_item_the_daily_digest_would_have_shown() -> None:
    """3/3/2 sums to 8. Triage keeps it — `score/rubrics.py` asks only that an
    item "plausibly clears" the bar — and the digest shows it. The brief's
    arithmetic is what actually enforces `weekly_min_total: 10`."""
    assert not passes_weekly_gate(
        _scored(1, signal=3, relevance=3, novelty=2),
        min_signal=3,
        min_relevance=3,
        min_total=10,
    )


def test_the_gate_keeps_an_item_that_clears_every_dimension() -> None:
    assert passes_weekly_gate(
        _scored(1, signal=3, relevance=3, novelty=4),
        min_signal=3,
        min_relevance=3,
        min_total=10,
    )


def test_a_strong_total_cannot_rescue_a_weak_dimension() -> None:
    """The bar is a conjunction, not a sum: 5/2/5 totals 12 but relevance 2 is
    below the floor. An item can be exciting and still not be *for you*."""
    assert not passes_weekly_gate(
        _scored(1, signal=5, relevance=2, novelty=5),
        min_signal=3,
        min_relevance=3,
        min_total=10,
    )


@pytest.mark.parametrize("dimension", ["signal", "relevance", "novelty"])
def test_a_missing_dimension_fails_the_gate_rather_than_passing_by_default(
    dimension: str,
) -> None:
    """An unscored dimension has not cleared anything. Coalescing it to a
    passing value would let a partial score row into the one report whose whole
    purpose is being tighter than the digest."""
    scored = _scored(
        1,
        signal=None if dimension == "signal" else 5,
        relevance=None if dimension == "relevance" else 5,
        novelty=None if dimension == "novelty" else 5,
    )
    assert not passes_weekly_gate(scored, min_signal=3, min_relevance=3, min_total=15)


# --------------------------------------------------------------------------- #
# noise marks
# --------------------------------------------------------------------------- #


def test_noise_marked_items_are_dropped_and_the_rest_keep_their_order() -> None:
    items = [_scored(1), _scored(2), _scored(3)]
    kept = drop_noise_marked(items, {2: ("noise",)})
    assert [scored.item.id for scored in kept] == [1, 3]


def test_dropping_noise_is_a_filter_not_a_reorder() -> None:
    """The result must stay a sub-sequence of the input — that is what keeps the
    brief a pure function of its window and lets it re-render byte-identically.

    The fixture arrives in genuine ranked order (descending score sums) with
    deliberately non-monotonic ids, so an implementation that re-sorted by id —
    or by anything but the incoming order — would produce a different list. Ids
    ascending in lockstep with rank would let that bug pass unnoticed."""
    ranked = [(7, 5), (2, 5), (9, 4), (1, 4), (5, 3), (8, 3), (4, 2)]
    items = [
        _scored(item_id, signal=score, relevance=score, novelty=score, source_id=f"s{item_id}")
        for item_id, score in ranked
    ]
    kept = drop_noise_marked(items, {9: ("noise",), 8: ("noise",)})
    assert [scored.item.id for scored in kept] == [7, 2, 1, 5, 4]


def test_a_contradictory_pair_of_marks_resolves_to_the_stronger_rung() -> None:
    """An item marked both `noise` and `useful` is a mis-click or a mind changed.
    It stays, because the ladder's rule is highest-rung-wins everywhere."""
    kept = drop_noise_marked([_scored(1)], {1: ("noise", "useful")})
    assert [scored.item.id for scored in kept] == [1]


def test_an_unmarked_week_yields_every_item() -> None:
    """Marking is optional. A week the operator never opened still produces a
    normal, score-ranked brief rather than an empty one."""
    items = [_scored(1), _scored(2)]
    assert drop_noise_marked(items, {}) == items


# --------------------------------------------------------------------------- #
# the partition
# --------------------------------------------------------------------------- #


def test_the_body_and_the_near_misses_are_disjoint() -> None:
    """Near-misses prompt for `missed`, a verdict that only makes sense for an
    item the brief did not show. An item in both lists would ask the operator to
    mark something they are looking at as missing."""
    items = [
        _scored(1, signal=5, relevance=5, novelty=5, source_id="a"),
        _scored(2, signal=3, relevance=3, novelty=2, source_id="b"),
        _scored(3, signal=4, relevance=4, novelty=4, source_id="c"),
        _scored(4, signal=1, relevance=1, novelty=1, source_id="d"),
    ]
    selection = select_weekly_items(items, {}, thresholds=_thresholds())

    body = {scored.item.id for scored in selection.items}
    near = {scored.item.id for scored in selection.near_misses}
    assert body == {1, 3}
    assert near == {2, 4}
    assert body & near == set()


def test_a_noise_marked_gate_failer_reaches_neither_list() -> None:
    """`noise` is dropped before the gate partitions, so a rejected item cannot
    take a body slot *or* come back as a near-miss asking to be marked."""
    items = [
        _scored(1, signal=5, relevance=5, novelty=5),
        _scored(2, signal=3, relevance=3, novelty=2, source_id="other"),
    ]
    selection = select_weekly_items(items, {2: ("noise",)}, thresholds=_thresholds())

    assert [scored.item.id for scored in selection.items] == [1]
    assert selection.near_misses == ()
    assert selection.gated_out_count == 0


def test_uncitable_items_are_dropped_before_they_can_take_a_slot() -> None:
    """An item with no URL cannot be cited, so rendering would drop it anyway
    (NEVER rule 7). Dropping it after selection would silently shorten the brief
    by one item instead of promoting the next candidate.

    `Item.url` is required, so this is built via `model_construct` — bypassing
    validation the way a corrupted row or a future nullable field might, the
    same way `test_daily.py` reaches this guard."""
    uncitable = DigestItem(
        item=Item.model_construct(
            id=1,
            source_id="x",
            source_type=SourceType.RSS,
            external_id=None,
            url="",
            canonical_url="https://example.com/x",
            title="An item with no URL",
            author=None,
            published_at=None,
            fetched_at=datetime(2026, 8, 5, 9, 0, tzinfo=UTC),
            summary=None,
            content=None,
            content_hash="deadbeef",
            lang="en",
            raw_path=None,
        ),
        signal=5,
        relevance=5,
        novelty=5,
        reasoning="This would otherwise lead the brief.",
        model="claude-haiku-4-5",
        rubric_version="triage-v3",
        scored_at=datetime(2026, 8, 5, 9, 0, tzinfo=UTC),
    )
    selection = select_weekly_items(
        [uncitable, _scored(2, source_id="other")], {}, thresholds=_thresholds(weekly_top_n=1)
    )
    assert [scored.item.id for scored in selection.items] == [2]


def test_the_per_source_cap_applies_over_the_whole_week() -> None:
    """The digest's own `daily_max_per_source` is reused deliberately: over seven
    days it is much tighter than over one, and one source dominating a week is
    one story, not N."""
    items = [_scored(index, source_id="chatty") for index in range(1, 6)]
    selection = select_weekly_items(items, {}, thresholds=_thresholds(daily_max_per_source=2))
    assert [scored.item.id for scored in selection.items] == [1, 2]
    assert selection.hidden_count == 3


def test_the_tighter_per_repo_cap_wins_for_a_release_watch() -> None:
    """For a `github` source the source id *is* the repo, so a repo shipping four
    versions in a week is one piece of news. Wiring this to `daily_max_per_source`
    instead would silently double every repo's weekly share — the shipped config
    sets 1 under a per-source 2."""
    items = [
        _scored(index, source_id="anthropics/claude-code", source_type=SourceType.GITHUB)
        for index in range(1, 5)
    ]
    selection = select_weekly_items(
        items,
        {},
        thresholds=_thresholds(daily_max_per_source=2, daily_max_per_github_repo=1),
    )
    assert [scored.item.id for scored in selection.items] == [1]


def test_the_gate_reads_each_threshold_into_its_own_dimension() -> None:
    """Asymmetric thresholds, so a swapped signal/relevance wiring is visible.
    With equal thresholds — the shipped config's shape — the mis-wire is
    invisible, and it is exactly the kind that survives a rename."""
    thresholds = _thresholds(weekly_min_signal=4, weekly_min_relevance=2, weekly_min_total=3)
    clears = _scored(1, signal=4, relevance=2, novelty=1)
    swapped = _scored(2, signal=2, relevance=4, novelty=1, source_id="other")

    selection = select_weekly_items([clears, swapped], {}, thresholds=thresholds)
    assert [scored.item.id for scored in selection.items] == [1]
    assert [scored.item.id for scored in selection.near_misses] == [2]


def test_top_n_truncates_the_ranking_rather_than_deciding_it() -> None:
    items = [_scored(index, source_id=f"s{index}") for index in range(1, 10)]
    selection = select_weekly_items(items, {}, thresholds=_thresholds(weekly_top_n=4))
    assert [scored.item.id for scored in selection.items] == [1, 2, 3, 4]
    assert selection.hidden_count == 5


def test_the_near_miss_list_shows_the_closest_misses_not_the_worst() -> None:
    """The footer reports how many items the gate turned away; the list shows
    only the top few, so a bad week does not bury the brief under its own
    rejects. Which few matters: `mark <id> missed` is worth offering on the
    items that nearly made it, not on the week's three weakest."""
    ranked = [(11, 3), (12, 3), (13, 2), (14, 2), (15, 1), (16, 1)]
    items = [
        _scored(item_id, signal=score, relevance=score, novelty=score, source_id=f"s{item_id}")
        for item_id, score in ranked
    ]
    selection = select_weekly_items(items, {}, thresholds=_thresholds(weekly_near_miss_n=3))

    assert [scored.item.id for scored in selection.near_misses] == [11, 12, 13]
    assert selection.gated_out_count == 6


def test_an_unset_near_miss_knob_omits_the_list_without_losing_the_count() -> None:
    items = [_scored(1, signal=1, relevance=1, novelty=1)]
    selection = select_weekly_items(items, {}, thresholds=_thresholds(weekly_near_miss_n=None))
    assert selection.near_misses == ()
    assert selection.gated_out_count == 1


def test_an_unset_top_n_means_the_brief_is_off() -> None:
    """Same degradation as `podcast_top_n`: an interests.yaml predating the brief
    produces no brief, rather than one sized by an invented default that would
    bill an Opus call the operator never asked for."""
    items = [_scored(1), _scored(2)]
    selection = select_weekly_items(items, {}, thresholds=_thresholds(weekly_top_n=None))
    assert selection.items == ()
    assert selection.hidden_count == 2


def test_every_item_is_rendered_counted_or_gated_out() -> None:
    """The footer has to reconcile: nothing may be silently dropped (CLAUDE.md
    §7). Body + hidden + gated-out must account for every citable, non-noise
    item in the window."""
    items = [
        *[_scored(index, source_id=f"pass{index}") for index in range(1, 6)],
        *[
            _scored(index, signal=1, relevance=1, novelty=1, source_id=f"fail{index}")
            for index in range(6, 9)
        ],
    ]
    selection = select_weekly_items(items, {}, thresholds=_thresholds(weekly_top_n=3))

    accounted = len(selection.items) + selection.hidden_count + selection.gated_out_count
    assert accounted == len(items)


def test_selection_is_deterministic_across_calls() -> None:
    """Same rows, same config, same result — the property the whole
    re-render-byte-identically guarantee rests on."""
    items = [_scored(index, source_id=f"s{index}") for index in range(1, 6)]
    thresholds = _thresholds()
    first = select_weekly_items(items, {}, thresholds=thresholds)
    second = select_weekly_items(items, {}, thresholds=thresholds)
    assert first == second


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _built(
    *,
    leads: list[WeeklyLead] | None = None,
    clusters: list[WeeklyCluster] | None = None,
    dropped: tuple[int, ...] = (),
) -> BuiltBrief:
    return BuiltBrief(
        brief=WeeklyBrief(leads=leads or [], clusters=clusters or []),
        model="claude-opus-5",
        input_tokens=1234,
        output_tokens=567,
        dropped_item_ids=dropped,
    )


def _selection(*ids: int) -> WeeklySelection:
    items = [_scored(item_id, source_id=f"s{item_id}") for item_id in ids]
    return select_weekly_items(items, {}, thresholds=_thresholds())


def test_the_body_renders_a_checkbox_for_every_selected_item() -> None:
    """The acceptance gate counts brief items, so every selected item must be
    markable — whether or not the synthesis happened to cite it. Attaching
    checkboxes to citations instead would let the model decide the denominator."""
    context = build_brief_context(
        _built(leads=[WeeklyLead(headline="H", why_it_matters="W", item_ids=[1])]),
        _selection(1, 2, 3),
        target_sunday=TARGET_SUNDAY,
    )
    rendered = render_brief(context)

    assert context.item_count == 3
    for item_id in (1, 2, 3):
        for verdict in CHECKBOX_VERDICTS:
            assert checkbox_marker(item_id, verdict) in rendered


def test_item_count_counts_items_not_cited_blocks() -> None:
    context = build_brief_context(
        _built(
            leads=[WeeklyLead(headline="H", why_it_matters="W", item_ids=[1])],
            clusters=[WeeklyCluster(title="T", narrative="N", item_ids=[1, 2])],
        ),
        _selection(1, 2, 3, 4),
        target_sunday=TARGET_SUNDAY,
    )
    assert context.item_count == 4


def _one_block(kind: str, *, text: str = "Invented", item_ids: list[int]) -> BuiltBrief:
    """A brief carrying exactly one lead, or exactly one cluster.

    Leads and clusters are separate loops in `build_brief_context`, so every
    guard has to be asserted on both — a test covering only clusters leaves the
    lead copy, which is the brief's most prominent claim, unguarded."""
    if kind == "lead":
        return _built(leads=[WeeklyLead(headline=text, why_it_matters="W", item_ids=item_ids)])
    return _built(clusters=[WeeklyCluster(title=text, narrative="N", item_ids=item_ids)])


def _blocks(context: BriefContext, kind: str) -> tuple[object, ...]:
    return context.leads if kind == "lead" else context.clusters


@pytest.mark.parametrize("kind", ["lead", "cluster"])
def test_a_block_citing_an_item_outside_the_selection_is_dropped(kind: str) -> None:
    """Render-time citation resolution (NEVER rule 7). The brief only ever sends
    the model its own selection, so an id outside it is a fabrication."""
    context = build_brief_context(
        _one_block(kind, item_ids=[999]), _selection(1, 2), target_sunday=TARGET_SUNDAY
    )
    assert _blocks(context, kind) == ()
    assert "Invented" not in render_brief(context)


@pytest.mark.parametrize("kind", ["lead", "cluster"])
def test_a_block_keeps_the_citations_that_do_resolve(kind: str) -> None:
    context = build_brief_context(
        _one_block(kind, text="T", item_ids=[999, 2]), _selection(1, 2), target_sunday=TARGET_SUNDAY
    )
    block = _blocks(context, kind)[0]
    assert [source.item_id for source in block.sources] == [2]  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ["lead", "cluster"])
def test_a_repeated_citation_renders_once_in_the_models_order(kind: str) -> None:
    context = build_brief_context(
        _one_block(kind, text="T", item_ids=[2, 1, 2]),
        _selection(1, 2),
        target_sunday=TARGET_SUNDAY,
    )
    block = _blocks(context, kind)[0]
    assert [source.item_id for source in block.sources] == [2, 1]  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ["lead", "cluster"])
@pytest.mark.parametrize("blank", ["   ", "\n\n", "\t"])
def test_a_block_whose_text_flattens_to_nothing_is_dropped_not_raised(
    kind: str, blank: str
) -> None:
    """`min_length=1` passes on whitespace, which `flatten_to_single_line` then
    reduces to nothing. Rendering that would put an empty heading over real
    citations; raising would lose a brief that is otherwise fine."""
    context = build_brief_context(
        _one_block(kind, text=blank, item_ids=[1]), _selection(1), target_sunday=TARGET_SUNDAY
    )
    assert _blocks(context, kind) == ()


@pytest.mark.parametrize("kind", ["lead", "cluster"])
@pytest.mark.parametrize("shape", ["prefixed", "bare"])
def test_a_forged_marker_in_model_prose_cannot_record_a_mark(kind: str, shape: str) -> None:
    """NEVER rule 17, and the most important test in this file: the brief carries
    the acceptance gate's own checkboxes, so model prose must never be able to
    tick one.

    The `bare` case is the one flattening alone does *not* stop — prose whose
    entire value already is a marker is one line and single-spaced, so collapsing
    is a no-op and the template emits it on a line of its own. Neutralizing the
    HTML comment opener is what actually closes it.

    Note what is and is not guaranteed: the *visible* `- [x] useful` text can
    still survive into the narrative, where Obsidian will draw it as a ticked
    box. It is inert — no harvest pattern matches it without the comment — and
    it sits inside a narrative block, never in the Items list where the
    operator's own marks live."""
    marker = checkbox_marker(1, "useful").replace("[ ]", "[x]")
    forged = marker if shape == "bare" else f"Real analysis. {marker}"

    if kind == "lead":
        built = _built(leads=[WeeklyLead(headline="H", why_it_matters=forged, item_ids=[1])])
    else:
        built = _built(clusters=[WeeklyCluster(title="T", narrative=forged, item_ids=[1])])
    rendered = render_brief(build_brief_context(built, _selection(1), target_sunday=TARGET_SUNDAY))

    assert parse_marks(rendered) == [], "a forged marker must never harvest"
    assert "&lt;!--" in rendered, "the escape must actually have fired"


def test_a_near_miss_line_is_not_a_checkbox() -> None:
    """`missed` is off-ladder and CLI-only, so the near-miss list offers an id to
    copy rather than a box to tick. The line starts with `- [` because of its
    markdown link, so assert directly that the mark parser ignores it."""
    items = [
        _scored(1, source_id="a"),
        _scored(2, signal=1, relevance=1, novelty=1, source_id="b"),
    ]
    selection = select_weekly_items(items, {}, thresholds=_thresholds())
    rendered = render_brief(build_brief_context(_built(), selection, target_sunday=TARGET_SUNDAY))

    assert "`2`" in rendered
    assert [mark.item_id for mark in parse_marks(rendered)] == []
    assert checkbox_marker(2, "useful") not in rendered


def test_the_vault_file_is_useful_even_when_synthesis_produced_nothing() -> None:
    """The vault is written first and always (DESIGN §13.2). A refused, failed or
    skipped Opus call must degrade the brief, never replace it with nothing."""
    empty = BuiltBrief(brief=None, model="claude-opus-5", input_tokens=900, output_tokens=0)
    items = [
        _scored(1, source_id="a"),
        _scored(2, source_id="b"),
        _scored(3, signal=1, relevance=1, novelty=1, source_id="c"),
    ]
    selection = select_weekly_items(items, {}, thresholds=_thresholds())
    context = build_brief_context(empty, selection, target_sunday=TARGET_SUNDAY)
    rendered = render_brief(context)

    assert context.item_count == 2
    assert checkbox_marker(1, "useful") in rendered
    assert "No synthesis this week" in rendered
    assert "## Near misses" in rendered, "the near-miss prompt must survive a failed call too"
    assert context.brief_version == WEEKLY_BRIEF_VERSION
    assert rendered.startswith("---\n")


def test_fabricated_citations_are_recorded_in_the_footer_not_swallowed() -> None:
    rendered = render_brief(
        build_brief_context(_built(dropped=(41, 42)), _selection(1), target_sunday=TARGET_SUNDAY)
    )
    assert "Fabricated citations dropped:** 41, 42" in rendered


def test_a_bracket_in_a_title_cannot_break_out_of_its_link() -> None:
    """Titles are world-authored — they come from a feed. An unescaped `]` would
    close the markdown link early and let a headline write raw markdown."""
    items = [_scored(1, title="Breaking] news")]
    selection = select_weekly_items(items, {}, thresholds=_thresholds())
    rendered = render_brief(
        build_brief_context(
            _built(clusters=[WeeklyCluster(title="T", narrative="N", item_ids=[1])]),
            selection,
            target_sunday=TARGET_SUNDAY,
        )
    )
    assert "[Breaking&#93; news](" in rendered


def test_a_bracket_in_a_near_miss_title_cannot_break_out_of_its_link() -> None:
    """The near-miss list is the one place whose whole purpose is copy-pasting an
    id, so a corrupted link there is worse than cosmetic."""
    items = [
        _scored(1, source_id="a"),
        _scored(2, signal=1, relevance=1, novelty=1, source_id="b", title="Odd] title"),
    ]
    selection = select_weekly_items(items, {}, thresholds=_thresholds())
    rendered = render_brief(build_brief_context(_built(), selection, target_sunday=TARGET_SUNDAY))
    assert "[Odd&#93; title](" in rendered


def test_the_brief_re_renders_byte_identically() -> None:
    built = _built(clusters=[WeeklyCluster(title="T", narrative="N", item_ids=[1])])
    selection = _selection(1, 2)
    first = render_brief(build_brief_context(built, selection, target_sunday=TARGET_SUNDAY))
    second = render_brief(build_brief_context(built, selection, target_sunday=TARGET_SUNDAY))
    assert first == second


def test_the_path_is_stable_and_lives_under_weekly() -> None:
    assert brief_path(Path("/vault"), target_sunday=TARGET_SUNDAY) == Path(
        "/vault/weekly/2026-08-09.md"
    )


def test_write_brief_creates_the_directory_and_overwrites_in_place(tmp_path: Path) -> None:
    built = _built(clusters=[WeeklyCluster(title="T", narrative="N", item_ids=[1])])
    selection = _selection(1)
    first = write_brief(built, selection, target_sunday=TARGET_SUNDAY, vault_dir=tmp_path)
    body = first.read_text(encoding="utf-8")
    second = write_brief(built, selection, target_sunday=TARGET_SUNDAY, vault_dir=tmp_path)

    assert first == second
    assert second.read_text(encoding="utf-8") == body


def test_render_brief_matches_the_golden_fixture() -> None:
    """Pins the whole file shape in one artifact: frontmatter, lead and cluster
    layout, the checkbox block, near-miss lines without checkboxes, and the
    fabricated-citation footer."""
    items = [
        _scored(1, signal=5, relevance=5, novelty=5, source_id="simonwillison", title="Story A"),
        _scored(2, signal=4, relevance=5, novelty=4, source_id="hn", title="Story B"),
        _scored(3, signal=4, relevance=4, novelty=3, source_id="openai-news", title="Story C"),
        _scored(4, signal=3, relevance=4, novelty=2, source_id="latent-space", title="Near miss D"),
    ]
    selection = select_weekly_items(items, {}, thresholds=_thresholds())
    built = _built(
        leads=[
            WeeklyLead(
                headline="Agent intrusions moved from theory to timeline",
                why_it_matters="Two independent write-ups landed the same week.",
                item_ids=[1, 2],
            )
        ],
        clusters=[
            WeeklyCluster(
                title="Adoption keeps outrunning the tooling",
                narrative="The gap shows up in every deployment report.",
                item_ids=[3],
            )
        ],
        dropped=(77,),
    )
    rendered = render_brief(build_brief_context(built, selection, target_sunday=TARGET_SUNDAY))

    assert rendered == GOLDEN_FIXTURE.read_text(encoding="utf-8")
