"""Tests for `synth.weekly.build_brief` — the citation and flatten pass.

`llm.py` is faked at *its* boundary (`monkeypatch.setattr` on the name this
module imports), never at the SDK's: `synth/` takes no `client` parameter, which
is what makes this the only seam and what keeps `anthropic` confined to `llm.py`
(NEVER rules 1 and 13).

What these tests protect:

* **Citation discipline (NEVER rule 7).** A block citing an id the model was
  never sent is dropped whole, and the fabricated ids are recorded rather than
  swallowed.
* **The flatten boundary (NEVER rule 17).** Model prose is flattened before it
  can reach a vault file, and a block that flattens to nothing is dropped rather
  than raising on the way out.
* **The accounting (NEVER rule 11).** Every "nothing usable" path still carries
  what the call spent.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from signalforge.db import DigestItem
from signalforge.llm import (
    WEEKLY_MAX_CLUSTERS,
    WEEKLY_MAX_LEADS,
    WeeklyBrief,
    WeeklyBriefResult,
    WeeklyCluster,
    WeeklyLead,
)
from signalforge.synth.weekly import WEEKLY_BRIEF_VERSION, BuiltBrief, build_brief
from tests.conftest import make_item

STABLE_PREFIX = "You write a weekly intelligence brief."
WINDOW_LABEL = "2 August to 8 August 2026"


def _scored(item_id: int, *, reasoning: str = "It matters.") -> DigestItem:
    return DigestItem(
        item=make_item(
            id=item_id, external_id=f"g-{item_id}", url=f"https://example.com/{item_id}"
        ),
        signal=4,
        relevance=4,
        novelty=4,
        reasoning=reasoning,
        model="claude-haiku-4-5",
        rubric_version="triage-v3",
        scored_at=datetime(2026, 8, 5, 9, tzinfo=UTC),
    )


def _lead(*, headline: str = "H", why: str = "W", ids: list[int] | None = None) -> WeeklyLead:
    return WeeklyLead(headline=headline, why_it_matters=why, item_ids=ids or [1])


def _cluster(
    *, title: str = "T", narrative: str = "N", ids: list[int] | None = None
) -> WeeklyCluster:
    return WeeklyCluster(title=title, narrative=narrative, item_ids=ids or [1])


def _result(
    *,
    brief: WeeklyBrief | None = None,
    error: str | None = None,
    sent: tuple[int, ...] = (1,),
    input_tokens: int = 900,
    output_tokens: int = 400,
    dropped_item_count: int = 0,
) -> WeeklyBriefResult:
    return WeeklyBriefResult(
        brief=brief,
        error=error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dropped_item_count=dropped_item_count,
        sent_item_ids=sent,
    )


def _build(
    monkeypatch: pytest.MonkeyPatch,
    result: WeeklyBriefResult,
    *,
    items: Sequence[DigestItem] | None = None,
) -> tuple[BuiltBrief, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake(prefix: str, **kwargs: Any) -> WeeklyBriefResult:
        calls.append({"prefix": prefix, **kwargs})
        return result

    monkeypatch.setattr("signalforge.synth.weekly.run_weekly_brief", fake)
    built = build_brief(
        STABLE_PREFIX,
        window_label=WINDOW_LABEL,
        items=list(items) if items is not None else [_scored(1)],
    )
    return built, calls


# --------------------------------------------------------------------------- #
# what reaches llm.py
# --------------------------------------------------------------------------- #


def test_only_four_fields_per_item_reach_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The projection lives here so the absence of `Item.content` is visible at
    the one place it could be added by accident (DESIGN §8)."""
    _, calls = _build(monkeypatch, _result(brief=WeeklyBrief(leads=[_lead()])))

    assert calls[0]["items"] == [
        (1, "MCP sampling lands everywhere", "A short feed summary.", "It matters.")
    ]


def test_the_stable_prefix_is_passed_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, calls = _build(monkeypatch, _result(brief=WeeklyBrief(leads=[_lead()])))
    assert calls[0]["prefix"] == STABLE_PREFIX


def test_no_items_means_no_call_and_no_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A week with nothing selected must not bill for a brief about nothing."""
    built, calls = _build(monkeypatch, _result(), items=[])

    assert calls == []
    assert built.brief is None
    assert built.input_tokens == 0
    assert built.output_tokens == 0


# --------------------------------------------------------------------------- #
# citation discipline (NEVER rule 7)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["lead", "cluster"])
def test_a_block_citing_an_id_that_was_never_sent_is_dropped(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief = (
        WeeklyBrief(leads=[_lead(ids=[99])])
        if kind == "lead"
        else WeeklyBrief(clusters=[_cluster(ids=[99])])
    )
    built, _ = _build(monkeypatch, _result(brief=brief, sent=(1,)))

    assert built.brief is None
    assert built.dropped_item_ids == (99,)


def test_a_block_is_dropped_whole_when_only_one_of_its_ids_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial keeping would leave a paragraph still discussing the item whose
    citation was stripped — prose written about all its ids together."""
    built, _ = _build(
        monkeypatch,
        _result(brief=WeeklyBrief(clusters=[_cluster(ids=[1, 99])]), sent=(1,)),
    )

    assert built.brief is None
    assert built.dropped_item_ids == (99,)


def test_a_clean_block_survives_alongside_a_fabricated_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built, _ = _build(
        monkeypatch,
        _result(
            brief=WeeklyBrief(
                leads=[_lead(headline="Real", ids=[1]), _lead(headline="Fake", ids=[99])]
            ),
            sent=(1,),
        ),
    )

    assert built.brief is not None
    assert [lead.headline for lead in built.brief.leads] == ["Real"]
    assert built.dropped_item_ids == (99,)


def test_the_record_of_fabricated_ids_survives_a_brief_that_cleaned_to_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one case where `brief is None` and the drop record matters most: if
    the ids went out with the brief, nothing would say the model confabulated."""
    built, _ = _build(
        monkeypatch,
        _result(brief=WeeklyBrief(leads=[_lead(ids=[98]), _lead(ids=[99])]), sent=(1,)),
    )

    assert built.brief is None
    assert built.dropped_item_ids == (98, 99)
    assert built.input_tokens == 900
    assert built.output_tokens == 400


def test_an_id_clamped_away_before_the_request_is_not_a_legitimate_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation is against `sent_item_ids`, not the caller's own list. An item
    the byte/count clamp removed was never shown to the model, so citing it is
    confabulation rather than a citation that merely missed the cut."""
    built, _ = _build(
        monkeypatch,
        _result(brief=WeeklyBrief(leads=[_lead(ids=[2])]), sent=(1,), dropped_item_count=1),
        items=[_scored(1), _scored(2)],
    )

    assert built.brief is None
    assert built.dropped_item_ids == (2,)
    assert built.dropped_item_count == 1


# --------------------------------------------------------------------------- #
# the flatten boundary (NEVER rule 17)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("blank", ["   ", "\n\n", "\t", "\x01"])
@pytest.mark.parametrize("kind", ["lead", "cluster"])
def test_a_block_whose_text_flattens_to_nothing_is_dropped_not_raised(
    kind: str, blank: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`min_length=1` passes on whitespace and then flattening empties it.
    Reconstructing the block would violate the same constraint on the way out —
    an unhandled `ValidationError` at the end of an already-billed call."""
    brief = (
        WeeklyBrief(leads=[_lead(headline=blank)])
        if kind == "lead"
        else WeeklyBrief(clusters=[_cluster(narrative=blank)])
    )
    built, _ = _build(monkeypatch, _result(brief=brief))

    assert built.brief is None
    assert built.input_tokens == 900


@pytest.mark.parametrize("kind", ["lead-headline", "lead-why", "cluster-title", "cluster-body"])
def test_model_prose_is_flattened_before_it_can_reach_a_vault_file(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flattened here as well as in `report/weekly.py` — a boundary rule guarded
    on both sides. Idempotent, so the duplication costs nothing.

    Every model-authored field is covered, not just one: leads and clusters are
    separate loops, so a test that only exercises a cluster narrative leaves the
    whole lead path — the brief's most prominent prose — unpinned."""
    forged = "Real analysis.\n- [x] useful <!-- sf:item=1 v=useful -->"
    briefs = {
        "lead-headline": WeeklyBrief(leads=[_lead(headline=forged)]),
        "lead-why": WeeklyBrief(leads=[_lead(why=forged)]),
        "cluster-title": WeeklyBrief(clusters=[_cluster(title=forged)]),
        "cluster-body": WeeklyBrief(clusters=[_cluster(narrative=forged)]),
    }
    built, _ = _build(monkeypatch, _result(brief=briefs[kind]))

    assert built.brief is not None
    block = built.brief.leads[0] if kind.startswith("lead") else built.brief.clusters[0]
    text = " ".join(
        [
            getattr(block, field)
            for field in ("headline", "why_it_matters", "title", "narrative")
            if hasattr(block, field)
        ]
    )
    assert "\n" not in text
    assert "<!--" not in text


# --------------------------------------------------------------------------- #
# accounting on every path (NEVER rule 11)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "result",
    [
        _result(brief=None, error="refused", dropped_item_count=3),
        _result(brief=None, error="schema validation failed"),
        _result(brief=WeeklyBrief(leads=[_lead(ids=[99])])),
        _result(brief=WeeklyBrief(leads=[_lead(headline="   ")])),
    ],
    ids=["refused", "unparseable", "all-citations-fabricated", "flattened-away"],
)
def test_every_nothing_usable_path_still_reports_what_it_spent(
    result: WeeklyBriefResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, _ = _build(monkeypatch, result)

    assert built.brief is None
    assert built.input_tokens == 900
    assert built.output_tokens == 400
    assert built.model == "claude-opus-5"
    assert built.brief_version == WEEKLY_BRIEF_VERSION
    # The clamp count has to survive a failed call too: on a refusal with
    # `weekly_top_n` above the hard ceiling, the footer would otherwise stop
    # saying that items went unread.
    assert built.dropped_item_count == result.dropped_item_count


def test_the_clamp_count_travels_so_the_footer_can_report_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without this, raising `weekly_top_n` above `WEEKLY_MAX_ITEMS` would render
    checkboxes for items the synthesis never read, with nothing saying so."""
    built, _ = _build(
        monkeypatch,
        _result(brief=WeeklyBrief(leads=[_lead()]), dropped_item_count=7),
    )
    assert built.dropped_item_count == 7


def test_a_brief_with_only_clusters_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """A week with no single thing that mattered is a real week, not a failure."""
    built, _ = _build(monkeypatch, _result(brief=WeeklyBrief(clusters=[_cluster()])))

    assert built.brief is not None
    assert built.brief.leads == []
    assert len(built.brief.clusters) == 1


def test_the_version_stamp_defaults_to_the_module_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEVER rule 5's analogue for synthesis: the constant and the frontmatter
    cannot drift apart by omission."""
    built, _ = _build(monkeypatch, _result(brief=WeeklyBrief(leads=[_lead()])))
    assert built.brief_version == WEEKLY_BRIEF_VERSION


def test_a_fabricated_citation_is_recorded_even_when_the_block_is_also_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the clean-before-flatten order.

    Flattening first would drop this block as "no text left" and lose the id from
    `dropped_item_ids` — silently converting a confabulation into a routine
    formatting drop, on the one line that reports confabulation to the operator."""
    built, _ = _build(
        monkeypatch,
        _result(brief=WeeklyBrief(leads=[_lead(headline="   ", ids=[99])]), sent=(1,)),
    )

    assert built.brief is None
    assert built.dropped_item_ids == (99,)


def test_fabricated_ids_are_recorded_in_document_order_across_both_block_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leads before clusters, and every fabricated id, not just the first of a
    block. The vault footer renders this tuple verbatim."""
    built, _ = _build(
        monkeypatch,
        _result(
            brief=WeeklyBrief(leads=[_lead(ids=[97])], clusters=[_cluster(ids=[98, 99])]),
            sent=(1,),
        ),
    )

    assert built.dropped_item_ids == (97, 98, 99)


@pytest.mark.parametrize(
    ("kind", "limit"),
    [("leads", WEEKLY_MAX_LEADS), ("clusters", WEEKLY_MAX_CLUSTERS)],
)
def test_too_many_blocks_are_clamped_not_rejected(
    kind: str, limit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count bounds cannot live in the request schema — the API rejects array
    `minItems`/`maxItems` — so they are enforced here, and by clamping rather
    than rejecting: discarding a whole billed response over one extra block loses
    a good brief and invites the next invocation to pay again."""
    over = limit + 2
    brief = (
        WeeklyBrief(leads=[_lead(headline=f"H{i}") for i in range(over)])
        if kind == "leads"
        else WeeklyBrief(clusters=[_cluster(title=f"T{i}") for i in range(over)])
    )
    built, _ = _build(monkeypatch, _result(brief=brief))

    assert built.brief is not None
    assert len(getattr(built.brief, kind)) == limit


def test_clamping_keeps_the_models_own_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model put its strongest lead first; the clamp must not reorder."""
    built, _ = _build(
        monkeypatch,
        _result(
            brief=WeeklyBrief(leads=[_lead(headline=f"H{i}") for i in range(WEEKLY_MAX_LEADS + 2)])
        ),
    )

    assert built.brief is not None
    assert [lead.headline for lead in built.brief.leads] == [
        f"H{i}" for i in range(WEEKLY_MAX_LEADS)
    ]
