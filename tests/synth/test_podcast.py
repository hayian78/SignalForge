"""Tests for `synth.podcast.build_script` — cleaning a raw model script into
something `report/podcast.py` can render (DESIGN §13.3).

Faked at `llm.py`'s boundary (CLAUDE.md §8, NEVER rule 13): `build_script`
calls `llm.run_podcast_script` directly with no client seam of its own (it
must not import `anthropic` at all — NEVER rule 1), so tests replace
`signalforge.synth.podcast.run_podcast_script` itself, the same pattern
`tests/curate/test_scout.py::fake_scout` uses for `llm.run_source_scout`.
"""

from __future__ import annotations

from typing import Any

import pytest

from signalforge.llm import (
    PODCAST_MAX_SCRIPT_CHARS,
    PODCAST_MODEL,
    LlmError,
    PodcastScript,
    PodcastScriptResult,
)
from signalforge.models import Item, SourceType
from signalforge.synth.podcast import PODCAST_SCRIPT_VERSION, build_script

STABLE_PREFIX = "You write a two-presenter podcast."
DATE_LABEL = "Friday, 7 August 2026"
PRESENTER_A = "Alex"
PRESENTER_B = "Sam"


def _item(item_id: int, **overrides: object) -> Item:
    fields: dict[str, object] = {
        "id": item_id,
        "source_id": "simonwillison",
        "source_type": SourceType.RSS,
        "url": f"https://example.com/{item_id}",
        "title": f"Title {item_id}",
        "summary": f"Summary {item_id}",
    }
    fields.update(overrides)
    return Item(**fields)  # type: ignore[arg-type]


def _turn(speaker: str, chars: int = 20) -> dict[str, str]:
    return {"speaker": speaker, "text": "x" * chars}


def _script(
    *, segment_count: int = 1, chars_per_turn: int = 20, turns_per_segment: int = 2
) -> dict[str, Any]:
    return {
        "intro_turns": [_turn("A", chars_per_turn), _turn("B", chars_per_turn)],
        "segments": [
            {
                "item_ids": [index + 1],
                "turns": [
                    _turn("A" if turn_index % 2 == 0 else "B", chars_per_turn)
                    for turn_index in range(turns_per_segment)
                ],
            }
            for index in range(segment_count)
        ],
        "outro_turns": [_turn("A", chars_per_turn), _turn("B", chars_per_turn)],
    }


def _result(
    script: dict[str, Any] | None,
    *,
    sent_item_ids: tuple[int, ...],
    input_tokens: int = 1_000,
    output_tokens: int = 500,
    dropped_item_count: int = 0,
) -> PodcastScriptResult:
    return PodcastScriptResult(
        script=PodcastScript.model_validate(script) if script is not None else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dropped_item_count=dropped_item_count,
        sent_item_ids=sent_item_ids,
    )


def _fake_run_podcast_script(
    monkeypatch: pytest.MonkeyPatch, responses: list[PodcastScriptResult | LlmError]
) -> list[dict[str, Any]]:
    """Replace the paid call with a scripted sequence of results, recording
    every call's kwargs — the same shape `tests/curate/test_scout.py`'s
    `fake_scout` uses for `llm.run_source_scout`. A scripted `LlmError`
    instance is raised rather than returned, for the "the retry fails
    before a response comes back" case."""
    calls: list[dict[str, Any]] = []
    remaining = list(responses)

    def _run(stable_system_prefix: str, **kwargs: Any) -> PodcastScriptResult:
        calls.append({"stable_system_prefix": stable_system_prefix, **kwargs})
        if not remaining:  # pragma: no cover - a test scripted too few
            raise AssertionError("build_script made more calls than scripted")
        next_response = remaining.pop(0)
        if isinstance(next_response, LlmError):
            raise next_response
        return next_response

    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _run)
    return calls


def _build(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[PodcastScriptResult | LlmError],
    *,
    item_count: int = 1,
    **overrides: object,
) -> tuple[Any, list[dict[str, Any]]]:
    calls = _fake_run_podcast_script(monkeypatch, responses)
    kwargs: dict[str, Any] = {
        "stable_prefix": STABLE_PREFIX,
        "date_label": DATE_LABEL,
        "presenter_a": PRESENTER_A,
        "presenter_b": PRESENTER_B,
        "items": [_item(index + 1) for index in range(item_count)],
    }
    kwargs.update(overrides)
    result = build_script(**kwargs)
    return result, calls


def _all_ids(count: int) -> tuple[int, ...]:
    return tuple(range(1, count + 1))


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_no_items_builds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = _build(monkeypatch, [], items=[])
    assert result.script is None
    assert (result.input_tokens, result.output_tokens) == (0, 0)  # no call was ever made
    assert calls == []


def test_items_with_no_id_never_reach_the_paid_call(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The empty-input guard runs *after* filtering out unsaved items (no
    `id` yet), not before — a slice of only-unsaved items must never reach
    `run_podcast_script` with an empty payload. The drop is logged with a
    count, not silent — a code-reviewer pass flagged the earlier version
    (no log at all) as the same invisible-clamp shape `dropped_item_count`
    exists to prevent elsewhere in this module."""
    unsaved = _item(1, id=None)
    with caplog.at_level("WARNING", logger="signalforge.synth.podcast"):
        result, calls = _build(monkeypatch, [], items=[unsaved])

    assert result.script is None
    assert result.error is not None, (
        "a lost episode now flips the run to `partial` — the reason must be recorded"
    )
    assert calls == []
    drop_records = [
        record
        for record in caplog.records
        if record.getMessage() == "podcast build_script dropped unsaved items with no id"
    ]
    assert [record.unsaved_count for record in drop_records] == [1]  # type: ignore[attr-defined]


def test_a_clean_script_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = _build(monkeypatch, [_result(_script(), sent_item_ids=(1,))])

    assert result.script is not None
    assert len(result.script.segments) == 1
    assert result.script_version == PODCAST_SCRIPT_VERSION
    assert result.model == PODCAST_MODEL
    assert result.dropped_item_ids == ()
    assert result.dropped_item_count == 0
    assert result.truncated is False
    assert (result.input_tokens, result.output_tokens) == (1_000, 500)
    assert len(calls) == 1


def test_every_turn_is_flattened(monkeypatch: pytest.MonkeyPatch) -> None:
    dirty = _script()
    dirty["intro_turns"][0]["text"] = "Line one\nLine two\t\ttabbed"
    result, _ = _build(monkeypatch, [_result(dirty, sent_item_ids=(1,))])

    assert result.script is not None
    assert result.script.intro_turns[0].text == "Line one Line two tabbed"


def test_a_control_character_only_turn_flattens_and_the_rest_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirty = _script()
    dirty["intro_turns"][0]["text"] = "Before\x07After"
    result, _ = _build(monkeypatch, [_result(dirty, sent_item_ids=(1,))])

    assert result.script is not None
    assert result.script.intro_turns[0].text == "BeforeAfter"


def test_dropped_item_count_is_threaded_onto_the_built_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _build(
        monkeypatch,
        [_result(_script(), sent_item_ids=(1,), dropped_item_count=3)],
    )

    assert result.script is not None
    assert result.dropped_item_count == 3


# --------------------------------------------------------------------------- #
# spend survives every "nothing usable" outcome — NEVER rule 11
# --------------------------------------------------------------------------- #


def test_a_refused_call_still_reports_what_it_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The most likely production instance of the accounting bug an
    architectural review caught: `run_podcast_script` refuses (or its
    response fails schema validation) and returns `script=None` directly —
    no cleaning ever runs, since there is nothing to clean. Real tokens
    were still billed for that response and must still come back."""
    refused = _result(None, sent_item_ids=(1,))
    result, calls = _build(monkeypatch, [refused], item_count=1)

    assert result.script is None
    assert (result.input_tokens, result.output_tokens) == (1_000, 500)
    assert len(calls) == 1


def test_a_retry_cleaned_to_nothing_still_reports_both_calls_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry-side twin of `test_dropping_every_segment_returns_none`:
    the first attempt is over-length and triggers the "shorter" retry, and
    the retry comes back citing only unknown ids — cleaned down to zero
    segments. Both calls were real and billed; the fallback here is
    "nothing usable," not "fall back to the first attempt," because the
    first attempt's own script no longer exists once the retry replaced it
    (see `build_script`'s docstring on why a retry replaces rather than
    merges)."""
    long_script = _script(segment_count=1, chars_per_turn=8_000)
    retry_script = _script(segment_count=1, chars_per_turn=20)
    retry_script["segments"][0]["item_ids"] = [999]
    result, calls = _build(
        monkeypatch,
        [
            _result(long_script, sent_item_ids=(1,)),
            _result(retry_script, sent_item_ids=(1,)),
        ],
        item_count=1,
    )

    assert result.script is None
    assert result.error is not None, (
        "a lost episode now flips the run to `partial` — the reason must be recorded"
    )
    assert len(calls) == 2
    assert (result.input_tokens, result.output_tokens) == (2_000, 1_000)
    assert result.dropped_item_ids == (999,)


# --------------------------------------------------------------------------- #
# citation discipline — NEVER rule 7, applied to item ids
# --------------------------------------------------------------------------- #


def test_a_segment_citing_an_unknown_item_id_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(segment_count=2)
    script["segments"][1]["item_ids"] = [999]  # not among the sent ids
    result, _ = _build(monkeypatch, [_result(script, sent_item_ids=(1, 2))], item_count=2)

    assert result.script is not None
    assert len(result.script.segments) == 1
    assert result.script.segments[0].item_ids == [1]
    assert result.dropped_item_ids == (999,)


def test_a_segment_with_one_unknown_id_among_several_is_dropped_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(segment_count=2)
    script["segments"][1]["item_ids"] = [2, 999]
    result, _ = _build(monkeypatch, [_result(script, sent_item_ids=(1, 2))], item_count=2)

    assert result.script is not None
    assert len(result.script.segments) == 1
    assert result.script.segments[0].item_ids == [1]
    assert result.dropped_item_ids == (999,)


def test_dropping_every_segment_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A script cleaned down to nothing still reports what the call spent —
    the accounting gap an architectural review caught: a bare `None` return
    here used to make a paid Opus call look free (NEVER rule 11). It also
    still reports *which* id was fabricated: the one case where every
    segment cited an unknown id is exactly the case where that record must
    not also be thrown away."""
    script = _script(segment_count=1)
    script["segments"][0]["item_ids"] = [999]
    result, _ = _build(monkeypatch, [_result(script, sent_item_ids=(1,))], item_count=1)

    assert result.script is None
    assert result.error is not None, (
        "a lost episode now flips the run to `partial` — the reason must be recorded"
    )
    assert (result.input_tokens, result.output_tokens) == (1_000, 500)
    assert result.dropped_item_ids == (999,)


def test_a_citation_beyond_sent_item_ids_is_dropped_even_though_it_is_a_real_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The confabulation-safety case an `llm-cost-guard`/`code-reviewer`
    review caught: `known_ids` must come from `PodcastScriptResult.sent_item_ids`
    (what `llm.py`'s `PODCAST_MAX_ITEMS` clamp actually sent), never from
    `build_script`'s own, larger `items` argument. Item 2 here is a
    perfectly real item the caller passed in — it just wasn't among the ids
    the model was actually shown — so citing it is exactly as much a
    fabrication as citing an id that was never real at all.
    """
    script = _script(segment_count=2)
    script["segments"][1]["item_ids"] = [2]
    # sent_item_ids narrower than item_count=2: the model was only shown id 1.
    result, _ = _build(monkeypatch, [_result(script, sent_item_ids=(1,))], item_count=2)

    assert result.script is not None
    assert len(result.script.segments) == 1
    assert result.script.segments[0].item_ids == [1]
    assert result.dropped_item_ids == (2,)


# --------------------------------------------------------------------------- #
# cleaning order — unknown-id segments dropped BEFORE the char cap
# --------------------------------------------------------------------------- #


def test_a_hallucinated_oversized_segment_is_dropped_before_it_can_trigger_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the only thing pushing a script over `PODCAST_MAX_SCRIPT_CHARS` is a
    segment citing an id the model was never given, dropping it first must
    make the script fit — and a script that already fits must never trigger
    a second, real, paid API call."""
    script = _script(segment_count=1, chars_per_turn=20)
    huge_hallucinated = {
        "item_ids": [999],
        "turns": [_turn("A", 10_000), _turn("B", 10_000)],
    }
    script["segments"].append(huge_hallucinated)
    result, calls = _build(monkeypatch, [_result(script, sent_item_ids=(1,))], item_count=1)

    assert result.script is not None
    assert len(calls) == 1  # no retry
    assert result.truncated is False
    assert result.dropped_item_ids == (999,)
    assert len(result.script.segments) == 1


# --------------------------------------------------------------------------- #
# whitespace/control-only turns — degrade, never crash
# --------------------------------------------------------------------------- #


def test_a_whitespace_only_turn_is_dropped_not_reconstructed_into_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`PodcastTurn.text` requires `min_length=1` on the model's *unflattened*
    output, so a turn of pure whitespace passes schema validation and only
    collapses to `""` once `flatten_to_single_line` runs. Reconstructing a
    `PodcastTurn` with that empty text would raise `ValidationError` out of
    `build_script` — this must instead drop the turn and keep going."""
    script = _script(segment_count=1)
    script["segments"][0]["turns"].append({"speaker": "A", "text": "   \t  "})
    result, _ = _build(monkeypatch, [_result(script, sent_item_ids=(1,))], item_count=1)

    assert result.script is not None
    # The original two turns survive; the whitespace-only third does not.
    assert len(result.script.segments[0].turns) == 2


def test_a_segment_left_with_no_turns_after_flattening_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(segment_count=2)
    script["segments"][1]["turns"] = [{"speaker": "A", "text": "\x07\x07"}]
    result, _ = _build(monkeypatch, [_result(script, sent_item_ids=(1, 2))], item_count=2)

    assert result.script is not None
    assert len(result.script.segments) == 1
    assert result.script.segments[0].item_ids == [1]


def test_an_intro_reduced_to_nothing_by_flattening_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(segment_count=1)
    script["intro_turns"] = [{"speaker": "A", "text": "\x07"}, {"speaker": "B", "text": "  "}]
    result, _ = _build(monkeypatch, [_result(script, sent_item_ids=(1,))], item_count=1)

    assert result.script is None
    assert result.error is not None, (
        "a lost episode now flips the run to `partial` — the reason must be recorded"
    )
    assert (result.input_tokens, result.output_tokens) == (1_000, 500)


# --------------------------------------------------------------------------- #
# the cut-off retry — the failure that used to lose an episode in silence
# --------------------------------------------------------------------------- #


def _unfinished(sent_item_ids: tuple[int, ...] = (1,)) -> PodcastScriptResult:
    """What `run_podcast_script` returns when the response hit `max_tokens`."""
    return PodcastScriptResult(
        script=None,
        error="podcast script call was cut off at its 12,288-token output ceiling",
        input_tokens=1_000,
        output_tokens=500,
        sent_item_ids=sent_item_ids,
        unfinished=True,
    )


def test_a_cut_off_first_attempt_retries_and_uses_what_comes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression that cost two real episodes (2026-08-15, 2026-08-17):
    a first attempt truncated at its output ceiling returned no script, and
    `build_script` gave up on the spot. It must instead spend its one retry
    on the single failure mode a "write it shorter" instruction can fix.
    """
    result, calls = _build(
        monkeypatch,
        [_unfinished(), _result(_script(chars_per_turn=20), sent_item_ids=(1,))],
        item_count=1,
    )

    assert result.script is not None
    assert len(calls) == 2
    assert calls[1]["retry_mode"] == "unfinished"
    # Both calls' spend, not just the one that produced something.
    assert (result.input_tokens, result.output_tokens) == (2_000, 1_000)


def test_a_failure_that_is_not_a_cut_off_never_pays_for_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`unfinished` is the whole trigger, not "the script was None". A
    refusal or genuinely malformed JSON would fail identically on a second
    attempt, so retrying it buys nothing and costs an Opus call.
    """
    refused = PodcastScriptResult(
        script=None,
        error="podcast script call was refused by safety classifiers",
        input_tokens=1_000,
        output_tokens=500,
        sent_item_ids=(1,),
    )
    result, calls = _build(monkeypatch, [refused], item_count=1)

    assert result.script is None
    assert len(calls) == 1
    assert result.error == "podcast script call was refused by safety classifiers"


def test_a_cut_off_retry_that_is_also_cut_off_reports_the_retrys_own_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two cut-offs in a row is a lost episode — but a *recorded* one, and
    recorded as the *retry's* failure rather than the first attempt's.

    The gap an `llm-cost-guard` review caught in the first cut of this fix:
    `result` was only reassigned when the retry produced a script, so a
    double cut-off put the first call's reason in `runs.errors` and a run
    that truncated twice looked exactly like one that truncated once. That
    is the single datum that says whether the retry's ceiling is set high
    enough, and the code was spending real money to throw it away.
    """
    first = _unfinished()
    second = PodcastScriptResult(
        script=None,
        error="the retry after a cut-off was cut off at its 8,192-token output ceiling",
        input_tokens=1_000,
        output_tokens=500,
        sent_item_ids=(1,),
        unfinished=True,
    )
    result, calls = _build(monkeypatch, [first, second], item_count=1)

    assert result.script is None
    assert len(calls) == 2
    assert result.error == second.error, "the first attempt's reason must not mask the retry's"
    assert result.error != first.error
    # Both attempts were billed; neither one's spend is lost with the episode.
    assert (result.input_tokens, result.output_tokens) == (2_000, 1_000)


def test_a_cut_off_retry_that_raises_keeps_the_first_calls_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same accounting rule the "shorter" retry's handler follows: an
    `LlmError` means nothing was spent on the retry, but letting it
    propagate would take the first call's real, billed tokens with it
    (NEVER rule 11).
    """
    result, calls = _build(monkeypatch, [_unfinished(), LlmError("network error")], item_count=1)

    assert result.script is None
    assert len(calls) == 2
    assert (result.input_tokens, result.output_tokens) == (1_000, 500)


def test_a_cut_off_run_never_also_pays_for_the_shorter_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The money guard. A run whose first attempt was cut off and whose
    retry came back *over the char cap* must not then make a third Opus
    call — `PODCAST_MONTHLY_CEILING_USD` is priced on one first call plus at
    most one retry, and a third would silently break the ceiling this whole
    feature's budget rests on. The over-long retry is truncated locally
    instead, which costs nothing.
    """
    long_script = _script(segment_count=5, chars_per_turn=2_000)
    result, calls = _build(
        monkeypatch,
        [
            _unfinished(sent_item_ids=_all_ids(5)),
            _result(long_script, sent_item_ids=_all_ids(5)),
            _result(_script(segment_count=5, chars_per_turn=20), sent_item_ids=_all_ids(5)),
        ],
        item_count=5,
    )

    assert result.script is not None
    assert len(calls) == 2  # never the third response, which was there to be taken
    assert result.truncated is True


# --------------------------------------------------------------------------- #
# the char cap — one "shorter" retry, then truncate at a segment boundary
# --------------------------------------------------------------------------- #


def test_a_short_script_never_triggers_a_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = _build(monkeypatch, [_result(_script(chars_per_turn=20), sent_item_ids=(1,))])

    assert result.script is not None
    assert len(calls) == 1


def test_a_retry_that_itself_produces_nothing_falls_back_to_truncating_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the "shorter" retry fails to parse (or is refused), `build_script`
    must fall back to the *first* attempt's script — over-long as it is —
    and truncate that, rather than treating a failed retry as "nothing to
    build." This path is easy to invert silently: the fallback is a plain
    `if retry.script is not None`, and nothing else in this suite exercises
    the `False` branch.
    """
    long_script = _script(segment_count=5, chars_per_turn=2_000)
    failed_retry = PodcastScriptResult(
        script=None, error="schema validation failed", sent_item_ids=_all_ids(5)
    )
    result, calls = _build(
        monkeypatch,
        [_result(long_script, sent_item_ids=_all_ids(5)), failed_retry],
        item_count=5,
    )

    assert result.script is not None
    assert len(calls) == 2
    assert result.truncated is True
    # The content is the *first* script's (2,000 chars/turn), truncated —
    # not empty, and not something invented from the failed retry.
    assert result.script.intro_turns[0].text == "x" * 2_000


def test_a_retry_that_raises_before_a_response_falls_back_without_losing_the_first_calls_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accounting bug an architectural review caught: `run_podcast_script`
    only raises `LlmError` for a failure *before* any response comes back
    (auth, network, the create call itself) — nothing was spent on the
    retry, but the first call's tokens are real and already billed. Letting
    the exception propagate out of `build_script` would take that record
    down with it. This must instead fall back to the first attempt's
    (already-cleaned, non-empty) script, truncated as needed, with both
    calls' spend intact — auth/network failures happen on any call, not
    just the first.
    """
    long_script = _script(segment_count=5, chars_per_turn=2_000)
    result, calls = _build(
        monkeypatch,
        [_result(long_script, sent_item_ids=_all_ids(5)), LlmError("network error")],
        item_count=5,
    )

    assert result.script is not None
    assert len(calls) == 2
    assert result.truncated is True
    assert result.script.intro_turns[0].text == "x" * 2_000
    # Only the first call's spend — the retry raised before billing anything.
    assert (result.input_tokens, result.output_tokens) == (1_000, 500)


def test_an_overlong_script_retries_once_and_uses_the_shorter_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_script = _script(segment_count=1, chars_per_turn=8_000)  # well over the cap
    short_script = _script(segment_count=1, chars_per_turn=20)  # comfortably under
    result, calls = _build(
        monkeypatch,
        [
            _result(long_script, sent_item_ids=(1,)),
            _result(short_script, sent_item_ids=(1,)),
        ],
        item_count=1,
    )

    assert result.script is not None
    assert len(calls) == 2
    assert calls[1]["retry_mode"] == "shorter"
    assert result.truncated is False
    # The result is the *retry's* content — each turn is 20 chars, not 8000.
    assert len(result.script.intro_turns[0].text) == 20


def test_a_retry_that_is_still_too_long_gets_truncated_at_a_segment_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_script = _script(segment_count=5, chars_per_turn=2_000)
    result, calls = _build(
        monkeypatch,
        [
            _result(long_script, sent_item_ids=_all_ids(5)),
            _result(long_script, sent_item_ids=_all_ids(5)),
        ],
        item_count=5,
    )

    assert result.script is not None
    assert len(calls) == 2  # no third API call for truncation
    assert result.truncated is True
    total_chars = sum(
        len(turn.text)
        for turn in (
            *result.script.intro_turns,
            *(t for seg in result.script.segments for t in seg.turns),
            *result.script.outro_turns,
        )
    )
    assert total_chars <= PODCAST_MAX_SCRIPT_CHARS
    assert len(result.script.segments) < 5  # at least one segment was dropped to fit


def test_truncation_drops_segments_from_the_end_not_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_script = _script(segment_count=5, chars_per_turn=2_000)
    result, _ = _build(
        monkeypatch,
        [
            _result(long_script, sent_item_ids=_all_ids(5)),
            _result(long_script, sent_item_ids=_all_ids(5)),
        ],
        item_count=5,
    )

    assert result.script is not None
    kept_ids = [seg.item_ids[0] for seg in result.script.segments]
    # Segments are 1..5 in order; truncating from the end keeps a prefix.
    assert kept_ids == list(range(1, len(kept_ids) + 1))


# --------------------------------------------------------------------------- #
# truncation, phase 1 — teaser an over-length segment before dropping it
# --------------------------------------------------------------------------- #


def test_an_overlong_trailing_segment_is_teasered_not_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator's own read-test finding: a lowest-ranked story that
    overflows the cap should still get a mention, not vanish outright.
    One segment with far more than `_TEASER_TURN_COUNT` turns is, on its
    own, enough to blow the cap — which always triggers the "shorter"
    retry first (`build_script`'s cap check runs before any truncation),
    so this scripts a second, still-over-length response for the retry
    and asserts the fallback teasers it down instead of dropping it."""
    long_segment_script = _script(chars_per_turn=2_000, turns_per_segment=10)
    result, calls = _build(
        monkeypatch,
        [
            _result(long_segment_script, sent_item_ids=(1,)),
            _result(long_segment_script, sent_item_ids=(1,)),
        ],
        item_count=1,
    )

    assert result.script is not None
    assert len(calls) == 2  # first attempt + the mandatory "shorter" retry
    assert len(result.script.segments) == 1  # teasered, not dropped
    assert len(result.script.segments[0].turns) == 2
    assert result.script.segments[0].item_ids == [1]  # citation survives the teaser
    assert result.truncated is True
    assert result.teasered_item_ids == (1,)
    assert result.truncation_dropped_item_ids == ()


def test_multiple_overlong_segments_are_all_teasered_before_any_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teasering happens from the end, one segment at a time, stopping as
    soon as the script fits — so with several over-length segments, the
    earliest (highest-ranked) ones may survive untouched while only the
    tail gets shrunk. All five items should keep at least a mention.

    Calibrated so the math is checkable by hand — `_script`'s intro/outro
    use `chars_per_turn` too, so they count: 5 segments x 4 turns x 800
    chars = 16,000, plus a 4-turn intro+outro at 800 chars = 3,200, is
    19,200, over the 14,000 cap. Walking from the end, segments 5-2
    teasered (4 -> 2 turns each, -1,600 chars apiece) brings the total to
    12,800 before segment 1 is even reached, so segment 1 is the one
    left full.
    """
    script = _script(segment_count=5, chars_per_turn=800, turns_per_segment=4)
    result, _ = _build(
        monkeypatch,
        [
            _result(script, sent_item_ids=_all_ids(5)),
            _result(script, sent_item_ids=_all_ids(5)),
        ],
        item_count=5,
    )

    assert result.script is not None
    assert result.truncated is True
    # Every item still has a segment — nothing was dropped, only shrunk.
    kept_ids = [seg.item_ids[0] for seg in result.script.segments]
    assert kept_ids == [1, 2, 3, 4, 5]
    # The highest-ranked segment (walked to last, from the end) survives
    # full; the rest were teasered down to fit.
    assert len(result.script.segments[0].turns) == 4
    assert all(len(seg.turns) == 2 for seg in result.script.segments[1:])
    assert result.teasered_item_ids == (2, 3, 4, 5)
    assert result.truncation_dropped_item_ids == ()
    total_chars = sum(
        len(turn.text)
        for turn in (
            *result.script.intro_turns,
            *(t for seg in result.script.segments for t in seg.turns),
            *result.script.outro_turns,
        )
    )
    assert total_chars <= PODCAST_MAX_SCRIPT_CHARS


def test_teasering_every_segment_still_over_cap_falls_back_to_dropping_from_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 2 — reached only once every segment is already a teaser and
    the script still doesn't fit. The tail should then be dropped entirely
    (same as the pre-teaser behavior), rather than looping forever or
    shrinking a segment below `_TEASER_TURN_COUNT`.

    Calibrated so both phases genuinely fire: 10 segments x 4 turns x
    1,000 chars = 40,000, plus a 4-turn intro+outro at 1,000 chars each =
    4,000, is 44,000, over the cap. Even fully teasered (10 x 2 x 1,000 =
    20,000, the intro/outro's 4,000 unchanged) it is 24,000 — *still* over
    the 14,000 cap, so phase 1 must shrink every segment and phase 2 must
    then drop some of the now-teasered survivors from the end: 5 pops at
    2,000 chars each lands exactly at 14,000, so 5 segments survive.
    """
    script = _script(segment_count=10, chars_per_turn=1_000, turns_per_segment=4)
    result, _ = _build(
        monkeypatch,
        [
            _result(script, sent_item_ids=_all_ids(10)),
            _result(script, sent_item_ids=_all_ids(10)),
        ],
        item_count=10,
    )

    assert result.script is not None
    assert result.truncated is True
    assert len(result.script.segments) < 10  # some segments genuinely dropped
    # Every survivor is a teaser — phase 1 shrank everything before phase 2
    # dropped any of them.
    assert all(len(seg.turns) == 2 for seg in result.script.segments)
    # The 5 lowest-ranked items were dropped outright by phase 2, not merely
    # teasered — a survivor is never double-counted as both.
    assert result.truncation_dropped_item_ids == (6, 7, 8, 9, 10)
    assert result.teasered_item_ids == (1, 2, 3, 4, 5)
    total_chars = sum(
        len(turn.text)
        for turn in (
            *result.script.intro_turns,
            *(t for seg in result.script.segments for t in seg.turns),
            *result.script.outro_turns,
        )
    )
    assert total_chars <= PODCAST_MAX_SCRIPT_CHARS
    # Every surviving segment is still a valid prefix in rank order.
    kept_ids = [seg.item_ids[0] for seg in result.script.segments]
    assert kept_ids == list(range(1, len(kept_ids) + 1))


def test_token_accounting_sums_the_initial_call_and_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_script = _script(segment_count=1, chars_per_turn=8_000)
    short_script = _script(segment_count=1, chars_per_turn=20)
    result, _ = _build(
        monkeypatch,
        [
            _result(long_script, sent_item_ids=(1,)),
            _result(short_script, sent_item_ids=(1,)),
        ],
        item_count=1,
    )

    assert result.script is not None
    # Each fake response reports 1_000/500 — the retry's spend must not vanish.
    assert (result.input_tokens, result.output_tokens) == (2_000, 1_000)
