"""Tests for `llm.run_podcast_script` — the daily script call (DESIGN §13.3).

The client is faked at `llm.py`'s boundary; the real Anthropic API is never
touched (CLAUDE.md §8, NEVER rule 13). Responses are real SDK `Message`/
`Usage`/`TextBlock` types, the same discipline `test_llm_scout.py` uses, so a
duck-typed stand-in can't pass a test the real API would fail.

What these tests protect:

* **The spend caps.** `PODCAST_MAX_ITEMS` and `PODCAST_MAX_ITEM_CONTENT_BYTES`
  clamp the item payload before a request is ever built, and the worst-case
  arithmetic (mirroring `test_llm_scout.py`'s) prices **two** calls per day —
  the first attempt and the "shorter" retry the code permits on every run,
  not an occasional exception — and proves that stays under
  `PODCAST_MONTHLY_CEILING_USD`.
* **The citation boundary (NEVER rule 7).** `sent_item_ids` on the result is
  the set the model actually saw, distinct from whatever larger list the
  caller passed in — this is what lets `synth.podcast.build_script` catch a
  segment citing an id that was clamped away before the request was built.
* **The cache boundary (NEVER rule 10).** The stable prefix never varies with
  the date — proven directly against `run_podcast_script`'s own request
  shape, not just against `build_podcast_stable_prefix`'s signature.
* **The accounting (NEVER rule 11).** Usage is recorded even when the
  response fails to parse — a malformed response still spent output tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from anthropic import APIError
from anthropic.types import Message, TextBlock, Usage

from signalforge.config import PODCAST_PRESENTER_NAME_MAX_LENGTH, load_interests
from signalforge.llm import (
    PODCAST_MAX_ITEM_CONTENT_BYTES,
    PODCAST_MAX_ITEM_SUMMARY_BYTES,
    PODCAST_MAX_ITEM_TITLE_BYTES,
    PODCAST_MAX_ITEMS,
    PODCAST_MAX_TOKENS,
    PODCAST_MODEL,
    PODCAST_MONTHLY_CEILING_USD,
    PODCAST_RETRY_MAX_TOKENS,
    PODCAST_TRUNCATION_RETRY_MAX_TOKENS,
    LlmError,
    run_podcast_script,
)
from signalforge.synth.podcast import build_podcast_stable_prefix

STABLE_PREFIX = "You write a two-presenter podcast."
DATE_LABEL = "Friday, 7 August 2026"
PRESENTER_A = "Alex"
PRESENTER_B = "Sam"

_VALID_SCRIPT: dict[str, Any] = {
    "intro_turns": [
        {"speaker": "A", "text": "Welcome back to the show."},
        {"speaker": "B", "text": "Let's get into it."},
    ],
    "segments": [
        {
            "item_ids": [1],
            "turns": [
                {"speaker": "A", "text": "First up, a new agent framework shipped."},
                {"speaker": "B", "text": "What's the guardrail story this time?"},
            ],
        }
    ],
    "outro_turns": [
        {"speaker": "A", "text": "That's it for today."},
        {"speaker": "B", "text": "See you tomorrow."},
    ],
}


def _items(count: int = 1) -> list[tuple[int, str, str | None, str | None]]:
    return [
        (item_id, f"Title {item_id}", f"Summary {item_id}", f"Content {item_id}")
        for item_id in range(1, count + 1)
    ]


def _message(
    *,
    script: dict[str, Any] | None = _VALID_SCRIPT,
    stop_reason: str = "end_turn",
    input_tokens: int = 1_000,
    output_tokens: int = 500,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    raw_text: str | None = None,
    no_text: bool = False,
) -> Message:
    """A real `Message`, shaped like a structured-output podcast response."""
    content: list[Any] = []
    if not no_text:
        text = raw_text if raw_text is not None else json.dumps(script)
        content.append(TextBlock(text=text, type="text"))
    return Message(
        id="msg_podcast",
        content=content,
        model=PODCAST_MODEL,
        role="assistant",
        type="message",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )


class FakeMessages:
    """Records every request and replays a scripted list of responses."""

    def __init__(self, responses: list[Message | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.requests.append(kwargs)
        if not self._responses:  # pragma: no cover - a test scripted too few
            raise AssertionError("run_podcast_script made more calls than scripted")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class FakeClient:
    def __init__(self, responses: list[Message | Exception]) -> None:
        self.messages = FakeMessages(responses)


def _run(responses: list[Message | Exception], **overrides: Any) -> tuple[Any, FakeClient]:
    client = FakeClient(responses)
    kwargs: dict[str, Any] = {
        "stable_system_prefix": STABLE_PREFIX,
        "date_label": DATE_LABEL,
        "presenter_a": PRESENTER_A,
        "presenter_b": PRESENTER_B,
        "items": _items(1),
        "client": client,
    }
    kwargs.update(overrides)
    result = run_podcast_script(**kwargs)
    return result, client


def _user_blocks(client: FakeClient, request_index: int = 0) -> list[dict[str, Any]]:
    content = client.messages.requests[request_index]["messages"][0]["content"]
    assert isinstance(content, list), "the user turn must be a list of content blocks"
    return content


def _base_text(client: FakeClient, request_index: int = 0) -> str:
    return str(_user_blocks(client, request_index)[0]["text"])


# --------------------------------------------------------------------------- #
# the happy path and the request shape
# --------------------------------------------------------------------------- #


def test_a_valid_response_becomes_a_validated_script() -> None:
    result, _ = _run([_message()])

    assert result.error is None
    assert result.script is not None
    assert len(result.script.segments) == 1
    assert result.script.segments[0].item_ids == [1]
    assert result.input_tokens == 1_000
    assert result.output_tokens == 500
    assert result.sent_item_ids == (1,)


def test_the_system_block_carries_cache_control_and_the_stable_prefix_verbatim() -> None:
    _, client = _run([_message()])

    system = client.messages.requests[0]["system"]
    assert system == [
        {
            "type": "text",
            "text": STABLE_PREFIX,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_the_user_turn_is_a_single_uncached_block_by_default() -> None:
    _, client = _run([_message()])

    blocks = _user_blocks(client)
    assert len(blocks) == 1
    assert "cache_control" not in blocks[0]


def test_the_user_turn_carries_date_presenters_and_items() -> None:
    _, client = _run([_message()])

    text = _base_text(client)
    assert DATE_LABEL in text
    assert PRESENTER_A in text
    assert PRESENTER_B in text
    assert '"item_id": 1' in text


def test_output_config_carries_effort_and_the_json_schema_format() -> None:
    _, client = _run([_message()])

    output_config = client.messages.requests[0]["output_config"]
    assert output_config["effort"] == "medium"
    assert output_config["format"]["type"] == "json_schema"


def test_uses_podcast_model_and_max_tokens_by_default() -> None:
    _, client = _run([_message()])

    request = client.messages.requests[0]
    assert request["model"] == PODCAST_MODEL
    assert request["max_tokens"] == PODCAST_MAX_TOKENS


# --------------------------------------------------------------------------- #
# the item clamp (PODCAST_MAX_ITEMS / PODCAST_MAX_ITEM_CONTENT_BYTES) — the
# money limit config cannot exceed
# --------------------------------------------------------------------------- #


def test_items_beyond_the_ceiling_are_clamped_and_recorded() -> None:
    result, client = _run([_message()], items=_items(PODCAST_MAX_ITEMS + 5))

    assert result.dropped_item_count == 5
    assert result.sent_item_ids == tuple(range(1, PODCAST_MAX_ITEMS + 1))
    # `json.dumps` with no `indent=` produces no embedded newlines, so the
    # JSON payload the prompt builder appends is always the text's last line.
    payload = json.loads(_base_text(client).rsplit("\n", 1)[-1])
    assert {entry["item_id"] for entry in payload} == set(range(1, PODCAST_MAX_ITEMS + 1))


def test_items_at_or_below_the_ceiling_are_not_clamped() -> None:
    result, _ = _run([_message()], items=_items(PODCAST_MAX_ITEMS))

    assert result.dropped_item_count == 0
    assert result.sent_item_ids == tuple(range(1, PODCAST_MAX_ITEMS + 1))


def test_an_item_beyond_the_clamp_is_never_a_legitimate_citation() -> None:
    """The confabulation-safety case: an item the caller passed in but that
    got clamped away must not appear in `sent_item_ids`, even though it is a
    perfectly real item from the caller's point of view — the model never
    saw it, so a segment citing it later is a fabrication, not a citation
    that missed the cut."""
    result, _ = _run([_message()], items=_items(PODCAST_MAX_ITEMS + 1))

    last_item_id = PODCAST_MAX_ITEMS + 1
    assert last_item_id not in result.sent_item_ids


def test_item_content_is_truncated_to_the_prompt_level_cap() -> None:
    long_content = "x" * (PODCAST_MAX_ITEM_CONTENT_BYTES + 5_000)
    items = [(1, "Title", "Summary", long_content)]
    _, client = _run([_message()], items=items)

    payload = json.loads(_base_text(client).rsplit("\n", 1)[-1])
    assert len(payload[0]["content"].encode("utf-8")) == PODCAST_MAX_ITEM_CONTENT_BYTES


def test_non_ascii_content_is_capped_in_bytes_not_characters() -> None:
    """The bug an `llm-cost-guard` review caught: a character-count cap
    bounds Python string length, not what gets billed. A CJK character is
    ~3 UTF-8 bytes; capping *characters* would let a non-Latin-script feed
    cost several times the intended byte budget once encoded, and
    `json.dumps`'s default `ensure_ascii=True` would inflate it again on
    top of that (see the next test). This asserts the *byte* length is at
    the cap, not the character count."""
    long_content = "日" * (PODCAST_MAX_ITEM_CONTENT_BYTES)  # 3 bytes/char — way over budget
    items = [(1, "Title", "Summary", long_content)]
    _, client = _run([_message()], items=items)

    payload = json.loads(_base_text(client).rsplit("\n", 1)[-1])
    truncated_content = payload[0]["content"]
    assert len(truncated_content.encode("utf-8")) <= PODCAST_MAX_ITEM_CONTENT_BYTES
    # Truncation lands on a whole character — decoding never raised, and
    # every character present really is "日" (no mangled trailing bytes).
    assert set(truncated_content) <= {"日"}
    assert len(truncated_content) > 0


def test_non_ascii_json_is_not_escaped_back_past_the_cap() -> None:
    """`ensure_ascii=True` (json.dumps's default) would re-encode every "日"
    as the 6-character escape `\\u65e5` — undoing `_truncate_utf8_json_safe`'s
    cap the moment the payload is serialized. This asserts the literal
    character reaches the rendered prompt text, not an escape sequence."""
    items = [(1, "Title", "Summary", "日本語のコンテンツ")]
    _, client = _run([_message()], items=items)

    text = _base_text(client)
    assert "日本語のコンテンツ" in text
    assert "\\u" not in text


def test_control_character_content_is_capped_after_json_escaping() -> None:
    """The bug a later `llm-cost-guard` review caught: even with
    `ensure_ascii=False`, `json.dumps` still escapes `"`, `\\`, and every C0
    control character — a raw `\\x01` becomes the 6-byte `\\u0001`. A cap
    enforced on *raw* UTF-8 bytes, before that escaping happens, would let
    this field cost up to 6x its intended size once serialized. This
    asserts the cap holds against the actual bytes reaching the rendered
    prompt text, not the pre-escaping byte count."""
    long_content = "\x01" * (PODCAST_MAX_ITEM_CONTENT_BYTES * 2)
    items = [(1, "Title", "Summary", long_content)]
    _, client = _run([_message()], items=items)

    text = _base_text(client)
    # The rendered prompt text is what gets tokenized and billed — assert
    # the *escaped* representation of this item's content, not some
    # unescaped intermediate, stays within budget.
    payload = json.loads(text.rsplit("\n", 1)[-1])
    content_field = payload[0]["content"]
    escaped_len = len(json.dumps(content_field, ensure_ascii=False).encode("utf-8")) - 2
    assert escaped_len <= PODCAST_MAX_ITEM_CONTENT_BYTES
    assert len(content_field) > 0


# --------------------------------------------------------------------------- #
# the "shorter" retry — a cheaper second call, never a cached one
# --------------------------------------------------------------------------- #


def test_shorter_appends_the_rewrite_instruction_as_a_separate_uncached_block() -> None:
    _, plain_client = _run([_message()])
    _, shorter_client = _run([_message()], retry_mode="shorter")

    blocks = _user_blocks(shorter_client)
    assert len(blocks) == 2
    assert "cache_control" not in blocks[0]
    assert "cache_control" not in blocks[1]
    assert "shorter" in str(blocks[1]["text"]).lower()
    # The base block (date/presenters/items) is byte-identical to a plain
    # call's only block — a retry mode only appends, never rewrites it.
    assert blocks[0]["text"] == _base_text(plain_client)


def test_shorter_keeps_the_same_system_prefix() -> None:
    _, client = _run([_message()], retry_mode="shorter")

    assert client.messages.requests[0]["system"][0]["text"] == STABLE_PREFIX


def test_shorter_uses_the_smaller_retry_token_ceiling() -> None:
    _, client = _run([_message()], retry_mode="shorter")

    assert client.messages.requests[0]["max_tokens"] == PODCAST_RETRY_MAX_TOKENS


# --------------------------------------------------------------------------- #
# the cut-off response — reported as its own reason, not as a parse failure
# --------------------------------------------------------------------------- #


def test_a_response_cut_off_at_the_ceiling_is_reported_as_unfinished() -> None:
    """`stop_reason="max_tokens"` is checked before parsing. Truncated JSON
    fails schema validation anyway, so the old code reached "no script" by
    accident — but recorded it as "schema validation failed", which reads as
    a model that wrote something malformed rather than one that ran out of
    room, and left the caller nothing to base a retry decision on. Two real
    episodes were lost that way.
    """
    result, _ = _run(
        [_message(raw_text='{"intro_turns": [{"speaker": "A", "te', stop_reason="max_tokens")]
    )

    assert result.script is None
    assert result.unfinished is True
    assert result.error is not None
    assert "cut off" in result.error
    # Billed for what it wrote, cut off or not (NEVER rule 11).
    assert (result.input_tokens, result.output_tokens) == (1_000, 500)


def test_an_ordinary_parse_failure_is_not_reported_as_unfinished() -> None:
    """The flag has to actually discriminate: malformed JSON that came back
    complete is not a cut-off, and must not trigger the retry that only a
    cut-off justifies paying for."""
    result, _ = _run([_message(raw_text="not json at all")])

    assert result.script is None
    assert result.unfinished is False


def test_a_refusal_is_not_reported_as_unfinished() -> None:
    result, _ = _run([_message(stop_reason="refusal")])

    assert result.script is None
    assert result.unfinished is False


def test_the_unfinished_retry_gets_more_room_than_the_shorter_one() -> None:
    """The point of a separate ceiling. Recovering from "ran out of room" with
    *less* room than the attempt that ran out would truncate again, having
    spent a real Opus call to do it."""
    _, client = _run([_message()], retry_mode="unfinished")

    assert client.messages.requests[0]["max_tokens"] == PODCAST_TRUNCATION_RETRY_MAX_TOKENS
    assert PODCAST_TRUNCATION_RETRY_MAX_TOKENS > PODCAST_RETRY_MAX_TOKENS


def test_the_unfinished_retry_tells_the_model_it_was_cut_off_not_that_it_ran_long() -> None:
    """Two instructions, not one shared. "Your previous script ran long" is
    simply false here — the model never finished, and does not know how long
    the script would have been."""
    _, client = _run([_message()], retry_mode="unfinished")

    blocks = _user_blocks(client)
    assert len(blocks) == 2
    assert "cut off" in str(blocks[1]["text"]).lower()
    assert "ran long" not in str(blocks[1]["text"]).lower()
    assert "cache_control" not in blocks[1]


# --------------------------------------------------------------------------- #
# degradation — never raises once a response has come back
# --------------------------------------------------------------------------- #


def test_malformed_json_is_recorded_with_usage_still_counted() -> None:
    result, _ = _run([_message(raw_text="not json", input_tokens=900, output_tokens=200)])

    assert result.script is None
    assert result.error is not None
    assert "schema validation failed" in result.error
    assert (result.input_tokens, result.output_tokens) == (900, 200)


def test_no_text_content_is_recorded_rather_than_raised() -> None:
    result, _ = _run([_message(no_text=True)])

    assert result.script is None
    assert result.error == "no text content in podcast script response"


def test_a_refusal_is_recorded_rather_than_raised() -> None:
    result, _ = _run([_message(script=None, stop_reason="refusal", no_text=True)])

    assert result.script is None
    assert result.error is not None
    assert "refused" in result.error


def test_zero_segments_is_recorded_as_an_error_not_a_crash() -> None:
    empty = {**_VALID_SCRIPT, "segments": []}
    result, _ = _run([_message(script=empty)])

    assert result.script is None
    assert result.error == "podcast script had no segments"


def test_an_unknown_speaker_fails_schema_validation() -> None:
    bad = json.loads(json.dumps(_VALID_SCRIPT))
    bad["segments"][0]["turns"][0]["speaker"] = "C"
    result, _ = _run([_message(script=bad)])

    assert result.script is None
    assert result.error is not None
    assert "schema validation failed" in result.error


def _api_error() -> APIError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return APIError("boom", request=request, body=None)


def test_a_failed_request_raises_llm_error_before_anything_is_spent() -> None:
    """Unlike the scout, nothing here can have been billed before the response
    comes back — this is a plain structured-output call, no server tools —
    so raising (rather than recording) is correct: there is no spend an
    exception could hide (NEVER rule 11 does not apply here the way it does
    to the scout's server-side search)."""
    with pytest.raises(LlmError, match="failed to generate podcast script"):
        run_podcast_script(
            STABLE_PREFIX,
            date_label=DATE_LABEL,
            presenter_a=PRESENTER_A,
            presenter_b=PRESENTER_B,
            items=_items(1),
            client=FakeClient([_api_error()]),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# NEVER rule 10 — no volatile data in the cached prefix
# --------------------------------------------------------------------------- #


def test_the_stable_prefix_signature_cannot_accept_a_date_or_run_id(
    repo_config_dir: Path,
) -> None:
    """`build_podcast_stable_prefix` takes only `interests` — no date/run-id
    parameter exists to pass one through, so byte-identity across calls is
    provable by construction. This is a necessary condition; the sufficient
    one — that `run_podcast_script` itself never lets a date leak into the
    cached block regardless of what it's given — is
    `test_two_dates_produce_the_byte_identical_system_block` below."""
    interests = load_interests(repo_config_dir)

    first = build_podcast_stable_prefix(interests)
    second = build_podcast_stable_prefix(interests)

    assert first == second


def test_the_stable_prefix_forbids_the_intro_and_outro_from_previewing_stories(
    repo_config_dir: Path,
) -> None:
    """`intro_turns`/`outro_turns` carry no `item_ids` and are never
    validated against `sent_item_ids` — unlike a segment, nothing stops a
    dropped citation or a length-truncated segment from leaving the intro
    describing content that never aired. A code-reviewer pass on Stage 3
    flagged this as a real gap in NEVER rule 7's citation discipline; the
    chosen fix is a prompt constraint (v2 of the prefix), not a runtime
    check, so this test pins the constraint's actual wording rather than
    behavior no unit test can observe without a live model."""
    interests = load_interests(repo_config_dir)

    prefix = build_podcast_stable_prefix(interests)

    assert "greeting and sign-off ONLY" in prefix
    assert "never a preview or recap of specific stories" in prefix


def test_two_dates_produce_the_byte_identical_system_block() -> None:
    """The real gate: call `run_podcast_script` itself with two different
    dates and prove the cached system block does not move. Unlike testing
    `build_podcast_stable_prefix` alone, this fails if a future edit ever
    threads `date_label` (or anything else volatile) into the system
    argument rather than the user turn."""
    _, client = _run([_message(), _message()], date_label="Monday, 3 August 2026")
    run_podcast_script(
        STABLE_PREFIX,
        date_label="Tuesday, 4 August 2026",
        presenter_a=PRESENTER_A,
        presenter_b=PRESENTER_B,
        items=_items(1),
        client=client,  # type: ignore[arg-type]
    )

    assert client.messages.requests[0]["system"] == client.messages.requests[1]["system"]
    # The dates *did* land somewhere — in the user turn, not the system block.
    assert "3 August" in _base_text(client, 0)
    assert "4 August" in _base_text(client, 1)


# --------------------------------------------------------------------------- #
# the worst-case cost, computed rather than assumed (mirrors test_llm_scout.py)
# --------------------------------------------------------------------------- #

IN_RATE_USD_PER_TOKEN = 5 / 1e6
OUT_RATE_USD_PER_TOKEN = 25 / 1e6
CACHE_WRITE_MULTIPLIER = 1.25
"""Ephemeral cache writes cost ~1.25x base input price, applied only to the
system prefix — the user turn (items, date, presenters) is never cached
(NEVER rule 10; see `run_podcast_script`'s docstring for why caching it was
deliberately rejected even though it would make the retry cheaper)."""

CALLS_PER_DAY = 2
"""**Not** an assumption — this is what the code in this file actually
permits: `synth.podcast.build_script` always makes the "shorter" retry when
a script runs long, and nothing bounds how often that happens. Pricing the
worst case at 1 call/day was the mistake an `llm-cost-guard` review caught
here — the code allows 2 every single day, so the ceiling must be priced
against 2, not against how often a retry is expected to be needed."""

CALLS_PER_MONTH = 30 * CALLS_PER_DAY
"""Daily cadence, rounded down from ~30.44 — conservative in the direction
that matters less here (fewer days undercounts monthly cost slightly);
kept a round number for readability, unlike `WEEKS_PER_MONTH`'s precise
4.33 in `test_llm_scout.py`, where the multiplier's precision mattered more
against a tighter ceiling."""

BYTES_PER_TOKEN = 1.5
"""Pessimistic UTF-8-bytes-per-token estimate — **bytes, not characters**,
and *not* the round-trip-optimistic 3.0 an earlier version of this test
used. That earlier value came from measuring a fixture made entirely of one
3-byte-per-character CJK ideograph and dividing bytes by that character
count — which silently assumes 1 token per CJK character, the *best* case
for CJK tokenization, not a pessimistic one. A fourth `llm-cost-guard`
review caught this: real non-Latin tokenization is documented to run
*denser* than 1 token/character for common scripts (~1.4-1.5 tokens/char
for Chinese in comparable BPE tokenizers) and considerably denser for
Hangul, Devanagari, and Thai (~2 tokens/char is the documented bad case).
At 3 bytes/character and ~2 tokens/character, that is 1.5 bytes/token —
this constant. There is no live tokenizer to measure against (NEVER rule
13), so this is reasoned from documented tokenizer behavior on comparable
BPE vocabularies, the same "no code enforces this, price it with margin
above the observed/documented worst, not at it" posture
`PESSIMISTIC_TOKENS_PER_SEARCH` (`test_llm_scout.py`) uses — and like that
constant, it should be replaced with a grounded figure once a real
Stage 3 run reports actual `input_tokens` for non-English content.

**This is the ceiling's thin edge**, flagged by a sixth `llm-cost-guard`
review, re-measured by a seventh after the intro/outro citation-discipline
prompt fix (`PODCAST_SCRIPT_VERSION` "podcast-v2") added ~340 bytes to the
stable prefix, and re-measured again by an eighth when
`PODCAST_MAX_TOKENS` was raised. `PODCAST_MONTHLY_CEILING_USD`'s headline
margin over the worst case computed at this ratio is not the margin that
matters — the ratio itself is the uncertain input, and the worst case is
sensitive to it.

**The sensitivity figures live in `PODCAST_MONTHLY_CEILING_USD`'s docstring
and are deliberately not repeated here.** An earlier version of this
docstring carried its own copy, which went stale the moment the ceiling
moved and then contradicted the canonical one — breaking, in the constant
whose whole purpose is pricing discipline, the rule DESIGN §8 states in so
many words: a number written in two places is a number that will disagree
with itself. The eighth review caught the copy still quoting $20.63 against
a $23 ceiling that no longer existed.

Two consequences that *do* belong here, because they are about this
constant rather than about the ceiling: grounding it from a real run's
actual `input_tokens` is overdue rather than aspirational — there are now
seven real runs to ground it against, and English content runs nearer 4
bytes/token than 1.5, so the honest figure would likely *lower* the computed
worst case substantially. And the stable prefix is not a free surface to
extend casually — a future addition of even a few hundred bytes should be
priced against the sensitivity table before it ships, not after."""


def _worst_case_item_tuples() -> list[tuple[int, str, str | None, str | None]]:
    """`PODCAST_MAX_ITEMS` items, every field padded past its byte cap with
    **adversarial** filler: the C0 control character `\\x01`, which
    `json.dumps` escapes to the 6-byte `\\u0001` regardless of
    `ensure_ascii`. This is what actually exercises
    `_truncate_utf8_json_safe`'s escape-aware bisection — a CJK-only fixture
    (what an earlier round of this test used) escapes to nothing extra
    under `ensure_ascii=False` and so cannot catch the escaping bug a later
    `llm-cost-guard` review found: per-field caps enforced on raw UTF-8
    bytes, measured *before* `json.dumps` escapes `"`, `\\`, and control
    characters, undercounted the bytes actually billed by as much as 6x for
    exactly this filler shape.

    Filler length is derived from the *largest* of the three per-field
    caps, not `PODCAST_MAX_ITEM_CONTENT_BYTES` alone — an earlier version
    hardcoded the content cap, which would silently under-measure if
    `PODCAST_MAX_ITEM_TITLE_BYTES` or `PODCAST_MAX_ITEM_SUMMARY_BYTES` were
    ever raised above it.
    """
    filler_chars = (
        max(
            PODCAST_MAX_ITEM_TITLE_BYTES,
            PODCAST_MAX_ITEM_SUMMARY_BYTES,
            PODCAST_MAX_ITEM_CONTENT_BYTES,
        )
        * 2
    )
    filler = "\x01" * filler_chars  # every field over-length even after escaping
    return [(item_id, filler, filler, filler) for item_id in range(1, PODCAST_MAX_ITEMS + 1)]


def _worst_case_monthly_usd(
    prefix_tokens: int, item_payload_tokens: int, retry_instruction_tokens: int
) -> float:
    """Prices one day as *two* calls: the first attempt at `PODCAST_MAX_TOKENS`
    output, and one retry — both paying full, uncached price for the identical
    item payload, since NEVER rule 10 keeps that payload out of the cached
    block on every call.

    The retry is priced at the **more expensive of the two retry modes**, in
    both its output ceiling and its instruction text, because which one fires
    is not something this arithmetic gets to assume. `build_script` makes at
    most one retry per run and the modes are mutually exclusive (`retry_spent`
    guards it), so pricing both would overstate the ceiling — but pricing the
    *cheaper* one would understate it, and understating is the direction every
    earlier round of this file got caught in.
    """
    worst_retry_max_tokens = max(PODCAST_RETRY_MAX_TOKENS, PODCAST_TRUNCATION_RETRY_MAX_TOKENS)
    first_call = (
        prefix_tokens * IN_RATE_USD_PER_TOKEN * CACHE_WRITE_MULTIPLIER
        + item_payload_tokens * IN_RATE_USD_PER_TOKEN
        + PODCAST_MAX_TOKENS * OUT_RATE_USD_PER_TOKEN
    )
    retry_call = (
        prefix_tokens * IN_RATE_USD_PER_TOKEN * CACHE_WRITE_MULTIPLIER
        + (item_payload_tokens + retry_instruction_tokens) * IN_RATE_USD_PER_TOKEN
        + worst_retry_max_tokens * OUT_RATE_USD_PER_TOKEN
    )
    per_day = first_call + retry_call
    return per_day * (CALLS_PER_MONTH / CALLS_PER_DAY)


def test_the_worst_case_podcast_cost_stays_within_the_recorded_ceiling(
    repo_config_dir: Path,
) -> None:
    """Guards the budget against any change to the prompt, the constants, or
    `interests.yaml` — the prefix is rendered from the real shipped config,
    at its real size, mirroring `test_llm_scout.py`'s `_rendered_prompt_tokens`.

    The item payload is measured from the **real** `_build_podcast_user_prompt`,
    not a parallel reconstruction of its shape. An earlier version of this
    test re-derived the payload from the imported caps instead of calling
    the function that actually builds it — which is exactly why it didn't
    catch the `ensure_ascii`/character-vs-byte bug an `llm-cost-guard`
    review found by hand: the caps were right, the *test* just wasn't
    exercising the code that applies them. Calling the private function
    directly is deliberate here, the same reasoning
    `tests/curate/test_scout.py` uses for `_REJECTED_PROMPT_LIMIT`: this is
    the one thing actually being measured, not incidental internals.
    """
    from signalforge.llm import (  # noqa: PLC0415 - the real things being measured
        _PODCAST_SHORTER_INSTRUCTION,
        _PODCAST_UNFINISHED_INSTRUCTION,
        _build_podcast_user_prompt,
    )

    interests = load_interests(repo_config_dir)
    prefix = build_podcast_stable_prefix(interests)
    prefix_tokens = int(len(prefix.encode("utf-8")) / BYTES_PER_TOKEN)

    # Presenter names priced at `PodcastChannelConfig`'s enforced bound, not
    # the short `PRESENTER_A`/`PRESENTER_B` fixtures other tests use — an
    # `llm-cost-guard` review flagged pricing below the enforced max_length
    # as the same "capped in one place, priced in another" pattern this
    # file's other worst-case fixes exist to close. Pydantic's `max_length`
    # counts Python characters, not bytes, and presenter names reach the
    # prompt unescaped (f-string interpolation, not `json.dumps` — no JSON
    # escaping applies) — so 100 four-byte astral characters (an all-emoji
    # name; a subsequent review round caught an "A" * 100 ASCII fixture
    # under-measuring this by ~300 bytes) is the real worst case, not 100
    # ASCII bytes.
    worst_case_presenter_a = "\U0001f600" * PODCAST_PRESENTER_NAME_MAX_LENGTH
    worst_case_presenter_b = "\U0001f601" * PODCAST_PRESENTER_NAME_MAX_LENGTH

    user_text, dropped, sent_ids = _build_podcast_user_prompt(
        date_label=DATE_LABEL,
        presenter_a=worst_case_presenter_a,
        presenter_b=worst_case_presenter_b,
        items=_worst_case_item_tuples(),
    )
    assert dropped == 0  # exactly PODCAST_MAX_ITEMS items — none clamped away
    assert len(sent_ids) == PODCAST_MAX_ITEMS
    item_payload_tokens = int(len(user_text.encode("utf-8")) / BYTES_PER_TOKEN)
    # The longer of the two retry instructions, matching `_worst_case_monthly_usd`'s
    # "price the worse mode" rule — not whichever one happens to fire more often.
    retry_instruction_tokens = int(
        max(
            len(_PODCAST_SHORTER_INSTRUCTION.encode("utf-8")),
            len(_PODCAST_UNFINISHED_INSTRUCTION.encode("utf-8")),
        )
        / BYTES_PER_TOKEN
    )

    worst = _worst_case_monthly_usd(prefix_tokens, item_payload_tokens, retry_instruction_tokens)

    assert worst <= PODCAST_MONTHLY_CEILING_USD, (
        f"worst case at PODCAST_MAX_ITEMS (non-ASCII filler), priced at {CALLS_PER_DAY} "
        f"calls/day, is ${worst:.2f}/month against a ${PODCAST_MONTHLY_CEILING_USD:.2f} "
        "budget. Lower PODCAST_MAX_ITEMS, the per-field *_BYTES caps, PODCAST_MAX_TOKENS, "
        "PODCAST_RETRY_MAX_TOKENS, or PODCAST_TRUNCATION_RETRY_MAX_TOKENS, or raise the "
        "budget deliberately with the operator."
    )
