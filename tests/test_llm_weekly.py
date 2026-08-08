"""Tests for `llm.run_weekly_brief` — the Sunday synthesis call (DESIGN §13, Phase 1).

The client is faked at `llm.py`'s boundary; the real Anthropic API is never
touched (CLAUDE.md §8, NEVER rule 13). Responses are real SDK `Message`/`Usage`/
`TextBlock` types, the same discipline `test_llm_podcast.py` uses, so a
duck-typed stand-in cannot pass a test the real API would fail.

What these tests protect:

* **The spend caps.** `WEEKLY_MAX_ITEMS` and the three per-field byte ceilings
  clamp the payload before a request is built, and the worst-case arithmetic
  prices **five** calls a month — a month can contain five Sundays — against
  `WEEKLY_MONTHLY_CEILING_USD`.
* **What is never sent.** The payload carries `(item_id, title, summary,
  reasoning)` and nothing else. `items.content` — which several of these rows do
  carry, fetched by the podcast path — must never reach this prompt.
* **The citation boundary (NEVER rule 7).** `sent_item_ids` is the set the model
  actually saw, narrower than the caller's list whenever the clamp bit.
* **The cache boundary (NEVER rule 10).** The stable prefix never varies with the
  window. There is deliberately no `cache_control` breakpoint here.
* **The accounting (NEVER rule 11).** Usage is recorded on every failure path — a
  malformed response still spent output tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from anthropic import APIError
from anthropic.types import Message, TextBlock, ThinkingBlock, Usage

from signalforge.config import load_interests
from signalforge.llm import (
    WEEKLY_MAX_ITEM_REASONING_BYTES,
    WEEKLY_MAX_ITEM_SUMMARY_BYTES,
    WEEKLY_MAX_ITEM_TITLE_BYTES,
    WEEKLY_MAX_ITEMS,
    WEEKLY_MAX_TOKENS,
    WEEKLY_MODEL,
    WEEKLY_MONTHLY_CEILING_USD,
    LlmError,
    run_weekly_brief,
)
from signalforge.synth.weekly import build_weekly_stable_prefix

STABLE_PREFIX = "You write a weekly intelligence brief."
WINDOW_LABEL = "2 August to 8 August 2026"

_VALID_BRIEF: dict[str, Any] = {
    "leads": [
        {
            "headline": "Agent intrusions moved from theory to timeline",
            "why_it_matters": "Two independent write-ups landed the same week.",
            "item_ids": [1],
        }
    ],
    "clusters": [
        {
            "title": "Adoption keeps outrunning the tooling",
            "narrative": "The gap shows up in every deployment report.",
            "item_ids": [1],
        }
    ],
}


def _items(count: int = 1) -> list[tuple[int, str, str | None, str]]:
    return [
        (item_id, f"Title {item_id}", f"Summary {item_id}", f"Reasoning {item_id}")
        for item_id in range(1, count + 1)
    ]


def _message(
    *,
    brief: dict[str, Any] | None = _VALID_BRIEF,
    stop_reason: str = "end_turn",
    input_tokens: int = 1_000,
    output_tokens: int = 500,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    raw_text: str | None = None,
    no_text: bool = False,
) -> Message:
    """A real `Message`, shaped like a structured-output weekly-brief response."""
    content: list[Any] = []
    if not no_text:
        content.append(
            TextBlock(text=raw_text if raw_text is not None else json.dumps(brief), type="text")
        )
    return Message(
        id="msg_weekly",
        content=content,
        model=WEEKLY_MODEL,
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
    def __init__(self, responses: list[Message | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.requests.append(kwargs)
        if not self._responses:  # pragma: no cover - a test scripted too few
            raise AssertionError("run_weekly_brief made more calls than scripted")
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
        "window_label": WINDOW_LABEL,
        "items": _items(),
        "client": client,
    }
    kwargs.update(overrides)
    return run_weekly_brief(STABLE_PREFIX, **kwargs), client


# --------------------------------------------------------------------------- #
# the request shape
# --------------------------------------------------------------------------- #


def test_a_valid_response_becomes_a_brief() -> None:
    result, _ = _run([_message()])

    assert result.error is None
    assert result.brief is not None
    assert result.brief.leads[0].headline.startswith("Agent intrusions")
    assert result.input_tokens == 1_000
    assert result.output_tokens == 500


def test_the_request_uses_the_weekly_model_effort_and_token_budget() -> None:
    _, client = _run([_message()])
    request = client.messages.requests[0]

    assert request["model"] == WEEKLY_MODEL
    assert request["max_tokens"] == WEEKLY_MAX_TOKENS
    assert request["output_config"]["effort"] == "high"
    assert request["output_config"]["format"]["type"] == "json_schema"


def test_the_stable_prefix_is_sent_verbatim_as_the_system_block() -> None:
    _, client = _run([_message()])
    system = client.messages.requests[0]["system"]

    assert len(system) == 1
    assert system[0]["text"] == STABLE_PREFIX


def test_there_is_no_cache_breakpoint() -> None:
    """One request a week, seven days apart, no retry — an ephemeral cache entry
    expires long before it could be read, so a breakpoint is pure write premium.
    The scout records the same reasoning at the same cadence (DESIGN §8).

    Asserted rather than left implicit: adding `cache_control` here looks like a
    free optimisation and is actually a 1.25x surcharge on every call."""
    _, client = _run([_message()])
    assert "cache_control" not in client.messages.requests[0]["system"][0]


def test_the_window_label_rides_in_the_user_turn_not_the_prefix() -> None:
    """NEVER rule 10, proven against the request the code actually builds rather
    than against the prefix builder's signature alone."""
    _, client = _run([_message()])
    request = client.messages.requests[0]

    assert WINDOW_LABEL not in request["system"][0]["text"]
    assert WINDOW_LABEL in request["messages"][0]["content"][0]["text"]


def test_two_windows_produce_a_byte_identical_system_block() -> None:
    _, first = _run([_message()], window_label="2 August to 8 August 2026")
    _, second = _run([_message()], window_label="9 August to 15 August 2026")

    assert first.messages.requests[0]["system"] == second.messages.requests[0]["system"]


def test_the_stable_prefix_builder_cannot_accept_a_date_or_a_run_id() -> None:
    """The structural half of NEVER rule 10: volatile data cannot enter the cached
    prefix because the function that builds it has nowhere to put it."""
    import inspect

    parameters = inspect.signature(build_weekly_stable_prefix).parameters
    assert list(parameters) == ["interests"]


def test_the_payload_carries_no_item_content() -> None:
    """DESIGN §8's whole cost argument for this call. Several of these rows do
    carry deep-read `content` from the podcast path, and the 4-tuple seam is what
    stops it riding along — asserted on the wire, not on the type."""
    _, client = _run([_message()])
    sent = client.messages.requests[0]["messages"][0]["content"][0]["text"]
    payload = json.loads(sent[sent.index("[") :])

    assert set(payload[0]) == {"item_id", "title", "summary", "why_it_was_kept"}
    assert "content" not in sent


# --------------------------------------------------------------------------- #
# the spend caps
# --------------------------------------------------------------------------- #


def test_items_beyond_the_ceiling_are_clamped_and_reported() -> None:
    result, client = _run([_message()], items=_items(WEEKLY_MAX_ITEMS + 5))

    assert result.dropped_item_count == 5
    assert len(result.sent_item_ids) == WEEKLY_MAX_ITEMS
    sent = client.messages.requests[0]["messages"][0]["content"][0]["text"]
    assert f'"item_id": {WEEKLY_MAX_ITEMS + 1}' not in sent


def test_a_clamped_away_id_is_not_a_legitimate_citation() -> None:
    """`sent_item_ids` is narrower than the caller's list whenever the clamp bit.
    A caller validating against its own input would accept a citation for an item
    the model never saw."""
    result, _ = _run([_message()], items=_items(WEEKLY_MAX_ITEMS + 1))
    assert (WEEKLY_MAX_ITEMS + 1) not in result.sent_item_ids


@pytest.mark.parametrize(
    ("field", "cap", "key"),
    [
        ("title", WEEKLY_MAX_ITEM_TITLE_BYTES, "title"),
        ("summary", WEEKLY_MAX_ITEM_SUMMARY_BYTES, "summary"),
        ("reasoning", WEEKLY_MAX_ITEM_REASONING_BYTES, "why_it_was_kept"),
    ],
)
def test_every_text_field_is_capped_in_json_escaped_bytes(field: str, cap: int, key: str) -> None:
    """Capped after JSON escaping, not before, and in bytes rather than characters.

    `\\x01` is the fixture that catches it: it escapes to the six characters
    `\\u0001`, so a cap applied to raw characters lets the billed payload run six
    times over. The summary field is the one that binds in practice — a real
    week's longest ran eight times its cap."""
    filler = "\x01" * (cap * 2)
    values = {"title": "t", "summary": "s", "reasoning": "r"}
    values[field] = filler
    _, client = _run(
        [_message()],
        items=[(1, values["title"], values["summary"], values["reasoning"])],
    )

    sent = client.messages.requests[0]["messages"][0]["content"][0]["text"]
    payload = json.loads(sent[sent.index("[") :])
    escaped_len = len(json.dumps(payload[0][key], ensure_ascii=False).encode("utf-8")) - 2
    assert escaped_len <= cap


def test_a_non_ascii_field_is_capped_in_bytes_not_characters() -> None:
    _, client = _run([_message()], items=[(1, "中" * WEEKLY_MAX_ITEM_TITLE_BYTES, "s", "r")])

    sent = client.messages.requests[0]["messages"][0]["content"][0]["text"]
    payload = json.loads(sent[sent.index("[") :])
    assert len(payload[0]["title"].encode("utf-8")) <= WEEKLY_MAX_ITEM_TITLE_BYTES


# --------------------------------------------------------------------------- #
# accounting on every failure path (NEVER rule 11)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("response", "fragment"),
    [
        (_message(stop_reason="refusal"), "refused"),
        (_message(no_text=True), "no text content"),
        (_message(raw_text="{not json"), "schema validation failed"),
        (_message(brief={"leads": [], "clusters": []}), "no leads and no clusters"),
    ],
)
def test_a_failed_response_still_reports_what_it_spent(response: Message, fragment: str) -> None:
    """A malformed response is billed exactly like a good one. A path that
    returned zero tokens here would drop real money out of `runs`.

    The clamp fields are asserted too, not just the tokens: all four failure
    paths funnel through one `failed()` closure, and a closure that quietly
    stopped passing `sent_item_ids` would hand callers an empty citation set —
    the NEVER rule 7 surface — with every token assertion still green."""
    result, _ = _run([response], items=_items(WEEKLY_MAX_ITEMS + 5))

    assert result.brief is None
    assert fragment in (result.error or "")
    assert result.input_tokens == 1_000
    assert result.output_tokens == 500
    assert result.dropped_item_count == 5
    assert len(result.sent_item_ids) == WEEKLY_MAX_ITEMS


def test_cache_tokens_fold_into_the_input_figure() -> None:
    result, _ = _run(
        [_message(input_tokens=100, cache_creation_input_tokens=20, cache_read_input_tokens=5)]
    )
    assert result.input_tokens == 125


def test_a_pre_response_api_error_raises_because_nothing_was_spent() -> None:
    error = APIError("boom", request=httpx.Request("POST", "https://api.anthropic.com"), body=None)
    with pytest.raises(LlmError, match="failed to generate weekly brief"):
        _run([error])


# --------------------------------------------------------------------------- #
# the worst-case cost, computed rather than assumed
# --------------------------------------------------------------------------- #

IN_RATE_USD_PER_TOKEN = 5 / 1e6
OUT_RATE_USD_PER_TOKEN = 25 / 1e6

CALLS_PER_MONTH = 5
"""**Not** 4.33, and not an assumption about cron.

A calendar month can contain five Sundays, and nothing in the code enforces a
weekly cadence — `--force` and a mis-wired invocation both bill again. Pricing a
ceiling off an expected cadence is the mistake DESIGN §8 names outright: "an
estimate that assumes good behaviour is a forecast, not a bound." Five is the
smallest number that is actually true of a month."""

BYTES_PER_TOKEN = 1.0
"""The pessimistic ratio, not the merely-conservative one.

`test_llm_podcast.py` prices at 1.5 and records at length why even that is an
estimate rather than a measurement, with a sensitivity table showing 1.0 as the
realistic bad case for a script with poor BPE coverage. This ceiling is priced
directly at that bad case instead of near it, because this payload is small
enough that the extra margin costs almost nothing — and a ceiling that only holds
at a favourable tokenization is not a ceiling.

Replace with a grounded figure once real runs report actual `input_tokens`."""


@pytest.mark.parametrize(
    ("filler", "label"),
    [("\x01", "control-character"), ("\U0001d11e", "astral-plane")],
)
def test_the_worst_case_weekly_cost_stays_within_the_recorded_ceiling(
    repo_config_dir: Path, filler: str, label: str
) -> None:
    """Guards the budget against any change to the prompt, the constants, or
    `interests.yaml`.

    Asserted at `WEEKLY_MAX_ITEMS`, never at the shipped `thresholds.weekly_top_n`
    — otherwise a pure YAML edit moves real spend with every test green (DESIGN
    §8). The payload is measured from the **real** `_build_weekly_user_prompt`
    rather than reconstructed from the imported caps: an earlier version of the
    podcast's equivalent re-derived the shape and therefore never exercised the
    code that applies the caps, which is exactly how the character-vs-byte bug
    survived.
    """
    from signalforge.llm import _build_weekly_user_prompt  # noqa: PLC0415

    interests = load_interests(repo_config_dir)
    prefix = build_weekly_stable_prefix(interests)
    prefix_tokens = len(prefix.encode("utf-8")) / BYTES_PER_TOKEN

    # Every field at its cap. Two fillers, because they fail differently:
    # `\x01` escapes identically under either `ensure_ascii` setting and so
    # cannot detect a flip to the default, while an astral-plane character
    # expands from 4 UTF-8 bytes to 12 escaped ones under `ensure_ascii=True`.
    # `_truncate_utf8_json_safe` measures against `ensure_ascii=False`, so that
    # flip would re-open the very gap the caps close — and with only the `\x01`
    # fixture, every assertion in this file stays green while it does.
    worst_item = (
        10**9,
        filler * WEEKLY_MAX_ITEM_TITLE_BYTES,
        filler * WEEKLY_MAX_ITEM_SUMMARY_BYTES,
        filler * WEEKLY_MAX_ITEM_REASONING_BYTES,
    )
    user_text, dropped, sent_ids = _build_weekly_user_prompt(
        window_label=WINDOW_LABEL,
        items=[worst_item] * (WEEKLY_MAX_ITEMS + 3),
    )
    assert dropped == 3, "the clamp must be what bounds the payload, not the caller"
    assert len(sent_ids) == WEEKLY_MAX_ITEMS

    user_tokens = len(user_text.encode("utf-8")) / BYTES_PER_TOKEN

    # No cache-write multiplier: there is no breakpoint (see `run_weekly_brief`).
    per_call = (prefix_tokens + user_tokens) * IN_RATE_USD_PER_TOKEN + (
        WEEKLY_MAX_TOKENS * OUT_RATE_USD_PER_TOKEN
    )
    monthly = per_call * CALLS_PER_MONTH

    assert monthly <= WEEKLY_MONTHLY_CEILING_USD, (
        f"worst case ${monthly:.2f}/month with {label} filler exceeds the "
        f"recorded ${WEEKLY_MONTHLY_CEILING_USD:.2f} ceiling"
    )


def test_a_brief_with_only_leads_or_only_clusters_is_accepted() -> None:
    """The rejection is `not leads AND not clusters`, never `or`.

    `clusters` has no `minItems`, so a lead-only brief is exactly what the schema
    permits on a week with no theme worth naming. Rejecting it would turn a good,
    fully-billed call into a reported failure — and Stage 6 would re-bill it."""
    leads_only, _ = _run([_message(brief={"leads": _VALID_BRIEF["leads"], "clusters": []})])
    clusters_only, _ = _run([_message(brief={"leads": [], "clusters": _VALID_BRIEF["clusters"]})])

    assert leads_only.brief is not None
    assert leads_only.error is None
    assert clusters_only.brief is not None
    assert clusters_only.error is None


def test_a_block_citing_nothing_is_a_schema_failure() -> None:
    """`item_ids` is `min_length=1` — the first line of NEVER rule 7.

    Stage 5's unknown-id cleaning cannot catch this: an empty list contains no
    unknown ids, so a lead with no citations would sail through it and land in
    the vault as a synthesized claim backed by nothing."""
    result, _ = _run(
        [
            _message(
                brief={
                    "leads": [{"headline": "H", "why_it_matters": "W", "item_ids": []}],
                    "clusters": [],
                }
            )
        ]
    )
    assert result.brief is None
    assert "schema validation failed" in (result.error or "")


def test_more_than_three_leads_is_not_a_validation_failure() -> None:
    """A count over the cap must not throw away a paid response.

    `synth.weekly` clamps instead. Rejecting here would lose an otherwise good
    brief *and* leave no vault file, which invites the next invocation to pay for
    the same week again."""
    lead = _VALID_BRIEF["leads"][0]
    result, _ = _run([_message(brief={"leads": [lead] * 4, "clusters": []})])

    assert result.brief is not None
    assert len(result.brief.leads) == 4


def test_the_request_schema_carries_no_array_bounds() -> None:
    """The API rejects `minItems`/`maxItems` on array types outright — a first
    live call returned `400 invalid_request_error` on exactly that. Pinned so a
    future edit re-adding them fails here rather than on a Sunday morning.

    Every count bound therefore lives in code: the prompt asks for the shape and
    `WEEKLY_MAX_LEADS`/`WEEKLY_MAX_CLUSTERS` clamp what comes back."""
    _, client = _run([_message()])
    schema = client.messages.requests[0]["output_config"]["format"]["schema"]

    assert "minItems" not in json.dumps(schema)
    assert "maxItems" not in json.dumps(schema)


def test_the_text_block_is_found_past_a_thinking_block() -> None:
    """At `effort: high` every real response leads with a thinking block, so the
    `block.type == "text"` filter is load-bearing on the *normal* path. Every
    other fixture here has a single block, which would never exercise it."""
    message = _message()
    message.content.insert(
        0, ThinkingBlock(thinking="weighing the week", signature="sig", type="thinking")
    )

    result, _ = _run([message])

    assert result.brief is not None


def test_an_item_with_no_summary_sends_an_empty_string() -> None:
    """`Item.summary` really is optional. A future edit dropping the `or ""` would
    crash the one weekly call of the month on an ordinary row."""
    _, client = _run([_message()], items=[(1, "Title", None, "Reasoning")])

    sent = client.messages.requests[0]["messages"][0]["content"][0]["text"]
    payload = json.loads(sent[sent.index("[") :])
    assert payload[0]["summary"] == ""


def test_the_prefix_states_the_citation_discipline_it_alone_enforces() -> None:
    """Pinned by wording, because no unit test can observe the behaviour.

    Most of this prompt's instructions have a runtime counterpart — flattening,
    unknown-id dropping, the lead cap. "Never make a claim no cited item
    supports" has none: it is the one NEVER rule 7 control that exists only as
    text, so deleting it must fail something. Same reasoning the podcast file
    applies to its intro/outro rule."""
    from signalforge.config import load_interests as _load  # noqa: PLC0415

    prefix = build_weekly_stable_prefix(_load(Path("config")))

    assert "Never cite an id you were not given" in prefix
    assert "never make a claim no cited item supports" in prefix
