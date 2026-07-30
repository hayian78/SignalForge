"""Tests for `llm.run_source_scout` — the weekly curation call (DESIGN §7.1, §8).

The client is faked at `llm.py`'s boundary; the real Anthropic API is never
touched (CLAUDE.md §8, NEVER rule 13). Unlike `test_llm.py`'s batch fakes, the
*responses* here are real SDK types — `Message`, `ToolUseBlock`, `Usage` — because
`run_source_scout` narrows content blocks with `isinstance`, and a duck-typed
stand-in would pass a test the real API would fail.

What these tests are actually protecting:

* **The spend caps.** `max_uses` on the search tool, the ceiling that config
  cannot exceed, and the one-request-per-run rule that makes the monthly ceiling
  provable rather than estimated. Each is real money if it breaks.
* **The accounting.** Tokens *and* `web_search_requests`. Search bills per call, so
  a run's cost is not derivable from tokens (NEVER rule 11).
* **Failure isolation.** One malformed proposal must not discard the others from a
  call that has already been paid for.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from anthropic import APIError
from anthropic.types import Message, ServerToolUsage, TextBlock, ToolUseBlock, Usage

from signalforge.llm import (
    SCOUT_EFFORT,
    SCOUT_MAX_SEARCHES_CEILING,
    SCOUT_MAX_TOKENS,
    SCOUT_MODEL,
    SCOUT_MONTHLY_CEILING_USD,
    SCOUT_PROPOSE_TOOL_NAME,
    LlmError,
    run_source_scout,
)
from signalforge.models import ProposalKind, ProposalTier

SYSTEM_PROMPT = "You advise on sources."
USER_PROMPT = "Here is the evidence."


def _proposal(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "add_rss",
        "target": "https://newvoice.example.com/feed",
        "source_id": "newvoice",
        "url": "https://newvoice.example.com/feed",
        "rationale": "Cited four times by items you marked useful.",
        "evidence": [{"url": "https://simonwillison.net/x/", "note": "links to it"}],
        "tier": "corpus",
    }
    payload.update(overrides)
    return payload


def _message(
    *,
    proposals: list[dict[str, Any]] | None = None,
    stop_reason: str = "tool_use",
    input_tokens: int = 40_000,
    output_tokens: int = 3_000,
    web_searches: int = 6,
    text: str | None = None,
) -> Message:
    """A real `Message`, shaped like the response the scout expects."""
    content: list[Any] = []
    if text is not None:
        content.append(TextBlock(text=text, type="text"))
    if proposals is not None:
        content.append(
            ToolUseBlock(
                id="toolu_scout",
                name=SCOUT_PROPOSE_TOOL_NAME,
                input={"proposals": proposals},
                type="tool_use",
            )
        )
    return Message(
        id="msg_scout",
        content=content,
        model=SCOUT_MODEL,
        role="assistant",
        type="message",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            server_tool_use=ServerToolUsage(web_search_requests=web_searches, web_fetch_requests=0),
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
            raise AssertionError("the scout made more calls than the test scripted")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class FakeClient:
    def __init__(self, responses: list[Message | Exception]) -> None:
        self.messages = FakeMessages(responses)


def _run(responses: list[Message | Exception], **overrides: Any) -> Any:
    client = FakeClient(responses)
    kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "max_searches": 12,
        "max_proposals": 5,
        "client": client,
    }
    kwargs.update(overrides)
    result = run_source_scout(**kwargs)
    return result, client


# --------------------------------------------------------------------------- #
# the happy path and the request shape
# --------------------------------------------------------------------------- #


def test_a_tool_call_becomes_validated_proposals() -> None:
    result, _ = _run([_message(proposals=[_proposal()])])

    assert result.errors == []
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.kind is ProposalKind.ADD_RSS
    assert proposal.tier is ProposalTier.CORPUS
    assert proposal.evidence[0].url == "https://simonwillison.net/x/"


def test_an_empty_proposal_list_is_a_valid_quiet_week() -> None:
    # Proposing nothing is a correct answer, not a failure — the prompt says so,
    # and a run that records an error for it would train the operator to ignore
    # the error line.
    result, _ = _run([_message(proposals=[])])

    assert result.proposals == []
    assert result.errors == []


def test_the_request_declares_web_search_with_the_configured_cap() -> None:
    _, client = _run([_message(proposals=[])], max_searches=9)

    tools = client.messages.requests[0]["tools"]
    search = next(tool for tool in tools if str(tool["type"]).startswith("web_search"))
    # `max_uses` is enforced server-side, so this is the real spend limit rather
    # than an instruction the model may ignore.
    assert search["max_uses"] == 9
    # Drops the nested search blocks from the assistant message once dynamic
    # filtering has consumed them — an output-token saving.
    assert search["response_inclusion"] == "excluded"


def test_the_request_does_not_declare_code_execution() -> None:
    # On `web_search_20260209`+ the API provisions the execution dynamic filtering
    # needs; declaring a second execution environment confuses the model.
    _, client = _run([_message(proposals=[])])

    types = {str(tool.get("type")) for tool in client.messages.requests[0]["tools"]}
    assert not any(name.startswith("code_execution") for name in types)


def test_the_request_carries_no_cache_control() -> None:
    """The one place the caching discipline is deliberately inverted (DESIGN §8).

    At a weekly cadence every cache entry has expired before the next call, so a
    breakpoint would pay the ~1.25x write premium for exactly zero reads. Asserted
    rather than commented, because "add cache_control everywhere" is exactly the
    well-meaning change a future reader would make.
    """
    _, client = _run([_message(proposals=[])])
    request = client.messages.requests[0]

    assert "cache_control" not in request
    assert "cache_control" not in str(request["system"])
    for tool in request["tools"]:
        assert "cache_control" not in tool


def test_the_request_uses_the_configured_model_effort_and_token_ceiling() -> None:
    _, client = _run([_message(proposals=[])])
    request = client.messages.requests[0]

    assert request["model"] == SCOUT_MODEL
    assert request["max_tokens"] == SCOUT_MAX_TOKENS
    assert request["output_config"]["effort"] == SCOUT_EFFORT
    # No sampling parameters: they are rejected outright on this model tier.
    assert "temperature" not in request
    assert "top_p" not in request


def test_the_prompts_are_passed_through_unchanged() -> None:
    # `llm.py` never writes prompt text (module docstring); it transports what
    # `curate/prompts.py` rendered.
    _, client = _run([_message(proposals=[])])
    request = client.messages.requests[0]

    assert request["system"] == SYSTEM_PROMPT
    assert request["messages"] == [{"role": "user", "content": USER_PROMPT}]


# --------------------------------------------------------------------------- #
# the spend caps
# --------------------------------------------------------------------------- #


def test_a_config_budget_above_the_ceiling_is_clamped_and_recorded() -> None:
    """A clamp the operator cannot see is a spend limit they cannot trust.

    They would configure 100, get 15, and never learn. The clamp therefore lands
    in `errors`, which the caller writes to `runs.errors`, which the next digest
    surfaces — the reports are the monitoring channel (DESIGN §7), not cron.log.
    """
    result, client = _run([_message(proposals=[])], max_searches=100)

    tools = client.messages.requests[0]["tools"]
    search = next(tool for tool in tools if str(tool["type"]).startswith("web_search"))
    assert search["max_uses"] == SCOUT_MAX_SEARCHES_CEILING
    assert len(result.errors) == 1
    assert "ceiling" in result.errors[0]
    assert str(SCOUT_MAX_SEARCHES_CEILING) in result.errors[0]


def test_a_budget_at_or_below_the_ceiling_passes_through_silently() -> None:
    result, client = _run([_message(proposals=[])], max_searches=SCOUT_MAX_SEARCHES_CEILING)

    tools = client.messages.requests[0]["tools"]
    search = next(tool for tool in tools if str(tool["type"]).startswith("web_search"))
    assert search["max_uses"] == SCOUT_MAX_SEARCHES_CEILING
    assert result.errors == []


def test_zero_searches_is_honoured_as_a_corpus_only_run() -> None:
    # A supported, documented mode: reason from the stored corpus and spend
    # nothing on search. It must reach the API as 0, not be treated as "unset".
    _, client = _run([_message(proposals=[])], max_searches=0)

    tools = client.messages.requests[0]["tools"]
    search = next(tool for tool in tools if str(tool["type"]).startswith("web_search"))
    assert search["max_uses"] == 0


def test_the_proposal_cap_reaches_the_tool_schema() -> None:
    _, client = _run([_message(proposals=[])], max_proposals=3)

    tools = client.messages.requests[0]["tools"]
    propose = next(tool for tool in tools if tool.get("name") == SCOUT_PROPOSE_TOOL_NAME)
    schema = propose["input_schema"]["properties"]["proposals"]
    assert schema["maxItems"] == 3


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #


def test_tokens_and_search_requests_are_both_recorded() -> None:
    result, _ = _run(
        [_message(proposals=[], input_tokens=41_000, output_tokens=3_800, web_searches=11)]
    )

    assert result.input_tokens == 41_000
    assert result.output_tokens == 3_800
    # The number the token counters cannot express: search bills per call.
    assert result.web_search_requests == 11


def test_cache_token_fields_are_folded_into_input_tokens() -> None:
    # Mirrors `run_triage_batch`: cache reads and writes are input tokens that
    # were really spent, so they must not vanish from the run's total.
    message = _message(proposals=[], input_tokens=1_000, web_searches=0)
    message.usage.cache_creation_input_tokens = 200
    message.usage.cache_read_input_tokens = 50

    result, _ = _run([message])

    assert result.input_tokens == 1_250


def test_usage_without_server_tool_use_does_not_lose_the_token_counts() -> None:
    # A response whose usage omits `server_tool_use` (no search was run) must
    # still contribute its tokens.
    message = _message(proposals=[], input_tokens=900, output_tokens=100)
    message.usage.server_tool_use = None

    result, _ = _run([message])

    assert (result.input_tokens, result.output_tokens) == (900, 100)
    assert result.web_search_requests == 0


# --------------------------------------------------------------------------- #
# pause_turn
# --------------------------------------------------------------------------- #


def _search_budget(request: dict[str, Any]) -> int:
    search = next(tool for tool in request["tools"] if str(tool["type"]).startswith("web_search"))
    return int(search["max_uses"])


def test_the_scout_makes_exactly_one_api_request() -> None:
    """One request per run is the fact that makes this call's ceiling provable.

    `max_tokens` bounds output *per request*, so resuming a paused turn multiplies
    it: at two resumes the enforced ceiling is 3 x 16,384 output tokens ~ $6.52/month,
    past this feature's budget, and no `max_tokens` that avoids truncation fixes
    that while resumes exist. With one request the absolute worst case is ~$2.35/month
    from code constants alone.

    Asserted rather than commented because re-adding a resume loop is a natural
    "improvement" that would silently triple the ceiling.
    """
    _, client = _run([_message(proposals=[_proposal()])])

    assert len(client.messages.requests) == 1


def test_a_paused_turn_is_recorded_and_not_resumed() -> None:
    # The cost of the single-request rule: a turn that pauses yields nothing this
    # week. Recorded so it reaches `runs.errors` and the next digest, because a run
    # that silently produced nothing looks identical to a quiet week.
    paused = _message(proposals=None, stop_reason="pause_turn", web_searches=4)

    result, client = _run([paused, _message(proposals=[_proposal()])])

    assert len(client.messages.requests) == 1, "must not resume"
    assert result.proposals == []
    assert any("paused" in error for error in result.errors)
    # The searches it did run are still on the invoice and still counted.
    assert result.web_search_requests == 4


def test_the_search_budget_reaches_the_only_request() -> None:
    _, client = _run([_message(proposals=[])], max_searches=6)

    assert _search_budget(client.messages.requests[0]) == 6


def test_the_worst_case_cost_stays_within_the_recorded_ceiling() -> None:
    """Guards the budget arithmetic against a later constant change.

    Recomputes the absolute worst case from the module's own constants — one
    request, searches maxed, output maxed — and checks it against the ceiling
    recorded beside them. Raising `SCOUT_MAX_TOKENS`, or the shipped search budget,
    without re-checking the sum fails here rather than on an invoice.
    """
    in_rate, out_rate, per_search_usd, weeks = 5 / 1e6, 25 / 1e6, 0.01, 4.33
    shipped_searches = 6  # config/sources.yaml curation.max_searches_per_run
    prompt_tokens = 2_500
    per_search_result_tokens = 2_000

    tokens_in = prompt_tokens + shipped_searches * per_search_result_tokens
    worst_per_run = (
        tokens_in * in_rate + SCOUT_MAX_TOKENS * out_rate + shipped_searches * per_search_usd
    )

    assert worst_per_run * weeks <= SCOUT_MONTHLY_CEILING_USD, (
        f"worst case is ${worst_per_run * weeks:.2f}/month against a "
        f"${SCOUT_MONTHLY_CEILING_USD:.2f} ceiling; re-tune the constants or "
        "renegotiate the budget with the operator"
    )


# --------------------------------------------------------------------------- #
# failure isolation
# --------------------------------------------------------------------------- #


def test_one_invalid_proposal_does_not_discard_the_others() -> None:
    # The searches are already paid for. Throwing away three good suggestions
    # because a fourth was malformed would waste that money twice (CLAUDE.md §7).
    result, _ = _run(
        [
            _message(
                proposals=[
                    _proposal(target="https://a.example/feed"),
                    _proposal(target="https://b.example/feed", evidence=[]),
                    _proposal(target="https://c.example/feed"),
                ]
            )
        ]
    )

    assert [proposal.target for proposal in result.proposals] == [
        "https://a.example/feed",
        "https://c.example/feed",
    ]
    assert len(result.errors) == 1
    assert "proposal 1" in result.errors[0]


def test_a_proposal_with_no_evidence_is_rejected() -> None:
    # `db.insert_proposal` raises on an uncited proposal, so catching it here
    # turns a wasted paid slot into a recorded error rather than a crash mid-run
    # (CLAUDE.md §5, NEVER rule 7).
    result, _ = _run([_message(proposals=[_proposal(evidence=[])])])

    assert result.proposals == []
    assert len(result.errors) == 1


def test_an_invented_kind_is_rejected_rather_than_stored() -> None:
    # An unknown kind would reach an applier that has no code path for it.
    result, _ = _run([_message(proposals=[_proposal(kind="rewrite_interests_yaml")])])

    assert result.proposals == []
    assert any("validation" in error for error in result.errors)


def test_an_unexpected_field_is_rejected() -> None:
    # `extra="forbid"`: a field the pipeline does not understand must not ride
    # along silently into a config edit.
    result, _ = _run([_message(proposals=[_proposal(auto_apply=True)])])

    assert result.proposals == []


def test_a_tool_input_without_a_proposals_list_is_recorded() -> None:
    message = _message(proposals=[])
    message.content[0].input = {"suggestions": []}  # type: ignore[union-attr]

    result, _ = _run([message])

    assert result.proposals == []
    assert any("no 'proposals' list" in error for error in result.errors)


def test_finishing_without_calling_the_tool_is_recorded() -> None:
    # A genuine "nothing to propose" is an empty tool call. Ending the turn with
    # prose instead means the run produced nothing usable, and that is worth a
    # line in the next digest rather than reading as a quiet week.
    result, _ = _run([_message(proposals=None, stop_reason="end_turn", text="I could not decide.")])

    assert result.proposals == []
    assert any("without calling" in error for error in result.errors)


def test_a_refusal_is_recorded_rather_than_raised() -> None:
    result, _ = _run([_message(proposals=None, stop_reason="refusal")])

    assert result.proposals == []
    assert any("refused" in error for error in result.errors)


def _api_error() -> APIError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return APIError("boom", request=request, body=None)


def test_a_failed_request_still_reports_the_searches_it_paid_for() -> None:
    """Searches run server-side before the response is read, so a failure that
    discarded the record would hide real spend (NEVER rule 11).

    Exercised through the single request the scout now makes: the failure path has
    to preserve whatever the run had, even though there is no resume to accumulate
    across any more.
    """
    result, _ = _run([_api_error()])

    assert result.proposals == []
    assert any("source scout call failed" in error for error in result.errors)
    # Nothing was read back, so the counts are zero — but the *shape* is intact and
    # the caller still gets a result to record rather than an exception.
    assert result.input_tokens == 0
    assert result.web_search_requests == 0


def test_an_api_error_is_recorded_rather_than_raised() -> None:
    # Deliberately unlike `run_triage_batch`, which raises. This call spends money
    # in two places across possibly several requests, so an exception thrown from
    # inside the loop would carry off the only record of that spend.
    result, _ = _run([_api_error()])

    assert result.proposals == []
    assert any("source scout call failed" in error for error in result.errors)


def test_llm_error_is_still_raised_when_no_client_can_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The one remaining `LlmError` path: a missing key fails before anything is
    # spent, so there is no partial result worth preserving and a hard failure is
    # the honest signal.
    monkeypatch.setattr("signalforge.llm.get_secret", lambda _name: None)

    with pytest.raises(LlmError, match="ANTHROPIC_API_KEY"):
        run_source_scout(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
            max_searches=6,
            max_proposals=5,
        )
