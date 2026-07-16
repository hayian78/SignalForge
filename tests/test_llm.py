"""Tests for `llm.py` — the only module allowed to import `anthropic`.

Every test here fakes the Anthropic Batches API at the client boundary
(CLAUDE.md §8, NEVER rule 13): no real SDK objects, no network. The fakes
mimic just the attribute shape `llm.py` reads (`processing_status`,
`custom_id`, `result.type`, `result.message.content`/`.usage`), which is
enough to exercise request-building, batching, structured-output parsing,
per-item error isolation, and token accounting without ever touching
`api.anthropic.com`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from anthropic import APIError
from anthropic.types.messages.batch_create_params import Request

from signalforge.config import InterestsConfig
from signalforge.llm import LlmError, TriageResult, run_triage_batch

# --------------------------------------------------------------------------- #
# Fakes — just enough of the Anthropic Batches API surface for `llm.py`
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FakeUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(frozen=True, slots=True)
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True, slots=True)
class FakeMessage:
    content: list[FakeTextBlock]
    usage: FakeUsage


@dataclass(frozen=True, slots=True)
class FakeResult:
    """Stands in for `MessageBatchSucceededResult` / `...ErroredResult` / etc.
    `type` is one of `succeeded | errored | canceled | expired` — `llm.py`
    only branches on that string, never on a real result subclass."""

    type: str
    message: FakeMessage | None = None


@dataclass(frozen=True, slots=True)
class FakeEntry:
    custom_id: str
    result: FakeResult


@dataclass(slots=True)
class FakeBatch:
    id: str
    processing_status: str


def _succeeded(item_ids: Sequence[int], *, triage: str = "keep") -> FakeResult:
    """A successful batch entry whose payload triages every id the same way."""
    payload = {
        "results": [
            {
                "item_id": item_id,
                "triage": triage,
                "signal": 4,
                "relevance": 4,
                "novelty": 3,
                "reasoning": f"Reasoning for item {item_id}.",
            }
            for item_id in item_ids
        ]
    }
    message = FakeMessage(
        content=[FakeTextBlock(text=json.dumps(payload))],
        usage=FakeUsage(input_tokens=500, output_tokens=120, cache_read_input_tokens=50),
    )
    return FakeResult(type="succeeded", message=message)


class FakeBatchesResource:
    """Fakes `client.messages.batches`.

    `poll_statuses` lets a test model a batch that takes a couple of polls to
    reach `ended`, without a real test ever sleeping for real — `run_triage_batch`
    is called with `poll_interval=0` in every test here.
    """

    def __init__(
        self,
        entries: Iterable[FakeEntry],
        *,
        poll_statuses: Sequence[str] = ("ended",),
        create_error: Exception | None = None,
        retrieve_error: Exception | None = None,
        results_error: Exception | None = None,
    ) -> None:
        self.entries = list(entries)
        self.poll_statuses = list(poll_statuses)
        self.create_error = create_error
        self.retrieve_error = retrieve_error
        self.results_error = results_error
        self.created_requests: list[Request] = []
        self._retrieve_calls = 0

    def create(self, *, requests: Sequence[Request]) -> FakeBatch:
        if self.create_error is not None:
            raise self.create_error
        self.created_requests = list(requests)
        return FakeBatch(id="batch_1", processing_status=self.poll_statuses[0])

    def retrieve(self, batch_id: str) -> FakeBatch:
        if self.retrieve_error is not None:
            raise self.retrieve_error
        self._retrieve_calls += 1
        index = min(self._retrieve_calls, len(self.poll_statuses) - 1)
        return FakeBatch(id=batch_id, processing_status=self.poll_statuses[index])

    def results(self, batch_id: str) -> Iterator[FakeEntry]:
        if self.results_error is not None:
            raise self.results_error
        return iter(self.entries)


@dataclass
class FakeMessagesResource:
    batches: FakeBatchesResource


@dataclass
class FakeAnthropicClient:
    """Just enough of `anthropic.Anthropic` for `run_triage_batch`."""

    messages: FakeMessagesResource

    @classmethod
    def with_batches(cls, batches: FakeBatchesResource) -> FakeAnthropicClient:
        return cls(messages=FakeMessagesResource(batches))


def make_interests(**overrides: object) -> InterestsConfig:
    data: dict[str, object] = {
        "thresholds": {"weekly_min_signal": 3, "weekly_min_relevance": 3, "weekly_min_total": 10},
    }
    data.update(overrides)
    return InterestsConfig.model_validate(data)


ITEMS: list[tuple[int, str, str | None]] = [
    (1, "MCP sampling lands everywhere", "A short summary."),
    (2, "Another framework announcement", "Marketing copy, no numbers."),
]


def _client(batches: FakeBatchesResource) -> Any:
    # `run_triage_batch`'s `client` parameter is typed `anthropic.Anthropic | None`;
    # the fake only implements the slice of that surface the function actually
    # calls. Cast at the call site rather than fighting the SDK's concrete type.
    return FakeAnthropicClient.with_batches(batches)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_empty_items_makes_no_api_call() -> None:
    batches = FakeBatchesResource(entries=[])
    result = run_triage_batch([], make_interests(), client=_client(batches))

    assert result.results == {}
    assert result.errors == {}
    assert result.input_tokens == 0 and result.output_tokens == 0


def test_successful_batch_returns_parsed_results_and_token_counts() -> None:
    batches = FakeBatchesResource(
        entries=[FakeEntry(custom_id="triage-0", result=_succeeded([1, 2]))]
    )

    result = run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)

    assert set(result.results) == {1, 2}
    assert all(isinstance(r, TriageResult) for r in result.results.values())
    assert result.results[1].triage == "keep"
    assert result.errors == {}
    # 500 input + 50 cache-read = 550; summed across the one group in this test.
    assert result.input_tokens == 550
    assert result.output_tokens == 120


def test_never_sends_content_only_title_and_summary() -> None:
    """NEVER rule 9 — structurally impossible to leak full article text."""
    batches = FakeBatchesResource(
        entries=[FakeEntry(custom_id="triage-0", result=_succeeded([1, 2]))]
    )

    run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)

    request = batches.created_requests[0]
    body = json.dumps(request, default=str)
    # The only per-item text in the payload is title + summary.
    assert "MCP sampling lands everywhere" in body
    assert "A short summary." in body


def test_batches_items_into_groups_of_batch_size() -> None:
    items = [(i, f"title {i}", "summary") for i in range(60)]
    entries = [
        FakeEntry(
            custom_id=f"triage-{group}",
            result=_succeeded(list(range(group * 25, min((group + 1) * 25, 60)))),
        )
        for group in range(3)
    ]
    batches = FakeBatchesResource(entries=entries)

    run_triage_batch(items, make_interests(), client=_client(batches), poll_interval=0.0)

    assert len(batches.created_requests) == 3


def test_cache_control_is_on_the_system_block() -> None:
    """DESIGN §8 caching discipline: the rubric+interests prefix is cached."""
    batches = FakeBatchesResource(
        entries=[FakeEntry(custom_id="triage-0", result=_succeeded([1, 2]))]
    )

    run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)

    request = batches.created_requests[0]
    system = request["params"]["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_polls_until_batch_ends() -> None:
    batches = FakeBatchesResource(
        entries=[FakeEntry(custom_id="triage-0", result=_succeeded([1, 2]))],
        poll_statuses=["in_progress", "in_progress", "ended"],
    )

    result = run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)

    assert set(result.results) == {1, 2}


# --------------------------------------------------------------------------- #
# Failure isolation and structured-output edge cases
# --------------------------------------------------------------------------- #


def test_a_failed_batch_request_records_a_per_item_error_for_every_item_in_it() -> None:
    batches = FakeBatchesResource(
        entries=[FakeEntry(custom_id="triage-0", result=FakeResult(type="errored"))]
    )

    result = run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)

    assert result.results == {}
    assert set(result.errors) == {1, 2}
    assert "errored" in result.errors[1]


def test_malformed_json_response_is_a_per_item_error_not_a_crash() -> None:
    message = FakeMessage(
        content=[FakeTextBlock(text="not json at all")],
        usage=FakeUsage(input_tokens=100, output_tokens=10),
    )
    batches = FakeBatchesResource(
        entries=[
            FakeEntry(custom_id="triage-0", result=FakeResult(type="succeeded", message=message))
        ]
    )

    result = run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)

    assert result.results == {}
    assert set(result.errors) == {1, 2}
    # Tokens were still spent producing the bad response — must still be counted.
    assert result.input_tokens == 100
    assert result.output_tokens == 10


def test_an_out_of_range_score_is_a_per_item_error_not_a_stored_score() -> None:
    payload = {
        "results": [
            {
                "item_id": 1,
                "triage": "keep",
                "signal": 9,  # out of the 1-5 range TriageResult enforces
                "relevance": 4,
                "novelty": 3,
                "reasoning": "x",
            },
            {
                "item_id": 2,
                "triage": "kill",
                "signal": 2,
                "relevance": 2,
                "novelty": 1,
                "reasoning": "y",
            },
        ]
    }
    message = FakeMessage(
        content=[FakeTextBlock(text=json.dumps(payload))],
        usage=FakeUsage(input_tokens=100, output_tokens=10),
    )
    batches = FakeBatchesResource(
        entries=[
            FakeEntry(custom_id="triage-0", result=FakeResult(type="succeeded", message=message))
        ]
    )

    result = run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)

    assert 1 in result.errors
    assert 2 in result.results
    assert result.results[2].triage == "kill"


def test_an_item_missing_from_the_response_is_recorded_as_an_error() -> None:
    """The model returned fewer results than it was given items."""
    batches = FakeBatchesResource(entries=[FakeEntry(custom_id="triage-0", result=_succeeded([1]))])

    result = run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)

    assert 1 in result.results
    assert 2 in result.errors
    assert "missing" in result.errors[2]


def test_batch_create_failure_raises_llm_error() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches")
    batches = FakeBatchesResource(
        entries=[], create_error=APIError("boom", request=request, body=None)
    )

    with pytest.raises(LlmError):
        run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)


def test_batch_retrieve_failure_raises_llm_error() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches")
    batches = FakeBatchesResource(
        entries=[],
        poll_statuses=["in_progress"],
        retrieve_error=APIError("boom", request=request, body=None),
    )

    with pytest.raises(LlmError):
        run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)


def test_batch_results_failure_raises_llm_error() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches")
    batches = FakeBatchesResource(
        entries=[], results_error=APIError("boom", request=request, body=None)
    )

    with pytest.raises(LlmError):
        run_triage_batch(ITEMS, make_interests(), client=_client(batches), poll_interval=0.0)


def test_poll_timeout_raises_llm_error() -> None:
    batches = FakeBatchesResource(entries=[], poll_statuses=["in_progress"])

    with pytest.raises(LlmError, match="did not complete"):
        run_triage_batch(
            ITEMS,
            make_interests(),
            client=_client(batches),
            poll_interval=0.0,
            max_poll_seconds=0.0,
        )


# --------------------------------------------------------------------------- #
# get_anthropic_client
# --------------------------------------------------------------------------- #


def test_get_anthropic_client_raises_without_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from signalforge.llm import get_anthropic_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LlmError, match="ANTHROPIC_API_KEY"):
        get_anthropic_client()


def test_get_anthropic_client_builds_a_client_from_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from signalforge.llm import get_anthropic_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    client = get_anthropic_client()
    assert client.api_key == "sk-ant-test-key"
