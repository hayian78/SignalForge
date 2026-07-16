"""The single Anthropic client wrapper — caching, batching, budget (DESIGN §8).

This is the **only** module in the codebase permitted to import the `anthropic`
SDK (CLAUDE.md §2, NEVER rule 1). `score/` and (later) `synth/` call through
the functions here; they must never touch `anthropic` directly.

Triage/scoring runs on `claude-haiku-4-5` via the **Batches API** (50% off),
structured outputs, ~25 items per request, on titles + summaries only — never
full `content` (NEVER rule 9). Every call returns its token counts alongside
its results so a caller cannot persist scores without also seeing what they
cost (NEVER rule 11) — see `TriageBatchResult`.

The prompt-caching discipline lives in `score/rubrics.py`: the frozen rubric +
`interests.yaml` is the cached prefix, carrying no timestamps or run IDs
(NEVER rule 10). This module reads that rendered text and attaches
`cache_control`; it never builds prompt text itself.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from signalforge.config import InterestsConfig, get_secret
from signalforge.score.rubrics import build_triage_system_prompt

__all__ = [
    "TRIAGE_BATCH_SIZE",
    "TRIAGE_MAX_TOKENS",
    "TRIAGE_MODEL",
    "LlmError",
    "TriageBatchResult",
    "TriageResult",
    "get_anthropic_client",
    "run_triage_batch",
]

logger = logging.getLogger(__name__)

TRIAGE_MODEL: Final = "claude-haiku-4-5"
"""Triage/scoring model (CLAUDE.md §6). Never Sonnet/Opus on a per-item path —
those are reserved for the 1-2 weekly/monthly synthesis calls (Phase 1+)."""

TRIAGE_BATCH_SIZE: Final = 25
"""Items grouped into one Messages request within the batch (DESIGN §8)."""

TRIAGE_MAX_TOKENS: Final = 4096
"""Output ceiling per batch request — a group of 25 short reasoning strings."""

_DEFAULT_POLL_INTERVAL_SECONDS: Final = 5.0
_DEFAULT_MAX_POLL_SECONDS: Final = 24 * 3600.0
"""Batches complete within 24h at the outside (Anthropic Batches API limit).

Known Phase 0 limitation: `run_triage_batch` submits and polls to completion
in one blocking call, so a slow batch (Anthropic's own guidance: "most
complete within 1 hour") holds up whatever invoked `signalforge score` — in
DESIGN §14's `ingest→score→daily` cron chain, that means the digest step too.
At Phase 0 volumes (~100-300 items/day) this is expected to resolve in
minutes, so the straightforward submit-and-wait shape was chosen over a
submit/collect split across two cron ticks; revisit if batches start taking
long enough to matter."""

_ANTHROPIC_API_KEY_ENV: Final = "ANTHROPIC_API_KEY"

_TRIAGE_OUTPUT_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "integer"},
                    "triage": {"type": "string", "enum": ["keep", "kill"]},
                    "signal": {"type": "integer"},
                    "relevance": {"type": "integer"},
                    "novelty": {"type": "integer"},
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "item_id",
                    "triage",
                    "signal",
                    "relevance",
                    "novelty",
                    "reasoning",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}
"""Structured-output schema. Numeric bounds (1-5) aren't expressible here per
the JSON-schema constraint list this SDK supports — `TriageResult` validates
them; a value outside 1-5 becomes a per-item error, not a stored score."""


class LlmError(Exception):
    """Any failure calling the Anthropic API — auth, network, batch-level error.

    Callers (`score/`) catch this and record it into `runs.errors` rather than
    letting it abort the run (CLAUDE.md §7, NEVER rule 12).
    """


class TriageResult(BaseModel):
    """One item's triage output — the `scores` row shape (DESIGN §5) minus the
    bookkeeping fields the caller attaches: `rubric_version`, `model`,
    `scored_at`. Kept out of this model on purpose, so `rubric_version` stays
    sourced from the `score/rubrics.py` constant rather than free-floating
    inside an LLM response the model could (in principle) hallucinate.
    """

    model_config = ConfigDict(extra="forbid")

    triage: Literal["keep", "kill"]
    signal: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    reasoning: str = Field(min_length=1)
    """One-paragraph why — always stored, per DESIGN §5's `reasoning NOT NULL`."""


@dataclass(slots=True)
class TriageBatchResult:
    """Per-item results, per-item errors, and the token spend that produced
    them — in one return value, so persisting scores without also reading
    `input_tokens`/`output_tokens` takes an extra line of code to skip, not
    zero (NEVER rule 11: no call may bypass token accounting).
    """

    results: dict[int, TriageResult] = field(default_factory=dict)
    """`item_id -> TriageResult`, for items whose batch entry parsed cleanly."""

    errors: dict[int, str] = field(default_factory=dict)
    """`item_id -> message`, for items whose request or response failed."""

    input_tokens: int = 0
    output_tokens: int = 0


def get_anthropic_client() -> anthropic.Anthropic:
    """Build the Anthropic client from the configured secret.

    Uses `config.get_secret` — the one mechanism for reading credentials
    (CLAUDE.md §16 NEVER rule; no second secrets path is invented here).
    Raises `LlmError` rather than a bare `KeyError`/`TypeError` so callers have
    one exception type to catch across every failure this module can produce.
    """
    secret = get_secret(_ANTHROPIC_API_KEY_ENV)
    if secret is None:
        raise LlmError(f"{_ANTHROPIC_API_KEY_ENV} is not set")
    return anthropic.Anthropic(api_key=secret.get_secret_value())


def _chunk(
    items: Sequence[tuple[int, str, str | None]], size: int
) -> list[list[tuple[int, str, str | None]]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _build_batch_request(
    custom_id: str,
    group: Sequence[tuple[int, str, str | None]],
    *,
    system_prompt: str,
    model: str,
) -> Request:
    """One Messages request scoring ~25 items — title + summary only, never
    `content` (NEVER rule 9): the tuple shape callers pass in makes it
    structurally impossible to leak full article text into this prompt.
    """
    payload = [
        {"item_id": item_id, "title": title, "summary": summary or ""}
        for item_id, title, summary in group
    ]
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=TRIAGE_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    # Frozen rubric + interests.yaml — no volatile data, so this
                    # prefix is identical across every batch, every day
                    # (DESIGN §8 caching discipline, NEVER rule 10).
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Triage and score each of these items. Return exactly one "
                        "result per item in `results`, each carrying the `item_id` "
                        "it was given so it can be matched back:\n"
                        + json.dumps(payload, sort_keys=True)
                    ),
                }
            ],
            output_config={"format": {"type": "json_schema", "schema": _TRIAGE_OUTPUT_SCHEMA}},
        ),
    )


def _record_group_error(outcome: TriageBatchResult, item_ids: Sequence[int], message: str) -> None:
    for item_id in item_ids:
        outcome.errors[item_id] = message


def _apply_batch_entry(
    entry: MessageBatchIndividualResponse,
    outcome: TriageBatchResult,
    *,
    expected_item_ids: Sequence[int],
) -> None:
    """Fold one Batches API result (one group of ~25 items) into `outcome`.

    Token counts are recorded whenever a message actually came back — even one
    that fails to parse still spent tokens, and that spend must never vanish
    from `runs.llm_input_tokens`/`llm_output_tokens` (NEVER rule 11).
    """
    result = entry.result
    custom_id = entry.custom_id

    if result.type != "succeeded":
        _record_group_error(outcome, expected_item_ids, f"batch request {result.type}")
        logger.warning(
            "triage batch request did not succeed",
            extra={"custom_id": custom_id, "status": result.type},
        )
        return

    message = result.message
    usage = message.usage
    outcome.input_tokens += (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )
    outcome.output_tokens += usage.output_tokens

    text = next((block.text for block in message.content if block.type == "text"), None)
    if text is None:
        _record_group_error(outcome, expected_item_ids, "no text content in response")
        return

    try:
        decoded = json.loads(text)
        raw_results = decoded["results"]
        if not isinstance(raw_results, list):
            raise TypeError("'results' is not a list")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _record_group_error(outcome, expected_item_ids, f"malformed batch response: {exc}")
        return

    seen: set[int] = set()
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        item_id = raw.get("item_id")
        if not isinstance(item_id, int):
            continue
        seen.add(item_id)
        # `TriageResult` deliberately has no `item_id` field (it is matching
        # metadata, not a scored dimension) and is `extra="forbid"`, so it must
        # be stripped before validation rather than merely ignored.
        fields = {key: value for key, value in raw.items() if key != "item_id"}
        try:
            outcome.results[item_id] = TriageResult.model_validate(fields)
        except ValidationError as exc:
            outcome.errors[item_id] = f"schema validation failed: {exc}"

    for item_id in expected_item_ids:
        if item_id not in seen:
            outcome.errors[item_id] = "missing from batch response"


def run_triage_batch(
    items: Sequence[tuple[int, str, str | None]],
    interests: InterestsConfig,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = TRIAGE_MODEL,
    batch_size: int = TRIAGE_BATCH_SIZE,
    poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    max_poll_seconds: float = _DEFAULT_MAX_POLL_SECONDS,
) -> TriageBatchResult:
    """Triage + score `items` via the Batches API. Titles + summaries only.

    `items` is `(item_id, title, summary)` tuples — deliberately not `Item`, so
    a caller cannot accidentally pass `content` into a triage prompt (NEVER
    rule 9). `interests` is rendered (and cache-controlled) into the system
    prompt by `score.rubrics.build_triage_system_prompt`.

    Pass `client` in tests to fake the Anthropic Batches API at this module's
    boundary (CLAUDE.md §8 — never call the real API in tests). Raises
    `LlmError` on any API failure; callers decide how that affects a run.
    """
    outcome = TriageBatchResult()
    if not items:
        return outcome

    active_client = client if client is not None else get_anthropic_client()
    system_prompt = build_triage_system_prompt(interests)

    groups = _chunk(list(items), batch_size)
    id_by_custom_id: dict[str, list[int]] = {}
    requests: list[Request] = []
    for index, group in enumerate(groups):
        custom_id = f"triage-{index}"
        id_by_custom_id[custom_id] = [item_id for item_id, _, _ in group]
        requests.append(
            _build_batch_request(custom_id, group, system_prompt=system_prompt, model=model)
        )

    try:
        batch = active_client.messages.batches.create(requests=requests)
    except anthropic.APIError as exc:
        raise LlmError(f"failed to create triage batch: {exc}") from exc

    elapsed = 0.0
    while batch.processing_status != "ended":
        if elapsed >= max_poll_seconds:
            raise LlmError(
                f"triage batch {batch.id} did not complete within {max_poll_seconds:.0f}s"
            )
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            batch = active_client.messages.batches.retrieve(batch.id)
        except anthropic.APIError as exc:
            raise LlmError(f"failed to poll triage batch {batch.id}: {exc}") from exc

    try:
        results = list(active_client.messages.batches.results(batch.id))
    except anthropic.APIError as exc:
        raise LlmError(f"failed to fetch triage batch results: {exc}") from exc

    for entry in results:
        custom_id = entry.custom_id
        _apply_batch_entry(entry, outcome, expected_item_ids=id_by_custom_id.get(custom_id, []))

    logger.info(
        "triage batch complete",
        extra={
            "item_count": len(items),
            "group_count": len(groups),
            "scored_count": len(outcome.results),
            "error_count": len(outcome.errors),
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
        },
    )
    return outcome
