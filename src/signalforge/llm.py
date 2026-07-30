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
from anthropic.types import (
    OutputConfigParam,
    ToolParam,
    ToolUnionParam,
    ToolUseBlock,
    WebSearchTool20260318Param,
)
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from signalforge.config import InterestsConfig, get_secret
from signalforge.models import ProposalKind, ProposalTier
from signalforge.score.rubrics import build_triage_system_prompt

__all__ = [
    "SCOUT_EFFORT",
    "SCOUT_MAX_SEARCHES_CEILING",
    "SCOUT_MAX_TOKENS",
    "SCOUT_MODEL",
    "SCOUT_MONTHLY_CEILING_USD",
    "SCOUT_PROPOSE_TOOL_NAME",
    "TRIAGE_BATCH_SIZE",
    "TRIAGE_MAX_TOKENS",
    "TRIAGE_MODEL",
    "WEB_SEARCH_USD_PER_REQUEST",
    "LlmError",
    "ScoutEvidence",
    "ScoutProposal",
    "ScoutResult",
    "TriageBatchResult",
    "TriageResult",
    "get_anthropic_client",
    "run_source_scout",
    "run_triage_batch",
]

logger = logging.getLogger(__name__)

TRIAGE_MODEL: Final = "claude-haiku-4-5"
"""Triage/scoring model (CLAUDE.md §6). Never Sonnet/Opus on a per-item path —
those are reserved for the 1-2 weekly/monthly synthesis calls (Phase 1+)."""

SCOUT_MODEL: Final = "claude-opus-5"
"""Source-curation scout (DESIGN §7.1, §8).

Opus on a **weekly, single-call** path — which is the existing Opus slot in §8's
budget table, not a new one, and emphatically not a per-item path. The question it
answers ("who is worth reading on these topics now, and which sources have gone
quiet") is judgment over a whole corpus at once; there is no per-item version of
it to make cheap."""

SCOUT_MAX_SEARCHES_CEILING: Final = 7
"""Hard cap on web searches per scout run, regardless of config.

Web search bills **per call** ($10/1,000) on top of the tokens its results consume,
so this is a spend limit, not a quality knob. `curation.max_searches_per_run` can
only lower it. Living here rather than in `config.py` is deliberate — model and cost
decisions belong to this module and nowhere else (CLAUDE.md §6) — and `config.py`
could not import it anyway without a cycle.

**7, chosen so that the ceiling is consistent with the budget.** It was 15, which
was not: the worst case at 15 searches is ~$3.13/month even on optimistic
assumptions, so any config value the code accepted between 7 and 15 would breach
`SCOUT_MONTHLY_CEILING_USD` while every test stayed green. A ceiling that permits
values the budget forbids is not a ceiling. 7 leaves one search of headroom above
the shipped 6, which is enough for the backstop job this constant exists to do —
catching a typo like `60`, not enabling a bigger appetite.

Wanting more searches is therefore a deliberate budget decision: raise
`SCOUT_MONTHLY_CEILING_USD`, re-run the arithmetic, and raise this. The test that
guards the ceiling will tell you if the sum stops working."""

SCOUT_MAX_TOKENS: Final = 10240
"""Output ceiling for one scout call: thinking plus a handful of proposals.

Output is billed for what is produced, not for the ceiling — but the ceiling is
still the *bound*, and on a single-request call at $25/M it is the largest term in
the worst case (~$0.26 of a ~$0.41 run). So it is set by two competing constraints
rather than by generosity:

* high enough that a run which has already paid for its searches does not truncate
  before the tool call lands, which would waste the whole run. Thinking counts
  against this on Opus 5, and 8192 was judged too tight for that reason.
* low enough that the worst case fits `SCOUT_MONTHLY_CEILING_USD` **at the search
  ceiling and on a pessimistic input assumption** — see the test.

10240 satisfies both with margin: a 5-proposal tool call is under 1k tokens, so this
leaves ~9k for thinking at `effort: "high"`."""

SCOUT_EFFORT: Final = "high"
"""Reasoning effort for the scout call.

`high` rather than `xhigh`/`max`: this is a judgment task over evidence that has
already been gathered, not a long-horizon agentic one, and the output is read by a
human who can reject it. The cheaper setting is also the one that keeps the weekly
figure inside §8's estimate."""

_WEB_SEARCH_TOOL_TYPE: Final = "web_search_20260318"
"""Web search with dynamic filtering and `response_inclusion`.

Dynamic filtering (available from `web_search_20260209`) has the model filter
results *before* they enter the context window, which is the single biggest lever
on this call's input tokens. `code_execution` is deliberately **not** declared
alongside it: on this tool version the API provisions the execution it needs
itself, and declaring a second execution environment confuses the model."""

SCOUT_PROPOSE_TOOL_NAME: Final = "propose_source_changes"
"""Name of the tool the scout calls to return its proposals.

Public so `curate/prompts.py` can name it in the instructions without keeping a
second copy of the string — the prompt telling the model to call a tool that does
not exist is a silent, expensive failure."""

WEB_SEARCH_USD_PER_REQUEST: Final = 0.01
"""What one web search costs: $10 per 1,000 requests (DESIGN §8).

Here because cost belongs to this module and nowhere else (CLAUDE.md §6), even
though the only consumer is a `status` readout. Reached by the CLI through
`curate.scout.search_spend_usd` rather than imported directly — `cli.py` does not
import this module, deliberately."""

SCOUT_MONTHLY_CEILING_USD: Final = 2.50
"""The agreed monthly budget for this feature, recorded next to the constants that
enforce it (DESIGN §7.1, §8).

Here rather than only in a doc because the two knobs below are tuned *to* it: a
future editor raising `curation.max_searches_per_run` needs the number in front of
them. Enforced by three facts, all checkable from code:

* exactly one API request per run — no resumes (see `run_source_scout`);
* `max_uses` caps searches server-side at `SCOUT_MAX_SEARCHES_CEILING` or below;
* `SCOUT_MAX_TOKENS` caps output for that single request.

The worst case is deliberately **not written out here.** Every arithmetic error found
while reviewing this feature was a figure that had drifted from what it described,
including an earlier version of this very docstring. It is computed instead, by
`test_the_worst_case_cost_stays_within_the_recorded_ceiling`, which sums four terms
it *reads* rather than assumes:

    input   the rendered prompt (measured from curate/prompts.py, at full evidence)
            + SCOUT_MAX_SEARCHES_CEILING x a pessimistic 6k tokens per search
    output  SCOUT_MAX_TOKENS, the enforced ceiling
    search  SCOUT_MAX_SEARCHES_CEILING x $0.01
    x 4.33 weeks, asserted <= this budget

Only the per-search input volume is an assumption, and it is pitched at 3x what
dynamic filtering is expected to deliver, because it is the one term no code
enforces. Raising a knob, editing the config, or growing the prompt past what this
budget affords fails that test rather than the invoice. A realistic run — the shipped
6 searches, expected search volume, ~4k of output — is ~$1.00/month."""

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


# --------------------------------------------------------------------------- #
# source curation scout (DESIGN §7.1)
# --------------------------------------------------------------------------- #


class ScoutEvidence(BaseModel):
    """One citation behind a proposal. At least one is mandatory (NEVER rule 7)."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    note: str = ""


class ScoutProposal(BaseModel):
    """One proposed `sources.yaml` change, as the model returned it.

    Validated here rather than trusted: this is the boundary where model output
    becomes something the pipeline will write into the operator's config, so the
    shape is checked before it can reach `db.insert_proposal`. `kind` is validated
    against `ProposalKind` — an invented kind is a rejected proposal, never a new
    edit shape the applier has never heard of.

    `weight` is a *suggestion*, and bounded (`curate/scout.py` clamps it). The
    operator asked for it on the reasoning that a scout arguing "this author is
    worth trusting" should be able to say so, and that the number is trivial to
    change while approving — it renders in the digest block and lands in an
    uncommitted `sources.yaml` diff either way. The bound exists because that
    review is a human skimming over coffee: a visible 1.3 gets judged, whereas
    nothing in the loop would catch a quietly-proposed 9.0 before it reweighted a
    source. Ongoing weight *tuning* is still Phase 2's `tune` job under DESIGN §11's
    ±0.1/month cap; this is only the value a new source starts at.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ProposalKind
    target: str = Field(min_length=1)
    """What is being added or retired: a feed URL, an `owner/repo` slug, or a
    keyword. Normalized into `proposals.dedup_key` by the caller, not here — the
    normalization differs per kind and belongs next to the code that knows which."""

    source_id: str | None = None
    """Proposed `sources.yaml` id for an added RSS feed. Absent for other kinds."""

    url: str | None = None
    """The feed URL for an added RSS source, when `target` is not itself the URL."""

    weight: float | None = Field(default=None, gt=0)
    """Suggested score multiplier for an added feed. Clamped by the caller; absent
    means the identity element (1.0), which is what most additions should get."""

    rationale: str = Field(min_length=1)
    evidence: list[ScoutEvidence] = Field(min_length=1)
    """Non-empty by validation. `db.insert_proposal` refuses an uncited proposal,
    so catching it here turns a wasted paid slot into a recorded error."""

    tier: ProposalTier = ProposalTier.WEB


@dataclass(slots=True)
class ScoutResult:
    """Proposals, per-proposal errors, and everything the run spent.

    Same shape and reasoning as `TriageBatchResult`: the spend comes back in the
    same value as the output, so persisting proposals without also seeing what
    they cost takes an extra line of code to skip rather than zero (NEVER rule 11).

    `web_search_requests` is the field the token counters cannot express — search
    bills per call, so a run's cost is not derivable from tokens alone.
    """

    proposals: list[ScoutProposal] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    """Per-proposal validation failures and run-level notes (including a clamped
    search budget). The caller writes these into `runs.errors` so they surface in
    the next digest rather than only in cron.log (DESIGN §7)."""

    input_tokens: int = 0
    output_tokens: int = 0
    web_search_requests: int = 0


def _propose_tool_schema(max_proposals: int) -> ToolParam:
    """The tool the scout calls to return its proposals.

    A **tool** rather than `output_config.format`, deliberately. Structured
    outputs are this module's normal habit (see `_TRIAGE_OUTPUT_SCHEMA`), but their
    documented interaction with *server-side* tools is unspecified, and this call
    is the one path in the pipeline that cannot be exercised against the real API
    in tests (NEVER rule 13) — so its first real run is the operator's. A custom
    tool alongside a server tool is explicitly supported, which makes it the
    lower-risk shape for the one thing that has to work first time.

    Nothing executes this tool: the model calling it *is* the answer, and the run
    ends there. `strict` is deliberately absent — `ScoutProposal` re-validates
    every field anyway, so requiring it would add a compatibility surface for a
    guarantee already held elsewhere.
    """
    return ToolParam(
        name=SCOUT_PROPOSE_TOOL_NAME,
        description=(
            "Submit your proposed changes to the operator's sources.yaml. Call this "
            "exactly once, after you have finished searching. Pass an empty list if "
            "the evidence does not support any change this week."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "proposals": {
                    "type": "array",
                    "maxItems": max_proposals,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": [kind.value for kind in ProposalKind],
                            },
                            "target": {
                                "type": "string",
                                "description": (
                                    "The feed URL, owner/repo slug, or keyword being "
                                    "added or retired."
                                ),
                            },
                            "source_id": {
                                "type": "string",
                                "description": (
                                    "Short stable id for an added RSS feed, e.g. "
                                    "'import-ai'. Omit for other kinds."
                                ),
                            },
                            "url": {"type": "string"},
                            "weight": {
                                "type": "number",
                                "description": (
                                    "Optional score multiplier for an added feed, where "
                                    "1.0 means no adjustment. Only propose one when the "
                                    "author's track record specifically justifies it, and "
                                    "keep it near 1.0 — the operator reads this number and "
                                    "will change it if they disagree."
                                ),
                            },
                            "rationale": {
                                "type": "string",
                                "description": (
                                    "One or two sentences for the operator, naming the "
                                    "specific evidence."
                                ),
                            },
                            "evidence": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string"},
                                        "note": {"type": "string"},
                                    },
                                    "required": ["url"],
                                    "additionalProperties": False,
                                },
                            },
                            "tier": {"type": "string", "enum": ["corpus", "web"]},
                        },
                        "required": ["kind", "target", "rationale", "evidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["proposals"],
            "additionalProperties": False,
        },
    )


def _scout_tools(*, searches: int, max_proposals: int) -> list[ToolUnionParam]:
    """The tool list for the run's single request.

    **`max_uses` is per API request, not per logical run**, which is why
    `run_source_scout` makes exactly one. A second request — a `pause_turn` resume —
    would arrive with a fresh full budget unless it were decremented, and would
    multiply the `max_tokens` output ceiling besides. One request removes both
    problems and makes `SCOUT_MAX_SEARCHES_CEILING` a real bound rather than a
    per-request one.

    Floored at 0 so a configured budget of 0 — the supported corpus-only mode — still
    produces a valid tool list, letting the model propose from stored evidence alone.
    """
    return [
        WebSearchTool20260318Param(
            type=_WEB_SEARCH_TOOL_TYPE,
            name="web_search",
            # Enforced server-side, so this is a real limit rather than an
            # instruction the model may ignore.
            max_uses=max(0, searches),
            # Drops the nested search blocks from the assistant message once dynamic
            # filtering has consumed them, which is an output-token saving. Safe
            # because this is a single-request call: nothing needs to echo search
            # content back on a later turn.
            response_inclusion="excluded",
        ),
        _propose_tool_schema(max_proposals),
    ]


def _effective_search_budget(requested: int, outcome: ScoutResult) -> int:
    """Clamp the configured search budget to this module's ceiling, loudly.

    Silently handing back a smaller number than the operator configured is the
    "config that quietly does nothing" failure `_StrictModel` exists to prevent —
    they would set 100, get 15, and never learn. So the clamp is recorded into
    `ScoutResult.errors`, which the caller writes to `runs.errors`, which the next
    digest surfaces (DESIGN §7's monitoring channel is the reports, not cron.log).
    """
    if requested <= SCOUT_MAX_SEARCHES_CEILING:
        return requested
    message = (
        f"curation.max_searches_per_run is {requested}, above llm.py's ceiling of "
        f"{SCOUT_MAX_SEARCHES_CEILING}; using {SCOUT_MAX_SEARCHES_CEILING}. Lower the "
        "config value to silence this."
    )
    logger.warning("clamped the scout search budget", extra={"requested": requested})
    outcome.errors.append(message)
    return SCOUT_MAX_SEARCHES_CEILING


def _accumulate_usage(outcome: ScoutResult, usage: object) -> None:
    """Fold one response's usage into the running total.

    Every attempt counts, including a paused turn that produced no proposals —
    those searches and tokens were spent and must not vanish from the run's
    accounting (NEVER rule 11). Attribute access is guarded because `usage` shapes
    grow over time and a missing field must not lose the fields that are present.
    """
    outcome.input_tokens += (
        int(getattr(usage, "input_tokens", 0) or 0)
        + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    )
    outcome.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
    server_tool_use = getattr(usage, "server_tool_use", None)
    if server_tool_use is not None:
        outcome.web_search_requests += int(getattr(server_tool_use, "web_search_requests", 0) or 0)


def _collect_proposals(tool_input: object, outcome: ScoutResult) -> None:
    """Validate the model's tool input into `ScoutProposal`s.

    One bad proposal is recorded and skipped rather than failing the run: the
    searches are already paid for, so discarding four good suggestions because a
    fifth omitted its citation would waste the money twice (CLAUDE.md §7).
    """
    raw_proposals = tool_input.get("proposals") if isinstance(tool_input, dict) else None
    if not isinstance(raw_proposals, list):
        outcome.errors.append(f"{SCOUT_PROPOSE_TOOL_NAME} input had no 'proposals' list")
        return
    for index, raw in enumerate(raw_proposals):
        try:
            outcome.proposals.append(ScoutProposal.model_validate(raw))
        except ValidationError as exc:
            outcome.errors.append(f"proposal {index} failed validation: {exc}")


def run_source_scout(
    *,
    system_prompt: str,
    user_prompt: str,
    max_searches: int,
    max_proposals: int,
    client: anthropic.Anthropic | None = None,
    model: str = SCOUT_MODEL,
) -> ScoutResult:
    """Ask the scout for `sources.yaml` changes, with web search, once.

    Prompt text is built by `curate/prompts.py` and passed in — this module never
    writes prompts (see the module docstring).

    **No `cache_control` anywhere in this call, deliberately.** The system prompt is
    stable enough to cache and caching it would still be wrong: the call runs once
    a week, every cache entry has expired long before the next one, so a breakpoint
    would pay the ~1.25x write premium for exactly zero reads. This is the one
    place in the pipeline where the caching discipline is correctly inverted; do
    not "fix" it (DESIGN §8).

    **Exactly one API request. A paused turn is not resumed.** That is a cost
    decision, and it is what makes this call's ceiling provable rather than estimated.
    `max_tokens` bounds output *per request*, so resuming multiplies it — enough to
    put the enforced ceiling several times past this feature's budget, and no
    `max_tokens` low enough to fix that is high enough to avoid truncating a run that
    has already paid for its searches. One request bounds the worst case from code
    constants alone; the arithmetic lives with those constants, in
    `SCOUT_MONTHLY_CEILING_USD`, rather than being restated here where it would drift
    out of step with them.

    What that costs: a turn that pauses yields nothing that week. Acceptable because
    searches are capped low enough that the server-side loop rarely reaches a pause,
    the pause is recorded and surfaces in the next digest, and the run happens again
    next Sunday. If pauses turn out to be common, that will be visible in
    `runs.errors` and is the moment to revisit with real data.

    **Never raises for an API failure** — deliberate, and unlike `run_triage_batch`.
    Searches and tokens are spent before the response is read, so an exception would
    take the only record of that spend with it. Everything — a clamped budget, a
    failed request, a pause, an invalid proposal, a turn that never called the
    tool — comes back inside `ScoutResult` for the caller to record. The one
    `LlmError` still possible comes from `get_anthropic_client`, which fails before
    anything has been spent.
    """
    outcome = ScoutResult()
    active_client = client if client is not None else get_anthropic_client()
    budget = _effective_search_budget(max_searches, outcome)

    try:
        response = active_client.messages.create(
            model=model,
            max_tokens=SCOUT_MAX_TOKENS,
            system=system_prompt,
            tools=_scout_tools(searches=budget, max_proposals=max_proposals),
            output_config=OutputConfigParam(effort=SCOUT_EFFORT),
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.APIError as exc:
        # Recorded, not raised: see the docstring. Searches may already have run.
        logger.warning("source scout call failed", extra={"error": str(exc)})
        outcome.errors.append(f"source scout call failed: {exc}")
        return _finished(outcome, budget)

    _accumulate_usage(outcome, response.usage)

    # `isinstance` rather than duck-typing on `.type`: the content union holds a
    # dozen block types (server tool results, thinking, code execution) and only a
    # real `ToolUseBlock` carries `.input`.
    tool_calls = [
        block
        for block in response.content
        if isinstance(block, ToolUseBlock) and block.name == SCOUT_PROPOSE_TOOL_NAME
    ]
    if tool_calls:
        # The model calling the tool *is* the answer; nothing executes it and no
        # tool_result goes back, so the exchange ends here.
        _collect_proposals(tool_calls[0].input, outcome)
    elif response.stop_reason == "pause_turn":
        outcome.errors.append(
            "scout turn paused mid-search and is not resumed (resuming multiplies the "
            "per-request output ceiling past this feature's budget); nothing proposed "
            "this run"
        )
    elif response.stop_reason == "refusal":
        outcome.errors.append("scout call was refused by safety classifiers")
    else:
        # A legitimate "nothing to propose" should have been an empty tool call, so
        # this is worth recording rather than reading as a quiet week.
        outcome.errors.append(
            f"scout finished with stop_reason={response.stop_reason!r} without calling "
            f"{SCOUT_PROPOSE_TOOL_NAME}"
        )
    return _finished(outcome, budget)


def _finished(outcome: ScoutResult, budget: int) -> ScoutResult:
    """Log the run's shape and spend, then hand it back."""
    logger.info(
        "source scout complete",
        extra={
            "proposal_count": len(outcome.proposals),
            "error_count": len(outcome.errors),
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "web_search_requests": outcome.web_search_requests,
            "search_budget": budget,
        },
    )
    return outcome
