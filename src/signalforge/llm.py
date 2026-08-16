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
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
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
    "PODCAST_EFFORT",
    "PODCAST_MAX_ITEM_CONTENT_BYTES",
    "PODCAST_MAX_ITEM_SUMMARY_BYTES",
    "PODCAST_MAX_ITEM_TITLE_BYTES",
    "PODCAST_MAX_ITEMS",
    "PODCAST_MAX_SCRIPT_CHARS",
    "PODCAST_MAX_TOKENS",
    "PODCAST_MODEL",
    "PODCAST_MONTHLY_CEILING_USD",
    "PODCAST_RETRY_MAX_TOKENS",
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
    "WEEKLY_EFFORT",
    "WEEKLY_MAX_ITEMS",
    "WEEKLY_MAX_ITEM_REASONING_BYTES",
    "WEEKLY_MAX_ITEM_SUMMARY_BYTES",
    "WEEKLY_MAX_ITEM_TITLE_BYTES",
    "WEEKLY_MAX_CLUSTERS",
    "WEEKLY_MAX_LEADS",
    "WEEKLY_MAX_TOKENS",
    "WEEKLY_MODEL",
    "WEEKLY_MONTHLY_CEILING_USD",
    "LlmError",
    "PodcastScript",
    "PodcastScriptResult",
    "PodcastSegment",
    "PodcastTurn",
    "ScoutEvidence",
    "ScoutProposal",
    "ScoutResult",
    "TriageBatchResult",
    "TriageResult",
    "WeeklyBrief",
    "WeeklyBriefResult",
    "WeeklyCluster",
    "WeeklyLead",
    "get_anthropic_client",
    "run_podcast_script",
    "run_source_scout",
    "run_triage_batch",
    "run_weekly_brief",
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

SCOUT_MAX_SEARCHES_CEILING: Final = 12
"""Hard cap on web searches per scout run, regardless of config.

Web search bills **per call** ($10/1,000) on top of the tokens its results consume,
so this is a spend limit, not a quality knob. `curation.max_searches_per_run` can
only lower it. Living here rather than in `config.py` is deliberate — model and cost
decisions belong to this module and nowhere else (CLAUDE.md §6) — and `config.py`
could not import it anyway without a cycle.

**12, a deliberate operator decision made after three real runs, not a default.**
It was 7, chosen only so the ceiling would not exceed a $2.50/month budget set
before any real run existed. Three runs on 2026-07-30/31 (6, 6, and 7 searches)
measured 22k-31k input tokens per search — 4-5x the 6k/search this feature's budget
had assumed — and the operator judged the later runs' results a real improvement
and wanted more research depth per run, not more proposals (`max_proposals_per_run`
is the separate, unchanged knob for that). 12 sits under the 15 this scout's search
tool was originally designed around, leaving headroom for a genuine typo (`60`) to
still be caught while affording real research breadth: enough to check a handful of
addition candidates are still active and cross-reference a few retirement calls,
without so many that results dilute rather than sharpen the proposals.

`SCOUT_MONTHLY_CEILING_USD` was raised alongside this to the real worst case at 12
searches on the corrected per-search assumption — raising one without the other is
exactly the "ceiling permits values the budget forbids" failure this constant exists
to prevent. Wanting more still means: re-run the arithmetic, get the operator's
sign-off, raise both. The test that guards the ceiling will tell you if the sum
stops working."""

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

SCOUT_MONTHLY_CEILING_USD: Final = 13.00
"""The agreed monthly budget for this feature, recorded next to the constants that
enforce it (DESIGN §7.1, §8).

Raised from $2.50 on 2026-07-31, the same day `SCOUT_MAX_SEARCHES_CEILING` went
from 7 to 12 — see that docstring for why. This is the operator's deliberate
decision, not a code default: three real runs cost ~$1.00-1.20 each at 6-7 searches,
and weekly-times-more-research-depth was judged worth paying for, against a stated
tolerance of "a dollar a run, still under a cup of coffee a month." Still small
against the $50 whole-pipeline alarm (CLAUDE.md §6), but no longer small against
this project's overall $5-10/month target — this one feature can now cost as much
as the rest of the pipeline combined, which is worth knowing next time total spend
is reviewed.

Set to the worst case at `SCOUT_MAX_SEARCHES_CEILING`'s new value with real margin,
not scraped to the edge of it: an initial version of this raise set both this and
`PESSIMISTIC_TOKENS_PER_SEARCH` right at the highest of only two known real runs,
leaving 2-5% headroom — and a cost-guard review then found a *third* real run
already sitting in `runs`, higher than either. Three samples spanning ~22k-31k
tokens/search in one day is not a distribution to price at its observed edge.

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
            + SCOUT_MAX_SEARCHES_CEILING x a pessimistic per-search token figure
    output  SCOUT_MAX_TOKENS, the enforced ceiling
    search  SCOUT_MAX_SEARCHES_CEILING x $0.01
    x 4.33 weeks, asserted <= this budget

The per-search input figure is the one term no code enforces, and it used to be a
guess (6k, "3x what dynamic filtering is expected to deliver") made before any real
run existed. It is now grounded in the three real runs that guess was wrong about,
priced with margin above their observed max rather than at it —
see `PESSIMISTIC_TOKENS_PER_SEARCH` in the test. Raising a knob, editing the config,
or growing the prompt past what this budget affords fails that test rather than the
invoice."""

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


# --------------------------------------------------------------------------- #
# podcast script (DESIGN §13.3 — the second recorded phase-gate exception)
# --------------------------------------------------------------------------- #


PODCAST_MODEL: Final = "claude-opus-5"
"""Same family as the scout (`SCOUT_MODEL`) — the existing Opus slot in §8's
budget table, not a new model tier. Writing a two-presenter dialogue that
cites specific items and stays in character is judgment, not a per-item
task, the same reasoning that puts the scout on Opus rather than Haiku."""

PODCAST_EFFORT: Final = "medium"
"""Lower than `SCOUT_EFFORT`'s `high`, deliberately: this is dialogue
assembly over evidence (`items`) already gathered by the caller, not a
search-and-judge task, and thinking is billed as output ($25/M, on top of
counting against `PODCAST_MAX_TOKENS` — the same fact `SCOUT_MAX_TOKENS`'s
docstring records). A daily call pays for its effort level thirty times a
month where the scout's weekly one pays 4.33 times, so the cheaper setting
that still produces a human-reviewable script (Stage 3's stop-and-ask, then
every day via the vault file) is the right default here even though it
would not be for a harder reasoning task."""

PODCAST_MAX_TOKENS: Final = 8192
"""Output ceiling for the first script call. Thinking counts against this
on Opus 5 (see `SCOUT_MAX_TOKENS`'s docstring for the same fact, learned the
hard way there) — if truncation from thinking proves common in practice,
raise this alongside `PODCAST_MONTHLY_CEILING_USD`. Kept at the plan's
original figure rather than pre-emptively raised, because Stage 3's
stop-and-ask (one real script, operator reads it before any TTS money is
wired) is exactly the checkpoint that would surface a too-tight ceiling
before it costs anything beyond this call."""

PODCAST_RETRY_MAX_TOKENS: Final = 4096
"""Output ceiling for the one "shorter" retry `run_podcast_script(shorter=True)`
makes. Deliberately half of `PODCAST_MAX_TOKENS`: the retry is asking for
*less* dialogue than the first attempt already produced too much of, so a
tighter cap is a legitimate brevity forcing-function as well as a cost one
— it is the lever that keeps a script this feature retries on from costing
close to double every time (see `PODCAST_MONTHLY_CEILING_USD`'s worked
worst case, which assumes the retry happens on *every* call as the
pessimistic case, exactly the assumption CLAUDE.md §6 requires)."""

PODCAST_MAX_ITEMS: Final = 6
"""Hard cap on items in one script call, regardless of what
`thresholds.podcast_top_n` is configured to (DESIGN's `interests.yaml`
deliberately leaves that value unbounded, the same "single-user tunable"
reasoning `daily_max_items` uses — CLAUDE.md §4). This is `podcast_top_n`'s
`SCOUT_MAX_SEARCHES_CEILING`: a spend limit enforced in code, in the one
module money decisions belong to (CLAUDE.md §6), that config can only lower.
`_build_podcast_user_prompt` clamps to it and returns both the drop count
and which ids actually made it into the prompt — the caller must treat the
returned id set, never its own input list, as "what the model could
legitimately cite" (see `PodcastScriptResult.sent_item_ids`)."""

PODCAST_MAX_ITEM_SUMMARY_BYTES: Final = 500
"""Per-item summary ceiling *inside the prompt*, in UTF-8 bytes as billed
after JSON escaping — not Python characters, and not raw UTF-8 bytes either
(see `_truncate_utf8_json_safe`'s docstring for why the unit matters).
`content` is not the only field an operator-editable config can inflate:
`sources.yaml`'s `defaults.max_summary_chars` bounds a feed summary only
with `gt=0` (`config.SourceDefaults`) — an operator raising it to, say,
8,000 is a pure YAML edit no test rejects, and without this cap that edit
would silently move real spend past every ceiling below, exactly the "a
config edit moves spend without failing anything" failure DESIGN §8 warns
about. Concrete version of the mistake this constant exists to prevent: an
earlier version of `_build_podcast_user_prompt` capped `content` and left
`summary` and `title` uncapped, and the shipped `max_summary_chars: 4000`
priced against that gap landed the worst case inside 1% of the ceiling."""

PODCAST_MAX_ITEM_TITLE_BYTES: Final = 300
"""Per-item title ceiling *inside the prompt*, in UTF-8 bytes, same
reasoning as `PODCAST_MAX_ITEM_SUMMARY_BYTES` — a feed's own title has no
length bound anywhere upstream (`models.Item.title` only flattens it to one
line, NEVER rule 17; it never truncates). 300 is generous for any real
headline and still closes the unbounded-input gap for the worst-case
arithmetic."""

PODCAST_MAX_ITEM_CONTENT_BYTES: Final = 8_000
"""Per-item content ceiling *inside the prompt*, in UTF-8 bytes — distinct
from, and tighter than, `ingest.fullcontent.MAX_FULL_CONTENT_CHARS`
(25,000 *characters*; that module's own truncation-unit gap is
pre-existing and out of this stage's scope), which bounds what gets
stored. Storage and prompt cost are different budgets: a kept article can
be worth archiving in full while still being worth trimming before it
becomes Opus input every day. 8,000 sits comfortably above the 2026-08-06
measurement's per-item average (~7,750 chars, that measurement itself in
English) so real scripts rarely feel it, while keeping `PODCAST_MAX_ITEMS`
items' worst case (every item at this ceiling, not the average) affordable
at daily cadence.

**Billed bytes, not characters and not raw UTF-8, deliberately** — two
separate gaps closed after the per-field caps themselves existed. First: a
character-count cap bounds Python string length, not what gets billed —
`json.dumps` with its default `ensure_ascii=True` re-encodes every
non-ASCII character as a 6-character `\\uXXXX` escape *after* a
character-based slice, so a non-Latin-script feed could cost up to ~6x this
cap's intended size; `run_podcast_script` sets `ensure_ascii=False` to
close that. Second: even with `ensure_ascii=False`, JSON still escapes
`"`, `\\`, and every C0 control character — a raw `\\n` or an unstripped
control byte in feed content expands after the byte-based cap already ran,
measured to land as high as $71/month against this ceiling from this field
alone, over 3x the ceiling and past the $50 whole-pipeline alarm by itself.
`_truncate_utf8_json_safe` bisects on the actual `json.dumps(...,
ensure_ascii=False)`-encoded length, so the bytes this constant bounds are
the bytes that reach the API, regardless of script or embedded escapes."""

PODCAST_MONTHLY_CEILING_USD: Final = 23.00
"""The operator's deliberate decision, made the same way `SCOUT_MONTHLY_CEILING_USD`
was: `docs/plans/podcast-channel.md` wrote down "~$6" before anyone had done the
arithmetic (that constant's own docstring records the identical shape — a
pre-arithmetic $2.50 corrected to a computed $13 once a cost-guard review priced
the real worst case). Not near $6 here either, and this figure has moved once
so far — from $17 to $23 — across five separate `llm-cost-guard` rounds, each
one finding a new way the arithmetic behind it was still optimistic rather
than pessimistic:

1. The "shorter" retry is a second Opus call the *code* permits on every
   single run — worst case prices two calls a day, not one.
2. Every field of every item (`title`, `summary`, `content`) has to be
   capped, because `sources.yaml`'s `max_summary_chars` is operator-editable
   YAML with no upper bound.
3. Those caps, and the worst-case test measuring them, have to be
   denominated in UTF-8 bytes, not Python characters — a character-based cap
   silently assumes ASCII content.
4. The bytes-to-tokens conversion itself has to use a genuinely pessimistic
   ratio for non-Latin scripts (documented Hangul/Devanagari/Thai
   tokenization density, ~1.5 bytes/token), not the *optimistic* ratio an
   earlier version of the worst-case test derived from a fixture where every
   character happened to tokenize at its cheapest possible rate.
5. Capping raw UTF-8 bytes still isn't capping billed bytes: `json.dumps`
   escapes `"`, `\\`, and every C0 control character even with
   `ensure_ascii=False`, and that escaping happens *after* a byte-based
   truncation runs. A cap enforced pre-escaping let one item field alone
   reach ~$71/month before `_truncate_utf8_json_safe` closed it by bisecting
   on the actual escaped, encoded length instead.

Five misses, five `llm-cost-guard` passes, one constant, each time landing on
a real dollar figure rather than an intended one — see
`test_the_worst_case_podcast_cost_stays_within_the_recorded_ceiling`.

**Honest note on the whole-pipeline alarm (CLAUDE.md §6):** this ceiling plus
`SCOUT_MONTHLY_CEILING_USD` ($13) sums to $36/month on paper — both ceilings at
their full margin. Against the $30 alarm this docstring was written under, that
was clearly past the line, and it said so rather than shrinking the show to fit.

**That escalation was answered on 2026-08-16 and the alarm is now $50**, so
$36 sits under it. The number changed; the reasoning did not, and it is kept
here because it is the record of how the band moved: measured spend had reached
≈$29.30/month, two reductions worth ≈$15.40/month were costed and offered, and
the operator declined them and raised the band instead (DESIGN §8). What that
buys is headroom, not permission — the $5–10 *target* is unchanged, and a
ceiling pair summing to $36 is still two thirds of the alarm on paper. The next
new LLM consumer faces exactly the question this note originally raised.

Enforced by six facts, all checkable from code, the same discipline
`SCOUT_MONTHLY_CEILING_USD` uses: at most two non-batch requests per script call
(the first attempt, and the one "shorter" retry `synth.podcast.build_script`
makes when it runs long — never more, and never fewer than the code permits);
`PODCAST_MAX_ITEMS` caps the item count; `PODCAST_MAX_ITEM_TITLE_BYTES`,
`PODCAST_MAX_ITEM_SUMMARY_BYTES`, and `PODCAST_MAX_ITEM_CONTENT_BYTES` cap
every text field of every item, in the actual JSON-escaped UTF-8 bytes that
reach the API (`_truncate_utf8_json_safe`), before either request is built;
`PODCAST_MAX_TOKENS`/`PODCAST_RETRY_MAX_TOKENS` cap output for each.
The worst case is deliberately not written out here — see
`SCOUT_MONTHLY_CEILING_USD`'s docstring for why arithmetic copied into prose
drifts; it is computed instead by the test named above, which reads every
term rather than assuming it, including a pessimistic bytes-per-token ratio
that should be replaced with a grounded one (the same way
`PESSIMISTIC_TOKENS_PER_SEARCH` was grounded in three real scout runs) once
a real script call has actually run."""

_PODCAST_OUTPUT_TURN_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "speaker": {"type": "string", "enum": ["A", "B"]},
        "text": {"type": "string"},
    },
    "required": ["speaker", "text"],
    "additionalProperties": False,
}

_PODCAST_OUTPUT_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "intro_turns": {"type": "array", "items": _PODCAST_OUTPUT_TURN_SCHEMA},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_ids": {"type": "array", "items": {"type": "integer"}},
                    "turns": {"type": "array", "items": _PODCAST_OUTPUT_TURN_SCHEMA},
                },
                "required": ["item_ids", "turns"],
                "additionalProperties": False,
            },
        },
        "outro_turns": {"type": "array", "items": _PODCAST_OUTPUT_TURN_SCHEMA},
    },
    "required": ["intro_turns", "segments", "outro_turns"],
    "additionalProperties": False,
}
"""Structured-output schema. Like `_TRIAGE_OUTPUT_SCHEMA`, deliberately loose
on constraints this SDK's JSON-schema subset can't express (non-empty arrays,
non-empty strings) — `PodcastTurn`/`PodcastSegment`/`PodcastScript` re-validate
those; a violation becomes a parse error recorded on the result, never a
crash and never silently accepted content."""

PODCAST_MAX_SCRIPT_CHARS: Final = 14_000
"""Caps the *generated dialogue only* — never article content. OpenRouter's
TTS request cap is 15,000 chars (`deliver/tts.py`, Stage 5); this leaves
headroom for that module's own per-line chunking, which will enforce the
same number again as a double defense (CLAUDE.md §6). Lives here, and
`synth/podcast.py` imports it directly rather than re-exporting it under a
second name, because `_PODCAST_SHORTER_INSTRUCTION` below needs the same
figure the "shorter" retry is judged against — one constant, not two that
can drift apart."""


class PodcastTurn(BaseModel):
    """One line of dialogue, as the model returned it."""

    model_config = ConfigDict(extra="forbid")

    speaker: Literal["A", "B"]
    text: str = Field(min_length=1)


class PodcastSegment(BaseModel):
    """One story (or cluster of related stories) discussed in the show.

    `item_ids` is the citation surface NEVER rule 7 requires: the model
    cites ids, never URLs. `synth.podcast.build_script` validates a segment's
    ids against the set actually sent to the model and drops the rest; the
    id-to-URL mapping itself is `report/podcast.py`'s job (Stage 4) — this
    module never touches `db.py`, so it never sees a URL at all.
    """

    model_config = ConfigDict(extra="forbid")

    item_ids: list[int] = Field(min_length=1)
    turns: list[PodcastTurn] = Field(min_length=1)


class PodcastScript(BaseModel):
    """The whole episode, as the model returned it — unvalidated against the
    known item id set and unflattened. `synth.podcast.build_script` does both
    before this becomes something `report/podcast.py` renders.

    `segments` deliberately has no `min_length`, unlike `intro_turns` and
    `outro_turns`: `synth.podcast._drop_unknown_segments` and
    `_truncate_to_fit` both construct an intermediate `PodcastScript` while
    cleaning the model's output, and either can legitimately empty the list
    before `build_script` decides whether anything survived. A model
    response with zero segments is instead rejected explicitly in
    `run_podcast_script` — a schema constraint here cannot tell "the model
    wrote nothing" apart from "cleanup removed everything," and only the
    former is actually a parse error.
    """

    model_config = ConfigDict(extra="forbid")

    intro_turns: list[PodcastTurn] = Field(min_length=1)
    segments: list[PodcastSegment] = Field(default_factory=list)
    outro_turns: list[PodcastTurn] = Field(min_length=1)


@dataclass(slots=True)
class PodcastScriptResult:
    """What one `run_podcast_script` call produced, and what it spent.

    Same shape and reasoning as `TriageBatchResult`/`ScoutResult`: usage is
    recorded whenever a response actually came back, even one whose content
    fails to parse — output tokens are billed for a malformed response too
    (NEVER rule 11) — so `error` and `script` do not both carry information;
    exactly one is meaningful per result.
    """

    script: PodcastScript | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    dropped_item_count: int = 0
    """Items the caller passed beyond `PODCAST_MAX_ITEMS`, clamped before the
    request was built — never sent to the model, so this cost nothing, but
    the caller needs the count to record the clamp (§7, mirrors
    `_effective_search_budget`'s recorded clamp)."""
    sent_item_ids: tuple[int, ...] = ()
    """The ids actually offered to the model, after the `PODCAST_MAX_ITEMS`
    clamp. This — never the caller's own, larger input list — is the "known
    ids" set `synth.podcast.build_script` must validate cited ids against.
    An id the caller passed in but that got clamped away was never shown to
    the model, so a segment citing it is exactly the confabulation NEVER
    rule 7 exists to catch, not a legitimate citation that missed the cut."""


_PODCAST_SHORTER_INSTRUCTION: Final = (
    "Your previous script ran long. Rewrite it shorter: keep every "
    "segment's item_ids the same, but tighten the dialogue so the full "
    f"script's spoken text totals well under {PODCAST_MAX_SCRIPT_CHARS:,} "
    "characters. Trim words, don't drop segments."
)


def _truncate_utf8_json_safe(text: str, max_bytes: int) -> str:
    """Truncate `text` so its `json.dumps(..., ensure_ascii=False)`-encoded
    form is at most `max_bytes` UTF-8 bytes — the unit this cap is actually
    billed in, not the raw UTF-8 length of `text` itself.

    A prior version of this function (`_truncate_utf8`) bounded raw UTF-8
    bytes and stopped there. That closed the character-vs-byte gap
    (`str[:n]` bounds Python characters, and a non-Latin script can run
    several bytes per character) but missed a second, independent expansion:
    JSON must escape `"`, `\\`, and every C0 control character regardless of
    `ensure_ascii`, and that escaping happens *after* truncation, not before.
    A `\\n` becomes the 2-byte escape `\\n`; a raw control byte like `\\x01`
    becomes the 6-byte `\\u0001`. An `llm-cost-guard` review measured the
    result against the shipped caps: a feed with ordinary newlines (a
    changelog or release-note shape, not adversarial input) priced the
    worst case at $30.58/month against a $23 ceiling, and unstripped control
    characters (nothing upstream removes them from `summary`/`content`) at
    $71.38/month — over 3x the ceiling and 1.4x the $50 whole-pipeline alarm
    from this one field. Binary-searching the truncation point against the
    actual `json.dumps` output, the same "measure the thing that's billed"
    move that motivated the byte-vs-character fix, closes this the same way.

    Monotonic by construction — extending a prefix can only add escaped
    bytes, never remove them — so bisecting on character count for the
    longest prefix whose escaped encoding still fits is correct and, for the
    few-thousand-character inputs this function ever sees, cheap: at most
    ~13 `json.dumps` calls, not called on any hot path (once per item field,
    twice per script call at most).
    """

    def _escaped_len(candidate: str) -> int:
        return len(json.dumps(candidate, ensure_ascii=False).encode("utf-8")) - 2

    if _escaped_len(text) <= max_bytes:
        return text
    lo, hi, best = 0, len(text), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if _escaped_len(candidate) <= max_bytes:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _build_podcast_user_prompt(
    *,
    date_label: str,
    presenter_a: str,
    presenter_b: str,
    items: Sequence[tuple[int, str, str | None, str | None]],
) -> tuple[str, int, tuple[int, ...]]:
    """The user turn's base content: date, presenter names, items.

    Deliberately built here rather than accepted pre-rendered — mirrors
    `_build_batch_request`'s payload construction, not `run_source_scout`'s
    pre-rendered-string shape, because `PODCAST_MAX_ITEMS` and the per-field
    byte ceilings have to clamp structured items before they become prompt
    text, not after (the caller cannot be trusted to enforce a *money* limit
    this module owns — CLAUDE.md §6). All three item fields are capped, not
    just `content`: `title` and `summary` are just as capable of inflating
    the prompt (`sources.yaml`'s `max_summary_chars` is operator-editable
    YAML with no upper bound), and a worst case that only bounds one of
    three fields — or bounds them in the wrong unit — is not a worst case.

    Returns `(prompt_text, dropped_item_count, sent_item_ids)`. `sent_item_ids`
    is the actual "what the model could legitimately cite" set — narrower
    than `items` whenever the clamp bit, and the caller must use it (not its
    own input list) when validating cited ids (`PodcastScriptResult.sent_item_ids`).
    """
    clamped = list(items[:PODCAST_MAX_ITEMS])
    dropped = len(items) - len(clamped)
    payload = [
        {
            "item_id": item_id,
            "title": _truncate_utf8_json_safe(title, PODCAST_MAX_ITEM_TITLE_BYTES),
            "summary": _truncate_utf8_json_safe(summary or "", PODCAST_MAX_ITEM_SUMMARY_BYTES),
            "content": _truncate_utf8_json_safe(content or "", PODCAST_MAX_ITEM_CONTENT_BYTES),
        }
        for item_id, title, summary, content in clamped
    ]
    text = (
        f"Today is {date_label}. Speaker A is {presenter_a}; speaker B is "
        f"{presenter_b}. Write today's episode from these items, citing each "
        "segment's item_id(s) exactly as given below — never an id not "
        "listed here:\n" + json.dumps(payload, sort_keys=True, ensure_ascii=False)
        # `ensure_ascii=False`, unlike `_build_batch_request`'s triage payload
        # (a separate, pre-existing gap out of this stage's scope): with the
        # default True, every non-ASCII character in an already-byte-capped
        # field gets re-encoded as a 6-character `\uXXXX` escape. This must
        # match the `ensure_ascii=False` `_truncate_utf8_json_safe` measures
        # against — flipping this back to the default would silently
        # re-open the gap that function closes.
    )
    sent_ids = tuple(item_id for item_id, _, _, _ in clamped)
    return text, dropped, sent_ids


def run_podcast_script(
    stable_system_prefix: str,
    *,
    date_label: str,
    presenter_a: str,
    presenter_b: str,
    items: Sequence[tuple[int, str, str | None, str | None]],
    client: anthropic.Anthropic | None = None,
    model: str = PODCAST_MODEL,
    shorter: bool = False,
) -> PodcastScriptResult:
    """Write one episode script from `items`. Titles, summaries, and full
    `content` all reach this call — unlike triage (NEVER rule 9), this is
    the top-N deep-read survivor slice DESIGN §6 reserves full content for.

    `stable_system_prefix` is the fully-rendered, cache-controlled prefix
    (`synth.podcast.build_podcast_stable_prefix`) — this module wraps it in
    the `cache_control` block itself, the same split `_build_batch_request`
    uses for triage's system prompt. Honestly: at the current shipped
    `interests.yaml`, that prefix renders under Anthropic's minimum
    cacheable prompt length, so the breakpoint is a no-op today — kept
    because a fuller show-format brief would clear it for free, and because
    the worst-case ceiling (`PODCAST_MONTHLY_CEILING_USD`) already prices
    every call as if it never reads from cache, so its inertness costs
    nothing to be wrong about. `items` is `(item_id, title, summary,
    content)` tuples, deliberately not `Item`, so an accidental extra field
    (e.g. a raw HTTP path) cannot ride along into the prompt.

    Items, date, and presenter names all stay in the user turn, after the
    `cache_control` breakpoint (NEVER rule 10) — deliberately *not* cached
    themselves, even though they are identical between the first attempt and
    the "shorter" retry: caching them would mean the retry reads real spoken
    content back from cache, and this feature's whole worst-case ceiling is
    priced assuming every call pays full price for its item payload (see
    `PODCAST_MONTHLY_CEILING_USD`). `PODCAST_RETRY_MAX_TOKENS`, not caching,
    is what keeps a retried call from costing close to double.

    `shorter=True` appends a rewrite instruction as a separate, uncached user
    content block, and uses `PODCAST_RETRY_MAX_TOKENS` in place of
    `PODCAST_MAX_TOKENS` — the caller (`synth.podcast.build_script`) passes
    this for its one retry when the first attempt ran over
    `PODCAST_MAX_SCRIPT_CHARS`. There is no `max_tokens` parameter to set
    independently — unlike `run_source_scout` (whose docstring gives the
    same reasoning), the output ceiling is a money decision this module
    owns outright, not a knob a caller should be able to widen by accident.

    Raises `LlmError` only for a failure before any response comes back
    (auth, network, the create call itself) — nothing was spent yet, unlike
    `run_source_scout`'s server-tool case. A response that arrives but fails
    to parse is never raised; it is recorded on `PodcastScriptResult.error`
    with its usage still counted (NEVER rule 11), matching
    `_apply_batch_entry`'s per-group error handling.
    """
    active_client = client if client is not None else get_anthropic_client()
    base_prompt, dropped, sent_ids = _build_podcast_user_prompt(
        date_label=date_label,
        presenter_a=presenter_a,
        presenter_b=presenter_b,
        items=items,
    )
    user_content: list[TextBlockParam] = [{"type": "text", "text": base_prompt}]
    effective_max_tokens = PODCAST_MAX_TOKENS
    if shorter:
        user_content.append({"type": "text", "text": _PODCAST_SHORTER_INSTRUCTION})
        effective_max_tokens = PODCAST_RETRY_MAX_TOKENS

    try:
        response = active_client.messages.create(
            model=model,
            max_tokens=effective_max_tokens,
            system=[
                {
                    "type": "text",
                    "text": stable_system_prefix,
                    # No timestamps or run ids in this text (NEVER rule 10) —
                    # see `build_podcast_stable_prefix`'s own docstring for
                    # what makes that true by construction, not by care.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={
                "effort": PODCAST_EFFORT,
                "format": {"type": "json_schema", "schema": _PODCAST_OUTPUT_SCHEMA},
            },
            messages=[MessageParam(role="user", content=user_content)],
        )
    except anthropic.APIError as exc:
        raise LlmError(f"failed to generate podcast script: {exc}") from exc

    usage = response.usage
    input_tokens = (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )
    output_tokens = usage.output_tokens

    if response.stop_reason == "refusal":
        logger.warning("podcast script call was refused")
        return PodcastScriptResult(
            error="podcast script call was refused by safety classifiers",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_count=dropped,
            sent_item_ids=sent_ids,
        )

    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        return PodcastScriptResult(
            error="no text content in podcast script response",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_count=dropped,
            sent_item_ids=sent_ids,
        )

    try:
        script = PodcastScript.model_validate_json(text)
    except ValidationError as exc:
        logger.warning("podcast script failed schema validation", extra={"error": str(exc)})
        return PodcastScriptResult(
            error=f"schema validation failed: {exc}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_count=dropped,
            sent_item_ids=sent_ids,
        )

    if not script.segments:
        # `PodcastScript.segments` has no `min_length` (see its docstring) —
        # this is the explicit rejection that constraint would otherwise
        # have given us, applied only to the model's own output.
        logger.warning("podcast script had no segments")
        return PodcastScriptResult(
            error="podcast script had no segments",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_count=dropped,
            sent_item_ids=sent_ids,
        )

    logger.info(
        "podcast script complete",
        extra={
            "segment_count": len(script.segments),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "dropped_item_count": dropped,
        },
    )
    return PodcastScriptResult(
        script=script,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dropped_item_count=dropped,
        sent_item_ids=sent_ids,
    )


# --------------------------------------------------------------------------- #
# Weekly Intelligence Brief (DESIGN §13, Phase 1)
#
# Phase 1's product and its acceptance gate. One Opus request per week, over
# titles, summaries and the stored triage reasoning of the week's top-N items —
# never `items.content`, which is what keeps this the cheapest of the three
# Opus consumers despite the highest effort setting.
# --------------------------------------------------------------------------- #

WEEKLY_MODEL: Final = "claude-opus-5"
"""Synthesis model for the weekly brief.

The same string as `SCOUT_MODEL`/`PODCAST_MODEL`, deliberately. DESIGN §8's table
still names `claude-opus-4-8` for this row; the two price identically and the repo
standardised on `claude-opus-5` when the scout shipped, so introducing a third
Opus id would fragment the prompt cache and the docs for nothing. The drift is
recorded in DESIGN rather than perpetuated here."""

WEEKLY_EFFORT: Final = "high"
"""Reasoning effort, chosen on cadence — the same argument `PODCAST_EFFORT`
records in reverse.

That constant justifies "medium" purely by frequency: a daily call pays for its
effort thirty times a month where a weekly one pays five. The brief is the weekly
case, so `SCOUT_EFFORT`'s "high" is the applicable precedent, and effort adds
nothing to the worst case because `WEEKLY_MAX_TOKENS` is what bounds the bill.
Operator decision, 2026-08-08: this is the product's centrepiece and worth
thinking hard about, with `WEEKLY_MAX_TOKENS` sized accordingly."""

WEEKLY_MAX_TOKENS: Final = 12_288
"""Output ceiling — sized for the *thinking*, not for the JSON.

Nothing in this module passes `thinking`, so adaptive thinking is on by default
and bills as output **and** counts against this number. Measured in this repo:
the scout at `effort: high` consumes 78% of its budget; the podcast at `medium`
consumes 43%. At 78% of 12,288 that leaves ~2,700 tokens for a structured output
sized at ~1,400 — comfortable. The 8,192 this was first specced at would have
left roughly 400 tokens of margin, and a truncated brief is a fully-billed call
that produces nothing."""

WEEKLY_MAX_ITEMS: Final = 24
"""Hard ceiling on items per call, which config can only lower.

`thresholds.weekly_top_n` is `ge=1` with no upper bound, so without this clamp a
one-line YAML edit would move real spend with every test green — the exact
failure `SCOUT_MAX_SEARCHES_CEILING` exists to prevent. Twice the shipped
`weekly_top_n: 12`, and affordable at 24 only because no `content` field is sent:
per-item worst-case bytes here are roughly a sixth of the podcast's."""

WEEKLY_MAX_ITEM_TITLE_BYTES: Final = 300
"""Mirrors `PODCAST_MAX_ITEM_TITLE_BYTES`. `Item.title` is flattened but not
length-bounded upstream."""

WEEKLY_MAX_ITEM_SUMMARY_BYTES: Final = 500
"""Mirrors `PODCAST_MAX_ITEM_SUMMARY_BYTES`, and it is the binding cap here.
Measured over a real week: the longest stored summary was ~4,000 bytes, eight
times this. The ceiling below is only true because this one actually clamps."""

WEEKLY_MAX_ITEM_REASONING_BYTES: Final = 600
"""Cap on `scores.reasoning`, which has no analogue in the podcast payload.

Unbounded model-authored prose: the rubric asks for one sentence, and nothing
enforces that. `report/daily.py`'s 320-character trim is a *display* bound on a
different surface and does not travel with the stored value."""

WEEKLY_MONTHLY_CEILING_USD: Final = 3.50
"""Hard worst-case ceiling on weekly-brief spend, enforced by a test.

A bound derived from what the code permits, never from what cron is expected to
do (DESIGN §8: "an estimate that assumes good behaviour is a forecast, not a
bound"). The arithmetic is deliberately not written out here — see
`SCOUT_MONTHLY_CEILING_USD` for why numbers copied into prose drift — but the
terms the test reads are: every field at its byte cap, `WEEKLY_MAX_ITEMS` items,
`WEEKLY_MAX_TOKENS` of output, a pessimistic bytes-per-token ratio, and **five**
calls a month rather than 4.33, because a month can contain five Sundays.

The margin is wider than the podcast's on purpose: that ceiling's own history
records a 340-byte prefix addition eating a third of its headroom, and this
prefix has its lead/cluster format and citation discipline still to grow.

**One term this bound deliberately omits**, shared by all four of the pipeline's
enforced ceilings: `get_anthropic_client` leaves the SDK's default
`max_retries=2`, so a request that times out *after* the model has generated can
bill up to three times before raising. Setting it to zero would trade a rare
double-bill for a lost run on every transient network blip across triage,
scout and podcast alike, which is the worse deal at this cadence. Recorded here
rather than silently excluded; if a real run ever shows it firing, the honest fix
is a 3x term in this bound, not a quieter docstring."""

_WEEKLY_OUTPUT_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "leads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "item_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["headline", "why_it_matters", "item_ids"],
                "additionalProperties": False,
            },
        },
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "narrative": {"type": "string"},
                    "item_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["title", "narrative", "item_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["leads", "clusters"],
    "additionalProperties": False,
}
"""Structured-output schema for the request.

**Carries no `minItems`/`maxItems`** — the API rejects both outright for array
types (`400 invalid_request_error: For 'array' type, property 'maxItems' is not
supported`), which a first live call found. So every count bound lives in code:
the prompt asks for the shape, `WEEKLY_MAX_LEADS`/`WEEKLY_MAX_CLUSTERS` clamp
what comes back, and `WEEKLY_MAX_TOKENS` is what actually bounds the dollars.

Loose in the same places for the same reason (non-empty strings, non-empty
`item_ids`): `WeeklyLead`/`WeeklyCluster`/`WeeklyBrief` re-validate those, so a
violation becomes a recorded parse error rather than a crash or silently accepted
content."""

WEEKLY_MAX_LEADS: Final = 3
"""How many "things that mattered" a brief may carry.

Clamped after the response rather than constrained in the schema, because the API
rejects array bounds — and clamping is the better failure mode anyway: rejecting a
whole paid response over a fourth lead throws away a good brief and, worse,
invites the next invocation to re-bill for it."""

WEEKLY_MAX_CLUSTERS: Final = 6
"""How many themes a brief may carry. Not a dollar bound — `WEEKLY_MAX_TOKENS` is
— but the thing that keeps a brief readable, and the counterpart to the prompt's
own instruction."""


class WeeklyLead(BaseModel):
    """One of "the 3 things that mattered" (DESIGN §13's lead block).

    `item_ids` is the citation surface NEVER rule 7 requires: the model cites
    ids, never URLs, and never prose it invented a source for. Validation of
    those ids against the set actually sent is `synth.weekly`'s job; resolving
    an id to a link is `report/weekly.py`'s. This module never sees a URL.
    """

    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    item_ids: list[int] = Field(min_length=1)


class WeeklyCluster(BaseModel):
    """A themed group of the week's items, with the narrative tying them together.

    Grouping *and* narrating are both done by the model here, which is a
    deliberate, temporary exception to the deterministic-vs-LLM split (DESIGN
    §8 puts "clustering math" in the deterministic column and only "cluster
    labeling and cross-source synthesis" in the LLM one). The deterministic
    input that split assumes — embeddings and distance-based clustering — is
    Phase 2 work and does not exist yet. When it lands, the grouping moves to
    it and only the narration stays here.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    narrative: str = Field(min_length=1)
    item_ids: list[int] = Field(min_length=1)


class WeeklyBrief(BaseModel):
    """The whole brief, as the model returned it — ids unvalidated, text unflattened.

    Neither list is bounded here. "The 3 things that mattered" is a headline, not
    a schema constraint — a hard floor on a thin week is an instruction to
    manufacture a third thing that did not matter — and the ceiling cannot live
    on this model either, because rejecting a whole paid response over one extra
    block is a worse outcome than clamping it. `synth.weekly` clamps to
    `WEEKLY_MAX_LEADS`/`WEEKLY_MAX_CLUSTERS` instead.

    Neither list is required to survive cleaning, so both tolerate being
    emptied: `synth.weekly` builds intermediate values of this model while
    dropping blocks with unknown citations, and a brief left with nothing is a
    decision `build_brief` makes explicitly rather than a `ValidationError`
    raised mid-clean. The same reasoning `PodcastScript.segments` records.
    """

    model_config = ConfigDict(extra="forbid")

    leads: list[WeeklyLead] = Field(default_factory=list)
    clusters: list[WeeklyCluster] = Field(default_factory=list)


@dataclass(slots=True)
class WeeklyBriefResult:
    """What one `run_weekly_brief` call produced, and what it spent.

    Same shape and reasoning as `PodcastScriptResult`: usage is recorded whenever
    a response actually came back, including one whose content fails to parse —
    output tokens are billed for a malformed response too (NEVER rule 11) — so
    `error` and `brief` never both carry information; exactly one is meaningful.

    `sent_item_ids` is the "what the model could legitimately cite" set, narrower
    than the caller's input list whenever `WEEKLY_MAX_ITEMS` clamped. Callers must
    validate cited ids against *this*, not their own input, or a clamped-away item
    reads as a legitimate citation.
    """

    brief: WeeklyBrief | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    dropped_item_count: int = 0
    sent_item_ids: tuple[int, ...] = ()


def _build_weekly_user_prompt(
    *,
    window_label: str,
    items: Sequence[tuple[int, str, str | None, str]],
) -> tuple[str, int, tuple[int, ...]]:
    """The user turn: the window's label and the week's items as JSON.

    Built here rather than accepted pre-rendered, for the reason
    `_build_podcast_user_prompt` records: `WEEKLY_MAX_ITEMS` and the per-field
    byte ceilings are *money* limits this module owns, and they have to clamp
    structured items before they become prompt text rather than after.

    All three text fields are capped. `summary` is the one that actually binds in
    practice (a real week's longest ran eight times its cap), but a worst case
    that bounds one field of three is not a worst case.

    Returns `(prompt_text, dropped_item_count, sent_item_ids)`.
    """
    clamped = list(items[:WEEKLY_MAX_ITEMS])
    dropped = len(items) - len(clamped)
    payload = [
        {
            "item_id": item_id,
            "title": _truncate_utf8_json_safe(title, WEEKLY_MAX_ITEM_TITLE_BYTES),
            "summary": _truncate_utf8_json_safe(summary or "", WEEKLY_MAX_ITEM_SUMMARY_BYTES),
            "why_it_was_kept": _truncate_utf8_json_safe(reasoning, WEEKLY_MAX_ITEM_REASONING_BYTES),
        }
        for item_id, title, summary, reasoning in clamped
    ]
    text = (
        f"These are the items that cleared the bar for the week of {window_label}. "
        "Write the brief from them, citing item_id(s) exactly as given below and "
        "never an id not listed here:\n" + json.dumps(payload, sort_keys=True, ensure_ascii=False)
        # `ensure_ascii=False` must match what `_truncate_utf8_json_safe` measures
        # against; the default True re-encodes every non-ASCII character in an
        # already-capped field as a 6-character escape, silently re-opening the
        # gap that function closes.
    )
    sent_ids = tuple(item_id for item_id, _, _, _ in clamped)
    return text, dropped, sent_ids


def run_weekly_brief(
    stable_system_prefix: str,
    *,
    window_label: str,
    items: Sequence[tuple[int, str, str | None, str]],
    client: anthropic.Anthropic | None = None,
    model: str = WEEKLY_MODEL,
) -> WeeklyBriefResult:
    """Write one weekly brief from `items`. Titles, summaries and stored triage
    reasoning only — never `items.content` (DESIGN §8).

    Each item is `(item_id, title, summary, reasoning)`, deliberately a tuple and
    not an `Item`: several of these rows *do* carry deep-read `content` from the
    podcast path, and an `Item` seam would let it ride along into the prompt
    unnoticed, multiplying the bill for material the brief was never priced for.

    **No `cache_control` breakpoint**, unlike the podcast's daily call. One
    request per run, seven days apart, no retry, so an ephemeral cache entry can
    never be read before it expires — a breakpoint would be pure write premium.
    This is the scout's recorded reasoning at the identical cadence (DESIGN §8),
    and the prefix is below Opus's cacheable minimum anyway, which would make the
    breakpoint a silent no-op rather than a cheap bet.

    Never raises once a response has been billed: a refusal, a missing text block
    and a schema failure each come back as a result carrying its usage
    (NEVER rule 11). Only a pre-response `APIError` raises, when nothing was spent.
    """
    active_client = client if client is not None else get_anthropic_client()
    prompt, dropped, sent_ids = _build_weekly_user_prompt(window_label=window_label, items=items)

    try:
        response = active_client.messages.create(
            model=model,
            max_tokens=WEEKLY_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": stable_system_prefix,
                    # No timestamps or run ids in this text (NEVER rule 10) — see
                    # `synth.weekly.build_weekly_stable_prefix`, which takes only
                    # `interests` so that is true by construction, not by care.
                }
            ],
            output_config={
                "effort": WEEKLY_EFFORT,
                "format": {"type": "json_schema", "schema": _WEEKLY_OUTPUT_SCHEMA},
            },
            messages=[MessageParam(role="user", content=[{"type": "text", "text": prompt}])],
        )
    except anthropic.APIError as exc:
        raise LlmError(f"failed to generate weekly brief: {exc}") from exc

    usage = response.usage
    input_tokens = (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )
    output_tokens = usage.output_tokens

    def failed(message: str) -> WeeklyBriefResult:
        return WeeklyBriefResult(
            error=message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_item_count=dropped,
            sent_item_ids=sent_ids,
        )

    if response.stop_reason == "refusal":
        logger.warning("weekly brief call was refused")
        return failed("weekly brief call was refused by safety classifiers")

    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        return failed("no text content in weekly brief response")

    try:
        brief = WeeklyBrief.model_validate_json(text)
    except ValidationError as exc:
        logger.warning("weekly brief failed schema validation", extra={"error": str(exc)})
        return failed(f"schema validation failed: {exc}")

    if not brief.leads and not brief.clusters:
        # Neither list has a `min_length` on the model (see `WeeklyBrief`), so
        # this is the explicit rejection that constraint would have given us,
        # applied only to the model's own output and never to a cleaned value.
        logger.warning("weekly brief had no leads and no clusters")
        return failed("weekly brief had no leads and no clusters")

    logger.info(
        "weekly brief complete",
        extra={
            "lead_count": len(brief.leads),
            "cluster_count": len(brief.clusters),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "dropped_item_count": dropped,
        },
    )
    return WeeklyBriefResult(
        brief=brief,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dropped_item_count=dropped,
        sent_item_ids=sent_ids,
    )
