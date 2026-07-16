"""Scoring orchestration — batches unscored items through `llm.py`, persists.

`score/` never makes HTTP calls to sources (CLAUDE.md §2, NEVER rule 2): it
reads only stored items via `db.py` and calls the LLM only through
`signalforge.llm`, never importing `anthropic` itself (NEVER rule 1).

Idempotent by construction (CLAUDE.md §3): `_fetch_unscored_items` selects
only items with no `scores` row, so re-running `signalforge score` sends
nothing already scored to the LLM and spends zero additional tokens.

Failure isolation matches `cli.py`'s ingest command: a batch that fails to
reach the API is recorded as a run-level error and the run ends there — no
exception escapes to abort a `signalforge` invocation (CLAUDE.md §7, NEVER
rule 12). Items whose *individual* result comes back malformed are recorded
per-item and simply remain unscored, to be retried the next time `score` runs
(they still carry no `scores` row, so `_fetch_unscored_items` picks them up
again — no special-casing needed for that retry).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

from signalforge.config import InterestsConfig
from signalforge.db import insert_score
from signalforge.score.rubrics import RUBRIC_VERSION

# `signalforge.llm` is imported lazily inside `score_unscored_items` rather
# than at module scope. `llm.py` itself imports `signalforge.score.rubrics` —
# and importing *any* submodule of `score` first runs this `__init__.py` — so
# an eager top-level `from signalforge.llm import ...` here would make the two
# modules' import order matter (whichever is imported first would find the
# other only partially initialized). Deferring the import to call time breaks
# the cycle without changing either module's public API.

__all__ = ["ScoreOutcome", "score_unscored_items"]

logger = logging.getLogger(__name__)

_RUN_LEVEL_SOURCE_ID = "*"
"""`runs.errors[].source_id` for a failure that belongs to the whole batch
call, not a single item — mirrors the convention `cli.py` uses for ingest."""


@dataclass(slots=True)
class ScoreOutcome:
    """What one `score` invocation achieved — the CLI's basis for `runs.status`
    and the `runs.llm_input_tokens`/`llm_output_tokens` columns."""

    items_scored: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)


def _fetch_unscored_items(conn: sqlite3.Connection) -> list[tuple[int, str, str | None]]:
    """Every item with no `scores` row yet, oldest first.

    The `LEFT JOIN ... WHERE scores.item_id IS NULL` is the whole idempotency
    mechanism (CLAUDE.md §3): an item that already has a score is invisible to
    this query, so it can never be sent to the LLM — and therefore never
    billed — twice.
    """
    rows = conn.execute(
        """
        SELECT items.id AS id, items.title AS title, items.summary AS summary
        FROM items
        LEFT JOIN scores ON scores.item_id = items.id
        WHERE scores.item_id IS NULL
        ORDER BY items.id
        """
    ).fetchall()
    return [(row["id"], row["title"], row["summary"]) for row in rows]


def _error_record(source_id: str, exc_or_message: BaseException | str) -> dict[str, str]:
    if isinstance(exc_or_message, BaseException):
        error_type = exc_or_message.__class__.__name__
        message = str(exc_or_message) or error_type
    else:
        error_type = "TriageItemError"
        message = exc_or_message
    return {
        "source_id": source_id,
        "error_type": error_type,
        "message": message,
        "occurred_at": datetime.now(UTC).isoformat(),
    }


def score_unscored_items(
    conn: sqlite3.Connection,
    interests: InterestsConfig,
    *,
    model: str | None = None,
) -> ScoreOutcome:
    """Score every currently-unscored item; persist results as they parse.

    Returns a `ScoreOutcome` for the caller (`cli.py`) to fold into a `runs`
    row. Never raises: an `LlmError` from the batch call is caught and
    recorded as a run-level error (`source_id == "*"`), matching how ingest's
    `cli.py` isolates a source failure from the rest of a run.

    `model` defaults to `signalforge.llm.TRIAGE_MODEL` when omitted — resolved
    inside this function (see the module docstring for why the `llm` import
    is deferred rather than module-level).
    """
    from signalforge.llm import TRIAGE_MODEL, LlmError, run_triage_batch

    resolved_model = model or TRIAGE_MODEL

    outcome = ScoreOutcome()
    pending = _fetch_unscored_items(conn)
    if not pending:
        return outcome

    # Captured *before* the blocking Batches-API call, not after: `run_triage_batch`
    # can take anywhere from minutes to (documented) up to 24h to return, and
    # `get_digest_items`/`count_killed_items` bucket by `scored_at`'s calendar date.
    # Stamping at completion time would let a slow batch straddle UTC midnight and
    # push every item in it into the next day's digest bucket instead of today's.
    scored_at = datetime.now(UTC)

    try:
        batch_result = run_triage_batch(pending, interests, model=resolved_model)
    except LlmError as exc:
        logger.exception("triage batch call failed; no items scored this run")
        outcome.errors.append(_error_record(_RUN_LEVEL_SOURCE_ID, exc))
        return outcome

    outcome.input_tokens = batch_result.input_tokens
    outcome.output_tokens = batch_result.output_tokens

    for item_id, result in batch_result.results.items():
        try:
            insert_score(
                conn,
                item_id=item_id,
                triage=result.triage,
                signal=result.signal,
                relevance=result.relevance,
                novelty=result.novelty,
                reasoning=result.reasoning,
                rubric_version=RUBRIC_VERSION,
                model=resolved_model,
                scored_at=scored_at,
            )
        except Exception as exc:  # sqlite3.Error and friends
            # One item's write failing must not lose the rest of the batch
            # (CLAUDE.md §7) — it stays unscored and is retried next run.
            logger.exception("persisting a score failed", extra={"item_id": item_id})
            outcome.errors.append(_error_record(str(item_id), exc))
        else:
            outcome.items_scored += 1

    for item_id, item_error in batch_result.errors.items():
        logger.warning("item failed triage", extra={"item_id": item_id, "reason": item_error})
        outcome.errors.append(_error_record(str(item_id), item_error))

    return outcome
