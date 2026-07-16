"""Tests for `score/__init__.py`'s orchestration.

Fakes `signalforge.llm.run_triage_batch` at the module boundary — never the
real Anthropic API (CLAUDE.md §8, NEVER rule 13). What matters here is
selection ("never re-score"), persistence, and failure isolation; `llm.py`'s
own batching/parsing behaviour is covered in `tests/test_llm.py`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from signalforge.config import InterestsConfig
from signalforge.db import connect, insert_score, migrate, upsert_item
from signalforge.llm import LlmError, TriageBatchResult, TriageResult
from signalforge.models import Item, SourceType
from signalforge.score import score_unscored_items


def make_interests(**overrides: object) -> InterestsConfig:
    data: dict[str, object] = {
        "thresholds": {"weekly_min_signal": 3, "weekly_min_relevance": 3, "weekly_min_total": 10},
    }
    data.update(overrides)
    return InterestsConfig.model_validate(data)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(tmp_path / "signalforge.db")
    migrate(connection)
    try:
        yield connection
    finally:
        connection.close()


def _make_item(**overrides: object) -> Item:
    fields: dict[str, object] = {
        "source_id": "simonwillison",
        "source_type": SourceType.RSS,
        "external_id": "guid-1",
        "url": "https://simonwillison.net/post",
        "title": "MCP sampling lands everywhere",
        "summary": "A short feed summary.",
        "fetched_at": datetime(2026, 7, 16, 6, 0, tzinfo=UTC),
    }
    fields.update(overrides)
    return Item(**fields)  # type: ignore[arg-type]


def _fake_run_triage_batch(
    result: TriageBatchResult | None = None, *, error: Exception | None = None
) -> Callable[..., TriageBatchResult]:
    def _fake(*args: object, **kwargs: object) -> TriageBatchResult:
        if error is not None:
            raise error
        return result if result is not None else TriageBatchResult()

    return _fake


def _triage_result(**overrides: object) -> TriageResult:
    fields: dict[str, object] = {
        "triage": "keep",
        "signal": 4,
        "relevance": 4,
        "novelty": 3,
        "reasoning": "Solid working example with benchmarks.",
    }
    fields.update(overrides)
    return TriageResult(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Selection / idempotency
# --------------------------------------------------------------------------- #


def test_no_unscored_items_makes_no_llm_call(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"n": 0}

    def _boom(*args: object, **kwargs: object) -> TriageBatchResult:
        called["n"] += 1
        raise AssertionError("must not be called with nothing to score")

    monkeypatch.setattr("signalforge.llm.run_triage_batch", _boom)
    outcome = score_unscored_items(conn, make_interests())

    assert called["n"] == 0
    assert outcome.items_scored == 0
    assert outcome.errors == []


def test_scores_a_pending_item_and_persists_it(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    item_id, _ = upsert_item(conn, _make_item())
    batch_result = TriageBatchResult(
        results={item_id: _triage_result()}, input_tokens=500, output_tokens=80
    )
    monkeypatch.setattr("signalforge.llm.run_triage_batch", _fake_run_triage_batch(batch_result))

    outcome = score_unscored_items(conn, make_interests())

    assert outcome.items_scored == 1
    assert outcome.errors == []
    assert (outcome.input_tokens, outcome.output_tokens) == (500, 80)

    row = conn.execute("SELECT * FROM scores WHERE item_id = ?", (item_id,)).fetchone()
    assert row["triage"] == "keep"
    assert row["signal"] == 4
    assert row["rubric_version"]
    assert row["model"]


def test_an_already_scored_item_is_never_sent_back_to_the_llm(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE.md §3: idempotency is non-negotiable. Running `score` twice must
    spend zero additional tokens on an item that already has a `scores` row."""
    item_id, _ = upsert_item(conn, _make_item())
    first_batch = TriageBatchResult(
        results={item_id: _triage_result()}, input_tokens=500, output_tokens=80
    )
    monkeypatch.setattr("signalforge.llm.run_triage_batch", _fake_run_triage_batch(first_batch))
    first = score_unscored_items(conn, make_interests())
    assert first.items_scored == 1

    def _must_not_be_called(*args: object, **kwargs: object) -> TriageBatchResult:
        raise AssertionError("an already-scored item must never reach the LLM again")

    monkeypatch.setattr("signalforge.llm.run_triage_batch", _must_not_be_called)
    second = score_unscored_items(conn, make_interests())

    assert second.items_scored == 0
    assert second.input_tokens == 0 and second.output_tokens == 0
    assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 1


def test_scoring_twice_in_a_row_leaves_scores_byte_for_byte_identical(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    item_id, _ = upsert_item(conn, _make_item())
    batch_result = TriageBatchResult(results={item_id: _triage_result()})
    monkeypatch.setattr("signalforge.llm.run_triage_batch", _fake_run_triage_batch(batch_result))
    score_unscored_items(conn, make_interests())
    before = [tuple(row) for row in conn.execute("SELECT * FROM scores").fetchall()]

    monkeypatch.setattr(
        "signalforge.llm.run_triage_batch",
        _fake_run_triage_batch(TriageBatchResult()),
    )
    score_unscored_items(conn, make_interests())
    after = [tuple(row) for row in conn.execute("SELECT * FROM scores").fetchall()]

    assert before == after


# --------------------------------------------------------------------------- #
# Failure isolation (CLAUDE.md §7, NEVER rule 12)
# --------------------------------------------------------------------------- #


def test_llm_error_is_recorded_and_nothing_is_scored(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    upsert_item(conn, _make_item())
    monkeypatch.setattr(
        "signalforge.llm.run_triage_batch",
        _fake_run_triage_batch(error=LlmError("batch creation failed")),
    )

    outcome = score_unscored_items(conn, make_interests())

    assert outcome.items_scored == 0
    assert len(outcome.errors) == 1
    assert outcome.errors[0]["source_id"] == "*"
    assert outcome.errors[0]["error_type"] == "LlmError"
    assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0


def test_an_unexpected_exception_from_the_batch_call_is_not_swallowed(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only `LlmError` is treated as an isolated failure — anything else is a
    bug and must propagate so the CLI's `runs` bookkeeping still sees it."""
    upsert_item(conn, _make_item())
    monkeypatch.setattr(
        "signalforge.llm.run_triage_batch",
        _fake_run_triage_batch(error=RuntimeError("unexpected")),
    )

    with pytest.raises(RuntimeError):
        score_unscored_items(conn, make_interests())


def test_a_per_item_triage_error_leaves_that_item_unscored_and_is_recorded(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    item_id, _ = upsert_item(conn, _make_item())
    batch_result = TriageBatchResult(errors={item_id: "missing from batch response"})
    monkeypatch.setattr("signalforge.llm.run_triage_batch", _fake_run_triage_batch(batch_result))

    outcome = score_unscored_items(conn, make_interests())

    assert outcome.items_scored == 0
    assert len(outcome.errors) == 1
    assert outcome.errors[0]["source_id"] == str(item_id)
    assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0


def test_a_persistence_failure_for_one_item_does_not_lose_the_rest_of_the_batch(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    good_id, _ = upsert_item(conn, _make_item())
    bad_id, _ = upsert_item(
        conn, _make_item(external_id="guid-2", url="https://simonwillison.net/other")
    )
    batch_result = TriageBatchResult(results={good_id: _triage_result(), bad_id: _triage_result()})
    monkeypatch.setattr("signalforge.llm.run_triage_batch", _fake_run_triage_batch(batch_result))

    real_insert_score = insert_score

    def _flaky_insert(
        conn: sqlite3.Connection,
        *,
        item_id: int,
        triage: str,
        signal: int | None,
        relevance: int | None,
        novelty: int | None,
        reasoning: str,
        rubric_version: str,
        model: str,
        scored_at: datetime,
    ) -> None:
        if item_id == bad_id:
            raise sqlite3.OperationalError("disk I/O error")
        real_insert_score(
            conn,
            item_id=item_id,
            triage=triage,
            signal=signal,
            relevance=relevance,
            novelty=novelty,
            reasoning=reasoning,
            rubric_version=rubric_version,
            model=model,
            scored_at=scored_at,
        )

    monkeypatch.setattr("signalforge.score.insert_score", _flaky_insert)

    outcome = score_unscored_items(conn, make_interests())

    assert outcome.items_scored == 1
    assert len(outcome.errors) == 1
    assert outcome.errors[0]["source_id"] == str(bad_id)
    assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 1
    assert conn.execute("SELECT 1 FROM scores WHERE item_id = ?", (good_id,)).fetchone()


def test_multiple_pending_items_are_all_selected(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = [
        upsert_item(conn, _make_item(external_id=f"guid-{n}", url=f"https://example.com/{n}"))[0]
        for n in range(3)
    ]
    batch_result = TriageBatchResult(results={i: _triage_result() for i in ids})
    seen_ids: list[int] = []

    def _capturing(
        items: list[tuple[int, str, str | None]], interests: InterestsConfig, **kwargs: object
    ) -> TriageBatchResult:
        seen_ids.extend(item_id for item_id, _, _ in items)
        return batch_result

    monkeypatch.setattr("signalforge.llm.run_triage_batch", _capturing)

    outcome = score_unscored_items(conn, make_interests())

    assert outcome.items_scored == 3
    assert sorted(seen_ids) == sorted(ids)
