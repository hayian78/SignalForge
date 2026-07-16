"""Integration test across the seam three separately-built agents didn't see:
`ingest` -> `score` -> `digest`. Each command has its own idempotency tests in
isolation (`tests/test_cli.py`, `tests/test_cli_score.py`,
`tests/test_digest_cli.py`); this file exists because the handoffs between
them — does a scored item's `scored_at` land in the digest bucket the same
run computes, does a second full pass add zero rows/scores/duplicate output —
are exactly where two independently-built halves can silently disagree.

The only network boundary is `ingest`'s RSS fetch, served by `respx`
(CLAUDE.md §8). The LLM boundary is faked at `signalforge.llm.run_triage_batch`
(NEVER rule 13) — never the real Anthropic API.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner, Result

from signalforge.cli import app
from signalforge.db import connection
from signalforge.llm import TriageBatchResult, TriageResult

runner = CliRunner()

_LOG_LEVEL: list[str] = ["--log-level", "WARNING"]

FEED_URL = "https://example.com/alpha/feed.xml"

FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>alpha blog</title>
  <entry>
    <title>alpha post</title>
    <link href="https://example.com/alpha/post-1"/>
    <id>urn:alpha:1</id>
    <updated>2026-07-15T12:30:00Z</updated>
    <summary>A short summary of the post.</summary>
  </entry>
</feed>
"""


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """The CLI owns global logging config; restore it after every test."""
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:
        yield
    finally:
        root.setLevel(level)
        root.handlers[:] = handlers


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    path = tmp_path / "config"
    path.mkdir()
    (path / "sources.yaml").write_text(
        "defaults:\n"
        "  fetch_timeout: 5\n"
        "  min_hn_points: 80\n"
        "  max_summary_chars: 4000\n"
        "rss:\n"
        f"  - id: alpha\n    url: {FEED_URL}\n",
        encoding="utf-8",
    )
    (path / "interests.yaml").write_text(
        "priority_topics: [agents.mcp]\n"
        "interests: [python]\n"
        "stack: [python]\n"
        "learning_goals: []\n"
        "architecture_philosophy: 'Local-first.'\n"
        "ignore:\n"
        "  topics: [crypto]\n"
        "  people: []\n"
        "  repos: []\n"
        "thresholds: {weekly_min_signal: 3, weekly_min_relevance: 3, weekly_min_total: 10,"
        " daily_max_items: 15}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "signalforge.db"


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "http_cache"


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    return tmp_path / "vault"


def _run(*args: str) -> Result:
    return runner.invoke(app, [*_LOG_LEVEL, *args])


def _ingest(config_dir: Path, db_path: Path, cache_dir: Path) -> Result:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"etag": '"a-1"'})
        )
        return _run(
            "ingest",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
        )


def _score(config_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Result:
    batch_result = TriageBatchResult(
        results={
            1: TriageResult(
                triage="keep", signal=5, relevance=5, novelty=4, reasoning="Good stuff."
            )
        },
        input_tokens=42,
        output_tokens=7,
    )
    monkeypatch.setattr("signalforge.llm.run_triage_batch", lambda *a, **k: batch_result)
    return _run("score", "--config-dir", str(config_dir), "--db", str(db_path))


def _digest(config_dir: Path, db_path: Path, vault_dir: Path, target_date: str) -> Result:
    return _run(
        "digest",
        "--config-dir",
        str(config_dir),
        "--db",
        str(db_path),
        "--vault-dir",
        str(vault_dir),
        "--date",
        target_date,
    )


def test_ingest_score_digest_chain_twice_is_fully_idempotent(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest_result = _ingest(config_dir, db_path, cache_dir)
    assert ingest_result.exit_code == 0

    score_result = _score(config_dir, db_path, monkeypatch)
    assert score_result.exit_code == 0

    with connection(db_path) as conn:
        scored_at = conn.execute("SELECT scored_at FROM scores WHERE item_id = 1").fetchone()[0]
    target_date = scored_at[:10]

    digest_result = _digest(config_dir, db_path, vault_dir, target_date)
    assert digest_result.exit_code == 0

    digest_path = vault_dir / "daily" / f"{target_date}.md"
    assert digest_path.exists()
    first_render = digest_path.read_text(encoding="utf-8")
    assert "alpha post" in first_render
    assert "https://example.com/alpha/post-1" in first_render

    # Second full pass: same feed, no new items to score, same digest date.
    ingest_result_2 = _ingest(config_dir, db_path, cache_dir)
    assert ingest_result_2.exit_code == 0

    def _must_not_be_called(*args: object, **kwargs: object) -> TriageBatchResult:
        raise AssertionError("nothing should be sent to the LLM on the second pass")

    monkeypatch.setattr("signalforge.llm.run_triage_batch", _must_not_be_called)
    score_result_2 = _run("score", "--config-dir", str(config_dir), "--db", str(db_path))
    assert score_result_2.exit_code == 0

    digest_result_2 = _digest(config_dir, db_path, vault_dir, target_date)
    assert digest_result_2.exit_code == 0
    second_render = digest_path.read_text(encoding="utf-8")

    with connection(db_path) as conn:
        item_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        score_count = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]

    assert item_count == 1
    assert score_count == 1
    assert second_render == first_render


def test_daily_command_runs_all_three_steps_in_one_invocation(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_result = TriageBatchResult(
        results={
            1: TriageResult(triage="keep", signal=5, relevance=5, novelty=4, reasoning="Good.")
        },
        input_tokens=1,
        output_tokens=1,
    )
    monkeypatch.setattr("signalforge.llm.run_triage_batch", lambda *a, **k: batch_result)

    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"etag": '"a-1"'})
        )
        result = _run(
            "daily",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
            "--vault-dir",
            str(vault_dir),
        )

    assert result.exit_code == 0
    with connection(db_path) as conn:
        kinds = [row[0] for row in conn.execute("SELECT kind FROM runs ORDER BY id")]
    assert kinds == ["ingest", "score", "daily"]

    today_digests = list((vault_dir / "daily").glob("*.md"))
    assert len(today_digests) == 1
    assert "alpha post" in today_digests[0].read_text(encoding="utf-8")


def test_daily_command_surfaces_a_config_error_but_still_exits(tmp_path: Path) -> None:
    result = _run(
        "daily",
        "--config-dir",
        str(tmp_path / "missing"),
        "--db",
        str(tmp_path / "data" / "signalforge.db"),
        "--cache-dir",
        str(tmp_path / "http_cache"),
        "--vault-dir",
        str(tmp_path / "vault"),
    )
    assert result.exit_code == 2
