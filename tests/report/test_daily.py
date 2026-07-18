"""Tests for the Daily Digest (`report/daily.py`, DESIGN §13).

Scope: pure assembly (`build_digest_context`), template rendering
(`render_digest`), and the write path (`write_digest`). Every test builds its
own throwaway DB via the `conn`/`make_item` fixtures in `tests/conftest.py` —
never the real `data/signalforge.db` (CLAUDE.md §8).

`report/` never calls an LLM (CLAUDE.md §2), so these tests insert `scores`
rows directly via SQL rather than depending on the concurrently-developed
`score/` pipeline's write path — that keeps this suite decoupled from a module
this task does not own.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from signalforge.db import upsert_item
from signalforge.models import Item, SourceType
from signalforge.report.daily import (
    DigestContext,
    _to_line,
    build_digest_context,
    digest_path,
    render_digest,
    utc_day_window,
    write_digest,
)
from tests.conftest import make_item

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FIXTURE = REPO_ROOT / "fixtures" / "daily_digest_golden.md"

TARGET_DATE = date(2026, 7, 16)
SCORED_AT = "2026-07-16T06:05:00+00:00"

MAX_ITEMS = 15
"""Mirrors the shipped `thresholds.daily_max_items`. The cap itself is config
(CLAUDE.md §4) — tests that exercise truncation pass a small value explicitly."""


def _insert_score(
    conn: sqlite3.Connection,
    item_id: int,
    *,
    triage: str = "keep",
    signal: int | None = 4,
    relevance: int | None = 4,
    novelty: int | None = 3,
    reasoning: str = "A perfectly ordinary reason this item matters.",
    rubric_version: str = "v1",
    model: str = "claude-haiku-4-5",
    scored_at: str = SCORED_AT,
) -> None:
    """Insert one `scores` row directly — no dependency on `score/`'s writer."""
    conn.execute(
        """
        INSERT INTO scores (
            item_id, triage, signal, relevance, novelty, reasoning,
            rubric_version, model, scored_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (item_id, triage, signal, relevance, novelty, reasoning, rubric_version, model, scored_at),
    )


def _insert_ingest_run_with_errors(conn: sqlite3.Connection, errors: list[dict[str, str]]) -> None:
    import json

    conn.execute(
        """
        INSERT INTO runs (kind, started_at, finished_at, status, items_new, errors)
        VALUES ('ingest', '2026-07-16T05:00:00+00:00', '2026-07-16T05:02:00+00:00',
                'partial', 3, ?)
        """,
        (json.dumps(errors),),
    )


# --------------------------------------------------------------------------- #
# build_digest_context — ordering, footer counts, empty-day handling
# --------------------------------------------------------------------------- #


def test_build_digest_context_orders_kept_items_by_total_score_desc(
    conn: sqlite3.Connection,
) -> None:
    low_id, _ = upsert_item(
        conn, make_item(external_id="g-low", url="https://example.com/low", title="Low scorer")
    )
    high_id, _ = upsert_item(
        conn, make_item(external_id="g-high", url="https://example.com/high", title="High scorer")
    )
    _insert_score(conn, low_id, signal=2, relevance=2, novelty=2)
    _insert_score(conn, high_id, signal=5, relevance=5, novelty=5)

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)

    assert [line.title for line in context.items] == ["High scorer", "Low scorer"]


def test_build_digest_context_excludes_killed_items_but_counts_them(
    conn: sqlite3.Connection,
) -> None:
    kept_id, _ = upsert_item(
        conn, make_item(external_id="g-keep", url="https://example.com/keep", title="Kept")
    )
    killed_id, _ = upsert_item(
        conn, make_item(external_id="g-kill", url="https://example.com/kill", title="Killed")
    )
    _insert_score(conn, kept_id, triage="keep")
    _insert_score(conn, killed_id, triage="kill", signal=1, relevance=1, novelty=1)

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)

    assert [line.title for line in context.items] == ["Kept"]
    assert context.killed_count == 1
    assert context.scored_count == 2


def test_build_digest_context_excludes_items_scored_on_a_different_date(
    conn: sqlite3.Connection,
) -> None:
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id, scored_at="2026-07-15T06:05:00+00:00")

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)

    assert context.items == ()
    assert context.killed_count == 0
    assert context.scored_count == 0


def test_build_digest_context_with_nothing_scored_is_empty_not_an_error(
    conn: sqlite3.Connection,
) -> None:
    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)

    assert context == DigestContext(
        date=TARGET_DATE,
        items=(),
        source_failures=(),
        killed_count=0,
        scored_count=0,
        hidden_kept_count=0,
    )


def test_why_it_matters_is_the_stored_reasoning_verbatim_when_short(
    conn: sqlite3.Connection,
) -> None:
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id, reasoning="Short and clear reasoning.")

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)

    assert context.items[0].why_it_matters == "Short and clear reasoning."


def test_why_it_matters_is_trimmed_for_a_long_reasoning_string(conn: sqlite3.Connection) -> None:
    item_id, _ = upsert_item(conn, make_item())
    long_reasoning = "word " * 200
    _insert_score(conn, item_id, reasoning=long_reasoning)

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)

    line = context.items[0].why_it_matters
    assert len(line) < len(long_reasoning)
    assert line.endswith("…")


def test_source_failures_come_from_the_latest_ingest_run(conn: sqlite3.Connection) -> None:
    _insert_ingest_run_with_errors(
        conn,
        [
            {
                "source_id": "interconnects",
                "error_type": "FetchError",
                "message": "HTTP 503",
                "occurred_at": "2026-07-16T05:01:00+00:00",
            }
        ],
    )

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)

    assert len(context.source_failures) == 1
    assert context.source_failures[0].source_id == "interconnects"
    assert context.source_failures[0].message == "HTTP 503"


def test_run_level_errors_are_excluded_from_source_failures(conn: sqlite3.Connection) -> None:
    """A crash outside any single source (`source_id == "*"`) is not a *source* failure."""
    _insert_ingest_run_with_errors(
        conn,
        [{"source_id": "*", "error_type": "RuntimeError", "message": "the network fell over"}],
    )

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)

    assert context.source_failures == ()


def test_no_ingest_run_yet_means_no_source_failures(conn: sqlite3.Connection) -> None:
    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    assert context.source_failures == ()


# --------------------------------------------------------------------------- #
# The daily_max_items cap — DESIGN §13's "5–15 kept items… 60-second read"
# --------------------------------------------------------------------------- #


def _seed_ranked_items(conn: sqlite3.Connection, count: int) -> None:
    """`count` kept items with strictly descending totals, so rank is unambiguous:
    item-1 scores highest, item-`count` lowest."""
    for rank in range(1, count + 1):
        item_id, _ = upsert_item(
            conn,
            make_item(
                external_id=f"ranked-{rank}",
                url=f"https://example.com/ranked-{rank}",
                title=f"Ranked item {rank}",
            ),
        )
        score = 5 - (rank - 1)  # 5, 4, 3, … — total drops by 3 per rank.
        _insert_score(conn, item_id, signal=score, relevance=score, novelty=score)


def test_cap_truncates_to_the_top_n_and_counts_the_rest(conn: sqlite3.Connection) -> None:
    _seed_ranked_items(conn, 5)

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=3)

    assert [line.title for line in context.items] == [
        "Ranked item 1",
        "Ranked item 2",
        "Ranked item 3",
    ]
    assert context.hidden_kept_count == 2
    # Kept-vs-killed semantics are untouched by the cap: all 5 were scored.
    assert context.killed_count == 0
    assert context.scored_count == 5


def test_cap_leaves_a_short_day_alone(conn: sqlite3.Connection) -> None:
    _seed_ranked_items(conn, 2)

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=3)

    assert len(context.items) == 2
    assert context.hidden_kept_count == 0


def test_cap_equal_to_kept_count_hides_nothing(conn: sqlite3.Connection) -> None:
    _seed_ranked_items(conn, 3)

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=3)

    assert len(context.items) == 3
    assert context.hidden_kept_count == 0


def test_top_n_is_deterministic_across_re_renders(conn: sqlite3.Connection) -> None:
    """Same date, same DB state, same cap ⇒ the same N items in the same order —
    the cap must never turn re-rendering into a shuffle (CLAUDE.md §3)."""
    _seed_ranked_items(conn, 5)

    first = build_digest_context(conn, target_date=TARGET_DATE, max_items=3)
    second = build_digest_context(conn, target_date=TARGET_DATE, max_items=3)

    assert first == second
    assert render_digest(first) == render_digest(second)


def test_truncation_footer_line_renders_only_when_items_are_hidden(
    conn: sqlite3.Connection,
) -> None:
    _seed_ranked_items(conn, 5)

    capped = render_digest(build_digest_context(conn, target_date=TARGET_DATE, max_items=3))
    uncapped = render_digest(build_digest_context(conn, target_date=TARGET_DATE, max_items=15))

    assert "2 more kept item(s) not shown" in capped
    assert "item_count: 3" in capped
    assert "kept_count: 5" in capped
    assert "Ranked item 4" not in capped

    assert "not shown" not in uncapped
    assert "item_count: 5" in uncapped
    assert "kept_count: 5" in uncapped


# --------------------------------------------------------------------------- #
# Citation discipline — NEVER rule 7
# --------------------------------------------------------------------------- #


def test_to_line_drops_an_item_with_no_url_rather_than_render_an_uncited_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`Item.url` is required, so this path should be unreachable in practice —
    guarded anyway (CLAUDE.md §5, NEVER rule 7) via `Item.model_construct`,
    which bypasses pydantic validation the way a corrupted row or a future
    nullable field might.
    """
    from signalforge.db import DigestItem

    item = Item.model_construct(
        id=1,
        source_id="x",
        source_type=SourceType.RSS,
        external_id=None,
        url="",
        canonical_url="https://example.com/x",
        title="An item with no URL",
        author=None,
        published_at=None,
        fetched_at=datetime.now(UTC),
        summary=None,
        content=None,
        content_hash="deadbeef",
        lang="en",
        raw_path=None,
    )
    scored = DigestItem(
        item=item,
        signal=5,
        relevance=5,
        novelty=5,
        reasoning="This would otherwise be a fine item.",
        model="claude-haiku-4-5",
        rubric_version="v1",
        scored_at=datetime.now(UTC),
    )

    with caplog.at_level("WARNING"):
        line = _to_line(scored)

    assert line is None
    assert "citation" in caplog.text.lower() or "url" in caplog.text.lower()


# --------------------------------------------------------------------------- #
# Rendering — golden file
# --------------------------------------------------------------------------- #


def test_render_digest_matches_the_golden_fixture(conn: sqlite3.Connection) -> None:
    id1, _ = upsert_item(
        conn,
        Item(
            source_id="simonwillison",
            source_type=SourceType.RSS,
            external_id="guid-1",
            url="https://simonwillison.net/2026/Jul/15/mcp-sampling/",
            title="MCP sampling lands everywhere",
            author="Simon Willison",
            published_at=datetime(2026, 7, 15, 12, 30, tzinfo=UTC),
            fetched_at=datetime(2026, 7, 16, 6, 0, tzinfo=UTC),
            summary="A short feed summary.",
        ),
    )
    id2, _ = upsert_item(
        conn,
        Item(
            source_id="hn",
            source_type=SourceType.HN,
            external_id="4242",
            url="https://example.com/agent-memory",
            title="A new approach to agent memory",
            fetched_at=datetime(2026, 7, 16, 6, 0, tzinfo=UTC),
            summary="Summary two.",
        ),
    )
    id3, _ = upsert_item(
        conn,
        Item(
            source_id="hn",
            source_type=SourceType.HN,
            external_id="9999",
            url="https://example.com/hype-post",
            title="Yet another hype post",
            fetched_at=datetime(2026, 7, 16, 6, 0, tzinfo=UTC),
            summary="Summary three.",
        ),
    )
    _insert_score(
        conn,
        id1,
        signal=5,
        relevance=4,
        novelty=3,
        reasoning=(
            "Working code and benchmarks showing real throughput gains on production MCP servers."
        ),
    )
    _insert_score(
        conn,
        id2,
        signal=4,
        relevance=5,
        novelty=4,
        reasoning=(
            "Directly touches agent memory, a stated learning goal, with a credible new mechanism."
        ),
    )
    _insert_score(
        conn,
        id3,
        triage="kill",
        signal=1,
        relevance=1,
        novelty=1,
        reasoning="Press release language, no artifact.",
    )
    _insert_ingest_run_with_errors(
        conn,
        [
            {
                "source_id": "interconnects",
                "error_type": "FetchError",
                "message": "HTTP 503",
                "occurred_at": "2026-07-16T05:01:00+00:00",
            }
        ],
    )

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    rendered = render_digest(context)

    expected = GOLDEN_FIXTURE.read_text(encoding="utf-8")
    assert rendered == expected


def test_render_digest_with_no_items_renders_sensibly(conn: sqlite3.Connection) -> None:
    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)

    rendered = render_digest(context)

    assert "item_count: 0" in rendered
    assert "No items cleared triage today" in rendered
    assert "No source failures in the last ingest run." in rendered
    assert "0 item(s) killed at triage · 0 scored today." in rendered


# --------------------------------------------------------------------------- #
# write_digest — idempotent overwrite (CLAUDE.md §3, NEVER rule 4)
# --------------------------------------------------------------------------- #


def test_write_digest_creates_the_expected_path(conn: sqlite3.Connection, tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"

    path = write_digest(conn, target_date=TARGET_DATE, vault_dir=vault_dir, max_items=MAX_ITEMS)

    assert path == vault_dir / "daily" / "2026-07-16.md"
    assert path.is_file()


def test_write_digest_twice_overwrites_rather_than_duplicating(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    vault_dir = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)

    first_path = write_digest(
        conn, target_date=TARGET_DATE, vault_dir=vault_dir, max_items=MAX_ITEMS
    )
    first_content = first_path.read_text(encoding="utf-8")

    # A second run for the same date, DB state unchanged: byte-for-byte no-op.
    second_path = write_digest(
        conn, target_date=TARGET_DATE, vault_dir=vault_dir, max_items=MAX_ITEMS
    )
    second_content = second_path.read_text(encoding="utf-8")

    assert second_path == first_path
    assert second_content == first_content
    assert list((vault_dir / "daily").glob("2026-07-16*")) == [first_path]


def test_write_digest_overwrite_reflects_updated_db_state(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Overwrite means overwrite: a re-run must show the *current* DB state,
    not stale content from the first render."""
    vault_dir = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item(title="First title"))
    _insert_score(conn, item_id, reasoning="First reasoning.")
    write_digest(conn, target_date=TARGET_DATE, vault_dir=vault_dir, max_items=MAX_ITEMS)

    conn.execute(
        "UPDATE scores SET reasoning = ? WHERE item_id = ?", ("Updated reasoning.", item_id)
    )
    path = write_digest(conn, target_date=TARGET_DATE, vault_dir=vault_dir, max_items=MAX_ITEMS)

    content = path.read_text(encoding="utf-8")
    assert "Updated reasoning." in content
    assert "First reasoning." not in content


def test_digest_path_is_stable_for_a_given_date(tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    assert digest_path(vault_dir, target_date=TARGET_DATE) == digest_path(
        vault_dir, target_date=TARGET_DATE
    )


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Crowding limits — daily_max_per_source, daily_max_per_github_repo
# --------------------------------------------------------------------------- #


def _insert_release(
    conn: sqlite3.Connection,
    *,
    repo: str,
    tag: str,
    published_at: datetime,
    total: int = 15,
) -> int:
    """One GitHub release item + score. `source_id` is the repo, as the real
    release-watch ingestor writes it."""
    item_id, _ = upsert_item(
        conn,
        make_item(
            source_id=repo,
            source_type=SourceType.GITHUB,
            external_id=f"{repo}@{tag}",
            url=f"https://github.com/{repo}/releases/tag/{tag}",
            title=f"{repo} {tag}",
            published_at=published_at,
        ),
    )
    per_dimension, remainder = divmod(total, 3)
    _insert_score(
        conn,
        item_id,
        signal=per_dimension + remainder,
        relevance=per_dimension,
        novelty=per_dimension,
    )
    return item_id


def _insert_post(conn: sqlite3.Connection, *, slug: str, total: int = 12) -> int:
    """One RSS post + score from the default `simonwillison` source."""
    item_id, _ = upsert_item(
        conn,
        make_item(
            external_id=slug,
            url=f"https://simonwillison.net/2026/Jul/15/{slug}/",
            title=f"Post {slug}",
        ),
    )
    per_dimension, remainder = divmod(total, 3)
    _insert_score(
        conn,
        item_id,
        signal=per_dimension + remainder,
        relevance=per_dimension,
        novelty=per_dimension,
    )
    return item_id


def test_github_repo_limit_collapses_a_version_pile_to_one(conn: sqlite3.Connection) -> None:
    """Four versions of one library landing in one window is one piece of news,
    not four — and must not eat four of the digest's slots."""
    for tag, total in [("3.2.0", 15), ("3.1.1", 14), ("3.1.0", 14), ("3.0.4", 14)]:
        _insert_release(
            conn,
            repo="stanfordnlp/dspy",
            tag=tag,
            published_at=datetime(2026, 4, 21, tzinfo=UTC),
            total=total,
        )

    context = build_digest_context(
        conn, target_date=TARGET_DATE, max_items=15, max_per_github_repo=1
    )

    assert [line.title for line in context.items] == ["stanfordnlp/dspy 3.2.0"]
    # The other three are still kept items — hidden, never silently dropped.
    assert context.hidden_kept_count == 3
    assert context.kept_count == 4


def test_github_repo_limit_keeps_the_best_release_not_the_newest(
    conn: sqlite3.Connection,
) -> None:
    """The regression this rule exists for: a prerelease publishes *after* the
    stable release it follows, so picking by recency hands the slot to a beta
    and drops the release that earned the score."""
    _insert_release(
        conn,
        repo="stanfordnlp/dspy",
        tag="3.2.0",
        published_at=datetime(2026, 4, 21, tzinfo=UTC),
        total=15,
    )
    _insert_release(
        conn,
        repo="stanfordnlp/dspy",
        tag="3.3.0b1",
        published_at=datetime(2026, 5, 28, tzinfo=UTC),
        total=10,
    )

    context = build_digest_context(
        conn, target_date=TARGET_DATE, max_items=15, max_per_github_repo=1
    )

    assert [line.title for line in context.items] == ["stanfordnlp/dspy 3.2.0"]


def test_github_repo_limit_is_per_repo_not_across_repos(conn: sqlite3.Connection) -> None:
    """Two repos each shipping a release are two separate pieces of news."""
    _insert_release(
        conn, repo="ollama/ollama", tag="v0.32.0", published_at=datetime(2026, 7, 14, tzinfo=UTC)
    )
    _insert_release(
        conn, repo="stanfordnlp/dspy", tag="3.2.0", published_at=datetime(2026, 7, 15, tzinfo=UTC)
    )

    context = build_digest_context(
        conn, target_date=TARGET_DATE, max_items=15, max_per_github_repo=1
    )

    assert len(context.items) == 2


def test_github_repo_limit_leaves_non_github_items_alone(conn: sqlite3.Connection) -> None:
    """The rule is about release piles; two posts from one blog are both still
    news. `daily_max_per_source` is what bounds those."""
    for n in range(3):
        _insert_post(conn, slug=f"post-{n}")

    context = build_digest_context(
        conn, target_date=TARGET_DATE, max_items=15, max_per_github_repo=1
    )

    assert len(context.items) == 3


def test_max_per_source_caps_a_prolific_source_and_promotes_the_tail(
    conn: sqlite3.Connection,
) -> None:
    """The whole point: a link blog sweeping the top of the ranking must not
    crowd out a lower-ranked item from a different source."""
    for rank in range(3):
        _insert_post(conn, slug=f"sw-{rank}", total=15)

    other, _ = upsert_item(
        conn,
        make_item(
            source_id="jxnl",
            external_id="lessons",
            url="https://jxnl.co/writing/lessons/",
            title="Lessons from industry leaders",
        ),
    )
    _insert_score(conn, other, signal=4, relevance=5, novelty=4)

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=15, max_per_source=2)

    assert [line.title for line in context.items] == [
        "Post sw-0",
        "Post sw-1",
        "Lessons from industry leaders",
    ]
    assert context.hidden_kept_count == 1


def test_max_per_source_keeps_each_source_s_best(conn: sqlite3.Connection) -> None:
    """The cap drops a source's *weakest* items — it takes the top slice of the
    ranking within a source, never an arbitrary slice."""
    for slug, total in [("weak", 9), ("best", 15), ("mid", 12)]:
        _insert_post(conn, slug=slug, total=total)

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=15, max_per_source=2)

    assert [line.title for line in context.items] == ["Post best", "Post mid"]


def test_limits_are_off_by_default(conn: sqlite3.Connection) -> None:
    """Absent config changes nothing — every kept item still renders."""
    for rank in range(3):
        _insert_post(conn, slug=f"sw-{rank}")
    for tag in ["3.2.0", "3.1.1"]:
        _insert_release(
            conn, repo="stanfordnlp/dspy", tag=tag, published_at=datetime(2026, 4, 21, tzinfo=UTC)
        )

    context = build_digest_context(conn, target_date=TARGET_DATE, max_items=15)

    assert len(context.items) == 5


def test_github_repo_limit_wins_over_the_looser_per_source_limit(
    conn: sqlite3.Connection,
) -> None:
    """A repo is also a source, so both limits match it. The tighter one must
    decide — otherwise `daily_max_per_github_repo: 1` would be a no-op."""
    for tag, total in [("3.2.0", 15), ("3.1.1", 14), ("3.1.0", 13)]:
        _insert_release(
            conn,
            repo="stanfordnlp/dspy",
            tag=tag,
            published_at=datetime(2026, 4, 21, tzinfo=UTC),
            total=total,
        )

    context = build_digest_context(
        conn,
        target_date=TARGET_DATE,
        max_items=15,
        max_per_source=2,
        max_per_github_repo=1,
    )

    assert [line.title for line in context.items] == ["stanfordnlp/dspy 3.2.0"]


def test_limits_never_reorder_the_ranking(conn: sqlite3.Connection) -> None:
    """Filtering must leave a sub-sequence of the ranking — a crowded-out item
    must not promote a lower-ranked one above a higher-ranked one."""
    _insert_post(conn, slug="sw-best", total=15)
    _insert_post(conn, slug="sw-second", total=14)
    _insert_post(conn, slug="sw-third", total=13)
    _insert_release(
        conn,
        repo="stanfordnlp/dspy",
        tag="3.2.0",
        published_at=datetime(2026, 4, 21, tzinfo=UTC),
        total=12,
    )

    context = build_digest_context(
        conn, target_date=TARGET_DATE, max_items=15, max_per_source=2, max_per_github_repo=1
    )

    titles = [line.title for line in context.items]
    assert titles == ["Post sw-best", "Post sw-second", "stanfordnlp/dspy 3.2.0"]


def test_crowding_limits_are_deterministic_across_re_renders(conn: sqlite3.Connection) -> None:
    """Filtering must never turn re-rendering into a shuffle (CLAUDE.md §3)."""
    for rank in range(4):
        _insert_post(conn, slug=f"sw-{rank}")
    _insert_release(
        conn, repo="stanfordnlp/dspy", tag="3.2.0", published_at=datetime(2026, 4, 21, tzinfo=UTC)
    )

    kwargs = {"max_items": 15, "max_per_source": 2, "max_per_github_repo": 1}
    first = build_digest_context(conn, target_date=TARGET_DATE, **kwargs)  # type: ignore[arg-type]
    second = build_digest_context(conn, target_date=TARGET_DATE, **kwargs)  # type: ignore[arg-type]

    assert first == second
    assert render_digest(first) == render_digest(second)


# --------------------------------------------------------------------------- #
# Timezone — the reader's local day, resolved from a UTC store (settings.yaml)
# --------------------------------------------------------------------------- #

SYDNEY = ZoneInfo("Australia/Sydney")  # UTC+10, no DST in July
NEW_YORK = ZoneInfo("America/New_York")


def test_utc_day_window_utc_is_plain_midnights() -> None:
    start, end = utc_day_window(date(2026, 7, 16), UTC)
    assert start == "2026-07-16T00:00:00+00:00"
    assert end == "2026-07-17T00:00:00+00:00"


def test_utc_day_window_shifts_for_a_positive_offset_zone() -> None:
    # Sydney is UTC+10, so local 2026-07-18 spans UTC 2026-07-17T14:00 .. 18T14:00.
    start, end = utc_day_window(date(2026, 7, 18), SYDNEY)
    assert start == "2026-07-17T14:00:00+00:00"
    assert end == "2026-07-18T14:00:00+00:00"


def test_utc_day_window_is_dst_correct_not_a_fixed_24h() -> None:
    # US spring-forward: 2026-03-08 is a 23-hour day in New York. The window
    # must be that real calendar day, not start+24h.
    start, end = utc_day_window(date(2026, 3, 8), NEW_YORK)
    span = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    assert span.total_seconds() == 23 * 3600


def test_digest_day_uses_the_configured_zone_not_utc(conn: sqlite3.Connection) -> None:
    """The actual 2026-07-18 bug: an item scored at 22:23 UTC on the 17th is
    08:23 on the 18th in Sydney, so it belongs to the Sydney-local 18th digest —
    exactly the day the old UTC logic left empty."""
    item_id, _ = upsert_item(conn, make_item(external_id="late", title="Late-night item"))
    _insert_score(conn, item_id, scored_at="2026-07-17T22:23:59+00:00")

    on_the_18th = build_digest_context(conn, target_date=date(2026, 7, 18), tz=SYDNEY, max_items=15)
    on_the_17th = build_digest_context(conn, target_date=date(2026, 7, 17), tz=SYDNEY, max_items=15)

    assert [line.title for line in on_the_18th.items] == ["Late-night item"]
    assert on_the_17th.items == ()


def test_utc_default_preserves_the_old_calendar(conn: sqlite3.Connection) -> None:
    """With no zone (tz defaults to UTC), the same item files under its UTC date —
    proving the range query is backward compatible with the date-prefix logic."""
    item_id, _ = upsert_item(conn, make_item(external_id="late", title="Late-night item"))
    _insert_score(conn, item_id, scored_at="2026-07-17T22:23:59+00:00")

    assert [
        line.title
        for line in build_digest_context(conn, target_date=date(2026, 7, 17), max_items=15).items
    ] == ["Late-night item"]
    assert build_digest_context(conn, target_date=date(2026, 7, 18), max_items=15).items == ()


def test_local_midnight_belongs_to_the_new_day_not_the_old(conn: sqlite3.Connection) -> None:
    """Half-open [start, end): an item at exactly local midnight is the first
    item of the new day, never the last of the previous one."""
    # Sydney midnight 2026-07-18T00:00+10:00 == 2026-07-17T14:00:00 UTC.
    item_id, _ = upsert_item(conn, make_item(external_id="mid", title="Midnight item"))
    _insert_score(conn, item_id, scored_at="2026-07-17T14:00:00+00:00")

    assert (
        build_digest_context(conn, target_date=date(2026, 7, 17), tz=SYDNEY, max_items=15).items
        == ()
    )
    assert [
        line.title
        for line in build_digest_context(
            conn, target_date=date(2026, 7, 18), tz=SYDNEY, max_items=15
        ).items
    ] == ["Midnight item"]


def test_fractional_seconds_at_the_window_seam(conn: sqlite3.Connection) -> None:
    """The whole range-comparison rests on lexical order matching time order at
    the seam: stored values carry microseconds (`...SS.ffffff+00:00`) while the
    bounds are whole-second (`...SS+00:00`), and `.` (0x2E) > `+` (0x2B) is what
    keeps a sub-second-into-the-day item on the new day. Lock that ordering in —
    a bound that regained fractional digits would break it silently."""
    # Sydney 2026-07-18 starts at 2026-07-17T14:00:00 UTC (the `start` bound).
    just_in, _ = upsert_item(
        conn, make_item(external_id="in", url="https://example.com/in", title="Just inside")
    )
    _insert_score(conn, just_in, scored_at="2026-07-17T14:00:00.000001+00:00")
    just_out, _ = upsert_item(
        conn, make_item(external_id="out", url="https://example.com/out", title="Just outside")
    )
    _insert_score(conn, just_out, scored_at="2026-07-17T13:59:59.999999+00:00")

    ctx = build_digest_context(conn, target_date=date(2026, 7, 18), tz=SYDNEY, max_items=15)
    assert [line.title for line in ctx.items] == ["Just inside"]


def test_killed_count_shares_the_digest_window(conn: sqlite3.Connection) -> None:
    """The footer's killed count must use the same local-day window as the kept
    items, or the counts stop reconciling across the UTC-midnight seam."""
    kept, _ = upsert_item(
        conn, make_item(external_id="k", url="https://example.com/kept", title="Kept")
    )
    _insert_score(conn, kept, scored_at="2026-07-17T22:00:00+00:00")
    killed, _ = upsert_item(
        conn, make_item(external_id="x", url="https://example.com/killed", title="Killed")
    )
    _insert_score(conn, killed, triage="kill", scored_at="2026-07-17T23:00:00+00:00")

    ctx = build_digest_context(conn, target_date=date(2026, 7, 18), tz=SYDNEY, max_items=15)
    assert ctx.killed_count == 1
    assert ctx.kept_count == 1
    assert ctx.scored_count == 2
