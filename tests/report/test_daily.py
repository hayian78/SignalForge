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

import json
import sqlite3
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from signalforge.curate.approvals import parse_proposal_marks
from signalforge.db import decide_proposal, insert_proposal, start_run, upsert_item
from signalforge.feedback import CHECKBOX_VERDICTS, checkbox_marker, parse_marks
from signalforge.models import (
    Item,
    ProposalKind,
    ProposalStatus,
    ProposalTier,
    SourceType,
)
from signalforge.report.daily import (
    DigestContext,
    _to_line,
    build_digest_context,
    digest_path,
    render_digest,
    utc_day_window,
    utc_range_window,
    write_digest,
)
from signalforge.score.taxonomy import TAXONOMY_VERSION
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
        line = _to_line(scored, {})

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


@pytest.mark.parametrize("tz", [UTC, SYDNEY, NEW_YORK])
@pytest.mark.parametrize("day", [date(2026, 3, 8), date(2026, 7, 18), date(2026, 11, 1)])
def test_a_one_day_range_is_exactly_the_day_window(tz: tzinfo, day: date) -> None:
    """`utc_day_window` is the single-day case of `utc_range_window` and nothing
    more. Asserting the identity — rather than trusting the delegation — is what
    lets the DST tests around this one keep guarding both callers at once."""
    assert utc_day_window(day, tz) == utc_range_window(day, day, tz)


def test_a_seven_day_range_spanning_dst_is_not_168_hours() -> None:
    """The weekly brief's window is seven *calendar* days, not a 168-hour slab.
    2026-03-08 is New York's spring-forward, so the week containing it is 167h —
    an hour a `start + 7 * 24h` implementation would silently hand to the
    following week, every year, in one direction only."""
    start, end = utc_range_window(date(2026, 3, 2), date(2026, 3, 8), NEW_YORK)
    span = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    assert span.total_seconds() == 167 * 3600

    # Autumn's fall-back is the mirror image: 169 hours, not 168.
    start, end = utc_range_window(date(2026, 10, 26), date(2026, 11, 1), NEW_YORK)
    span = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    assert span.total_seconds() == 169 * 3600


def test_consecutive_weekly_windows_tile_without_gap_or_overlap() -> None:
    """Each Sunday's brief covers the seven days *before* it, so one week's
    `end` is the next week's `start` exactly. A gap here would strand a day's
    items in no brief at all — permanently, since nothing re-scores them.

    Deliberately straddling New York's spring-forward: in a zone with no
    transition a broken `start + 7 * 24h` implementation tiles just as cleanly,
    so this would pass on the very bug the test above exists to catch."""
    first = utc_range_window(date(2026, 3, 1), date(2026, 3, 7), NEW_YORK)
    second = utc_range_window(date(2026, 3, 8), date(2026, 3, 14), NEW_YORK)
    assert first[1] == second[0]


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


# --------------------------------------------------------------------------- #
# The source-curation approval block (DESIGN §7.1)
# --------------------------------------------------------------------------- #

PROPOSALS_GOLDEN_FIXTURE = REPO_ROOT / "fixtures" / "daily_digest_proposals_golden.md"
SURFACE_DATE = date(2026, 7, 16)
SETTLED_DAYS = 14
"""Mirrors the shipped `curation.settled_display_days`; the window is config."""


def _add_proposal(
    conn: sqlite3.Connection,
    *,
    kind: ProposalKind = ProposalKind.ADD_RSS,
    dedup_key: str = "https://newvoice.example.com/feed",
    payload: dict[str, object] | None = None,
    rationale: str = "Cited three times this month by items you marked useful.",
    evidence: list[dict[str, str]] | None = None,
    probe: dict[str, object] | None = None,
    tier: ProposalTier = ProposalTier.WEB,
    status: ProposalStatus = ProposalStatus.PENDING,
    surface_date: date = SURFACE_DATE,
) -> int:
    proposal_id = insert_proposal(
        conn,
        # A real run id, as production always has: a proposal is the audit trail
        # for why a `sources.yaml` edit happened, so it is never run-less.
        run_id=start_run(conn, "curate", started_at=datetime(2026, 7, 16, 6, 30, tzinfo=UTC)),
        kind=kind,
        dedup_key=dedup_key,
        payload=payload if payload is not None else {"id": "newvoice", "url": dedup_key},
        rationale=rationale,
        evidence=evidence
        if evidence is not None
        else [{"url": "https://simonwillison.net/2026/Jul/12/link/", "note": "linked twice"}],
        probe=probe,
        tier=tier,
        status=status,
        surface_date=surface_date,
        created_at=datetime(2026, 7, 16, 6, 0, tzinfo=UTC),
    )
    assert proposal_id is not None
    return proposal_id


def _proposal_context(conn: sqlite3.Connection, **kwargs: object) -> DigestContext:
    return build_digest_context(
        conn,
        target_date=kwargs.pop("target_date", TARGET_DATE),  # type: ignore[arg-type]
        max_items=MAX_ITEMS,
        settled_display_days=kwargs.pop("settled_display_days", SETTLED_DAYS),  # type: ignore[arg-type]
    )


def test_a_pending_proposal_renders_both_checkboxes_that_the_harvester_parses(
    conn: sqlite3.Connection,
) -> None:
    """The wire format has one definition, and this proves the round trip.

    The template renders through `proposal_marker` and the harvester reads through
    `PROPOSAL_MARK_RE`; a digest whose checkboxes the harvester cannot parse would
    silently swallow every decision the operator makes.
    """
    proposal_id = _add_proposal(conn)

    rendered = render_digest(_proposal_context(conn))

    ticked = rendered.replace("- [ ] approve", "- [x] approve")
    marks = parse_proposal_marks(ticked)
    assert [(mark.proposal_id, mark.decision) for mark in marks] == [(proposal_id, "approve")]


def test_a_settled_proposal_renders_no_checkbox(conn: sqlite3.Connection) -> None:
    """A decided proposal is a record, not a question. Offering a box would invite a
    second decision that `db.decide_proposal`'s pending guard would then ignore."""
    proposal_id = _add_proposal(conn)
    decide_proposal(
        conn,
        proposal_id=proposal_id,
        status=ProposalStatus.APPROVED,
        decided_at=datetime(2026, 7, 16, 7, 0, tzinfo=UTC),
    )

    rendered = render_digest(_proposal_context(conn))

    assert "sf:proposal=" not in rendered
    assert "approved 2026-07-16" in rendered


def test_a_pending_proposal_follows_forward_into_later_digests(
    conn: sqlite3.Connection,
) -> None:
    """An unanswered question must not scroll out of sight (DESIGN §7.1)."""
    _add_proposal(conn, surface_date=date(2026, 7, 1))

    rendered = render_digest(_proposal_context(conn, target_date=TARGET_DATE))

    assert "sf:proposal=" in rendered


def test_a_proposal_surfacing_tomorrow_is_not_shown_today(conn: sqlite3.Connection) -> None:
    _add_proposal(conn, surface_date=date(2026, 7, 20))

    rendered = render_digest(_proposal_context(conn, target_date=TARGET_DATE))

    assert "Proposed source changes" not in rendered


def test_a_settled_proposal_drops_out_after_its_display_window(
    conn: sqlite3.Connection,
) -> None:
    """Counted from the day it surfaced, so an old digest still shows its own week."""
    proposal_id = _add_proposal(conn, surface_date=date(2026, 7, 1))
    decide_proposal(
        conn,
        proposal_id=proposal_id,
        status=ProposalStatus.REJECTED,
        decided_at=datetime(2026, 7, 2, 7, 0, tzinfo=UTC),
    )

    within = render_digest(_proposal_context(conn, target_date=date(2026, 7, 14)))
    beyond = render_digest(_proposal_context(conn, target_date=date(2026, 7, 16)))

    assert "rejected" in within
    assert "Proposed source changes" not in beyond


def test_an_arxiv_proposal_is_no_longer_tagged_staged(conn: sqlite3.Connection) -> None:
    """`ingest/arxiv.py` shipped 2026-08-07: applying this now has a real effect,
    so the digest must not claim otherwise (see `ProposalKind.is_staged`)."""
    _add_proposal(
        conn,
        kind=ProposalKind.ADD_ARXIV_KEYWORD,
        dedup_key="interpretability",
        payload={"target": "interpretability"},
    )

    rendered = render_digest(_proposal_context(conn))

    assert "staged" not in rendered


def test_a_suggested_weight_is_shown_with_how_to_override_it(
    conn: sqlite3.Connection,
) -> None:
    """It is part of what a tick approves, so it cannot be invisible."""
    _add_proposal(
        conn, payload={"id": "newvoice", "url": "https://newvoice.example.com/feed", "weight": 1.3}
    )

    rendered = render_digest(_proposal_context(conn))

    assert "suggested weight:** 1.3" in rendered
    assert "edit it in the diff" in rendered


def test_probe_facts_render_beside_the_proposal(conn: sqlite3.Connection) -> None:
    _add_proposal(
        conn,
        probe={
            "ok": True,
            "items_total": 24,
            "items_in_window": 3,
            "median_summary_chars": 1240,
            "newest_published_at": "2026-07-15T09:00:00+00:00",
            "label": "Nathan Lambert",
        },
    )

    rendered = render_digest(_proposal_context(conn))

    assert "24 entries, 3 recent, median 1240 chars, newest 2026-07-15" in rendered
    assert "latest by Nathan Lambert" in rendered


def test_an_invalid_proposal_is_recorded_as_considered_not_offered(
    conn: sqlite3.Connection,
) -> None:
    """It was never shown for approval, but "we looked and it 404s" is worth a line.

    Without it a candidate the operator might have wanted vanishes with no trace
    that it was ever considered.
    """
    _add_proposal(
        conn,
        status=ProposalStatus.INVALID,
        probe={"ok": False, "error": "HTTP 404", "status_code": 404},
    )

    rendered = render_digest(_proposal_context(conn))

    assert "sf:proposal=" not in rendered
    assert "not shown — HTTP 404" in rendered


def test_a_malformed_probe_blob_shortens_the_line_rather_than_breaking_the_digest(
    conn: sqlite3.Connection,
) -> None:
    """`db._decode_probe` is tolerant on purpose; this is why (CLAUDE.md §7)."""
    _add_proposal(conn)
    conn.execute("UPDATE proposals SET probe = ?", ("not json at all",))

    rendered = render_digest(_proposal_context(conn))

    assert "sf:proposal=" in rendered
    assert "checked:" not in rendered


def test_no_proposals_means_no_block_at_all(conn: sqlite3.Connection) -> None:
    """A digest from before curation existed must render exactly as it did."""
    rendered = render_digest(_proposal_context(conn))

    assert "Proposed source changes" not in rendered


def test_the_proposal_block_matches_the_golden_fixture(conn: sqlite3.Connection) -> None:
    """The block as a human actually reads it — pending, staged, and settled together."""
    _add_proposal(
        conn,
        payload={"id": "interconnects", "url": "https://newvoice.example.com/feed", "weight": 1.2},
        probe={
            "ok": True,
            "items_total": 24,
            "items_in_window": 3,
            "median_summary_chars": 1240,
            "newest_published_at": "2026-07-15T09:00:00+00:00",
            "label": "Nathan Lambert",
        },
    )
    _add_proposal(
        conn,
        kind=ProposalKind.RETIRE_GITHUB_REPO,
        dedup_key="block/goose",
        payload={"target": "block/goose"},
        rationale="0 useful against 4 noise marks this month; every release was a version bump.",
        evidence=[{"url": "https://github.com/block/goose/releases", "note": "bare tags"}],
        probe={"ok": True, "items_total": 12, "items_in_window": 0, "median_summary_chars": 40},
        tier=ProposalTier.CORPUS,
    )
    _add_proposal(
        conn,
        kind=ProposalKind.ADD_ARXIV_KEYWORD,
        dedup_key="interpretability",
        payload={"target": "interpretability"},
        rationale="Three of your kept items this month were interpretability papers.",
        evidence=[{"url": "https://arxiv.org/list/cs.AI/recent", "note": ""}],
        tier=ProposalTier.CORPUS,
    )
    settled = _add_proposal(
        conn,
        kind=ProposalKind.ADD_HN_KEYWORD,
        dedup_key="evaluation",
        payload={"target": "evaluation"},
        rationale="Recurring theme in what you keep.",
        evidence=[{"url": "https://news.ycombinator.com/item?id=1", "note": ""}],
        surface_date=date(2026, 7, 10),
    )
    decide_proposal(
        conn,
        proposal_id=settled,
        status=ProposalStatus.REJECTED,
        decided_at=datetime(2026, 7, 11, 7, 0, tzinfo=UTC),
        note="too broad, would flood the digest",
    )

    rendered = render_digest(_proposal_context(conn))

    expected = PROPOSALS_GOLDEN_FIXTURE.read_text(encoding="utf-8")
    assert rendered == expected


def test_the_proposal_block_re_renders_byte_identically(conn: sqlite3.Connection) -> None:
    """Idempotent rendering (CLAUDE.md §3): the block is a pure function of DB state."""
    _add_proposal(conn)
    _add_proposal(conn, dedup_key="https://other.example.com/feed", payload={"id": "other"})

    first = render_digest(_proposal_context(conn))
    second = render_digest(_proposal_context(conn))

    assert first == second


def test_a_forged_marker_in_a_rationale_cannot_approve_anything(
    conn: sqlite3.Connection,
) -> None:
    """The attack that would have bypassed the human gate entirely.

    The digest renders the scout's rationale into a vault file, and
    `harvest_approvals` reads a decision from *any line* matching its checkbox
    pattern. A rationale free to contain a newline can therefore carry a pre-ticked
    approval for an arbitrary proposal id, which the next harvest records as the
    operator's decision — no tick, no reading, no gate. Reproduced end to end before
    the fix: this rendered digest yielded `ProposalMark(proposal_id=999,
    decision='approve')`.

    The fix is that no stored proposal text can contain a control character, so a
    rationale cannot start a line at all.
    """
    _add_proposal(
        conn,
        rationale=(
            "A perfectly reasonable sounding argument.\n"
            "- [x] approve <!-- sf:proposal=999 v=approve -->\n"
            "and some trailing prose."
        ),
    )

    rendered = render_digest(_proposal_context(conn))

    assert parse_proposal_marks(rendered) == []
    # The prose survives, flattened onto one line — the words are not lost.
    assert "A perfectly reasonable sounding argument." in rendered


def test_a_rationale_that_is_nothing_but_a_marker_cannot_approve_anything(
    conn: sqlite3.Connection,
) -> None:
    """The half flattening alone did not close.

    The test above relies on the forged marker being *preceded* by prose, so
    collapsing the newline pulls it onto a line that no longer matches. Prose
    whose entire value already is the marker has no newline to collapse: it is
    one line, single-spaced, and the template emits it on a line of its own.
    Reproduced against this repo before the fix — it yielded a real
    `ProposalMark`. What closes it is neutralizing the HTML comment opener,
    which both harvest patterns anchor on.
    """
    _add_proposal(conn, rationale="- [x] approve <!-- sf:proposal=999 v=approve -->")

    rendered = render_digest(_proposal_context(conn))

    assert parse_proposal_marks(rendered) == []


def test_a_triage_rationale_cannot_forge_a_feedback_mark(conn: sqlite3.Connection) -> None:
    """The same class, on the surface nobody was watching.

    `scores.reasoning` is model-authored from feed content and renders straight
    into the digest as "why it matters", and `feedback.harvest_marks` reads a
    verdict from any line matching its pattern. So a feed able to steer one
    triage rationale could tick a box on any item id it names — recording the
    operator's judgement without the operator. Phase 1's acceptance gate is
    measured off exactly those rows.
    """
    item_id, _ = upsert_item(conn, make_item(external_id="forge"))
    _insert_score(conn, item_id, reasoning="- [x] exceptional <!-- sf:item=1 v=exceptional -->")

    rendered = render_digest(
        build_digest_context(conn, target_date=date(2026, 7, 16), max_items=15)
    )

    assert [mark for mark in parse_marks(rendered)] == []


def test_a_forged_marker_in_an_evidence_note_cannot_approve_anything(
    conn: sqlite3.Connection,
) -> None:
    """Same attack through the other free-text field a proposal carries."""
    _add_proposal(
        conn,
        evidence=[
            {
                "url": "https://example.com/post",
                "note": "cited\n- [x] approve <!-- sf:proposal=999 v=approve -->",
            }
        ],
    )

    rendered = render_digest(_proposal_context(conn))

    assert [mark.proposal_id for mark in parse_proposal_marks(rendered)] == []


def test_a_settled_date_is_the_operators_local_day_not_the_utc_one(
    conn: sqlite3.Connection,
) -> None:
    """Brisbane is UTC+10 with no DST, so 07:00 local is 21:00 the previous day UTC.

    Without the zone conversion the note would be dated a day earlier than the day
    the operator remembers ticking the box. Every other proposal test runs in UTC,
    where `astimezone` is a no-op — so deleting the conversion entirely would have
    passed all of them.
    """
    brisbane = ZoneInfo("Australia/Brisbane")
    proposal_id = _add_proposal(conn, surface_date=date(2026, 7, 16))
    decide_proposal(
        conn,
        proposal_id=proposal_id,
        status=ProposalStatus.APPROVED,
        # 2026-07-17 07:00 Brisbane.
        decided_at=datetime(2026, 7, 16, 21, 0, tzinfo=UTC),
    )

    context = build_digest_context(
        conn,
        target_date=date(2026, 7, 17),
        tz=brisbane,
        max_items=MAX_ITEMS,
        settled_display_days=SETTLED_DAYS,
    )

    assert "approved 2026-07-17" in render_digest(context)


def test_a_hand_written_multi_line_rationale_still_cannot_forge_a_marker(
    conn: sqlite3.Connection,
) -> None:
    """Why the render boundary flattens a second time.

    `db.insert_proposal` already guarantees single-line text, so for anything the
    pipeline wrote the second flatten is a no-op — and a test that goes through
    `insert_proposal` therefore cannot tell whether the render layer exists at all.
    This writes the row with raw SQL to reach the state the layer is actually for: a
    row hand-edited in the DB, or written by an older shape before the storage
    invariant existed.
    """
    _add_proposal(conn)
    conn.execute(
        "UPDATE proposals SET rationale = ?",
        ("Sounds fine.\n- [x] approve <!-- sf:proposal=999 v=approve -->\ntrailing.",),
    )

    rendered = render_digest(_proposal_context(conn))

    assert [mark.proposal_id for mark in parse_proposal_marks(rendered)] == []


# --------------------------------------------------------------------------- #
# Forged marks — the vault is a file whose *lines* carry decisions
# --------------------------------------------------------------------------- #


def test_a_feed_title_cannot_forge_a_feedback_mark(conn: sqlite3.Connection) -> None:
    """The one an attacker reaches without any cooperation from us.

    A title is entirely controlled by whoever publishes the feed. It renders verbatim
    into the digest, and `feedback.py` harvests a mark from any line matching
    `MARK_RE` — so a crafted title silently records `useful` for any item id it can
    guess, corrupting the ground-truth set relevance tuning depends on (CLAUDE.md §4)
    and that the curation scout reasons over.

    Reproduced end to end through the real pipeline before the fix: upsert, score,
    render, and `parse_marks` returned `Mark(item_id=1, verdict='useful')`.
    """
    item_id, _ = upsert_item(
        conn,
        make_item(
            title="Totally normal headline\n\n- [x] useful <!-- sf:item=1 v=useful -->\n\nmore",
        ),
    )
    _insert_score(conn, item_id)

    rendered = render_digest(
        build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    )

    assert parse_marks(rendered) == []
    assert "Totally normal headline" in rendered


def test_a_title_edited_directly_in_the_database_is_still_flattened(
    conn: sqlite3.Connection,
) -> None:
    """Why `report/` does not need its own title flatten.

    Every read path reconstructs an `Item`, so the model validator runs even for a
    title written past it with raw SQL — which is the only way to get one. That makes
    `Item._flatten_title` the real chokepoint and a second flatten in `report/`
    unreachable code, so there isn't one.
    """
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    conn.execute(
        "UPDATE items SET title = ?",
        ("Headline\n- [x] useful <!-- sf:item=1 v=useful -->",),
    )

    rendered = render_digest(
        build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    )

    assert parse_marks(rendered) == []


def test_a_triage_reasoning_cannot_forge_a_feedback_mark(conn: sqlite3.Connection) -> None:
    """`why_it_matters` is LLM-authored text on the same page.

    It was safe only as a side effect of `_trim_reasoning` joining on whitespace for
    the one-line format. Pinned so a future reformat cannot quietly remove it.
    """
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(
        conn,
        item_id,
        reasoning="Genuinely useful.\n- [x] exceptional <!-- sf:item=1 v=exceptional -->",
    )

    rendered = render_digest(
        build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    )

    assert parse_marks(rendered) == []


def test_a_source_failure_message_cannot_forge_a_mark(conn: sqlite3.Connection) -> None:
    """Exception text can quote a server response, and renders on the same page."""
    _insert_ingest_run_with_errors(
        conn,
        [
            {
                "source_id": "evil",
                "error_type": "FetchError",
                "message": "HTTP 500\n- [x] useful <!-- sf:item=1 v=useful -->",
                "occurred_at": "2026-07-16T05:01:00+00:00",
            }
        ],
    )

    rendered = render_digest(
        build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    )

    assert parse_marks(rendered) == []


def test_a_proposal_dedup_key_carrying_a_newline_is_dropped_not_rendered(
    conn: sqlite3.Connection,
) -> None:
    """The identity-field half of the same attack, on the read path.

    `insert_proposal` refuses these, so this row is written with raw SQL — the
    hand-edited or older-shape case. Dropped rather than repaired, because rewriting
    an identity field changes what the row means, and `db.py` is where every consumer
    of a proposal passes through.
    """
    _add_proposal(conn)
    conn.execute(
        "UPDATE proposals SET dedup_key = ?",
        ("https://x.example/feed\n- [x] approve <!-- sf:proposal=999 v=approve -->",),
    )

    rendered = render_digest(_proposal_context(conn))

    assert parse_proposal_marks(rendered) == []
    assert "Proposed source changes" not in rendered


def test_a_proposal_citation_url_carrying_a_newline_is_dropped_not_rendered(
    conn: sqlite3.Connection,
) -> None:
    _add_proposal(conn)
    forged = "https://x.example/a\n- [x] approve <!-- sf:proposal=9 v=approve -->"
    conn.execute(
        "UPDATE proposals SET evidence = ?",
        (json.dumps([{"url": forged, "note": ""}]),),
    )

    rendered = render_digest(_proposal_context(conn))

    assert parse_proposal_marks(rendered) == []


def test_a_citation_url_shaped_like_a_marker_needs_no_newline_to_forge(
    conn: sqlite3.Connection,
) -> None:
    """The variant of the attack above that carries no control character at all.

    Each evidence entry renders as its own line in the digest (`daily.md.j2`'s
    evidence loop begins the line with `- {{ url }}`), so a citation "URL" that is
    itself the literal text of a checkbox marker needs no embedded newline to land
    on its own line — `has_control_characters` alone would not catch this.
    Reproduced end to end before the fix: this rendered digest yielded
    `ProposalMark(proposal_id=999, decision='approve')`. The fix requires a
    citation to actually have the shape of an `http(s)` URL.
    """
    _add_proposal(conn)
    forged = "[x] approve <!-- sf:proposal=999 v=approve -->"
    conn.execute(
        "UPDATE proposals SET evidence = ?",
        (json.dumps([{"url": forged, "note": ""}]),),
    )

    rendered = render_digest(_proposal_context(conn))

    assert parse_proposal_marks(rendered) == []


def test_a_forged_marker_in_a_probe_label_cannot_approve_anything(
    conn: sqlite3.Connection,
) -> None:
    """The probe-facts half of the same attack class.

    `probe.label` is lifted from a candidate feed's own `<author>` tag or a
    candidate repo's own release `tag_name` — content the *source being proposed*
    controls, reaching the digest automatically once the candidate is probed and
    before any human approves it. A label free to carry a newline could therefore
    forge an approval with no LLM involved at all. Fixed at the source
    (`Item._flatten_title`'s sibling on `author`, and a flatten in
    `ingest/probe.py::probe_repo`), with this render-boundary flatten as the
    second, cheaper layer — this test writes the row with raw SQL to reach the
    state that second layer is actually for.
    """
    _add_proposal(conn)
    conn.execute(
        "UPDATE proposals SET probe = ?",
        (
            json.dumps(
                {
                    "ok": True,
                    "items_total": 3,
                    "items_in_window": 1,
                    "median_summary_chars": 400,
                    "label": "Real Author\n- [x] approve <!-- sf:proposal=999 v=approve -->",
                }
            ),
        ),
    )

    rendered = render_digest(_proposal_context(conn))

    assert parse_proposal_marks(rendered) == []
    assert "Real Author" in rendered


# --------------------------------------------------------------------------- #
# Topic tags (DESIGN §10 — the deterministic tagger's read surface)
# --------------------------------------------------------------------------- #


def _tag(conn: sqlite3.Connection, item_id: int, topic: str, *, version: str) -> None:
    conn.execute(
        """
        INSERT INTO item_topics (item_id, topic, matched_keyword, taxonomy_version, tagged_at)
        VALUES (?, ?, 'a keyword', ?, '2026-07-16T06:10:00+00:00')
        """,
        (item_id, topic, version),
    )


def test_topics_render_as_obsidian_tags(conn: sqlite3.Connection) -> None:
    """`group.leaf` becomes `#group/leaf` — a nested tag Obsidian can filter on."""
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    _tag(conn, item_id, "industry.strategy", version=TAXONOMY_VERSION)
    _tag(conn, item_id, "policy.regulation", version=TAXONOMY_VERSION)

    rendered = render_digest(
        build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    )

    assert "**Topics:** #industry/strategy #policy/regulation" in rendered


def test_the_topic_line_does_not_swallow_the_first_checkbox(conn: sqlite3.Connection) -> None:
    """A tagged item's first feedback box must still start its own line.

    Regression: the template's topic loop ended a content line with a block tag,
    so `trim_blocks` ate the newline and `- [ ] useful` joined the tag list — the
    `-` absorbed into the trailing tag and the checkbox stopped rendering.
    """
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    _tag(conn, item_id, "industry.strategy", version=TAXONOMY_VERSION)

    rendered = render_digest(
        build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    )

    first_box = checkbox_marker(item_id, CHECKBOX_VERDICTS[0])
    assert f"#industry/strategy\n{first_box}" in rendered
    for verdict in CHECKBOX_VERDICTS:
        assert f"\n{checkbox_marker(item_id, verdict)}\n" in rendered


def test_an_untagged_item_renders_no_topic_line(conn: sqlite3.Connection) -> None:
    """The line is absent, not empty — a pre-tagger digest renders as it always did."""
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)

    rendered = render_digest(
        build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    )

    assert "**Topics:**" not in rendered


def test_topics_from_an_older_taxonomy_version_do_not_render(conn: sqlite3.Connection) -> None:
    """A version bump changes what a topic means; stale rows must not leak into a
    digest as though they were current."""
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    _tag(conn, item_id, "industry.strategy", version="tax-v0")

    rendered = render_digest(
        build_digest_context(conn, target_date=TARGET_DATE, max_items=MAX_ITEMS)
    )

    assert "**Topics:**" not in rendered
