"""Tests for `db.py` — idempotency, the two-UNIQUE upsert, and the merge rules.

The Phase 0 acceptance gate is "a double-run produces zero duplicates"
(DESIGN §16). These tests hold that gate, and hold it at the stronger bar
CLAUDE.md §3 sets: a re-run is a byte-for-byte no-op, not merely duplicate-free.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from signalforge.db import (
    MIGRATIONS,
    SCHEMA_VERSION,
    connect,
    connection,
    finish_run,
    get_feedback,
    get_item,
    get_item_by_canonical_url,
    migrate,
    record_feedback,
    start_run,
    upsert_item,
)
from signalforge.models import SourceType, compute_content_hash
from tests.conftest import FIXED_FETCHED_AT, dump_table, make_item

PHASE0_TABLES = {"items", "scores", "runs", "feedback"}
# Phase 2/3 tables from DESIGN §5. Present in the design, absent from the code
# until their phase gate opens (NEVER rule 15).
DEFERRED_TABLES = {
    "embeddings",
    "clusters",
    "cluster_members",
    "trends",
    "insights",
    "insight_citations",
    "impact_assessments",
}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def _schema_sql(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    rows = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    return [tuple(row) for row in rows]


# --------------------------------------------------------------------------- #
# migrations
# --------------------------------------------------------------------------- #


def test_migrate_creates_phase0_tables(conn: sqlite3.Connection) -> None:
    assert _table_names(conn) >= PHASE0_TABLES


def test_migrate_does_not_create_phase2_or_phase3_tables(conn: sqlite3.Connection) -> None:
    # Phase gate: building these before Phase 0's acceptance gate is met is a
    # regression, not progress (CLAUDE.md §1, NEVER rule 15).
    assert _table_names(conn) & DEFERRED_TABLES == set()


def test_migrate_sets_user_version_to_schema_version(conn: sqlite3.Connection) -> None:
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION


def test_migrate_is_idempotent(conn: sqlite3.Connection) -> None:
    before = _schema_sql(conn)
    assert migrate(conn) == SCHEMA_VERSION
    assert migrate(conn) == SCHEMA_VERSION
    assert _schema_sql(conn) == before


def test_migrate_does_not_destroy_existing_data(conn: sqlite3.Connection) -> None:
    item_id, _ = upsert_item(conn, make_item())
    migrate(conn)
    assert get_item(conn, item_id) is not None


def test_migrate_refuses_a_future_schema_version(conn: sqlite3.Connection) -> None:
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="newer than this code understands"):
        migrate(conn)


def test_connect_enables_wal(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


def test_connection_migrates_and_closes(db_path: Path) -> None:
    with connection(db_path) as conn:
        assert _table_names(conn) >= PHASE0_TABLES
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_connection_creates_the_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "nested" / "signalforge.db"
    with connection(nested) as conn:
        assert _table_names(conn) >= PHASE0_TABLES
    assert nested.is_file()


def test_migrations_are_append_only_and_ordered() -> None:
    versions = [migration.version for migration in MIGRATIONS]
    assert versions == sorted(set(versions))
    assert versions[0] == 1


def test_schema_version_is_two_after_the_feedback_dedup_migration() -> None:
    # Migration 2 (feedback dedup index) is the last one; SCHEMA_VERSION derives
    # from it. If this drops, a fresh DB stops getting the unique index.
    assert SCHEMA_VERSION == 2
    assert MIGRATIONS[-1].version == 2


def test_feedback_dedup_unique_index_exists(conn: sqlite3.Connection) -> None:
    indexes = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    }
    assert "ux_feedback_item_verdict" in indexes


# --------------------------------------------------------------------------- #
# feedback — Phase 1 mark capture (DESIGN §11)
# --------------------------------------------------------------------------- #

FEEDBACK_AT = datetime(2026, 7, 23, 8, 0, 0, tzinfo=UTC)


def test_record_feedback_inserts_a_new_row_and_returns_true(conn: sqlite3.Connection) -> None:
    item_id, _ = upsert_item(conn, make_item())

    recorded = record_feedback(
        conn, item_id=item_id, verdict="useful", note=None, created_at=FEEDBACK_AT
    )

    assert recorded is True
    rows = get_feedback(conn, item_id)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "useful"


def test_record_feedback_is_idempotent_on_the_same_item_and_verdict(
    conn: sqlite3.Connection,
) -> None:
    item_id, _ = upsert_item(conn, make_item())

    first = record_feedback(
        conn, item_id=item_id, verdict="useful", note=None, created_at=FEEDBACK_AT
    )
    second = record_feedback(
        conn, item_id=item_id, verdict="useful", note="second try", created_at=FEEDBACK_AT
    )

    assert first is True
    assert second is False  # ON CONFLICT DO NOTHING — no new row
    assert len(get_feedback(conn, item_id)) == 1


def test_record_feedback_allows_two_distinct_verdicts_on_one_item(
    conn: sqlite3.Connection,
) -> None:
    item_id, _ = upsert_item(conn, make_item())

    # Distinct `created_at`: the migration-1 PRIMARY KEY is (item_id, created_at),
    # so two verdicts for one item need distinct timestamps to coexist. The new
    # unique index (item_id, verdict) is what blocks a *duplicate* verdict.
    later = FEEDBACK_AT.replace(hour=FEEDBACK_AT.hour + 1)
    assert record_feedback(
        conn, item_id=item_id, verdict="useful", note=None, created_at=FEEDBACK_AT
    )
    assert record_feedback(conn, item_id=item_id, verdict="noise", note=None, created_at=later)

    verdicts = {row["verdict"] for row in get_feedback(conn, item_id)}
    assert verdicts == {"useful", "noise"}


def test_record_feedback_stores_the_note(conn: sqlite3.Connection) -> None:
    item_id, _ = upsert_item(conn, make_item())

    record_feedback(
        conn,
        item_id=item_id,
        verdict="missed",
        note="should have been surfaced",
        created_at=FEEDBACK_AT,
    )

    assert get_feedback(conn, item_id)[0]["note"] == "should have been surfaced"


def test_record_feedback_for_a_nonexistent_item_raises(conn: sqlite3.Connection) -> None:
    # The FK to items(id) is enforced (PRAGMA foreign_keys = ON), so writing a
    # mark for an unknown id raises rather than storing an orphan — which is why
    # the harvester and the CLI both pre-check the item exists (CLAUDE.md §7).
    with pytest.raises(sqlite3.IntegrityError):
        record_feedback(conn, item_id=9999, verdict="useful", note=None, created_at=FEEDBACK_AT)


def test_items_table_has_both_unique_constraints(conn: sqlite3.Connection) -> None:
    # Both constraints are load-bearing; the upsert exists to span them.
    unique_cols = {
        tuple(
            info["name"] for info in conn.execute(f"PRAGMA index_info({row['name']!r})").fetchall()
        )
        for row in conn.execute("PRAGMA index_list('items')").fetchall()
        if row["unique"]
    }
    assert ("canonical_url",) in unique_cols
    assert ("source_id", "external_id") in unique_cols


# --------------------------------------------------------------------------- #
# Idempotency — the Phase 0 acceptance gate
# --------------------------------------------------------------------------- #


def test_double_ingest_of_the_same_item_yields_one_row(conn: sqlite3.Connection) -> None:
    first_id, first_new = upsert_item(conn, make_item())
    second_id, second_new = upsert_item(conn, make_item())

    assert first_new is True
    assert second_new is False, "is_new must be True only on the call that created the row"
    assert first_id == second_id
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1


def test_double_ingest_leaves_db_state_byte_for_byte_identical(conn: sqlite3.Connection) -> None:
    # The strong form of the gate (CLAUDE.md §3): a re-run is a true no-op, not
    # merely duplicate-free. "No duplicates" would still permit churning
    # fetched_at or content_hash on every cron tick.
    upsert_item(conn, make_item())
    after_first = dump_table(conn, "items")

    # Tomorrow's cron re-fetches the same unchanged feed entry: identical payload,
    # later fetched_at. Passing a byte-identical Item would make this assertion
    # vacuous — the clock is the one field a re-run really does carry anew.
    upsert_item(conn, make_item(fetched_at=datetime(2026, 7, 17, 6, 0, 0, tzinfo=UTC)))
    after_second = dump_table(conn, "items")

    assert after_second == after_first


def test_reingest_preserves_fetched_at_as_first_seen(conn: sqlite3.Connection) -> None:
    # first-seen, not last-seen: this is what makes the re-run a no-op, and it is
    # the honest answer to "when did this enter our world?"
    item_id, _ = upsert_item(conn, make_item())
    later = make_item(fetched_at=datetime(2026, 8, 1, 6, 0, 0, tzinfo=UTC))
    upsert_item(conn, later)

    stored = get_item(conn, item_id)
    assert stored is not None
    assert stored.fetched_at == FIXED_FETCHED_AT


def test_ingesting_a_whole_batch_twice_is_a_no_op(conn: sqlite3.Connection) -> None:
    batch = [
        make_item(external_id=f"guid-{n}", url=f"https://example.com/post-{n}") for n in range(5)
    ]
    for item in batch:
        upsert_item(conn, item)
    after_first = dump_table(conn, "items")

    new_count = sum(is_new for _, is_new in (upsert_item(conn, item) for item in batch))

    assert new_count == 0, "a second run must report zero new items — no double-spend downstream"
    assert dump_table(conn, "items") == after_first


# --------------------------------------------------------------------------- #
# The two-UNIQUE upsert — every collision path
# --------------------------------------------------------------------------- #


def test_collision_on_neither_key_inserts_a_fresh_row(conn: sqlite3.Connection) -> None:
    first_id, first_new = upsert_item(conn, make_item())
    second_id, second_new = upsert_item(
        conn,
        make_item(external_id="guid-2", url="https://example.com/other"),
    )

    assert (first_new, second_new) == (True, True)
    assert first_id != second_id
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2


def test_collision_on_canonical_url_only_updates_the_existing_row(
    conn: sqlite3.Connection,
) -> None:
    # The same post republished under a new guid — one document, one row.
    first_id, _ = upsert_item(conn, make_item(external_id="guid-1"))
    second_id, is_new = upsert_item(conn, make_item(external_id="guid-RENAMED", title="Retitled"))

    assert (second_id, is_new) == (first_id, False)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
    stored = get_item(conn, first_id)
    assert stored is not None
    assert stored.title == "Retitled"
    assert stored.external_id == "guid-1", "external_id is write-once"


def test_collision_on_source_and_external_id_only_updates_the_existing_row(
    conn: sqlite3.Connection,
) -> None:
    # The same guid now pointing at a different URL — the publisher moved it.
    first_id, _ = upsert_item(conn, make_item(url="https://example.com/old"))
    second_id, is_new = upsert_item(conn, make_item(url="https://example.com/new"))

    assert (second_id, is_new) == (first_id, False)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
    stored = get_item(conn, first_id)
    assert stored is not None
    assert stored.url == "https://example.com/old", "url is write-once"
    assert stored.canonical_url == "https://example.com/old"


def test_collision_on_both_keys_updates_the_single_matching_row(
    conn: sqlite3.Connection,
) -> None:
    first_id, _ = upsert_item(conn, make_item())
    second_id, is_new = upsert_item(conn, make_item(title="Updated title"))

    assert (second_id, is_new) == (first_id, False)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1


def test_same_external_id_from_a_different_source_does_not_collide(
    conn: sqlite3.Connection,
) -> None:
    # The UNIQUE is (source_id, external_id): guid "1" from two feeds is two items.
    first_id, _ = upsert_item(conn, make_item(source_id="feed-a", external_id="1"))
    second_id, is_new = upsert_item(
        conn,
        make_item(source_id="feed-b", external_id="1", url="https://example.com/b"),
    )

    assert is_new is True
    assert first_id != second_id


def test_null_external_ids_do_not_collide_with_each_other(conn: sqlite3.Connection) -> None:
    # SQLite treats NULLs as distinct in a UNIQUE index, so two guid-less items
    # from one feed are two rows — they can only dedup on canonical_url.
    first_id, first_new = upsert_item(
        conn, make_item(external_id=None, url="https://example.com/a")
    )
    second_id, second_new = upsert_item(
        conn, make_item(external_id=None, url="https://example.com/b")
    )

    assert (first_new, second_new) == (True, True)
    assert first_id != second_id
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2


def test_null_external_id_items_still_dedup_on_canonical_url(conn: sqlite3.Connection) -> None:
    first_id, _ = upsert_item(conn, make_item(external_id=None, url="https://example.com/a"))
    second_id, is_new = upsert_item(
        conn, make_item(external_id=None, url="https://www.example.com/a/?utm_source=hn")
    )

    assert (second_id, is_new) == (first_id, False)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1


def test_disagreeing_keys_resolve_to_the_canonical_url_row_and_merge_nothing(
    conn: sqlite3.Connection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Row A holds the URL, row B holds the guid. The incoming item bridges them.
    # canonical_url wins (it is the document's cross-source identity), and
    # NOTHING is merged or deleted — silently destroying a row that scores or
    # feedback reference is far worse than leaving two.
    row_a_id, _ = upsert_item(
        conn, make_item(external_id="guid-A", url="https://example.com/shared")
    )
    row_b_id, _ = upsert_item(
        conn, make_item(external_id="guid-B", url="https://example.com/other")
    )

    with caplog.at_level("WARNING"):
        resolved_id, is_new = upsert_item(
            conn,
            make_item(external_id="guid-B", url="https://example.com/shared", title="Bridging"),
        )

    assert is_new is False
    assert resolved_id == row_a_id, "canonical_url is the winner"
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2, "no row was deleted"
    assert "two different existing rows" in caplog.text

    row_a = get_item(conn, row_a_id)
    row_b = get_item(conn, row_b_id)
    assert row_a is not None and row_b is not None
    assert row_a.title == "Bridging", "the canonical_url match absorbed the update"
    assert row_a.external_id == "guid-A", "identity untouched — no merge of B's guid"
    assert row_b.title == "MCP sampling lands everywhere", "row B is untouched"


# --------------------------------------------------------------------------- #
# Merge rules — these protect money and correctness
# --------------------------------------------------------------------------- #


def test_reingest_with_null_content_never_wipes_a_paid_for_deep_read(
    conn: sqlite3.Connection,
) -> None:
    # THE money test (CLAUDE.md §6): a top-N deep read paid an LLM call to fetch
    # this full text. The next daily re-ingest carries feed data only
    # (content=None) and must not clear it, or the next run pays again.
    item_id, _ = upsert_item(conn, make_item())
    upsert_item(conn, make_item(content="The full article text, expensively fetched."))

    upsert_item(conn, make_item(content=None))

    stored = get_item(conn, item_id)
    assert stored is not None
    assert stored.content == "The full article text, expensively fetched."


@pytest.mark.parametrize("field", ["author", "summary", "published_at", "raw_path", "content"])
def test_reingest_with_a_null_payload_field_never_wipes_stored_data(
    conn: sqlite3.Connection, field: str
) -> None:
    # Freshest NON-NULL: a re-ingest carries only feed-level data, so a missing
    # field means "I don't know", never "delete what you have".
    rich = make_item(
        author="Simon Willison",
        summary="A short feed summary.",
        content="Full text.",
        raw_path="data/http_cache/abc.json",
    )
    item_id, _ = upsert_item(conn, rich)
    upsert_item(conn, make_item(**{field: None}))

    stored = get_item(conn, item_id)
    assert stored is not None
    assert getattr(stored, field) == getattr(rich, field)


def test_reingest_with_a_fresher_non_null_value_wins(conn: sqlite3.Connection) -> None:
    item_id, _ = upsert_item(conn, make_item(author="Old Author", summary="Old summary."))
    upsert_item(conn, make_item(author="New Author", summary="New summary."))

    stored = get_item(conn, item_id)
    assert stored is not None
    assert stored.author == "New Author"
    assert stored.summary == "New summary."


def test_content_hash_always_agrees_with_the_stored_title_and_summary(
    conn: sqlite3.Connection,
) -> None:
    # A stale hash silently poisons exact dedup — the hash must be recomputed
    # from the MERGED text, not carried over from either side.
    item_id, _ = upsert_item(conn, make_item(title="Original", summary="Original summary."))
    upsert_item(conn, make_item(title="Corrected", summary=None))

    stored = get_item(conn, item_id)
    assert stored is not None
    assert stored.title == "Corrected"
    assert stored.summary == "Original summary.", "NULL summary did not wipe"
    assert stored.content_hash == compute_content_hash("Corrected", "Original summary.")


def test_content_hash_is_recomputed_even_when_the_incoming_item_carries_a_stale_one(
    conn: sqlite3.Connection,
) -> None:
    item_id, _ = upsert_item(conn, make_item(title="Original"))
    upsert_item(conn, make_item(title="Retitled", content_hash="0" * 64))

    stored = get_item(conn, item_id)
    assert stored is not None
    assert stored.content_hash != "0" * 64
    assert stored.content_hash == compute_content_hash("Retitled", "A short feed summary.")


def test_identity_columns_are_never_rewritten_by_an_update(conn: sqlite3.Connection) -> None:
    original = make_item(
        source_id="feed-a",
        external_id="guid-1",
        url="https://example.com/post?utm_source=rss",
    )
    item_id, _ = upsert_item(conn, original)

    # Same canonical_url, but every identity field differs.
    upsert_item(
        conn,
        make_item(
            source_id="feed-a",
            external_id="guid-CHANGED",
            url="https://www.example.com/post/#frag",
            title="Changed",
        ),
    )

    stored = get_item(conn, item_id)
    assert stored is not None
    assert stored.source_id == "feed-a"
    assert stored.external_id == "guid-1"
    assert stored.url == original.url
    assert stored.canonical_url == original.canonical_url
    assert stored.title == "Changed", "payload still merged"


def test_url_and_canonical_url_stay_consistent_after_an_update(
    conn: sqlite3.Connection,
) -> None:
    # They move together or not at all: a canonical_url that is not
    # canonicalize_url(url) would make citations point somewhere we never fetched.
    from signalforge.models import canonicalize_url

    item_id, _ = upsert_item(conn, make_item(url="https://example.com/post?utm_source=rss"))
    upsert_item(conn, make_item(url="https://example.com/post?utm_source=rss", title="V2"))

    stored = get_item(conn, item_id)
    assert stored is not None
    assert stored.canonical_url == canonicalize_url(stored.url)


def test_source_type_is_pinned_to_its_first_seen_value(conn: sqlite3.Connection) -> None:
    # source_type is identity, not payload. "Not part of a UNIQUE key" is not the
    # test for identity: the UNIQUE keys exist to make dedup *findable*, whereas
    # identity is about which fields describe where a row came from. source_type is
    # functionally determined by source_id — it names the adapter that owns that key
    # in sources.yaml — so letting it follow the last writer while source_id stays
    # pinned would make the pair incoherent. It moves with source_id or not at all.
    item_id, _ = upsert_item(conn, make_item(source_type=SourceType.RSS))
    upsert_item(conn, make_item(source_type=SourceType.NEWSLETTER))

    stored = get_item(conn, item_id)
    assert stored is not None
    assert stored.source_type is SourceType.RSS


def test_a_post_reaching_us_from_two_sources_keeps_a_coherent_source_pair(
    conn: sqlite3.Connection,
) -> None:
    # Cross-source dedup is designed behaviour, not an edge case: the same post from
    # an RSS feed and from HN lands on one canonical_url, so the second upsert is an
    # update by a *different* adapter. It fires on anything that trends on HN — the
    # highest-signal subset — and scoring weights by source, so an incoherent pair
    # would silently misweight exactly the items that matter most.
    url = "https://simonwillison.net/2026/Jul/15/mcp-sampling/"
    item_id, was_new = upsert_item(
        conn, make_item(source_id="simonwillison", source_type=SourceType.RSS, url=url)
    )
    assert was_new

    same_id, was_new_again = upsert_item(
        conn, make_item(source_id="hn", source_type=SourceType.HN, external_id="4242", url=url)
    )
    assert (same_id, was_new_again) == (item_id, False)

    stored = get_item(conn, item_id)
    assert stored is not None
    # First writer wins the attribution, and both halves of the pair agree.
    assert (stored.source_id, stored.source_type) == ("simonwillison", SourceType.RSS)


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #


def test_get_item_round_trips_every_field(conn: sqlite3.Connection) -> None:
    original = make_item(content="Full text.", raw_path="data/http_cache/abc.json", lang="fr")
    item_id, _ = upsert_item(conn, original)

    stored = get_item(conn, item_id)
    assert stored is not None
    assert stored.model_dump(exclude={"id"}) == original.model_dump(exclude={"id"})
    assert stored.id == item_id


def test_get_item_returns_none_for_a_missing_id(conn: sqlite3.Connection) -> None:
    assert get_item(conn, 9999) is None


def test_get_item_by_canonical_url_finds_the_row(conn: sqlite3.Connection) -> None:
    item = make_item()
    item_id, _ = upsert_item(conn, item)

    found = get_item_by_canonical_url(conn, item.canonical_url)
    assert found is not None
    assert found.id == item_id


def test_get_item_by_canonical_url_returns_none_when_absent(conn: sqlite3.Connection) -> None:
    assert get_item_by_canonical_url(conn, "https://example.com/nope") is None


def test_stored_datetimes_are_iso_8601_text(conn: sqlite3.Connection) -> None:
    # DESIGN §5 stores datetimes as ISO 8601 TEXT so they sort lexicographically.
    upsert_item(conn, make_item())
    row = conn.execute("SELECT fetched_at, published_at FROM items").fetchone()
    assert datetime.fromisoformat(row["fetched_at"]) == FIXED_FETCHED_AT
    assert row["fetched_at"] == FIXED_FETCHED_AT.isoformat()


# --------------------------------------------------------------------------- #
# runs — no silent runs (CLAUDE.md §3)
# --------------------------------------------------------------------------- #


def test_start_run_writes_a_row(conn: sqlite3.Connection) -> None:
    started = datetime(2026, 7, 16, 6, 0, tzinfo=UTC)
    run_id = start_run(conn, "ingest", started_at=started)

    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["kind"] == "ingest"
    assert row["started_at"] == started.isoformat()
    assert row["finished_at"] is None
    assert row["status"] is None


def test_start_run_defaults_counters_to_zero(conn: sqlite3.Connection) -> None:
    run_id = start_run(conn, "ingest", started_at=datetime(2026, 7, 16, 6, 0, tzinfo=UTC))
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert (row["items_new"], row["llm_input_tokens"], row["llm_output_tokens"]) == (0, 0, 0)


def test_finish_run_records_status_counts_and_tokens(conn: sqlite3.Connection) -> None:
    started = datetime(2026, 7, 16, 6, 0, tzinfo=UTC)
    finished = datetime(2026, 7, 16, 6, 5, tzinfo=UTC)
    run_id = start_run(conn, "ingest", started_at=started)

    finish_run(
        conn,
        run_id,
        status="ok",
        finished_at=finished,
        items_new=12,
        llm_input_tokens=3400,
        llm_output_tokens=210,
    )

    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["status"] == "ok"
    assert row["finished_at"] == finished.isoformat()
    assert row["items_new"] == 12
    assert (row["llm_input_tokens"], row["llm_output_tokens"]) == (3400, 210)
    assert row["errors"] is None


def test_finish_run_serializes_per_source_errors_as_json(conn: sqlite3.Connection) -> None:
    # One broken source never aborts a run, but its failure must never vanish
    # either — the reports are the monitoring channel (CLAUDE.md §7).
    run_id = start_run(conn, "ingest", started_at=datetime(2026, 7, 16, 6, 0, tzinfo=UTC))
    errors = [
        {"source_id": "interconnects", "error": "HTTP 503"},
        {"source_id": "arxiv", "error": "timeout after 20s"},
    ]

    finish_run(
        conn,
        run_id,
        status="partial",
        finished_at=datetime(2026, 7, 16, 6, 5, tzinfo=UTC),
        errors=errors,
    )

    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["status"] == "partial"
    assert json.loads(row["errors"]) == errors


def test_finish_run_can_record_a_failed_run(conn: sqlite3.Connection) -> None:
    run_id = start_run(conn, "ingest", started_at=datetime(2026, 7, 16, 6, 0, tzinfo=UTC))
    finish_run(
        conn,
        run_id,
        status="failed",
        finished_at=datetime(2026, 7, 16, 6, 1, tzinfo=UTC),
        errors=[{"error": "config missing"}],
    )
    row = conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["status"] == "failed"


def test_each_start_run_gets_a_distinct_id(conn: sqlite3.Connection) -> None:
    started = datetime(2026, 7, 16, 6, 0, tzinfo=UTC)
    ids = [start_run(conn, kind, started_at=started) for kind in ("ingest", "score", "daily")]
    assert len(set(ids)) == 3
