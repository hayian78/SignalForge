"""Tests for Phase 1 feedback capture (`signalforge/feedback.py`, DESIGN §11).

Scope: the pure `parse_marks` parser, the template↔parser round trip (proving
the rendered checkbox marker and the parser agree), and `harvest_marks` — its
DB writes, its skip-and-log of unknown ids, and its run-twice idempotency.

Every DB is a throwaway `conn` from `tests/conftest.py`; the vault is always a
`tmp_path` directory. `harvest_marks` only ever *reads* vault files (NEVER
rule 8), so these tests also assert the marked file is left byte-for-byte
unchanged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from signalforge.db import get_feedback, upsert_item
from signalforge.feedback import (
    HarvestResult,
    Mark,
    checkbox_marker,
    harvest_marks,
    parse_marks,
)
from signalforge.report.daily import build_digest_context, render_digest
from tests.conftest import make_item

TARGET_DATE_STR = "2026-07-16"
SCORED_AT = "2026-07-16T06:05:00+00:00"


def _insert_score(conn: sqlite3.Connection, item_id: int, *, triage: str = "keep") -> None:
    conn.execute(
        """
        INSERT INTO scores (
            item_id, triage, signal, relevance, novelty, reasoning,
            rubric_version, model, scored_at
        ) VALUES (?, ?, 4, 4, 3, 'A reason this item matters.', 'v1',
                  'claude-haiku-4-5', ?)
        """,
        (item_id, triage, SCORED_AT),
    )


# --------------------------------------------------------------------------- #
# parse_marks — pure
# --------------------------------------------------------------------------- #


def test_parse_marks_recovers_a_checked_useful_box() -> None:
    text = "- [x] useful <!-- sf:item=42 v=useful -->"
    assert parse_marks(text) == [Mark(item_id=42, verdict="useful")]


def test_parse_marks_ignores_an_unchecked_box() -> None:
    text = "- [ ] useful <!-- sf:item=42 v=useful -->"
    assert parse_marks(text) == []


def test_parse_marks_recovers_both_boxes_when_both_are_checked() -> None:
    text = "- [x] useful <!-- sf:item=7 v=useful -->\n- [x] noise <!-- sf:item=7 v=noise -->\n"
    assert parse_marks(text) == [
        Mark(item_id=7, verdict="useful"),
        Mark(item_id=7, verdict="noise"),
    ]


def test_parse_marks_accepts_capital_x() -> None:
    text = "- [X] noise <!-- sf:item=9 v=noise -->"
    assert parse_marks(text) == [Mark(item_id=9, verdict="noise")]


def test_parse_marks_ignores_non_mark_and_malformed_lines() -> None:
    text = (
        "# A heading\n"
        "Some prose mentioning sf:item=1 in passing.\n"
        "- [x] useful\n"  # no marker comment
        "- [x] useful <!-- sf:item=abc v=useful -->\n"  # non-numeric id
        "- [x] useful <!-- sf:item=3 v=noise -->\n"  # label/verdict disagree
        "**Link:** https://example.com/x\n"
    )
    assert parse_marks(text) == []


def test_parse_marks_follows_document_order() -> None:
    text = "- [x] noise <!-- sf:item=2 v=noise -->\n- [x] useful <!-- sf:item=1 v=useful -->\n"
    assert parse_marks(text) == [
        Mark(item_id=2, verdict="noise"),
        Mark(item_id=1, verdict="useful"),
    ]


# --------------------------------------------------------------------------- #
# Round trip — the rendered marker and the parser must agree
# --------------------------------------------------------------------------- #


def test_checkbox_marker_round_trips_through_parse_marks() -> None:
    line = checkbox_marker(42, "useful")
    # Flip the rendered (always-empty) box to checked, as a reader would.
    checked = line.replace("- [ ]", "- [x]", 1)
    assert parse_marks(checked) == [Mark(item_id=42, verdict="useful")]


def test_rendered_digest_marker_round_trips(conn: sqlite3.Connection) -> None:
    """Render a real digest, check a box in the text, and prove `parse_marks`
    recovers the exact (id, verdict) — the template marker and the parser agree."""
    from datetime import date

    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)

    rendered = render_digest(
        build_digest_context(conn, target_date=date(2026, 7, 16), max_items=15)
    )
    # The rendered digest carries only empty boxes (idempotent re-render).
    assert parse_marks(rendered) == []

    # A reader ticks the "useful" box for this item.
    empty = checkbox_marker(item_id, "useful")
    checked = empty.replace("- [ ]", "- [x]", 1)
    assert empty in rendered
    marked_text = rendered.replace(empty, checked, 1)

    assert parse_marks(marked_text) == [Mark(item_id=item_id, verdict="useful")]


# --------------------------------------------------------------------------- #
# harvest_marks — DB writes, unknown-id skip, idempotency, vault-read-only
# --------------------------------------------------------------------------- #


def _write_daily(vault_dir: Path, name: str, text: str) -> Path:
    daily = vault_dir / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    path = daily / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_harvest_marks_records_the_checked_marks(conn: sqlite3.Connection, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    _write_daily(
        vault,
        TARGET_DATE_STR,
        f"- [x] useful <!-- sf:item={item_id} v=useful -->\n"
        f"- [ ] noise <!-- sf:item={item_id} v=noise -->\n",
    )

    result = harvest_marks(conn, vault)

    assert result.files_scanned == 1
    assert result.marks_found == 1
    assert result.rows_recorded == 1
    rows = get_feedback(conn, item_id)
    assert [row["verdict"] for row in rows] == ["useful"]


def test_harvest_marks_skips_an_unknown_item_id_without_raising(
    conn: sqlite3.Connection, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    vault = tmp_path / "vault"
    # No item 9999 exists — a hand-edited or stale id must never abort the run.
    _write_daily(vault, TARGET_DATE_STR, "- [x] useful <!-- sf:item=9999 v=useful -->\n")

    with caplog.at_level("WARNING"):
        result = harvest_marks(conn, vault)

    assert result.marks_found == 1
    assert result.rows_recorded == 0
    assert "unknown item" in caplog.text.lower()


def test_harvest_marks_over_an_empty_vault_is_a_no_op(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    vault = tmp_path / "vault"
    (vault / "daily").mkdir(parents=True)

    result = harvest_marks(conn, vault)

    assert result == HarvestResult(files_scanned=0, marks_found=0, rows_recorded=0)


def test_harvest_marks_run_twice_records_zero_new_rows_the_second_time(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Harvest-then-overwrite must be safe to run before every render: the same
    checked box harvested twice records ONE row (CLAUDE.md §3, DESIGN §11)."""
    vault = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    path = _write_daily(
        vault, TARGET_DATE_STR, f"- [x] useful <!-- sf:item={item_id} v=useful -->\n"
    )
    original_bytes = path.read_bytes()

    first = harvest_marks(conn, vault)
    second = harvest_marks(conn, vault)

    assert first.rows_recorded == 1
    assert second.marks_found == 1  # still finds it in the file
    assert second.rows_recorded == 0  # but records nothing new
    assert len(get_feedback(conn, item_id)) == 1
    # Vault file was only read, never rewritten (NEVER rule 8).
    assert path.read_bytes() == original_bytes


def test_harvest_marks_records_both_verdicts_on_one_item(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Both boxes ticked on one item: the harvest must record BOTH rows (matching
    the CLI path) without raising, despite the migration-1 PK (item_id,
    created_at) — distinct per-mark timestamps keep them from colliding."""
    vault = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    _write_daily(
        vault,
        TARGET_DATE_STR,
        f"- [x] useful <!-- sf:item={item_id} v=useful -->\n"
        f"- [x] noise <!-- sf:item={item_id} v=noise -->\n",
    )

    first = harvest_marks(conn, vault)

    assert first.marks_found == 2
    assert first.rows_recorded == 2
    assert {row["verdict"] for row in get_feedback(conn, item_id)} == {"useful", "noise"}

    # Idempotency still holds with distinct timestamps: the UNIQUE(item_id,
    # verdict) index makes a re-harvest a no-op.
    second = harvest_marks(conn, vault)
    assert second.rows_recorded == 0
    assert len(get_feedback(conn, item_id)) == 2
