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
from datetime import date
from pathlib import Path

import pytest

from signalforge.db import get_feedback, upsert_item
from signalforge.feedback import (
    CHECKBOX_VERDICTS,
    HARVEST_DIRS,
    LADDER,
    VERDICTS,
    HarvestResult,
    Mark,
    checkbox_marker,
    harvest_marks,
    highest_rung,
    parse_marks,
)
from signalforge.report.daily import build_digest_context, digest_path, render_digest
from signalforge.report.podcast import script_path
from signalforge.report.weekly import brief_path
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


@pytest.mark.parametrize("verdict", CHECKBOX_VERDICTS)
def test_every_checkbox_verdict_round_trips(verdict: str) -> None:
    """Whatever the vocabulary offers as a box, the parser must recover — so a
    new rung can never render without being harvestable."""
    checked = checkbox_marker(42, verdict).replace("- [ ]", "- [x]", 1)
    assert parse_marks(checked) == [Mark(item_id=42, verdict=verdict)]


def test_ladder_is_ordinal_and_excludes_missed() -> None:
    """The rungs are ranked weakest-first and `missed` sits off the ladder.

    Guards the Phase 1 gate trap (DESIGN §16): aggregations must reduce an item
    to its highest rung, which is only definable if this order is stable.
    """
    assert LADDER == ("noise", "useful", "exceptional")
    assert "missed" not in LADDER
    assert set(LADDER) == set(CHECKBOX_VERDICTS)
    assert set(CHECKBOX_VERDICTS) < set(VERDICTS)


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


@pytest.mark.parametrize(
    ("first_verdict", "second_verdict"),
    [
        ("useful", "noise"),
        # Two rungs of the same ladder: the case an aggregation must collapse to
        # the highest rather than counting twice (DESIGN §11).
        ("useful", "exceptional"),
    ],
)
def test_harvest_marks_records_both_verdicts_on_one_item(
    conn: sqlite3.Connection, tmp_path: Path, first_verdict: str, second_verdict: str
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
        f"- [x] {first_verdict} <!-- sf:item={item_id} v={first_verdict} -->\n"
        f"- [x] {second_verdict} <!-- sf:item={item_id} v={second_verdict} -->\n",
    )

    first = harvest_marks(conn, vault)

    assert first.marks_found == 2
    assert first.rows_recorded == 2
    verdicts = {row["verdict"] for row in get_feedback(conn, item_id)}
    assert verdicts == {first_verdict, second_verdict}
    # Two rungs coexist as separate rows, so an item's rating is the highest of
    # them — the reduction every aggregation owes the Phase 1 gate (DESIGN §16).
    rungs = [v for v in verdicts if v in LADDER]
    assert max(rungs, key=LADDER.index) == max((first_verdict, second_verdict), key=LADDER.index)

    # Idempotency still holds with distinct timestamps: the UNIQUE(item_id,
    # verdict) index makes a re-harvest a no-op.
    second = harvest_marks(conn, vault)
    assert second.rows_recorded == 0
    assert len(get_feedback(conn, item_id)) == 2


# --------------------------------------------------------------------------- #
# highest_rung — the one reduction every aggregation goes through
# --------------------------------------------------------------------------- #


def test_highest_rung_of_no_marks_is_none() -> None:
    assert highest_rung(()) is None


@pytest.mark.parametrize("verdict", LADDER)
def test_highest_rung_of_a_single_rung_is_that_rung(verdict: str) -> None:
    assert highest_rung([verdict]) == verdict


def test_highest_rung_picks_the_strongest_not_the_first() -> None:
    """Order of the input must not matter — `UNIQUE(item_id, verdict)` stores
    each mark separately and says nothing about which was ticked first."""
    assert highest_rung(["noise", "exceptional", "useful"]) == "exceptional"
    assert highest_rung(["exceptional", "noise"]) == "exceptional"


def test_highest_rung_ignores_off_ladder_verdicts() -> None:
    """`missed` is in `VERDICTS` but absent from `LADDER`; the naive
    `max(..., key=LADDER.index)` would raise `ValueError` here."""
    assert highest_rung(["missed"]) is None
    assert highest_rung(["missed", "useful"]) == "useful"


def test_highest_rung_covers_every_ladder_rung() -> None:
    """Guards the reduction against a rung being added to `LADDER` without a
    thought for how it aggregates — a new rung lands here first."""
    assert highest_rung(LADDER) == LADDER[-1]
    assert set(LADDER) <= set(VERDICTS)


# --------------------------------------------------------------------------- #
# the weekly brief's marks (DESIGN §11 — the acceptance gate's sensor)
# --------------------------------------------------------------------------- #


def _write_weekly(vault_dir: Path, name: str, text: str) -> Path:
    weekly = vault_dir / "weekly"
    weekly.mkdir(parents=True, exist_ok=True)
    path = weekly / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_harvest_reads_the_weekly_directory_too(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Phase 1's gate is "≥80% of brief items rated useful or better". Those marks
    are ticked on the brief, so without this the gate has no sensor at all."""
    vault = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    _write_weekly(vault, "2026-08-09", f"- [x] useful <!-- sf:item={item_id} v=useful -->\n")

    result = harvest_marks(conn, vault)

    assert result == HarvestResult(files_scanned=1, marks_found=1, rows_recorded=1)


def test_harvest_scans_both_directories_in_one_pass(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """One pass, not two commands: `digest` and `podcast` both call this, and
    either may be the first to see a brief the operator ticked."""
    vault = tmp_path / "vault"
    first, _ = upsert_item(conn, make_item(external_id="a", url="https://example.com/a"))
    second, _ = upsert_item(conn, make_item(external_id="b", url="https://example.com/b"))
    _insert_score(conn, first)
    _insert_score(conn, second)
    _write_daily(vault, "2026-08-08", f"- [x] useful <!-- sf:item={first} v=useful -->\n")
    _write_weekly(vault, "2026-08-09", f"- [x] noise <!-- sf:item={second} v=noise -->\n")

    result = harvest_marks(conn, vault)

    assert result == HarvestResult(files_scanned=2, marks_found=2, rows_recorded=2)
    stored = {
        (row[0], row[1]) for row in conn.execute("SELECT item_id, verdict FROM feedback").fetchall()
    }
    assert stored == {(first, "useful"), (second, "noise")}


def test_the_same_mark_in_daily_and_weekly_records_exactly_one_row(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """The property the whole change rests on. An item carried by both the digest
    and the brief is the normal case, not the exception, and ticking it in each
    place must not double-count it toward the gate."""
    vault = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    marker = f"- [x] useful <!-- sf:item={item_id} v=useful -->\n"
    _write_daily(vault, "2026-08-08", marker)
    _write_weekly(vault, "2026-08-09", marker)

    result = harvest_marks(conn, vault)

    assert result.marks_found == 2
    assert result.rows_recorded == 1
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE item_id = ? AND verdict = 'useful'",
            (item_id,),
        ).fetchone()[0]
        == 1
    )


def test_harvest_is_a_no_op_when_the_weekly_directory_does_not_exist(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """`Path.glob` on a missing directory yields nothing rather than raising, so a
    vault predating the brief keeps working. Asserted explicitly so a future
    refactor to `os.scandir` cannot quietly break it."""
    vault = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    _write_daily(vault, "2026-08-08", f"- [x] useful <!-- sf:item={item_id} v=useful -->\n")
    assert not (vault / "weekly").exists()

    result = harvest_marks(conn, vault)

    assert result == HarvestResult(files_scanned=1, marks_found=1, rows_recorded=1)


def test_the_podcast_directory_is_never_harvested(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """The episode template renders no checkbox, so `podcast/` is deliberately
    outside `HARVEST_DIRS`. Pinned because "scan every subdirectory" is the
    tempting simplification, and it would make the episode an input surface."""
    vault = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    podcast = vault / "podcast"
    podcast.mkdir(parents=True)
    (podcast / "2026-08-09.md").write_text(
        f"- [x] useful <!-- sf:item={item_id} v=useful -->\n", encoding="utf-8"
    )

    assert harvest_marks(conn, vault) == HarvestResult(
        files_scanned=0, marks_found=0, rows_recorded=0
    )


def test_the_same_item_marked_differently_in_each_file_records_both(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """The ladder-upgrade path, and the one case a shared timestamp would lose.

    The migration-1 primary key is `(item_id, created_at)`, so two verdicts on one
    item collide unless each mark gets its own stamp. `harvest_marks` offsets by a
    single counter that runs across *all* directories; a per-directory counter
    would look identical in `files_scanned`/`marks_found` and silently drop the
    second row into the `except` branch.

    This is the expected path, not an edge case: an item marked `useful` on
    Tuesday's digest and promoted to `exceptional` on Sunday's brief is exactly
    what "an item's rating is its highest rung" is for, and exactly what Phase 1's
    "useful **or better**" gate reads.
    """
    vault = tmp_path / "vault"
    item_id, _ = upsert_item(conn, make_item())
    _insert_score(conn, item_id)
    _write_daily(vault, "2026-08-08", f"- [x] useful <!-- sf:item={item_id} v=useful -->\n")
    _write_weekly(
        vault, "2026-08-09", f"- [x] exceptional <!-- sf:item={item_id} v=exceptional -->\n"
    )

    result = harvest_marks(conn, vault)

    assert result == HarvestResult(files_scanned=2, marks_found=2, rows_recorded=2)
    rows = conn.execute(
        "SELECT verdict, created_at FROM feedback WHERE item_id = ? ORDER BY created_at",
        (item_id,),
    ).fetchall()
    assert [row[0] for row in rows] == ["useful", "exceptional"]
    assert rows[0][1] != rows[1][1], "each mark needs its own stamp or the PK collides"
    assert highest_rung([row[0] for row in rows]) == "exceptional"


def test_every_directory_harvested_is_one_a_report_actually_writes(tmp_path: Path) -> None:
    """`HARVEST_DIRS` names a vault layout that `report/*.py` owns. Renaming a
    report's directory would otherwise unplug the harvest silently — every test
    green, and the acceptance gate quietly without a sensor."""
    target = date(2026, 8, 9)
    assert digest_path(tmp_path, target_date=target).parent.name in HARVEST_DIRS
    assert brief_path(tmp_path, target_sunday=target).parent.name in HARVEST_DIRS
    assert script_path(tmp_path, date=target).parent.name not in HARVEST_DIRS, (
        "an episode is a delivery channel, never an input surface (DESIGN §13.2)"
    )
