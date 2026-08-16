"""Deterministic topic tagger tests (DESIGN §10, `score/taxonomy.py`).

No LLM is faked here because none is called — that is the point of the module
(NEVER rule 3). Every test drives off a taxonomy built in the test or loaded
from the shipped `config/taxonomy.yaml`; no keyword is written into an
assertion by hand, so the tests stay honest when the operator edits the tree
(CLAUDE.md §4).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from signalforge.config import TaxonomyConfig, load_taxonomy
from signalforge.db import connection, topics_for_items, upsert_item
from signalforge.score.taxonomy import (
    TAXONOMY_VERSION,
    compile_taxonomy,
    match_topics,
    stale_topics,
    tag_untagged_items,
)
from tests.conftest import dump_table, make_item

REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@pytest.fixture
def shipped_taxonomy() -> TaxonomyConfig:
    """The real `config/taxonomy.yaml`. A tagger that does not work against the
    operator's actual tree is not working."""
    return load_taxonomy(REPO_CONFIG_DIR)


@pytest.fixture
def taxonomy() -> TaxonomyConfig:
    """A small tree with the shapes worth testing: a multi-word phrase, a short
    word that could match inside a longer one, and punctuation."""
    return TaxonomyConfig.model_validate(
        {
            "agents": {"autonomy": {"keywords": ["autonomous agent", "agentic"]}},
            "policy": {"regulation": {"keywords": ["ai act", "export control"]}},
            "tooling": {"runtime": {"keywords": ["node.js"]}},
        }
    )


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "signalforge.db"


def _add_item(conn: sqlite3.Connection, *, title: str, summary: str | None, ext: str) -> int:
    item_id, _ = upsert_item(
        conn,
        make_item(
            title=title,
            summary=summary,
            external_id=ext,
            url=f"https://example.invalid/{ext}",
        ),
    )
    return item_id


def _topics(conn: sqlite3.Connection, item_id: int) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT topic FROM item_topics WHERE item_id = ? ORDER BY topic", (item_id,)
        )
    ]


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def test_every_shipped_leaf_matches_its_own_keywords(shipped_taxonomy: TaxonomyConfig) -> None:
    """Each `group.leaf` in the operator's real tree is reachable.

    Drives off the loaded config rather than a hand-written list, so adding a
    leaf to `taxonomy.yaml` extends this test automatically.
    """
    compiled = compile_taxonomy(shipped_taxonomy)
    for group, leaves in shipped_taxonomy.root.items():
        for leaf, node in leaves.items():
            for keyword in node.keywords:
                matches = match_topics(f"Report on {keyword} today", None, compiled)
                assert f"{group}.{leaf}" in [m.topic for m in matches], (
                    f"{group}.{leaf} keyword {keyword!r} did not match its own text"
                )


def test_a_keyword_inside_a_longer_word_does_not_match(taxonomy: TaxonomyConfig) -> None:
    """The bounded-match guarantee. Without it a keyword list rots silently."""
    compiled = compile_taxonomy(taxonomy)

    assert match_topics("Reagentic subagentics", None, compiled) == []
    assert [m.topic for m in match_topics("An agentic workflow", None, compiled)] == [
        "agents.autonomy"
    ]


def test_punctuated_keywords_match(taxonomy: TaxonomyConfig) -> None:
    """`.` is escaped, and the bounds do not break on a keyword ending in punctuation."""
    compiled = compile_taxonomy(taxonomy)

    assert [m.topic for m in match_topics("Shipping on node.js", None, compiled)] == [
        "tooling.runtime"
    ]
    assert match_topics("Shipping on nodeXjs", None, compiled) == []


def test_a_phrase_split_across_a_line_break_still_matches(taxonomy: TaxonomyConfig) -> None:
    compiled = compile_taxonomy(taxonomy)

    matches = match_topics("Title", "The new export\ncontrol rules", compiled)

    assert [m.topic for m in matches] == ["policy.regulation"]


def test_matching_is_case_insensitive(taxonomy: TaxonomyConfig) -> None:
    compiled = compile_taxonomy(taxonomy)

    assert [m.topic for m in match_topics("The AI ACT passes", None, compiled)] == [
        "policy.regulation"
    ]


def test_summary_is_searched_as_well_as_title(taxonomy: TaxonomyConfig) -> None:
    compiled = compile_taxonomy(taxonomy)

    assert match_topics("An unremarkable headline", "about the ai act", compiled)


def test_a_topic_matches_at_most_once(taxonomy: TaxonomyConfig) -> None:
    """Both keywords of one leaf firing is still one topic — `UNIQUE` forbids two."""
    compiled = compile_taxonomy(taxonomy)

    matches = match_topics("An autonomous agent doing agentic things", None, compiled)

    assert [m.topic for m in matches] == ["agents.autonomy"]


def test_no_match_is_an_empty_list(taxonomy: TaxonomyConfig) -> None:
    compiled = compile_taxonomy(taxonomy)

    assert match_topics("A post about sourdough", "Bread, mostly.", compiled) == []


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_tagging_writes_a_row_per_matched_topic(db: Path, taxonomy: TaxonomyConfig) -> None:
    with connection(db) as conn:
        item_id = _add_item(
            conn,
            title="The AI Act meets the autonomous agent",
            summary=None,
            ext="a",
        )

        outcome = tag_untagged_items(conn, taxonomy)

        assert outcome.items_tagged == 1
        assert outcome.topics_written == 2
        assert _topics(conn, item_id) == ["agents.autonomy", "policy.regulation"]


def test_the_matched_keyword_is_stored(db: Path, taxonomy: TaxonomyConfig) -> None:
    """The evidence an operator needs to tell a precise match from a lucky one."""
    with connection(db) as conn:
        _add_item(conn, title="New export control rules", summary=None, ext="a")

        tag_untagged_items(conn, taxonomy)

        row = conn.execute("SELECT matched_keyword, taxonomy_version FROM item_topics").fetchone()
        assert row[0] == "export control"
        assert row[1] == TAXONOMY_VERSION


def test_an_item_matching_nothing_writes_no_rows(db: Path, taxonomy: TaxonomyConfig) -> None:
    with connection(db) as conn:
        _add_item(conn, title="A post about sourdough", summary="Bread.", ext="a")

        outcome = tag_untagged_items(conn, taxonomy)

        assert outcome.items_examined == 1
        assert outcome.items_tagged == 0
        assert dump_table(conn, "item_topics") == []


def test_running_twice_produces_identical_state(db: Path, taxonomy: TaxonomyConfig) -> None:
    """Idempotency (CLAUDE.md §3, NEVER rule 4) — the rule the whole pipeline rests on."""
    with connection(db) as conn:
        _add_item(conn, title="The AI Act arrives", summary=None, ext="a")

        tag_untagged_items(conn, taxonomy)
        first = dump_table(conn, "item_topics")

        second_outcome = tag_untagged_items(conn, taxonomy)

        assert dump_table(conn, "item_topics") == first
        assert second_outcome.items_examined == 0, "an already-tagged item is not re-examined"


def test_a_new_item_is_tagged_without_disturbing_the_old_ones(
    db: Path, taxonomy: TaxonomyConfig
) -> None:
    with connection(db) as conn:
        _add_item(conn, title="The AI Act arrives", summary=None, ext="a")
        tag_untagged_items(conn, taxonomy)
        before = dump_table(conn, "item_topics")

        _add_item(conn, title="An agentic runtime", summary=None, ext="b")
        outcome = tag_untagged_items(conn, taxonomy)

        assert outcome.items_examined == 1
        assert dump_table(conn, "item_topics")[: len(before)] == before


def test_an_unmatched_item_is_re_examined_next_run(db: Path, taxonomy: TaxonomyConfig) -> None:
    """It writes no row, so a later taxonomy edit can still catch it."""
    with connection(db) as conn:
        _add_item(conn, title="A post about sourdough", summary=None, ext="a")

        tag_untagged_items(conn, taxonomy)
        second = tag_untagged_items(conn, taxonomy)

        assert second.items_examined == 1


def test_a_taxonomy_with_no_keywords_is_a_no_op(db: Path) -> None:
    empty = TaxonomyConfig.model_validate({})
    with connection(db) as conn:
        _add_item(conn, title="The AI Act arrives", summary=None, ext="a")

        outcome = tag_untagged_items(conn, empty)

        assert outcome.items_examined == 0
        assert dump_table(conn, "item_topics") == []


def test_topics_for_items_groups_by_id(db: Path, taxonomy: TaxonomyConfig) -> None:
    with connection(db) as conn:
        first = _add_item(conn, title="The AI Act meets an agentic future", summary=None, ext="a")
        second = _add_item(conn, title="A post about sourdough", summary=None, ext="b")
        tag_untagged_items(conn, taxonomy)

        grouped = topics_for_items(conn, [first, second], taxonomy_version=TAXONOMY_VERSION)

        assert grouped == {first: ["agents.autonomy", "policy.regulation"]}
        assert topics_for_items(conn, [], taxonomy_version=TAXONOMY_VERSION) == {}


# --------------------------------------------------------------------------- #
# Stale-keyword warning
# --------------------------------------------------------------------------- #


def test_a_leaf_that_never_matched_is_stale(db: Path, taxonomy: TaxonomyConfig) -> None:
    now = datetime(2026, 8, 16, tzinfo=UTC)
    with connection(db) as conn:
        _add_item(conn, title="The AI Act arrives", summary=None, ext="a")
        tag_untagged_items(conn, taxonomy)

        stale = stale_topics(conn, taxonomy, now=now, days=60)

        assert "policy.regulation" not in stale
        assert "agents.autonomy" in stale
        assert "tooling.runtime" in stale


def test_a_leaf_that_matched_long_ago_is_stale(db: Path, taxonomy: TaxonomyConfig) -> None:
    now = datetime(2026, 8, 16, tzinfo=UTC)
    with connection(db) as conn:
        _add_item(conn, title="The AI Act arrives", summary=None, ext="a")
        tag_untagged_items(conn, taxonomy)
        conn.execute(
            "UPDATE item_topics SET tagged_at = ?", ((now - timedelta(days=90)).isoformat(),)
        )

        assert "policy.regulation" in stale_topics(conn, taxonomy, now=now, days=60)
        assert "policy.regulation" not in stale_topics(conn, taxonomy, now=now, days=180)
