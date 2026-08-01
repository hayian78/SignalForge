"""Tests for `curate/apply.py` — the only code that edits the operator's config.

Two properties carry the weight, and both are asserted against the **real shipped
`config/sources.yaml`** rather than a toy fixture, because the thing that would
break is the real file's shape:

* **Every existing comment survives every edit.** Those comments are the recorded
  reasoning behind every past pruning decision — the institutional memory a
  `safe_load`/`safe_dump` round-trip would silently delete.
* **A no-op means "already applied", never "could not find it".** Conflating the
  two lets a proposal be marked `applied` while the file is unchanged, silently
  discarding the operator's approval. A real bug did exactly that during
  development, so it is pinned here.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path

import pytest

from signalforge import db
from signalforge.config import load_sources
from signalforge.curate.apply import apply_approved_proposals, apply_to_text
from signalforge.models import ProposalKind, ProposalStatus, ProposalTier
from tests.conftest import REPO_ROOT

TODAY = Date(2026, 7, 30)
APPLIED_AT = datetime(2026, 7, 30, 6, 0, tzinfo=UTC)

# A feed and a repo that are genuinely in the shipped config *right now*, resolved
# at import rather than hardcoded.
#
# These tests deliberately run against the real `config/sources.yaml` (see the
# module docstring), which makes them the one suite an approved curation proposal
# can break: `curate apply` retires feeds, and a hardcoded `deepmind` here turned
# a successful retirement into four red tests that say nothing about the applier.
# The property under test is "a retire edit preserves comments", not "this
# particular feed exists", so the target is read from the file the test already
# loads.
_SHIPPED_SOURCES = load_sources(REPO_ROOT / "config")
RETIRE_RSS_TARGET = _SHIPPED_SOURCES.rss[0].id
RETIRE_GITHUB_TARGET = (
    _SHIPPED_SOURCES.github.releases[0]
    if _SHIPPED_SOURCES.github and _SHIPPED_SOURCES.github.releases
    else "block/goose"
)


@pytest.fixture
def config_dir(tmp_path: Path, repo_config_dir: Path) -> Path:
    """A writable copy of the real shipped config.

    A copy of the genuine file, not a fixture: the applier's whole job is surviving
    contact with this file's actual shape — nested rss entries, a `releases:` list
    two levels deep, inline keyword lists, and 100+ comment lines.
    """
    target = tmp_path / "config"
    target.mkdir()
    for name in ("sources.yaml", "interests.yaml"):
        shutil.copy(repo_config_dir / name, target / name)
    return target


def _comment_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip().startswith("#")]


def make_proposal(
    *,
    proposal_id: int = 1,
    kind: ProposalKind = ProposalKind.ADD_RSS,
    dedup_key: str = "https://newvoice.example.com/feed",
    payload: dict[str, object] | None = None,
    rationale: str = "Cited four times by items you marked useful.",
    status: ProposalStatus = ProposalStatus.APPROVED,
) -> db.Proposal:
    return db.Proposal(
        id=proposal_id,
        run_id=1,
        kind=kind,
        dedup_key=dedup_key,
        payload=payload
        if payload is not None
        else {"id": "newvoice", "url": "https://newvoice.example.com/feed"},
        rationale=rationale,
        evidence=({"url": "https://simonwillison.net/x/", "note": "links to it"},),
        probe=None,
        tier=ProposalTier.CORPUS,
        status=status,
        surface_date=TODAY,
        created_at=APPLIED_AT,
        decided_at=None,
        decision_note=None,
        applied_at=None,
    )


ALL_EDIT_KINDS = [
    pytest.param(
        make_proposal(kind=ProposalKind.ADD_RSS),
        id="add_rss",
    ),
    pytest.param(
        make_proposal(
            proposal_id=2,
            kind=ProposalKind.RETIRE_RSS,
            dedup_key=RETIRE_RSS_TARGET,
            payload={"id": RETIRE_RSS_TARGET},
        ),
        id="retire_rss",
    ),
    pytest.param(
        make_proposal(
            proposal_id=3, kind=ProposalKind.ADD_GITHUB_REPO, dedup_key="openai/codex", payload={}
        ),
        id="add_github_repo",
    ),
    pytest.param(
        make_proposal(
            proposal_id=4,
            kind=ProposalKind.RETIRE_GITHUB_REPO,
            dedup_key=RETIRE_GITHUB_TARGET,
            payload={},
        ),
        id="retire_github_repo",
    ),
    pytest.param(
        make_proposal(
            proposal_id=5, kind=ProposalKind.ADD_HN_KEYWORD, dedup_key="evaluation", payload={}
        ),
        id="add_hn_keyword",
    ),
    pytest.param(
        make_proposal(
            proposal_id=6, kind=ProposalKind.REMOVE_HN_KEYWORD, dedup_key="agi", payload={}
        ),
        id="remove_hn_keyword",
    ),
    pytest.param(
        make_proposal(
            proposal_id=7,
            kind=ProposalKind.ADD_ARXIV_KEYWORD,
            dedup_key="interpretability",
            payload={},
        ),
        id="add_arxiv_keyword",
    ),
    pytest.param(
        make_proposal(
            proposal_id=8,
            kind=ProposalKind.REMOVE_ARXIV_KEYWORD,
            dedup_key="multimodal",
            payload={},
        ),
        id="remove_arxiv_keyword",
    ),
]


# --------------------------------------------------------------------------- #
# the golden property: comments survive
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("proposal", ALL_EDIT_KINDS)
def test_every_edit_preserves_every_existing_comment(
    proposal: db.Proposal, repo_config_dir: Path
) -> None:
    """The reason this module edits text instead of round-tripping YAML.

    `config/sources.yaml` carries 100+ comment lines recording why each source is
    there and why several were removed. A `safe_dump` round-trip would delete all of
    them; this asserts that every one survives every kind of edit, against the real
    file rather than a fixture.
    """
    text = (repo_config_dir / "sources.yaml").read_text(encoding="utf-8")
    before = _comment_lines(text)
    assert len(before) > 50, "the shipped config should be comment-heavy; check the fixture"

    result = apply_to_text(text, proposal, today=TODAY, current=load_sources(repo_config_dir))
    assert result is not None, "every kind in this list applies to the shipped config"

    after = _comment_lines(result)
    assert set(before) <= set(after)


@pytest.mark.parametrize("proposal", ALL_EDIT_KINDS)
def test_every_edit_still_validates(proposal: db.Proposal, config_dir: Path) -> None:
    # A config that does not parse is a 6am failure for every command, so this is
    # the floor for all eight edit shapes.
    path = config_dir / "sources.yaml"
    result = apply_to_text(
        path.read_text(encoding="utf-8"), proposal, today=TODAY, current=load_sources(config_dir)
    )
    assert result is not None
    path.write_text(result, encoding="utf-8")

    load_sources(config_dir)  # raises ConfigError on failure


@pytest.mark.parametrize("proposal", ALL_EDIT_KINDS)
def test_every_edit_records_a_dated_provenance_comment(
    proposal: db.Proposal, repo_config_dir: Path
) -> None:
    # The next person reading this file should not be able to tell which lines a
    # human wrote and which the scout proposed — only what the reason was.
    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        proposal,
        today=TODAY,
        current=load_sources(repo_config_dir),
    )
    assert result is not None
    assert f"# 2026-07-30 curate #{proposal.id}:" in result
    assert proposal.rationale[:30] in result


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #


def test_an_added_feed_lands_at_the_end_of_the_rss_block(repo_config_dir: Path) -> None:
    """Appends go to the bottom of the list, not the top.

    Pinned because an indentation bug in `_block_bounds` made every block appear to
    end immediately after its header, so new entries landed above the first existing
    one. Valid YAML, wrong place, and invisible unless asserted.
    """
    text = (repo_config_dir / "sources.yaml").read_text(encoding="utf-8")
    current = load_sources(repo_config_dir)
    last_existing_id = current.rss[-1].id

    result = apply_to_text(text, make_proposal(), today=TODAY, current=current)
    assert result is not None
    lines = result.splitlines()

    assert lines.index("  - id: newvoice") > lines.index(f"  - id: {last_existing_id}")
    assert result.rindex("newvoice") > result.rindex(last_existing_id)


def test_an_added_repo_lands_at_the_end_of_the_releases_block(repo_config_dir: Path) -> None:
    text = (repo_config_dir / "sources.yaml").read_text(encoding="utf-8")
    current = load_sources(repo_config_dir)
    assert current.github is not None
    last_repo = current.github.releases[-1]

    result = apply_to_text(
        text,
        make_proposal(
            proposal_id=3, kind=ProposalKind.ADD_GITHUB_REPO, dedup_key="openai/codex", payload={}
        ),
        today=TODAY,
        current=current,
    )
    assert result is not None
    assert result.index("- openai/codex") > result.index(f"- {last_repo}")


def test_a_retirement_can_be_reversed_by_deleting_the_comment_prefix(
    repo_config_dir: Path,
) -> None:
    """Commented-out lines keep their nesting, so un-commenting restores valid YAML.

    That is the point of commenting out rather than deleting: the operator can
    reverse a retirement in an editor. Flattening the indentation would leave the
    `url:` line at the entry's own column and break the un-comment silently.
    """
    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            proposal_id=2,
            kind=ProposalKind.RETIRE_RSS,
            dedup_key="simonwillison",
            payload={"id": "simonwillison"},
        ),
        today=TODAY,
        current=load_sources(repo_config_dir),
    )
    assert result is not None
    lines = result.splitlines()
    start = next(index for index, line in enumerate(lines) if "curate #2" in line)
    commented = lines[start + 1 : start + 4]

    assert commented[0] == "  # - id: simonwillison"
    restored = [line.replace("# ", "", 1) for line in commented]
    assert restored[0] == "  - id: simonwillison"
    assert restored[1] == "    url: https://simonwillison.net/atom/everything/"
    assert restored[2] == "    weight: 1.3"


def test_the_trailing_newline_is_preserved(repo_config_dir: Path) -> None:
    # `splitlines` drops it, and a config file that loses its final newline makes a
    # noisy, meaningless line in every future git diff.
    text = (repo_config_dir / "sources.yaml").read_text(encoding="utf-8")
    assert text.endswith("\n")

    result = apply_to_text(
        text, make_proposal(), today=TODAY, current=load_sources(repo_config_dir)
    )

    assert result is not None
    assert result.endswith("\n")
    assert not result.endswith("\n\n")


# --------------------------------------------------------------------------- #
# no-op vs failure — the dangerous distinction
# --------------------------------------------------------------------------- #


def test_adding_a_feed_that_already_exists_is_a_no_op(repo_config_dir: Path) -> None:
    current = load_sources(repo_config_dir)
    existing = current.rss[0]

    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(dedup_key=existing.url, payload={"id": existing.id, "url": existing.url}),
        today=TODAY,
        current=current,
    )

    assert result is None


def test_adding_a_feed_whose_url_already_exists_under_another_id_is_a_no_op(
    repo_config_dir: Path,
) -> None:
    # Same feed, different proposed id. Canonicalized URL comparison is what stops
    # one source being ingested twice under two ids.
    current = load_sources(repo_config_dir)
    existing = current.rss[0]

    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            dedup_key=existing.url, payload={"id": "a-different-id", "url": existing.url}
        ),
        today=TODAY,
        current=current,
    )

    assert result is None


def test_retiring_an_absent_feed_is_a_no_op(repo_config_dir: Path) -> None:
    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            proposal_id=2,
            kind=ProposalKind.RETIRE_RSS,
            dedup_key="never-configured",
            payload={"id": "never-configured"},
        ),
        today=TODAY,
        current=load_sources(repo_config_dir),
    )

    assert result is None


def test_a_target_present_in_config_but_unlocatable_in_text_raises(
    repo_config_dir: Path,
) -> None:
    """The failure that must never be mistaken for a no-op.

    If the parsed config says a source is there but the text edit cannot find its
    lines, the file shape is not what the applier expects. Returning None would mark
    the proposal `applied` with the file untouched — silently discarding the
    operator's approval. A `_block_bounds` bug did exactly this, and it was
    invisible from outside.

    Simulated by handing `apply_to_text` a config that claims a feed the text does
    not contain, which is precisely the state that bug produced.
    """
    current = load_sources(repo_config_dir)
    text_without_the_entry = "\n".join(
        line
        for line in (repo_config_dir / "sources.yaml").read_text(encoding="utf-8").splitlines()
        if f"id: {RETIRE_RSS_TARGET}" not in line
    )

    with pytest.raises(ValueError, match="could not be located"):
        apply_to_text(
            text_without_the_entry,
            make_proposal(
                proposal_id=2,
                kind=ProposalKind.RETIRE_RSS,
                dedup_key=RETIRE_RSS_TARGET,
                payload={"id": RETIRE_RSS_TARGET},
            ),
            today=TODAY,
            current=current,
        )


def test_adding_a_feed_whose_id_is_taken_by_a_different_feed_raises(
    repo_config_dir: Path,
) -> None:
    """An id collision is not an already-applied edit, and must not read as one.

    The two look identical from the outside — the id is in the config either way —
    but they mean opposite things. `existing.url` already configured means the work
    is done; a *different* feed holding the id means the operator's approved
    addition has nowhere to land. Returning None for the second marks the proposal
    `applied` with the file untouched and the feed never added.
    """
    current = load_sources(repo_config_dir)
    existing = current.rss[0]

    with pytest.raises(ValueError, match="already uses"):
        apply_to_text(
            (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
            make_proposal(
                dedup_key="https://brand-new.example.com/feed",
                payload={"id": existing.id, "url": "https://brand-new.example.com/feed"},
            ),
            today=TODAY,
            current=current,
        )


INJECTED_URL = (
    "https://evil.example.com/feed\n"
    "defaults:\n"
    "  fetch_timeout: 1\n"
    "  min_hn_points: 0\n"
    "  max_summary_chars: 1\n"
    "  max_item_age_days: 9999\n"
)


def test_a_url_carrying_a_newline_cannot_inject_yaml(repo_config_dir: Path) -> None:
    """The one corruption the re-validation safety net cannot catch.

    `urlsplit` strips embedded newlines before parsing, so this string validates as
    a well-formed `https://` URL while still containing them. Written into a line
    with an f-string it stops being one entry and becomes several — and the result
    is **valid** YAML, where a duplicate top-level key silently wins, so
    `load_sources` succeeds and nothing is reverted. The operator ticks "add this
    feed" and their `defaults:` block is replaced.

    Reachable from the scout's web search, which makes a page it happens to read an
    input to the operator's git-tracked config. Refused at the write site so it holds
    for every path in, not just the scout's.
    """
    with pytest.raises(ValueError, match="control character"):
        apply_to_text(
            (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
            make_proposal(
                dedup_key="https://evil.example.com/feed",
                payload={"id": "evil", "url": INJECTED_URL},
            ),
            today=TODAY,
            current=load_sources(repo_config_dir),
        )


@pytest.mark.parametrize(
    "control",
    ["\n", "\r", "\t", "\x00", "\x7f"],
)
def test_no_control_character_survives_into_a_written_value(
    control: str, repo_config_dir: Path
) -> None:
    """Not just newline: the guard is an allowlist over the whole control range.

    `\\r` alone reflows a line in some editors, `\\t` breaks YAML indentation
    outright, and a NUL in a config file is a debugging session nobody wants.
    """
    with pytest.raises(ValueError, match="control character"):
        apply_to_text(
            (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
            make_proposal(
                dedup_key=f"https://x.example.com/feed{control}",
                payload={"id": "x", "url": f"https://x.example.com/feed{control}"},
            ),
            today=TODAY,
            current=load_sources(repo_config_dir),
        )


def test_adding_a_keyword_that_differs_only_in_case_is_a_no_op(
    tmp_path: Path, repo_config_dir: Path
) -> None:
    """The case-race the RSS and GitHub kinds had closed, for the keyword kinds.

    `scout._clean_keyword` lowercases every proposed keyword and `config.py` puts no
    case rule on the configured list, so a `keywords: [LLM]` config against a
    proposed `llm` is an ordinary case. Compared case-sensitively it appends a
    duplicate to the operator's file.
    """
    config_dir = _write_config(tmp_path, repo_config_dir, keywords="[LLM, agent]")

    result = apply_to_text(
        (config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            proposal_id=20, kind=ProposalKind.ADD_HN_KEYWORD, dedup_key="llm", payload={}
        ),
        today=TODAY,
        current=load_sources(config_dir),
    )

    assert result is None


def test_removing_a_keyword_that_differs_only_in_case_still_removes_it(
    tmp_path: Path, repo_config_dir: Path
) -> None:
    """Otherwise the row is marked applied and the keyword is never removed."""
    config_dir = _write_config(tmp_path, repo_config_dir, keywords="[LLM, agent]")

    result = apply_to_text(
        (config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            proposal_id=21, kind=ProposalKind.REMOVE_HN_KEYWORD, dedup_key="llm", payload={}
        ),
        today=TODAY,
        current=load_sources(config_dir),
    )

    assert result is not None
    assert "keywords: [agent]" in result


def test_removing_an_arxiv_keyword_that_differs_only_in_case_still_removes_it(
    tmp_path: Path, repo_config_dir: Path
) -> None:
    config_dir = _write_config(tmp_path, repo_config_dir, arxiv_keyword="Reasoning")

    result = apply_to_text(
        (config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            proposal_id=22,
            kind=ProposalKind.REMOVE_ARXIV_KEYWORD,
            dedup_key="reasoning",
            payload={},
        ),
        today=TODAY,
        current=load_sources(config_dir),
    )

    assert result is not None
    assert "    # - Reasoning" in result


def _write_config(
    tmp_path: Path,
    repo_config_dir: Path,
    *,
    keywords: str = "[llm]",
    arxiv_keyword: str = "reasoning",
) -> Path:
    """A tiny config whose casing is chosen by the test rather than by the repo.

    The shipped `sources.yaml` is all-lowercase, so the case-race cases above cannot
    be expressed against it. This is the one place in this file that does not use the
    real config, and only because the property under test is about casing the real
    file does not contain.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "sources.yaml").write_text(
        "defaults:\n"
        "  fetch_timeout: 20\n"
        "  min_hn_points: 80\n"
        "  max_summary_chars: 4000\n"
        "  max_item_age_days: 7\n"
        "rss: []\n"
        f"hackernews:\n  keywords: {keywords}\n"
        f"arxiv:\n  categories: [cs.AI]\n  require_keywords:\n    - {arxiv_keyword}\n",
        encoding="utf-8",
    )
    shutil.copy(repo_config_dir / "interests.yaml", config_dir / "interests.yaml")
    return config_dir


def test_retiring_a_repo_whose_case_differs_from_the_config_still_retires_it(
    repo_config_dir: Path,
) -> None:
    """GitHub slugs are case-insensitive; a literal `in` check is not.

    `OpenHands/OpenHands` proposed as `openhands/openhands` used to miss the
    membership check, return None, and be recorded as applied with the watch still
    in place — the silent discard this module's docstring is about.
    """
    current = load_sources(repo_config_dir)
    configured = current.github.releases[0] if current.github else ""
    assert configured, "the shipped config must have at least one release watch"

    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            proposal_id=9,
            kind=ProposalKind.RETIRE_GITHUB_REPO,
            dedup_key=configured.upper(),
            payload={},
        ),
        today=TODAY,
        current=current,
    )

    assert result is not None
    assert f"    # - {configured}" in result


def test_adding_a_repo_that_differs_only_in_case_is_a_no_op(repo_config_dir: Path) -> None:
    """The same repo must not be watched twice under two spellings (NEVER rule 4)."""
    current = load_sources(repo_config_dir)
    configured = current.github.releases[0] if current.github else ""

    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            proposal_id=10,
            kind=ProposalKind.ADD_GITHUB_REPO,
            dedup_key=configured.upper(),
            payload={},
        ),
        today=TODAY,
        current=current,
    )

    assert result is None


def test_adding_an_hn_keyword_that_is_already_present_is_a_no_op(repo_config_dir: Path) -> None:
    """The dual-write recovery path for the inline-list kinds.

    A crash between writing the YAML and flipping the row to `applied` leaves this
    exact state. Without a membership check the keyword kinds fell through to the
    "present in config but unlocatable" error, so recovery reported a file-shape
    failure for a file that was already correct.
    """
    current = load_sources(repo_config_dir)
    keyword = current.hackernews.keywords[0] if current.hackernews else ""
    assert keyword, "the shipped config must have at least one HN keyword"

    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            proposal_id=11, kind=ProposalKind.ADD_HN_KEYWORD, dedup_key=keyword, payload={}
        ),
        today=TODAY,
        current=current,
    )

    assert result is None


def test_removing_an_hn_keyword_that_is_already_gone_is_a_no_op(repo_config_dir: Path) -> None:
    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            proposal_id=12,
            kind=ProposalKind.REMOVE_HN_KEYWORD,
            dedup_key="never-a-keyword",
            payload={},
        ),
        today=TODAY,
        current=load_sources(repo_config_dir),
    )

    assert result is None


def test_adding_a_feed_with_no_source_id_raises(repo_config_dir: Path) -> None:
    with pytest.raises(ValueError, match="no source id"):
        apply_to_text(
            (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
            make_proposal(payload={"url": "https://x.example/feed"}),
            today=TODAY,
            current=load_sources(repo_config_dir),
        )


# --------------------------------------------------------------------------- #
# apply_approved_proposals — the DB-and-file orchestration
# --------------------------------------------------------------------------- #


def _store(conn: sqlite3.Connection, proposal: db.Proposal, *, approve: bool = True) -> int:
    run_id = db.start_run(conn, "curate", started_at=APPLIED_AT)
    proposal_id = db.insert_proposal(
        conn,
        run_id=run_id,
        kind=proposal.kind,
        dedup_key=proposal.dedup_key,
        payload=proposal.payload,
        rationale=proposal.rationale,
        evidence=list(proposal.evidence),
        probe=None,
        tier=proposal.tier,
        status=ProposalStatus.PENDING,
        surface_date=TODAY,
        created_at=APPLIED_AT,
    )
    assert proposal_id is not None
    if approve:
        db.decide_proposal(
            conn,
            proposal_id=proposal_id,
            status=ProposalStatus.APPROVED,
            decided_at=APPLIED_AT,
        )
    return proposal_id


def test_applying_writes_the_file_and_marks_the_proposal(
    conn: sqlite3.Connection, config_dir: Path
) -> None:
    proposal_id = _store(conn, make_proposal())

    outcome = apply_approved_proposals(conn, config_dir, today=TODAY, applied_at_now=APPLIED_AT)

    assert outcome.text_changed is True
    assert [change.proposal_id for change in outcome.changes] == [proposal_id]
    assert outcome.errors == []
    stored = db.get_proposal(conn, proposal_id)
    assert stored is not None
    assert stored.status is ProposalStatus.APPLIED
    assert "newvoice" in (config_dir / "sources.yaml").read_text(encoding="utf-8")


def test_applying_twice_leaves_the_file_byte_identical(
    conn: sqlite3.Connection, config_dir: Path
) -> None:
    # CLAUDE.md §3's bar: a re-run is a no-op, not merely duplicate-free. The status
    # guard stops the second pass, and the membership check would stop it even if
    # the guard were bypassed.
    _store(conn, make_proposal())
    apply_approved_proposals(conn, config_dir, today=TODAY, applied_at_now=APPLIED_AT)
    after_first = (config_dir / "sources.yaml").read_text(encoding="utf-8")

    second = apply_approved_proposals(conn, config_dir, today=TODAY, applied_at_now=APPLIED_AT)

    assert second.changes == []
    assert second.text_changed is False
    assert (config_dir / "sources.yaml").read_text(encoding="utf-8") == after_first


def test_an_edit_already_on_disk_is_recorded_as_applied_not_failed(
    conn: sqlite3.Connection, config_dir: Path
) -> None:
    """The dual-write recovery path.

    The YAML write and the status flip are two writes. If a crash lands between
    them, the row stays `approved` with the edit already on disk. The next run must
    recognise that as done rather than appending it a second time.
    """
    proposal_id = _store(conn, make_proposal())
    # Simulate the crash: apply the text edit by hand, leave the row approved.
    path = config_dir / "sources.yaml"
    applied_text = apply_to_text(
        path.read_text(encoding="utf-8"),
        make_proposal(proposal_id=proposal_id),
        today=TODAY,
        current=load_sources(config_dir),
    )
    assert applied_text is not None
    path.write_text(applied_text, encoding="utf-8")

    outcome = apply_approved_proposals(conn, config_dir, today=TODAY, applied_at_now=APPLIED_AT)

    assert len(outcome.changes) == 1
    assert outcome.changes[0].already_applied is True
    assert outcome.errors == []
    assert outcome.text_changed is False
    assert applied_text.count("- id: newvoice") == 1
    assert path.read_text(encoding="utf-8").count("- id: newvoice") == 1
    stored = db.get_proposal(conn, proposal_id)
    assert stored is not None
    assert stored.status is ProposalStatus.APPLIED


def test_a_pending_proposal_is_not_applied(conn: sqlite3.Connection, config_dir: Path) -> None:
    # Nothing changes the feed without a tick (DESIGN §7.1). `apply` reads approved
    # rows only.
    _store(conn, make_proposal(), approve=False)
    before = (config_dir / "sources.yaml").read_text(encoding="utf-8")

    outcome = apply_approved_proposals(conn, config_dir, today=TODAY, applied_at_now=APPLIED_AT)

    assert outcome.changes == []
    assert (config_dir / "sources.yaml").read_text(encoding="utf-8") == before


def test_one_failing_proposal_does_not_stop_the_others(
    conn: sqlite3.Connection, config_dir: Path
) -> None:
    # Failure isolation (CLAUDE.md §7): the broken one is recorded into `errors` for
    # `runs.errors`, and the good one still lands.
    broken = _store(
        conn,
        make_proposal(
            proposal_id=99,
            dedup_key="https://broken.example/feed",
            payload={"url": "https://broken.example/feed"},  # no id -> raises
        ),
    )
    good = _store(conn, make_proposal())

    outcome = apply_approved_proposals(conn, config_dir, today=TODAY, applied_at_now=APPLIED_AT)

    assert [change.proposal_id for change in outcome.changes] == [good]
    assert len(outcome.errors) == 1
    assert f"proposal-{broken}" == outcome.errors[0]["source_id"]
    broken_row = db.get_proposal(conn, broken)
    assert broken_row is not None
    assert broken_row.status is ProposalStatus.APPROVED, "a failure must not mark it applied"


def test_nothing_is_written_when_there_is_nothing_approved(
    conn: sqlite3.Connection, config_dir: Path
) -> None:
    before = (config_dir / "sources.yaml").read_text(encoding="utf-8")

    outcome = apply_approved_proposals(conn, config_dir, today=TODAY, applied_at_now=APPLIED_AT)

    assert outcome.changes == []
    assert outcome.errors == []
    assert (config_dir / "sources.yaml").read_text(encoding="utf-8") == before


def test_a_control_character_in_a_rationale_is_dropped_from_the_comment(
    repo_config_dir: Path,
) -> None:
    """The rationale is prose, so a stray byte is dropped rather than refused.

    Whitespace collapsing already handles the characters that could end the comment
    and start a YAML line. This covers the rest of the control range, which cannot
    corrupt the parse but would live in the operator's config invisibly. Raising here
    would discard an approved change over a cosmetic flaw in its explanation.
    """
    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(rationale="Cited\x1b[31m four times\x00 by items you marked useful."),
        today=TODAY,
        current=load_sources(repo_config_dir),
    )

    assert result is not None
    assert not any(ord(char) < 0x20 or ord(char) == 0x7F for char in "".join(result.splitlines()))
    assert "Cited[31m four times by items you marked useful." in result


def test_a_suggested_weight_is_written_beside_the_added_feed(repo_config_dir: Path) -> None:
    """The operator's lever for overruling it: a line in a diff they already read."""
    result = apply_to_text(
        (repo_config_dir / "sources.yaml").read_text(encoding="utf-8"),
        make_proposal(
            payload={"id": "newvoice", "url": "https://newvoice.example.com/feed", "weight": 1.3}
        ),
        today=TODAY,
        current=load_sources(repo_config_dir),
    )

    assert result is not None
    assert "    weight: 1.3" in result


def test_no_weight_line_is_written_when_none_was_suggested(repo_config_dir: Path) -> None:
    """A feed at the identity element should look like every other unweighted entry."""
    before = (repo_config_dir / "sources.yaml").read_text(encoding="utf-8")
    result = apply_to_text(
        before,
        make_proposal(payload={"id": "newvoice", "url": "https://newvoice.example.com/feed"}),
        today=TODAY,
        current=load_sources(repo_config_dir),
    )

    assert result is not None
    assert result.count("weight:") == before.count("weight:")
