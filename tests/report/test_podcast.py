"""Tests for the podcast vault script (`report/podcast.py`, DESIGN §13.3 Stage 4).

Scope: pure assembly (`build_script_context`), template rendering
(`render_script`), the round trip (`parse_script`), and the write path
(`write_script`). Every test builds its own throwaway DB via the `conn`/
`make_item` fixtures in `tests/conftest.py` — never the real
`data/signalforge.db` (CLAUDE.md §8). This module never calls an LLM
(CLAUDE.md §2), so tests construct `BuiltScript` values directly rather
than depending on `synth/podcast.py`'s (separately tested) LLM-facing path.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from signalforge.db import upsert_item
from signalforge.llm import PodcastScript, PodcastSegment, PodcastTurn
from signalforge.report.podcast import (
    CoverageGap,
    ScriptContext,
    ScriptSegment,
    ScriptTurn,
    SourceLink,
    build_script_context,
    parse_script,
    render_script,
    script_path,
    write_script,
)
from signalforge.synth.podcast import BuiltScript
from tests.conftest import make_item

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FIXTURE = REPO_ROOT / "fixtures" / "podcast_script_golden.md"

TARGET_DATE = date(2026, 8, 7)
PRESENTER_A = "Alex"
PRESENTER_B = "Sam"


def _seed_items(conn: sqlite3.Connection) -> dict[str, int]:
    """The five items the golden fixture's script cites, in the fixture's
    own order — a bad-citation and a truncation drop each need a real row
    to resolve a title/url from, even though neither ends up in a rendered
    segment."""
    ids: dict[str, int] = {}
    for key, url, title in [
        ("a", "https://example.com/a", "Story A"),
        ("b", "https://example.com/b", "Story B"),
        ("c", "https://example.com/c", "Story C, teasered"),
        ("d", "https://example.com/d", "Bad citation story"),
        ("e", "https://example.com/e", "Truncated-away story"),
    ]:
        item_id, _ = upsert_item(conn, make_item(external_id=key, url=url, title=title))
        ids[key] = item_id
    return ids


def _built_script(ids: dict[str, int]) -> BuiltScript:
    """The `BuiltScript` the golden fixture renders — two story segments
    (one full, one teasered), plus one bad-citation drop and one
    truncation drop, matching every branch `podcast.md.j2` has."""
    script = PodcastScript(
        intro_turns=[
            PodcastTurn(speaker="A", text="Good morning, let's get into it."),
            PodcastTurn(speaker="B", text="Great lineup today."),
        ],
        segments=[
            PodcastSegment(
                item_ids=[ids["a"]],
                turns=[
                    PodcastTurn(speaker="A", text="First up, Story A."),
                    PodcastTurn(speaker="B", text="Interesting stuff."),
                ],
            ),
            PodcastSegment(
                item_ids=[ids["b"], ids["c"]],
                turns=[PodcastTurn(speaker="A", text="Now, B and C together.")],
            ),
        ],
        outro_turns=[
            PodcastTurn(speaker="A", text="That's the show, see you tomorrow."),
            PodcastTurn(speaker="B", text="See you then."),
        ],
    )
    return BuiltScript(
        script=script,
        script_version="podcast-v2",
        model="claude-opus-5",
        dropped_item_ids=(ids["d"],),
        truncation_dropped_item_ids=(ids["e"],),
        teasered_item_ids=(ids["c"],),
        truncated=True,
        input_tokens=1_000,
        output_tokens=500,
    )


# --------------------------------------------------------------------------- #
# build_script_context — assembly, citation resolution, coverage gaps
# --------------------------------------------------------------------------- #


def test_build_script_context_resolves_sources_and_splits_coverage_gaps_by_reason(
    conn: sqlite3.Connection,
) -> None:
    ids = _seed_items(conn)
    built = _built_script(ids)

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    assert context.item_count == 3  # a, b, c — d and e never aired
    labels = [segment.label for segment in context.segments]
    assert labels == ["Intro", "Segment 1", "Segment 2", "Outro"]

    segment_2 = context.segments[2]
    assert segment_2.teased is True
    assert [source.item_id for source in segment_2.sources] == [ids["b"], ids["c"]]

    segment_1 = context.segments[1]
    assert segment_1.teased is False

    assert context.coverage_gaps == (
        CoverageGap(
            item_id=ids["d"],
            title="Bad citation story",
            url="https://example.com/d",
            reason="bad_citation",
        ),
        CoverageGap(
            item_id=ids["e"],
            title="Truncated-away story",
            url="https://example.com/e",
            reason="truncated",
        ),
    )


def test_speaker_letters_map_to_presenter_display_names(conn: sqlite3.Connection) -> None:
    ids = _seed_items(conn)
    built = _built_script(ids)

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    intro = context.segments[0]
    assert [turn.speaker for turn in intro.turns] == [PRESENTER_A, PRESENTER_B]


def test_a_script_that_built_to_none_renders_an_empty_episode_with_gaps_still_shown(
    conn: sqlite3.Connection,
) -> None:
    """`built.script is None` (no items, a refused call, everything cleaned
    away) still renders a file — a quiet day is a visible fact, not a
    missing one — and any citation drops recorded before the script went
    to None (e.g. every segment was a bad citation) still show up."""
    ids = _seed_items(conn)
    built = BuiltScript(
        script=None,
        script_version="podcast-v2",
        model="claude-opus-5",
        dropped_item_ids=(ids["d"],),
        input_tokens=500,
        output_tokens=0,
    )

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    assert context.segments == ()
    assert context.item_count == 0
    assert context.coverage_gaps == (
        CoverageGap(
            item_id=ids["d"],
            title="Bad citation story",
            url="https://example.com/d",
            reason="bad_citation",
        ),
    )


def test_a_segment_whose_only_citation_is_missing_from_the_db_is_not_rendered(
    conn: sqlite3.Connection,
) -> None:
    """NEVER rule 7, enforced a second time at render — a row deleted (or
    never committed) between script generation and vault write must not
    leave a segment rendering with zero real citations."""
    ids = _seed_items(conn)
    missing_id = max(ids.values()) + 1_000  # never inserted
    script = PodcastScript(
        intro_turns=[PodcastTurn(speaker="A", text="Morning.")],
        segments=[
            PodcastSegment(
                item_ids=[ids["a"]], turns=[PodcastTurn(speaker="A", text="Story A airs fine.")]
            ),
            PodcastSegment(
                item_ids=[missing_id], turns=[PodcastTurn(speaker="A", text="This never airs.")]
            ),
        ],
        outro_turns=[PodcastTurn(speaker="A", text="Bye.")],
    )
    built = BuiltScript(script=script, script_version="podcast-v2", model="claude-opus-5")

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    labels = [segment.label for segment in context.segments]
    assert labels == ["Intro", "Segment 1", "Outro"]  # "Segment 2" dropped
    assert context.item_count == 1


def test_every_segment_unresolvable_renders_no_episode_at_all(conn: sqlite3.Connection) -> None:
    """If every story segment loses its last citation at render time, the
    intro/outro are dropped too — a show that opened and closed around zero
    stories is "nothing usable," the same outcome `build_script` already
    returns for a script cleaned down to nothing at build time."""
    never_inserted = 999_999
    script = PodcastScript(
        intro_turns=[PodcastTurn(speaker="A", text="Morning.")],
        segments=[
            PodcastSegment(
                item_ids=[never_inserted], turns=[PodcastTurn(speaker="A", text="Ghost story.")]
            )
        ],
        outro_turns=[PodcastTurn(speaker="A", text="Bye.")],
    )
    built = BuiltScript(script=script, script_version="podcast-v2", model="claude-opus-5")

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    assert context.segments == ()
    assert context.item_count == 0


def test_a_bad_citation_id_that_resolves_to_no_row_still_renders_a_linkless_gap(
    conn: sqlite3.Connection,
) -> None:
    """The common case: a fabricated id almost never names a real row.
    Losing the gap silently would make the one drop reason worth a paper
    trail — the model inventing something — the one that vanishes."""
    ids = _seed_items(conn)
    fabricated_id = 999_999  # never inserted
    script = PodcastScript(
        intro_turns=[PodcastTurn(speaker="A", text="Morning.")],
        segments=[
            PodcastSegment(item_ids=[ids["a"]], turns=[PodcastTurn(speaker="A", text="Story A.")])
        ],
        outro_turns=[PodcastTurn(speaker="A", text="Bye.")],
    )
    built = BuiltScript(
        script=script,
        script_version="podcast-v2",
        model="claude-opus-5",
        dropped_item_ids=(fabricated_id,),
    )

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    assert context.coverage_gaps == (
        CoverageGap(item_id=fabricated_id, title=None, url=None, reason="bad_citation"),
    )
    rendered = render_script(context)
    assert "item 999999" in rendered
    assert parse_script(rendered) == context


def test_an_id_in_both_drop_sets_renders_only_once(conn: sqlite3.Connection) -> None:
    """`synth.podcast`'s disjointness invariant (an id can only ever be a
    bad citation *or* a truncation drop, never both) should already make
    this unreachable — `_coverage_gaps`'s `seen` guard exists as a second,
    independent layer rather than a dependency on that invariant holding
    forever. One row, first-reason-wins, not two contradictory ones."""
    ids = _seed_items(conn)
    script = PodcastScript(
        intro_turns=[PodcastTurn(speaker="A", text="Morning.")],
        segments=[
            PodcastSegment(item_ids=[ids["a"]], turns=[PodcastTurn(speaker="A", text="Story A.")])
        ],
        outro_turns=[PodcastTurn(speaker="A", text="Bye.")],
    )
    built = BuiltScript(
        script=script,
        script_version="podcast-v2",
        model="claude-opus-5",
        dropped_item_ids=(ids["d"],),
        truncation_dropped_item_ids=(ids["d"],),
    )

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    assert context.coverage_gaps == (
        CoverageGap(
            item_id=ids["d"],
            title="Bad citation story",
            url="https://example.com/d",
            reason="bad_citation",
        ),
    )


def test_an_item_cited_in_a_surviving_segment_is_never_listed_as_a_coverage_gap(
    conn: sqlite3.Connection,
) -> None:
    """Nothing forbids the model citing the same item across two segments.
    If one of those segments was later dropped for length, the item must
    not be branded "not covered" while another segment still airs it."""
    ids = _seed_items(conn)
    script = PodcastScript(
        intro_turns=[PodcastTurn(speaker="A", text="Morning.")],
        segments=[
            PodcastSegment(
                item_ids=[ids["a"]], turns=[PodcastTurn(speaker="A", text="Story A airs fully.")]
            )
        ],
        outro_turns=[PodcastTurn(speaker="A", text="Bye.")],
    )
    built = BuiltScript(
        script=script,
        script_version="podcast-v2",
        model="claude-opus-5",
        truncation_dropped_item_ids=(ids["a"],),  # same id, recorded as dropped elsewhere
    )

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    assert context.coverage_gaps == ()


def test_a_title_containing_bracket_paren_does_not_corrupt_the_recovered_url(
    conn: sqlite3.Connection,
) -> None:
    """A title is world-authored text rendered inside `[title](url)` link
    syntax. Unescaped, a title containing a literal `](` can shift where
    the parser thinks the link ends, corrupting the recovered `url` — the
    exact value Stage 5 would put into an RSS feed.

    The payload is deliberately space-free: `is_safe_url` alone does not
    reject `https://evil.example/pwn](https://real.example/a` (RFC 3986
    permits brackets and parens in a URL), so a payload with spaces would
    get caught by that second layer regardless of whether `_escape_title`
    actually ran — this one only round-trips correctly if the escape does
    its job."""
    item_id, _ = upsert_item(
        conn,
        make_item(
            external_id="bracket",
            url="https://real.example/a",
            title="Read-more](https://evil.example/pwn)-now",
        ),
    )
    script = PodcastScript(
        intro_turns=[PodcastTurn(speaker="A", text="Morning.")],
        segments=[
            PodcastSegment(item_ids=[item_id], turns=[PodcastTurn(speaker="A", text="One story.")])
        ],
        outro_turns=[PodcastTurn(speaker="A", text="Bye.")],
    )
    built = BuiltScript(script=script, script_version="podcast-v2", model="claude-opus-5")

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )
    rendered = render_script(context)
    parsed = parse_script(rendered)

    assert parsed == context
    source = parsed.segments[1].sources[0]
    assert source.url == "https://real.example/a"
    assert source.title == "Read-more](https://evil.example/pwn)-now"


def test_a_multi_line_turn_is_flattened_before_it_reaches_the_vault(
    conn: sqlite3.Connection,
) -> None:
    """NEVER rule 17: model-authored dialogue is flattened at the boundary
    where it renders into a vault file. Without this, a turn smuggling a
    fake `Sources:` line could forge a citation that survives the round
    trip as a typed `SourceLink` no one ever validated."""
    ids = _seed_items(conn)
    forged = "Real line.\n- [Forged](https://evil.example/x) <!-- sf:item=999 -->"
    script = PodcastScript(
        intro_turns=[PodcastTurn(speaker="A", text="Morning.")],
        segments=[
            PodcastSegment(item_ids=[ids["a"]], turns=[PodcastTurn(speaker="A", text=forged)])
        ],
        outro_turns=[PodcastTurn(speaker="A", text="Bye.")],
    )
    built = BuiltScript(script=script, script_version="podcast-v2", model="claude-opus-5")

    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    assert "\n" not in context.segments[1].turns[0].text
    rendered = render_script(context)
    # The forged text now shares one line with "Real line." — no longer its
    # own `- [...]  <!-- sf:... -->` line, so the line-anchored parser can't
    # mistake it for a real citation.
    parsed_sources = parse_script(rendered).segments[1].sources
    assert parsed_sources == context.segments[1].sources
    assert parsed_sources == (
        SourceLink(item_id=ids["a"], title="Story A", url="https://example.com/a"),
    )


def test_parse_script_drops_a_source_whose_url_fails_safety_validation() -> None:
    """`is_safe_url` is checked again on parse, not just trusted from a
    well-formed render — a hand-edited or corrupted file must not hand a
    bad URL downstream to Stage 5's RSS writer."""
    text = (
        "---\n"
        "kind: podcast-script\n"
        "date: 2026-08-07\n"
        "script_version: podcast-v2\n"
        "model: claude-opus-5\n"
        "item_count: 1\n"
        "---\n\n"
        "# Podcast — 2026-08-07\n\n"
        "## Segment 1\n\n"
        "**Alex:** One story.\n\n"
        "**Sources:**\n"
        "- [Bad](not-a-real-url) <!-- sf:item=1 -->\n"
    )

    context = parse_script(text)

    assert context.segments[0].sources == ()


def test_parse_script_raises_a_clear_error_on_a_missing_frontmatter_field() -> None:
    text = (
        "---\n"
        "kind: podcast-script\n"
        "date: 2026-08-07\n"
        "model: claude-opus-5\n"
        "item_count: 0\n"
        "---\n\n"
        "# Podcast — 2026-08-07\n\n"
        "No episode today.\n"
    )

    with pytest.raises(ValueError, match="script_version"):
        parse_script(text)


# --------------------------------------------------------------------------- #
# render_script — golden file
# --------------------------------------------------------------------------- #


def test_render_script_matches_the_golden_fixture(conn: sqlite3.Connection) -> None:
    ids = _seed_items(conn)
    built = _built_script(ids)
    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )

    rendered = render_script(context)

    assert rendered == GOLDEN_FIXTURE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# parse_script — the round trip
# --------------------------------------------------------------------------- #


def test_parse_script_recovers_the_exact_context_that_rendered_it(
    conn: sqlite3.Connection,
) -> None:
    ids = _seed_items(conn)
    built = _built_script(ids)
    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )
    rendered = render_script(context)

    assert parse_script(rendered) == context


def test_re_rendering_a_parsed_context_is_byte_identical(conn: sqlite3.Connection) -> None:
    ids = _seed_items(conn)
    built = _built_script(ids)
    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )
    rendered = render_script(context)

    assert render_script(parse_script(rendered)) == rendered


def test_parse_script_round_trips_the_empty_episode_too(conn: sqlite3.Connection) -> None:
    built = BuiltScript(script=None, script_version="podcast-v2", model="claude-opus-5")
    context = build_script_context(
        conn, built, date=TARGET_DATE, presenter_a=PRESENTER_A, presenter_b=PRESENTER_B
    )
    rendered = render_script(context)

    assert parse_script(rendered) == context
    assert render_script(parse_script(rendered)) == rendered


def test_parse_script_raises_on_a_file_with_no_frontmatter() -> None:
    with pytest.raises(ValueError, match="frontmatter"):
        parse_script("# Not a podcast script\n\nJust some text.\n")


def test_parse_script_skips_a_coverage_gap_marker_with_an_unrecognized_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hand-edited or future-version file might carry a reason this
    parser doesn't know — skip that one gap rather than raising or
    fabricating a `Literal` value that was never validated."""
    text = (
        "---\n"
        "kind: podcast-script\n"
        "date: 2026-08-07\n"
        "script_version: podcast-v2\n"
        "model: claude-opus-5\n"
        "item_count: 0\n"
        "---\n\n"
        "# Podcast — 2026-08-07\n\n"
        "No episode today.\n\n"
        "## Coverage gaps\n\n"
        "*1 item(s) not covered:*\n\n"
        "- [Mystery](https://example.com/x) — dropped: who knows "
        "<!-- sf:item=42 reason=mystery -->\n"
    )

    context = parse_script(text)

    assert context.coverage_gaps == ()


# --------------------------------------------------------------------------- #
# write_script — the write path, idempotency
# --------------------------------------------------------------------------- #


def test_script_path_is_the_same_path_every_time(tmp_path: Path) -> None:
    assert script_path(tmp_path, date=TARGET_DATE) == tmp_path / "podcast" / "2026-08-07.md"


def test_write_script_writes_the_rendered_file_at_script_path(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    ids = _seed_items(conn)
    built = _built_script(ids)

    path = write_script(
        conn,
        built,
        date=TARGET_DATE,
        presenter_a=PRESENTER_A,
        presenter_b=PRESENTER_B,
        vault_dir=tmp_path,
    )

    assert path == script_path(tmp_path, date=TARGET_DATE)
    assert path.read_text(encoding="utf-8") == GOLDEN_FIXTURE.read_text(encoding="utf-8")


def test_write_script_overwrites_rather_than_appends_on_a_second_run(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    ids = _seed_items(conn)
    built = _built_script(ids)

    first = write_script(
        conn,
        built,
        date=TARGET_DATE,
        presenter_a=PRESENTER_A,
        presenter_b=PRESENTER_B,
        vault_dir=tmp_path,
    )
    second = write_script(
        conn,
        built,
        date=TARGET_DATE,
        presenter_a=PRESENTER_A,
        presenter_b=PRESENTER_B,
        vault_dir=tmp_path,
    )

    assert first == second
    assert first.read_text(encoding="utf-8") == GOLDEN_FIXTURE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# a hand-built ScriptContext round trips too — not just one derived from a DB
# --------------------------------------------------------------------------- #


def test_a_hand_built_context_round_trips() -> None:
    context = ScriptContext(
        date=TARGET_DATE,
        script_version="podcast-v2",
        model="claude-opus-5",
        item_count=1,
        segments=(
            ScriptSegment(label="Intro", turns=(ScriptTurn(speaker="Alex", text="Hi there."),)),
            ScriptSegment(
                label="Segment 1",
                turns=(ScriptTurn(speaker="Alex", text="One story today."),),
                sources=(
                    SourceLink(item_id=1, title="Only Story", url="https://example.com/only"),
                ),
                teased=True,
            ),
            ScriptSegment(label="Outro", turns=(ScriptTurn(speaker="Alex", text="Bye."),)),
        ),
    )

    rendered = render_script(context)

    assert parse_script(rendered) == context
