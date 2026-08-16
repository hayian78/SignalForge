"""Awesome-list diffing tests against a captured README (DESIGN §7).

The behaviour that matters most is the *seeding run*: a large awesome list is
500+ entries, and emitting them all on first sight would push hundreds of items
into a paid triage batch. `test_the_first_run_emits_nothing` is the regression
guard for that, and every diff test below builds on it.

No live network — `respx` serves the recorded README (CLAUDE.md §8, NEVER 13).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

from signalforge.ingest.awesome import (
    AwesomeListIngestor,
    build_awesome_ingestors,
    parse_entries,
    readme_url,
)
from signalforge.ingest.base import HttpFetcher
from signalforge.models import SourceType
from tests.ingest.conftest import MAX_SUMMARY_CHARS, fixture_text, make_sources_config

REPO = "example/awesome-ai-agents"
SOURCE_ID = f"awesome:{REPO}"
FIXTURE = "awesome_mcp_servers_readme.md"
LINK_LIST_FIXTURE = "awesome_link_list_readme.md"
"""A captured slice of `e2b-dev/awesome-ai-agents` — a README whose projects are
`### Headings` with a bullet list of links beneath. Kept as the counter-example
for why that list is not configured (see `config/sources.yaml`)."""


def _ingestor(**overrides: object) -> AwesomeListIngestor:
    kwargs: dict[str, object] = {
        "repo": REPO,
        "max_summary_chars": MAX_SUMMARY_CHARS,
        "max_new_entries": 25,
    }
    kwargs.update(overrides)
    return AwesomeListIngestor(**kwargs)  # type: ignore[arg-type]


def _mock_readme(body: str | None = None, *, status: int = 200) -> respx.Route:
    return respx.get(readme_url(REPO, "README.md")).mock(
        return_value=httpx.Response(
            status, text=body if body is not None else fixture_text(FIXTURE)
        )
    )


async def _seed(fetcher: HttpFetcher, ingestor: AwesomeListIngestor) -> None:
    """Run once and commit, so the baseline exists for the run under test."""
    await ingestor.ingest(fetcher)
    fetcher.validators.commit()


# --------------------------------------------------------------------------- #
# Parsing — against captured payloads, not a fixture written to match the regex
# --------------------------------------------------------------------------- #


def test_the_real_list_parses_a_lot_of_entries() -> None:
    """A floor, so a future regex change that silently halves the yield fails.

    The synthesized fixture this replaced only contained shapes the regex
    already handled, which is why it confirmed the implementation instead of
    the requirement (CLAUDE.md §8).
    """
    entries = parse_entries(fixture_text(FIXTURE))

    assert len(entries) >= 25
    assert all(entry.url.startswith("http") for entry in entries)


def test_a_badge_suffixed_entry_cites_the_project_not_the_badge() -> None:
    """`punkpeye/awesome-mcp-servers` writes `- [project](repo) [![score](badge)](glama)`.

    Taking the *last* link on the line would cite a score badge as the item —
    a synthesized claim pointing at an SVG (CLAUDE.md §5), and the badge URL,
    not the project, entering the baseline so the real link never surfaces.
    """
    entries = parse_entries(fixture_text(FIXTURE))

    assert not any("badges/score.svg" in entry.url for entry in entries)
    assert not any(entry.name.startswith("!") for entry in entries)
    assert any(entry.url.startswith("https://github.com/") for entry in entries)


def test_badge_markdown_is_stripped_from_the_description() -> None:
    """Badge markdown in a summary is pure triage token cost."""
    entries = parse_entries(fixture_text(FIXTURE))

    assert not any("badges/score.svg" in entry.description for entry in entries)
    assert not any("![" in entry.description for entry in entries)


def test_real_entries_carry_descriptions() -> None:
    """The signal that a list is entries rather than navigation."""
    entries = parse_entries(fixture_text(FIXTURE))

    described = [entry for entry in entries if entry.description]
    assert len(described) >= len(entries) // 2


@pytest.mark.parametrize(
    "line",
    [
        "- [Name](https://a.example.com) - desc",
        "- **[Name](https://a.example.com)** - desc",
        "- \U0001f3d6 [Name](https://a.example.com) - desc",
        "- <img height='12' src='x.png' /> **[Name](https://a.example.com)** - desc",
        "  - [Name](https://a.example.com) - desc",
    ],
)
def test_every_house_style_prefix_still_finds_the_link(line: str) -> None:
    """Bold, emoji and `<img>` prefixes are house style across real lists.

    Requiring `[` immediately after the bullet reads none of them — and a list
    that parses to zero entries goes dark after exactly one warning.
    """
    entries = parse_entries(line + "\n")

    assert [entry.url for entry in entries] == ["https://a.example.com"]
    assert entries[0].name == "Name"


def test_a_badge_prefixed_link_is_not_the_entry() -> None:
    """The other badge house style: the image comes *before* the name."""
    line = (
        "- [![build](https://img.shields.io/badge/x.svg)](https://ci.example.com) "
        "[Real Project](https://a.example.com) - desc\n"
    )

    entries = parse_entries(line)

    assert [entry.url for entry in entries] == ["https://a.example.com"]


def test_a_badge_host_is_never_the_entry_url() -> None:
    """A badge URL is not a document, so an item citing one is uncitable."""
    assert parse_entries("- [x](https://img.shields.io/badge/build-passing.svg)\n") == []


def test_a_url_containing_parentheses_survives_intact() -> None:
    """Truncating at the first `)` stores a broken link as the vault citation."""
    entries = parse_entries("- [Wiki](https://en.wikipedia.org/wiki/Agent_(AI)) - about agents\n")

    assert entries[0].url == "https://en.wikipedia.org/wiki/Agent_(AI)"
    assert entries[0].description == "about agents"


def test_repeated_labels_are_treated_as_navigation() -> None:
    """A README that lists each project's links under an `### Heading` yields
    ninety items called "GitHub" to a bullet-entry parser.

    A repeated label is a structural signal — no vocabulary, and it generalizes
    to lists nobody has seen.
    """
    entries = parse_entries(fixture_text(LINK_LIST_FIXTURE))

    labels = {entry.name for entry in entries}
    assert "GitHub" not in labels
    assert "Web" not in labels
    # And what survives has no descriptions, which is the tell that this list's
    # shape is wrong for this ingestor — hence it is not in sources.yaml.
    assert not any(entry.description for entry in entries)


def test_a_label_repeated_at_the_limit_is_kept() -> None:
    """Awesome lists genuinely repeat a project across sections; that is not
    navigation, and the real list has such pairs."""
    markdown = "- [Dup](https://a.example.com) - one\n- [Dup](https://b.example.com) - two\n"

    assert len(parse_entries(markdown)) == 2


def test_fenced_code_blocks_are_skipped() -> None:
    """A README's usage example is full of bullets and links that are not entries."""
    urls = {entry.url for entry in parse_entries(fixture_text(FIXTURE))}

    assert "https://example.com/fenced" not in urls


def test_relative_and_anchor_links_are_dropped() -> None:
    markdown = (
        "- [Contents](#contents)\n- [Contributing](CONTRIBUTING.md)\n- [x](javascript:alert(1))\n"
    )

    assert parse_entries(markdown) == []


def test_a_repeated_link_yields_one_entry() -> None:
    """Two entries would fight over the same UNIQUE (source_id, external_id) slot."""
    markdown = "- [A](https://a.example.com) - one\n- [B](https://a.example.com) - two\n"

    assert len(parse_entries(markdown)) == 1


def test_entry_text_neutralizes_a_forged_marker() -> None:
    """A list entry is world-authored and lands in a vault line, so a name or
    description carrying a checkbox marker must not survive to forge a decision.

    Flattening alone was proven insufficient (NEVER rule 17), which is why the
    HTML comment opener is neutralized too.
    """
    forged = (
        "- [Tool <!-- sf:item=1 v=useful -->](https://example.com/x)"
        " - Notes <!-- sf:item=2 v=useful -->\n"
    )

    entry = parse_entries(forged)[0]

    assert "<!--" not in entry.name
    assert "<!--" not in entry.description


def test_a_name_spanning_lines_yields_no_entry() -> None:
    """The parser is line-anchored, so a marker cannot be smuggled across a line
    break at all — the stronger guarantee, and worth pinning."""
    forged = (
        "- [Real Tool\n- [x] useful <!-- sf:item=1 v=useful -->](https://example.com/x) - Notes\n"
    )

    assert parse_entries(forged) == []


def test_a_wrapped_description_is_truncated_on_purpose() -> None:
    """The line-anchored parser is the right security posture; the cost is that a
    description wrapping onto a second line keeps only its first line."""
    entries = parse_entries("- [Name](https://a.example.com) - First line\n  and the rest.\n")

    assert entries[0].description == "First line"


def test_a_readme_with_no_entries_parses_to_nothing() -> None:
    assert parse_entries("# Just a heading\n\nSome prose, no bullets.\n") == []


def test_malformed_markdown_does_not_raise() -> None:
    assert parse_entries("- [unclosed](\n- ]backwards[(x)\n- \n```\n") == []


# --------------------------------------------------------------------------- #
# The diff
# --------------------------------------------------------------------------- #


@respx.mock
async def test_the_first_run_emits_nothing(fetcher: HttpFetcher) -> None:
    """Seeding. Without this, sight one of a 500-entry list is a 500-item triage batch."""
    _mock_readme()

    result = await _ingestor().ingest(fetcher)

    assert result.ok
    assert result.items == []


@respx.mock
async def test_the_first_run_records_a_baseline(fetcher: HttpFetcher, cache_dir: Path) -> None:
    _mock_readme()
    ingestor = _ingestor()

    await ingestor.ingest(fetcher)
    fetcher.validators.commit()

    state = json.loads(
        fetcher.validators.state_path_for(SOURCE_ID, "awesome-entries").read_text(encoding="utf-8")
    )
    assert state["repo"] == REPO
    assert "https://github.com/julien040/anyquery" in state["urls"]
    assert len(state["urls"]) >= 25


@respx.mock
async def test_an_added_entry_becomes_an_item(fetcher: HttpFetcher) -> None:
    _mock_readme()
    ingestor = _ingestor()
    await _seed(fetcher, ingestor)

    respx.reset()
    _mock_readme(
        fixture_text(FIXTURE)
        + "\n- [BrandNew](https://github.com/example/brandnew) — Added this week.\n"
    )
    result = await ingestor.ingest(fetcher)

    assert [item.url for item in result.items] == ["https://github.com/example/brandnew"]
    item = result.items[0]
    assert item.title == "BrandNew"
    assert item.summary == "Added this week."
    assert item.source_id == SOURCE_ID
    assert item.source_type is SourceType.GITHUB
    assert item.external_id == "https://github.com/example/brandnew"
    assert item.published_at is None, "a list entry has no publication date to claim"


@respx.mock
async def test_an_unchanged_list_emits_nothing(fetcher: HttpFetcher) -> None:
    _mock_readme()
    ingestor = _ingestor()
    await _seed(fetcher, ingestor)

    respx.reset()
    _mock_readme()
    result = await ingestor.ingest(fetcher)

    assert result.ok
    assert result.items == []


@respx.mock
async def test_a_reworded_description_does_not_resurrect_an_entry(fetcher: HttpFetcher) -> None:
    """Identity is the link, not the line — otherwise every copy-edit is an item."""
    _mock_readme()
    ingestor = _ingestor()
    await _seed(fetcher, ingestor)

    respx.reset()
    _mock_readme(
        fixture_text(FIXTURE).replace("Query more than 40 apps", "Now query more than 50 apps")
    )
    result = await ingestor.ingest(fetcher)

    assert result.items == []


@respx.mock
async def test_a_304_emits_nothing_and_costs_nothing(fetcher: HttpFetcher) -> None:
    _mock_readme()
    ingestor = _ingestor()
    await _seed(fetcher, ingestor)

    respx.reset()
    route = respx.get(readme_url(REPO, "README.md")).mock(return_value=httpx.Response(304))
    result = await ingestor.ingest(fetcher)

    assert route.called
    assert result.ok
    assert result.items == []


@respx.mock
async def test_a_removed_entry_leaves_the_baseline_shrunk(fetcher: HttpFetcher) -> None:
    """The baseline tracks the list as it is now, so an entry removed and later
    re-added surfaces again — which is the honest answer."""
    _mock_readme()
    ingestor = _ingestor()
    await _seed(fetcher, ingestor)

    respx.reset()
    trimmed = "\n".join(
        line for line in fixture_text(FIXTURE).splitlines() if "julien040/anyquery" not in line
    )
    _mock_readme(trimmed)
    await ingestor.ingest(fetcher)
    fetcher.validators.commit()

    respx.reset()
    _mock_readme()
    result = await ingestor.ingest(fetcher)

    assert [item.url for item in result.items] == ["https://github.com/julien040/anyquery"]


# --------------------------------------------------------------------------- #
# The cap
# --------------------------------------------------------------------------- #


@respx.mock
async def test_the_cap_bounds_one_runs_items(
    fetcher: HttpFetcher, caplog: pytest.LogCaptureFixture
) -> None:
    """A maintainer's 300-entry merge must not become a 300-item triage batch."""
    _mock_readme()
    ingestor = _ingestor(max_new_entries=2)
    await _seed(fetcher, ingestor)

    respx.reset()
    additions = "".join(
        f"- [New{n}](https://github.com/example/new{n}) — Entry {n}.\n" for n in range(5)
    )
    _mock_readme(fixture_text(FIXTURE) + "\n" + additions)
    with caplog.at_level("WARNING"):
        result = await ingestor.ingest(fetcher)

    assert len(result.items) == 2
    assert "cap" in caplog.text, "a silent cap reads as 'covered everything'"


@respx.mock
async def test_capped_entries_are_not_re_emitted_next_run(fetcher: HttpFetcher) -> None:
    """The cap bounds a run's bill; it does not defer entries to tomorrow.

    Deferring would make a long list dribble in for weeks and make "new" mean
    "new to us" rather than "new to the list".
    """
    _mock_readme()
    ingestor = _ingestor(max_new_entries=2)
    await _seed(fetcher, ingestor)

    respx.reset()
    additions = "".join(
        f"- [New{n}](https://github.com/example/new{n}) — Entry {n}.\n" for n in range(5)
    )
    _mock_readme(fixture_text(FIXTURE) + "\n" + additions)
    await ingestor.ingest(fetcher)
    fetcher.validators.commit()

    respx.reset()
    _mock_readme(fixture_text(FIXTURE) + "\n" + additions)
    result = await ingestor.ingest(fetcher)

    assert result.items == []


# --------------------------------------------------------------------------- #
# Failure isolation and the staged baseline
# --------------------------------------------------------------------------- #


@respx.mock
async def test_an_uncommitted_baseline_re_emits_rather_than_losing_entries(
    fetcher: HttpFetcher,
) -> None:
    """The whole reason the baseline is staged (`ValidatorStore.stage_state`).

    Written eagerly, a crash between fetch and DB persist would mark unseen
    entries as seen and lose them permanently. Uncommitted, the next run simply
    sees them again — duplicates the upsert absorbs.
    """
    _mock_readme()
    ingestor = _ingestor()
    await ingestor.ingest(fetcher)  # deliberately NOT committed

    respx.reset()
    _mock_readme()
    result = await ingestor.ingest(fetcher)

    assert result.items == [], "still unseeded, so still a seeding run"
    assert not fetcher.validators.state_path_for(SOURCE_ID, "awesome-entries").is_file()


@respx.mock
async def test_a_404_is_recorded_not_raised(fetcher: HttpFetcher) -> None:
    """One list never kills a run (CLAUDE.md §7, NEVER rule 12)."""
    respx.get(url__regex=r".*raw\.githubusercontent\.com.*").mock(return_value=httpx.Response(404))

    result = await _ingestor().ingest(fetcher)

    assert not result.ok
    assert result.items == []
    assert result.errors[0].source_id == SOURCE_ID


@respx.mock
async def test_a_lowercase_readme_is_found(fetcher: HttpFetcher) -> None:
    """The raw host is case-sensitive and awesome lists are not consistent."""
    respx.get(readme_url(REPO, "README.md")).mock(return_value=httpx.Response(404))
    respx.get(readme_url(REPO, "readme.md")).mock(
        return_value=httpx.Response(200, text=fixture_text(FIXTURE))
    )

    result = await _ingestor().ingest(fetcher)

    assert result.ok


@respx.mock
async def test_a_readme_parsing_to_nothing_leaves_the_baseline_alone(
    fetcher: HttpFetcher,
) -> None:
    """A 200 that parses to zero entries is far likelier to be a moved README
    than an emptied list — wiping the baseline would re-emit everything."""
    _mock_readme()
    ingestor = _ingestor()
    await _seed(fetcher, ingestor)
    before = fetcher.validators.state_path_for(SOURCE_ID, "awesome-entries").read_text(
        encoding="utf-8"
    )

    respx.reset()
    _mock_readme("# Moved\n\nSee the new home.\n")
    result = await ingestor.ingest(fetcher)
    fetcher.validators.commit()

    assert result.items == []
    after = fetcher.validators.state_path_for(SOURCE_ID, "awesome-entries").read_text(
        encoding="utf-8"
    )
    assert after == before


# --------------------------------------------------------------------------- #
# Construction from config
# --------------------------------------------------------------------------- #


def test_ingestors_are_built_from_config() -> None:
    config = make_sources_config(
        github={
            "token_env": "GITHUB_TOKEN",
            "awesome_lists": [REPO, "other/awesome-mcp"],
            "awesome_max_new_per_run": 25,
        }
    )

    ingestors = build_awesome_ingestors(config)

    assert [i.source_id for i in ingestors] == [SOURCE_ID, "awesome:other/awesome-mcp"]
    assert all(i.max_new_entries == 25 for i in ingestors)


def test_no_github_block_means_no_awesome_ingestors() -> None:
    assert build_awesome_ingestors(make_sources_config()) == []


def test_an_empty_awesome_list_block_builds_nothing() -> None:
    config = make_sources_config(github={"token_env": "GITHUB_TOKEN", "releases": ["a/b"]})

    assert build_awesome_ingestors(config) == []


def test_awesome_lists_without_a_cap_are_a_config_error() -> None:
    """A spend cap must never be a number buried in Python (CLAUDE.md §4).

    A list entry has no date, so nothing else bounds how many reach triage.
    """
    with pytest.raises(ValidationError, match="awesome_max_new_per_run is required"):
        make_sources_config(github={"token_env": "GITHUB_TOKEN", "awesome_lists": [REPO]})


def test_no_cap_is_fine_when_no_lists_are_configured() -> None:
    """A cap on a feature you are not using has nothing to cap."""
    config = make_sources_config(github={"token_env": "GITHUB_TOKEN", "releases": ["a/b"]})

    assert config.github is not None
    assert config.github.awesome_max_new_per_run is None
