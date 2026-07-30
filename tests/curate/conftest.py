"""Fixtures for the curation suite.

Every HTTP interaction is a recorded payload served by `respx`, and `llm.py` is
faked at its boundary — nothing here reaches a live network or the real Anthropic
API (CLAUDE.md §8, NEVER rules 11 and 13).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signalforge import llm
from signalforge.config import CurationConfig, InterestsConfig, SourcesConfig, load_interests

FIXTURES = Path(__file__).parent.parent / "fixtures"

MAX_SUMMARY_CHARS = 4000
"""Test-side stand-in for `defaults.max_summary_chars`; production reads it from
config (NEVER rule 6)."""

MAX_ITEM_AGE_DAYS = 3650
"""Deliberately huge, NOT the shipped 7: the recorded feeds carry real capture
dates, so a shipped-width window would make the suite rot as they age."""


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def make_sources(**overrides: object) -> SourcesConfig:
    """A minimal valid `SourcesConfig`, with no sources unless asked for."""
    data: dict[str, object] = {
        "defaults": {
            "fetch_timeout": 5,
            "min_hn_points": 80,
            "max_summary_chars": MAX_SUMMARY_CHARS,
            "max_item_age_days": MAX_ITEM_AGE_DAYS,
        },
        "rss": [],
    }
    data.update(overrides)
    return SourcesConfig.model_validate(data)


def make_curation(**overrides: object) -> CurationConfig:
    """A `curation:` block. Values mirror the shipped ones unless overridden."""
    data: dict[str, object] = {
        "max_proposals_per_run": 5,
        "max_searches_per_run": 6,
        "yield_window_days": 30,
        "settled_display_days": 14,
    }
    data.update(overrides)
    return CurationConfig.model_validate(data)


def make_scout_proposal(**overrides: object) -> llm.ScoutProposal:
    """A valid `ScoutProposal` as the model would have returned it."""
    data: dict[str, object] = {
        "kind": "add_rss",
        "target": "https://newvoice.example.com/feed",
        "source_id": "newvoice",
        "rationale": "Cited three times this month by items you marked useful.",
        "evidence": [{"url": "https://simonwillison.net/x/", "note": "links to it"}],
        "tier": "web",
    }
    data.update(overrides)
    return llm.ScoutProposal.model_validate(data)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Scratch stand-in for `data/http_cache/` — never the real one."""
    path = tmp_path / "http_cache"
    path.mkdir()
    return path


@pytest.fixture
def interests(repo_config_dir: Path) -> InterestsConfig:
    """The real shipped interests — the scout prompt is built from them."""
    return load_interests(repo_config_dir)
