"""`curate/prompts.py` — the scout's system prompt (DESIGN §7.1).

Kept intentionally small: `test_llm_scout.py` covers the request shape end to
end. This file guards one thing specifically — `_staged_rule()` degrading
correctly when no `ProposalKind` is staged, after a real bug where an
unconditional f-string interpolation of an empty `_STAGED_KINDS` rendered
"Proposals of kind  are staged only" into the weekly Opus scout's paid prompt
(CLAUDE.md §6 — the priciest single call in the pipeline).
"""

from __future__ import annotations

from pathlib import Path

from signalforge.config import load_interests
from signalforge.curate.prompts import build_scout_system_prompt

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def test_no_malformed_staged_kinds_instruction_when_nothing_is_staged() -> None:
    """As of `ingest/arxiv.py` (2026-08-07), no `ProposalKind` is staged — the
    prompt must not render a rule naming an empty kind list."""
    prompt = build_scout_system_prompt(load_interests(CONFIG_DIR))

    assert "kind  are staged" not in prompt
    assert "staged only" not in prompt


def test_no_double_blank_line_where_the_staged_rule_would_be() -> None:
    prompt = build_scout_system_prompt(load_interests(CONFIG_DIR))

    assert "\n\n\n" not in prompt
