"""Tests for `score/rubrics.py` — the versioned prompt and its rendering.

Nothing here calls an LLM; this is pure-Python string assembly (CLAUDE.md §8).
"""

from __future__ import annotations

from signalforge.config import InterestsConfig
from signalforge.score.rubrics import RUBRIC_VERSION, build_triage_system_prompt


def make_interests(**overrides: object) -> InterestsConfig:
    data: dict[str, object] = {
        "priority_topics": ["agents.mcp", "engineering.code-gen"],
        "interests": ["python", "sqlite"],
        "stack": ["python", "fastapi"],
        "learning_goals": ["agent memory architectures"],
        "architecture_philosophy": "Local-first, deterministic pipelines.",
        "ignore": {"topics": ["crypto"], "people": [], "repos": []},
        "thresholds": {
            "weekly_min_signal": 3,
            "weekly_min_relevance": 3,
            "weekly_min_total": 10,
            "daily_max_items": 15,
        },
    }
    data.update(overrides)
    return InterestsConfig.model_validate(data)


def test_prompt_is_deterministic_for_the_same_config() -> None:
    """DESIGN §8 caching discipline: byte-identical prefix for byte-identical
    config is what lets many days of batch requests share one cache entry."""
    interests = make_interests()
    assert build_triage_system_prompt(interests) == build_triage_system_prompt(interests)


def test_prompt_contains_the_rubric_anchors() -> None:
    prompt = build_triage_system_prompt(make_interests())
    assert "Signal — substance vs noise" in prompt
    assert "Personal relevance" in prompt
    assert "Novelty" in prompt
    assert "keep" in prompt and "kill" in prompt


def test_prompt_embeds_the_interests_config() -> None:
    prompt = build_triage_system_prompt(make_interests())
    assert "agents.mcp" in prompt
    assert "crypto" in prompt
    assert "Local-first, deterministic pipelines." in prompt


def test_prompt_carries_no_obviously_volatile_content() -> None:
    """NEVER rule 10 — no timestamps or run IDs in the cached prefix.

    Not exhaustive, but catches the two easiest ways to accidentally bust the
    cache: an ISO year appearing where it shouldn't, and a run-id-shaped token.
    """
    prompt = build_triage_system_prompt(make_interests())
    assert "run_id" not in prompt.lower()
    assert "2026-" not in prompt  # no embedded timestamp


def test_two_different_orderings_of_the_same_lists_render_identically() -> None:
    """Sorted rendering — config authors shouldn't be able to bust the cache by
    reordering a YAML list that means the same thing either way."""
    a = make_interests(interests=["python", "sqlite"])
    b = make_interests(interests=["sqlite", "python"])
    assert build_triage_system_prompt(a) == build_triage_system_prompt(b)


def test_rubric_version_is_a_non_empty_string() -> None:
    assert isinstance(RUBRIC_VERSION, str) and RUBRIC_VERSION


def test_keep_rule_references_config_thresholds_not_hardcoded_numbers() -> None:
    """The keep bar must name the `thresholds` keys, never inline the numbers —
    otherwise tuning `interests.yaml` silently contradicts the prompt (CLAUDE.md
    §4 / NEVER rule 6). Regression for the literals that used to live here."""
    prompt = build_triage_system_prompt(make_interests())
    assert "weekly_min_signal" in prompt
    assert "weekly_min_relevance" in prompt
    assert "weekly_min_total" in prompt
    # The old hardcoded bar must not reappear in the instruction prose.
    assert "signal >= 3" not in prompt
    assert "(signal + relevance + novelty) >= 10" not in prompt
