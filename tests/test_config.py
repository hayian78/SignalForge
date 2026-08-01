"""Tests for `config.py` — config is data, not code (CLAUDE.md §4).

Three properties carry the weight:

* the shipped `config/*.yaml` must validate — those are the files cron reads at
  06:00, and a parse test is the only thing standing between a typo and a dead
  morning run;
* the shapes DESIGN §7 and §11 specify must parse, so every config block stays
  exercised;
* a typo'd key must be a loud error, because config that silently does nothing
  is the worst failure mode for a file the user tunes by hand.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest
from pydantic import SecretStr

from signalforge.config import (
    SETTINGS_FILENAME,
    ConfigError,
    EmailChannelConfig,
    SettingsConfig,
    SourceDefaults,
    Thresholds,
    get_secret,
    load_interests,
    load_settings,
    load_sources,
)

# --------------------------------------------------------------------------- #
# Realistic config fixtures, mirroring the examples in DESIGN §7 and §11.
#
# These are hand-synced COPIES of the doc, and that is all they can be: a
# constant compared against a constant detects drift in itself and nothing else.
# So they make no claim about `docs/DESIGN.md` staying in step — they are just
# full, realistic configs to exercise every block.
#
# The drift that actually matters is code <-> the YAML that ships, and THAT is
# checked mechanically against the real file (see the shipped-config section).
# DESIGN is a design doc, free to describe a target the code hasn't reached yet
# (§5 documents Phase 2/3 tables we deliberately don't create); doc drift here
# surfaces as a loud ConfigError naming the key, not a silent wrong answer.
# --------------------------------------------------------------------------- #

SOURCES_YAML = textwrap.dedent("""
    defaults:
      fetch_timeout: 20
      min_hn_points: 80
      max_summary_chars: 4000
      max_item_age_days: 7

    rss:
      - id: simonwillison
        url: https://simonwillison.net/atom/everything/
        weight: 1.3            # score multiplier: trusted author
      - id: interconnects
        url: https://www.interconnects.ai/feed

    github:
      token_env: GITHUB_TOKEN
      releases: [Aider-AI/aider, langchain-ai/langgraph, modelcontextprotocol/specification,
                 ollama/ollama, vllm-project/vllm, BerriAI/litellm, anthropics/claude-code,
                 stanfordnlp/dspy, pydantic/pydantic-ai, ggml-org/llama.cpp,
               huggingface/transformers]
      awesome_lists: [e2b-dev/awesome-ai-agents, punkpeye/awesome-mcp-servers]

    arxiv:
      categories: [cs.AI, cs.CL, cs.LG, cs.SE]
      require_keywords: [agent, context, retrieval, inference, evaluation,
                         fine-tuning, quantization, reasoning, embedding, tool use]

    hackernews:
      keywords: [llm, claude, mcp, agent, rag, inference, ollama, vllm]
""")

DESIGN_INTERESTS_YAML = textwrap.dedent("""
    priority_topics: [industry.strategy, frontier.capabilities, enterprise.adoption,
                      agents.autonomy, policy.regulation, ai.research-direction]
    interests: [ai-strategy, frontier-models, enterprise-ai, ai-policy, agents,
                thought-leadership, claude-code, local-first, trading-systems]
    stack: [python, typescript, sqlite, docker, wsl]
    learning_goals: [where AI capabilities are heading over the next 6-24 months,
                     enterprise AI adoption and business impact, AI policy and governance,
                     agent memory architectures, production llm evaluation]
    architecture_philosophy: >
      Local-first, deterministic pipelines, low operational cost, monolith-by-default,
      boring technology, human-in-the-loop.
    ignore:
      topics: [crypto, web3]
      people: []
      repos: []
    thresholds: {weekly_min_signal: 3, weekly_min_relevance: 3, weekly_min_total: 10,
                 daily_max_items: 15}
""")

MINIMAL_SOURCES_YAML = textwrap.dedent("""
    defaults:
      fetch_timeout: 20
      min_hn_points: 80
      max_summary_chars: 4000
      max_item_age_days: 7
""")

MINIMAL_INTERESTS_YAML = textwrap.dedent("""
    thresholds: {weekly_min_signal: 3, weekly_min_relevance: 3, weekly_min_total: 10,
                 daily_max_items: 15}
""")


def write_sources(config_dir: Path, yaml_text: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "sources.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


def write_interests(config_dir: Path, yaml_text: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "interests.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# sources.yaml
# --------------------------------------------------------------------------- #


def test_design_section_7_sources_yaml_parses(tmp_path: Path) -> None:
    # Every block and field shape DESIGN §7 specifies, in one realistic config.
    # This exercises the models; it does not police the doc — see the fixture
    # comment above for why that job belongs to the shipped-config tests.
    write_sources(tmp_path, SOURCES_YAML)
    config = load_sources(tmp_path)

    assert config.defaults.fetch_timeout == 20
    assert config.defaults.min_hn_points == 80
    assert config.defaults.max_summary_chars == 4000
    assert config.defaults.max_item_age_days == 7
    assert [source.id for source in config.rss] == ["simonwillison", "interconnects"]
    assert config.rss[0].weight == 1.3
    assert config.rss[1].weight == 1.0, "weight defaults to the identity element"
    assert config.github is not None
    assert config.github.token_env == "GITHUB_TOKEN"
    assert "Aider-AI/aider" in config.github.releases
    assert len(config.github.releases) == 11
    assert config.github.awesome_lists == [
        "e2b-dev/awesome-ai-agents",
        "punkpeye/awesome-mcp-servers",
    ]
    assert config.arxiv is not None
    assert config.arxiv.categories == ["cs.AI", "cs.CL", "cs.LG", "cs.SE"]
    assert config.hackernews is not None
    assert "mcp" in config.hackernews.keywords


def test_sources_optional_blocks_may_be_absent(tmp_path: Path) -> None:
    write_sources(tmp_path, MINIMAL_SOURCES_YAML)
    config = load_sources(tmp_path)
    assert (config.github, config.arxiv, config.hackernews) == (None, None, None)
    assert config.rss == []


def test_sources_defaults_block_is_required(tmp_path: Path) -> None:
    write_sources(tmp_path, "rss: []\n")
    with pytest.raises(ConfigError, match="defaults"):
        load_sources(tmp_path)


@pytest.mark.parametrize(
    "knob", ["fetch_timeout", "min_hn_points", "max_summary_chars", "max_item_age_days"]
)
def test_omitting_a_source_default_is_an_error_not_a_silent_default(knob: str) -> None:
    # SourceDefaults has NO Python fallbacks: a tuning knob that silently
    # materializes in code is exactly NEVER rule 6. `max_summary_chars` is the
    # sharpest case — a Python default on the triage cost knob is a number that
    # sets the monthly bill while being invisible in the file the user tunes.
    values = {
        "fetch_timeout": 20,
        "min_hn_points": 80,
        "max_summary_chars": 4000,
        "max_item_age_days": 7,
    }
    del values[knob]
    with pytest.raises(ValueError, match=knob):
        SourceDefaults(**values)


@pytest.mark.parametrize("size", [0, -1])
def test_non_positive_max_summary_chars_is_rejected(tmp_path: Path, size: int) -> None:
    # Truncating summaries to 0 chars would send triage titles only, silently
    # gutting scoring quality rather than failing loudly.
    write_sources(
        tmp_path,
        "defaults:\n  fetch_timeout: 20\n  min_hn_points: 80\n"
        f"  max_summary_chars: {size}\n  max_item_age_days: 7\n",
    )
    with pytest.raises(ConfigError, match="max_summary_chars"):
        load_sources(tmp_path)


@pytest.mark.parametrize("days", [0, -1])
def test_non_positive_max_item_age_days_is_rejected(tmp_path: Path, days: int) -> None:
    # A zero-day window keeps only undated items — an ingest that silently
    # drops everything fresh rather than failing loudly.
    write_sources(
        tmp_path,
        "defaults:\n  fetch_timeout: 20\n  min_hn_points: 80\n"
        f"  max_summary_chars: 4000\n  max_item_age_days: {days}\n",
    )
    with pytest.raises(ConfigError, match="max_item_age_days"):
        load_sources(tmp_path)


def test_typod_key_raises_config_error_naming_the_key(tmp_path: Path) -> None:
    # The user edits the file, nothing changes, no signal — the failure mode
    # extra="forbid" exists to kill.
    write_sources(
        tmp_path,
        textwrap.dedent("""
            defaults:
              fetch_timeout: 20
              min_hn_points: 80
              fetch_timout: 30
        """),
    )
    with pytest.raises(ConfigError, match="fetch_timout"):
        load_sources(tmp_path)


def test_typod_key_at_the_root_raises_config_error_naming_the_key(tmp_path: Path) -> None:
    write_sources(tmp_path, MINIMAL_SOURCES_YAML + "rrs: []\n")
    with pytest.raises(ConfigError, match="rrs"):
        load_sources(tmp_path)


def test_typod_key_on_an_rss_source_raises_config_error_naming_the_key(tmp_path: Path) -> None:
    write_sources(
        tmp_path,
        MINIMAL_SOURCES_YAML
        + textwrap.dedent("""
            rss:
              - id: simonwillison
                url: https://simonwillison.net/atom/everything/
                wieght: 1.3
        """),
    )
    with pytest.raises(ConfigError, match="wieght"):
        load_sources(tmp_path)


def test_duplicate_rss_ids_are_rejected(tmp_path: Path) -> None:
    # source_id becomes items.source_id; two feeds sharing one would silently
    # merge their identity in (source_id, external_id).
    write_sources(
        tmp_path,
        MINIMAL_SOURCES_YAML
        + textwrap.dedent("""
            rss:
              - id: simonwillison
                url: https://simonwillison.net/atom/everything/
              - id: simonwillison
                url: https://simonwillison.net/atom/entries/
        """),
    )
    with pytest.raises(ConfigError, match="duplicate rss source id"):
        load_sources(tmp_path)


@pytest.mark.parametrize("timeout", [0, -1])
def test_non_positive_fetch_timeout_is_rejected(tmp_path: Path, timeout: int) -> None:
    write_sources(tmp_path, f"defaults:\n  fetch_timeout: {timeout}\n  min_hn_points: 80\n")
    with pytest.raises(ConfigError, match="fetch_timeout"):
        load_sources(tmp_path)


@pytest.mark.parametrize(
    "slug",
    ["aider", "Aider-AI/aider/extra", "/aider", "Aider-AI/"],
)
def test_malformed_repo_slug_is_rejected(tmp_path: Path, slug: str) -> None:
    write_sources(
        tmp_path,
        MINIMAL_SOURCES_YAML + f"github:\n  token_env: GITHUB_TOKEN\n  releases: [{slug!r}]\n",
    )
    with pytest.raises(ConfigError, match="owner/repo"):
        load_sources(tmp_path)


# --------------------------------------------------------------------------- #
# interests.yaml
# --------------------------------------------------------------------------- #


def test_design_section_11_interests_yaml_parses(tmp_path: Path) -> None:
    write_interests(tmp_path, DESIGN_INTERESTS_YAML)
    config = load_interests(tmp_path)

    assert "industry.strategy" in config.priority_topics
    assert config.stack == [
        "python",
        "typescript",
        "sqlite",
        "docker",
        "wsl",
    ]
    assert config.learning_goals == [
        "where AI capabilities are heading over the next 6-24 months",
        "enterprise AI adoption and business impact",
        "AI policy and governance",
        "agent memory architectures",
        "production llm evaluation",
    ]
    assert "Local-first" in config.architecture_philosophy
    assert config.ignore.topics == ["crypto", "web3"]
    assert config.ignore.people == []
    assert config.thresholds.weekly_min_signal == 3
    assert config.thresholds.weekly_min_relevance == 3
    assert config.thresholds.weekly_min_total == 10
    assert config.thresholds.daily_max_items == 15


def test_thresholds_block_is_required(tmp_path: Path) -> None:
    write_interests(tmp_path, "priority_topics: [agents.mcp]\n")
    with pytest.raises(ConfigError, match="thresholds"):
        load_interests(tmp_path)


@pytest.mark.parametrize(
    "knob",
    ["weekly_min_signal", "weekly_min_relevance", "weekly_min_total", "daily_max_items"],
)
def test_omitting_a_threshold_is_an_error_not_a_silent_default(knob: str) -> None:
    # Thresholds is the canonical NEVER rule 6 case: the weekly-brief gate and
    # the daily cap must be readable in YAML, never a number hiding in Python.
    values = {
        "weekly_min_signal": 3,
        "weekly_min_relevance": 3,
        "weekly_min_total": 10,
        "daily_max_items": 15,
    }
    del values[knob]
    with pytest.raises(ValueError, match=knob):
        Thresholds(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weekly_min_signal", 0),
        ("weekly_min_signal", 6),
        ("weekly_min_relevance", 0),
        ("weekly_min_relevance", 6),
        ("weekly_min_total", 2),
        ("weekly_min_total", 16),
        ("daily_max_items", 0),
    ],
)
def test_out_of_range_thresholds_are_rejected(field: str, value: int) -> None:
    # Scores are 1-5 on three dimensions (DESIGN §9); a threshold outside the
    # reachable range is a gate that always or never opens. A zero-item daily
    # cap is a digest that renders nothing, ever.
    values = {
        "weekly_min_signal": 3,
        "weekly_min_relevance": 3,
        "weekly_min_total": 10,
        "daily_max_items": 15,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        Thresholds(**values)


def test_typod_key_in_interests_raises_config_error_naming_the_key(tmp_path: Path) -> None:
    write_interests(tmp_path, MINIMAL_INTERESTS_YAML + "priorty_topics: [agents.mcp]\n")
    with pytest.raises(ConfigError, match="priorty_topics"):
        load_interests(tmp_path)


def test_interests_optional_blocks_default_to_empty(tmp_path: Path) -> None:
    # Empty lists are the identity element, not a tuning knob — safe as defaults.
    write_interests(tmp_path, MINIMAL_INTERESTS_YAML)
    config = load_interests(tmp_path)
    assert config.priority_topics == []
    assert config.ignore.topics == []
    assert config.architecture_philosophy == ""


# --------------------------------------------------------------------------- #
# curation block (DESIGN §7.1)
# --------------------------------------------------------------------------- #

VALID_CURATION_YAML = textwrap.dedent("""
    curation:
      max_proposals_per_run: 5
      max_searches_per_run: 12
      yield_window_days: 30
      settled_display_days: 14
""")


def test_curation_block_is_optional(tmp_path: Path) -> None:
    # Absent means the feature is off. That is different from the required blocks
    # (`defaults`, `thresholds`): those hold knobs a run always needs, whereas an
    # operator can legitimately not want a weekly scout at all.
    write_sources(tmp_path, MINIMAL_SOURCES_YAML)

    assert load_sources(tmp_path).curation is None


def test_curation_block_parses(tmp_path: Path) -> None:
    write_sources(tmp_path, MINIMAL_SOURCES_YAML + VALID_CURATION_YAML)

    curation = load_sources(tmp_path).curation

    assert curation is not None
    assert curation.max_proposals_per_run == 5
    assert curation.max_searches_per_run == 12
    assert curation.yield_window_days == 30
    assert curation.settled_display_days == 14


@pytest.mark.parametrize(
    "field",
    [
        "max_proposals_per_run",
        "max_searches_per_run",
        "yield_window_days",
        "settled_display_days",
    ],
)
def test_every_curation_field_is_required(tmp_path: Path, field: str) -> None:
    # No Python fallback for any of them (CLAUDE.md §10 rule 6). A default buried
    # in code is a spend limit nobody can see — and one of these is real money.
    partial = "\n".join(
        line for line in VALID_CURATION_YAML.splitlines() if not line.strip().startswith(field)
    )
    write_sources(tmp_path, MINIMAL_SOURCES_YAML + partial + "\n")

    with pytest.raises(ConfigError, match=field):
        load_sources(tmp_path)


def test_curation_rejects_an_unknown_key(tmp_path: Path) -> None:
    # extra="forbid" everywhere: a typo'd knob that silently does nothing is the
    # worst failure mode for config-as-data.
    write_sources(
        tmp_path, MINIMAL_SOURCES_YAML + VALID_CURATION_YAML + "  max_serches_per_run: 3\n"
    )

    with pytest.raises(ConfigError, match="max_serches_per_run"):
        load_sources(tmp_path)


def test_curation_allows_zero_searches(tmp_path: Path) -> None:
    # A supported, meaningful mode: the scout reasons from the stored corpus alone
    # and spends nothing on search. This is the cheapest useful configuration, so
    # it must not be validated away as if it were a mistake.
    write_sources(
        tmp_path,
        MINIMAL_SOURCES_YAML
        + VALID_CURATION_YAML.replace("max_searches_per_run: 12", "max_searches_per_run: 0"),
    )

    curation = load_sources(tmp_path).curation
    assert curation is not None
    assert curation.max_searches_per_run == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_proposals_per_run", 0),  # a run that can propose nothing is a no-op
        ("max_proposals_per_run", 21),  # one week must not be able to rewire the feed
        ("max_searches_per_run", -1),  # negative spend is meaningless
        ("yield_window_days", 0),  # an empty window makes every source look dead
        ("yield_window_days", 400),  # beyond a year the "lately" question stops meaning anything
        ("settled_display_days", -1),
        ("settled_display_days", 91),  # the block would grow without bound
    ],
)
def test_curation_rejects_out_of_range_values(tmp_path: Path, field: str, value: int) -> None:
    write_sources(
        tmp_path,
        MINIMAL_SOURCES_YAML
        + "\n".join(
            f"  {field}: {value}" if line.strip().startswith(field) else line
            for line in VALID_CURATION_YAML.splitlines()
        )
        + "\n",
    )

    with pytest.raises(ConfigError, match=field):
        load_sources(tmp_path)


# --------------------------------------------------------------------------- #
# The shipped config — the files that actually run at 06:00
#
# Every other test in this file validates a string constant, which proves the
# *models* work and proves nothing about the YAML on disk. These are the only
# tests that would catch a typo, a drifted key, or a knob someone deleted from
# the real `config/`. They are deliberately about shape, not content: asserting
# a specific feed list here would make every `/add-source` edit a test failure.
# --------------------------------------------------------------------------- #


def test_shipped_sources_yaml_parses(repo_config_dir: Path) -> None:
    # The file cron actually reads. A validation error here is a 6am failure;
    # this moves it to CI. (It cannot see a dead URL — that needs the network,
    # which tests never touch, NEVER rule 13. It does see a malformed key.)
    config = load_sources(repo_config_dir)

    assert config.defaults.fetch_timeout > 0
    assert config.defaults.max_summary_chars > 0
    assert config.defaults.max_item_age_days >= 1
    assert config.rss, "sources.yaml ships with no RSS feeds — Phase 0 has nothing to ingest"
    assert config.github is not None
    assert config.github.releases, "the github block ships with no repos to poll"


def test_shipped_sources_yaml_has_unique_rss_ids(repo_config_dir: Path) -> None:
    # source_id becomes items.source_id; a duplicate silently merges two feeds'
    # identity in the (source_id, external_id) UNIQUE key. The validator enforces
    # this — the assertion here is that the *shipped* file satisfies it.
    config = load_sources(repo_config_dir)
    ids = [source.id for source in config.rss]
    assert len(ids) == len(set(ids))


def test_shipped_sources_yaml_names_env_vars_not_tokens(repo_config_dir: Path) -> None:
    # The config/ directory is git-tracked. A token pasted into `token_env` is the
    # leak NEVER rule 16 exists to prevent, and this is the file it would land in.
    config = load_sources(repo_config_dir)
    assert config.github is not None
    assert config.github.token_env == config.github.token_env.upper()
    assert not config.github.token_env.lower().startswith(
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")
    )


def test_shipped_interests_yaml_parses(repo_config_dir: Path) -> None:
    config = load_interests(repo_config_dir)

    # Thresholds gate the weekly brief; a missing one is a run that produces
    # nothing rather than a run that fails.
    assert 1 <= config.thresholds.weekly_min_signal <= 5
    assert 1 <= config.thresholds.weekly_min_relevance <= 5
    assert 3 <= config.thresholds.weekly_min_total <= 15
    assert config.thresholds.daily_max_items >= 1
    # The crowding limits are what stop one prolific source (a link blog, a
    # release watch shipping four versions) from sweeping the digest. They are
    # optional in the model, so assert the shipped file actually sets them —
    # deleting either key is a silent revert, not a failure.
    assert config.thresholds.daily_max_per_source is not None
    assert config.thresholds.daily_max_per_github_repo is not None
    assert config.thresholds.daily_max_per_github_repo <= config.thresholds.daily_max_per_source, (
        "the per-repo limit is the tighter of the two; above the per-source cap it is a no-op"
    )
    assert config.priority_topics, "interests.yaml defines 'relevant to me' — it cannot be empty"


def test_shipped_sources_yaml_configures_curation(repo_config_dir: Path) -> None:
    # The block is optional in the model (absent = feature off), so assert the
    # shipped file actually sets it — deleting the key silently disables the
    # weekly scout rather than failing anything.
    config = load_sources(repo_config_dir)

    assert config.curation is not None
    # No bound assertions here: the model's own `ge=`/`le=` already hold by
    # construction if `load_sources` returned, so restating them would be a test
    # that cannot fail. What is worth asserting is what the model does *not*
    # enforce — that the block is present at all, and that the money knob is sane.
    # The one cap that is real money: web search bills per call, not per token
    # (DESIGN §8). A shipped value above the ceiling in `llm.py` is clamped there,
    # not here, but a wildly large number in the file is worth catching in CI.
    assert config.curation.max_searches_per_run <= 20, (
        "max_searches_per_run is a spend cap; a value this high wants a deliberate "
        "decision and a matching ceiling in llm.py"
    )


def test_shipped_configs_load_together(repo_config_dir: Path) -> None:
    # The pair a run actually needs, loaded the way a run loads them.
    assert load_sources(repo_config_dir) is not None
    assert load_interests(repo_config_dir) is not None


# --------------------------------------------------------------------------- #
# Loader error handling
# --------------------------------------------------------------------------- #


def test_missing_sources_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_sources(tmp_path)


def test_missing_interests_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_interests(tmp_path)


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    write_sources(tmp_path, "defaults: {fetch_timeout: 20\nrss: [\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_sources(tmp_path)


def test_empty_file_raises_config_error(tmp_path: Path) -> None:
    write_sources(tmp_path, "")
    with pytest.raises(ConfigError, match="empty"):
        load_sources(tmp_path)


def test_non_mapping_top_level_raises_config_error(tmp_path: Path) -> None:
    write_sources(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_sources(tmp_path)


def test_config_error_names_the_file_and_every_bad_field(tmp_path: Path) -> None:
    path = write_sources(tmp_path, "defaults:\n  fetch_timeout: nope\n  min_hn_points: -1\n")
    with pytest.raises(ConfigError) as excinfo:
        load_sources(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "defaults.fetch_timeout" in message
    assert "defaults.min_hn_points" in message


# --------------------------------------------------------------------------- #
# Secrets — environment only, never YAML (NEVER rule 16)
# --------------------------------------------------------------------------- #


def test_get_secret_reads_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_realtokenvalue")
    secret = get_secret("GITHUB_TOKEN")
    assert secret is not None
    assert secret.get_secret_value() == "ghp_realtokenvalue"


def test_get_secret_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # None, not an exception: GitHub works unauthenticated at 60 req/hr, so the
    # caller decides whether the credential is optional or fatal.
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert get_secret("GITHUB_TOKEN") is None


def test_get_secret_treats_a_blank_value_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "   ")
    assert get_secret("GITHUB_TOKEN") is None


def test_get_secret_never_logs_the_secret_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_realtokenvalue")
    with caplog.at_level(logging.DEBUG, logger="signalforge.config"):
        get_secret("GITHUB_TOKEN")
    assert "ghp_realtokenvalue" not in caplog.text


def test_secret_is_not_stringified_by_repr_or_str(monkeypatch: pytest.MonkeyPatch) -> None:
    # SecretStr's whole job: an accidental log line or traceback renders stars.
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_realtokenvalue")
    secret = get_secret("GITHUB_TOKEN")
    assert "ghp_realtokenvalue" not in repr(secret)
    assert "ghp_realtokenvalue" not in str(secret)


def test_secrets_are_not_settable_from_yaml(tmp_path: Path) -> None:
    # A token in git-tracked YAML is the leak NEVER rule 16 exists to prevent.
    # SourcesConfig has no field to receive one, so extra="forbid" rejects it.
    write_sources(tmp_path, MINIMAL_SOURCES_YAML + "github_token: ghp_leaked\n")
    with pytest.raises(ConfigError, match="github_token"):
        load_sources(tmp_path)


def test_config_error_for_an_inline_token_does_not_echo_the_token(tmp_path: Path) -> None:
    # The error must name the mistake without copying the credential into a log,
    # a traceback, or a CI transcript.
    write_sources(
        tmp_path,
        MINIMAL_SOURCES_YAML + "github:\n  token_env: ghp_abc123realtoken\n",
    )
    with pytest.raises(ConfigError) as excinfo:
        load_sources(tmp_path)

    message = str(excinfo.value)
    assert "ghp_abc123realtoken" not in message
    assert "NAME of an environment variable" in message


def test_token_env_must_name_an_env_var(tmp_path: Path) -> None:
    write_sources(
        tmp_path,
        MINIMAL_SOURCES_YAML + "github:\n  token_env: github_pat_11ABCDEF\n",
    )
    with pytest.raises(ConfigError, match="NAME of an environment variable"):
        load_sources(tmp_path)


def test_plausible_env_var_names_are_accepted(tmp_path: Path) -> None:
    write_sources(tmp_path, MINIMAL_SOURCES_YAML + "github:\n  token_env: MY_GH_TOKEN_2\n")
    config = load_sources(tmp_path)
    assert config.github is not None
    assert config.github.token_env == "MY_GH_TOKEN_2"


@pytest.mark.parametrize("name", ["GITHUB_PAT", "GITHUB_TOKEN", "GH_PAT", "MY_GITHUB_TOKEN"])
def test_env_var_names_resembling_a_token_prefix_are_accepted(tmp_path: Path, name: str) -> None:
    # The guard matches token SHAPES (`<prefix>_<body>`), not name prefixes, so
    # `GITHUB_PAT` — a natural env-var NAME that lowercases to `github_pat` — must
    # not be mistaken for a pasted `github_pat_<body>` token. Regression for the
    # false-positive the old bare-prefix check produced. (A name that literally
    # begins `ghp_`/`gho_` is still rejected — that shape is indistinguishable
    # from a real token and no one names an env var that.)
    write_sources(tmp_path, MINIMAL_SOURCES_YAML + f"github:\n  token_env: {name}\n")
    config = load_sources(tmp_path)
    assert config.github is not None
    assert config.github.token_env == name


@pytest.mark.parametrize(
    "token",
    ["ghp_abc123", "gho_abc123", "ghu_abc123", "ghs_abc123", "ghr_abc123", "github_pat_11ABCDEF"],
)
def test_pasted_github_tokens_are_rejected(tmp_path: Path, token: str) -> None:
    write_sources(tmp_path, MINIMAL_SOURCES_YAML + f"github:\n  token_env: {token}\n")
    with pytest.raises(ConfigError, match="NAME of an environment variable"):
        load_sources(tmp_path)


def test_secret_str_is_the_declared_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    assert isinstance(get_secret("GITHUB_TOKEN"), SecretStr)


@pytest.mark.parametrize("knob", ["daily_max_per_source", "daily_max_per_github_repo"])
@pytest.mark.parametrize("bad", [0, -1])
def test_crowding_limits_reject_meaningless_values(knob: str, bad: int) -> None:
    # A limit of 0 would render an empty digest rather than "no limit" — the
    # off switch is omitting the key, so the model must not accept a count
    # that silently empties the report.
    values = {
        "weekly_min_signal": 3,
        "weekly_min_relevance": 3,
        "weekly_min_total": 10,
        "daily_max_items": 15,
        knob: bad,
    }
    with pytest.raises(ValueError, match=knob):
        Thresholds(**values)


@pytest.mark.parametrize("knob", ["daily_max_per_source", "daily_max_per_github_repo"])
def test_crowding_limits_are_optional(knob: str) -> None:
    # Unlike the four required thresholds, these default to "no limit" — an
    # existing interests.yaml stays valid and behaves exactly as before.
    values = {
        "weekly_min_signal": 3,
        "weekly_min_relevance": 3,
        "weekly_min_total": 10,
        "daily_max_items": 15,
    }
    assert getattr(Thresholds(**values), knob) is None


# --------------------------------------------------------------------------- #
# settings.yaml — app & locale (the timezone that resolves the reader's day)
# --------------------------------------------------------------------------- #


def test_settings_defaults_to_utc() -> None:
    # UTC is the safe global default: an operator who never sets a zone gets
    # correct (if not local) behaviour, not a crash.
    settings = SettingsConfig()
    assert settings.timezone == "UTC"
    assert settings.tzinfo.key == "UTC"


def test_settings_accepts_a_valid_iana_zone() -> None:
    settings = SettingsConfig(timezone="Australia/Sydney")
    assert settings.tzinfo.key == "Australia/Sydney"


@pytest.mark.parametrize("bad", ["Mars/Phobos", "GMT+10", "Sydney", ""])
def test_settings_rejects_an_unknown_zone_at_load(bad: str) -> None:
    # A typo'd zone silently falling back to UTC is the invisible-config failure
    # _StrictModel exists to prevent — it must raise, naming the field.
    with pytest.raises(ValueError, match="timezone"):
        SettingsConfig(timezone=bad)


def test_settings_vault_dir_defaults_to_repo_vault() -> None:
    # Unset → the historical location, so an existing install's digests keep
    # landing in the repo's vault/ (CLAUDE.md §4 — the default lives in config).
    assert SettingsConfig().vault_dir == Path("vault")


def test_settings_vault_dir_expands_user_and_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    # A committed example can carry a portable ~/... path and a real file a
    # /mnt/c/... one, without an absolute home leaking into the repo.
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("OBSIDIAN", "/mnt/c/Users/tester/Obsidian")
    # str is coerced to Path by pydantic after the before-validator expands it;
    # the annotation is Path, so the deliberate str inputs are ignored for mypy.
    expanded_home = SettingsConfig(vault_dir="~/Obsidian/SignalForge")  # type: ignore[arg-type]
    assert expanded_home.vault_dir == Path("/home/tester/Obsidian/SignalForge")
    expanded_var = SettingsConfig(vault_dir="$OBSIDIAN/SignalForge")  # type: ignore[arg-type]
    assert expanded_var.vault_dir == Path("/mnt/c/Users/tester/Obsidian/SignalForge")


def test_settings_vault_dir_rejects_unset_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unset $VAR is left verbatim by expandvars; a nonsense path silently
    # misfiling digests is the invisible-misconfig _StrictModel guards against —
    # so it must raise at load, not defer to the first digest.
    monkeypatch.delenv("DEFINITELY_UNSET_VAULT_VAR", raising=False)
    with pytest.raises(ValueError, match="unexpanded environment variable"):
        SettingsConfig(vault_dir="$DEFINITELY_UNSET_VAULT_VAR/vault")  # type: ignore[arg-type]


def test_settings_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="locale"):
        SettingsConfig(locale="en_AU")  # type: ignore[call-arg]


def test_load_settings_missing_file_is_not_an_error(tmp_path: Path) -> None:
    # Unlike sources/interests, a wholly-absent settings.yaml is tolerated and
    # yields UTC defaults — existing installs predating the file still run.
    assert not (tmp_path / SETTINGS_FILENAME).exists()
    assert load_settings(tmp_path).timezone == "UTC"


def test_load_settings_present_file_is_validated_strictly(tmp_path: Path) -> None:
    (tmp_path / SETTINGS_FILENAME).write_text("timezone: Pluto/Nowhere\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="timezone"):
        load_settings(tmp_path)


def test_load_settings_reads_the_zone(tmp_path: Path) -> None:
    (tmp_path / SETTINGS_FILENAME).write_text("timezone: America/New_York\n", encoding="utf-8")
    assert load_settings(tmp_path).tzinfo.key == "America/New_York"


def test_shipped_settings_yaml_parses(repo_config_dir: Path) -> None:
    # The shipped file must load and name a real zone — a broken settings.yaml
    # would break every digest's day boundary.
    settings = load_settings(repo_config_dir)
    assert settings.tzinfo is not None


# --------------------------------------------------------------------------- #
# delivery — the outbound push channel (DESIGN §13.2)
# --------------------------------------------------------------------------- #

DELIVERY_YAML = """
timezone: Australia/Brisbane
delivery:
  email:
    enabled: true
    api_key_env: RESEND_API_KEY
    from_address: SignalForge <digest@example.com>
    to:
      - reader@example.com
"""


def test_settings_without_a_delivery_block_delivers_nowhere() -> None:
    # The historical shape. Delivery is additive: absent means the vault is the
    # only destination, which must stay the default forever.
    assert SettingsConfig().delivery.email is None


def test_load_settings_reads_the_email_channel(tmp_path: Path) -> None:
    (tmp_path / SETTINGS_FILENAME).write_text(DELIVERY_YAML, encoding="utf-8")

    email = load_settings(tmp_path).delivery.email

    assert email is not None
    assert email.enabled is True
    assert email.api_key_env == "RESEND_API_KEY"
    assert email.to == ["reader@example.com"]


def test_email_channel_requires_an_explicit_enabled() -> None:
    """An off-switch you reach by deleting the block is not an off-switch."""
    with pytest.raises(ValueError, match="enabled"):
        EmailChannelConfig(  # type: ignore[call-arg]
            api_key_env="RESEND_API_KEY",
            from_address="a@example.com",
            to=["b@example.com"],
        )


def test_email_channel_rejects_a_pasted_api_key() -> None:
    """The likely mistake: the key value where the env-var name belongs.

    Shaped like a real Resend key — lowercase `re_` prefix, ~35 characters — because
    that is what the validator discriminates on.
    """
    with pytest.raises(ValueError, match="NAME of an environment variable"):
        EmailChannelConfig(
            enabled=True,
            api_key_env="re_1a2B3c4D_5e6F7g8H9i0J1k2L3m4N5o6P",
            from_address="a@example.com",
            to=["b@example.com"],
        )


def test_email_channel_accepts_an_env_var_name_that_merely_starts_like_a_key() -> None:
    # `RE_DIGEST_KEY` is a legitimate name; only the `re_<body>` token shape is out.
    assert (
        EmailChannelConfig(
            enabled=True,
            api_key_env="RE_DIGEST_KEY",
            from_address="a@example.com",
            to=["b@example.com"],
        ).api_key_env
        == "RE_DIGEST_KEY"
    )


@pytest.mark.parametrize(
    "address",
    [
        "not-an-address",
        "@example.com",
        "nobody@",
        "nobody@localhost",
        "nobody@.com",
        "nobody@example..com",
        # `parseaddr` repairs these by deleting interior whitespace, yielding an
        # address the operator never wrote. Validating the repair and storing the
        # original is exactly the refuse-don't-repair rule (DESIGN §13.1).
        "nobody@example.com x",
        "nobody @example.com",
    ],
)
def test_email_channel_rejects_an_unparseable_address(address: str) -> None:
    with pytest.raises(ValueError, match="email address"):
        EmailChannelConfig(
            enabled=True,
            api_key_env="RESEND_API_KEY",
            from_address=address,
            to=["b@example.com"],
        )


@pytest.mark.parametrize("field", ["from_address", "to"])
def test_email_channel_refuses_a_control_character_in_an_address(field: str) -> None:
    """Header injection, refused rather than repaired (DESIGN §13.1).

    These strings end up in message headers. A `\\r\\n` in one is not a formatting
    quirk, it is a second header — and an address is an identity field, so
    silently rewriting it would change who the mail goes to.
    """
    injected = "ok@example.com\r\nBcc: attacker@evil.example"
    kwargs: dict[str, object] = {
        "enabled": True,
        "api_key_env": "RESEND_API_KEY",
        "from_address": "a@example.com",
        "to": ["b@example.com"],
    }
    kwargs[field] = injected if field == "from_address" else [injected]

    with pytest.raises(ValueError, match="control character"):
        EmailChannelConfig(**kwargs)  # type: ignore[arg-type]


def test_email_channel_accepts_a_display_name_form() -> None:
    """The containment check must not reject the form the example documents."""
    channel = EmailChannelConfig(
        enabled=True,
        api_key_env="RESEND_API_KEY",
        from_address="SignalForge <digest@example.com>",
        to=["Reader <reader@example.com>"],
    )
    assert channel.from_address == "SignalForge <digest@example.com>"


def test_email_channel_rejects_two_addresses_in_one_string() -> None:
    """A comma-separated pair would be a smuggled second recipient."""
    with pytest.raises(ValueError, match="email address"):
        EmailChannelConfig(
            enabled=True,
            api_key_env="RESEND_API_KEY",
            from_address="a@example.com",
            to=["b@example.com, attacker@evil.example"],
        )


def test_email_channel_requires_at_least_one_recipient() -> None:
    with pytest.raises(ValueError, match="to"):
        EmailChannelConfig(
            enabled=True,
            api_key_env="RESEND_API_KEY",
            from_address="a@example.com",
            to=[],
        )


def test_email_channel_rejects_unknown_keys() -> None:
    # `extra="forbid"` reaches the nested block too, so a typo'd delivery key is
    # a startup error rather than a setting that silently does nothing.
    with pytest.raises(ValueError, match="subject_prefix"):
        EmailChannelConfig(  # type: ignore[call-arg]
            enabled=True,
            api_key_env="RESEND_API_KEY",
            from_address="a@example.com",
            to=["b@example.com"],
            subject_prefix="[SF] ",
        )


def test_a_provider_key_is_rejected_rather_than_silently_ignored(tmp_path: Path) -> None:
    """`provider:` was removed rather than kept as a field nothing reads.

    `extra="forbid"` turns a stale line in someone's `settings.yaml` into a startup
    error naming the key, which is the whole point of strict config — and it pins
    that the field is gone deliberately, not by accident.
    """
    (tmp_path / SETTINGS_FILENAME).write_text(
        DELIVERY_YAML + "    provider: resend\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="provider"):
        load_settings(tmp_path)
