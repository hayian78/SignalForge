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
    ConfigError,
    Secrets,
    SourceDefaults,
    Thresholds,
    get_secret,
    load_interests,
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
    priority_topics: [agents.mcp, engineering.code-gen, engineering.context,
                      models.local, retrieval.rag]
    interests: [python, fastapi, sqlite, duckdb, claude-code, trading-systems, local-first]
    stack: [python, typescript, fastapi, sqlite, postgres, docker, wsl]
    learning_goals: [agent memory architectures, production llm evaluation]
    architecture_philosophy: >
      Local-first, deterministic pipelines, low operational cost, monolith-by-default,
      boring technology, human-in-the-loop.
    ignore:
      topics: [crypto, web3, model-release-hype]
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

    assert "agents.mcp" in config.priority_topics
    assert config.stack == [
        "python",
        "typescript",
        "fastapi",
        "sqlite",
        "postgres",
        "docker",
        "wsl",
    ]
    assert config.learning_goals == [
        "agent memory architectures",
        "production llm evaluation",
    ]
    assert "Local-first" in config.architecture_philosophy
    assert config.ignore.topics == ["crypto", "web3", "model-release-hype"]
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
    assert not config.github.token_env.lower().startswith(("ghp", "github_pat"))


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


def test_secrets_model_reads_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    secrets = Secrets(_env_file=None)  # type: ignore[call-arg]

    assert secrets.github_token is not None
    assert secrets.github_token.get_secret_value() == "ghp_x"
    assert secrets.anthropic_api_key is not None
    assert secrets.anthropic_api_key.get_secret_value() == "sk-ant-x"


def test_secrets_default_to_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    secrets = Secrets(_env_file=None)  # type: ignore[call-arg]
    assert (secrets.github_token, secrets.anthropic_api_key) == (None, None)


def test_secrets_repr_does_not_leak_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
    secrets = Secrets(_env_file=None)  # type: ignore[call-arg]
    assert "sk-ant-supersecret" not in repr(secrets)
    assert "sk-ant-supersecret" not in str(secrets.model_dump())


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
