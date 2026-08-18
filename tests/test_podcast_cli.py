"""Integration tests for the `podcast` CLI command and its wiring into
`daily`/`status` (Stage 6, DESIGN §13.3).

Boundaries faked the same way the rest of this suite fakes them (CLAUDE.md §8,
NEVER rule 13): LLM triage via `signalforge.llm.run_triage_batch`, the podcast
script call via `signalforge.synth.podcast.run_podcast_script` (the same seam
`tests/synth/test_podcast.py` uses), and every HTTP boundary — ingest's RSS
fetch, the deep-read article fetch, TTS, and R2 — via respx. ffmpeg and the mp3
concat step are faked the same way `tests/deliver/test_podcast.py` fakes them,
so this suite runs identically with or without a real ffmpeg binary.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner, Result

from signalforge.cli import app
from signalforge.db import connection
from signalforge.deliver.tts import tts_spend_usd
from signalforge.feedback import checkbox_marker
from signalforge.llm import PodcastScript, PodcastScriptResult, TriageBatchResult, TriageResult

runner = CliRunner()

_LOG_LEVEL: list[str] = ["--log-level", "WARNING"]

FEED_URL = "https://example.com/alpha/feed.xml"
ARTICLE_URL = "https://example.com/alpha/post-1"
TTS_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"
R2_ENDPOINT = "https://account123.r2.cloudflarestorage.com"
BUCKET = "signalforge-podcast"
PRESENTER_A = "Alex"
PRESENTER_B = "Sam"

FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>alpha blog</title>
  <entry>
    <title>alpha post</title>
    <link href="https://example.com/alpha/post-1"/>
    <id>urn:alpha:1</id>
    <updated>2026-07-15T12:30:00Z</updated>
    <summary>A short summary of the post.</summary>
  </entry>
</feed>
"""

_ARTICLE_HTML = (
    "<html><body><article>"
    + "".join(
        f"<p>Paragraph {n} of the article body, long enough for trafilatura to "
        "extract cleanly and reliably every time this test runs.</p>\n"
        for n in range(20)
    )
    + "</article></body></html>"
).encode()


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """The CLI owns global logging config; restore it after every test."""
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:
        yield
    finally:
        root.setLevel(level)
        root.handlers[:] = handlers


@pytest.fixture(autouse=True)
def _fake_ffmpeg_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")


@pytest.fixture(autouse=True)
def _fake_concat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("signalforge.deliver.tts._concat_mp3", lambda chunks: b"|".join(chunks))


@pytest.fixture(autouse=True)
def _secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real env vars, so `get_secret` never reaches a developer's real `.env`."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret-key")


def _sources_yaml() -> str:
    return (
        "defaults:\n"
        "  fetch_timeout: 5\n"
        "  min_hn_points: 80\n"
        "  max_summary_chars: 4000\n"
        # Huge on purpose: the inline feed carries fixed dates.
        "  max_item_age_days: 3650\n"
        "rss:\n"
        f"  - id: alpha\n    url: {FEED_URL}\n"
    )


def _interests_yaml(*, podcast_top_n: int | None) -> str:
    thresholds: dict[str, int] = {
        "weekly_min_signal": 3,
        "weekly_min_relevance": 3,
        "weekly_min_total": 10,
        "daily_max_items": 15,
    }
    if podcast_top_n is not None:
        thresholds["podcast_top_n"] = podcast_top_n
    thresholds_line = "{" + ", ".join(f"{k}: {v}" for k, v in thresholds.items()) + "}"
    return (
        "priority_topics: [agents.mcp]\n"
        "interests: [python]\n"
        "stack: [python]\n"
        "learning_goals: []\n"
        "architecture_philosophy: 'Local-first.'\n"
        "ignore:\n"
        "  topics: [crypto]\n"
        "  people: []\n"
        "  repos: []\n"
        f"thresholds: {thresholds_line}\n"
    )


def _settings_yaml(*, with_podcast: bool) -> str:
    base = "timezone: UTC\n"
    if not with_podcast:
        return base
    return base + (
        "delivery:\n"
        "  podcast:\n"
        "    enabled: true\n"
        "    tts_api_key_env: OPENROUTER_API_KEY\n"
        "    tts_model: hexgrad/kokoro-82m\n"
        "    voice_a: af_bella\n"
        "    voice_b: am_adam\n"
        f"    presenter_a: {PRESENTER_A}\n"
        f"    presenter_b: {PRESENTER_B}\n"
        f"    r2_endpoint: {R2_ENDPOINT}\n"
        f"    r2_bucket: {BUCKET}\n"
        "    r2_access_key_env: R2_ACCESS_KEY_ID\n"
        "    r2_secret_key_env: R2_SECRET_ACCESS_KEY\n"
        "    public_base_url: https://pub-example.r2.dev/prefix\n"
        "    retention_episodes: 30\n"
        "    feed_title: SignalForge Daily\n"
    )


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    path = tmp_path / "config"
    path.mkdir()
    (path / "sources.yaml").write_text(_sources_yaml(), encoding="utf-8")
    (path / "interests.yaml").write_text(_interests_yaml(podcast_top_n=1), encoding="utf-8")
    (path / "settings.yaml").write_text(_settings_yaml(with_podcast=True), encoding="utf-8")
    return path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "signalforge.db"


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "http_cache"


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    return tmp_path / "vault"


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    return tmp_path / "audio"


def _run(*args: str) -> Result:
    return runner.invoke(app, [*_LOG_LEVEL, *args])


def _ingest(config_dir: Path, db_path: Path, cache_dir: Path) -> Result:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"etag": '"a-1"'})
        )
        return _run(
            "ingest",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
        )


def _score(config_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Result:
    batch_result = TriageBatchResult(
        results={
            1: TriageResult(triage="keep", signal=5, relevance=5, novelty=4, reasoning="Good.")
        },
        input_tokens=42,
        output_tokens=7,
    )
    monkeypatch.setattr("signalforge.llm.run_triage_batch", lambda *a, **k: batch_result)
    return _run("score", "--config-dir", str(config_dir), "--db", str(db_path))


def _script_dict(item_id: int = 1) -> dict[str, Any]:
    def turn(speaker: str, text: str) -> dict[str, str]:
        return {"speaker": speaker, "text": text}

    return {
        "intro_turns": [
            turn("A", "Good morning, welcome to the show."),
            turn("B", "Great to be here."),
        ],
        "segments": [
            {
                "item_ids": [item_id],
                "turns": [
                    turn("A", "Here's today's top story."),
                    turn("B", "Sounds interesting."),
                ],
            }
        ],
        "outro_turns": [turn("A", "That's all for today."), turn("B", "See you tomorrow.")],
    }


def _fake_run_podcast_script(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _fake(stable_system_prefix: str, **kwargs: Any) -> PodcastScriptResult:
        calls.append({"stable_system_prefix": stable_system_prefix, **kwargs})
        return PodcastScriptResult(
            script=PodcastScript.model_validate(_script_dict()),
            input_tokens=1_000,
            output_tokens=500,
            dropped_item_count=0,
            sent_item_ids=(1,),
        )

    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _fake)
    return calls


def _mock_r2_success(mock: respx.MockRouter) -> None:
    mock.put(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(return_value=httpx.Response(200))
    mock.get(f"{R2_ENDPOINT}/{BUCKET}").mock(
        return_value=httpx.Response(
            200,
            content=b'<?xml version="1.0"?><ListBucketResult '
            b'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"></ListBucketResult>',
        )
    )
    mock.delete(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(return_value=httpx.Response(204))


def _podcast(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    *,
    r2_down: bool = False,
    extra_args: list[str] | None = None,
) -> Result:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(ARTICLE_URL).mock(return_value=httpx.Response(200, content=_ARTICLE_HTML))
        mock.post(TTS_ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio-chunk"))
        if r2_down:
            # A plain non-retryable status (403), not a transport error: the
            # storage.py retry policy treats 500/502/503/504 and transport
            # errors as retryable with real exponential backoff, which would
            # make this test sleep for real seconds it does not need to.
            mock.put(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(
                return_value=httpx.Response(403)
            )
            mock.get(f"{R2_ENDPOINT}/{BUCKET}").mock(return_value=httpx.Response(403))
        else:
            _mock_r2_success(mock)
        return _run(
            "podcast",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
            "--vault-dir",
            str(vault_dir),
            "--audio-dir",
            str(audio_dir),
            *(extra_args or []),
        )


def _scored_date(db_path: Path) -> str:
    with connection(db_path) as conn:
        scored_at = str(
            conn.execute("SELECT scored_at FROM scores WHERE item_id = 1").fetchone()[0]
        )
    return scored_at[:10]


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_podcast_command_writes_a_script_synthesizes_and_publishes(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    target_date = _scored_date(db_path)

    calls = _fake_run_podcast_script(monkeypatch)
    result = _podcast(
        config_dir, db_path, cache_dir, vault_dir, audio_dir, extra_args=["--date", target_date]
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1

    script_file = vault_dir / "podcast" / f"{target_date}.md"
    assert script_file.exists()
    assert "Here's today's top story." in script_file.read_text(encoding="utf-8")

    assert (audio_dir / f"signalforge-{target_date}.mp3").exists()

    with connection(db_path) as conn:
        run = conn.execute(
            "SELECT status, tts_characters, llm_input_tokens FROM runs WHERE kind = 'podcast'"
        ).fetchone()
        delivery = conn.execute(
            "SELECT channel, report_kind FROM deliveries WHERE channel = 'podcast'"
        ).fetchone()

    assert run["status"] == "ok"
    assert run["tts_characters"] > 0
    assert run["llm_input_tokens"] == 1_000
    assert delivery is not None
    assert delivery["report_kind"] == "podcast"


def test_a_lost_episode_is_recorded_as_partial_with_the_reason_in_runs_errors(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The silence that made this bug expensive. Two real episodes were lost
    to output-ceiling truncation across four days and both runs closed as
    `status="ok"` with an empty `errors` column — invisible to
    `signalforge status` and to the next digest, which CLAUDE.md §7 makes the
    monitoring channel. The episode is still lost here; what must not be lost
    is the *record* of it.
    """
    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    target_date = _scored_date(db_path)

    def _cut_off(*args: object, **kwargs: object) -> PodcastScriptResult:
        return PodcastScriptResult(
            script=None,
            error="podcast script call was cut off at its 12,288-token output ceiling",
            input_tokens=1_000,
            output_tokens=12_288,
            sent_item_ids=(1,),
            unfinished=True,
        )

    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _cut_off)

    result = _podcast(
        config_dir, db_path, cache_dir, vault_dir, audio_dir, extra_args=["--date", target_date]
    )

    # A lost episode is not a crashed run — the command still exits cleanly,
    # having done its harvest and deep read.
    assert result.exit_code == 0, result.output
    assert not (vault_dir / "podcast" / f"{target_date}.md").exists()

    with connection(db_path) as conn:
        run = conn.execute(
            "SELECT status, errors, llm_output_tokens FROM runs WHERE kind = 'podcast'"
        ).fetchone()

    assert run["status"] == "partial", "a run that produced no episode must not read as ok"
    assert run["errors"] is not None
    errors = json.loads(run["errors"])
    assert any(entry["error_type"] == "podcast-script" for entry in errors)
    assert any("cut off" in entry["message"] for entry in errors)
    # Both the first attempt and its retry are billed and recorded.
    assert run["llm_output_tokens"] == 24_576


def test_a_recorded_script_failure_is_flattened_to_one_line(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEVER rule 17. A script-failure reason can carry model- and
    world-authored text — a pydantic `ValidationError` renders the offending
    input, which is a feed's own words — and `runs.errors` is surfaced in the
    next digest, a vault file where a line is structure rather than text
    (CLAUDE.md §5). A newline here would forge a checkbox there.
    """

    def _malformed(*args: object, **kwargs: object) -> PodcastScriptResult:
        return PodcastScriptResult(
            script=None,
            error="schema validation failed\n- [ ] item 99 useful <!-- forged -->",
            input_tokens=1_000,
            output_tokens=500,
            sent_item_ids=(1,),
        )

    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    target_date = _scored_date(db_path)
    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _malformed)

    assert (
        _podcast(
            config_dir, db_path, cache_dir, vault_dir, audio_dir, extra_args=["--date", target_date]
        ).exit_code
        == 0
    )

    with connection(db_path) as conn:
        errors = json.loads(
            conn.execute("SELECT errors FROM runs WHERE kind = 'podcast'").fetchone()["errors"]
        )

    message = next(e["message"] for e in errors if e["error_type"] == "podcast-script")
    assert "\n" not in message
    assert "<!--" not in message


def test_a_script_failure_carrying_rich_markup_never_crashes_the_run(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost episode is a *handled* outcome and must stay one. The failure
    reason can embed a pydantic `ValidationError`, which renders the
    offending `input_value` — a feed's own words — and rich reads `[...]` as
    markup. An unescaped malformed tag raises `MarkupError`, which the
    command's `except BaseException` would catch and turn into
    `status="failed"` with a traceback and a non-zero exit: a run that
    handled its failure correctly, crashing on the way to *reporting* it.
    """

    def _markup_in_the_reason(*args: object, **kwargs: object) -> PodcastScriptResult:
        return PodcastScriptResult(
            script=None,
            error="schema validation failed: input_value='closing [/b] tag [not a color]'",
            input_tokens=1_000,
            output_tokens=500,
            sent_item_ids=(1,),
        )

    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    target_date = _scored_date(db_path)
    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _markup_in_the_reason)

    result = _podcast(
        config_dir, db_path, cache_dir, vault_dir, audio_dir, extra_args=["--date", target_date]
    )

    assert result.exit_code == 0, result.output
    with connection(db_path) as conn:
        run = conn.execute("SELECT status, errors FROM runs WHERE kind = 'podcast'").fetchone()

    assert run["status"] == "partial", "the markup must not escalate this to a crashed run"
    assert "[/b]" in json.loads(run["errors"])[0]["message"]


def test_a_recorded_script_failure_is_capped_in_length(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`runs.errors` is a record, not a log. A `ValidationError` over a
    truncated 12k-token response renders the offending input and runs to
    multiple KB, and this string goes verbatim into the errors blob and the
    status table — same cap `ingest/probe.py` applies for the same reason.
    """

    def _enormous(*args: object, **kwargs: object) -> PodcastScriptResult:
        return PodcastScriptResult(
            script=None,
            error="schema validation failed: " + ("x" * 20_000),
            input_tokens=1_000,
            output_tokens=500,
            sent_item_ids=(1,),
        )

    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    target_date = _scored_date(db_path)
    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _enormous)

    assert (
        _podcast(
            config_dir, db_path, cache_dir, vault_dir, audio_dir, extra_args=["--date", target_date]
        ).exit_code
        == 0
    )

    with connection(db_path) as conn:
        errors = json.loads(
            conn.execute("SELECT errors FROM runs WHERE kind = 'podcast'").fetchone()["errors"]
        )
    assert len(errors[0]["message"]) <= 200


def test_podcast_command_second_run_is_fully_idempotent(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    target_date = _scored_date(db_path)

    calls = _fake_run_podcast_script(monkeypatch)
    first = _podcast(
        config_dir, db_path, cache_dir, vault_dir, audio_dir, extra_args=["--date", target_date]
    )
    assert first.exit_code == 0, first.output
    assert len(calls) == 1

    script_file = vault_dir / "podcast" / f"{target_date}.md"
    first_render = script_file.read_text(encoding="utf-8")

    def _must_not_be_called(*args: object, **kwargs: object) -> PodcastScriptResult:
        raise AssertionError("nothing should be sent to the LLM on the second pass")

    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _must_not_be_called)

    # No TTS/R2 routes registered at all on the second pass: `deliver_podcast`
    # must short-circuit on `delivery_exists` before making any HTTP call.
    with respx.mock(assert_all_called=False):
        second = _run(
            "podcast",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
            "--vault-dir",
            str(vault_dir),
            "--audio-dir",
            str(audio_dir),
            "--date",
            target_date,
        )
    assert second.exit_code == 0, second.output
    assert script_file.read_text(encoding="utf-8") == first_render

    with connection(db_path) as conn:
        podcast_runs = conn.execute(
            "SELECT tts_characters, llm_input_tokens FROM runs WHERE kind = 'podcast' ORDER BY id"
        ).fetchall()
        delivery_count = conn.execute(
            "SELECT COUNT(*) AS n FROM deliveries WHERE channel = 'podcast'"
        ).fetchone()["n"]

    assert len(podcast_runs) == 2
    assert (podcast_runs[1]["tts_characters"], podcast_runs[1]["llm_input_tokens"]) == (0, 0)
    assert delivery_count == 1


# --------------------------------------------------------------------------- #
# off-switches — a clean no-op, not a crash
# --------------------------------------------------------------------------- #


def test_podcast_top_n_unset_is_a_clean_noop(
    tmp_path: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "sources.yaml").write_text(_sources_yaml(), encoding="utf-8")
    (config_dir / "interests.yaml").write_text(
        _interests_yaml(podcast_top_n=None), encoding="utf-8"
    )
    (config_dir / "settings.yaml").write_text(_settings_yaml(with_podcast=True), encoding="utf-8")

    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0

    def _must_not_be_called(*args: object, **kwargs: object) -> PodcastScriptResult:
        raise AssertionError("the podcast channel is off; nothing should call the LLM")

    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _must_not_be_called)

    with respx.mock(assert_all_called=False):
        result = _run(
            "podcast",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
            "--vault-dir",
            str(vault_dir),
            "--audio-dir",
            str(audio_dir),
        )

    assert result.exit_code == 0, result.output
    assert not (vault_dir / "podcast").exists()

    with connection(db_path) as conn:
        run = conn.execute("SELECT status FROM runs WHERE kind = 'podcast'").fetchone()
    assert run["status"] == "ok"


def test_podcast_without_a_configured_delivery_block_is_a_clean_noop(
    tmp_path: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "sources.yaml").write_text(_sources_yaml(), encoding="utf-8")
    (config_dir / "interests.yaml").write_text(_interests_yaml(podcast_top_n=1), encoding="utf-8")
    (config_dir / "settings.yaml").write_text(_settings_yaml(with_podcast=False), encoding="utf-8")

    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0

    def _must_not_be_called(*args: object, **kwargs: object) -> PodcastScriptResult:
        raise AssertionError("no delivery.podcast: block; nothing should call the LLM")

    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _must_not_be_called)

    with respx.mock(assert_all_called=False):
        result = _run(
            "podcast",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
            "--vault-dir",
            str(vault_dir),
            "--audio-dir",
            str(audio_dir),
        )

    assert result.exit_code == 0, result.output
    assert not (vault_dir / "podcast").exists()


# --------------------------------------------------------------------------- #
# NEVER rule 19 — a dead channel never fails the run
# --------------------------------------------------------------------------- #


def test_a_dead_r2_is_a_recorded_error_not_a_failed_run(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    target_date = _scored_date(db_path)

    _fake_run_podcast_script(monkeypatch)
    result = _podcast(
        config_dir,
        db_path,
        cache_dir,
        vault_dir,
        audio_dir,
        r2_down=True,
        extra_args=["--date", target_date],
    )

    # The script is on disk either way (DESIGN §13.2's vault-first invariant);
    # a dead R2 must not fail the command (CLAUDE.md NEVER rule 19).
    assert result.exit_code == 0, result.output
    assert (vault_dir / "podcast" / f"{target_date}.md").exists()

    with connection(db_path) as conn:
        run = conn.execute("SELECT status, errors FROM runs WHERE kind = 'podcast'").fetchone()
        delivery = conn.execute(
            "SELECT 1 AS x FROM deliveries WHERE channel = 'podcast'"
        ).fetchone()

    assert run["status"] == "ok"
    assert run["errors"] is not None
    assert delivery is None


# --------------------------------------------------------------------------- #
# daily — the fifth isolated step
# --------------------------------------------------------------------------- #


def test_daily_runs_podcast_as_a_fifth_isolated_step(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "signalforge.llm.run_triage_batch",
        lambda *a, **k: TriageBatchResult(
            results={
                1: TriageResult(triage="keep", signal=5, relevance=5, novelty=4, reasoning="Good.")
            },
            input_tokens=1,
            output_tokens=1,
        ),
    )
    _fake_run_podcast_script(monkeypatch)

    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"etag": '"a-1"'})
        )
        mock.get(ARTICLE_URL).mock(return_value=httpx.Response(200, content=_ARTICLE_HTML))
        mock.post(TTS_ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio-chunk"))
        _mock_r2_success(mock)
        result = _run(
            "daily",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
            "--vault-dir",
            str(vault_dir),
            "--audio-dir",
            str(audio_dir),
        )

    assert result.exit_code == 0, result.output
    with connection(db_path) as conn:
        kinds = [row[0] for row in conn.execute("SELECT kind FROM runs ORDER BY id")]
    assert kinds == ["ingest", "score", "daily", "podcast"]

    today_scripts = list((vault_dir / "podcast").glob("*.md"))
    assert len(today_scripts) == 1


def test_daily_isolates_a_podcast_step_failure_from_the_already_written_digest(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`digest` runs before `podcast` in the `daily` chain — a podcast step
    that dies outright (not merely a delivery failure, which never fails the
    run per NEVER rule 19, but a genuine exception from the one part of
    `podcast` that is not delivery — script generation) must not erase or
    block that already-written digest file.
    """
    monkeypatch.setattr(
        "signalforge.llm.run_triage_batch",
        lambda *a, **k: TriageBatchResult(
            results={
                1: TriageResult(triage="keep", signal=5, relevance=5, novelty=4, reasoning="Good.")
            },
            input_tokens=1,
            output_tokens=1,
        ),
    )

    def _broken_script_call(*args: object, **kwargs: object) -> PodcastScriptResult:
        raise RuntimeError("the script model call blew up before any response came back")

    monkeypatch.setattr("signalforge.synth.podcast.run_podcast_script", _broken_script_call)

    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"etag": '"a-1"'})
        )
        mock.get(ARTICLE_URL).mock(return_value=httpx.Response(200, content=_ARTICLE_HTML))
        result = _run(
            "daily",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
            "--vault-dir",
            str(vault_dir),
            "--audio-dir",
            str(audio_dir),
        )

    assert result.exit_code != 0

    today_digests = list((vault_dir / "daily").glob("*.md"))
    assert len(today_digests) == 1
    assert "alpha post" in today_digests[0].read_text(encoding="utf-8")

    with connection(db_path) as conn:
        kinds = [row[0] for row in conn.execute("SELECT kind FROM runs ORDER BY id")]
    assert kinds == ["ingest", "score", "daily", "podcast"]


# --------------------------------------------------------------------------- #
# status — the TTS spend line
# --------------------------------------------------------------------------- #


def test_status_shows_a_tts_spend_line_priced_at_zero_before_any_episode(
    config_dir: Path, db_path: Path
) -> None:
    result = _run("status", "--config-dir", str(config_dir), "--db", str(db_path))
    assert result.exit_code == 0, result.output
    assert "TTS characters" in result.output
    assert "$0.00" in result.output


def test_status_prices_recorded_tts_characters_at_the_configured_model(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    target_date = _scored_date(db_path)

    _fake_run_podcast_script(monkeypatch)
    assert (
        _podcast(
            config_dir, db_path, cache_dir, vault_dir, audio_dir, extra_args=["--date", target_date]
        ).exit_code
        == 0
    )

    with connection(db_path) as conn:
        chars = conn.execute(
            "SELECT SUM(tts_characters) AS n FROM runs WHERE kind = 'podcast'"
        ).fetchone()["n"]
    assert chars > 0
    expected_dollars = tts_spend_usd(chars, "hexgrad/kokoro-82m")

    result = _run("status", "--config-dir", str(config_dir), "--db", str(db_path))
    assert result.exit_code == 0, result.output
    tts_line = result.output.split("TTS characters")[1].splitlines()[0]
    assert f"{chars:,}" in tts_line
    assert f"${expected_dollars:.2f}" in tts_line


# --------------------------------------------------------------------------- #
# the marks decide the running order (DESIGN §13.3, §14's split schedule)
# --------------------------------------------------------------------------- #


def _write_marked_digest(vault_dir: Path, *, item_id: int, verdict: str) -> Path:
    """A minimal vault digest carrying one *checked* mark.

    Deliberately hand-built from `checkbox_marker` rather than rendered by
    `digest`: this asserts the podcast command harvests whatever the vault
    actually says, and the marker's wire format is already round-trip tested in
    `tests/test_feedback.py`.
    """
    daily = vault_dir / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    path = daily / "2026-07-15.md"
    marker = checkbox_marker(item_id, verdict).replace("- [ ]", "- [x]")
    path.write_text(f"# Daily Digest\n\n- alpha post\n{marker}\n", encoding="utf-8")
    return path


def test_podcast_harvests_vault_marks_before_selecting(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam the split schedule depends on: a mark ticked after the digest ran
    is still only in the vault markdown, so the podcast must harvest it itself or
    rank against yesterday's marks (DESIGN §14)."""
    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    _fake_run_podcast_script(monkeypatch)
    _write_marked_digest(vault_dir, item_id=1, verdict="useful")

    result = _podcast(
        config_dir,
        db_path,
        cache_dir,
        vault_dir,
        audio_dir,
        extra_args=["--date", _scored_date(db_path)],
    )

    assert result.exit_code == 0, result.output
    with connection(db_path) as conn:
        verdicts = [row[0] for row in conn.execute("SELECT verdict FROM feedback")]
    assert verdicts == ["useful"]


def test_a_noise_mark_keeps_an_item_off_the_episode(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the split schedule: the only kept item is marked
    `noise`, so there is nothing left to record — no Opus call, no script, and a
    clean `ok` run rather than a failure."""
    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    script_calls = _fake_run_podcast_script(monkeypatch)
    _write_marked_digest(vault_dir, item_id=1, verdict="noise")

    result = _podcast(
        config_dir,
        db_path,
        cache_dir,
        vault_dir,
        audio_dir,
        extra_args=["--date", _scored_date(db_path)],
    )

    assert result.exit_code == 0, result.output
    assert "nothing to record" in result.output
    assert script_calls == []  # never billed an Opus call for an episode with no items
    assert not (vault_dir / "podcast").exists()
    with connection(db_path) as conn:
        status = conn.execute("SELECT status FROM runs WHERE kind = 'podcast'").fetchone()[0]
    assert status == "ok"


def test_an_unmarked_day_still_records_a_normal_episode(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marking is upside, never a gate: an empty vault must not stop the episode."""
    assert _ingest(config_dir, db_path, cache_dir).exit_code == 0
    assert _score(config_dir, db_path, monkeypatch).exit_code == 0
    _fake_run_podcast_script(monkeypatch)

    result = _podcast(
        config_dir,
        db_path,
        cache_dir,
        vault_dir,
        audio_dir,
        extra_args=["--date", _scored_date(db_path)],
    )

    assert result.exit_code == 0, result.output
    assert len(list((vault_dir / "podcast").glob("*.md"))) == 1


def test_daily_no_podcast_stops_after_the_digest(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    vault_dir: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evening half of the split schedule (DESIGN §14): digest written, no
    episode, no `podcast` run row — and no TTS or R2 call to mock, which is
    itself the assertion that nothing tried to publish."""
    monkeypatch.setattr(
        "signalforge.llm.run_triage_batch",
        lambda *a, **k: TriageBatchResult(
            results={
                1: TriageResult(triage="keep", signal=5, relevance=5, novelty=4, reasoning="Good.")
            },
            input_tokens=1,
            output_tokens=1,
        ),
    )
    script_calls = _fake_run_podcast_script(monkeypatch)

    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"etag": '"a-1"'})
        )
        result = _run(
            "daily",
            "--no-podcast",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
            "--vault-dir",
            str(vault_dir),
            "--audio-dir",
            str(audio_dir),
        )

    assert result.exit_code == 0, result.output
    with connection(db_path) as conn:
        kinds = [row[0] for row in conn.execute("SELECT kind FROM runs ORDER BY id")]
    assert kinds == ["ingest", "score", "daily"]
    assert script_calls == []
    assert not (vault_dir / "podcast").exists()
    assert list((vault_dir / "daily").glob("*.md"))  # the digest still landed
