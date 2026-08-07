"""Tests for `deliver/podcast.py` — the podcast channel (DESIGN §13.3, Stage 5).

Every test builds its own throwaway DB (`conn` fixture), vault, and audio
directory under `tmp_path` — never the real filesystem or `data/signalforge.db`
(CLAUDE.md §8). TTS and R2 are always mocked via `respx` (NEVER rule 13); every
test in this file also fakes `shutil.which("ffmpeg")` and `_concat_mp3` (see
`tests/deliver/test_tts.py`), so the suite runs identically with or without
ffmpeg installed.

Four properties carry the weight:

* **Run twice, nothing new happens.** `delivery_exists` short-circuits a
  second delivery for the same date: 0 TTS characters, 0 uploads, 0 new
  `deliveries` rows (NEVER rule 4).
* **A cached local mp3 skips synthesis, not delivery.** The freshness/
  send-log guards and a cached mp3 are two independent idempotency levers —
  the second only ever matters when the first hasn't fired yet (e.g. a prior
  run synthesized but failed before uploading).
* **Retention prunes exactly what falls outside the window, once.** Three
  remote episodes with `retention_episodes=2` deletes exactly one; pruning
  the same state again deletes nothing.
* **Every refusal is named**, and TTS spend already incurred is never lost
  to a later failure (the same NEVER rule 11 discipline `synth/podcast.py`
  already applies to Opus tokens, applied here to TTS characters).
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
import respx
from pydantic import SecretStr

from signalforge.config import DeliveryConfig, PodcastChannelConfig, SettingsConfig
from signalforge.db import delivery_exists
from signalforge.deliver.podcast import (
    CHANNEL_NAME,
    REPORT_KIND_PODCAST,
    _Feed,
    _feed_dates,
    _FeedEpisode,
    _prune,
    _prune_keep_dates,
    _r2_prefix,
    _remote_key,
    audio_path,
    deliver_podcast,
    render_feed,
)
from signalforge.report.podcast import (
    ScriptContext,
    ScriptSegment,
    ScriptTurn,
    SourceLink,
    render_script,
    script_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FIXTURE = REPO_ROOT / "fixtures" / "podcast_feed_golden.xml"

TTS_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"
R2_ENDPOINT = "https://account123.r2.cloudflarestorage.com"
BUCKET = "signalforge-podcast"

TARGET_DATE = Date(2026, 8, 7)
PRESENTER_A = "Alex"
PRESENTER_B = "Sam"


@pytest.fixture(autouse=True)
def _fake_ffmpeg_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")


@pytest.fixture(autouse=True)
def _fake_concat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("signalforge.deliver.tts._concat_mp3", lambda chunks: b"|".join(chunks))


@pytest.fixture(autouse=True)
def _secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real env vars, so `get_secret` never reaches a developer's real
    `.env` during a test run — same pattern `test_deliver_cli.py` uses."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret-key")


def _config(**overrides: object) -> PodcastChannelConfig:
    fields: dict[str, object] = {
        "enabled": True,
        "tts_api_key_env": "OPENROUTER_API_KEY",
        "tts_model": "hexgrad/kokoro-82m",
        "voice_a": "af_bella",
        "voice_b": "am_adam",
        "presenter_a": PRESENTER_A,
        "presenter_b": PRESENTER_B,
        "r2_endpoint": R2_ENDPOINT,
        "r2_bucket": BUCKET,
        "r2_access_key_env": "R2_ACCESS_KEY_ID",
        "r2_secret_key_env": "R2_SECRET_ACCESS_KEY",
        "public_base_url": "https://pub-example.r2.dev/prefix",
        "retention_episodes": 30,
        "feed_title": "SignalForge Daily",
    }
    fields.update(overrides)
    return PodcastChannelConfig(**fields)  # type: ignore[arg-type]


def _settings(vault_dir: Path, *, config: PodcastChannelConfig | None) -> SettingsConfig:
    return SettingsConfig(
        timezone="UTC", vault_dir=vault_dir, delivery=DeliveryConfig(podcast=config)
    )


def _write_vault_script(
    vault_dir: Path, *, date: Date, item_count: int = 1, empty: bool = False
) -> str:
    """Write a valid vault script and return its rendered text."""
    segments = (
        ()
        if empty
        else (
            ScriptSegment(
                label="Intro", turns=(ScriptTurn(speaker=PRESENTER_A, text="Good morning."),)
            ),
            ScriptSegment(
                label="Segment 1",
                turns=(
                    ScriptTurn(speaker=PRESENTER_A, text="Here's today's story."),
                    ScriptTurn(speaker=PRESENTER_B, text="Sounds great."),
                ),
                sources=(SourceLink(item_id=1, title="A Story", url="https://example.com/a"),),
            ),
            ScriptSegment(
                label="Outro", turns=(ScriptTurn(speaker=PRESENTER_A, text="See you tomorrow."),)
            ),
        )
    )
    context = ScriptContext(
        date=date,
        script_version="podcast-v2",
        model="claude-opus-5",
        item_count=item_count,
        segments=segments,
    )
    rendered = render_script(context)
    path = script_path(vault_dir, date=date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return rendered


def _mock_r2_success() -> None:
    respx.put(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(return_value=httpx.Response(200))
    respx.get(f"{R2_ENDPOINT}/{BUCKET}").mock(
        return_value=httpx.Response(
            200,
            content=b'<?xml version="1.0"?><ListBucketResult '
            b'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"></ListBucketResult>',
        )
    )
    respx.delete(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(return_value=httpx.Response(204))


# --------------------------------------------------------------------------- #
# render_feed — golden file
# --------------------------------------------------------------------------- #


def test_render_feed_matches_the_golden_fixture() -> None:
    feed = _Feed(
        title="SignalForge Daily",
        public_base_url="https://pub-example.r2.dev/prefix",
        episodes=(
            _FeedEpisode(
                title="SignalForge Daily — 2026-08-07",
                description="3 stories from SignalForge's daily digest.",
                pub_date_rfc822="Fri, 07 Aug 2026 06:00:00 -0000",
                guid="2026-08-07",
                audio_url="https://pub-example.r2.dev/prefix/signalforge-2026-08-07.mp3",
                audio_bytes=123456,
            ),
            _FeedEpisode(
                title="SignalForge Daily — 2026-08-06",
                description="5 stories from SignalForge's daily digest.",
                pub_date_rfc822="Thu, 06 Aug 2026 06:00:00 -0000",
                guid="2026-08-06",
                audio_url="https://pub-example.r2.dev/prefix/signalforge-2026-08-06.mp3",
                audio_bytes=98765,
            ),
        ),
    )

    rendered = render_feed(feed)

    assert rendered == GOLDEN_FIXTURE.read_text(encoding="utf-8")


def test_render_feed_escapes_an_ampersand_in_the_title() -> None:
    feed = _Feed(
        title="Signal & Forge",
        public_base_url="https://pub-example.r2.dev/prefix",
        episodes=(),
    )

    rendered = render_feed(feed)

    assert "Signal &amp; Forge" in rendered
    assert "Signal & Forge<" not in rendered


# --------------------------------------------------------------------------- #
# _feed_dates — always a subset of the caller's keep_dates
# --------------------------------------------------------------------------- #


def test_feed_dates_filters_keep_dates_down_to_ones_with_local_audio(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    keep_dates = [Date(2026, 8, 7), Date(2026, 8, 6), Date(2026, 8, 5)]
    for day in (6, 7):  # day 5 is in the keep set but has no local mp3
        audio_path(audio_dir, date=Date(2026, 8, day)).parent.mkdir(parents=True, exist_ok=True)
        audio_path(audio_dir, date=Date(2026, 8, day)).write_bytes(b"x")

    dates = _feed_dates(keep_dates=keep_dates, audio_dir=audio_dir)

    assert dates == [Date(2026, 8, 7), Date(2026, 8, 6)]


def test_feed_dates_never_exceeds_keep_dates_even_with_more_local_audio(tmp_path: Path) -> None:
    """The bug a code-reviewer pass reproduced end to end: local audio for a
    date *outside* the keep set must not let that date leak into the feed —
    `_feed_dates` filters `keep_dates`, it does not recompute its own,
    larger set from local disk."""
    audio_dir = tmp_path / "audio"
    keep_dates = [Date(2026, 8, 7)]  # e.g. retention_episodes=1
    for day in (6, 7):  # local audio exists for both, but only day 7 is kept
        audio_path(audio_dir, date=Date(2026, 8, day)).parent.mkdir(parents=True, exist_ok=True)
        audio_path(audio_dir, date=Date(2026, 8, day)).write_bytes(b"x")

    dates = _feed_dates(keep_dates=keep_dates, audio_dir=audio_dir)

    assert dates == [Date(2026, 8, 7)]


# --------------------------------------------------------------------------- #
# _prune_keep_dates — vault-only, so a wiped local cache never causes deletion
# --------------------------------------------------------------------------- #


def test_prune_keep_dates_does_not_require_local_audio(tmp_path: Path) -> None:
    """The bug a code-reviewer pass demonstrated end to end: if the keep set
    required local audio, a wiped `data/audio/` would make every remote
    episode look prunable. `_prune_keep_dates` must not have that shape."""
    vault_dir = tmp_path / "vault"
    for day in (5, 6, 7):
        _write_vault_script(vault_dir, date=Date(2026, 8, day))
    # No local audio directory at all — simulates a wiped/fresh-clone cache.

    dates = _prune_keep_dates(vault_dir=vault_dir, retention_episodes=30)

    assert dates == [Date(2026, 8, 7), Date(2026, 8, 6), Date(2026, 8, 5)]


def test_prune_keep_dates_is_capped_and_newest_first(tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    for day in (4, 5, 6, 7):
        _write_vault_script(vault_dir, date=Date(2026, 8, day))

    dates = _prune_keep_dates(vault_dir=vault_dir, retention_episodes=2)

    assert dates == [Date(2026, 8, 7), Date(2026, 8, 6)]


# --------------------------------------------------------------------------- #
# _r2_prefix / _remote_key — objects must live where the feed says they do
# --------------------------------------------------------------------------- #


def test_r2_prefix_is_the_path_segment_of_public_base_url() -> None:
    config = _config(public_base_url="https://pub-example.r2.dev/some-unguessable-prefix")
    assert _r2_prefix(config) == "some-unguessable-prefix/"


def test_r2_prefix_is_empty_when_the_feed_is_served_from_the_bucket_root() -> None:
    config = _config(public_base_url="https://pub-example.r2.dev")
    assert _r2_prefix(config) == ""


def test_remote_key_matches_the_path_segment_the_feed_advertises() -> None:
    """The bug: an object written under `_remote_key`'s prefix must live at
    the exact path `_feed_episode`'s `audio_url` advertises (built from
    `public_base_url` directly) — otherwise every enclosure 404s. R2's
    public dev-domain serving maps a URL path straight onto an object key
    with no rewriting, so "prefix in the advertised URL" and "prefix on the
    written object" must be the literal same string.
    """
    config = _config(public_base_url="https://pub-example.r2.dev/some-unguessable-prefix")
    basename = "signalforge-2026-08-07.mp3"

    remote_key = _remote_key(config, basename)
    advertised_path = urlsplit(f"{config.public_base_url}/{basename}").path.strip("/")

    assert remote_key == advertised_path


# --------------------------------------------------------------------------- #
# _prune — exactly what the plan's gate asks for
# --------------------------------------------------------------------------- #


@respx.mock
def test_prune_deletes_exactly_the_dates_outside_retention(tmp_path: Path) -> None:

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    config = _config()
    # Prefixed to match `config.public_base_url`'s path — a real R2 bucket
    # configured that way would never return unprefixed keys, and an
    # unprefixed fixture here would let `_prune`'s `removeprefix(r2_prefix)`
    # step regress to a no-op without failing this test.
    remote_keys = [
        f"{_r2_prefix(config)}signalforge-2026-08-05.mp3",
        f"{_r2_prefix(config)}signalforge-2026-08-06.mp3",
        f"{_r2_prefix(config)}signalforge-2026-08-07.mp3",
    ]
    respx.get(f"{R2_ENDPOINT}/{BUCKET}").mock(
        return_value=httpx.Response(
            200,
            content=(
                '<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                + "".join(f"<Contents><Key>{k}</Key></Contents>" for k in remote_keys)
                + "</ListBucketResult>"
            ).encode(),
        )
    )
    delete_route = respx.delete(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(
        return_value=httpx.Response(204)
    )

    with httpx.Client() as client:
        deleted = _prune(
            client,
            config=config,
            retained_dates={Date(2026, 8, 6), Date(2026, 8, 7)},
            audio_dir=audio_dir,
            access_key=SecretStr("k"),
            secret_key=SecretStr("s"),
            now=datetime.now(UTC),
        )

    assert deleted == 1
    assert delete_route.call_count == 1
    assert delete_route.calls.last.request.url.path.endswith("prefix/signalforge-2026-08-05.mp3")
    assert delete_route.calls.last.request.url.path.endswith("2026-08-05.mp3")


@respx.mock
def test_pruning_the_same_state_a_second_time_deletes_nothing(tmp_path: Path) -> None:

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    config = _config()
    remaining_keys = [
        f"{_r2_prefix(config)}signalforge-2026-08-06.mp3",
        f"{_r2_prefix(config)}signalforge-2026-08-07.mp3",
    ]
    respx.get(f"{R2_ENDPOINT}/{BUCKET}").mock(
        return_value=httpx.Response(
            200,
            content=(
                '<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                + "".join(f"<Contents><Key>{k}</Key></Contents>" for k in remaining_keys)
                + "</ListBucketResult>"
            ).encode(),
        )
    )
    delete_route = respx.delete(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(
        return_value=httpx.Response(204)
    )

    with httpx.Client() as client:
        deleted = _prune(
            client,
            config=config,
            retained_dates={Date(2026, 8, 6), Date(2026, 8, 7)},
            audio_dir=audio_dir,
            access_key=SecretStr("k"),
            secret_key=SecretStr("s"),
            now=datetime.now(UTC),
        )

    assert deleted == 0
    assert delete_route.call_count == 0


# --------------------------------------------------------------------------- #
# deliver_podcast — guard chain refusals
# --------------------------------------------------------------------------- #


def test_a_disabled_channel_returns_none(conn: sqlite3.Connection, tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    settings = _settings(vault_dir, config=_config(enabled=False))

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=tmp_path / "audio",
        today=TARGET_DATE,
    )

    assert outcome is None


def test_an_unconfigured_channel_returns_none(conn: sqlite3.Connection, tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    settings = _settings(vault_dir, config=None)

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=tmp_path / "audio",
        today=TARGET_DATE,
    )

    assert outcome is None


def test_a_stale_date_is_refused_by_the_freshness_window(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    vault_dir = tmp_path / "vault"
    settings = _settings(vault_dir, config=_config())

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=Date(2020, 1, 1),
        vault_dir=vault_dir,
        audio_dir=tmp_path / "audio",
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is False
    assert "freshness" not in outcome.reason  # message names days, not the word "freshness"
    assert "day(s) from today" in outcome.reason


def test_missing_secrets_are_named_and_refused(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    vault_dir = tmp_path / "vault"
    _write_vault_script(vault_dir, date=TARGET_DATE)
    settings = _settings(vault_dir, config=_config())

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=tmp_path / "audio",
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is False
    assert "OPENROUTER_API_KEY" in outcome.reason


def test_a_missing_vault_script_is_refused(conn: sqlite3.Connection, tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    settings = _settings(vault_dir, config=_config())

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=tmp_path / "audio",
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is False
    assert "nothing to deliver" in outcome.reason


def test_an_empty_episode_script_is_refused(conn: sqlite3.Connection, tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    _write_vault_script(vault_dir, date=TARGET_DATE, empty=True)
    settings = _settings(vault_dir, config=_config())

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=tmp_path / "audio",
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is False
    assert "no episode content" in outcome.reason


@respx.mock
def test_a_backfill_outside_retention_is_refused_before_any_tts_spend(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A code-reviewer pass demonstrated this end to end: with a small
    `retention_episodes` and a newer script already in the vault, a backfill
    for an older date used to be synthesized, uploaded, and immediately
    pruned by this same run's own retention step — real TTS money spent to
    publish an episode that never survived the function call, with
    `record_delivery` then permanently suppressing any retry."""
    vault_dir = tmp_path / "vault"
    audio_dir = tmp_path / "audio"
    older_date = Date(2026, 8, 6)
    newer_date = Date(2026, 8, 7)
    _write_vault_script(vault_dir, date=older_date)
    _write_vault_script(vault_dir, date=newer_date)
    settings = _settings(vault_dir, config=_config(retention_episodes=1))
    tts_route = respx.post(TTS_ENDPOINT).mock(
        return_value=httpx.Response(200, content=b"audio-chunk")
    )

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=older_date,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=newer_date,
    )

    assert outcome is not None
    assert outcome.sent is False
    assert "retention_episodes=1" in outcome.reason
    assert tts_route.call_count == 0
    assert not audio_path(audio_dir, date=older_date).is_file()
    assert not delivery_exists(
        conn, channel=CHANNEL_NAME, report_kind=REPORT_KIND_PODCAST, target_date=older_date
    )


@respx.mock
def test_a_wiped_local_audio_directory_does_not_delete_the_remote_back_catalogue(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """`data/audio/` is not backed up (DESIGN §14); a fresh clone or a wiped
    cache must not read as "these episodes don't exist anymore" and prune
    real, already-paid-for R2 objects. Simulates the wipe by delivering a
    new date while three *older* dates have vault scripts but no local mp3
    at all — none of the three should be deleted."""
    vault_dir = tmp_path / "vault"
    audio_dir = tmp_path / "audio"
    older_dates = [Date(2026, 8, 4), Date(2026, 8, 5), Date(2026, 8, 6)]
    for date in older_dates:
        _write_vault_script(vault_dir, date=date)  # vault script survives; no local mp3
    _write_vault_script(vault_dir, date=TARGET_DATE)
    settings = _settings(vault_dir, config=_config(retention_episodes=30))
    respx.post(TTS_ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio-chunk"))
    respx.put(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(return_value=httpx.Response(200))
    remote_keys = [f"prefix/signalforge-{d.isoformat()}.mp3" for d in [*older_dates, TARGET_DATE]]
    respx.get(f"{R2_ENDPOINT}/{BUCKET}").mock(
        return_value=httpx.Response(
            200,
            content=(
                '<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                + "".join(f"<Contents><Key>{k}</Key></Contents>" for k in remote_keys)
                + "</ListBucketResult>"
            ).encode(),
        )
    )
    delete_route = respx.delete(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(
        return_value=httpx.Response(204)
    )

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is True
    assert delete_route.call_count == 0


# --------------------------------------------------------------------------- #
# deliver_podcast — the happy path
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_successful_delivery_synthesizes_uploads_and_records(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    vault_dir = tmp_path / "vault"
    audio_dir = tmp_path / "audio"
    _write_vault_script(vault_dir, date=TARGET_DATE, item_count=1)
    settings = _settings(vault_dir, config=_config())
    tts_route = respx.post(TTS_ENDPOINT).mock(
        return_value=httpx.Response(200, content=b"audio-chunk")
    )
    _mock_r2_success()

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is True
    assert outcome.tts_characters > 0
    assert tts_route.call_count > 0
    assert audio_path(audio_dir, date=TARGET_DATE).is_file()
    assert delivery_exists(
        conn, channel=CHANNEL_NAME, report_kind=REPORT_KIND_PODCAST, target_date=TARGET_DATE
    )


@respx.mock
def test_running_delivery_twice_does_nothing_new_the_second_time(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    vault_dir = tmp_path / "vault"
    audio_dir = tmp_path / "audio"
    _write_vault_script(vault_dir, date=TARGET_DATE)
    settings = _settings(vault_dir, config=_config())
    tts_route = respx.post(TTS_ENDPOINT).mock(
        return_value=httpx.Response(200, content=b"audio-chunk")
    )
    put_route = respx.put(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(
        return_value=httpx.Response(200)
    )
    respx.get(f"{R2_ENDPOINT}/{BUCKET}").mock(
        return_value=httpx.Response(
            200,
            content=b'<?xml version="1.0"?><ListBucketResult '
            b'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"></ListBucketResult>',
        )
    )

    first = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=TARGET_DATE,
    )
    tts_calls_after_first = tts_route.call_count
    put_calls_after_first = put_route.call_count
    row_count_after_first = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]

    second = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=TARGET_DATE,
    )

    assert first is not None and first.sent is True
    assert second is not None and second.sent is False
    assert "already delivered" in second.reason
    assert second.tts_characters == 0
    assert tts_route.call_count == tts_calls_after_first
    assert put_route.call_count == put_calls_after_first
    assert conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == row_count_after_first
    assert row_count_after_first == 1


@respx.mock
def test_a_cached_local_mp3_skips_synthesis_but_still_delivers(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Simulates a prior run that synthesized successfully but crashed
    before uploading — the mp3 is on disk, but no `deliveries` row exists
    yet. The retry must not re-pay for TTS."""
    vault_dir = tmp_path / "vault"
    audio_dir = tmp_path / "audio"
    _write_vault_script(vault_dir, date=TARGET_DATE)
    audio_path(audio_dir, date=TARGET_DATE).parent.mkdir(parents=True, exist_ok=True)
    audio_path(audio_dir, date=TARGET_DATE).write_bytes(b"already-synthesized-audio")
    settings = _settings(vault_dir, config=_config())
    tts_route = respx.post(TTS_ENDPOINT).mock(
        return_value=httpx.Response(200, content=b"audio-chunk")
    )
    _mock_r2_success()

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is True
    assert outcome.tts_characters == 0
    assert tts_route.call_count == 0


@respx.mock
def test_a_synthesis_failure_is_reported_and_nothing_is_recorded(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    vault_dir = tmp_path / "vault"
    audio_dir = tmp_path / "audio"
    _write_vault_script(vault_dir, date=TARGET_DATE)
    settings = _settings(vault_dir, config=_config())
    respx.post(TTS_ENDPOINT).mock(return_value=httpx.Response(500))

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is False
    assert "synthesis failed" in outcome.reason
    assert not audio_path(audio_dir, date=TARGET_DATE).is_file()
    assert not delivery_exists(
        conn, channel=CHANNEL_NAME, report_kind=REPORT_KIND_PODCAST, target_date=TARGET_DATE
    )


@respx.mock
def test_an_upload_failure_after_synthesis_still_reports_the_spend(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """NEVER rule 11, applied to TTS characters: a real spend that already
    happened must not be lost just because the next step failed."""
    vault_dir = tmp_path / "vault"
    audio_dir = tmp_path / "audio"
    _write_vault_script(vault_dir, date=TARGET_DATE)
    settings = _settings(vault_dir, config=_config())
    respx.post(TTS_ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio-chunk"))
    respx.put(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(return_value=httpx.Response(403))

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is False
    assert outcome.tts_characters > 0
    assert audio_path(audio_dir, date=TARGET_DATE).is_file()  # cached for the retry


@respx.mock
def test_a_prune_failure_does_not_block_recording_delivery(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    vault_dir = tmp_path / "vault"
    audio_dir = tmp_path / "audio"
    _write_vault_script(vault_dir, date=TARGET_DATE)
    settings = _settings(vault_dir, config=_config())
    respx.post(TTS_ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio-chunk"))
    respx.put(url__regex=rf"{R2_ENDPOINT}/{BUCKET}/.*").mock(return_value=httpx.Response(200))
    respx.get(f"{R2_ENDPOINT}/{BUCKET}").mock(return_value=httpx.Response(500))

    outcome = deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=TARGET_DATE,
    )

    assert outcome is not None
    assert outcome.sent is True
    assert delivery_exists(
        conn, channel=CHANNEL_NAME, report_kind=REPORT_KIND_PODCAST, target_date=TARGET_DATE
    )


@respx.mock
def test_the_body_hash_is_the_sha256_of_the_vault_script_text(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    vault_dir = tmp_path / "vault"
    audio_dir = tmp_path / "audio"
    script_text = _write_vault_script(vault_dir, date=TARGET_DATE)
    settings = _settings(vault_dir, config=_config())
    respx.post(TTS_ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio-chunk"))
    _mock_r2_success()

    deliver_podcast(
        conn,
        settings=settings,
        target_date=TARGET_DATE,
        vault_dir=vault_dir,
        audio_dir=audio_dir,
        today=TARGET_DATE,
    )

    row = conn.execute(
        "SELECT body_hash FROM deliveries WHERE channel = ?", (CHANNEL_NAME,)
    ).fetchone()
    assert row["body_hash"] == hashlib.sha256(script_text.encode("utf-8")).hexdigest()
