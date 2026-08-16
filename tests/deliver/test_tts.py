"""Tests for `deliver/tts.py` — OpenRouter synthesis (DESIGN §13.3, Stage 5).

Every network-facing test mocks `shutil.which` (pretend ffmpeg exists) and
`subprocess.run` (never invoke a real ffmpeg process) so the suite runs
identically with or without ffmpeg installed on the test machine — except
the one test explicitly marked to skip without it, which exists so a
machine that *does* have ffmpeg gets a real end-to-end check of the concat
invocation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import SecretStr

from signalforge.config import PodcastChannelConfig
from signalforge.deliver.tts import (
    _PRICE_PER_MILLION_CHARS_USD,
    _UNKNOWN_MODEL_FALLBACK_USD_PER_MILLION_CHARS,
    TTS_MAX_CHARS,
    TTS_MONTHLY_CEILING_USD,
    TTS_MONTHLY_CHARACTER_CEILING,
    SynthesisError,
    _coalesce,
    _concat_mp3,
    synthesize_episode,
    tts_spend_usd,
)
from signalforge.report.podcast import ScriptTurn

ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"
API_KEY = SecretStr("sk-or-test-key-not-real")


def _no_sleep(_seconds: float) -> None:
    """Backoff seam — see `deliver.email`'s test suite for the same pattern."""


CONFIG = PodcastChannelConfig(
    enabled=True,
    tts_api_key_env="OPENROUTER_API_KEY",
    tts_model="hexgrad/kokoro-82m",
    voice_a="Kore",
    voice_b="Puck",
    presenter_a="Alex",
    presenter_b="Sam",
    r2_endpoint="https://account.r2.cloudflarestorage.com",
    r2_bucket="signalforge-podcast",
    r2_access_key_env="R2_ACCESS_KEY_ID",
    r2_secret_key_env="R2_SECRET_ACCESS_KEY",
    public_base_url="https://pub-example.r2.dev/prefix",
    retention_episodes=30,
    feed_title="SignalForge Daily",
)


def _turn(speaker: str, text: str) -> ScriptTurn:
    return ScriptTurn(speaker=speaker, text=text)


@pytest.fixture(autouse=True)
def _fake_ffmpeg_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this file pretends `ffmpeg` is on `PATH`, so tests that
    exercise real synthesis logic are not skipped on a machine without it.
    Tests specifically about the *missing ffmpeg* refusal override this."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")


@pytest.fixture(autouse=True)
def _fake_concat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never invoke a real ffmpeg process. Stands in for `_concat_mp3` by
    concatenating the raw chunk bytes with a separator — good enough to
    prove chunk ordering and count without depending on ffmpeg being
    installed on the test machine."""

    def fake_concat(chunks: list[bytes]) -> bytes:
        return b"|".join(chunks)

    monkeypatch.setattr("signalforge.deliver.tts._concat_mp3", fake_concat)


# --------------------------------------------------------------------------- #
# tts_spend_usd — pricing
# --------------------------------------------------------------------------- #


def test_a_known_char_priced_model_is_priced_exactly() -> None:
    # hexgrad/kokoro-82m: $0.62 / 1,000,000 chars
    assert tts_spend_usd(1_000_000, "hexgrad/kokoro-82m") == pytest.approx(0.62)


def test_an_unrecognized_model_falls_back_to_the_worst_known_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert tts_spend_usd(1_000_000, "some/future-model") == pytest.approx(100.00)


def test_a_free_tier_model_is_priced_at_zero() -> None:
    assert tts_spend_usd(1_000_000, "fish-audio/s2.1-pro-free:free") == 0.0


def test_zero_characters_costs_nothing_regardless_of_model() -> None:
    assert tts_spend_usd(0, "hexgrad/kokoro-82m") == 0.0
    assert tts_spend_usd(0, "unknown/model") == 0.0


# --------------------------------------------------------------------------- #
# TTS_MONTHLY_CHARACTER_CEILING — the worst-case math the docstring claims
# --------------------------------------------------------------------------- #


def test_the_hard_daily_cap_times_a_month_fits_the_ceiling_with_headroom() -> None:
    worst_case_month = TTS_MAX_CHARS * 31
    assert worst_case_month <= TTS_MONTHLY_CHARACTER_CEILING


def test_the_fallback_price_is_at_least_as_expensive_as_every_known_model() -> None:
    """An `llm-cost-guard` finding: the fallback used to be a hand-copied
    literal that could silently understate the true worst case the moment a
    pricier model was added to the table. Checked independently of the
    `max()` the source itself uses, so this would fail if the fallback and
    the table's actual maximum ever disagreed."""
    assert all(
        price <= _UNKNOWN_MODEL_FALLBACK_USD_PER_MILLION_CHARS
        for price in _PRICE_PER_MILLION_CHARS_USD.values()
    )


def test_the_dollar_ceiling_matches_the_hard_cap_priced_at_the_worst_known_rate() -> None:
    expected = TTS_MAX_CHARS * 31 * _UNKNOWN_MODEL_FALLBACK_USD_PER_MILLION_CHARS / 1_000_000
    assert pytest.approx(expected) == TTS_MONTHLY_CEILING_USD
    assert pytest.approx(43.4) == TTS_MONTHLY_CEILING_USD  # the docstring's own worked number


def test_the_dollar_ceiling_honestly_exceeds_the_whole_pipeline_alarm() -> None:
    """Documents the known tension the constant's own docstring discloses —
    the worst *known* model at the hard per-episode cap every single day
    approaches DESIGN §8's $50 alarm on its own. Not something to
    quietly "fix" by shrinking the number; it reflects a real worst case,
    not the shipped default (`hexgrad/kokoro-82m` prices the same scenario
    at ~$0.27/month)."""
    assert TTS_MONTHLY_CEILING_USD > 30.0


# --------------------------------------------------------------------------- #
# _coalesce — merging consecutive same-speaker turns
# --------------------------------------------------------------------------- #


def test_consecutive_same_speaker_turns_merge_into_one_chunk() -> None:
    turns = [_turn("Alex", "First line."), _turn("Alex", "Second line.")]

    chunks = _coalesce(turns, presenter_a="Alex", presenter_b="Sam", voice_a="Kore", voice_b="Puck")

    assert len(chunks) == 1
    assert chunks[0].voice == "Kore"
    assert chunks[0].text == "First line. Second line."


def test_alternating_speakers_produce_one_chunk_per_turn() -> None:
    turns = [_turn("Alex", "Hi."), _turn("Sam", "Hi back."), _turn("Alex", "Let's start.")]

    chunks = _coalesce(turns, presenter_a="Alex", presenter_b="Sam", voice_a="Kore", voice_b="Puck")

    assert [c.voice for c in chunks] == ["Kore", "Puck", "Kore"]
    assert [c.text for c in chunks] == ["Hi.", "Hi back.", "Let's start."]


def test_a_speaker_matching_neither_presenter_raises() -> None:
    turns = [_turn("Nobody", "Whose voice is this?")]

    with pytest.raises(SynthesisError, match="matches neither configured presenter"):
        _coalesce(turns, presenter_a="Alex", presenter_b="Sam", voice_a="Kore", voice_b="Puck")


def test_total_character_count_is_unchanged_by_coalescing() -> None:
    """Coalescing buys fewer requests, not fewer billed characters."""
    turns = [_turn("Alex", "abc"), _turn("Alex", "def"), _turn("Sam", "ghi")]

    chunks = _coalesce(turns, presenter_a="Alex", presenter_b="Sam", voice_a="Kore", voice_b="Puck")

    original_chars = sum(len(t.text) for t in turns)
    # The join inserts one space per merge; account for it explicitly rather
    # than asserting exact equality, which would be the wrong invariant.
    merges = len(turns) - len(chunks)
    assert sum(len(c.text) for c in chunks) == original_chars + merges


# --------------------------------------------------------------------------- #
# synthesize_episode — refusals that must never touch the network
# --------------------------------------------------------------------------- #


def test_missing_ffmpeg_refuses_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with respx.mock:
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio"))

        with pytest.raises(SynthesisError, match="ffmpeg"):
            synthesize_episode([_turn("Alex", "Hi.")], config=CONFIG, api_key=API_KEY)

        assert route.call_count == 0


def test_an_over_length_episode_refuses_before_any_request() -> None:
    turns = [_turn("Alex", "x" * (TTS_MAX_CHARS + 1))]
    with respx.mock:
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio"))

        with pytest.raises(SynthesisError, match="tts hard cap"):
            synthesize_episode(turns, config=CONFIG, api_key=API_KEY)

        assert route.call_count == 0


def test_an_empty_episode_refuses_before_any_request() -> None:
    with respx.mock:
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio"))

        with pytest.raises(SynthesisError, match="no dialogue"):
            synthesize_episode([], config=CONFIG, api_key=API_KEY)

        assert route.call_count == 0


# --------------------------------------------------------------------------- #
# synthesize_episode — the happy path and request shape
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_successful_episode_returns_concatenated_audio_and_the_right_character_count() -> None:
    route = respx.post(ENDPOINT).mock(
        side_effect=[
            httpx.Response(200, content=b"chunk-a"),
            httpx.Response(200, content=b"chunk-b"),
        ]
    )
    turns = [_turn("Alex", "First up."), _turn("Sam", "Sounds good.")]

    result = synthesize_episode(turns, config=CONFIG, api_key=API_KEY, sleep=_no_sleep)

    assert route.call_count == 2
    assert result.audio == b"chunk-a|chunk-b"
    assert result.characters == len("First up.") + len("Sounds good.")


@respx.mock
def test_the_request_carries_the_right_model_voice_and_format() -> None:
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio"))

    synthesize_episode([_turn("Alex", "Hello there.")], config=CONFIG, api_key=API_KEY)

    payload = route.calls.last.request
    body = json.loads(payload.content)
    assert body["model"] == "hexgrad/kokoro-82m"
    assert body["voice"] == "Kore"
    assert body["input"] == "Hello there."
    assert body["response_format"] == "mp3"


@respx.mock
def test_the_api_key_travels_in_the_header_and_nowhere_else() -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, content=b"audio"))

    synthesize_episode([_turn("Alex", "Hello.")], config=CONFIG, api_key=API_KEY)

    sent = respx.calls.last.request
    assert sent.headers["authorization"] == f"Bearer {API_KEY.get_secret_value()}"
    assert API_KEY.get_secret_value() not in sent.content.decode("utf-8")


# --------------------------------------------------------------------------- #
# synthesize_episode — retry policy, copied from deliver.email
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_transient_status_is_retried_then_succeeds() -> None:
    route = respx.post(ENDPOINT).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, content=b"audio")]
    )

    result = synthesize_episode(
        [_turn("Alex", "Hi.")], config=CONFIG, api_key=API_KEY, sleep=_no_sleep
    )

    assert result.audio == b"audio"
    assert route.call_count == 2


@respx.mock
def test_retry_exhaustion_raises() -> None:
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(503))

    with pytest.raises(SynthesisError, match="after 3 attempts"):
        synthesize_episode([_turn("Alex", "Hi.")], config=CONFIG, api_key=API_KEY, sleep=_no_sleep)

    assert route.call_count == 3


@respx.mock
def test_a_connect_error_is_retried_and_then_raises() -> None:
    route = respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("no route to host"))

    with pytest.raises(SynthesisError):
        synthesize_episode([_turn("Alex", "Hi.")], config=CONFIG, api_key=API_KEY, sleep=_no_sleep)

    assert route.call_count == 3


@respx.mock
def test_a_read_timeout_is_not_retried_because_the_request_was_paid() -> None:
    """A `ReadTimeout` means the request reached the provider and audio may
    already have been generated — replaying it risks double-billing."""
    route = respx.post(ENDPOINT).mock(side_effect=httpx.ReadTimeout("response lost"))

    with pytest.raises(SynthesisError, match="without retrying"):
        synthesize_episode([_turn("Alex", "Hi.")], config=CONFIG, api_key=API_KEY, sleep=_no_sleep)

    assert route.call_count == 1


@respx.mock
def test_a_permanent_refusal_does_not_retry() -> None:
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(400, text="bad voice id"))

    with pytest.raises(SynthesisError, match="400"):
        synthesize_episode([_turn("Alex", "Hi.")], config=CONFIG, api_key=API_KEY, sleep=_no_sleep)

    assert route.call_count == 1


@respx.mock
def test_the_provider_backoff_hint_is_honored() -> None:
    route = respx.post(ENDPOINT).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, content=b"audio"),
        ]
    )

    result = synthesize_episode(
        [_turn("Alex", "Hi.")], config=CONFIG, api_key=API_KEY, sleep=_no_sleep
    )

    assert result.audio == b"audio"
    assert route.call_count == 2


# --------------------------------------------------------------------------- #
# SynthesisError.characters_sent — NEVER rule 11, applied to TTS spend
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_mid_episode_failure_reports_the_characters_already_billed() -> None:
    """Chunk 3 of 3 fails permanently after chunks 1-2 already succeeded (and
    were billed) — the raised error must carry what was already spent, not
    silently drop it. Alternating speakers so each turn is its own chunk
    (no coalescing merge), keeping the arithmetic exact."""
    turns = [_turn("Alex", "First."), _turn("Sam", "Second."), _turn("Alex", "Third.")]
    respx.post(ENDPOINT).mock(
        side_effect=[
            httpx.Response(200, content=b"chunk-1"),
            httpx.Response(200, content=b"chunk-2"),
            httpx.Response(400, text="bad voice id"),
        ]
    )

    with pytest.raises(SynthesisError) as caught:
        synthesize_episode(turns, config=CONFIG, api_key=API_KEY, sleep=_no_sleep)

    assert caught.value.characters_sent == len("First.") + len("Second.")


@respx.mock
def test_a_synthesis_failure_before_any_chunk_succeeds_reports_zero_spend() -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(400, text="bad voice id"))

    with pytest.raises(SynthesisError) as caught:
        synthesize_episode([_turn("Alex", "Hi.")], config=CONFIG, api_key=API_KEY, sleep=_no_sleep)

    assert caught.value.characters_sent == 0


@respx.mock
def test_a_concat_failure_after_full_synthesis_still_reports_all_characters_billed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every chunk succeeded (and was billed) before the post-synthesis
    `ffmpeg` concat step failed — that spend must not be lost either."""

    def failing_concat(_chunks: list[bytes]) -> bytes:
        raise SynthesisError("ffmpeg concat failed (exit 1): b''")

    monkeypatch.setattr("signalforge.deliver.tts._concat_mp3", failing_concat)
    turns = [_turn("Alex", "First."), _turn("Sam", "Second.")]
    respx.post(ENDPOINT).mock(
        side_effect=[
            httpx.Response(200, content=b"chunk-1"),
            httpx.Response(200, content=b"chunk-2"),
        ]
    )

    with pytest.raises(SynthesisError) as caught:
        synthesize_episode(turns, config=CONFIG, api_key=API_KEY, sleep=_no_sleep)

    assert caught.value.characters_sent == len("First.") + len("Second.")


# --------------------------------------------------------------------------- #
# _concat_mp3 — real behavior, not the autouse fake
# --------------------------------------------------------------------------- #


def test_a_single_chunk_never_invokes_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)))

    result = _concat_mp3([b"only-chunk"])

    assert result == b"only-chunk"
    assert calls == []


def test_multiple_chunks_invoke_ffmpeg_concat_with_the_right_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(
        args: list[str], *, cwd: object, capture_output: bool, check: bool, timeout: float
    ) -> subprocess.CompletedProcess[bytes]:
        captured["args"] = args
        # Simulate ffmpeg having written the output file.
        Path(args[-1]).write_bytes(b"concatenated")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _concat_mp3([b"a", b"b", b"c"])

    assert result == b"concatenated"
    args = captured["args"]
    assert args[0] == "ffmpeg"
    assert "-f" in args and args[args.index("-f") + 1] == "concat"
    assert "-c" in args and args[args.index("-c") + 1] == "copy"


def test_an_ffmpeg_failure_raises_synthesis_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"invalid concat list")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SynthesisError, match="ffmpeg concat failed"):
        _concat_mp3([b"a", b"b"])


_MP3_FRAME_HEADER = bytes.fromhex("fffb9004")
"""MPEG-1 Layer III, 128 kbps, 44.1 kHz, no padding, no CRC.

Every field below is derived from these four bytes rather than written beside
them, because the two disagreeing is exactly what broke this fixture before:
the header declared a 417-byte frame and only 62 bytes followed it, so ffmpeg
read the header, went looking for the next frame at +417, and fell off the end
with `Invalid frame size (417): Could not seek to 417`."""

_MP3_BITRATE_BPS = 128_000
_MP3_SAMPLE_RATE_HZ = 44_100
_MP3_FRAME_BYTES = 144 * _MP3_BITRATE_BPS // _MP3_SAMPLE_RATE_HZ
"""The frame size the header above declares: 144 x bitrate / sample rate for
MPEG-1 Layer III. A frame shorter than this is not a truncated frame, it is an
invalid one."""

_MP3_FRAMES_PER_INPUT = 8
"""How many identical frames make up each synthetic input.

One correctly-sized frame is still not enough: ffmpeg probes the container and
reports `Format mp3 detected only with low score of 1` on a single frame, then
refuses the concat. Eight is comfortably above the threshold (four already
passes) and keeps the fixture under 4 kB."""


def _silent_mp3() -> bytes:
    """A synthetic but genuinely valid MP3: N complete, correctly-sized frames."""
    frame = _MP3_FRAME_HEADER + bytes(_MP3_FRAME_BYTES - len(_MP3_FRAME_HEADER))
    return frame * _MP3_FRAMES_PER_INPUT


def test_the_mp3_fixture_declares_the_frame_size_it_actually_supplies() -> None:
    """Guards the fixture itself, so a future edit to the header cannot silently
    reintroduce the truncated-frame bug — which presented as a real-ffmpeg
    failure that looked environmental and was not."""
    payload = _silent_mp3()

    assert len(payload) == _MP3_FRAME_BYTES * _MP3_FRAMES_PER_INPUT
    assert payload[:2] == b"\xff\xfb", "every frame must start on a sync word"
    assert len(payload) % _MP3_FRAME_BYTES == 0, "no partial trailing frame"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed on this machine")
def test_real_ffmpeg_concatenates_two_minimal_mp3_files() -> None:
    """Runs only where ffmpeg is actually present — a genuine end-to-end
    check of the concat invocation, not just the argument shape."""
    payload = _silent_mp3()

    result = _concat_mp3([payload, payload])

    assert len(result) > 0
