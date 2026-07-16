"""Ingestor protocol + the shared HTTP fetcher (DESIGN §7 "Fetch mechanics").

Everything in `ingest/` is deterministic plain Python: fetch, parse, normalize.
This module never imports `llm.py` and never calls an LLM (CLAUDE.md §2,
NEVER rules 1-3).

Three things live here:

* **`Ingestor`** — the protocol every per-source adapter implements.
* **`HttpFetcher`** — one `httpx.AsyncClient` with conditional GET, `tenacity`
  retries honoring `Retry-After`, a concurrency limit, and raw-payload
  archiving into `data/http_cache/`.
* **`run_ingestors`** — the structural failure isolation (CLAUDE.md §7, NEVER
  rule 12). Ingestors are gathered concurrently and *every* exception becomes an
  `IngestError` record returned alongside whatever items succeeded. One broken
  source cannot abort a run because the runner never lets an exception escape.

### Where ETag state lives, and why

In sidecar JSON next to the archived payloads, under
`data/http_cache/<source>/_meta/<key>.json` — not in SQLite.

* `db.py` owns the schema (CLAUDE.md §3) and DESIGN §5 fixes its tables; an
  etag table would mean a migration in a module this one must not touch.
* Conditional-GET state is HTTP cache metadata about a *response*, and the
  response body already lives on disk here. Keeping the validator beside the
  body it validates means the two cannot disagree.
* It is regenerable: deleting `data/http_cache/` costs exactly one
  unconditional refetch per source. That is the correct blast radius for a
  cache, and it keeps `signalforge.db` free of state that isn't pipeline state.

`_meta/` deliberately sits *outside* the dated payload directories so the
90-day payload prune (DESIGN §7) never silently discards validators and forces a
full refetch of every source.

### Validators are committed, not written

A validator is a promise that we will never need the response again, so it is
only made durable once the caller confirms the items reached the database — see
`ValidatorStore` and `IngestRun.commit_validators`. `ingest/` stays free of
`db.py` (CLAUDE.md §2): the fetcher stages, the CLI confirms.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

from signalforge.models import Item, SourceType

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_USER_AGENT",
    "FetchError",
    "FetchResponse",
    "HttpFetcher",
    "IngestError",
    "IngestResult",
    "IngestRun",
    "Ingestor",
    "ValidatorStore",
    "run_ingestors",
    "truncate_summary",
]

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT: Final = "signalforge/0.1 (+https://github.com/signalforge; personal digest)"
DEFAULT_MAX_CONCURRENCY: Final = 4
DEFAULT_MAX_ATTEMPTS: Final = 3

# Transient by definition: a retry may plausibly succeed. 403 is absent on
# purpose — GitHub returns it for an exhausted rate limit whose reset is up to
# an hour away, and hammering it would make things worse, so it surfaces as a
# FetchError the ingestor records instead.
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_RETRY_AFTER_SECONDS: Final = 60.0
_UNSET: Final = object()
"""Sentinel distinguishing "no staged entry" from a staged deletion (None)."""
_UNSAFE_PATH_CHARS: Final = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# Errors and results
# --------------------------------------------------------------------------- #


def truncate_summary(
    value: object, *, max_chars: int, collapse_whitespace: bool = True
) -> str | None:
    """Normalize an arbitrary payload field into an `Item.summary`, or None.

    Takes `object` because it is applied straight to values decoded from feeds
    and JSON APIs, where a field documented as a string may be null, absent, or
    something else entirely. Non-strings and blanks become None rather than
    `"None"`.

    `max_chars` is required and comes from `defaults.max_summary_chars` in
    `sources.yaml` — it is the triage cost knob (triage reads titles + summaries
    only, DESIGN §8), so it is config, not a Python constant (NEVER rule 6).
    It is threaded in by the caller because loading config is not `ingest/`'s
    job.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = " ".join(value.split()) if collapse_whitespace else value.strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


class FetchError(Exception):
    """A non-retryable (or retry-exhausted) HTTP failure.

    Carries the status and headers so an ingestor can log context — e.g.
    GitHub's `x-ratelimit-reset` — before recording the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str,
        status_code: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.headers: Mapping[str, str] = dict(headers or {})


class _RetryableStatus(Exception):
    """Internal: a retryable status code, carrying any `Retry-After` hint."""

    def __init__(self, response: httpx.Response, retry_after: float | None) -> None:
        super().__init__(f"HTTP {response.status_code} from {response.request.url}")
        self.response = response
        self.retry_after = retry_after


class IngestError(BaseModel):
    """One structured per-source failure — the payload of `runs.errors`.

    Errors are *returned*, never swallowed (CLAUDE.md §7). `as_record()` gives
    the JSON-safe mapping `db.finish_run(errors=...)` expects.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: SourceType
    message: str
    error_type: str
    occurred_at: datetime

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        source_id: str,
        source_type: SourceType,
    ) -> IngestError:
        """Build a record from a caught exception. Never includes a traceback —
        `runs.errors` is read by a human in tomorrow's digest, not a debugger."""
        return cls(
            source_id=source_id,
            source_type=source_type,
            message=str(exc) or exc.__class__.__name__,
            error_type=exc.__class__.__name__,
            occurred_at=datetime.now(UTC),
        )

    def as_record(self) -> dict[str, str]:
        """JSON-serializable form for `runs.errors`."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "error_type": self.error_type,
            "message": self.message,
            "occurred_at": self.occurred_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What one ingestor (or a whole run) produced: items *and* errors.

    Both fields are always present. A partial success — 40 items and one dead
    feed — is the normal case, not an exceptional one.
    """

    items: list[Item] = field(default_factory=list)
    errors: list[IngestError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing failed."""
        return not self.errors

    @classmethod
    def merge(cls, results: Sequence[IngestResult]) -> IngestResult:
        """Concatenate results from several ingestors, order preserved."""
        items: list[Item] = []
        errors: list[IngestError] = []
        for result in results:
            items.extend(result.items)
            errors.extend(result.errors)
        return cls(items=items, errors=errors)


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #


class IngestRun:
    """A completed ingest: the items, the errors, and the uncommitted validators.

    What `ingest_all` hands back. It is `IngestResult` plus the one thing only
    the caller can decide — whether the fetch is *finished*, meaning the items
    actually reached the database.

    Until `commit_validators()` is called, nothing about this run's
    conditional-GET state is durable, so the next run refetches unconditionally
    and re-yields the same items (`upsert_item` dedupes them). Forgetting the
    call costs bandwidth; it cannot cost data. That is deliberate: the default
    must fail toward refetching.
    """

    def __init__(
        self,
        *,
        items: list[Item],
        errors: list[IngestError],
        validators: ValidatorStore,
    ) -> None:
        self.items = items
        self.errors = errors
        self.validators = validators

    @property
    def ok(self) -> bool:
        """True when no source failed."""
        return not self.errors

    @property
    def source_ids(self) -> set[str]:
        """Every source that produced at least one item."""
        return {item.source_id for item in self.items}

    def error_records(self) -> list[dict[str, str]]:
        """The `runs.errors` payload — JSON-ready, no custom encoder needed."""
        return [error.as_record() for error in self.errors]

    def commit_validators(self, source_ids: Iterable[str] | None = None) -> int:
        """Confirm these items are persisted; make their validators durable.

        Call this **after** `db.upsert_item` has succeeded — that is the whole
        point. Returns the number of validators written.

        Pass `source_ids` to confirm only some sources: if one source's writes
        failed, commit the rest and let that one refetch next run rather than
        forcing every source to refetch (CLAUDE.md §7).
        """
        return self.validators.commit(source_ids)


@runtime_checkable
class Ingestor(Protocol):
    """One source adapter.

    Implementations are constructed from `sources.yaml` (never hardcoded —
    CLAUDE.md §4) and hand back normalized `Item`s. An implementation *may*
    raise: `run_ingestors` converts that into an `IngestError`. Returning the
    error inside an `IngestResult` is preferred when the ingestor can keep some
    items, which is why the return type carries both.
    """

    source_id: str
    """Key into `sources.yaml`; becomes `Item.source_id`."""

    source_type: SourceType

    async def ingest(self, fetcher: HttpFetcher) -> IngestResult:
        """Fetch, parse, and normalize. Must not call an LLM (NEVER rule 2)."""
        ...


async def run_ingestors(
    ingestors: Sequence[Ingestor],
    fetcher: HttpFetcher,
) -> IngestResult:
    """Run every ingestor concurrently and collect `(items, errors)`.

    This is where failure isolation is *structural* rather than a convention:
    `return_exceptions=True` means a raising ingestor is delivered to us as a
    value, so there is no code path in which one source aborts the run
    (CLAUDE.md §7, NEVER rule 12). Concurrency is bounded inside `fetcher`.
    """
    if not ingestors:
        return IngestResult()

    outcomes = await asyncio.gather(
        *(ingestor.ingest(fetcher) for ingestor in ingestors),
        return_exceptions=True,
    )

    results: list[IngestResult] = []
    for ingestor, outcome in zip(ingestors, outcomes, strict=True):
        if isinstance(outcome, IngestResult):
            results.append(outcome)
            continue
        if isinstance(outcome, asyncio.CancelledError):  # pragma: no cover - shutdown path
            raise outcome
        logger.exception(
            "ingestor raised; recording as a source error and continuing",
            exc_info=outcome,
            extra={"source_id": ingestor.source_id},
        )
        results.append(
            IngestResult(
                errors=[
                    IngestError.from_exception(
                        outcome,
                        source_id=ingestor.source_id,
                        source_type=ingestor.source_type,
                    )
                ]
            )
        )
    return IngestResult.merge(results)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class ValidatorStore:
    """Conditional-GET validators, staged in memory until the caller commits.

    ### Why staging exists

    A validator written at fetch time is a promise that we will never need that
    response again. We cannot make that promise: `ingest_all` only *returns*
    items, and the CLI persists them afterwards. If the process dies between the
    two, an eagerly-written ETag makes the next run 304 — and a 304 yields
    nothing, so the items are gone for good.

    That defeats DESIGN §14's "a missed run self-heals on the next one": the
    7-day lookback can only heal a gap it is allowed to *see*, and a 304 hides
    it. So a validator becomes durable only once the caller confirms the items
    reached the database — `commit()`.

    ### Failing safe

    Every unhappy path here degrades to "refetch next run", never to "skip":

    * `commit()` never called → nothing durable → unconditional refetch.
    * one source's persistence failed → commit the others, refetch that one
      (per-source isolation, CLAUDE.md §7).
    * corrupt or unwritable sidecar → treated as absent → refetch.

    A wasted request is cheap; a silently dropped item is not.

    ### Deletions stage too

    Removing a validator is staged exactly like writing one, so *nothing* here
    touches the disk before `commit()` — that symmetry is what lets a caller
    (`--dry-run`) fetch without mutating the cache it promised not to touch.

    An uncommitted deletion leaves the old sidecar in place, which looks like it
    risks a stale 304 — it does not. Both deletion paths are only reachable
    *after* a 200 response to a request that already carried those exact
    validators, and a server that just answered 200 to a conditional request
    answers 200 to the identical one next run. The stale validator cannot
    produce a 304 it did not produce a moment ago.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        # value None == a staged deletion; a dict == staged validators to write.
        self._pending: dict[tuple[str, str], dict[str, str] | None] = {}

    def path_for(self, source_id: str, key: str) -> Path:
        """Sidecar location. Outside the dated payload dirs, so the 90-day prune
        cannot strand a validator and force a full refetch of every source."""
        return self.cache_dir / _safe_component(source_id) / "_meta" / f"{key}.json"

    def read(self, source_id: str, key: str) -> dict[str, str]:
        """The validators to send for one key; empty when there are none.

        A staged *deletion* is honored immediately — we have already decided
        that validator is wrong, so re-sending it could only produce a 304 we
        would have to distrust.

        A staged *write* is deliberately NOT honored: it is unconfirmed, so
        offering it back could earn a 304 on a re-fetch within the same run and
        yield nothing for a caller that expects items. Ignoring it falls through
        to the committed sidecar, which is the safe direction — at worst an
        unnecessary refetch, never a silent skip.
        """
        if self._pending.get((source_id, key), _UNSET) is None:
            return {}

        path = self.path_for(source_id, key)
        if not path.is_file():
            return {}
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt sidecar is not a source failure: drop it and refetch
            # unconditionally. The cache is regenerable by construction.
            logger.warning(
                "discarding unreadable http cache metadata",
                extra={"source_id": source_id, "path": str(path)},
            )
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(k): str(v)
            for k, v in data.items()
            if k in {"etag", "last_modified"} and isinstance(v, str)
        }

    def stage(self, source_id: str, key: str, response: httpx.Response) -> None:
        """Hold this response's validators pending a `commit()`.

        A response carrying no validators stages a *deletion*, so any existing
        sidecar is dropped on commit rather than left to describe a response we
        no longer hold.
        """
        etag = response.headers.get("etag")
        last_modified = response.headers.get("last-modified")
        if not etag and not last_modified:
            self._pending[(source_id, key)] = None
            return
        payload: dict[str, str] = {"url": str(response.request.url)}
        if etag:
            payload["etag"] = etag
        if last_modified:
            payload["last_modified"] = last_modified
        self._pending[(source_id, key)] = payload

    def invalidate(self, source_id: str, key: str) -> None:
        """Stage the removal of this key's validators, staged and committed alike.

        Like every other mutation here, it lands on `commit()` — a fetch that is
        never confirmed leaves the cache exactly as it found it.
        """
        self._pending[(source_id, key)] = None

    def pending_sources(self) -> set[str]:
        """Source ids with staged, uncommitted changes (writes or deletions)."""
        return {source_id for source_id, _ in self._pending}

    def pending_count(self) -> int:
        """Staged changes awaiting commit, counting writes and deletions alike."""
        return len(self._pending)

    def commit(self, source_ids: Iterable[str] | None = None) -> int:
        """Apply staged changes to disk. Returns how many were applied.

        Writes and deletions land together — this is the only method that
        touches the cache. `source_ids=None` commits everything; otherwise only
        the named sources, which is what lets 8 good sources keep their 304s
        while a 9th that failed to persist refetches. Applied entries are
        cleared, so a second call is a no-op rather than a double write.
        """
        selected = set(source_ids) if source_ids is not None else None
        applied = 0
        for (source_id, key), payload in list(self._pending.items()):
            if selected is not None and source_id not in selected:
                continue
            path = self.path_for(source_id, key)
            try:
                if payload is None:
                    path.unlink(missing_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            except OSError as exc:
                # Failing to apply a validator change only costs a refetch; it
                # must never fail the run that already produced good items.
                logger.warning(
                    "could not update http cache metadata; will refetch next run",
                    extra={"source_id": source_id, "path": str(path), "error": str(exc)},
                )
            else:
                applied += 1
            del self._pending[(source_id, key)]
        logger.debug("committed conditional-get validator changes", extra={"count": applied})
        return applied


@dataclass(frozen=True, slots=True)
class FetchResponse:
    """A successful, non-304 response plus where its bytes were archived."""

    url: str
    status_code: int
    content: bytes
    headers: Mapping[str, str]
    raw_path: str | None
    """Archive location relative to the cache root; goes on `Item.raw_path`."""

    @property
    def text(self) -> str:
        """Decoded body. Undecodable bytes are replaced, never raised on — a
        feed with one bad byte still has 30 good entries."""
        return self.content.decode("utf-8", errors="replace")


def _safe_component(value: str) -> str:
    """Make an arbitrary source_id (`Aider-AI/aider`) a safe path component."""
    cleaned = _UNSAFE_PATH_CHARS.sub("-", value).strip("-.")
    return cleaned or "unnamed"


def _cache_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _parse_retry_after(value: str | None) -> float | None:
    """Parse `Retry-After` — delta-seconds or an HTTP-date (RFC 9110)."""
    if not value:
        return None
    stripped = value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _wait_honoring_retry_after(state: RetryCallState) -> float:
    """Exponential backoff, floored by the server's `Retry-After` (DESIGN §7).

    The server's hint wins when it is longer than our backoff — that is the
    whole point of the header — but it is capped so a hostile or buggy value
    cannot stall a cron run for an hour.
    """
    base = wait_exponential(multiplier=1, min=1, max=30)(state)
    exc = state.outcome.exception() if state.outcome is not None else None
    if isinstance(exc, _RetryableStatus) and exc.retry_after is not None:
        return max(base, min(exc.retry_after, _MAX_RETRY_AFTER_SECONDS))
    return base


class HttpFetcher:
    """The shared async HTTP client for every ingestor.

    Responsibilities (DESIGN §7): conditional GET so an unchanged feed costs one
    304 and nothing else, retries with exponential backoff honoring
    `Retry-After`, a global concurrency cap for politeness, and archiving raw
    payloads so a parser bug can be fixed and replayed without refetching.

    Use as an async context manager; it owns its `httpx.AsyncClient`. Tests
    intercept at the transport layer via `respx`, so there is no client-injection
    seam to maintain here.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        timeout: float,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        validators: ValidatorStore | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.validators = validators or ValidatorStore(cache_dir)
        """Staged conditional-GET state. Outlives this fetcher on purpose: the
        caller commits it only after the items have been persisted."""
        self._max_attempts = max_attempts
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"user-agent": DEFAULT_USER_AGENT},
        )

    async def __aenter__(self) -> HttpFetcher:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying client."""
        await self._client.aclose()

    # -- cache layout ------------------------------------------------------- #

    def _archive_path(self, source_id: str, key: str, *, when: datetime) -> Path:
        day = when.strftime("%Y%m%d")
        return self.cache_dir / _safe_component(source_id) / day / f"{key}.raw"

    def _archive(self, source_id: str, key: str, content: bytes) -> str | None:
        """Write the raw payload; return its path relative to the cache root.

        A relative path keeps `items.raw_path` portable across machines and
        checkouts — the DB is regenerable, but a moved repo shouldn't orphan
        every archived payload. Best-effort: a full disk must not fail an
        otherwise good fetch, so a write error is logged, not raised.
        """
        path = self._archive_path(source_id, key, when=datetime.now(UTC))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        except OSError as exc:
            logger.warning(
                "could not archive raw payload",
                extra={"source_id": source_id, "path": str(path), "error": str(exc)},
            )
            return None
        return path.relative_to(self.cache_dir).as_posix()

    def invalidate(self, url: str, *, source_id: str, cache_key: str | None = None) -> None:
        """Forget the conditional-GET validators for one URL.

        For when a 200 is *technically* cacheable but useless to us, so a 304
        next run would be a wrong answer rather than a cheap one. The GitHub
        ingestor needs this: an empty `/releases` payload has a perfectly stable
        ETag, and caching it means every later run 304s and never reaches the
        `/tags` fallback the repo actually depends on.

        Only drops the validator sidecar; the archived payload is left alone.
        """
        self.validators.invalidate(source_id, _cache_key(cache_key or url))

    # -- fetching ----------------------------------------------------------- #

    async def _attempt(self, url: str, headers: Mapping[str, str]) -> httpx.Response:
        async with self._semaphore:
            response = await self._client.get(url, headers=dict(headers))
        if response.status_code in _RETRYABLE_STATUS:
            raise _RetryableStatus(
                response, _parse_retry_after(response.headers.get("retry-after"))
            )
        return response

    async def get(
        self,
        url: str,
        *,
        source_id: str,
        headers: Mapping[str, str] | None = None,
        conditional: bool = True,
        cache_key: str | None = None,
    ) -> FetchResponse | None:
        """GET `url`, returning None when the server says 304 Not Modified.

        A 304 yields no response object at all — that is the contract that makes
        "most daily RSS fetches return 304 and cost nothing" (DESIGN §7) true:
        callers cannot accidentally reparse an unchanged body, because there is
        no body to reparse.

        `cache_key` separates conditional-GET state for URLs that share a source
        (HN runs several queries under one `source_id`); it defaults to the URL.

        Raises `FetchError` on a non-2xx response, on retry exhaustion, or on a
        transport failure. Callers record that; they never let it escape a run.
        """
        key = _cache_key(cache_key or url)
        request_headers: dict[str, str] = dict(headers or {})

        if conditional:
            validators = self.validators.read(source_id, key)
            if etag := validators.get("etag"):
                request_headers["if-none-match"] = etag
            if last_modified := validators.get("last_modified"):
                request_headers["if-modified-since"] = last_modified

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_attempts),
                wait=_wait_honoring_retry_after,
                retry=retry_if_exception_type((_RetryableStatus, httpx.TransportError)),
                reraise=True,
            ):
                with attempt:
                    response = await self._attempt(url, request_headers)
        except _RetryableStatus as exc:
            raise FetchError(
                f"HTTP {exc.response.status_code} after {self._max_attempts} attempts",
                url=url,
                status_code=exc.response.status_code,
                headers=exc.response.headers,
            ) from exc
        except httpx.TransportError as exc:
            raise FetchError(f"transport error: {exc}", url=url) from exc

        if response.status_code == httpx.codes.NOT_MODIFIED:
            logger.debug(
                "not modified", extra={"source_id": source_id, "url": url, "status_code": 304}
            )
            return None

        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise FetchError(
                f"HTTP {response.status_code}",
                url=url,
                status_code=response.status_code,
                headers=response.headers,
            )

        if conditional:
            # Staged, not written: durable only once the caller confirms these
            # items were persisted. See `ValidatorStore`.
            self.validators.stage(source_id, key, response)

        raw_path = self._archive(source_id, key, response.content)
        logger.debug(
            "fetched",
            extra={
                "source_id": source_id,
                "url": url,
                "status_code": response.status_code,
                "bytes": len(response.content),
            },
        )
        return FetchResponse(
            url=str(response.request.url),
            status_code=response.status_code,
            content=response.content,
            headers=dict(response.headers),
            raw_path=raw_path,
        )
