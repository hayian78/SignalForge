"""Cloudflare R2 object storage — SigV4 PUT/DELETE/LIST, hand-signed over httpx.

Part of the podcast channel, the second recorded phase-gate exception
(DESIGN §13.3). No `boto3`: R2 is S3-compatible, and the podcast channel needs exactly three
operations against exactly one bucket, so a full AWS SDK would be a large,
general-purpose dependency in service of a narrow, fixed need (CLAUDE.md §9
— no dependency the design explicitly doesn't call for). AWS Signature
Version 4 is a stable, documented algorithm (unchanged since 2011); the risk
of hand-signing is confined to this one module and is caught by
`test_storage.py`'s known-answer vectors, computed independently rather than
by round-tripping this module's own output through itself.

Path-style addressing throughout (`{endpoint}/{bucket}/{key}`), not virtual-
hosted (`{bucket}.{endpoint}`): `r2_endpoint` is the account-level endpoint,
and path-style avoids depending on the bucket name being a valid DNS label.
Region is always `"auto"` — R2's own fixed pseudo-region, confirmed against
Cloudflare's docs (2026-08-07); there is nothing to configure here.

`public_base_url` (the unguessable-prefix URL a podcast app subscribes to)
is a *different* URL from anything in this module: this module authenticates
against `r2_endpoint` to write and delete objects, while `public_base_url`
is what a listener's phone reads from, unauthenticated. `deliver/podcast.py`
is the only place those two need to meet.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final
from urllib.parse import quote

import httpx
from pydantic import SecretStr
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

from signalforge.ingest.base import DEFAULT_USER_AGENT, parse_retry_after

__all__ = [
    "StorageError",
    "delete_object",
    "list_object_keys",
    "put_object",
]

logger = logging.getLogger(__name__)

_SIGNING_REGION: Final = "auto"
_SIGNING_SERVICE: Final = "s3"

_MAX_ATTEMPTS: Final = 3
_MAX_RETRY_AFTER_SECONDS: Final = 60.0
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
"""Deliberately local, not shared with `deliver.email`/`deliver.tts` — see
`deliver.email._RETRYABLE_STATUS`'s docstring for why a policy set is not
worth sharing even when the values match."""


class StorageError(Exception):
    """An R2 operation failed after retries, or was refused outright.

    Every operation here is idempotent (a PUT to a key overwrites; a DELETE
    of an absent key is not an error condition worth distinguishing from one
    that existed), so — unlike `deliver.tts.SynthesisError` — retrying on
    *any* transport error, not just connect-phase ones, is safe: replaying a
    PUT whose response was lost just overwrites the same bytes again.
    """


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_digest(key: bytes, data: str) -> bytes:
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret_key: str, date_stamp: str, *, region: str, service: str) -> bytes:
    """The 4-step HMAC chain AWS SigV4 derives a request-scoped key through —
    date, region, service, `aws4_request` — so the raw secret key is used
    exactly once, here, and never appears in the request itself.

    `region`/`service` are parameters, not `_SIGNING_REGION`/`_SIGNING_SERVICE`
    directly, so this function (and `_authorization_header` below) can be
    exercised against AWS's own published SigV4 test vectors — which use
    `us-east-1`/`service`, not R2's `auto`/`s3` — as a genuine known-answer
    test, rather than one that can only ever check this module against
    itself. `_signed_headers` is where R2's actual values get supplied.
    """
    k_date = _hmac_digest(f"AWS4{secret_key}".encode(), date_stamp)
    k_region = _hmac_digest(k_date, region)
    k_service = _hmac_digest(k_region, service)
    return _hmac_digest(k_service, "aws4_request")


def _canonical_uri(path: str) -> str:
    """URI-encode each path segment, preserving `/` separators — SigV4's
    canonical URI must encode reserved characters in a key (spaces, `%`,
    etc.) without touching the segment boundaries the signature also covers.
    """
    return "/".join(quote(segment, safe="") for segment in path.split("/"))


def _canonical_query_string(params: dict[str, str]) -> str:
    return "&".join(
        f"{quote(key, safe='-_.~')}={quote(value, safe='-_.~')}"
        for key, value in sorted(params.items())
    )


def _authorization_header(
    *,
    method: str,
    canonical_uri: str,
    query_params: dict[str, str],
    headers: dict[str, str],
    payload_hash: str,
    access_key: str,
    secret_key: str,
    now: datetime,
    region: str = _SIGNING_REGION,
    service: str = _SIGNING_SERVICE,
) -> str:
    """The full `Authorization` header value for one request.

    `headers` must already contain every header the request will actually
    send that needs to be signed (lowercase keys) — `host`,
    `x-amz-content-sha256`, `x-amz-date`, and `content-type` for a PUT. This
    function does not add anything to the request; it only computes the
    signature over what the caller already built, so there is exactly one
    place a header could be sent unsigned: the caller forgetting to include
    it in both the request and this dict, not a mismatch between the two.
    """
    signed_header_names = sorted(headers)
    canonical_headers = "".join(f"{name}:{headers[name].strip()}\n" for name in signed_header_names)
    signed_headers = ";".join(signed_header_names)
    canonical_request = "\n".join(
        [
            method,
            canonical_uri,
            _canonical_query_string(query_params),
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    # `now.astimezone(UTC)` rather than trusting the caller — the same
    # normalization `_signed_headers` applies to the `x-amz-date` *header*
    # value, done again here so the credential scope's date and the
    # string-to-sign's timestamp can never disagree with it.
    now = now.astimezone(UTC)
    date_stamp = now.strftime("%Y%m%d")
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            _sha256_hex(canonical_request.encode("utf-8")),
        ]
    )
    signing_key = _signing_key(secret_key, date_stamp, region=region, service=service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )


def _signed_headers(
    *,
    method: str,
    host: str,
    path: str,
    query_params: dict[str, str],
    payload: bytes,
    content_type: str | None,
    access_key: str,
    secret_key: str,
    now: datetime,
) -> dict[str, str]:
    """Every header one signed request needs, including `Authorization`.

    `now` is converted to UTC here rather than trusted — `strftime` formats
    whatever wall-clock fields a datetime carries and this function always
    appends a literal `Z`, so a caller passing a non-UTC-aware (or naive)
    `now` would sign the wrong instant and get an opaque 403 from R2, not a
    type error."""
    payload_hash = _sha256_hex(payload)
    amz_date = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    to_sign = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    if content_type is not None:
        to_sign["content-type"] = content_type
    authorization = _authorization_header(
        method=method,
        canonical_uri=_canonical_uri(path),
        query_params=query_params,
        headers=to_sign,
        payload_hash=payload_hash,
        access_key=access_key,
        secret_key=secret_key,
        now=now,
    )
    return {**to_sign, "Authorization": authorization, "User-Agent": DEFAULT_USER_AGENT}


class _RetryableStatus(Exception):
    def __init__(self, status_code: int, retry_after: float | None) -> None:
        super().__init__(f"r2 returned HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


def _wait_honoring_retry_after(state: RetryCallState) -> float:
    """Exponential backoff, floored by the provider's `Retry-After` — same
    rule as `deliver.email._wait_honoring_retry_after`."""
    base: float = wait_exponential(multiplier=1, min=1, max=10)(state)
    exc = state.outcome.exception() if state.outcome is not None else None
    if isinstance(exc, _RetryableStatus) and exc.retry_after is not None:
        return max(base, min(exc.retry_after, _MAX_RETRY_AFTER_SECONDS))
    return base


def _request_with_retry(
    client: httpx.Client,
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    content: bytes | None,
    sleep: Callable[[int | float], None],
) -> httpx.Response:
    def once() -> httpx.Response:
        response = client.request(method, url, headers=headers, content=content)
        if response.status_code in _RETRYABLE_STATUS:
            raise _RetryableStatus(
                response.status_code, parse_retry_after(response.headers.get("retry-after"))
            )
        return response

    try:
        for attempt in Retrying(
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            wait=_wait_honoring_retry_after,
            retry=retry_if_exception_type((_RetryableStatus, httpx.TransportError)),
            sleep=sleep,
            reraise=True,
        ):
            with attempt:
                return once()
    except (_RetryableStatus, httpx.TransportError) as exc:
        raise StorageError(f"r2 {method} failed after {_MAX_ATTEMPTS} attempts: {exc}") from exc
    raise StorageError("r2 request produced no result")  # pragma: no cover


def put_object(
    client: httpx.Client,
    *,
    endpoint: str,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
    access_key: SecretStr,
    secret_key: SecretStr,
    now: datetime,
    sleep: Callable[[int | float], None] = time.sleep,
) -> None:
    """Write `body` to `key`, overwriting whatever was there — PUT is the
    idempotency lever for both the audio file and `feed.xml` (CLAUDE.md §3):
    re-running a delivery re-uploads the same bytes to the same key rather
    than accumulating versions."""
    endpoint = endpoint.rstrip("/")
    host = _host(endpoint)
    path = f"/{bucket}/{key}"
    headers = _signed_headers(
        method="PUT",
        host=host,
        path=path,
        query_params={},
        payload=body,
        content_type=content_type,
        access_key=access_key.get_secret_value(),
        secret_key=secret_key.get_secret_value(),
        now=now,
    )
    response = _request_with_retry(
        client, method="PUT", url=f"{endpoint}{path}", headers=headers, content=body, sleep=sleep
    )
    if not 200 <= response.status_code < 300:
        raise StorageError(
            f"r2 put {key!r} returned HTTP {response.status_code}: {response.text[:300]}"
        )
    logger.info("uploaded object to r2", extra={"bucket": bucket, "key": key, "bytes": len(body)})


def delete_object(
    client: httpx.Client,
    *,
    endpoint: str,
    bucket: str,
    key: str,
    access_key: SecretStr,
    secret_key: SecretStr,
    now: datetime,
    sleep: Callable[[int | float], None] = time.sleep,
) -> None:
    """Delete `key`. A 404 is treated as success — S3's own DELETE semantics
    are "the key does not exist" either way, and retrying a delivery whose
    prune step half-completed must not fail on a key an earlier attempt
    already removed."""
    endpoint = endpoint.rstrip("/")
    host = _host(endpoint)
    path = f"/{bucket}/{key}"
    headers = _signed_headers(
        method="DELETE",
        host=host,
        path=path,
        query_params={},
        payload=b"",
        content_type=None,
        access_key=access_key.get_secret_value(),
        secret_key=secret_key.get_secret_value(),
        now=now,
    )
    response = _request_with_retry(
        client, method="DELETE", url=f"{endpoint}{path}", headers=headers, content=None, sleep=sleep
    )
    if not (200 <= response.status_code < 300 or response.status_code == 404):
        raise StorageError(
            f"r2 delete {key!r} returned HTTP {response.status_code}: {response.text[:300]}"
        )
    logger.info("deleted object from r2", extra={"bucket": bucket, "key": key})


def list_object_keys(
    client: httpx.Client,
    *,
    endpoint: str,
    bucket: str,
    prefix: str,
    access_key: SecretStr,
    secret_key: SecretStr,
    now: datetime,
    sleep: Callable[[int | float], None] = time.sleep,
) -> list[str]:
    """Every key under `prefix` in `bucket`.

    Not paginated: `ListObjectsV2`'s default page is 1,000 keys, and this
    bucket only ever holds one `feed.xml` plus at most
    `PodcastChannelConfig.retention_episodes` (capped at 90) audio files —
    orders of magnitude under that, so pagination would be complexity with
    no reachable case to exercise it.
    """
    endpoint = endpoint.rstrip("/")
    host = _host(endpoint)
    path = f"/{bucket}"
    query_params = {"list-type": "2", "prefix": prefix}
    headers = _signed_headers(
        method="GET",
        host=host,
        path=path,
        query_params=query_params,
        payload=b"",
        content_type=None,
        access_key=access_key.get_secret_value(),
        secret_key=secret_key.get_secret_value(),
        now=now,
    )
    url = f"{endpoint}{path}?{_canonical_query_string(query_params)}"
    response = _request_with_retry(
        client, method="GET", url=url, headers=headers, content=None, sleep=sleep
    )
    if not 200 <= response.status_code < 300:
        raise StorageError(
            f"r2 list {prefix!r} returned HTTP {response.status_code}: {response.text[:300]}"
        )
    return _parse_list_keys(response.content)


def _host(endpoint: str) -> str:
    """`endpoint` is `https://<account-id>.r2.cloudflarestorage.com` — the
    signed `Host` header is the same string minus the scheme, since SigV4
    signs the header value the server actually receives, not the URL."""
    return endpoint.removeprefix("https://").removeprefix("http://").rstrip("/")


def _local_name(tag: str) -> str:
    """Strip an XML namespace URI from `{uri}LocalName`, so parsing does not
    depend on whether R2's `ListObjectsV2` response declares the S3 XML
    namespace exactly the way AWS's own service does."""
    return tag.rsplit("}", 1)[-1]


def _parse_list_keys(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)  # noqa: S314 - R2's own response, not arbitrary untrusted XML
    return [elem.text for elem in root.iter() if _local_name(elem.tag) == "Key" and elem.text]
