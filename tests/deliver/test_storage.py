"""Tests for `deliver/storage.py` — hand-rolled SigV4 over httpx (Stage 5).

The known-answer test below is the load-bearing one: it calls
`_authorization_header` (the generic signing primitive, independent of any
R2-specific defaults) against AWS's own published SigV4 test vector
("get-vanilla" from the aws-sig-v4-test-suite, region `us-east-1`, service
`service` — not R2's `auto`/`s3`), so it validates the actual cryptography
against a signature this module did not compute, not merely against
itself. Every other test in this file checks *wiring* — the right method,
headers, and URL shape reaching httpx — on the strength of that one proof
that the underlying signature math is correct.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from pydantic import SecretStr

from signalforge.deliver.storage import (
    StorageError,
    _authorization_header,
    delete_object,
    list_object_keys,
    put_object,
)

ENDPOINT = "https://account123.r2.cloudflarestorage.com"
BUCKET = "signalforge-podcast"
ACCESS_KEY = SecretStr("test-access-key-id")
SECRET_KEY = SecretStr("test-secret-access-key")
NOW = datetime(2026, 8, 7, 6, 0, 0, tzinfo=UTC)


def _no_sleep(_seconds: float) -> None:
    pass


# --------------------------------------------------------------------------- #
# Known-answer test — AWS's own published SigV4 test vector
# --------------------------------------------------------------------------- #


def test_authorization_header_matches_the_published_aws_sigv4_test_vector() -> None:
    """The "get-vanilla" case from the AWS SigV4 test suite: a bare GET to
    `/` on `example.amazonaws.com`, dated 2015-08-30T12:36:00Z, region
    `us-east-1`, service `service`, signed with the well-known
    `AKIDEXAMPLE` / `wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY` example
    credential pair AWS publishes throughout its own documentation."""
    now = datetime(2015, 8, 30, 12, 36, 0, tzinfo=UTC)
    empty_payload_hash = hashlib.sha256(b"").hexdigest()

    authorization = _authorization_header(
        method="GET",
        canonical_uri="/",
        query_params={},
        headers={"host": "example.amazonaws.com", "x-amz-date": "20150830T123600Z"},
        payload_hash=empty_payload_hash,
        access_key="AKIDEXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        now=now,
        region="us-east-1",
        service="service",
    )

    assert authorization == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
        "SignedHeaders=host;x-amz-date, "
        "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
    )


def test_a_different_secret_key_produces_a_different_signature() -> None:
    """Guards against a signer that ignores the secret key entirely — the
    kind of bug a single fixed known-answer test alone would not catch if
    the "expected" value were computed with the same broken code."""
    now = datetime(2015, 8, 30, 12, 36, 0, tzinfo=UTC)
    empty_payload_hash = hashlib.sha256(b"").hexdigest()
    headers = {"host": "example.amazonaws.com", "x-amz-date": "20150830T123600Z"}

    def sign(secret_key: str) -> str:
        return _authorization_header(
            method="GET",
            canonical_uri="/",
            query_params={},
            headers=headers,
            payload_hash=empty_payload_hash,
            access_key="AKIDEXAMPLE",
            secret_key=secret_key,
            now=now,
            region="us-east-1",
            service="service",
        )

    assert sign("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY") != sign("a-completely-different-secret")


# --------------------------------------------------------------------------- #
# put_object — request shape and idempotent overwrite semantics
# --------------------------------------------------------------------------- #


@respx.mock
def test_put_object_sends_the_body_to_the_right_path_style_url() -> None:
    route = respx.put(f"{ENDPOINT}/{BUCKET}/signalforge-2026-08-07.mp3").mock(
        return_value=httpx.Response(200)
    )

    with httpx.Client() as client:
        put_object(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            key="signalforge-2026-08-07.mp3",
            body=b"fake mp3 bytes",
            content_type="audio/mpeg",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )

    assert route.call_count == 1
    sent = route.calls.last.request
    assert sent.content == b"fake mp3 bytes"
    assert sent.headers["content-type"] == "audio/mpeg"


@respx.mock
def test_put_object_signs_a_content_sha256_header_matching_the_body() -> None:
    route = respx.put(f"{ENDPOINT}/{BUCKET}/feed.xml").mock(return_value=httpx.Response(200))
    body = b"<rss></rss>"

    with httpx.Client() as client:
        put_object(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            key="feed.xml",
            body=body,
            content_type="application/rss+xml",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )

    sent = route.calls.last.request
    assert sent.headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
    assert "AWS4-HMAC-SHA256" in sent.headers["authorization"]


@respx.mock
def test_a_trailing_slash_on_the_endpoint_does_not_double_the_path_separator() -> None:
    """Before this was fixed, only the signed `Host` header stripped a
    trailing slash (`_host`) — the actual request URL still concatenated
    the raw endpoint, so `https://host.example/` produced a request path of
    `//bucket/key`, mismatched against what was signed."""
    endpoint_with_slash = f"{ENDPOINT}/"
    route = respx.put(f"{ENDPOINT}/{BUCKET}/feed.xml").mock(return_value=httpx.Response(200))

    with httpx.Client() as client:
        put_object(
            client,
            endpoint=endpoint_with_slash,
            bucket=BUCKET,
            key="feed.xml",
            body=b"x",
            content_type="application/rss+xml",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )

    assert route.call_count == 1
    assert "//" not in route.calls.last.request.url.path


@respx.mock
def test_a_key_containing_a_slash_signs_and_requests_the_same_path() -> None:
    """The normal shape after `deliver.podcast._r2_prefix`: a key like
    `some-prefix/signalforge-2026-08-07.mp3`. `_canonical_uri` encodes each
    path segment separately and rejoins with `/`, so this must reach the
    server as one nested path, not a percent-encoded slash that would
    request a literal `%2F`-containing single segment instead."""
    route = respx.put(f"{ENDPOINT}/{BUCKET}/some-prefix/signalforge-2026-08-07.mp3").mock(
        return_value=httpx.Response(200)
    )

    with httpx.Client() as client:
        put_object(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            key="some-prefix/signalforge-2026-08-07.mp3",
            body=b"audio",
            content_type="audio/mpeg",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )

    assert route.call_count == 1
    assert route.calls.last.request.url.path == f"/{BUCKET}/some-prefix/signalforge-2026-08-07.mp3"


@respx.mock
def test_a_second_put_to_the_same_key_overwrites_rather_than_erroring() -> None:
    """The idempotency lever the plan names: re-uploading a key just
    replaces it, no special-casing needed (CLAUDE.md §3, NEVER rule 4)."""
    route = respx.put(f"{ENDPOINT}/{BUCKET}/feed.xml").mock(return_value=httpx.Response(200))

    with httpx.Client() as client:
        for body in (b"<rss>v1</rss>", b"<rss>v2</rss>"):
            put_object(
                client,
                endpoint=ENDPOINT,
                bucket=BUCKET,
                key="feed.xml",
                body=body,
                content_type="application/rss+xml",
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY,
                now=NOW,
                sleep=_no_sleep,
            )

    assert route.call_count == 2
    assert route.calls.last.request.content == b"<rss>v2</rss>"


@respx.mock
def test_a_put_refusal_raises_storage_error() -> None:
    respx.put(f"{ENDPOINT}/{BUCKET}/feed.xml").mock(
        return_value=httpx.Response(403, text="SignatureDoesNotMatch")
    )

    with httpx.Client() as client, pytest.raises(StorageError, match="403"):
        put_object(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            key="feed.xml",
            body=b"x",
            content_type="application/rss+xml",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )


# --------------------------------------------------------------------------- #
# delete_object — 404 is success, not failure
# --------------------------------------------------------------------------- #


@respx.mock
def test_delete_object_succeeds_on_200() -> None:
    respx.delete(f"{ENDPOINT}/{BUCKET}/old.mp3").mock(return_value=httpx.Response(204))

    with httpx.Client() as client:
        delete_object(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            key="old.mp3",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )  # must not raise


@respx.mock
def test_delete_object_treats_a_404_as_already_gone_not_a_failure() -> None:
    """A retried prune step whose first attempt already deleted the key
    must not fail the second time."""
    respx.delete(f"{ENDPOINT}/{BUCKET}/already-gone.mp3").mock(return_value=httpx.Response(404))

    with httpx.Client() as client:
        delete_object(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            key="already-gone.mp3",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )  # must not raise


@respx.mock
def test_delete_object_raises_on_a_real_failure() -> None:
    respx.delete(f"{ENDPOINT}/{BUCKET}/x.mp3").mock(return_value=httpx.Response(500))

    with httpx.Client() as client, pytest.raises(StorageError):
        delete_object(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            key="x.mp3",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )


# --------------------------------------------------------------------------- #
# list_object_keys — query shape and XML parsing
# --------------------------------------------------------------------------- #


_LIST_RESPONSE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>signalforge-podcast</Name>
  <Prefix>signalforge-</Prefix>
  <Contents><Key>signalforge-2026-08-05.mp3</Key></Contents>
  <Contents><Key>signalforge-2026-08-06.mp3</Key></Contents>
  <Contents><Key>signalforge-2026-08-07.mp3</Key></Contents>
</ListBucketResult>"""


@respx.mock
def test_list_object_keys_parses_every_key_from_the_xml_response() -> None:
    respx.get(f"{ENDPOINT}/{BUCKET}").mock(
        return_value=httpx.Response(200, content=_LIST_RESPONSE_XML)
    )

    with httpx.Client() as client:
        keys = list_object_keys(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            prefix="signalforge-",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )

    assert keys == [
        "signalforge-2026-08-05.mp3",
        "signalforge-2026-08-06.mp3",
        "signalforge-2026-08-07.mp3",
    ]


@respx.mock
def test_list_object_keys_sends_list_type_2_and_the_prefix_as_query_params() -> None:
    route = respx.get(f"{ENDPOINT}/{BUCKET}").mock(
        return_value=httpx.Response(200, content=b"<ListBucketResult></ListBucketResult>")
    )

    with httpx.Client() as client:
        list_object_keys(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            prefix="signalforge-",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )

    query = parse_qs(urlsplit(str(route.calls.last.request.url)).query)
    assert query["list-type"] == ["2"]
    assert query["prefix"] == ["signalforge-"]


@respx.mock
def test_list_object_keys_on_an_empty_bucket_returns_an_empty_list() -> None:
    respx.get(f"{ENDPOINT}/{BUCKET}").mock(
        return_value=httpx.Response(
            200,
            content=b'<?xml version="1.0"?><ListBucketResult '
            b'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"></ListBucketResult>',
        )
    )

    with httpx.Client() as client:
        keys = list_object_keys(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            prefix="signalforge-",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )

    assert keys == []


# --------------------------------------------------------------------------- #
# Retry policy — idempotent operations retry on any transport error
# --------------------------------------------------------------------------- #


@respx.mock
def test_put_object_retries_a_transient_status_then_succeeds() -> None:
    route = respx.put(f"{ENDPOINT}/{BUCKET}/feed.xml").mock(
        side_effect=[httpx.Response(503), httpx.Response(200)]
    )

    with httpx.Client() as client:
        put_object(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            key="feed.xml",
            body=b"x",
            content_type="application/rss+xml",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )

    assert route.call_count == 2


@respx.mock
def test_put_object_retries_a_read_timeout_because_the_operation_is_idempotent() -> None:
    """Unlike `deliver.tts`, a PUT replay just overwrites the same bytes —
    safe to retry even when the response (not just the connection) was lost."""
    route = respx.put(f"{ENDPOINT}/{BUCKET}/feed.xml").mock(
        side_effect=[httpx.ReadTimeout("lost"), httpx.Response(200)]
    )

    with httpx.Client() as client:
        put_object(
            client,
            endpoint=ENDPOINT,
            bucket=BUCKET,
            key="feed.xml",
            body=b"x",
            content_type="application/rss+xml",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            now=NOW,
            sleep=_no_sleep,
        )

    assert route.call_count == 2
