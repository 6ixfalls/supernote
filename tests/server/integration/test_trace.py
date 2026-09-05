import asyncio
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient

from supernote.server.app import (
    ACCESS_LOG_FORMAT,
    OMITTED_SENSITIVE_BODY,
    TRACE_BODY_LOG_LIMIT,
    _bounded_body,
    _redact_url,
    _sanitize_body,
    _sanitize_headers,
    is_binary_content_type,
    try_parse_json,
)


@pytest.fixture
def mock_trace_log(tmp_path: Path) -> str:
    """Enable trace log for this module."""
    log_file = tmp_path / "trace.log"
    return str(log_file)


async def test_trace_logging(
    client: TestClient,
    mock_trace_log: str,
) -> None:
    """Verify that requests and responses are logged to the trace log."""

    # Make a simple request
    resp = await client.get("/api/file/query/server")
    assert resp.status == 200
    resp_text = await resp.text()

    # Check trace log
    log_path = Path(mock_trace_log)
    content = ""
    # Wait for the async file write thread to finish writing and flushing
    for _ in range(20):
        if log_path.exists():
            content = log_path.read_text().strip()
            if "/api/file/query/server" in content:
                break
        await asyncio.sleep(0.05)

    assert log_path.exists()
    entry = json.loads(content)

    # Check request fields
    assert "request" in entry
    assert entry["request"]["method"] == "GET"
    assert "/api/file/query/server" in entry["request"]["url"]

    # Check response fields
    assert "response" in entry
    assert entry["response"]["status"] == 200
    assert "headers" in entry["response"]

    # Check response body match
    logged_body = entry["response"]["body"]
    assert isinstance(logged_body, dict)  # Should be parsed as dict
    assert logged_body == json.loads(resp_text)


def test_try_parse_json() -> None:
    # Valid JSON
    assert try_parse_json('{"a": 1}') == {"a": 1}
    # Invalid JSON
    assert try_parse_json('{"a": 1') == '{"a": 1'
    # None
    assert try_parse_json(None) is None
    # A number json string
    assert try_parse_json("123") == 123


def test_binary_content_type_check() -> None:
    assert is_binary_content_type("application/octet-stream")
    assert is_binary_content_type("image/png")
    assert is_binary_content_type("application/pdf")
    assert not is_binary_content_type("application/json")
    assert not is_binary_content_type("text/plain")
    assert not is_binary_content_type("text/html")


def test_trace_redaction_is_case_insensitive_and_recursive() -> None:
    headers = _sanitize_headers(
        {
            "AUTHORIZATION": "Bearer secret",
            "X-Access-Token": "jwt",
            "Cookie": "session=secret",
            "Content-Type": "application/json",
        }
    )
    assert headers == {
        "AUTHORIZATION": "***",
        "X-Access-Token": "***",
        "Cookie": "***",
        "Content-Type": "application/json",
    }

    body = _sanitize_body(
        '{"password":"one","nested":{"accessToken":"two"},'
        '"items":[{"clientSecret":"three"}],"safe":"visible"}'
    )
    assert body == {
        "password": "***",
        "nested": {"accessToken": "***"},
        "items": [{"clientSecret": "***"}],
        "safe": "visible",
    }


def test_trace_url_redaction_handles_case_and_duplicate_credentials() -> None:
    redacted = _redact_url(
        "https://example.test/path?TOKEN=one&token=two&Signature=three&safe=yes"
    )
    assert "one" not in redacted
    assert "two" not in redacted
    assert "three" not in redacted
    assert "safe=yes" in redacted


def test_trace_body_is_bounded() -> None:
    logged = _bounded_body(b"x" * (TRACE_BODY_LOG_LIMIT + 100))
    assert logged == ("x" * TRACE_BODY_LOG_LIMIT) + "... (truncated)"


def test_access_log_does_not_include_request_target_or_referrer() -> None:
    assert "%r" not in ACCESS_LOG_FORMAT
    assert "Referer" not in ACCESS_LOG_FORMAT


async def test_sensitive_route_bodies_are_omitted(
    client: TestClient,
    mock_trace_log: str,
) -> None:
    password = "must-not-appear"
    response = await client.post(
        "/api/official/user/account/login/new",
        json={
            "account": "nobody@example.com",
            "password": password,
            "timestamp": "1",
            "loginMethod": "2",
        },
    )
    assert response.status == 401

    content = Path(mock_trace_log).read_text()
    assert password not in content
    entry = json.loads(content.strip())
    assert entry["request"]["body"] == OMITTED_SENSITIVE_BODY
    assert entry["response"]["body"] == OMITTED_SENSITIVE_BODY
