"""Tests for GitHubClient: retry logic, rate-limit handling, timeouts, error responses.

All tests use mocked HTTP transports — zero network access required.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
import pytest

from prospectpilot.github_client import (
    GitHubAPIError,
    GitHubClient,
    RateLimitExhausted,
    RateLimitInfo,
)


# ------------------------------------------------------------------
# Helpers: mock transport for httpx
# ------------------------------------------------------------------

class MockTransport(httpx.BaseTransport):
    """Programmable httpx transport for testing without network access."""

    def __init__(self, responses: Optional[List[httpx.Response]] = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_count = 0
        self.requests: List[httpx.Request] = []

    def add_response(
        self,
        status_code: int = 200,
        json_data: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        hdrs = dict(headers or {})
        body = b""
        if json_data is not None:
            import json as _json
            body = _json.dumps(json_data).encode()
            hdrs.setdefault("content-type", "application/json")
        self._responses.append(
            httpx.Response(status_code=status_code, headers=hdrs, content=body)
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self._call_count += 1
        if not self._responses:
            raise httpx.TimeoutException("No more mocked responses")
        return self._responses.pop(0)

    @property
    def call_count(self) -> int:
        return self._call_count


def _make_client(transport: MockTransport, **kwargs: Any) -> GitHubClient:
    """Create a GitHubClient with a mocked transport (no real network)."""
    client = GitHubClient(token="test-token-fake", max_retries=kwargs.get("max_retries", 2), **{k: v for k, v in kwargs.items() if k != "max_retries"})
    # Replace the internal httpx.Client with one using our transport
    client._client = httpx.Client(
        base_url=GitHubClient.BASE_URL,
        headers=client._client.headers,
        timeout=client._client.timeout,
        transport=transport,
    )
    return client


# ------------------------------------------------------------------
# Tests: successful responses
# ------------------------------------------------------------------

def test_search_repositories_success() -> None:
    transport = MockTransport()
    transport.add_response(
        json_data={"total_count": 1, "items": [{"full_name": "org/repo"}]},
        headers={"x-ratelimit-remaining": "50", "x-ratelimit-limit": "60", "x-ratelimit-reset": str(int(time.time()) + 3600)},
    )
    client = _make_client(transport)
    result = client.search_repositories("topic:cli")
    assert result["total_count"] == 1
    assert result["items"][0]["full_name"] == "org/repo"
    assert transport.call_count == 1


def test_get_repository_success() -> None:
    transport = MockTransport()
    transport.add_response(
        json_data={"full_name": "org/repo", "stargazers_count": 100},
        headers={"x-ratelimit-remaining": "50", "x-ratelimit-limit": "60", "x-ratelimit-reset": str(int(time.time()) + 3600)},
    )
    client = _make_client(transport)
    result = client.get_repository("org", "repo")
    assert result["full_name"] == "org/repo"


def test_get_organization_success() -> None:
    transport = MockTransport()
    transport.add_response(
        json_data={"login": "hashicorp", "name": "HashiCorp"},
        headers={"x-ratelimit-remaining": "50", "x-ratelimit-limit": "60", "x-ratelimit-reset": str(int(time.time()) + 3600)},
    )
    client = _make_client(transport)
    result = client.get_organization("hashicorp")
    assert result["login"] == "hashicorp"


def test_get_releases_success() -> None:
    transport = MockTransport()
    transport.add_response(
        json_data=[{"tag_name": "v1.0.0"}, {"tag_name": "v0.9.0"}],
        headers={"x-ratelimit-remaining": "50", "x-ratelimit-limit": "60", "x-ratelimit-reset": str(int(time.time()) + 3600)},
    )
    client = _make_client(transport)
    result = client.get_releases("org", "repo", per_page=2)
    assert len(result) == 2
    assert result[0]["tag_name"] == "v1.0.0"


# ------------------------------------------------------------------
# Tests: error handling
# ------------------------------------------------------------------

def test_404_raises_github_api_error() -> None:
    transport = MockTransport()
    transport.add_response(status_code=404, json_data={"message": "Not Found"})
    client = _make_client(transport)
    with pytest.raises(GitHubAPIError, match="not found"):
        client.get_organization("nonexistent-org-xyz")
    assert transport.call_count == 1  # No retry on 404


def test_422_raises_without_retry() -> None:
    transport = MockTransport()
    transport.add_response(status_code=422, json_data={"message": "Validation Failed"})
    client = _make_client(transport)
    with pytest.raises(GitHubAPIError, match="422"):
        client.search_repositories("invalid query with bad syntax \\\\\\")
    assert transport.call_count == 1  # No retry on 422


# ------------------------------------------------------------------
# Tests: retry logic
# ------------------------------------------------------------------

def test_retry_on_500_then_success() -> None:
    """Verifies retry on server error, succeeding on second attempt."""
    transport = MockTransport()
    transport.add_response(status_code=500, json_data={"message": "Internal Server Error"})
    transport.add_response(
        json_data={"login": "recovered"},
        headers={"x-ratelimit-remaining": "50", "x-ratelimit-limit": "60", "x-ratelimit-reset": str(int(time.time()) + 3600)},
    )
    # Use max_retries=1 and patch sleep to avoid waiting
    client = _make_client(transport, max_retries=1)
    with patch("prospectpilot.github_client.time.sleep"):
        result = client.get_organization("test")
    assert result["login"] == "recovered"
    assert transport.call_count == 2


def test_retry_exhaustion_on_persistent_500() -> None:
    """Verifies that persistent server errors exhaust retries and raise."""
    transport = MockTransport()
    for _ in range(4):
        transport.add_response(status_code=500, json_data={"message": "Server Error"})
    client = _make_client(transport, max_retries=2)
    with patch("prospectpilot.github_client.time.sleep"):
        with pytest.raises(GitHubAPIError, match="failed after 3 attempts"):
            client.get_organization("failing-org")
    assert transport.call_count == 3


def test_retry_on_timeout_then_success() -> None:
    """Verifies retry after httpx timeout, succeeding on second attempt."""
    transport = MockTransport()
    # First response raises timeout (simulated by no responses available)
    transport.add_response(
        json_data={"login": "recovered"},
        headers={"x-ratelimit-remaining": "50", "x-ratelimit-limit": "60", "x-ratelimit-reset": str(int(time.time()) + 3600)},
    )

    class TimeoutThenSuccessTransport(httpx.BaseTransport):
        def __init__(self, inner: MockTransport):
            self._inner = inner
            self._calls = 0
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            self._calls += 1
            if self._calls == 1:
                raise httpx.ReadTimeout("Simulated timeout")
            return self._inner.handle_request(request)
        @property
        def call_count(self) -> int:
            return self._calls

    timeout_transport = TimeoutThenSuccessTransport(transport)
    client = GitHubClient(token="test-token-fake", max_retries=2)
    client._client = httpx.Client(
        base_url=GitHubClient.BASE_URL,
        headers=client._client.headers,
        timeout=client._client.timeout,
        transport=timeout_transport,
    )
    with patch("prospectpilot.github_client.time.sleep"):
        result = client.get_organization("test")
    assert result["login"] == "recovered"
    assert timeout_transport.call_count == 2


# ------------------------------------------------------------------
# Tests: rate-limit handling
# ------------------------------------------------------------------

def test_rate_limit_headers_parsed() -> None:
    """Verifies that rate-limit headers are parsed and stored."""
    transport = MockTransport()
    reset_time = int(time.time()) + 3600
    transport.add_response(
        json_data={"login": "test"},
        headers={
            "x-ratelimit-remaining": "42",
            "x-ratelimit-limit": "60",
            "x-ratelimit-reset": str(reset_time),
        },
    )
    client = _make_client(transport)
    client.get_organization("test")
    rl = client.last_rate_limit
    assert rl is not None
    assert rl.remaining == 42
    assert rl.limit == 60


def test_rate_limit_exhausted_raises_when_wait_too_long() -> None:
    """Verifies RateLimitExhausted is raised when reset is too far in the future."""
    transport = MockTransport()
    far_future = int(time.time()) + 7200  # 2 hours from now
    transport.add_response(
        status_code=403,
        json_data={"message": "API rate limit exceeded"},
        headers={
            "x-ratelimit-remaining": "0",
            "x-ratelimit-limit": "60",
            "x-ratelimit-reset": str(far_future),
        },
    )
    client = _make_client(transport, max_rate_limit_wait=30.0)
    with pytest.raises(RateLimitExhausted, match="rate limit exhausted"):
        client.search_repositories("test")


def test_429_with_short_retry_after() -> None:
    """Verifies that 429 with short Retry-After waits and retries."""
    transport = MockTransport()
    transport.add_response(
        status_code=429,
        json_data={"message": "rate limited"},
        headers={"retry-after": "1"},
    )
    transport.add_response(
        json_data={"login": "recovered"},
        headers={"x-ratelimit-remaining": "50", "x-ratelimit-limit": "60", "x-ratelimit-reset": str(int(time.time()) + 3600)},
    )
    client = _make_client(transport, max_retries=1)
    with patch("prospectpilot.github_client.time.sleep"):
        result = client.get_organization("test")
    assert result["login"] == "recovered"
    assert transport.call_count == 2


# ------------------------------------------------------------------
# Tests: RateLimitInfo
# ------------------------------------------------------------------

def test_rate_limit_info_from_headers() -> None:
    headers = httpx.Headers({
        "x-ratelimit-remaining": "10",
        "x-ratelimit-limit": "60",
        "x-ratelimit-reset": str(int(time.time()) + 600),
    })
    info = RateLimitInfo.from_headers(headers)
    assert info is not None
    assert info.remaining == 10
    assert info.limit == 60
    assert not info.is_exhausted


def test_rate_limit_info_exhausted() -> None:
    headers = httpx.Headers({
        "x-ratelimit-remaining": "2",
        "x-ratelimit-limit": "60",
        "x-ratelimit-reset": str(int(time.time()) + 600),
    })
    info = RateLimitInfo.from_headers(headers)
    assert info is not None
    assert info.is_exhausted  # remaining <= 3


def test_rate_limit_info_from_missing_headers() -> None:
    headers = httpx.Headers({})
    info = RateLimitInfo.from_headers(headers)
    assert info is None


def test_secondary_rate_limit_handled() -> None:
    """Verifies that 403 secondary rate limit with Retry-After header is waited and retried."""
    transport = MockTransport()
    transport.add_response(
        status_code=403,
        json_data={"message": "You have exceeded a secondary rate limit."},
        headers={"retry-after": "1"},
    )
    transport.add_response(
        json_data={"login": "recovered-secondary"},
        headers={"x-ratelimit-remaining": "50", "x-ratelimit-limit": "60", "x-ratelimit-reset": str(int(time.time()) + 3600)},
    )
    client = _make_client(transport, max_retries=1)
    with patch("prospectpilot.github_client.time.sleep"):
        result = client.get_organization("test")
    assert result["login"] == "recovered-secondary"
    assert transport.call_count == 2


def test_search_repositories_pagination() -> None:
    """Verifies that search_repositories forwards the page parameter."""
    transport = MockTransport()
    transport.add_response(
        json_data={"total_count": 50, "items": [{"full_name": "org/repo-page-2"}]},
        headers={"x-ratelimit-remaining": "50", "x-ratelimit-limit": "60", "x-ratelimit-reset": str(int(time.time()) + 3600)},
    )
    client = _make_client(transport)
    result = client.search_repositories("topic:cli", page=2, per_page=15)
    assert result["items"][0]["full_name"] == "org/repo-page-2"
    # Verify request query params
    req = transport.requests[0]
    assert "page=2" in str(req.url)
    assert "per_page=15" in str(req.url)


# ------------------------------------------------------------------
# Tests: token security
# ------------------------------------------------------------------

def test_token_not_logged(caplog) -> None:
    """Verifies that the actual token value is never present in log output."""
    import logging
    secret_token = "ghp_SuperSecretToken123456789"
    with caplog.at_level(logging.DEBUG, logger="prospectpilot.github"):
        _client = GitHubClient(token=secret_token)
    assert secret_token not in caplog.text
    # The log should mention "token" but not the actual value
    _client.close()
