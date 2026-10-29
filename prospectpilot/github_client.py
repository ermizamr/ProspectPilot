"""GitHub REST API client with rate limiting, retries, and timeout handling.

Provides a thin, typed wrapper around the GitHub REST API with:
- Configurable HTTP timeouts (connect + read)
- Bounded retries with exponential backoff for transient failures (429, 5xx, timeouts)
- Rate-limit header parsing and proactive waiting
- Structured error types for callers to handle explicitly

Never logs API tokens or other secrets.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("prospectpilot.github")


class GitHubAPIError(Exception):
    """Raised when the GitHub API returns an unrecoverable error."""

    def __init__(
        self, message: str, status_code: int = 0, response_body: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitExhausted(GitHubAPIError):
    """Raised when GitHub rate limit is exhausted and cannot be waited out within budget."""

    pass


class RateLimitInfo:
    """Parsed GitHub API rate-limit state from response headers."""

    __slots__ = ("remaining", "limit", "reset_at")

    def __init__(self, remaining: int, limit: int, reset_at: float) -> None:
        self.remaining = remaining
        self.limit = limit
        self.reset_at = reset_at

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 3

    @property
    def wait_seconds(self) -> float:
        return max(0.0, self.reset_at - time.time() + 1.0)

    @classmethod
    def from_headers(cls, headers: httpx.Headers) -> Optional["RateLimitInfo"]:
        try:
            remaining = int(headers.get("x-ratelimit-remaining", "-1"))
            limit = int(headers.get("x-ratelimit-limit", "-1"))
            reset_at = float(headers.get("x-ratelimit-reset", "0"))
            if remaining < 0 or limit < 0:
                return None
            return cls(remaining=remaining, limit=limit, reset_at=reset_at)
        except (ValueError, TypeError):
            return None


class GitHubClient:
    """Synchronous GitHub REST API client with retry, backoff, and rate-limit awareness.

    Usage::

        client = GitHubClient()  # reads GITHUB_TOKEN from env
        results = client.search_repositories("topic:developer-tools language:python")
        client.close()
    """

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        max_retries: int = 3,
        max_rate_limit_wait: float = 60.0,
    ) -> None:
        resolved_token = token or os.getenv("GITHUB_TOKEN")

        headers: Dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ProspectPilot/0.2.0",
        }
        if resolved_token:
            headers["Authorization"] = f"Bearer {resolved_token}"
            logger.info("GitHub client initialized with authentication token")
        else:
            logger.warning(
                "GitHub client initialized WITHOUT token — rate limit is 60 req/hr. "
                "Set GITHUB_TOKEN for 5,000 req/hr."
            )

        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=10.0,
                pool=10.0,
            ),
        )
        self._max_retries = max_retries
        self._max_rate_limit_wait = max_rate_limit_wait
        self._last_rate_limit: Optional[RateLimitInfo] = None

    @property
    def last_rate_limit(self) -> Optional[RateLimitInfo]:
        """Most recently observed rate-limit state from GitHub response headers."""
        return self._last_rate_limit

    def _request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute an HTTP request with retries and rate-limit handling.

        Retries on: timeouts, 429, 5xx. Raises on: 4xx (except 429), exhausted retries.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            # Pre-check cached rate limit
            if self._last_rate_limit and self._last_rate_limit.is_exhausted:
                wait = self._last_rate_limit.wait_seconds
                if wait > self._max_rate_limit_wait:
                    raise RateLimitExhausted(
                        f"GitHub rate limit exhausted. Reset in {wait:.0f}s "
                        f"(exceeds max wait of {self._max_rate_limit_wait:.0f}s). "
                        f"Remaining: {self._last_rate_limit.remaining}/{self._last_rate_limit.limit}",
                        status_code=429,
                    )
                logger.info(
                    "Rate limit low (%d remaining), waiting %.1fs",
                    self._last_rate_limit.remaining,
                    wait,
                )
                time.sleep(wait)

            try:
                response = self._client.request(method, path, params=params)
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self._max_retries:
                    backoff = 2**attempt
                    logger.warning(
                        "Request timeout (attempt %d/%d): %s. Retrying in %ds",
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                continue
            except httpx.HTTPError as exc:
                last_error = exc
                logger.error("HTTP transport error: %s", exc)
                if attempt < self._max_retries:
                    time.sleep(2**attempt)
                continue

            # Parse rate-limit from every response
            rl = RateLimitInfo.from_headers(response.headers)
            if rl is not None:
                self._last_rate_limit = rl
                logger.debug(
                    "Rate limit: %d/%d remaining, resets at %s",
                    rl.remaining,
                    rl.limit,
                    time.ctime(rl.reset_at),
                )

            # Success
            if response.status_code == 200:
                return response.json()

            # Rate limit via 403 (primary or secondary/abuse)
            if response.status_code == 403:
                is_secondary = "secondary rate limit" in response.text.lower() or "abuse" in response.text.lower()
                retry_after_hdr = response.headers.get("retry-after")
                if is_secondary or (rl and rl.is_exhausted) or retry_after_hdr:
                    wait = float(retry_after_hdr) if retry_after_hdr else (rl.wait_seconds if rl else float(2**attempt * 5))
                    if wait > self._max_rate_limit_wait:
                        raise RateLimitExhausted(
                            f"GitHub rate limit exhausted (403 {'secondary' if is_secondary else 'primary'}). "
                            f"Reset in {wait:.0f}s (exceeds max wait of {self._max_rate_limit_wait:.0f}s). "
                            f"Set GITHUB_TOKEN for higher limits.",
                            status_code=403,
                            response_body=response.text[:500],
                        )
                    logger.info("Rate limited (403), waiting %.1fs for reset", wait)
                    time.sleep(wait)
                    continue

            # Explicit 429
            if response.status_code == 429:
                retry_after = float(
                    response.headers.get("retry-after", str(2**attempt))
                )
                if retry_after > self._max_rate_limit_wait:
                    raise RateLimitExhausted(
                        f"GitHub 429 rate limit. Retry-After: {retry_after}s",
                        status_code=429,
                        response_body=response.text[:500],
                    )
                logger.info("Rate limited (429), waiting %.1fs", retry_after)
                time.sleep(retry_after)
                continue

            # 404 — resource not found, don't retry
            if response.status_code == 404:
                raise GitHubAPIError(
                    f"GitHub resource not found: {path}",
                    status_code=404,
                    response_body=response.text[:500],
                )

            # Server errors — retry
            if response.status_code >= 500:
                last_error = GitHubAPIError(
                    f"GitHub server error {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                )
                if attempt < self._max_retries:
                    backoff = 2**attempt
                    logger.warning(
                        "Server error %d (attempt %d/%d), retrying in %ds",
                        response.status_code,
                        attempt + 1,
                        self._max_retries + 1,
                        backoff,
                    )
                    time.sleep(backoff)
                continue

            # Other client errors (401, 422, etc.) — don't retry
            raise GitHubAPIError(
                f"GitHub API error {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
                response_body=response.text[:500],
            )

        # All retries exhausted
        raise GitHubAPIError(
            f"GitHub request failed after {self._max_retries + 1} attempts: {last_error}",
            status_code=getattr(last_error, "status_code", 0),
        )

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def search_repositories(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """Search GitHub repositories with pagination. Returns the full search API response."""
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "page": max(1, page),
            "per_page": min(max(1, per_page), 30),
        }
        return self._request("GET", "/search/repositories", params=params)

    def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get detailed metadata for a single repository."""
        return self._request("GET", f"/repos/{owner}/{repo}")

    def get_organization(self, org: str) -> Dict[str, Any]:
        """Get organization profile metadata."""
        return self._request("GET", f"/orgs/{org}")

    def get_releases(
        self, owner: str, repo: str, per_page: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent releases for a repository."""
        return self._request(
            "GET", f"/repos/{owner}/{repo}/releases", params={"per_page": per_page}
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
