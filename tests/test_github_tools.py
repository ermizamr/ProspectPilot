"""Tests for GitHub tools: schema validation, response parsing, deduplication, empty results.

All tests use a mock GitHubClient — zero network access required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from prospectpilot.github_client import GitHubAPIError, GitHubClient
from prospectpilot.tools.base import ToolRegistry
from prospectpilot.tools.github_tools import register_github_tools


# ------------------------------------------------------------------
# Mock GitHubClient
# ------------------------------------------------------------------

class MockGitHubClient:
    """Deterministic mock of GitHubClient for offline tool testing."""

    def __init__(self) -> None:
        self._search_results: Optional[Dict[str, Any]] = None
        self._repo_data: Optional[Dict[str, Any]] = None
        self._org_data: Optional[Dict[str, Any]] = None
        self._releases_data: Optional[List[Dict[str, Any]]] = None
        self._should_raise: Optional[Exception] = None
        self.call_log: List[str] = []

    def set_search_results(self, data: Dict[str, Any]) -> None:
        self._search_results = data

    def set_repo_data(self, data: Dict[str, Any]) -> None:
        self._repo_data = data

    def set_org_data(self, data: Dict[str, Any]) -> None:
        self._org_data = data

    def set_releases_data(self, data: List[Dict[str, Any]]) -> None:
        self._releases_data = data

    def set_error(self, exc: Exception) -> None:
        self._should_raise = exc

    def search_repositories(self, **kwargs: Any) -> Dict[str, Any]:
        self.call_log.append(f"search_repositories({kwargs})")
        if self._should_raise:
            raise self._should_raise
        return self._search_results or {"total_count": 0, "items": []}

    def get_repository(self, **kwargs: Any) -> Dict[str, Any]:
        self.call_log.append(f"get_repository({kwargs})")
        if self._should_raise:
            raise self._should_raise
        return self._repo_data or {}

    def get_organization(self, **kwargs: Any) -> Dict[str, Any]:
        self.call_log.append(f"get_organization({kwargs})")
        if self._should_raise:
            raise self._should_raise
        return self._org_data or {}

    def get_releases(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.call_log.append(f"get_releases({kwargs})")
        if self._should_raise:
            raise self._should_raise
        return self._releases_data or []


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SAMPLE_REPO_API_RESPONSE = {
    "full_name": "hashicorp/terraform",
    "name": "terraform",
    "owner": {"login": "hashicorp", "type": "Organization"},
    "description": "Terraform enables you to safely and predictably create, change, and improve infrastructure.",
    "html_url": "https://github.com/hashicorp/terraform",
    "stargazers_count": 42000,
    "forks_count": 9500,
    "language": "Go",
    "topics": ["infrastructure", "terraform", "devops", "cloud"],
    "license": {"spdx_id": "BUSL-1.1"},
    "updated_at": "2026-10-25T10:00:00Z",
    "created_at": "2014-04-10T00:00:00Z",
    "open_issues_count": 1800,
    "archived": False,
    "fork": False,
}

SAMPLE_ORG_API_RESPONSE = {
    "login": "hashicorp",
    "name": "HashiCorp",
    "description": "Consistent workflows to provision, secure, connect, and run any infrastructure.",
    "html_url": "https://github.com/hashicorp",
    "blog": "https://www.hashicorp.com",
    "company": "HashiCorp",
    "location": "San Francisco, CA",
    "public_repos": 450,
    "created_at": "2013-01-01T00:00:00Z",
}

SAMPLE_SEARCH_RESPONSE = {
    "total_count": 1,
    "items": [SAMPLE_REPO_API_RESPONSE],
}


@pytest.fixture
def mock_github() -> MockGitHubClient:
    client = MockGitHubClient()
    client.set_search_results(SAMPLE_SEARCH_RESPONSE)
    client.set_repo_data(SAMPLE_REPO_API_RESPONSE)
    client.set_org_data(SAMPLE_ORG_API_RESPONSE)
    client.set_releases_data([
        {"tag_name": "v1.10.0", "name": "v1.10.0", "published_at": "2026-10-20T00:00:00Z", "html_url": "https://github.com/hashicorp/terraform/releases/tag/v1.10.0", "prerelease": False},
    ])
    return client


@pytest.fixture
def github_registry(mock_github: MockGitHubClient) -> ToolRegistry:
    registry = ToolRegistry()
    register_github_tools(registry, mock_github)  # type: ignore[arg-type]
    return registry


# ------------------------------------------------------------------
# Tests: tool registration and schema generation
# ------------------------------------------------------------------

def test_github_tools_registered(github_registry: ToolRegistry) -> None:
    """All four GitHub tools should be registered."""
    schemas = github_registry.get_openai_tools()
    tool_names = {s["function"]["name"] for s in schemas}
    assert tool_names == {"search_github_repos", "get_repo_details", "get_org_profile", "get_recent_releases"}


def test_github_tool_schemas_valid(github_registry: ToolRegistry) -> None:
    """All GitHub tool schemas should have proper OpenAI structure."""
    for schema in github_registry.get_openai_tools():
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


# ------------------------------------------------------------------
# Tests: search_github_repos
# ------------------------------------------------------------------

def test_search_github_repos_success(github_registry: ToolRegistry) -> None:
    result = github_registry.execute("search_github_repos", {"query": "topic:devops"})
    assert result.success is True
    assert result.data["total_count"] == 1
    assert len(result.data["repos"]) == 1
    repo = result.data["repos"][0]
    assert repo["full_name"] == "hashicorp/terraform"
    assert repo["owner_login"] == "hashicorp"
    assert repo["owner_type"] == "Organization"
    assert repo["stars"] == 42000
    assert "retrieved_at" in repo


def test_search_github_repos_empty(github_registry: ToolRegistry, mock_github: MockGitHubClient) -> None:
    mock_github.set_search_results({"total_count": 0, "items": []})
    result = github_registry.execute("search_github_repos", {"query": "nonexistent-topic-xyz"})
    assert result.success is True
    assert result.data["total_count"] == 0
    assert result.data["repos"] == []


def test_search_github_repos_missing_query(github_registry: ToolRegistry) -> None:
    """Missing required 'query' should fail with Pydantic validation."""
    result = github_registry.execute("search_github_repos", {})
    assert result.success is False
    assert "validation failed" in result.error.lower()


# ------------------------------------------------------------------
# Tests: get_repo_details
# ------------------------------------------------------------------

def test_get_repo_details_success(github_registry: ToolRegistry) -> None:
    result = github_registry.execute("get_repo_details", {"owner": "hashicorp", "repo": "terraform"})
    assert result.success is True
    assert result.data["full_name"] == "hashicorp/terraform"
    assert result.data["language"] == "Go"
    assert "devops" in result.data["topics"]


def test_get_repo_details_api_error(github_registry: ToolRegistry, mock_github: MockGitHubClient) -> None:
    mock_github.set_error(GitHubAPIError("Not found", status_code=404))
    result = github_registry.execute("get_repo_details", {"owner": "nonexistent", "repo": "nope"})
    assert result.success is False
    assert "Not found" in result.error


def test_get_repo_details_missing_args(github_registry: ToolRegistry) -> None:
    result = github_registry.execute("get_repo_details", {"owner": "hashicorp"})
    assert result.success is False
    assert "validation failed" in result.error.lower()


# ------------------------------------------------------------------
# Tests: get_org_profile
# ------------------------------------------------------------------

def test_get_org_profile_success(github_registry: ToolRegistry) -> None:
    result = github_registry.execute("get_org_profile", {"org": "hashicorp"})
    assert result.success is True
    assert result.data["login"] == "hashicorp"
    assert result.data["name"] == "HashiCorp"
    assert result.data["blog"] == "https://www.hashicorp.com"
    assert "retrieved_at" in result.data


def test_get_org_profile_api_error(github_registry: ToolRegistry, mock_github: MockGitHubClient) -> None:
    mock_github.set_error(GitHubAPIError("Not found", status_code=404))
    result = github_registry.execute("get_org_profile", {"org": "nonexistent"})
    assert result.success is False


# ------------------------------------------------------------------
# Tests: get_recent_releases
# ------------------------------------------------------------------

def test_get_recent_releases_success(github_registry: ToolRegistry) -> None:
    result = github_registry.execute("get_recent_releases", {"owner": "hashicorp", "repo": "terraform"})
    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["tag_name"] == "v1.10.0"


def test_get_recent_releases_empty(github_registry: ToolRegistry, mock_github: MockGitHubClient) -> None:
    mock_github.set_releases_data([])
    result = github_registry.execute("get_recent_releases", {"owner": "test", "repo": "empty"})
    assert result.success is True
    assert result.data == []


# ------------------------------------------------------------------
# Tests: unregistered tool
# ------------------------------------------------------------------

def test_unregistered_tool_in_github_registry(github_registry: ToolRegistry) -> None:
    result = github_registry.execute("hack_the_planet", {})
    assert result.success is False
    assert "not registered" in result.error


def test_search_github_repos_with_page(github_registry: ToolRegistry) -> None:
    result = github_registry.execute("search_github_repos", {"query": "topic:devops", "page": 2, "per_page": 5})
    assert result.success is True
    assert result.data["page"] == 2


def test_get_org_profile_url_cleaning(github_registry: ToolRegistry, mock_github: MockGitHubClient) -> None:
    # Bare domain without scheme
    mock_github.set_org_data({
        "login": "bare-domain-org",
        "name": "Bare Domain",
        "blog": "example.com",
        "html_url": "https://github.com/bare-domain-org",
    })
    result = github_registry.execute("get_org_profile", {"org": "bare-domain-org"})
    assert result.success is True
    assert result.data["blog"] == "https://example.com"
