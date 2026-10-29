"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pytest

from prospectpilot.models import QualificationCriteria
from prospectpilot.tools.base import ToolRegistry
from prospectpilot.tools.github_tools import register_github_tools


@pytest.fixture
def default_criteria() -> QualificationCriteria:
    return QualificationCriteria(
        target_focus="Developer Tools",
        require_developer_audience=True,
        min_team_size=10,
        max_team_size=100,
    )


# ==============================================================================
# GitHub Mock Client and Sample Data
# ==============================================================================

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


class MockGitHubClient:
    """Deterministic mock of GitHubClient for offline tool and controller testing."""

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
        q = str(kwargs.get("query") or kwargs.get("q") or "")
        if "rare_nonexistent_keyword" in q:
            return {"total_count": 0, "items": []}
        if "database" in q:
            return {
                "total_count": 1,
                "items": [{
                    "full_name": "schemacraft/schemacraft",
                    "name": "schemacraft",
                    "owner": {"login": "schemacraft", "type": "Organization"},
                    "description": "Database schema migration safety platform",
                    "html_url": "https://github.com/schemacraft/schemacraft",
                    "stargazers_count": 3500,
                    "forks_count": 400,
                    "language": "TypeScript",
                    "topics": ["database", "postgresql", "migration"],
                    "license": {"spdx_id": "Apache-2.0"},
                    "updated_at": "2026-10-25T10:00:00Z",
                    "created_at": "2021-01-01T00:00:00Z",
                    "open_issues_count": 25,
                    "archived": False,
                    "fork": False,
                }],
            }
        if "security" in q:
            return {
                "total_count": 1,
                "items": [{
                    "full_name": "devsecgate/scanner",
                    "name": "scanner",
                    "owner": {"login": "devsecgate", "type": "Organization"},
                    "description": "DevSecOps vulnerability scanner",
                    "html_url": "https://github.com/devsecgate/scanner",
                    "stargazers_count": 1200,
                    "forks_count": 150,
                    "language": "Go",
                    "topics": ["security", "scanner", "devops"],
                    "license": {"spdx_id": "MIT"},
                    "updated_at": "2026-10-25T10:00:00Z",
                    "created_at": "2020-01-01T00:00:00Z",
                    "open_issues_count": 10,
                    "archived": False,
                    "fork": False,
                }],
            }
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
        org = kwargs.get("org", "")
        if org == "devsecgate":
            return {
                "login": "devsecgate",
                "name": "DevSecGate",
                "description": "Application security and secret scanning",
                "html_url": "https://github.com/devsecgate",
                "blog": "https://devsecgate.io",
                "company": "DevSecGate Inc",
                "location": "Boston, MA",
                "public_repos": 15,
                "created_at": "2020-01-01T00:00:00Z",
            }
        if org == "schemacraft":
            return {
                "login": "schemacraft",
                "name": "SchemaCraft",
                "description": "Database schema migration safety platform",
                "html_url": "https://github.com/schemacraft",
                "blog": "https://schemacraft.io",
                "company": "SchemaCraft Inc",
                "location": "San Francisco, CA",
                "public_repos": 20,
                "created_at": "2021-01-01T00:00:00Z",
            }
        return self._org_data or {}

    def get_releases(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.call_log.append(f"get_releases({kwargs})")
        if self._should_raise:
            raise self._should_raise
        return self._releases_data or []


@pytest.fixture
def mock_github() -> MockGitHubClient:
    client = MockGitHubClient()
    client.set_search_results(SAMPLE_SEARCH_RESPONSE)
    client.set_repo_data(SAMPLE_REPO_API_RESPONSE)
    client.set_org_data(SAMPLE_ORG_API_RESPONSE)
    client.set_releases_data([
        {
            "tag_name": "v1.10.0",
            "name": "v1.10.0",
            "published_at": "2026-10-20T00:00:00Z",
            "html_url": "https://github.com/hashicorp/terraform/releases/tag/v1.10.0",
            "prerelease": False,
        },
    ])
    return client


@pytest.fixture
def github_registry(mock_github: MockGitHubClient) -> ToolRegistry:
    registry = ToolRegistry()
    register_github_tools(registry, mock_github)  # type: ignore[arg-type]
    return registry
