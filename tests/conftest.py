"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from prospectpilot.dataset import CompanyDataset
from prospectpilot.models import (
    CompanyProfile,
    CompanySignals,
    QualificationCriteria,
)


@pytest.fixture
def sample_dev_profile() -> CompanyProfile:
    return CompanyProfile(
        id="comp-dev-01",
        name="DevToolCo",
        domain="devtool.io",
        industry="Developer Tools",
        description="High performance distributed tracing for backend engineers.",
        is_synthetic_demo_data=True,
        product_category="Observability",
        target_audience="Software Engineers, SREs",
        employee_count=45,
        funding_stage="Series A",
        tech_stack=["Go", "Rust", "Kubernetes"],
        signals=CompanySignals(
            has_public_github=True,
            has_active_blog=True,
            blog_focus="Kernel eBPF internals",
            hiring_roles=["Senior Go Engineer", "DevRel Advocate"]
        ),
    )


@pytest.fixture
def sample_consumer_profile() -> CompanyProfile:
    return CompanyProfile(
        id="comp-consumer-01",
        name="ChoreBuddy",
        domain="chorebuddy.app",
        industry="Consumer Mobile Apps",
        description="Gamified household chore tracker for families.",
        is_synthetic_demo_data=True,
        product_category="B2C Mobile App",
        target_audience="Parents and families",
        employee_count=15,
        funding_stage="Seed",
        tech_stack=["Flutter", "Firebase"],
        signals=CompanySignals(
            has_public_github=False,
            has_active_blog=False,
            hiring_roles=["Marketing Specialist"]
        ),
    )


@pytest.fixture
def sample_sparse_profile() -> CompanyProfile:
    return CompanyProfile(
        id="comp-sparse-01",
        name="StealthDev",
        domain="stealthdev.internal",
        industry="Developer Tools",
        description="Next-gen compiler in stealth.",
        is_synthetic_demo_data=True,
        product_category="Compilers",
        target_audience=None,
        employee_count=None,
        funding_stage=None,
        tech_stack=["Rust"],
        signals=CompanySignals(
            has_public_github=True,
            has_active_blog=False,
            hiring_roles=[]
        ),
    )


@pytest.fixture
def default_criteria() -> QualificationCriteria:
    return QualificationCriteria(
        target_focus="Developer Tools",
        require_developer_audience=True,
        min_team_size=10,
        max_team_size=100,
    )


@pytest.fixture
def real_demo_dataset() -> CompanyDataset:
    return CompanyDataset.load_from_file()


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
