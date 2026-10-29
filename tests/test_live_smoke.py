"""Opt-in live smoke tests for real API integration.

These tests are NEVER required for the offline test suite.
Run with: pytest -m live
Skip with: pytest -m "not live"  (default behavior)

Prerequisites:
- GitHub API access (works unauthenticated at 60 req/hr, or authenticated with GITHUB_TOKEN at 5,000 req/hr)
- LLM provider configured (any one of: OPENAI_API_KEY, GROQ_API_KEY, etc.) for full pipeline
"""

from __future__ import annotations

import os

import pytest

# All tests in this file require the 'live' marker
pytestmark = pytest.mark.live


def _has_llm_key() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENAI_BASE_URL")
    )


class TestGitHubLiveSmoke:
    """Live smoke tests for the GitHub REST API client against public endpoints."""

    def test_search_repositories_live(self) -> None:
        from prospectpilot.github_client import GitHubClient

        client = GitHubClient()
        try:
            result = client.search_repositories("topic:developer-tools language:go stars:>500", per_page=3)
            assert "total_count" in result
            assert result["total_count"] > 0
            assert len(result["items"]) >= 1
            item = result["items"][0]
            assert "full_name" in item
            assert "stargazers_count" in item
            assert "owner" in item
        finally:
            client.close()

    def test_get_organization_live(self) -> None:
        from prospectpilot.github_client import GitHubClient

        client = GitHubClient()
        try:
            result = client.get_organization("hashicorp")
            assert result["login"] == "hashicorp"
            assert "public_repos" in result
            assert result["public_repos"] > 0
        finally:
            client.close()

    def test_get_repository_live(self) -> None:
        from prospectpilot.github_client import GitHubClient

        client = GitHubClient()
        try:
            result = client.get_repository("hashicorp", "terraform")
            assert result["full_name"] == "hashicorp/terraform"
            assert result["stargazers_count"] > 10000
        finally:
            client.close()

    def test_get_releases_live(self) -> None:
        from prospectpilot.github_client import GitHubClient

        client = GitHubClient()
        try:
            result = client.get_releases("hashicorp", "terraform", per_page=2)
            assert isinstance(result, list)
            if result:
                assert "tag_name" in result[0]
        finally:
            client.close()

    def test_github_tools_integration_live(self) -> None:
        """Full integration: register tools, execute, parse responses against live GitHub API."""
        from prospectpilot.github_client import GitHubClient
        from prospectpilot.tools.base import ToolRegistry
        from prospectpilot.tools.github_tools import register_github_tools

        client = GitHubClient()
        registry = ToolRegistry()
        register_github_tools(registry, client)

        try:
            # Search
            result = registry.execute(
                "search_github_repos",
                {"query": "topic:cli language:rust stars:>1000", "per_page": 2},
            )
            assert result.success is True
            assert result.data["returned_count"] >= 1

            # Get org
            first_owner = result.data["repos"][0]["owner_login"]
            org_result = registry.execute("get_org_profile", {"org": first_owner})
            # May be Organization or User; either returns or 404s cleanly
            assert org_result.execution_time_ms > 0
        finally:
            client.close()


@pytest.mark.skipif(
    not _has_llm_key(),
    reason="LLM API key required (OPENAI_API_KEY, GROQ_API_KEY, etc.) for full agent pipeline smoke test",
)
class TestFullPipelineLiveSmoke:
    """End-to-end live test: LLM + GitHub API -> scored evaluations."""

    def test_full_live_pipeline(self) -> None:
        from prospectpilot.agent.controller import AgentController
        from prospectpilot.agent.llm import OpenAICompatibleClient
        from prospectpilot.github_client import GitHubClient
        from prospectpilot.models import QualificationCriteria
        from prospectpilot.tools.github_tools import register_github_tools

        llm = OpenAICompatibleClient()
        github_client = GitHubClient()

        controller = AgentController(
            llm_client=llm,
            max_iterations=6,
            max_tool_calls=12,
            max_time_seconds=120.0,
            github_tools_registered=True,
        )
        register_github_tools(controller.registry, github_client)

        try:
            result = controller.run(
                goal="Find developer CLI tools written in Go or Rust with active releases",
                criteria=QualificationCriteria(target_focus="Developer CLI Tools"),
            )

            assert result.status in ("SUCCESS", "PARTIAL_BUDGET_EXHAUSTED")
            assert result.total_tool_calls >= 1
            assert result.is_synthetic_demo_run is False
            if result.evaluations:
                for ev in result.evaluations:
                    assert ev.is_synthetic_demo_data is False
                    assert ev.company_id
                    assert ev.score_breakdown.total_score >= 0
                    assert len(ev.evidence) >= 1
        finally:
            github_client.close()
