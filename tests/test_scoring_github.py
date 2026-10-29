"""Tests for GitHub prospect scoring, evidence validation, and deduplication.

Verifies that the deterministic scoring engine correctly handles GitHub-sourced
evidence with proper epistemic disclaimers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from prospectpilot.models import (
    GitHubOrgSummary,
    GitHubProspect,
    GitHubRepoSummary,
    QualificationCriteria,
    QualificationStatus,
    evaluate_github_prospect_deterministic,
    _is_recently_updated,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_repo(
    full_name: str = "org/repo",
    owner_login: str = "org",
    owner_type: str = "Organization",
    stars: int = 1000,
    language: str = "Go",
    topics: list[str] | None = None,
    description: str = "A developer CLI tool",
    updated_at: str | None = None,
) -> GitHubRepoSummary:
    return GitHubRepoSummary(
        full_name=full_name,
        owner_login=owner_login,
        owner_type=owner_type,
        name=full_name.split("/")[-1],
        description=description,
        html_url=f"https://github.com/{full_name}",
        stars=stars,
        forks=100,
        language=language,
        topics=topics or ["cli", "developer-tools"],
        license_name="MIT",
        updated_at=updated_at or _now_iso(),
        created_at="2020-01-01T00:00:00Z",
        open_issues_count=50,
        archived=False,
        is_fork=False,
        retrieved_at=_now_iso(),
    )


def _make_prospect(
    org_login: str = "hashicorp",
    org_name: str = "HashiCorp",
    org_blog: str | None = "https://www.hashicorp.com",
    org_description: str | None = "Infrastructure automation",
    org_company_field: str | None = "HashiCorp",
    repos: list[GitHubRepoSummary] | None = None,
    total_stars: int | None = None,
    has_recent_releases: bool = True,
) -> GitHubProspect:
    if repos is None:
        repos = [
            _make_repo("hashicorp/terraform", "hashicorp", stars=42000, language="Go",
                       topics=["infrastructure", "terraform", "devops"]),
            _make_repo("hashicorp/vault", "hashicorp", stars=31000, language="Go",
                       topics=["security", "secrets", "devops"]),
            _make_repo("hashicorp/consul", "hashicorp", stars=28000, language="Go",
                       topics=["service-mesh", "microservices"]),
        ]
    ts = total_stars if total_stars is not None else sum(r.stars for r in repos)
    return GitHubProspect(
        org_login=org_login,
        org_name=org_name,
        org_url=f"https://github.com/{org_login}",
        org_blog=org_blog,
        org_description=org_description,
        org_public_repos=50,
        org_company_field=org_company_field,
        repos=repos,
        top_languages=sorted({r.language for r in repos if r.language}),
        total_stars=ts,
        has_recent_releases=has_recent_releases,
        retrieved_at=_now_iso(),
        source_urls=[f"https://github.com/{org_login}"],
        is_synthetic_demo_data=False,
    )


@pytest.fixture
def default_criteria() -> QualificationCriteria:
    return QualificationCriteria(target_focus="Developer Tools")


# ------------------------------------------------------------------
# Tests: scoring — qualified prospect
# ------------------------------------------------------------------

def test_qualified_github_prospect(default_criteria: QualificationCriteria) -> None:
    """A strong org like HashiCorp with devops topics, many stars, blog, and releases should qualify."""
    prospect = _make_prospect()
    eval_result = evaluate_github_prospect_deterministic(prospect, default_criteria)

    assert eval_result.qualification_status == QualificationStatus.QUALIFIED
    assert eval_result.score_breakdown.total_score >= 70
    assert eval_result.score_breakdown.developer_alignment_score > 0
    assert eval_result.score_breakdown.scale_and_stage_score > 0
    assert eval_result.score_breakdown.engagement_signals_score > 0
    assert eval_result.company_id == "hashicorp"
    assert eval_result.company_name == "HashiCorp"
    assert eval_result.is_synthetic_demo_data is False


def test_qualified_prospect_has_source_urls(default_criteria: QualificationCriteria) -> None:
    prospect = _make_prospect()
    eval_result = evaluate_github_prospect_deterministic(prospect, default_criteria)
    source_urls = [e.source_url for e in eval_result.evidence if e.source_url]
    assert len(source_urls) >= 1


# ------------------------------------------------------------------
# Tests: scoring — low-quality prospect
# ------------------------------------------------------------------

def test_low_relevance_prospect(default_criteria: QualificationCriteria) -> None:
    """A personal account with few stars and no dev-tool topics should score low."""
    repos = [
        _make_repo(
            "user/my-dotfiles", "user", owner_type="User", stars=5,
            language="Shell", topics=["dotfiles"], description="My personal dotfiles",
        ),
    ]
    prospect = _make_prospect(
        org_login="user", org_name=None, org_blog=None, org_description=None,
        org_company_field=None, repos=repos, has_recent_releases=False,
    )
    eval_result = evaluate_github_prospect_deterministic(prospect, default_criteria)

    assert eval_result.qualification_status in (
        QualificationStatus.DISQUALIFIED,
        QualificationStatus.INSUFFICIENT_DATA,
    )
    assert eval_result.score_breakdown.total_score < 50


# ------------------------------------------------------------------
# Tests: epistemic disclaimers
# ------------------------------------------------------------------

def test_missing_information_includes_headcount(default_criteria: QualificationCriteria) -> None:
    """GitHub-sourced evaluations must explicitly note unavailable data."""
    prospect = _make_prospect()
    eval_result = evaluate_github_prospect_deterministic(prospect, default_criteria)
    missing_lower = [m.lower() for m in eval_result.missing_information]
    assert any("headcount" in m for m in missing_lower)
    assert any("revenue" in m or "funding" in m for m in missing_lower)


def test_unverified_company_identity(default_criteria: QualificationCriteria) -> None:
    """When org company field is empty, it should be flagged as unverified."""
    prospect = _make_prospect(org_company_field=None)
    eval_result = evaluate_github_prospect_deterministic(prospect, default_criteria)
    missing_lower = [m.lower() for m in eval_result.missing_information]
    assert any("unverified" in m for m in missing_lower)


def test_verified_company_identity(default_criteria: QualificationCriteria) -> None:
    """When org company field is set, unverified warning should NOT appear."""
    prospect = _make_prospect(org_company_field="HashiCorp, Inc.")
    eval_result = evaluate_github_prospect_deterministic(prospect, default_criteria)
    missing_lower = [m.lower() for m in eval_result.missing_information]
    assert not any("unverified" in m for m in missing_lower)


# ------------------------------------------------------------------
# Tests: evidence groundedness
# ------------------------------------------------------------------

def test_evidence_items_have_fields(default_criteria: QualificationCriteria) -> None:
    """Every evidence item must have field_name, observed_value, and relevance."""
    prospect = _make_prospect()
    eval_result = evaluate_github_prospect_deterministic(prospect, default_criteria)
    for item in eval_result.evidence:
        assert item.field_name
        assert item.observed_value is not None
        assert item.relevance


# ------------------------------------------------------------------
# Tests: scoring categories
# ------------------------------------------------------------------

def test_star_scoring_tiers(default_criteria: QualificationCriteria) -> None:
    """Verify star-based scoring produces correct point tiers."""
    # High stars
    high = _make_prospect(total_stars=10000)
    eval_high = evaluate_github_prospect_deterministic(high, default_criteria)
    
    # Low stars
    repos_low = [_make_repo("org/small", "org", stars=10, topics=["cli", "developer-tools"])]
    low = _make_prospect(repos=repos_low, total_stars=10)
    eval_low = evaluate_github_prospect_deterministic(low, default_criteria)

    assert eval_high.score_breakdown.scale_and_stage_score > eval_low.score_breakdown.scale_and_stage_score


def test_org_vs_user_scoring(default_criteria: QualificationCriteria) -> None:
    """Organization accounts should score higher on signals than personal accounts."""
    org_repos = [_make_repo("org/tool", "org", owner_type="Organization", topics=["cli", "developer-tools"])]
    user_repos = [_make_repo("user/tool", "user", owner_type="User", topics=["cli", "developer-tools"])]

    org_prospect = _make_prospect(repos=org_repos, org_blog="https://org.com", org_company_field="Org Inc")
    user_prospect = _make_prospect(
        org_login="user", repos=user_repos, org_blog=None,
        org_company_field=None, org_description=None,
    )

    eval_org = evaluate_github_prospect_deterministic(org_prospect, default_criteria)
    eval_user = evaluate_github_prospect_deterministic(user_prospect, default_criteria)

    assert eval_org.score_breakdown.engagement_signals_score > eval_user.score_breakdown.engagement_signals_score


# ------------------------------------------------------------------
# Tests: helper functions
# ------------------------------------------------------------------

def test_is_recently_updated_true() -> None:
    recent = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    assert _is_recently_updated(recent) is True


def test_is_recently_updated_false() -> None:
    old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
    assert _is_recently_updated(old) is False


def test_is_recently_updated_none() -> None:
    assert _is_recently_updated(None) is False


def test_is_recently_updated_invalid_format() -> None:
    assert _is_recently_updated("not-a-date") is False


# ------------------------------------------------------------------
# Tests: deduplication in controller
# ------------------------------------------------------------------

def test_controller_deduplicates_orgs(mock_github) -> None:
    """Verifies that examining the same org twice doesn't produce duplicate evaluations."""
    from prospectpilot.agent.controller import AgentController
    from prospectpilot.agent.llm import MockLLMClient, ModelDecision
    from prospectpilot.tools.github_tools import register_github_tools

    # Mock that calls get_org_profile twice for the same org, then finishes
    mock = MockLLMClient(decisions=[
        ModelDecision(
            is_tool_call=True,
            tool_name="get_org_profile",
            tool_arguments={"org": "hashicorp"},
        ),
        ModelDecision(
            is_tool_call=True,
            tool_name="get_org_profile",
            tool_arguments={"org": "hashicorp"},  # duplicate
        ),
        ModelDecision(
            is_tool_call=False,
            content="Done examining hashicorp.",
        ),
    ])

    controller = AgentController(
        llm_client=mock,
        max_iterations=5,
        max_tool_calls=10,
        github_tools_registered=True,
    )
    register_github_tools(controller.registry, mock_github)  # type: ignore[arg-type]

    result = controller.run(goal="Find developer tools")
    # Should have exactly 1 evaluation despite 2 get_org_profile calls
    assert len(result.evaluations) == 1
    assert result.evaluations[0].company_id == "hashicorp"


def test_controller_max_time_budget() -> None:
    """Verifies that the controller enforces wall-clock time limits."""
    from prospectpilot.agent.controller import AgentController
    from prospectpilot.agent.llm import MockLLMClient, ModelDecision
    from prospectpilot.dataset import CompanyDataset

    # Slow mock that takes time
    class SlowMock:
        def __init__(self):
            self._call_count = 0

        def generate_decision(self, messages, tools):
            self._call_count += 1
            import time
            time.sleep(0.01)
            return ModelDecision(
                is_tool_call=True,
                tool_name="search_companies",
                tool_arguments={"query": "tools"},
            )

    dataset = CompanyDataset.load_from_file()
    controller = AgentController(
        llm_client=SlowMock(),
        dataset=dataset,
        max_iterations=100,
        max_tool_calls=100,
        max_time_seconds=0.05,  # Very short time budget
    )

    result = controller.run(goal="Find tools")
    assert result.status == "PARTIAL_BUDGET_EXHAUSTED"
