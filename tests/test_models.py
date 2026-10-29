"""Tests for domain schemas and model validation."""

import pytest
from prospectpilot.models import (
    AgentRunResult,
    CompanyEvaluation,
    EvidenceItem,
    GitHubOrgSummary,
    GitHubProspect,
    GitHubRepoSummary,
    QualificationCriteria,
    QualificationStatus,
    ScoreBreakdown,
)


def test_qualification_criteria_defaults() -> None:
    """Verifies default values for qualification criteria."""
    criteria = QualificationCriteria()
    assert criteria.target_focus == "Developer Tools"
    assert criteria.require_developer_audience is True
    assert criteria.min_team_size == 5
    assert criteria.max_team_size == 150
    assert criteria.require_public_signals is False


def test_score_breakdown_validation() -> None:
    """Verifies ScoreBreakdown fields and total score computation."""
    breakdown = ScoreBreakdown(
        developer_alignment_score=35,
        scale_and_stage_score=25,
        engagement_signals_score=20,
        total_score=80,
        rules_applied=["+25 Dev relevance", "+10 traction"],
    )
    assert breakdown.total_score == 80
    assert len(breakdown.rules_applied) == 2


def test_github_repo_summary_parsing() -> None:
    """Verifies that GitHubRepoSummary parses repo attributes properly."""
    repo = GitHubRepoSummary(
        full_name="hashicorp/terraform",
        owner_login="hashicorp",
        owner_type="Organization",
        name="terraform",
        html_url="https://github.com/hashicorp/terraform",
        stars=42000,
        forks=9500,
        language="Go",
        topics=["infrastructure", "cloud"],
        retrieved_at="2026-10-28T00:00:00Z",
    )
    assert repo.full_name == "hashicorp/terraform"
    assert repo.stars == 42000
    assert "cloud" in repo.topics
    assert repo.is_fork is False


def test_github_org_summary_parsing() -> None:
    """Verifies that GitHubOrgSummary parses org metadata properly."""
    org = GitHubOrgSummary(
        login="hashicorp",
        name="HashiCorp",
        html_url="https://github.com/hashicorp",
        blog="https://hashicorp.com",
        company="HashiCorp",
        public_repos=450,
        retrieved_at="2026-10-28T00:00:00Z",
    )
    assert org.login == "hashicorp"
    assert org.name == "HashiCorp"
    assert org.public_repos == 450


def test_agent_run_result_serialization() -> None:
    """Verifies that AgentRunResult serializes and deserializes cleanly."""
    result = AgentRunResult(
        goal="Find devops tools",
        criteria=QualificationCriteria(),
        status="SUCCESS",
        evaluations=[
            CompanyEvaluation(
                company_id="hashicorp",
                company_name="HashiCorp",
                domain="https://hashicorp.com",
                qualification_status=QualificationStatus.QUALIFIED,
                score_breakdown=ScoreBreakdown(
                    developer_alignment_score=40,
                    scale_and_stage_score=30,
                    engagement_signals_score=25,
                    total_score=95,
                ),
                evidence=[
                    EvidenceItem(
                        field_name="stars",
                        observed_value=42000,
                        relevance="High traction",
                        source_url="https://github.com/hashicorp/terraform",
                    )
                ],
                missing_information=[],
                explanation="Top-tier devops tool provider.",
                recommended_action="Outreach Recommended",
                is_synthetic_demo_data=False,
            )
        ],
        summary="Found 1 prospect.",
        total_iterations=2,
        total_tool_calls=2,
        total_tokens=150,
        duration_seconds=1.25,
        is_synthetic_demo_run=False,
    )

    json_str = result.model_dump_json(indent=2)
    assert '"status": "SUCCESS"' in json_str
    assert '"is_synthetic_demo_run": false' in json_str

    restored = AgentRunResult.model_validate_json(json_str)
    assert restored.evaluations[0].company_id == "hashicorp"
    assert restored.is_synthetic_demo_run is False
