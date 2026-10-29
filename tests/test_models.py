"""Tests for domain schemas and deterministic qualification scoring."""

import pytest
from prospectpilot.models import (
    CompanyProfile,
    CompanySignals,
    QualificationCriteria,
    QualificationStatus,
    evaluate_company_deterministic,
)


def test_company_profile_validation(sample_dev_profile: CompanyProfile) -> None:
    """Verifies that CompanyProfile parses properly and generates valid summaries."""
    assert sample_dev_profile.id == "comp-dev-01"
    assert sample_dev_profile.is_synthetic_demo_data is True
    summary = sample_dev_profile.to_summary()
    assert summary["id"] == "comp-dev-01"
    assert "signals" not in summary  # summary is compact


def test_deterministic_scoring_qualified(
    sample_dev_profile: CompanyProfile,
    default_criteria: QualificationCriteria,
) -> None:
    """Dev tool with matching size and signals scores >= 70 and qualifies."""
    evaluation = evaluate_company_deterministic(sample_dev_profile, default_criteria)
    assert evaluation.qualification_status == QualificationStatus.QUALIFIED
    assert evaluation.score_breakdown.total_score >= 70
    assert evaluation.score_breakdown.developer_alignment_score > 0
    assert evaluation.score_breakdown.scale_and_stage_score > 0
    assert evaluation.score_breakdown.engagement_signals_score > 0
    assert len(evaluation.evidence) >= 3
    assert len(evaluation.missing_information) == 0
    assert "Outreach Recommended" in evaluation.recommended_action


def test_deterministic_scoring_disqualified(
    sample_consumer_profile: CompanyProfile,
    default_criteria: QualificationCriteria,
) -> None:
    """Consumer app targeting non-technical audience fails developer criteria."""
    evaluation = evaluate_company_deterministic(sample_consumer_profile, default_criteria)
    assert evaluation.qualification_status == QualificationStatus.DISQUALIFIED
    assert evaluation.score_breakdown.developer_alignment_score == 0
    assert evaluation.score_breakdown.total_score < 50
    assert "Do Not Contact" in evaluation.recommended_action


def test_deterministic_scoring_insufficient_data(
    sample_sparse_profile: CompanyProfile,
    default_criteria: QualificationCriteria,
) -> None:
    """Company missing critical audience and scale info gets flagged as INSUFFICIENT_DATA."""
    evaluation = evaluate_company_deterministic(sample_sparse_profile, default_criteria)
    assert evaluation.qualification_status == QualificationStatus.INSUFFICIENT_DATA
    assert len(evaluation.missing_information) >= 2
    assert any("target_audience" in item for item in evaluation.missing_information)
    assert any("employee_count" in item for item in evaluation.missing_information)
    assert "Manual Research Needed" in evaluation.recommended_action


def test_evidence_groundedness(sample_dev_profile: CompanyProfile, default_criteria: QualificationCriteria) -> None:
    """Checks that every generated evidence item references an actual field."""
    evaluation = evaluate_company_deterministic(sample_dev_profile, default_criteria)
    for item in evaluation.evidence:
        assert item.field_name
        assert item.observed_value is not None
        assert item.relevance
