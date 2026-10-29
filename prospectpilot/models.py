"""Domain models, Pydantic schemas, and deterministic qualification logic."""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class QualificationStatus(str, Enum):
    """Transparent qualification states."""
    QUALIFIED = "QUALIFIED"
    DISQUALIFIED = "DISQUALIFIED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class CompanySignals(BaseModel):
    """Public engagement and developer outreach signals."""
    has_public_github: bool = False
    has_active_blog: bool = False
    blog_focus: Optional[str] = None
    recent_changelog: Optional[str] = None
    hiring_roles: List[str] = Field(default_factory=list)


class CompanyProfile(BaseModel):
    """Controlled representation of a company from the demonstration dataset."""
    id: str
    name: str
    domain: str
    industry: str
    description: str
    is_synthetic_demo_data: bool = True
    product_category: str
    target_audience: Optional[str] = None
    employee_count: Optional[int] = None
    funding_stage: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)
    pricing_model: Optional[str] = None
    signals: CompanySignals = Field(default_factory=CompanySignals)

    def to_summary(self) -> dict[str, Any]:
        """Compact view returned during broad search to conserve model context."""
        return {
            "id": self.id,
            "name": self.name,
            "industry": self.industry,
            "product_category": self.product_category,
            "description": self.description,
            "employee_count": self.employee_count,
            "is_synthetic_demo_data": self.is_synthetic_demo_data,
        }


class QualificationCriteria(BaseModel):
    """User-specified criteria for outreach qualification."""
    target_focus: str = Field(
        default="Developer Tools",
        description="Core keyword or vertical to prioritize"
    )
    require_developer_audience: bool = Field(
        default=True,
        description="Whether target audience must be engineers or technical practitioners"
    )
    min_team_size: Optional[int] = Field(
        default=5,
        description="Minimum employee count if known"
    )
    max_team_size: Optional[int] = Field(
        default=150,
        description="Maximum employee count if known"
    )
    require_public_signals: bool = Field(
        default=False,
        description="Whether public GitHub or active tech blog is strictly required"
    )


class EvidenceItem(BaseModel):
    """Verifiable claim tied directly to an observed source attribute."""
    field_name: str = Field(description="Exact attribute key from source data")
    observed_value: Any = Field(description="Literal value extracted from dataset")
    relevance: str = Field(description="Why this fact supports or hurts qualification")


class ScoreBreakdown(BaseModel):
    """Transparent rubric score computed by deterministic Python logic."""
    developer_alignment_score: int = Field(ge=0, le=40, description="Max 40 pts")
    scale_and_stage_score: int = Field(ge=0, le=30, description="Max 30 pts")
    engagement_signals_score: int = Field(ge=0, le=30, description="Max 30 pts")
    total_score: int = Field(ge=0, le=100, description="Sum of category scores")
    rules_applied: List[str] = Field(default_factory=list)


class CompanyEvaluation(BaseModel):
    """Structured, evidence-backed evaluation result for a single company."""
    company_id: str
    company_name: str
    domain: str
    qualification_status: QualificationStatus
    score_breakdown: ScoreBreakdown
    evidence: List[EvidenceItem]
    missing_information: List[str]
    explanation: str
    recommended_action: str
    is_synthetic_demo_data: bool = True


class AgentRunResult(BaseModel):
    """Final synthesis report delivered to the user at the end of the run."""
    goal: str
    criteria: QualificationCriteria
    status: str = Field(description="SUCCESS, PARTIAL_BUDGET_EXHAUSTED, or FAILED")
    evaluations: List[CompanyEvaluation] = Field(default_factory=list)
    summary: str
    total_iterations: int
    total_tool_calls: int
    duration_seconds: float
    is_synthetic_demo_run: bool = True


# ==============================================================================
# Deterministic Scoring Engine (Enforced in Python, not LLM guesswork)
# ==============================================================================

DEV_AUDIENCE_KEYWORDS = {
    "software engineer", "software developers", "developer", "sre", 
    "platform engineer", "backend", "devops", "qa automation", "technical"
}

DEV_TECH_KEYWORDS = {
    "go", "rust", "python", "typescript", "kubernetes", "ebpf", 
    "docker", "openapi", "postgresql", "graphql", "github actions"
}


def evaluate_company_deterministic(
    profile: CompanyProfile,
    criteria: QualificationCriteria,
    model_explanation: Optional[str] = None,
    model_recommended_action: Optional[str] = None,
) -> CompanyEvaluation:
    """Computes a deterministic qualification score and evidence list for a company.

    This ensures that scores are mathematically verifiable, transparent,
    and cannot be hallucinated by the language model.
    """
    dev_score = 0
    scale_score = 0
    signal_score = 0
    rules: List[str] = []
    evidence: List[EvidenceItem] = []
    missing_info: List[str] = []

    # 1. Developer Alignment (Max 40)
    if profile.target_audience:
        audience_lower = profile.target_audience.lower()
        if any(kw in audience_lower for kw in DEV_AUDIENCE_KEYWORDS):
            dev_score += 25
            rules.append("+25: Target audience explicitly contains software engineers/developers")
            evidence.append(EvidenceItem(
                field_name="target_audience",
                observed_value=profile.target_audience,
                relevance="Directly targets technical developers"
            ))
        else:
            rules.append("+0: Target audience does not target software developers")
            evidence.append(EvidenceItem(
                field_name="target_audience",
                observed_value=profile.target_audience,
                relevance="Audience is non-technical or consumer-facing"
            ))
    else:
        missing_info.append("target_audience is unknown")
        rules.append("+0: Missing target audience data")

    # Tech stack developer affinity (up to 15)
    matching_tech = [t for t in profile.tech_stack if t.lower() in DEV_TECH_KEYWORDS]
    if matching_tech:
        tech_points = min(15, len(matching_tech) * 3)
        dev_score += tech_points
        rules.append(f"+{tech_points}: Tech stack features {len(matching_tech)} modern developer technologies ({', '.join(matching_tech)})")
        evidence.append(EvidenceItem(
            field_name="tech_stack",
            observed_value=profile.tech_stack,
            relevance=f"Demonstrates engineering depth with {', '.join(matching_tech)}"
        ))
    else:
        if not profile.tech_stack:
            missing_info.append("tech_stack is not specified")
        rules.append("+0: No prominent developer-focused technologies in stack")

    # 2. Company Scale & Stage Fit (Max 30)
    if profile.employee_count is not None:
        evidence.append(EvidenceItem(
            field_name="employee_count",
            observed_value=profile.employee_count,
            relevance=f"Team size of {profile.employee_count} employees"
        ))
        min_s = criteria.min_team_size or 0
        max_s = criteria.max_team_size or 1000000
        if min_s <= profile.employee_count <= max_s:
            scale_score += 20
            rules.append(f"+20: Team size ({profile.employee_count}) fits target range [{min_s}, {max_s}]")
        else:
            rules.append(f"+5: Team size ({profile.employee_count}) outside optimal range [{min_s}, {max_s}]")
            scale_score += 5
    else:
        missing_info.append("employee_count is unknown")
        rules.append("+0: Unknown employee count")

    if profile.funding_stage:
        scale_score += 10
        rules.append(f"+10: Verified funding stage ({profile.funding_stage})")
        evidence.append(EvidenceItem(
            field_name="funding_stage",
            observed_value=profile.funding_stage,
            relevance="Known venture stage indicates business stability"
        ))
    else:
        missing_info.append("funding_stage is unknown")
        rules.append("+0: Unknown funding stage")

    # 3. Outreach & Engagement Signals (Max 30)
    if profile.signals.has_public_github:
        signal_score += 10
        rules.append("+10: Public GitHub presence exists")
        evidence.append(EvidenceItem(
            field_name="signals.has_public_github",
            observed_value=True,
            relevance="Open-source repository available for developer engagement"
        ))

    if profile.signals.has_active_blog:
        signal_score += 10
        rules.append("+10: Active engineering/product blog")
        evidence.append(EvidenceItem(
            field_name="signals.has_active_blog",
            observed_value=True,
            relevance=f"Blog focus: {profile.signals.blog_focus or 'Active'}"
        ))

    dev_hiring = [r for r in profile.signals.hiring_roles if any(k in r.lower() for k in ["engineer", "developer", "devrel", "architect"])]
    if dev_hiring:
        signal_score += 10
        rules.append(f"+10: Actively recruiting developer roles ({', '.join(dev_hiring)})")
        evidence.append(EvidenceItem(
            field_name="signals.hiring_roles",
            observed_value=dev_hiring,
            relevance="Actively expanding developer team"
        ))

    total = dev_score + scale_score + signal_score

    # Determine status
    if len(missing_info) >= 2 and total < 70:
        status = QualificationStatus.INSUFFICIENT_DATA
    elif total >= 70:
        status = QualificationStatus.QUALIFIED
    else:
        status = QualificationStatus.DISQUALIFIED

    # Default explanation / recommended action if not supplied by model
    default_action = {
        QualificationStatus.QUALIFIED: f"Outreach Recommended: Craft personalized email referencing {profile.signals.blog_focus or 'their developer product'}.",
        QualificationStatus.DISQUALIFIED: "Do Not Contact: Profile does not match developer outreach criteria.",
        QualificationStatus.INSUFFICIENT_DATA: "Manual Research Needed: Investigate missing data points before initiating outreach.",
    }[status]

    explanation = model_explanation or f"Scored {total}/100. " + " ".join(rules[:3])
    action = model_recommended_action or default_action

    return CompanyEvaluation(
        company_id=profile.id,
        company_name=profile.name,
        domain=profile.domain,
        qualification_status=status,
        score_breakdown=ScoreBreakdown(
            developer_alignment_score=dev_score,
            scale_and_stage_score=scale_score,
            engagement_signals_score=signal_score,
            total_score=total,
            rules_applied=rules,
        ),
        evidence=evidence,
        missing_information=missing_info,
        explanation=explanation,
        recommended_action=action,
        is_synthetic_demo_data=profile.is_synthetic_demo_data,
    )
