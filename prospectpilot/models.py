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
    source_url: Optional[str] = Field(default=None, description="URL where this evidence was observed")


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
    is_synthetic_demo_data: bool = False


class AgentRunResult(BaseModel):
    """Final synthesis report delivered to the user at the end of the run."""
    goal: str
    criteria: QualificationCriteria
    status: str = Field(description="SUCCESS, PARTIAL_BUDGET_EXHAUSTED, or FAILED")
    evaluations: List[CompanyEvaluation] = Field(default_factory=list)
    summary: str
    total_iterations: int
    total_tool_calls: int
    total_tokens: int = 0
    duration_seconds: float
    is_synthetic_demo_run: bool = False


# ==============================================================================
# GitHub-Sourced Models and Evidence-Based Scoring
# ==============================================================================

from datetime import datetime, timedelta, timezone

DEV_TECH_KEYWORDS = {
    "go", "rust", "python", "typescript", "kubernetes", "ebpf", 
    "docker", "openapi", "postgresql", "graphql", "github actions"
}


class GitHubRepoSummary(BaseModel):
    """Summary of a GitHub repository, parsed from API responses."""
    full_name: str = Field(description="'owner/repo' format")
    owner_login: str
    owner_type: str = Field(description="'Organization' or 'User'")
    name: str
    description: Optional[str] = None
    html_url: str
    stars: int = 0
    forks: int = 0
    language: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    license_name: Optional[str] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None
    open_issues_count: int = 0
    archived: bool = False
    is_fork: bool = False
    retrieved_at: str = Field(description="ISO 8601 timestamp of when this data was fetched")


class GitHubOrgSummary(BaseModel):
    """Summary of a GitHub organization, parsed from API responses."""
    login: str
    name: Optional[str] = None
    description: Optional[str] = None
    html_url: str
    blog: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    public_repos: int = 0
    created_at: Optional[str] = None
    retrieved_at: str = Field(description="ISO 8601 timestamp of when this data was fetched")


class GitHubProspect(BaseModel):
    """Aggregated prospect built from GitHub evidence.

    Groups repositories under an organization with source URLs and timestamps.
    """
    org_login: str
    org_name: Optional[str] = None
    org_url: str
    org_blog: Optional[str] = None
    org_description: Optional[str] = None
    org_public_repos: int = 0
    org_company_field: Optional[str] = None
    repos: List[GitHubRepoSummary] = Field(default_factory=list)
    top_languages: List[str] = Field(default_factory=list)
    total_stars: int = 0
    has_recent_releases: bool = False
    retrieved_at: str
    source_urls: List[str] = Field(default_factory=list)
    is_synthetic_demo_data: bool = False


# Keywords for matching GitHub topics against developer-tool relevance
DEV_TOOL_TOPIC_KEYWORDS = frozenset({
    "developer-tools", "devtools", "sdk", "api", "cli", "framework",
    "library", "testing", "monitoring", "observability", "tracing",
    "debugging", "deployment", "ci-cd", "infrastructure", "container",
    "kubernetes", "docker", "terraform", "database", "orm", "migration",
    "security", "linting", "ide", "plugin", "build-tool", "compiler",
    "devops", "platform", "backend", "cloud-native", "open-source",
    "serverless", "microservices", "package-manager",
})

# Keywords for matching GitHub descriptions against developer-tool relevance
DEV_TOOL_DESC_KEYWORDS = frozenset({
    "developer", "engineering", "sdk", "api", "cli", "framework",
    "library", "testing", "monitoring", "observability", "tracing",
    "debugging", "deployment", "infrastructure", "container",
    "kubernetes", "docker", "terraform", "database", "migration",
    "security", "linting", "compiler", "devops", "platform", "backend",
    "open source", "open-source", "toolkit", "build tool",
})


def _is_recently_updated(updated_at: Optional[str], days: int = 90) -> bool:
    """Check if an ISO 8601 timestamp is within the last N days."""
    if not updated_at:
        return False
    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - updated) < timedelta(days=days)
    except (ValueError, TypeError):
        return False


def evaluate_github_prospect_deterministic(
    prospect: GitHubProspect,
    criteria: QualificationCriteria,
) -> CompanyEvaluation:
    """Compute a deterministic qualification score for a GitHub-evidenced prospect.

    Scoring rubric:
      1. Developer Relevance (max 40): topic/description keyword matches + language affinity
      2. Activity & Scale (max 30): stars, recent updates, releases, repo count
      3. Organization Signals (max 30): is org, has blog, has description, has company field

    IMPORTANT: GitHub data is evidence of public development activity, NOT proof of
    company headcount, revenue, purchasing intent, or current hiring. Unavailable
    data is explicitly marked as unknown.
    """
    dev_score = 0
    scale_score = 0
    signal_score = 0
    rules: List[str] = []
    evidence: List[EvidenceItem] = []
    missing_info: List[str] = []

    # --- Epistemic disclaimers: what GitHub CANNOT tell us ---
    missing_info.append("Company headcount: unavailable from GitHub API")
    missing_info.append("Revenue and funding stage: unavailable from GitHub API")
    if not prospect.org_company_field:
        missing_info.append(
            "Organization-to-company identity: unverified (org 'company' field not set)"
        )

    # Collect all topics and descriptions from repos
    all_topics: set[str] = set()
    all_desc_words: set[str] = set()
    for repo in prospect.repos:
        all_topics.update(t.lower() for t in repo.topics)
        if repo.description:
            all_desc_words.update(repo.description.lower().split())

    # ---- 1. Developer Relevance (max 40) ----
    matching_topics = all_topics & DEV_TOOL_TOPIC_KEYWORDS
    matching_desc = all_desc_words & DEV_TOOL_DESC_KEYWORDS

    # Also check org description
    if prospect.org_description:
        org_desc_words = set(prospect.org_description.lower().split())
        matching_desc |= org_desc_words & DEV_TOOL_DESC_KEYWORDS

    if matching_topics or matching_desc:
        dev_score += 25
        keyword_sample = sorted(matching_topics | matching_desc)[:5]
        rules.append(
            f"+25: Developer-tool relevance matched via keywords: {', '.join(keyword_sample)}"
        )
        evidence.append(EvidenceItem(
            field_name="topics_and_description",
            observed_value=keyword_sample,
            relevance="Repository topics/descriptions match developer-tool keywords",
            source_url=prospect.org_url,
        ))
    else:
        rules.append("+0: No developer-tool keywords found in topics or descriptions")

    # Language affinity (up to 15)
    all_languages = {r.language.lower() for r in prospect.repos if r.language}
    matching_langs = all_languages & DEV_TECH_KEYWORDS
    if matching_langs:
        lang_points = min(15, len(matching_langs) * 3)
        dev_score += lang_points
        rules.append(
            f"+{lang_points}: Languages include {', '.join(sorted(matching_langs))}"
        )
        evidence.append(EvidenceItem(
            field_name="languages",
            observed_value=sorted(matching_langs),
            relevance=f"Repos use {len(matching_langs)} developer-relevant languages",
            source_url=prospect.org_url,
        ))
    else:
        rules.append("+0: No developer-focused languages detected")

    # ---- 2. Activity & Scale (max 30) ----
    total_stars = prospect.total_stars
    if total_stars >= 5000:
        scale_score += 15
        rules.append(f"+15: High community traction ({total_stars:,} total stars)")
    elif total_stars >= 500:
        scale_score += 10
        rules.append(f"+10: Moderate community traction ({total_stars:,} total stars)")
    elif total_stars >= 50:
        scale_score += 5
        rules.append(f"+5: Some community traction ({total_stars:,} total stars)")
    else:
        rules.append(f"+0: Low star count ({total_stars:,} total stars)")

    evidence.append(EvidenceItem(
        field_name="total_stars",
        observed_value=total_stars,
        relevance=f"Combined star count across {len(prospect.repos)} examined repos",
        source_url=prospect.org_url,
    ))

    # Recent activity
    any_recent = any(_is_recently_updated(r.updated_at) for r in prospect.repos)
    if any_recent:
        scale_score += 5
        rules.append("+5: At least one repo updated within the last 90 days")
    else:
        rules.append("+0: No repos updated in the last 90 days")

    # Releases
    if prospect.has_recent_releases:
        scale_score += 5
        rules.append("+5: Has recent releases")
        evidence.append(EvidenceItem(
            field_name="has_recent_releases",
            observed_value=True,
            relevance="Active release cadence indicates maintained software",
        ))
    else:
        rules.append("+0: No recent releases found")

    # Multiple repos
    if len(prospect.repos) >= 3:
        scale_score += 5
        rules.append(f"+5: Multiple public repos examined ({len(prospect.repos)})")
    else:
        rules.append(f"+0: Few repos examined ({len(prospect.repos)})")

    # ---- 3. Organization Signals (max 30) ----
    # Is an actual organization?
    is_org = any(r.owner_type == "Organization" for r in prospect.repos)
    if is_org:
        signal_score += 10
        rules.append("+10: GitHub account is an Organization (not personal)")
        evidence.append(EvidenceItem(
            field_name="owner_type",
            observed_value="Organization",
            relevance="Confirmed GitHub Organization account",
            source_url=prospect.org_url,
        ))
    else:
        rules.append("+0: GitHub account appears to be a personal user, not an Organization")
        missing_info.append(
            "Owner is a personal GitHub account — may not represent a company"
        )

    if prospect.org_blog:
        signal_score += 10
        rules.append(f"+10: Organization has a blog/website: {prospect.org_blog}")
        evidence.append(EvidenceItem(
            field_name="org_blog",
            observed_value=prospect.org_blog,
            relevance="Public web presence beyond GitHub",
            source_url=prospect.org_url,
        ))
    else:
        rules.append("+0: No blog/website listed on organization profile")

    if prospect.org_description:
        signal_score += 5
        rules.append("+5: Organization has a description")
    else:
        rules.append("+0: No organization description")

    if prospect.org_company_field:
        signal_score += 5
        rules.append(f"+5: Company field set: {prospect.org_company_field}")
        evidence.append(EvidenceItem(
            field_name="org_company",
            observed_value=prospect.org_company_field,
            relevance="Self-reported company identity on GitHub",
            source_url=prospect.org_url,
        ))
    else:
        rules.append("+0: Company field not set on GitHub profile")

    total = dev_score + scale_score + signal_score

    # Qualification status
    # For GitHub prospects, we always have at least 2 "missing" items (headcount, revenue),
    # so we use a higher threshold for INSUFFICIENT_DATA
    critical_missing = sum(1 for m in missing_info if "unverified" in m or "personal" in m)
    if critical_missing >= 1 and total < 70:
        status = QualificationStatus.INSUFFICIENT_DATA
    elif total >= 70:
        status = QualificationStatus.QUALIFIED
    elif total < 50:
        status = QualificationStatus.DISQUALIFIED
    else:
        status = QualificationStatus.INSUFFICIENT_DATA

    # Default actions
    display_name = prospect.org_name or prospect.org_login
    default_actions = {
        QualificationStatus.QUALIFIED: (
            f"Outreach Recommended: Reference their open-source work at {prospect.org_url}. "
            f"Verify company identity before contacting."
        ),
        QualificationStatus.DISQUALIFIED: (
            "Do Not Contact: GitHub profile does not match developer-tool outreach criteria."
        ),
        QualificationStatus.INSUFFICIENT_DATA: (
            "Manual Research Needed: Verify organization-to-company identity and product "
            "relevance before initiating outreach."
        ),
    }

    domain = prospect.org_blog or prospect.org_url
    explanation = f"Scored {total}/100 based on GitHub evidence. " + " ".join(rules[:3])

    return CompanyEvaluation(
        company_id=prospect.org_login,
        company_name=display_name,
        domain=domain,
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
        recommended_action=default_actions[status],
        is_synthetic_demo_data=False,
    )
