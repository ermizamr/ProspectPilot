"""GitHub API tools for searching repositories, inspecting orgs, and fetching releases.

Each tool uses Pydantic schemas for argument validation and registers into the
existing ToolRegistry. Handlers parse GitHub API responses into typed summaries
with retrieval timestamps and source URLs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from prospectpilot.github_client import GitHubClient
from prospectpilot.models import GitHubOrgSummary, GitHubRepoSummary
from prospectpilot.tools.base import Tool, ToolRegistry


# ------------------------------------------------------------------
# Argument schemas (Pydantic-validated)
# ------------------------------------------------------------------

class SearchGitHubReposArgs(BaseModel):
    """Arguments for searching GitHub repositories."""
    query: str = Field(
        description=(
            "GitHub search query. Supports qualifiers like 'topic:developer-tools', "
            "'language:python', 'stars:>100'. Example: 'topic:cli language:go stars:>50'"
        )
    )
    sort: str = Field(
        default="stars",
        description="Sort field: 'stars', 'forks', 'updated', or 'help-wanted-issues'.",
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page number to retrieve for paginated searches (starts at 1).",
    )
    per_page: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Number of results per page (1-30).",
    )


class GetRepoDetailsArgs(BaseModel):
    """Arguments for fetching detailed repository metadata."""
    owner: str = Field(description="Repository owner (user or org login).")
    repo: str = Field(description="Repository name.")


class GetOrgProfileArgs(BaseModel):
    """Arguments for fetching a GitHub organization profile."""
    org: str = Field(description="Organization login name (e.g. 'hashicorp').")


class GetRecentReleasesArgs(BaseModel):
    """Arguments for fetching recent releases of a repository."""
    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    per_page: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of releases to fetch (1-10).",
    )


# ------------------------------------------------------------------
# Response parsers
# ------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_url(url: Optional[str]) -> Optional[str]:
    """Cleans blog/website URLs, ensuring https:// scheme if bare domain."""
    if not url:
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        return f"https://{cleaned}"
    return cleaned


def _parse_repo_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a GitHub API repository object into a GitHubRepoSummary dict."""
    owner = item.get("owner") or {}
    license_info = item.get("license") or {}
    desc = item.get("description")
    return GitHubRepoSummary(
        full_name=item.get("full_name", ""),
        owner_login=owner.get("login", ""),
        owner_type=owner.get("type", "User"),
        name=item.get("name", ""),
        description=desc.strip() if desc else None,
        html_url=item.get("html_url", ""),
        stars=item.get("stargazers_count", 0),
        forks=item.get("forks_count", 0),
        language=item.get("language"),
        topics=item.get("topics") or [],
        license_name=license_info.get("spdx_id") if license_info else None,
        updated_at=item.get("updated_at"),
        created_at=item.get("created_at"),
        open_issues_count=item.get("open_issues_count", 0),
        archived=item.get("archived", False),
        is_fork=item.get("fork", False),
        retrieved_at=_now_iso(),
    ).model_dump()


def _parse_org_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a GitHub API organization object into a GitHubOrgSummary dict."""
    desc = data.get("description")
    company = data.get("company")
    return GitHubOrgSummary(
        login=data.get("login", ""),
        name=data.get("name"),
        description=desc.strip() if desc else None,
        html_url=data.get("html_url", ""),
        blog=_clean_url(data.get("blog")),
        company=company.strip() if company else None,
        location=data.get("location"),
        public_repos=data.get("public_repos", 0),
        created_at=data.get("created_at"),
        retrieved_at=_now_iso(),
    ).model_dump()


# ------------------------------------------------------------------
# Tool registration
# ------------------------------------------------------------------

def register_github_tools(registry: ToolRegistry, github_client: GitHubClient) -> None:
    """Register GitHub research tools backed by a live GitHubClient."""

    def search_repos_handler(
        query: str,
        sort: str = "stars",
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        raw = github_client.search_repositories(query=query, sort=sort, page=page, per_page=per_page)
        items = raw.get("items", [])
        repos = [_parse_repo_summary(item) for item in items]
        return {
            "total_count": raw.get("total_count", 0),
            "page": page,
            "returned_count": len(repos),
            "repos": repos,
        }

    def get_repo_details_handler(owner: str, repo: str) -> Dict[str, Any]:
        raw = github_client.get_repository(owner=owner, repo=repo)
        return _parse_repo_summary(raw)

    def get_org_profile_handler(org: str) -> Dict[str, Any]:
        raw = github_client.get_organization(org=org)
        return _parse_org_summary(raw)

    def get_recent_releases_handler(
        owner: str, repo: str, per_page: int = 5
    ) -> List[Dict[str, Any]]:
        raw = github_client.get_releases(owner=owner, repo=repo, per_page=per_page)
        return [
            {
                "tag_name": r.get("tag_name", ""),
                "name": r.get("name", ""),
                "published_at": r.get("published_at"),
                "html_url": r.get("html_url", ""),
                "prerelease": r.get("prerelease", False),
            }
            for r in raw
        ]

    registry.register(Tool(
        name="search_github_repos",
        description=(
            "Search GitHub repositories by keyword, topic, language, or star count. "
            "Returns compact summaries with owner info, stars, language, and topics. "
            "Use GitHub search qualifiers like 'topic:cli language:go stars:>100'."
        ),
        args_schema=SearchGitHubReposArgs,
        handler=search_repos_handler,
    ))

    registry.register(Tool(
        name="get_repo_details",
        description=(
            "Get detailed metadata for a specific GitHub repository: stars, forks, "
            "language, topics, license, recent activity, and owner organization info."
        ),
        args_schema=GetRepoDetailsArgs,
        handler=get_repo_details_handler,
    ))

    registry.register(Tool(
        name="get_org_profile",
        description=(
            "Get a GitHub organization's profile: name, description, blog URL, "
            "company field, location, and public repo count. Use this to assess "
            "whether a repository owner is a company with public developer presence."
        ),
        args_schema=GetOrgProfileArgs,
        handler=get_org_profile_handler,
    ))

    registry.register(Tool(
        name="get_recent_releases",
        description=(
            "Get recent releases for a repository. Useful for assessing whether "
            "a project is actively maintained and shipping updates."
        ),
        args_schema=GetRecentReleasesArgs,
        handler=get_recent_releases_handler,
    ))
