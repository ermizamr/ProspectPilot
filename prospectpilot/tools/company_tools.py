"""Concrete tools for searching and retrieving company profiles from the dataset."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from prospectpilot.dataset import CompanyDataset
from prospectpilot.tools.base import Tool, ToolRegistry


class SearchCompaniesArgs(BaseModel):
    """Arguments for searching the demonstration company dataset."""
    query: Optional[str] = Field(
        default=None,
        description="Keyword to match against company name, product description, or tech stack (e.g. 'eBPF', 'PostgreSQL', 'API testing')."
    )
    industry: Optional[str] = Field(
        default=None,
        description="Filter by industry name (e.g. 'Developer Tools', 'Human Resources', 'Accounting')."
    )
    min_size: Optional[int] = Field(
        default=None,
        description="Minimum employee count if specified by criteria."
    )
    max_size: Optional[int] = Field(
        default=None,
        description="Maximum employee count if specified by criteria."
    )


class GetCompanyProfileArgs(BaseModel):
    """Arguments for retrieving the complete profile of a specific company."""
    company_id: str = Field(
        ...,
        description="The unique stable identifier of the company (e.g. 'comp-001')."
    )


def register_company_tools(registry: ToolRegistry, dataset: CompanyDataset) -> None:
    """Registers the research tools backed by the in-memory dataset."""

    def search_handler(
        query: Optional[str] = None,
        industry: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        results = dataset.search(
            query=query,
            industry=industry,
            min_size=min_size,
            max_size=max_size,
        )
        return results

    def profile_handler(company_id: str) -> Dict[str, Any]:
        profile = dataset.get_profile(company_id)
        if not profile:
            raise KeyError(f"Company ID '{company_id}' not found in demonstration dataset. Available IDs: {dataset.list_all_ids()}")
        return profile.model_dump()

    registry.register(
        Tool(
            name="search_companies",
            description="Search the company dataset by keyword, industry, or employee count. Returns compact summaries with company IDs.",
            args_schema=SearchCompaniesArgs,
            handler=search_handler,
        )
    )

    registry.register(
        Tool(
            name="get_company_profile",
            description="Retrieve the complete profile, technical details, tech stack, and public engagement signals for a specific company ID.",
            args_schema=GetCompanyProfileArgs,
            handler=profile_handler,
        )
    )
