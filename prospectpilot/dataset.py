"""Dataset loading and in-memory search engine for local company profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from prospectpilot.models import CompanyProfile


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "companies.json"


class CompanyDataset:
    """In-memory search and retrieval engine for the demonstration company dataset."""

    def __init__(self, profiles: List[CompanyProfile]) -> None:
        self._profiles = {p.id: p for p in profiles}

    @classmethod
    def load_from_file(cls, path: Optional[Path] = None) -> CompanyDataset:
        """Loads and validates the company profiles from JSON."""
        target_path = path or DEFAULT_DATASET_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Demonstration dataset not found at: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        profiles = [CompanyProfile.model_validate(item) for item in raw_data]
        return cls(profiles)

    def get_profile(self, company_id: str) -> Optional[CompanyProfile]:
        """Retrieves complete profile by company ID."""
        return self._profiles.get(company_id)

    def search(
        self,
        query: Optional[str] = None,
        industry: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> List[dict[str, Any]]:
        """Searches company profiles and returns compact summaries.

        Compact summaries are intentionally returned to avoid flooding
        the agent's context window with unnecessary fields during broad discovery.
        """
        results: List[dict[str, Any]] = []

        query_tokens = [t.lower() for t in query.split()] if query else []
        industry_norm = industry.lower() if industry else None

        for profile in self._profiles.values():
            # Filter by industry if specified
            if industry_norm and industry_norm not in profile.industry.lower():
                continue

            # Filter by employee count bounds if specified and profile has count
            if profile.employee_count is not None:
                if min_size is not None and profile.employee_count < min_size:
                    continue
                if max_size is not None and profile.employee_count > max_size:
                    continue
            elif min_size is not None or max_size is not None:
                # If size filter is strict and profile has unknown size, skip or keep depending on query
                pass

            # Search text matching
            if query_tokens:
                searchable_text = f"{profile.name} {profile.industry} {profile.product_category} {profile.description} {' '.join(profile.tech_stack)}".lower()
                # Check if any token matches
                if not any(token in searchable_text for token in query_tokens):
                    continue

            results.append(profile.to_summary())

        return results

    def list_all_ids(self) -> List[str]:
        """Returns all company IDs in the dataset."""
        return list(self._profiles.keys())

    def __len__(self) -> int:
        return len(self._profiles)
