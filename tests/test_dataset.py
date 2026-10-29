"""Tests for demonstration dataset loading and in-memory search."""

import pytest
from prospectpilot.dataset import CompanyDataset


def test_dataset_loading(real_demo_dataset: CompanyDataset) -> None:
    """Verifies that the demo JSON dataset loads and contains synthetic records."""
    assert len(real_demo_dataset) >= 8
    all_ids = real_demo_dataset.list_all_ids()
    assert "comp-001" in all_ids
    assert "comp-007" in all_ids


def test_get_profile_by_id(real_demo_dataset: CompanyDataset) -> None:
    """Verifies exact retrieval by ID."""
    profile = real_demo_dataset.get_profile("comp-001")
    assert profile is not None
    assert profile.name == "TracePulse"
    assert profile.is_synthetic_demo_data is True
    assert "eBPF" in profile.tech_stack

    missing = real_demo_dataset.get_profile("non-existent-id")
    assert missing is None


def test_search_by_keyword(real_demo_dataset: CompanyDataset) -> None:
    """Search matches names, descriptions, or tech stack."""
    results = real_demo_dataset.search(query="eBPF")
    assert len(results) >= 1
    assert any(r["id"] == "comp-001" for r in results)

    db_results = real_demo_dataset.search(query="PostgreSQL")
    assert len(db_results) >= 1


def test_search_by_industry_filter(real_demo_dataset: CompanyDataset) -> None:
    """Filter limits results to target industry."""
    dev_results = real_demo_dataset.search(industry="Developer Tools")
    assert len(dev_results) >= 3
    for r in dev_results:
        assert "Developer Tools" in r["industry"]


def test_search_by_size_bounds(real_demo_dataset: CompanyDataset) -> None:
    """Filter limits results based on min and max employee count."""
    results = real_demo_dataset.search(min_size=40, max_size=70)
    assert len(results) >= 2
    for r in results:
        if r["employee_count"] is not None:
            assert 40 <= r["employee_count"] <= 70
