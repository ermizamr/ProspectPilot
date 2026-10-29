"""Shared pytest fixtures."""

from pathlib import Path
import pytest

from prospectpilot.dataset import CompanyDataset
from prospectpilot.models import (
    CompanyProfile,
    CompanySignals,
    QualificationCriteria,
)


@pytest.fixture
def sample_dev_profile() -> CompanyProfile:
    return CompanyProfile(
        id="comp-dev-01",
        name="DevToolCo",
        domain="devtool.io",
        industry="Developer Tools",
        description="High performance distributed tracing for backend engineers.",
        is_synthetic_demo_data=True,
        product_category="Observability",
        target_audience="Software Engineers, SREs",
        employee_count=45,
        funding_stage="Series A",
        tech_stack=["Go", "Rust", "Kubernetes"],
        signals=CompanySignals(
            has_public_github=True,
            has_active_blog=True,
            blog_focus="Kernel eBPF internals",
            hiring_roles=["Senior Go Engineer", "DevRel Advocate"]
        ),
    )


@pytest.fixture
def sample_consumer_profile() -> CompanyProfile:
    return CompanyProfile(
        id="comp-consumer-01",
        name="ChoreBuddy",
        domain="chorebuddy.app",
        industry="Consumer Mobile Apps",
        description="Gamified household chore tracker for families.",
        is_synthetic_demo_data=True,
        product_category="B2C Mobile App",
        target_audience="Parents and families",
        employee_count=15,
        funding_stage="Seed",
        tech_stack=["Flutter", "Firebase"],
        signals=CompanySignals(
            has_public_github=False,
            has_active_blog=False,
            hiring_roles=["Marketing Specialist"]
        ),
    )


@pytest.fixture
def sample_sparse_profile() -> CompanyProfile:
    return CompanyProfile(
        id="comp-sparse-01",
        name="StealthDev",
        domain="stealthdev.internal",
        industry="Developer Tools",
        description="Next-gen compiler in stealth.",
        is_synthetic_demo_data=True,
        product_category="Compilers",
        target_audience=None,
        employee_count=None,
        funding_stage=None,
        tech_stack=["Rust"],
        signals=CompanySignals(
            has_public_github=True,
            has_active_blog=False,
            hiring_roles=[]
        ),
    )


@pytest.fixture
def default_criteria() -> QualificationCriteria:
    return QualificationCriteria(
        target_focus="Developer Tools",
        require_developer_audience=True,
        min_team_size=10,
        max_team_size=100,
    )


@pytest.fixture
def real_demo_dataset() -> CompanyDataset:
    return CompanyDataset.load_from_file()
