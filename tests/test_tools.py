"""Tests for tool validation, schema generation, and sandboxed execution."""

import pytest
from prospectpilot.dataset import CompanyDataset
from prospectpilot.tools.base import ToolRegistry, ToolResult
from prospectpilot.tools.company_tools import register_company_tools


@pytest.fixture
def registry(real_demo_dataset: CompanyDataset) -> ToolRegistry:
    reg = ToolRegistry()
    register_company_tools(reg, real_demo_dataset)
    return reg


def test_search_companies_tool_success(registry: ToolRegistry) -> None:
    """Verifies that search_companies returns valid data."""
    res = registry.execute("search_companies", {"query": "eBPF"})
    assert res.success is True
    assert isinstance(res.data, list)
    assert len(res.data) >= 1
    assert res.data[0]["id"] == "comp-001"
    assert res.execution_time_ms >= 0


def test_get_company_profile_tool_success(registry: ToolRegistry) -> None:
    """Verifies that get_company_profile returns the complete profile dict."""
    res = registry.execute("get_company_profile", {"company_id": "comp-001"})
    assert res.success is True
    assert isinstance(res.data, dict)
    assert res.data["name"] == "TracePulse"
    assert "tech_stack" in res.data
    assert "signals" in res.data


def test_tool_argument_validation_failure(registry: ToolRegistry) -> None:
    """Verifies that missing or invalid arguments fail gracefully with Pydantic validation errors."""
    # Missing required 'company_id'
    res = registry.execute("get_company_profile", {})
    assert res.success is False
    assert "validation failed" in res.error.lower()
    assert "company_id" in res.error


def test_tool_runtime_error_unknown_company(registry: ToolRegistry) -> None:
    """Verifies that an unknown company ID returns a structured error message rather than crashing."""
    res = registry.execute("get_company_profile", {"company_id": "comp-unknown-999"})
    assert res.success is False
    assert "not found in demonstration dataset" in res.error


def test_tool_openai_schema_generation(registry: ToolRegistry) -> None:
    """Verifies that registered tools generate valid OpenAI function schemas."""
    schemas = registry.get_openai_tools()
    assert len(schemas) == 2

    tool_names = [s["function"]["name"] for s in schemas]
    assert "search_companies" in tool_names
    assert "get_company_profile" in tool_names

    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_unregistered_tool_call(registry: ToolRegistry) -> None:
    """Calling an unknown tool returns a clean error listing available tools."""
    res = registry.execute("execute_arbitrary_code", {})
    assert res.success is False
    assert "is not registered" in res.error
    assert "search_companies" in res.error
