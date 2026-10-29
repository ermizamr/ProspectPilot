"""Tests for tool validation, schema generation, and sandboxed execution."""

from typing import Any, Dict
from pydantic import BaseModel, Field
import pytest

from prospectpilot.tools.base import Tool, ToolRegistry, ToolResult


class SampleArgs(BaseModel):
    query: str = Field(description="Search query string")
    limit: int = Field(default=10, description="Max results")


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    def sample_handler(query: str, limit: int = 10) -> Dict[str, Any]:
        if query == "trigger_error":
            raise RuntimeError("Underlying service exploded")
        return {"query": query, "count": limit, "items": ["result1", "result2"]}

    reg.register(
        Tool(
            name="sample_tool",
            description="A sample tool for testing registry functionality.",
            args_schema=SampleArgs,
            handler=sample_handler,
        )
    )
    return reg


def test_tool_execution_success(registry: ToolRegistry) -> None:
    """Verifies that executing a registered tool returns success and data."""
    res = registry.execute("sample_tool", {"query": "devops", "limit": 5})
    assert res.success is True
    assert res.data == {"query": "devops", "count": 5, "items": ["result1", "result2"]}
    assert res.execution_time_ms >= 0
    assert "result1" in res.to_observation_string()


def test_tool_argument_validation_failure(registry: ToolRegistry) -> None:
    """Verifies that missing or invalid arguments fail gracefully with Pydantic validation errors."""
    # Missing required 'query'
    res = registry.execute("sample_tool", {})
    assert res.success is False
    assert res.error is not None
    assert "validation failed" in res.error.lower()
    assert "query" in res.error
    assert "TOOL ERROR: Argument validation failed" in res.to_observation_string()


def test_tool_runtime_error_sandboxing(registry: ToolRegistry) -> None:
    """Verifies that an exception in handler returns a structured error message without crashing."""
    res = registry.execute("sample_tool", {"query": "trigger_error"})
    assert res.success is False
    assert res.error is not None
    assert "Underlying service exploded" in res.error
    assert "TOOL ERROR: Execution error in 'sample_tool'" in res.to_observation_string()


def test_tool_openai_schema_generation(registry: ToolRegistry) -> None:
    """Verifies that registered tools generate valid OpenAI function schemas."""
    schemas = registry.get_openai_tools()
    assert len(schemas) == 1

    fn = schemas[0]["function"]
    assert fn["name"] == "sample_tool"
    assert "description" in fn
    assert fn["parameters"]["type"] == "object"
    assert "query" in fn["parameters"]["properties"]


def test_unregistered_tool_call(registry: ToolRegistry) -> None:
    """Calling an unknown tool returns a clean error listing available tools."""
    res = registry.execute("execute_arbitrary_code", {})
    assert res.success is False
    assert res.error is not None
    assert "is not registered" in res.error
    assert "sample_tool" in res.error
