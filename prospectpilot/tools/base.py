"""Sandboxed tool interface, schema generation, and execution registry."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError


class ToolResult(BaseModel):
    """Encapsulates the deterministic outcome of a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    raw_arguments: Dict[str, Any] = {}

    def to_observation_string(self) -> str:
        """Formats the result cleanly for LLM consumption."""
        if not self.success:
            return f"TOOL ERROR: {self.error}"
        if self.data is None:
            return "SUCCESS: No content returned."
        if isinstance(self.data, (dict, list)):
            import json
            return json.dumps(self.data, indent=2)
        return str(self.data)


class Tool:
    """A bounded tool with explicit Pydantic schema validation and error isolation."""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        handler: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.handler = handler

    def execute(self, raw_args: Dict[str, Any]) -> ToolResult:
        """Validates arguments via Pydantic and safely invokes handler.

        Never crashes the caller on invalid arguments or runtime errors.
        """
        start = time.perf_counter()
        # 1. Schema Validation
        try:
            validated_args = self.args_schema.model_validate(raw_args or {})
        except ValidationError as val_err:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Argument validation failed: {val_err.errors()}",
                execution_time_ms=round(elapsed, 2),
                raw_arguments=raw_args,
            )

        # 2. Safe execution
        try:
            # Pass validated fields as kwargs
            result_data = self.handler(**validated_args.model_dump())
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=True,
                data=result_data,
                execution_time_ms=round(elapsed, 2),
                raw_arguments=raw_args,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Execution error in '{self.name}': {str(exc)}",
                execution_time_ms=round(elapsed, 2),
                raw_arguments=raw_args,
            )

    def to_openai_schema(self) -> Dict[str, Any]:
        """Produces OpenAI-compatible function definition."""
        schema = self.args_schema.model_json_schema()
        # Clean metadata if needed
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    """Central registry and dispatcher for sandboxed agent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Registers a tool."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def execute(self, name: str, raw_args: Dict[str, Any]) -> ToolResult:
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' is not registered. Available tools: {list(self._tools.keys())}",
                raw_arguments=raw_args,
            )
        return tool.execute(raw_args)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]
