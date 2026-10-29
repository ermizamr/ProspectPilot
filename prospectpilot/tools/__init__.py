"""ProspectPilot tool system."""

from prospectpilot.tools.base import Tool, ToolRegistry, ToolResult
from prospectpilot.tools.github_tools import register_github_tools

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "register_github_tools",
]
