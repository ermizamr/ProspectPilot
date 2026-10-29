"""Deterministic adaptive mock agent policy for offline demonstrations and testing."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from prospectpilot.agent.llm import ModelDecision


def create_adaptive_decision_handler() -> Callable[[List[Dict[str, Any]]], ModelDecision]:
    """An agent policy that dynamically inspects conversation history and adapts its next action."""

    def handler(messages: List[Dict[str, Any]]) -> ModelDecision:
        last_message = messages[-1]

        # 1. Initial Step: If last message is user prompt, initiate search via GitHub tools
        if last_message["role"] == "user":
            user_text = last_message["content"]
            if "database" in user_text.lower():
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="search_github_repos",
                    tool_arguments={"query": "topic:database"},
                )
            elif "security" in user_text.lower():
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="search_github_repos",
                    tool_arguments={"query": "topic:security"},
                )
            elif "devops" in user_text.lower() or "infrastructure" in user_text.lower() or "backend" in user_text.lower():
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="search_github_repos",
                    tool_arguments={"query": "topic:devops"},
                )
            else:
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="search_github_repos",
                    tool_arguments={"query": "topic:rare_nonexistent_keyword"},
                )

        # 2. Subsequent Step: Adapt to the Tool Observation
        if last_message["role"] == "tool":
            obs_raw = last_message["content"]
            try:
                obs_data = json.loads(obs_raw)
            except Exception:
                obs_data = {}

            # Handle search_github_repos observation
            if isinstance(obs_data, dict) and "repos" in obs_data:
                repos = obs_data.get("repos", [])
                if len(repos) > 0:
                    first_owner = repos[0].get("owner_login", "")
                    return ModelDecision(
                        is_tool_call=True,
                        tool_name="get_org_profile",
                        tool_arguments={"org": first_owner},
                    )
                else:
                    # Adaptive branch B: Search returned 0 candidates -> adapt by broadening search
                    return ModelDecision(
                        is_tool_call=True,
                        tool_name="search_github_repos",
                        tool_arguments={"query": "topic:developer-tools"},
                    )

            # Handle get_org_profile observation -> conclude research
            elif isinstance(obs_data, dict) and ("login" in obs_data or "name" in obs_data):
                org_name = obs_data.get("name") or obs_data.get("login") or "Unknown"
                return ModelDecision(
                    is_tool_call=False,
                    content=f"Evaluated candidate {org_name}. Qualified based on active GitHub developer presence.",
                )

        # Default fallback
        return ModelDecision(
            is_tool_call=False,
            content="Concluded research.",
        )

    return handler
