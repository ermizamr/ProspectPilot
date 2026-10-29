"""Deterministic adaptive mock agent policy for offline demonstrations and testing."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from prospectpilot.agent.llm import ModelDecision


def create_adaptive_decision_handler() -> Callable[[List[Dict[str, Any]]], ModelDecision]:
    """An agent policy that dynamically inspects conversation history and adapts its next action."""

    def handler(messages: List[Dict[str, Any]]) -> ModelDecision:
        last_message = messages[-1]

        # 1. Initial Step: If last message is user prompt, initiate search
        if last_message["role"] == "user":
            user_text = last_message["content"]
            if "database" in user_text.lower():
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="search_companies",
                    tool_arguments={"query": "database"},
                )
            elif "security" in user_text.lower():
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="search_companies",
                    tool_arguments={"query": "security"},
                )
            elif "devops" in user_text.lower() or "infrastructure" in user_text.lower():
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="search_companies",
                    tool_arguments={"query": "eBPF"},
                )
            else:
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="search_companies",
                    tool_arguments={"query": "rare_nonexistent_keyword"},
                )

        # 2. Subsequent Step: Adapt to the Tool Observation
        if last_message["role"] == "tool":
            obs_raw = last_message["content"]
            try:
                obs_data = json.loads(obs_raw)
            except Exception:
                obs_data = []

            # Adaptive branch A: Search returned results -> inspect the first candidate
            if isinstance(obs_data, list) and len(obs_data) > 0:
                first_id = obs_data[0]["id"]
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="get_company_profile",
                    tool_arguments={"company_id": first_id},
                )

            # Adaptive branch B: Search returned 0 candidates -> adapt by broadening search
            elif isinstance(obs_data, list) and len(obs_data) == 0:
                return ModelDecision(
                    is_tool_call=True,
                    tool_name="search_companies",
                    tool_arguments={"industry": "Developer Tools"},
                )

            # Adaptive branch C: Profile was retrieved -> conclude research
            elif isinstance(obs_data, dict) and "name" in obs_data:
                comp_name = obs_data.get("name", "Unknown")
                return ModelDecision(
                    is_tool_call=False,
                    content=f"Evaluated candidate {comp_name}. Qualified based on developer tools focus.",
                )

        # Default fallback
        return ModelDecision(
            is_tool_call=False,
            content="Concluded research.",
        )

    return handler
