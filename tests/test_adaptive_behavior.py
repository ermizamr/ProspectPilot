"""Test proving genuinely adaptive agent control based on previous tool observations.

Demonstrates:
1. Tool observations returned in Step N directly dictate the tool and arguments selected in Step N+1.
2. An empty search observation causes the agent to adapt its strategy (broadening search)
   rather than blindly querying a nonexistent profile.
"""

import json
from typing import Any, Dict, List
import pytest

from prospectpilot.agent.controller import AgentController
from prospectpilot.agent.llm import MockLLMClient, ModelDecision
from prospectpilot.dataset import CompanyDataset


def create_adaptive_decision_handler():
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


def test_adaptive_branching_database_vs_security(real_demo_dataset: CompanyDataset) -> None:
    """Verifies that different goals lead to different search queries,

    which return different candidate IDs, which dynamically trigger different
    subsequent profile retrieval calls.
    """
    # Scenario A: Database search -> finds SchemaCraft (comp-002)
    mock_db = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    controller_db = AgentController(llm_client=mock_db, dataset=real_demo_dataset, max_iterations=5)
    result_db = controller_db.run(goal="Find database tools")

    assert result_db.status == "SUCCESS"
    assert len(result_db.evaluations) == 1
    assert result_db.evaluations[0].company_id == "comp-002"
    assert result_db.evaluations[0].company_name == "SchemaCraft"

    # Scenario B: Security search -> finds DevSecGate (comp-008)
    mock_sec = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    controller_sec = AgentController(llm_client=mock_sec, dataset=real_demo_dataset, max_iterations=5)
    result_sec = controller_sec.run(goal="Find security tools")

    assert result_sec.status == "SUCCESS"
    assert len(result_sec.evaluations) == 1
    assert result_sec.evaluations[0].company_id == "comp-008"
    assert result_sec.evaluations[0].company_name == "DevSecGate"


def test_adaptive_recovery_from_empty_observation(real_demo_dataset: CompanyDataset) -> None:
    """Verifies that when an observation returns 0 matches, the agent policy

    observes the empty list and adapts by broadening its query to the entire industry,
    rather than failing or requesting an invalid company.
    """
    mock_adaptive = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    controller = AgentController(llm_client=mock_adaptive, dataset=real_demo_dataset, max_iterations=6)

    # Goal triggers initial search with 'rare_nonexistent_keyword'
    result = controller.run(goal="Find companies matching quantum computing")

    # The agent performed:
    # 1. search(rare_nonexistent_keyword) -> returns []
    # 2. observed [] -> dynamically adapted to search(industry="Developer Tools") -> returns candidates
    # 3. observed candidate list -> retrieved profile for first candidate
    # 4. concluded research
    assert result.total_tool_calls >= 3
    assert result.status == "SUCCESS"
    assert len(result.evaluations) >= 1
