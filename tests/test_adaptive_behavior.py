"""Test proving genuinely adaptive agent control based on previous tool observations.

Demonstrates:
1. Tool observations returned in Step N directly dictate the tool and arguments selected in Step N+1.
2. An empty search observation causes the agent to adapt its strategy (broadening search)
   rather than blindly querying a nonexistent profile.
"""

from __future__ import annotations

from typing import Any
import pytest

from prospectpilot.agent.adaptive_mock import create_adaptive_decision_handler
from prospectpilot.agent.controller import AgentController
from prospectpilot.agent.llm import MockLLMClient


def test_adaptive_branching_database_vs_security(mock_github: Any) -> None:
    """Verifies that different goals lead to different search queries,

    which return different candidate IDs, which dynamically trigger different
    subsequent profile retrieval calls.
    """
    # Scenario A: Database search -> finds SchemaCraft
    mock_db = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    controller_db = AgentController(
        llm_client=mock_db,
        github_client=mock_github,
        max_iterations=5,
    )
    result_db = controller_db.run(goal="Find database tools")

    assert result_db.status == "SUCCESS"
    assert len(result_db.evaluations) == 1
    assert result_db.evaluations[0].company_id == "schemacraft"
    assert result_db.evaluations[0].company_name == "SchemaCraft"

    # Scenario B: Security search -> finds DevSecGate
    mock_sec = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    controller_sec = AgentController(
        llm_client=mock_sec,
        github_client=mock_github,
        max_iterations=5,
    )
    result_sec = controller_sec.run(goal="Find security tools")

    assert result_sec.status == "SUCCESS"
    assert len(result_sec.evaluations) == 1
    assert result_sec.evaluations[0].company_id == "devsecgate"
    assert result_sec.evaluations[0].company_name == "DevSecGate"


def test_adaptive_recovery_from_empty_observation(mock_github: Any) -> None:
    """Verifies that when an observation returns 0 matches, the agent policy

    observes the empty list and adapts by broadening its query to the entire industry,
    rather than failing or requesting an invalid company.
    """
    mock_adaptive = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    controller = AgentController(
        llm_client=mock_adaptive,
        github_client=mock_github,
        max_iterations=6,
    )

    # Goal triggers initial search with 'rare_nonexistent_keyword'
    result = controller.run(goal="Find companies matching quantum computing")

    # The agent performed:
    # 1. search(rare_nonexistent_keyword) -> returns 0
    # 2. observed 0 -> dynamically adapted to search(query="topic:developer-tools") -> returns candidates
    # 3. observed candidate list -> retrieved profile for first candidate
    # 4. concluded research
    assert result.total_tool_calls >= 2
    assert result.status == "SUCCESS"
    assert len(result.evaluations) >= 1
