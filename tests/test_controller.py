"""Tests for the bounded agent controller and execution budget enforcement."""

import pytest
from prospectpilot.agent.controller import AgentController
from prospectpilot.agent.llm import MockLLMClient, ModelDecision
from prospectpilot.dataset import CompanyDataset
from prospectpilot.models import QualificationCriteria, QualificationStatus


def test_bounded_loop_max_iterations(real_demo_dataset: CompanyDataset) -> None:
    """Verifies that the controller strictly enforces maximum iterations."""
    # Endless tool caller mock
    endless_mock = MockLLMClient(
        decisions=[
            ModelDecision(
                is_tool_call=True,
                tool_name="search_companies",
                tool_arguments={"query": "tools"},
            )
            for _ in range(10)
        ]
    )

    controller = AgentController(
        llm_client=endless_mock,
        dataset=real_demo_dataset,
        max_iterations=3,
        max_tool_calls=10,
    )

    result = controller.run(goal="Find tools")
    assert result.total_iterations == 3
    assert result.status == "PARTIAL_BUDGET_EXHAUSTED"
    assert "maximum iteration budget" in result.summary.lower()


def test_bounded_loop_max_tool_calls(real_demo_dataset: CompanyDataset) -> None:
    """Verifies that the controller halts when tool call budget is exceeded."""
    loop_mock = MockLLMClient(
        decisions=[
            ModelDecision(
                is_tool_call=True,
                tool_name="search_companies",
                tool_arguments={"query": "test"},
            ),
            ModelDecision(
                is_tool_call=True,
                tool_name="search_companies",
                tool_arguments={"query": "test2"},
            ),
        ]
    )

    controller = AgentController(
        llm_client=loop_mock,
        dataset=real_demo_dataset,
        max_iterations=10,
        max_tool_calls=1,
    )

    result = controller.run(goal="Find tools")
    assert result.total_tool_calls == 1
    assert result.status == "PARTIAL_BUDGET_EXHAUSTED"


def test_llm_exception_recovery(real_demo_dataset: CompanyDataset) -> None:
    """Verifies safe failure handling when the LLM provider throws an exception."""
    class FailingLLM:
        def generate_decision(self, messages, tools):
            raise ConnectionError("Upstream LLM network timeout")

    controller = AgentController(
        llm_client=FailingLLM(),
        dataset=real_demo_dataset,
    )

    result = controller.run(goal="Find tools")
    assert result.status == "FAILED"


def test_full_agent_workflow(real_demo_dataset: CompanyDataset) -> None:
    """Verifies complete end-to-end workflow: search -> inspect profile -> finish."""
    mock = MockLLMClient(
        decisions=[
            ModelDecision(
                is_tool_call=True,
                tool_name="search_companies",
                tool_arguments={"query": "eBPF"},
            ),
            ModelDecision(
                is_tool_call=True,
                tool_name="get_company_profile",
                tool_arguments={"company_id": "comp-001"},
            ),
            ModelDecision(
                is_tool_call=False,
                content="TracePulse is an ideal target building eBPF telemetry with 42 employees and active tech blog.",
            ),
        ]
    )

    controller = AgentController(
        llm_client=mock,
        dataset=real_demo_dataset,
        max_iterations=5,
    )

    result = controller.run(
        goal="Find software companies building developer tools with active engineering presence",
        criteria=QualificationCriteria(target_focus="Developer Tools"),
    )

    assert result.status == "SUCCESS"
    assert result.total_tool_calls == 2
    assert len(result.evaluations) == 1
    eval_item = result.evaluations[0]
    assert eval_item.company_id == "comp-001"
    assert eval_item.company_name == "TracePulse"
    assert eval_item.qualification_status == QualificationStatus.QUALIFIED
    assert eval_item.score_breakdown.total_score >= 70
    assert "TracePulse" in result.summary
