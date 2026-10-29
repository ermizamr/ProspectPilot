"""Tests for the bounded agent controller and execution budget enforcement."""

from typing import Any
import pytest
from prospectpilot.agent.controller import AgentController
from prospectpilot.agent.llm import MockLLMClient, ModelDecision
from prospectpilot.models import QualificationCriteria, QualificationStatus


def test_bounded_loop_max_iterations(mock_github: Any) -> None:
    """Verifies that the controller strictly enforces maximum iterations."""
    endless_mock = MockLLMClient(
        decisions=[
            ModelDecision(
                is_tool_call=True,
                tool_name="search_github_repos",
                tool_arguments={"query": "topic:devops"},
            )
            for _ in range(10)
        ]
    )

    controller = AgentController(
        llm_client=endless_mock,
        github_client=mock_github,
        max_iterations=3,
        max_tool_calls=10,
    )

    result = controller.run(goal="Find tools")
    assert result.total_iterations == 3
    assert result.status == "PARTIAL_BUDGET_EXHAUSTED"
    assert "maximum iteration budget" in result.summary.lower()


def test_bounded_loop_max_tool_calls(mock_github: Any) -> None:
    """Verifies that the controller halts when tool call budget is exceeded."""
    loop_mock = MockLLMClient(
        decisions=[
            ModelDecision(
                is_tool_call=True,
                tool_name="search_github_repos",
                tool_arguments={"query": "topic:devops"},
            ),
            ModelDecision(
                is_tool_call=True,
                tool_name="search_github_repos",
                tool_arguments={"query": "topic:security"},
            ),
        ]
    )

    controller = AgentController(
        llm_client=loop_mock,
        github_client=mock_github,
        max_iterations=10,
        max_tool_calls=1,
    )

    result = controller.run(goal="Find tools")
    assert result.total_tool_calls == 1
    assert result.status == "PARTIAL_BUDGET_EXHAUSTED"


def test_llm_exception_recovery(mock_github: Any) -> None:
    """Verifies safe failure handling when the LLM provider throws an exception."""
    class FailingLLM:
        def generate_decision(self, messages, tools):
            raise ConnectionError("Upstream LLM network timeout")

    controller = AgentController(
        llm_client=FailingLLM(),
        github_client=mock_github,
    )

    result = controller.run(goal="Find tools")
    assert result.status == "FAILED"
    assert "LLM generation failed: Upstream LLM network timeout" in result.summary


def test_tool_call_id_preservation(mock_github: Any) -> None:
    """Verifies that the provider-assigned tool_call_id is preserved in conversation messages."""
    recorded_messages = []

    class InspectingMock(MockLLMClient):
        def generate_decision(self, messages, tools):
            recorded_messages.append(list(messages))
            return super().generate_decision(messages, tools)

    mock = InspectingMock(
        decisions=[
            ModelDecision(
                is_tool_call=True,
                tool_call_id="call_provider_specific_999",
                tool_name="search_github_repos",
                tool_arguments={"query": "topic:devops"},
            ),
            ModelDecision(
                is_tool_call=False,
                content="Finished research.",
            ),
        ]
    )

    controller = AgentController(
        llm_client=mock,
        github_client=mock_github,
        max_iterations=3,
    )

    result = controller.run(goal="Find tools")
    assert result.status == "SUCCESS"

    # Inspect the messages sent in the second call
    second_call_messages = recorded_messages[1]
    assistant_msg = [m for m in second_call_messages if m["role"] == "assistant"][0]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]

    assert assistant_msg["tool_calls"][0]["id"] == "call_provider_specific_999"
    assert tool_msg["tool_call_id"] == "call_provider_specific_999"


def test_malformed_tool_call_arguments_recovery(mock_github: Any) -> None:
    """Verifies that malformed or invalid arguments returned by model are caught by schema

    validation and fed back as tool error observation without crashing the agent.
    """
    recorded_messages = []

    class InspectingMock(MockLLMClient):
        def generate_decision(self, messages, tools):
            recorded_messages.append(list(messages))
            return super().generate_decision(messages, tools)

    mock = InspectingMock(
        decisions=[
            # Model emits malformed argument structure for get_org_profile (missing required owner)
            ModelDecision(
                is_tool_call=True,
                tool_call_id="call_malformed_1",
                tool_name="get_org_profile",
                tool_arguments={"invalid_param": 123},
            ),
            # Model receives validation error observation and corrects itself
            ModelDecision(
                is_tool_call=False,
                content="Self-corrected after observing validation error.",
            ),
        ]
    )

    controller = AgentController(
        llm_client=mock,
        github_client=mock_github,
        max_iterations=3,
    )

    result = controller.run(goal="Find company")
    assert result.status == "SUCCESS"

    second_call_messages = recorded_messages[1]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "TOOL ERROR: Argument validation failed" in tool_msg["content"]


def test_full_agent_workflow(mock_github: Any) -> None:
    """Verifies complete end-to-end workflow: search -> inspect org -> finish."""
    mock = MockLLMClient(
        decisions=[
            ModelDecision(
                is_tool_call=True,
                tool_name="search_github_repos",
                tool_arguments={"query": "topic:devops"},
            ),
            ModelDecision(
                is_tool_call=True,
                tool_name="get_org_profile",
                tool_arguments={"org": "hashicorp"},
            ),
            ModelDecision(
                is_tool_call=False,
                content="HashiCorp is an ideal target building Terraform infrastructure tools with 42,000 stars.",
            ),
        ]
    )

    controller = AgentController(
        llm_client=mock,
        github_client=mock_github,
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
    assert eval_item.company_id == "hashicorp"
    assert eval_item.company_name == "HashiCorp"
    assert eval_item.qualification_status == QualificationStatus.QUALIFIED
    assert eval_item.score_breakdown.total_score >= 70
    assert "HashiCorp" in result.summary


def test_keyboard_interrupt_recovery(mock_github: Any) -> None:
    """Verifies that SIGINT (KeyboardInterrupt) preserves partial progress and doesn't crash."""
    class InterruptingMock:
        def __init__(self):
            self.calls = 0

        def generate_decision(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                    return ModelDecision(
                        is_tool_call=True,
                        tool_name="get_org_profile",
                        tool_arguments={"org": "hashicorp"},
                    )
            raise KeyboardInterrupt()

    controller = AgentController(
        llm_client=InterruptingMock(),
        github_client=mock_github,
        max_iterations=5,
    )

    result = controller.run(goal="Find tools")
    assert result.status == "INTERRUPTED_BY_USER"
    assert "interrupted by user" in result.summary.lower()
    # Verified that the org examined before interrupt was evaluated and saved!
    assert len(result.evaluations) == 1
    assert result.evaluations[0].company_id == "hashicorp"
