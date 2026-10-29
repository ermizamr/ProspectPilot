"""Bounded ReAct agent controller with budget enforcement and deterministic qualification."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from prospectpilot.agent.llm import LLMClient, ModelDecision
from prospectpilot.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from prospectpilot.dataset import CompanyDataset
from prospectpilot.models import (
    AgentRunResult,
    CompanyEvaluation,
    CompanyProfile,
    QualificationCriteria,
    evaluate_company_deterministic,
)
from prospectpilot.tools.base import ToolRegistry, ToolResult
from prospectpilot.tools.company_tools import register_company_tools
from prospectpilot.tracing import ExecutionTrace, TraceStep


class AgentController:
    """Orchestrates the bounded agent loop with strict Python-enforced budgets."""

    def __init__(
        self,
        llm_client: LLMClient,
        dataset: Optional[CompanyDataset] = None,
        max_iterations: int = 6,
        max_tool_calls: int = 10,
        on_step: Optional[Any] = None,
    ) -> None:
        self.llm_client = llm_client
        self.dataset = dataset or CompanyDataset.load_from_file()
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.on_step = on_step

        # Setup tool registry
        self.registry = ToolRegistry()
        register_company_tools(self.registry, self.dataset)

    def run(
        self,
        goal: str,
        criteria: Optional[QualificationCriteria] = None,
    ) -> AgentRunResult:
        """Executes the bounded research loop."""
        criteria = criteria or QualificationCriteria()
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        trace = ExecutionTrace(run_id=run_id, goal=goal)

        # Profiles examined during this run
        examined_profiles: Dict[str, CompanyProfile] = {}

        # Conversation history for the LLM
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(goal, criteria)},
        ]

        openai_tools = self.registry.get_openai_tools()
        step_idx = 0
        final_summary: str = ""
        run_status: str = "SUCCESS"

        while step_idx < self.max_iterations:
            step_idx += 1
            trace.total_iterations = step_idx

            # 1. Enforce Tool Call Budget before calling LLM
            if trace.total_tool_calls >= self.max_tool_calls:
                run_status = "PARTIAL_BUDGET_EXHAUSTED"
                trace.record_step(TraceStep(
                    step_index=step_idx,
                    action_type="SYSTEM_EVENT",
                    decision_summary="Max tool calls budget exceeded; forcing termination.",
                ))
                break

            # 2. Invoke Model
            model_start = time.perf_counter()
            try:
                decision: ModelDecision = self.llm_client.generate_decision(
                    messages=messages,
                    tools=openai_tools,
                )
            except Exception as exc:
                trace.record_step(TraceStep(
                    step_index=step_idx,
                    action_type="SYSTEM_EVENT",
                    error=f"LLM generation failed: {exc}",
                ))
                run_status = "FAILED"
                break

            model_elapsed_ms = round((time.perf_counter() - model_start) * 1000, 2)

            # 3. Handle Decision: Tool Call vs Final Text
            if decision.is_tool_call and decision.tool_name:
                tool_call_id = f"call_{uuid.uuid4().hex[:6]}"

                # Execute validated tool in sandbox
                tool_result: ToolResult = self.registry.execute(
                    name=decision.tool_name,
                    raw_args=decision.tool_arguments,
                )

                # If this tool call retrieved a complete profile, track it for qualification
                if (
                    decision.tool_name == "get_company_profile"
                    and tool_result.success
                    and isinstance(tool_result.data, dict)
                ):
                    try:
                        profile = CompanyProfile.model_validate(tool_result.data)
                        examined_profiles[profile.id] = profile
                    except Exception:
                        pass

                # Record in trace
                obs_preview = tool_result.to_observation_string()
                if len(obs_preview) > 300:
                    obs_preview = obs_preview[:300] + "... [truncated in trace]"

                step_record = TraceStep(
                    step_index=step_idx,
                    duration_ms=model_elapsed_ms + tool_result.execution_time_ms,
                    action_type="TOOL_EXECUTION",
                    decision_summary=f"Called '{decision.tool_name}'",
                    tool_name=decision.tool_name,
                    tool_arguments=decision.tool_arguments,
                    tool_success=tool_result.success,
                    tool_observation_preview=obs_preview,
                    error=tool_result.error,
                    prompt_tokens=decision.prompt_tokens,
                    completion_tokens=decision.completion_tokens,
                )
                trace.record_step(step_record)
                if self.on_step:
                    self.on_step(step_record)

                # Append to messages in standard tool calling format
                messages.append({
                    "role": "assistant",
                    "content": decision.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": decision.tool_name,
                                "arguments": json.dumps(decision.tool_arguments),
                            },
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": decision.tool_name,
                    "content": tool_result.to_observation_string(),
                })

            else:
                # Model decided it has enough evidence and finished with text
                final_summary = decision.content or "Completed research."
                step_record = TraceStep(
                    step_index=step_idx,
                    duration_ms=model_elapsed_ms,
                    action_type="MODEL_DECISION",
                    decision_summary="Model finished research with final synthesis report.",
                    prompt_tokens=decision.prompt_tokens,
                    completion_tokens=decision.completion_tokens,
                )
                trace.record_step(step_record)
                if self.on_step:
                    self.on_step(step_record)
                run_status = "SUCCESS"
                break

        # Check if loop hit max iterations without model voluntarily stopping
        if step_idx >= self.max_iterations and run_status == "SUCCESS" and not final_summary:
            run_status = "PARTIAL_BUDGET_EXHAUSTED"
            final_summary = "Terminated due to maximum iteration budget."

        # 4. Perform Deterministic Python Qualification on all Examined Profiles
        evaluations: List[CompanyEvaluation] = []
        for profile in examined_profiles.values():
            eval_res = evaluate_company_deterministic(
                profile=profile,
                criteria=criteria,
            )
            evaluations.append(eval_res)

        trace.finalize(reason=run_status)

        return AgentRunResult(
            goal=goal,
            criteria=criteria,
            status=run_status,
            evaluations=evaluations,
            summary=final_summary or "Research completed.",
            total_iterations=trace.total_iterations,
            total_tool_calls=trace.total_tool_calls,
            duration_seconds=trace.total_duration_seconds,
            is_synthetic_demo_run=True,
        )
