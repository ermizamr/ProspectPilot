"""Execution tracing, step timing, and audit trail for agent runs."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    """Detailed record of a single step in the agent control loop."""
    step_index: int
    timestamp: float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    action_type: str  # "MODEL_DECISION", "TOOL_EXECUTION", "SYSTEM_EVENT"
    decision_summary: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    tool_success: Optional[bool] = None
    tool_observation_preview: Optional[str] = None
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ExecutionTrace(BaseModel):
    """Complete audit trail for an entire agent run."""
    run_id: str
    goal: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration_seconds: float = 0.0
    steps: List[TraceStep] = Field(default_factory=list)
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    termination_reason: str = "IN_PROGRESS"

    def record_step(self, step: TraceStep) -> None:
        """Appends a validated trace step and updates aggregate metrics."""
        self.steps.append(step)
        if step.tool_name:
            self.total_tool_calls += 1
        self.total_prompt_tokens += step.prompt_tokens
        self.total_completion_tokens += step.completion_tokens

    def finalize(self, reason: str) -> None:
        """Marks trace as completed."""
        self.end_time = time.time()
        self.total_duration_seconds = round(self.end_time - self.start_time, 2)
        self.termination_reason = reason

    def to_summary_dict(self) -> Dict[str, Any]:
        """Provides a clean summary dictionary suitable for JSON logging or reporting."""
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "duration_seconds": self.total_duration_seconds,
            "total_iterations": self.total_iterations,
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "termination_reason": self.termination_reason,
            "steps": [s.model_dump() for s in self.steps],
        }
