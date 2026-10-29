"""Bounded ReAct agent controller with budget enforcement and deterministic qualification.

Orchestrates research using GitHub tools, deduplicates examined organizations,
enforces iteration/tool-call/time budgets, and applies deterministic Python
scoring to all examined prospects.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from prospectpilot.agent.llm import LLMClient, ModelDecision
from prospectpilot.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from prospectpilot.models import (
    AgentRunResult,
    CompanyEvaluation,
    GitHubOrgSummary,
    GitHubProspect,
    GitHubRepoSummary,
    QualificationCriteria,
    evaluate_github_prospect_deterministic,
)
from prospectpilot.tools.base import ToolRegistry, ToolResult
from prospectpilot.tracing import ExecutionTrace, TraceStep


class AgentController:
    """Orchestrates the bounded agent loop with strict Python-enforced budgets.

    Budget enforcement:
    - max_iterations: Maximum loop iterations
    - max_tool_calls: Maximum total tool invocations
    - max_time_seconds: Wall-clock time limit for the entire run
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: Optional[ToolRegistry] = None,
        github_client: Optional[Any] = None,
        max_iterations: int = 6,
        max_tool_calls: int = 10,
        max_time_seconds: float = 300.0,
        on_step: Optional[Callable[..., Any]] = None,
        github_tools_registered: bool = False,
        **kwargs: Any,
    ) -> None:
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_time_seconds = max_time_seconds
        self.on_step = on_step

        if registry is not None:
            self.registry = registry
        else:
            from prospectpilot.github_client import GitHubClient
            from prospectpilot.tools.github_tools import register_github_tools

            self.registry = ToolRegistry()
            client = github_client or GitHubClient()
            register_github_tools(self.registry, client)

    def register_tools_into(self, registry: ToolRegistry) -> None:
        """Allow external callers to register additional tools."""
        self.registry = registry

    def run(
        self,
        goal: str,
        criteria: Optional[QualificationCriteria] = None,
    ) -> AgentRunResult:
        """Executes the bounded research loop."""
        criteria = criteria or QualificationCriteria()
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        trace = ExecutionTrace(run_id=run_id, goal=goal)
        run_start_time = time.perf_counter()

        # Tracking: GitHub repos and orgs (deduplication by key)
        examined_github_repos: Dict[str, GitHubRepoSummary] = {}
        examined_github_orgs: Dict[str, GitHubOrgSummary] = {}
        github_orgs_with_releases: set[str] = set()

        # Conversation history for the LLM
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(goal, criteria)},
        ]

        openai_tools = self.registry.get_openai_tools()
        step_idx = 0
        final_summary: str = ""
        run_status: str = "SUCCESS"

        try:
            while step_idx < self.max_iterations:
                step_idx += 1
                trace.total_iterations = step_idx

                # 1. Budget Guard: Execution Time limit check
                elapsed = time.perf_counter() - run_start_time
                if elapsed >= self.max_time_seconds:
                    run_status = "PARTIAL_BUDGET_EXHAUSTED"
                    final_summary = f"Terminated due to time budget limit ({self.max_time_seconds:.1f}s exceeded)."
                    step_record = TraceStep(
                        step_index=step_idx,
                        action_type="SYSTEM_EVENT",
                        decision_summary=f"Time budget of {self.max_time_seconds}s exceeded. Finalizing partial results.",
                    )
                    trace.record_step(step_record)
                    if self.on_step:
                        self.on_step(step_record)
                    break

                # 2. Budget Guard: Tool Call count limit check
                if trace.total_tool_calls >= self.max_tool_calls:
                    run_status = "PARTIAL_BUDGET_EXHAUSTED"
                    final_summary = f"Terminated due to maximum tool call budget ({self.max_tool_calls} calls limit reached)."
                    step_record = TraceStep(
                        step_index=step_idx,
                        action_type="SYSTEM_EVENT",
                        decision_summary=f"Tool call budget of {self.max_tool_calls} reached. Halting tool loop.",
                    )
                    trace.record_step(step_record)
                    if self.on_step:
                        self.on_step(step_record)
                    break

                # 3. Request LLM decision with sandboxed execution
                model_start = time.perf_counter()
                try:
                    decision: ModelDecision = self.llm_client.generate_decision(
                        messages=messages,
                        tools=openai_tools,
                    )
                except Exception as exc:
                    run_status = "FAILED"
                    final_summary = f"LLM generation failed: {exc}"
                    step_record = TraceStep(
                        step_index=step_idx,
                        action_type="SYSTEM_EVENT",
                        decision_summary="LLM failed to return a valid response.",
                        error=str(exc),
                    )
                    trace.record_step(step_record)
                    if self.on_step:
                        self.on_step(step_record)
                    break

                model_elapsed_ms = (time.perf_counter() - model_start) * 1000

                # 4. Process Decision: Tool Call vs Final Text
                if decision.is_tool_call and decision.tool_name:
                    tool_call_id = decision.tool_call_id or f"call_{uuid.uuid4().hex[:12]}"

                    # Execute the requested tool via registry
                    tool_result: ToolResult = self.registry.execute(
                        decision.tool_name,
                        decision.tool_arguments,
                    )

                    # Deduplicate and track entities for post-loop qualification
                    self._track_tool_result(
                        tool_name=decision.tool_name,
                        tool_result=tool_result,
                        examined_github_repos=examined_github_repos,
                        examined_github_orgs=examined_github_orgs,
                        github_orgs_with_releases=github_orgs_with_releases,
                    )

                    # Record step in trace
                    step_record = TraceStep(
                        step_index=step_idx,
                        duration_ms=model_elapsed_ms + tool_result.execution_time_ms,
                        action_type="TOOL_EXECUTION",
                        tool_name=decision.tool_name,
                        tool_call_id=tool_call_id,
                        tool_arguments=decision.tool_arguments,
                        tool_success=tool_result.success,
                        tool_observation_preview=tool_result.to_observation_string()[:200],
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
                        "content": decision.content or None,
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

        except KeyboardInterrupt:
            run_status = "INTERRUPTED_BY_USER"
            final_summary = final_summary or "Execution interrupted by user (SIGINT). Preserving gathered research."
            step_record = TraceStep(
                step_index=step_idx,
                action_type="SYSTEM_EVENT",
                decision_summary="User interrupt (SIGINT) received; gracefully finalizing partial evaluations.",
            )
            trace.record_step(step_record)
            if self.on_step:
                self.on_step(step_record)

        # Check if loop hit max iterations without model voluntarily stopping
        if step_idx >= self.max_iterations and run_status == "SUCCESS" and not final_summary:
            run_status = "PARTIAL_BUDGET_EXHAUSTED"
            final_summary = "Terminated due to maximum iteration budget."

        # 5. Perform Deterministic Python Qualification on all Examined Entities
        evaluations: List[CompanyEvaluation] = []

        # Score GitHub prospects (group repos by org)
        github_prospects = self._build_github_prospects(
            examined_github_orgs=examined_github_orgs,
            examined_github_repos=examined_github_repos,
            github_orgs_with_releases=github_orgs_with_releases,
        )
        for prospect in github_prospects:
            eval_res = evaluate_github_prospect_deterministic(
                prospect=prospect,
                criteria=criteria,
            )
            evaluations.append(eval_res)

        # Sort by score descending
        evaluations.sort(key=lambda e: e.score_breakdown.total_score, reverse=True)

        trace.finalize(reason=run_status)

        return AgentRunResult(
            goal=goal,
            criteria=criteria,
            status=run_status,
            evaluations=evaluations,
            summary=final_summary or "Research completed.",
            total_iterations=trace.total_iterations,
            total_tool_calls=trace.total_tool_calls,
            total_tokens=trace.total_prompt_tokens + trace.total_completion_tokens,
            duration_seconds=trace.total_duration_seconds,
            is_synthetic_demo_run=False,
        )

    def _track_tool_result(
        self,
        tool_name: str,
        tool_result: ToolResult,
        examined_github_repos: Dict[str, GitHubRepoSummary],
        examined_github_orgs: Dict[str, GitHubOrgSummary],
        github_orgs_with_releases: set[str],
    ) -> None:
        """Track tool results for post-loop scoring. Handles parse failures gracefully."""
        if not tool_result.success or tool_result.data is None:
            return

        try:
            if tool_name == "get_repo_details" and isinstance(tool_result.data, dict):
                repo = GitHubRepoSummary.model_validate(tool_result.data)
                examined_github_repos[repo.full_name] = repo  # dedup by full_name

            elif tool_name == "get_org_profile" and isinstance(tool_result.data, dict):
                org = GitHubOrgSummary.model_validate(tool_result.data)
                examined_github_orgs[org.login] = org  # dedup by login

            elif tool_name == "search_github_repos" and isinstance(tool_result.data, dict):
                # Track repos from search results too
                for repo_data in tool_result.data.get("repos", []):
                    try:
                        repo = GitHubRepoSummary.model_validate(repo_data)
                        examined_github_repos[repo.full_name] = repo
                    except Exception:
                        pass

            elif tool_name == "get_recent_releases" and isinstance(tool_result.data, list):
                # Track which org has releases (from the tool arguments)
                if tool_result.raw_arguments.get("owner"):
                    github_orgs_with_releases.add(tool_result.raw_arguments["owner"])

        except Exception:
            # Parse failures are non-fatal — we just won't score this entity
            pass

    @staticmethod
    def _build_github_prospects(
        examined_github_orgs: Dict[str, GitHubOrgSummary],
        examined_github_repos: Dict[str, GitHubRepoSummary],
        github_orgs_with_releases: set[str],
    ) -> List[GitHubProspect]:
        """Build GitHubProspect objects by grouping repos under their orgs."""
        from datetime import datetime, timezone

        prospects: List[GitHubProspect] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        # Group repos by owner
        repos_by_owner: Dict[str, List[GitHubRepoSummary]] = {}
        for repo in examined_github_repos.values():
            repos_by_owner.setdefault(repo.owner_login, []).append(repo)

        # Build prospects for orgs that were explicitly examined
        for org_login, org_summary in examined_github_orgs.items():
            org_repos = repos_by_owner.get(org_login, [])
            languages = sorted({r.language for r in org_repos if r.language})
            total_stars = sum(r.stars for r in org_repos)
            source_urls = [org_summary.html_url] + [r.html_url for r in org_repos]

            prospects.append(GitHubProspect(
                org_login=org_login,
                org_name=org_summary.name,
                org_url=org_summary.html_url,
                org_blog=org_summary.blog,
                org_description=org_summary.description,
                org_public_repos=org_summary.public_repos,
                org_company_field=org_summary.company,
                repos=org_repos,
                top_languages=languages,
                total_stars=total_stars,
                has_recent_releases=org_login in github_orgs_with_releases,
                retrieved_at=now_iso,
                source_urls=source_urls,
                is_synthetic_demo_data=False,
            ))

        return prospects
