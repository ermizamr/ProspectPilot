"""CLI entrypoint for ProspectPilot with rich live execution trace and qualification table."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from prospectpilot.agent.controller import AgentController
from prospectpilot.agent.llm import LLMClient, MockLLMClient, OpenAICompatibleClient
from prospectpilot.dataset import CompanyDataset
from prospectpilot.models import (
    AgentRunResult,
    QualificationCriteria,
    QualificationStatus,
)

# Load local .env
load_dotenv()

console = Console()


def create_demo_adaptive_mock() -> MockLLMClient:
    """Creates a deterministic adaptive mock that demonstrates genuine decision-making

    from previous observations without requiring network or API keys.
    """
    from tests.test_adaptive_behavior import create_adaptive_decision_handler
    return MockLLMClient(dynamic_handler=create_adaptive_decision_handler())


def run_cli(
    goal: str,
    target_focus: str = "Developer Tools",
    mock_mode: bool = False,
    max_iterations: int = 6,
    as_json: bool = False,
) -> int:
    """Executes the agent run and prints live traces and structured summaries."""

    criteria = QualificationCriteria(target_focus=target_focus)

    if not as_json:
        console.print(
            Panel.fit(
                "[bold cyan]ProspectPilot[/bold cyan] - Agentic Research & Qualification System\n"
                f"[dim]Goal:[/dim] [white]{goal}[/white]\n"
                f"[dim]Mode:[/dim] [{'yellow' if mock_mode else 'green'}]{'Deterministic Adaptive Mock (Offline)' if mock_mode else 'Live Provider'}[/{'yellow' if mock_mode else 'green'}] | "
                f"[dim]Focus:[/dim] {criteria.target_focus}",
                border_style="cyan",
            )
        )

    # Instantiate LLM client
    llm: LLMClient
    if mock_mode:
        llm = create_demo_adaptive_mock()
    else:
        try:
            llm = OpenAICompatibleClient()
        except ValueError as err:
            console.print(f"[bold red]Configuration Error:[/bold red] {err}")
            console.print("\n[yellow]Tip:[/yellow] Run with [bold]--mock[/bold] for an immediate offline demo with no API key needed.")
            return 1

    dataset = CompanyDataset.load_from_file()

    if not as_json:
        console.print("\n[bold]Starting Bounded Research Loop...[/bold]\n")

    def print_trace_step(step: Any) -> None:
        if as_json:
            return
        if step.action_type == "TOOL_EXECUTION":
            console.print(
                f"  [bold blue][Step {step.step_index}][/bold blue] Tool selected: [bold yellow]{step.tool_name}[/bold yellow] | Args: [cyan]{step.tool_arguments}[/cyan]"
            )
            if step.tool_success:
                obs_summary = step.tool_observation_preview.replace("\n", " ")
                if len(obs_summary) > 120:
                    obs_summary = obs_summary[:120] + "..."
                console.print(f"    [dim]↳ Observation ({step.duration_ms:.1f}ms):[/dim] {obs_summary}")
            else:
                console.print(f"    [bold red]↳ Error ({step.duration_ms:.1f}ms):[/bold red] {step.error}")
        elif step.action_type == "MODEL_DECISION":
            console.print(
                f"  [bold green][Step {step.step_index}][/bold green] Decision: [italic]{step.decision_summary}[/italic]"
            )
        elif step.action_type == "SYSTEM_EVENT":
            console.print(
                f"  [bold magenta][Step {step.step_index}][/bold magenta] System: {step.decision_summary or step.error}"
            )

    controller = AgentController(
        llm_client=llm,
        dataset=dataset,
        max_iterations=max_iterations,
        on_step=print_trace_step if not as_json else None,
    )

    start_time = time.perf_counter()
    result: AgentRunResult = controller.run(goal=goal, criteria=criteria)
    elapsed = time.perf_counter() - start_time

    if as_json:
        print(result.model_dump_json(indent=2))
        return 0

    # Render Step-by-Step Execution Summary
    console.print(f"\n[bold green]Loop Completed in {elapsed:.2f}s[/bold green] (Status: [bold]{result.status}[/bold])")
    console.print(f"Iterations: {result.total_iterations} | Tool Calls: {result.total_tool_calls}\n")

    # Render Evaluated Companies Table
    if result.evaluations:
        table = Table(title="Qualified Prospect Candidates", border_style="blue")
        table.add_column("Company / Domain", style="cyan", no_wrap=False)
        table.add_column("Status", justify="center")
        table.add_column("Score (0-100)", justify="center")
        table.add_column("Key Evidence", style="dim")
        table.add_column("Missing / Uncertainty", style="magenta")
        table.add_column("Recommended Action", style="bold")

        for ev in result.evaluations:
            status_color = {
                QualificationStatus.QUALIFIED: "green",
                QualificationStatus.DISQUALIFIED: "red",
                QualificationStatus.INSUFFICIENT_DATA: "yellow",
            }[ev.qualification_status]

            status_text = f"[{status_color}]{ev.qualification_status.value}[/{status_color}]"
            score_text = f"{ev.score_breakdown.total_score} (Dev:{ev.score_breakdown.developer_alignment_score}, Scale:{ev.score_breakdown.scale_and_stage_score})"

            evidence_bullets = "\n".join(f"• {e.field_name}: {e.observed_value}" for e in ev.evidence[:2])
            missing_bullets = "\n".join(f"• {m}" for m in ev.missing_information) if ev.missing_information else "None"

            table.add_row(
                f"{ev.company_name}\n[dim]{ev.domain}[/dim]",
                status_text,
                score_text,
                evidence_bullets,
                missing_bullets,
                ev.recommended_action,
            )

        console.print(table)
    else:
        console.print("[yellow]No company profiles were inspected during this run.[/yellow]")

    # Print Final Synthesis Note
    console.print(
        Panel(
            f"[bold]Agent Synthesis Summary:[/bold]\n{result.summary}",
            title="Executive Summary",
            border_style="green",
        )
    )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProspectPilot: Agentic Python Research & Qualification System"
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Find database developer tools and assess qualification for outreach.",
        help="Outreach research goal for the agent.",
    )
    parser.add_argument(
        "--focus",
        type=str,
        default="Developer Tools",
        help="Target vertical or keyword criteria.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline using deterministic adaptive mock (no API key required).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="Maximum loop iterations.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured Pydantic report as JSON.",
    )

    args = parser.parse_args()
    code = run_cli(
        goal=args.goal,
        target_focus=args.focus,
        mock_mode=args.mock,
        max_iterations=args.max_iterations,
        as_json=args.json,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
