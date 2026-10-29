"""CLI entrypoint for ProspectPilot with simple, delightful user experience."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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

# Curated outreach research presets
PRESETS: Dict[str, Dict[str, str]] = {
    "backend": {
        "goal": "Find real developer-tool companies whose products are relevant to backend engineers, and identify evidence-based outreach angles.",
        "focus": "Backend Developer Tools",
        "description": "Backend infrastructure, SDKs, distributed systems & APM tools",
    },
    "devops": {
        "goal": "Find cloud-native infrastructure, CI/CD automation, and container developer tools with public codebases.",
        "focus": "DevOps & Cloud-Native Tools",
        "description": "Kubernetes, Docker, CI/CD, Terraform & platform engineering",
    },
    "security": {
        "goal": "Find application security testing (SAST/DAST), secret scanning, and software supply chain security developer tools.",
        "focus": "DevSecOps & AppSec Tools",
        "description": "Security scanning, vulnerability management & secure coding tools",
    },
    "database": {
        "goal": "Find database DevOps, schema migration safety, and ORM developer tooling companies with active repositories.",
        "focus": "Database DevOps & Reliability",
        "description": "PostgreSQL/MySQL tooling, schema CI/CD, connection pooling & ORMs",
    },
    "api": {
        "goal": "Find API testing, OpenAPI contract-first mocking, and microservices developer tool companies.",
        "focus": "API Quality & Testing Tools",
        "description": "API mocking, contract testing, OpenAPI 3.1 & gRPC tooling",
    },
}


def is_live_llm_configured() -> bool:
    """Check if any supported LLM provider API key or base URL is present in env."""
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENAI_BASE_URL")
    )


def create_demo_adaptive_mock() -> MockLLMClient:
    """Creates a deterministic adaptive mock that demonstrates genuine decision-making

    from previous observations without requiring network or API keys.
    """
    from prospectpilot.agent.adaptive_mock import create_adaptive_decision_handler
    return MockLLMClient(dynamic_handler=create_adaptive_decision_handler())


def _export_csv(result: AgentRunResult, path: Path) -> None:
    """Export evaluation results to CSV with evidence and source URLs."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "company_id", "company_name", "domain", "qualification_status",
            "total_score", "dev_alignment_score", "scale_score", "signal_score",
            "evidence_summary", "missing_information", "recommended_action",
            "source_urls", "is_synthetic",
        ])
        for ev in result.evaluations:
            evidence_summary = "; ".join(
                f"{e.field_name}={e.observed_value}" for e in ev.evidence[:5]
            )
            source_urls = "; ".join(
                e.source_url for e in ev.evidence if e.source_url
            )
            missing = "; ".join(ev.missing_information)
            writer.writerow([
                ev.company_id,
                ev.company_name,
                ev.domain,
                ev.qualification_status.value,
                ev.score_breakdown.total_score,
                ev.score_breakdown.developer_alignment_score,
                ev.score_breakdown.scale_and_stage_score,
                ev.score_breakdown.engagement_signals_score,
                evidence_summary,
                missing,
                ev.recommended_action,
                source_urls,
                ev.is_synthetic_demo_data,
            ])


def _export_json_file(result: AgentRunResult, path: Path) -> None:
    """Export structured report to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))


def prompt_interactive_preset() -> Tuple[str, str]:
    """Simple, elegant interactive prompt for choosing outreach goals."""
    from rich.prompt import Prompt

    console.print("\n[bold cyan]Select an Outreach Research Vertical:[/bold cyan]\n")
    preset_keys = list(PRESETS.keys())
    for idx, key in enumerate(preset_keys, 1):
        info = PRESETS[key]
        console.print(f"  [bold cyan][{idx}][/bold cyan] [bold]{key.capitalize()}[/bold] — [dim]{info['description']}[/dim]")
    console.print(f"  [bold cyan][{len(preset_keys) + 1}][/bold cyan] [bold]Custom Goal[/bold] — [dim]Enter your own custom criteria[/dim]\n")

    choice = Prompt.ask("Choose an option", default="1", choices=[str(i) for i in range(1, len(preset_keys) + 2)])
    choice_idx = int(choice) - 1
    if choice_idx < len(preset_keys):
        selected_key = preset_keys[choice_idx]
        return PRESETS[selected_key]["goal"], PRESETS[selected_key]["focus"]
    else:
        custom_goal = Prompt.ask("\nEnter your custom research goal")
        custom_focus = Prompt.ask("Enter primary target focus", default="Developer Tools")
        return custom_goal, custom_focus


def run_cli(
    goal: str,
    target_focus: str = "Developer Tools",
    mock_mode: bool = False,
    max_iterations: int = 10,
    max_tool_calls: int = 20,
    max_time_seconds: float = 300.0,
    as_json: bool = False,
    export_path: Optional[str] = None,
) -> int:
    """Executes the agent run and prints live traces and structured summaries."""

    criteria = QualificationCriteria(target_focus=target_focus)

    if not mock_mode:
        if not is_live_llm_configured():
            console.print("[bold red]Configuration Error:[/bold red] No live LLM provider API key detected.\n")
            console.print("ProspectPilot conducts live research using the GitHub REST API and an LLM.")
            console.print("Set an API key in your environment or .env file (see .env.example):\n")
            console.print("  • export GROQ_API_KEY=\"gsk_...\"      (Free fast tier: console.groq.com)")
            console.print("  • export OPENAI_API_KEY=\"sk-...\"     (OpenAI gpt-4o-mini)")
            console.print("  • export GEMINI_API_KEY=\"AIzaSy...\"  (Google AI Studio)")
            console.print("  • export OPENAI_BASE_URL=\"http://localhost:11434/v1\" (Local Ollama)")
            console.print("\n[dim]To run offline unit tests with synthetic fixtures, pass: prospectpilot --mock[/dim]")
            return 1

    if not as_json:
        mode_label = "Offline Mock (Synthetic Fixtures)" if mock_mode else "Live GitHub REST API Research"
        mode_color = "yellow" if mock_mode else "green"
        console.print(
            Panel.fit(
                "[bold cyan]ProspectPilot[/bold cyan] — Agentic Developer Outreach Research\n"
                f"[dim]Goal:[/dim] [white]{goal}[/white]\n"
                f"[dim]Mode:[/dim] [{mode_color}]{mode_label}[/{mode_color}] | "
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
            return 1

    controller: AgentController

    if mock_mode:
        # Synthetic mode with demo dataset (for tests only)
        dataset = CompanyDataset.load_from_file()
        controller = AgentController(
            llm_client=llm,
            dataset=dataset,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            max_time_seconds=max_time_seconds,
            on_step=_make_trace_printer(as_json),
        )
    else:
        # Live mode with GitHub tools
        from prospectpilot.github_client import GitHubClient
        from prospectpilot.tools.github_tools import register_github_tools

        github_token = os.getenv("GITHUB_TOKEN")
        if not as_json:
            if github_token:
                console.print("[green]✓[/green] GITHUB_TOKEN active (5,000 req/hr authenticated rate limit)")
            else:
                console.print(
                    "[yellow]⚠[/yellow] No GITHUB_TOKEN set — running unauthenticated (60 req/hr limit). "
                    "Set GITHUB_TOKEN for 5,000 req/hr."
                )

        github_client = GitHubClient()
        controller = AgentController(
            llm_client=llm,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            max_time_seconds=max_time_seconds,
            on_step=_make_trace_printer(as_json),
            github_tools_registered=True,
        )
        register_github_tools(controller.registry, github_client)

    if not as_json:
        console.print("[bold]Executing Bounded Research Loop...[/bold]\n")

    start_time = time.perf_counter()
    result: AgentRunResult = controller.run(goal=goal, criteria=criteria)
    elapsed = time.perf_counter() - start_time

    if as_json:
        print(result.model_dump_json(indent=2))
    else:
        _print_results(result, elapsed)

    # Handle file export if requested
    if export_path:
        out_path = Path(export_path)
        if out_path.suffix.lower() == ".json":
            _export_json_file(result, out_path)
            if not as_json:
                console.print(f"\n[green]✓[/green] JSON report exported to: [bold]{out_path}[/bold]")
        else:
            _export_csv(result, out_path)
            if not as_json:
                console.print(f"\n[green]✓[/green] CSV report exported to: [bold]{out_path}[/bold]")

    return 0


def _make_trace_printer(as_json: bool) -> Optional[Any]:
    """Returns a rich, formatted trace step callback or None if JSON mode."""
    if as_json:
        return None

    tool_labels = {
        "search_github_repos": "🔎 [bold cyan]Search Repos[/bold cyan]",
        "get_org_profile": "🏢 [bold blue]Inspect Org[/bold blue]",
        "get_repo_details": "📦 [bold magenta]Inspect Repo[/bold magenta]",
        "get_recent_releases": "🏷️  [bold yellow]Check Releases[/bold yellow]",
        "search_companies": "🔎 [bold cyan]Search Companies[/bold cyan]",
        "get_company_profile": "🏢 [bold blue]Inspect Profile[/bold blue]",
    }

    def print_trace_step(step: Any) -> None:
        if step.action_type == "TOOL_EXECUTION":
            action_label = tool_labels.get(step.tool_name, f"🔧 [bold]{step.tool_name}[/bold]")
            arg_str = ", ".join(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in step.tool_arguments.items())
            if len(arg_str) > 75:
                arg_str = arg_str[:75] + "..."

            console.print(f"  [dim][Step {step.step_index}][/dim] {action_label} [dim]({arg_str})[/dim]")
            if step.tool_success:
                obs_summary = (step.tool_observation_preview or "").replace("\n", " ").strip()
                if len(obs_summary) > 115:
                    obs_summary = obs_summary[:115] + "..."
                console.print(f"         [dim green]↳[/dim green] [dim]{obs_summary}[/dim]")
            else:
                console.print(f"         [bold red]↳ Error ({step.duration_ms:.1f}ms):[/bold red] {step.error}")
        elif step.action_type == "MODEL_DECISION":
            console.print(f"  [dim][Step {step.step_index}][/dim] 💡 [bold green]Model Decision[/bold green] [italic]{step.decision_summary}[/italic]")
        elif step.action_type == "SYSTEM_EVENT":
            console.print(f"  [dim][Step {step.step_index}][/dim] ⚙️  [bold magenta]System[/bold magenta]: {step.decision_summary or step.error}")

    return print_trace_step


def _print_results(result: AgentRunResult, elapsed: float) -> None:
    """Render the structured results table and summary."""
    status_style = "green" if result.status == "SUCCESS" else "yellow"
    console.print(
        f"\n[bold {status_style}]Loop Completed in {elapsed:.2f}s[/bold {status_style}] "
        f"(Status: [bold]{result.status}[/bold])"
    )
    console.print(
        f"Iterations: {result.total_iterations} | "
        f"Tool Calls: {result.total_tool_calls} | "
        f"Tokens: {result.total_tokens} | "
        f"Data Source: {'Synthetic Demo' if result.is_synthetic_demo_run else 'Live GitHub REST API'}\n"
    )

    if result.evaluations:
        table = Table(title="Qualified Prospect Report", border_style="blue")
        table.add_column("Organization / Domain", style="cyan", no_wrap=False)
        table.add_column("Qualification", justify="center")
        table.add_column("Score", justify="center")
        table.add_column("Key Evidence", style="dim", max_width=48)
        table.add_column("Missing / Unknown", style="magenta", max_width=38)
        table.add_column("Recommended Outreach Angle", style="bold", max_width=48)

        for ev in result.evaluations:
            status_text = {
                QualificationStatus.QUALIFIED: "[bold green]✔ QUALIFIED[/bold green]",
                QualificationStatus.DISQUALIFIED: "[bold red]✘ DISQUALIFIED[/bold red]",
                QualificationStatus.INSUFFICIENT_DATA: "[bold yellow]⚠ UNCERTAIN[/bold yellow]",
            }[ev.qualification_status]

            score_text = (
                f"{ev.score_breakdown.total_score}/100\n"
                f"[dim](D:{ev.score_breakdown.developer_alignment_score} "
                f"S:{ev.score_breakdown.scale_and_stage_score} "
                f"E:{ev.score_breakdown.engagement_signals_score})[/dim]"
            )

            evidence_bullets = "\n".join(
                f"• {e.field_name}: {e.observed_value}" for e in ev.evidence[:3]
            )
            missing_bullets = (
                "\n".join(f"• {m}" for m in ev.missing_information[:3])
                if ev.missing_information
                else "None"
            )

            data_tag = "" if not ev.is_synthetic_demo_data else "\n[dim](synthetic)[/dim]"

            table.add_row(
                f"{ev.company_name}\n[dim]{ev.domain}[/dim]{data_tag}",
                status_text,
                score_text,
                evidence_bullets,
                missing_bullets,
                ev.recommended_action,
            )

        console.print(table)
    else:
        console.print("[yellow]No prospects were examined during this run.[/yellow]")

    console.print(
        Panel(
            f"[bold]Agent Synthesis Summary:[/bold]\n{result.summary}",
            title="Executive Summary",
            border_style="green",
        )
    )

    # Next steps tip
    console.print("\n[dim]💡 Quick tips:[/dim]")
    console.print("  [dim]• Export findings to CSV: [bold]prospectpilot -o prospects.csv[/bold][/dim]")
    console.print("  [dim]• Try a security preset:  [bold]prospectpilot -p security[/bold][/dim]")
    console.print("  [dim]• Interactive vertical:   [bold]prospectpilot -i[/bold][/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prospectpilot",
        description="ProspectPilot: Agentic Developer-Prospect Research System",
        epilog="Examples:\n"
               "  prospectpilot                      # Instant run with smart defaults\n"
               "  prospectpilot -p devops            # Run DevOps & Cloud-Native preset\n"
               "  prospectpilot -p security -o out.csv  # Security research + export CSV\n"
               "  prospectpilot -i                   # Interactive vertical picker\n"
               "  prospectpilot --mock               # Force offline demo mode\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=list(PRESETS.keys()),
        default=None,
        help="Quick outreach preset: backend, devops, security, database, api.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive vertical & goal selector.",
    )
    parser.add_argument(
        "--goal", "-g",
        type=str,
        default=None,
        help="Custom research goal for the agent.",
    )
    parser.add_argument(
        "--focus",
        type=str,
        default=None,
        help="Target vertical or keyword criteria (default: 'Developer Tools').",
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Run offline using synthetic fixtures (deterministic testing only).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Export results to file (.csv or .json).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Export results to CSV file at the given path (alias for -o path.csv).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured Pydantic report as JSON to stdout.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum loop iterations (default: 10).",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=20,
        help="Maximum total tool calls (default: 20).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=300.0,
        help="Maximum wall-clock execution time in seconds (default: 300).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed debug/network transport logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Resolve goal and focus:
    # 1. Interactive picker if -i
    # 2. Preset if -p
    # 3. Explicit --goal / --focus
    # 4. Default backend preset
    goal: str
    focus: str

    if args.interactive:
        goal, focus = prompt_interactive_preset()
    elif args.preset:
        preset_info = PRESETS[args.preset]
        goal = args.goal or preset_info["goal"]
        focus = args.focus or preset_info["focus"]
    else:
        default_preset = PRESETS["backend"]
        goal = args.goal or default_preset["goal"]
        focus = args.focus or args.focus or default_preset["focus"]

    export_path = args.output or args.csv

    code = run_cli(
        goal=goal,
        target_focus=focus,
        mock_mode=args.mock,
        max_iterations=args.max_iterations,
        max_tool_calls=args.max_tool_calls,
        max_time_seconds=args.max_time,
        as_json=args.json,
        export_path=export_path,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
