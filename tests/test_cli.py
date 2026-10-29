"""Tests for CLI options, presets, smart defaults, and file export."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest

from prospectpilot.agent.adaptive_mock import create_adaptive_decision_handler
from prospectpilot.agent.llm import MockLLMClient
from prospectpilot.cli import PRESETS, run_cli


def test_cli_presets_exist() -> None:
    """Verifies that all standard research presets are configured."""
    assert "backend" in PRESETS
    assert "devops" in PRESETS
    assert "security" in PRESETS
    assert "database" in PRESETS
    assert "api" in PRESETS
    for p in PRESETS.values():
        assert p["goal"]
        assert p["focus"]


def test_cli_default_backend(mock_github: Any) -> None:
    """Verifies run_cli with default goal completes successfully."""
    llm = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    code = run_cli(
        goal=PRESETS["backend"]["goal"],
        target_focus=PRESETS["backend"]["focus"],
        llm_client=llm,
        github_client=mock_github,
        max_iterations=5,
    )
    assert code == 0


def test_cli_security_preset(mock_github: Any) -> None:
    """Verifies security preset runs and scores prospects."""
    llm = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    code = run_cli(
        goal=PRESETS["security"]["goal"],
        target_focus=PRESETS["security"]["focus"],
        llm_client=llm,
        github_client=mock_github,
        max_iterations=5,
    )
    assert code == 0


def test_cli_csv_file_export(tmp_path: Path, mock_github: Any) -> None:
    """Verifies CSV file export writes expected headers and records."""
    csv_out = tmp_path / "prospects.csv"
    llm = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    code = run_cli(
        goal=PRESETS["backend"]["goal"],
        target_focus=PRESETS["backend"]["focus"],
        llm_client=llm,
        github_client=mock_github,
        export_path=str(csv_out),
    )
    assert code == 0
    assert csv_out.exists()
    content = csv_out.read_text(encoding="utf-8")
    assert "company_id,company_name,domain" in content
    assert "hashicorp" in content.lower()


def test_cli_json_file_export(tmp_path: Path, mock_github: Any) -> None:
    """Verifies JSON file export writes valid JSON report."""
    json_out = tmp_path / "report.json"
    llm = MockLLMClient(dynamic_handler=create_adaptive_decision_handler())
    code = run_cli(
        goal=PRESETS["database"]["goal"],
        target_focus=PRESETS["database"]["focus"],
        llm_client=llm,
        github_client=mock_github,
        export_path=str(json_out),
    )
    assert code == 0
    assert json_out.exists()
    content = json_out.read_text(encoding="utf-8")
    assert '"status": "SUCCESS"' in content
    assert "schemacraft" in content.lower()


def test_cli_live_mode_without_keys_fails_helpfully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that running without configured keys returns code 1 with clean error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    code = run_cli(
        goal="Test goal",
    )
    assert code == 1
