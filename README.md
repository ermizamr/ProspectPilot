# ProspectPilot

[![CI](https://github.com/ermizamr/ProspectPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/ermizamr/ProspectPilot/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Type Checked](https://img.shields.io/badge/type%20checked-mypy-blue.svg)
![Tests](https://img.shields.io/badge/tests-73%20passed-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Evidence-Backed Developer-Prospecting Research Agent**  
> Bounded Python ReAct loop discovering and qualifying developer tools via the official GitHub REST API.

---

## Overview

**ProspectPilot** is a production-grade, typed Python agent that discovers, researches, and qualifies developer-tool organizations for outreach campaigns using public data.

- **Real Public Evidence**: Queries GitHub REST API for repos, org profiles, languages, and releases via `httpx`.
- **HTTP Resilience**: Timeouts (10s/30s), exponential backoff on 429/5xx, and rate-limit tracking.
- **Deterministic Python Scoring**: LLM gathers evidence, but Python computes the mathematical score (0–100).
- **Epistemic Honesty**: GitHub data proves code activity—not headcount, revenue, or hiring. Unknowns are explicitly labeled.
- **Strict Budget Guards**: Configurable iteration, tool-call, and execution time bounds.

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/ermizamr/ProspectPilot.git
cd ProspectPilot

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure Credentials
Set your LLM provider and optional GitHub token in `.env` (see `.env.example`):
```bash
# GitHub Token (Optional: raises limit from 60 to 5,000 req/hr)
export GITHUB_TOKEN="ghp_..."

# Any OpenAI-compatible LLM provider:
export GROQ_API_KEY="gsk_..."       # Free developer tier (console.groq.com)
# export OPENAI_API_KEY="sk-..."    # OpenAI gpt-4o-mini
# export GEMINI_API_KEY="AIzaSy..."  # Google AI Studio
# export OPENAI_BASE_URL="http://localhost:11434/v1" # Local Ollama
```

### 3. Usage
```bash
# Default run (Backend Developer Tools)
prospectpilot

# Use curated vertical presets
prospectpilot -p security              # Application security & SAST tools
prospectpilot -p devops                # Kubernetes, Docker & CI/CD tools
prospectpilot -p database              # Database DevOps & migration tools
prospectpilot -p api                   # API testing & mock server tools

# Export scored findings to CSV or JSON
prospectpilot -p security -o leads.csv
prospectpilot -p devops -o report.json

# Interactive vertical picker
prospectpilot -i

# Custom research objective
prospectpilot -g "Find developer tools for Rust web developers" -o rust_tools.csv
```

---

## Architecture

```
User Goal & Criteria
         │
         ▼
┌────────────────────────────────────────────────────────┐
│             AgentController (controller.py)            │
│  - Enforces max_iterations, tool calls, and time budget│
│  - Entity deduplication, timing & token accounting     │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
       Tool Call Request        Observation Feedback
                │                        │
                ▼                        │
┌──────────────────────────────────────┐ │
│       Tool Registry (base.py)        │ │
│  - Pydantic v2 schema validation     │ │
│  - Exception isolation & sandboxing  │ │
└───────────────┬──────────────────────┘ │
                │                        │
                ▼                        │
┌──────────────────────────────────────┐ │
│  GitHubClient (github_client.py)     │─┘
│  - search_github_repos               │
│  - get_org_profile / get_repo_details│
│  - get_recent_releases               │
└──────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Deterministic Scoring Engine (models.py)               │
│  - Relevance (40) + Activity (30) + Signals (30) = 100 │
│  - Attaches source URLs, timestamps & unknowns         │
└────────────────────────────────────────────────────────┘
```

---

## Epistemic Grounding

| What GitHub Proves (Evidence) | What GitHub Does NOT Prove (Labeled UNKNOWN) |
| :--- | :--- |
| Public repos, languages, stars, forks | Company headcount / employee count |
| Release cadence & update timestamps | Revenue, funding stage, or business health |
| Listed website, blog, and company string | Purchasing intent or active hiring budget |

---

## Testing & Quality

```bash
# Run 73 deterministic offline tests (0.30s, 0 network calls)
pytest

# Static type check (15 source files, 0 errors)
mypy prospectpilot

# Opt-in live GitHub API smoke test
pytest -m live
```

---

## License

[MIT](LICENSE) © 2026 Ermyas Misiker
