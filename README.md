# ProspectPilot

[![CI](https://github.com/ermizamr/ProspectPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/ermizamr/ProspectPilot/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Type Checked](https://img.shields.io/badge/type%20checked-mypy-blue.svg)
![Tests](https://img.shields.io/badge/tests-79%20passed-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Evidence-Backed Developer-Prospecting Research Agent**  
> *Production-grade Python ReAct loop discovering and qualifying developer tools via the official GitHub REST API.*

---

## Overview

**ProspectPilot** is a production-grade, bounded agentic Python CLI system that discovers, researches, and qualifies developer-tool organizations for evidence-backed outreach campaigns.

The system interacts directly with the **official GitHub REST API** to discover real repositories, inspect organization metadata, analyze release cadence, attach verifiable source URLs and retrieval timestamps, and compute deterministic qualification scores.

*(Synthetic fixtures are maintained strictly for deterministic offline unit testing.)*

---

## Key Engineering Decisions

| Requirement | Implementation Decision | Engineering Justification |
| :--- | :--- | :--- |
| **No Framework Baggage** | Pure Python Bounded Loop (`controller.py`) | Avoids brittle abstraction layers (e.g. LangChain, CrewAI). Budget enforcement, step timing, and token accounting remain completely deterministic and inspectable. |
| **Real Public Evidence** | Official GitHub REST API (`github_client.py`) | Gathers verifiable public repository data, organization profiles, and release history via `httpx`. Supports unauthenticated access (60 req/hr) or token-authenticated access (5,000 req/hr). |
| **HTTP Resilience** | Exponential Backoff & Rate-Limit Tracking | Implements configurable timeouts (10s connect, 30s read), bounded retries with backoff on 429/5xx, rate-limit header parsing (`X-RateLimit-*`), and proactive waiting. Never retries non-retryable 4xx errors. |
| **Strict Tool Sandboxing** | Pydantic Schema Validation (`tools/base.py`) | Every tool validates input arguments and structures outputs via Pydantic v2. Malformed arguments return structured feedback to the model rather than crashing the runtime. |
| **Deterministic Scoring** | Python Rubric Scorer (`models.py`) | The model navigates research and tool selection, but **Python computes the mathematical score (0–100)** and applies qualification thresholds. Hallucinated scores are structurally impossible. |
| **Epistemic Honesty** | Explicit Unknowns & Disclaimers | GitHub data reflects public development activity—**not** headcount, revenue, purchasing intent, or current hiring. Missing data is explicitly marked as `UNKNOWN`. Unverified company identities are labeled accordingly. |
| **Multi-Resource Budgets** | Iteration, Tool-Call, and Wall-Clock Limits | Enforces `max_iterations`, `max_tool_calls`, and `max_time_seconds` with graceful partial degradation (`PARTIAL_BUDGET_EXHAUSTED`). |
| **Deduplication** | In-Memory Identity Tracking | Prevents redundant tool calls and duplicate prospect evaluations for already-examined organizations. |
| **Credential Safety** | Zero-Leak Logging | API keys, bearer tokens, and secrets are strictly redacted and never logged or serialized to traces. |

---

## Project Architecture

```
User Goal & Qualification Criteria
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│             AgentController (controller.py)            │
│  - Enforces max_iterations, tool calls, and time budget│
│  - Step timing, token accounting, and audit trace      │
│  - Deduplication of examined repositories and orgs     │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
       Tool Call Request        Observation Feedback
                │                        │
                ▼                        │
┌──────────────────────────────────────┐ │
│       Tool Registry (base.py)        │ │
│  - Pydantic v2 argument validation   │ │
│  - Runtime exception isolation       │ │
└───────────────┬──────────────────────┘ │
                │                        │
        ┌───────┴───────┐                │
        ▼               ▼                │
┌──────────────┐ ┌─────────────────────┐ │
│ GitHub Tools │ │ Synthetic Tools     │ │
│ (Live Mode)  │ │ (Offline Mock Mode) │ │
│  - search    │ │  - search_companies │ │
│  - repos     │ │  - get_profile      │ │
│  - orgs      │ └──────────┬──────────┘ │
│  - releases  │            │            │
└───────┬──────┘            ▼            │
        │            Local Dataset       │
        ▼            (companies.json)────┘
┌──────────────────────────────────────┐
│  GitHubClient (github_client.py)     │
│  - HTTP timeouts (connect/read)      │
│  - Bounded retries + backoff         │
│  - Rate-limit header parsing         │
└──────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Deterministic Scoring Engine (models.py)               │
│  - evaluate_github_prospect_deterministic              │
│  - evaluate_company_deterministic                      │
│  - Produces EvidenceItem with source URLs & timestamps │
└────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Setup Environment
```bash
# Clone and enter directory
cd ProspectPilot

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode (registers the 'prospectpilot' command)
pip install -e ".[dev]"
```

### 2. Configure Credentials
Set your LLM API key and optional GitHub token in `.env` (see `.env.example`):
```bash
# GitHub Token (Optional but recommended: 5,000 req/hr vs 60 req/hr)
export GITHUB_TOKEN="ghp_..."

# Choose ONE LLM Provider:
export GROQ_API_KEY="gsk_..."      # Free fast tier via console.groq.com
# export OPENAI_API_KEY="sk-..."   # OpenAI gpt-4o-mini
# export GEMINI_API_KEY="AIzaSy..." # Google AI Studio
# export OPENAI_BASE_URL="http://localhost:11434/v1" # Local Ollama
```

### 3. Run Live Research
ProspectPilot runs live against the official GitHub REST API:
```bash
# Run with default Backend Developer Tools objective
prospectpilot

# Run curated vertical presets
prospectpilot -p security              # Application security & SAST tools
prospectpilot -p devops                # Cloud-native & Kubernetes tools
prospectpilot -p database              # Database DevOps & migration tools
prospectpilot -p api                   # API testing & mocking tools

# Interactive vertical selector
prospectpilot -i

# Export findings to CSV or JSON
prospectpilot -p security -o security_leads.csv
prospectpilot -o report.json

# Custom research goal
prospectpilot -g "Find developer tools for Rust web developers" -o rust_leads.csv
```

### 4. Run Offline Test Suite (0.33s execution, 0 network calls)
```bash
pytest
```
*Executes 79 deterministic unit and integration tests covering scoring, tool schema validation, budget enforcement, HTTP retry logic, rate-limit backoff, secondary rate-limit handling, pagination, SIGINT recovery, CLI presets, and file export.*

### 5. Run Strict Static Type Checking
```bash
mypy prospectpilot
```
*Validates 100% type annotations across all 15 source files with zero errors.*

### 6. Run Opt-In Live Smoke Tests
```bash
pytest -m live
```
*Runs 5 verified live tests hitting the real GitHub REST API to validate repository search, organization lookup, release inspection, and tool dispatch.*

---

## Offline Testing Flag

For deterministic local unit test debugging without network access, pass `--mock`:
```bash
prospectpilot --mock -p security
```
*(Synthetic fixtures in `data/companies.json` are maintained strictly for deterministic offline unit testing.)*

# 2. Configure LLM Provider (Choose one OpenAI-compatible provider)
# Option A: Groq Cloud (Free Developer Tier)
export GROQ_API_KEY="gsk_..."
export PROSPECTPILOT_MODEL="llama-3.3-70b-versatile"

# Option B: OpenAI
export OPENAI_API_KEY="sk-..."
export PROSPECTPILOT_MODEL="gpt-4o-mini"

# Option C: Google Gemini (Free via AI Studio OpenAI endpoint)
export GEMINI_API_KEY="AIzaSy..."
export PROSPECTPILOT_MODEL="gemini-1.5-flash"

# Option D: Local Ollama (100% offline LLM)
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
export PROSPECTPILOT_MODEL="llama3.1"
```

Verify your provider connection before running:
```bash
python smoke_test_provider.py
```

Run live research:
```bash
python -m prospectpilot.cli \
  --goal "Find developer-tool companies relevant to backend engineers and identify outreach angles" \
  --csv live_prospects.csv
```

---

## CLI Options & Usage

```
usage: prospectpilot [-h] [-p {backend,devops,security,database,api}] [-i]
                     [-g GOAL] [--focus FOCUS] [-m] [--live] [-o PATH]
                     [--csv PATH] [--json] [--max-iterations MAX_ITERATIONS]
                     [--max-tool-calls MAX_TOOL_CALLS] [--max-time MAX_TIME]
                     [-v]
```

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--preset` | `-p` | `backend` | Curated research vertical: `backend`, `devops`, `security`, `database`, `api`. |
| `--interactive`| `-i` | `False` | Interactive vertical picker menu with guided prompt. |
| `--output` | `-o` | `None` | Export results directly to file (auto-detects `.csv` or `.json` from extension). |
| `--goal` | `-g` | *Preset goal* | Custom natural language research objective for the LLM. |
| `--focus` | | *Preset focus* | Primary target keyword or vertical filter. |
| `--mock` | `-m` | *Auto* | Force offline mode using local deterministic dataset (default when no key is set). |
| `--live` | | `False` | Force live GitHub + LLM mode (fails helpfully if credentials are not configured). |
| `--json` | | `False` | Output structured Pydantic report as formatted JSON to stdout. |
| `--csv` | | `None` | Alias for `-o path.csv`. |
| `--max-iterations`| | `10` | Maximum control loop iterations before forced termination. |
| `--max-tool-calls` | | `20` | Upper bound on total tool invocations across the run. |
| `--max-time` | | `300.0` | Maximum wall-clock execution time in seconds. |
| `--verbose` | `-v` | `False` | Stream network requests, retries, and rate-limit transitions to stderr. |

---

## Epistemic Honesty: What GitHub Data Can and Cannot Prove

A central design principle of ProspectPilot is **epistemic discipline**:

### What GitHub Data Proves (Verified Evidence)
- Open-source repositories owned by an organization or user.
- Primary programming languages, topics, star counts, and fork counts.
- Active release cadence (tags, publication dates).
- Recent commit/update timestamps.
- Self-reported website/blog URLs and description fields.

### What GitHub Data Does NOT Prove (Explicitly Labeled as Unknown)
- **Company Headcount**: Star counts and repository volume do not correlate directly with employee count. Flagged as `UNKNOWN`.
- **Revenue & Funding**: Not present in the GitHub API. Flagged as `UNKNOWN`.
- **Purchasing Intent & Current Hiring**: Public development activity does not prove budget or active hiring for proprietary tooling. Flagged as `UNKNOWN`.
- **Company Identity**: An organization account may belong to an open-source collective, hobby project, or enterprise. If the profile's `company` attribute is absent, it is flagged as `Unverified organization-to-company identity`.

---

## Deterministic Scoring Rubric

### GitHub Evidence Rubric (Live Mode)
1. **Developer Relevance (Max 40 pts)**:
   - Topic/description keyword match (CLI, SDK, eBPF, database, testing, observability, etc.): `+25 pts`
   - Modern backend technologies (Go, Rust, Python, TypeScript, Kubernetes, Docker, etc.): `+3 pts` per language (up to `+15 pts`)
2. **Activity & Scale (Max 30 pts)**:
   - Community traction: $\ge 5,000$ stars (`+15 pts`), $\ge 500$ stars (`+10 pts`), $\ge 50$ stars (`+5 pts`)
   - Recent update within 90 days: `+5 pts`
   - Verified active releases: `+5 pts`
   - Multi-repository developer presence ($\ge 3$ repos): `+5 pts`
3. **Organization Signals (Max 30 pts)**:
   - Account is a confirmed GitHub Organization: `+10 pts`
   - Organization lists public website/blog: `+10 pts`
   - Organization has profile description: `+5 pts`
   - Organization specifies company identity: `+5 pts`

**Qualification Status:**
- `QUALIFIED`: Total score $\ge 70$
- `DISQUALIFIED`: Total score $< 50$
- `INSUFFICIENT_DATA`: Score $50–69$, or critical identity attributes unverified.

---

## Test Suite Overview

```
============================= test session starts ==============================
collected 85 items / 6 deselected / 79 selected

tests/test_adaptive_behavior.py ..                                       [  2%]
tests/test_cli.py ......                                                 [ 10%]
tests/test_controller.py .......                                         [ 18%]
tests/test_dataset.py .....                                              [ 25%]
tests/test_github_client.py ..................                           [ 48%]
tests/test_github_tools.py ...............                               [ 67%]
tests/test_models.py .....                                               [ 73%]
tests/test_scoring_github.py ...............                             [ 92%]
tests/test_tools.py ......                                               [100%]

======================= 79 passed, 6 deselected in 0.33s =======================
```

### Coverage Highlights
- **HTTP Resilience (`test_github_client.py`)**: 404/422 non-retryable handling, 500 retry-with-backoff, retry exhaustion, timeout recovery, `X-RateLimit` header parsing, `RateLimitExhausted` threshold checks, and credential leak prevention.
- **Tool Sandbox (`test_github_tools.py`)**: Schema validation, missing argument rejection, API error translation to structured observations, empty search result recovery.
- **Scoring & Evidence Grounding (`test_scoring_github.py`)**: Mathematical point tiers, epistemic disclaimer enforcement, source URL citations, organization deduplication, and execution time budget enforcement.
- **Adaptive Control (`test_adaptive_behavior.py`, `test_controller.py`)**: Observation-driven query adjustment, tool call ID preservation, iteration and tool-call budget limits.
- **Live Smoke (`test_live_smoke.py`)**: 5 opt-in live tests hitting GitHub REST API endpoints with verified roundtrips.

---

## Five-Minute Walkthrough Guide

Structure your technical evaluation video around these core pillars:

### Minute 1: Problem Statement & Pure Python Architecture
- **Show**: Directory structure (`prospectpilot/`).
- **Explain**: *"Rather than relying on opaque agent frameworks like LangChain, which obscure loop state and complicate budget enforcement, ProspectPilot uses a pure-Python ReAct controller with typed Pydantic schemas, strict budget guards, and complete execution tracing."*

### Minute 2: Real Public Data Integration & HTTP Resilience
- **Show**: `prospectpilot/github_client.py` and `prospectpilot/tools/github_tools.py`.
- **Explain**: *"We interact directly with GitHub's REST API. We implement connect/read timeouts, exponential backoff on 5xx and 429 errors, proactive rate-limit tracking via response headers, and safe credential handling—tokens are never logged."*

### Minute 3: Epistemic Honesty & Deterministic Python Scoring
- **Show**: `prospectpilot/models.py` (`evaluate_github_prospect_deterministic`).
- **Explain**: *"The model directs research, but Python calculates the qualification score. Every claim attaches a verifiable source URL and timestamp. Most importantly, we enforce epistemic honesty: GitHub data proves code activity, not headcount or revenue. Unavailable fields are explicitly marked as UNKNOWN."*

### Minute 4: Live CLI Demo & CSV Export
- **Show**: Run `python -m prospectpilot.cli --mock --csv prospects.csv`.
- **Explain**: *"Here is the execution trace. In Step 1, the agent queries the tools, observes the response, adapts to inspect a candidate profile, and produces a scored qualification table. We can also export this directly to a structured CSV with evidence citations."*

### Minute 5: Fast Deterministic Testing & Live Smoke Verification
- **Show**: Run `mypy prospectpilot` (0 issues), run `pytest` (73 tests passing in ~0.35s), then `pytest -m live` (5 live GitHub API smoke tests passing).
- **Explain**: *"Our offline test suite executes 73 tests in under 350 milliseconds with zero network dependencies, verifying budget exhaustion, rate limits, secondary rate limits, and error recovery. Live tests are opt-in and verify genuine external API integration."*

---

## Production Readiness & Known Limitations

To maintain absolute engineering integrity, the following limitations are explicitly noted:

| Feature | Current Implementation Status | What is Needed for Full Production |
| :--- | :--- | :--- |
| **API Integration** | Direct GitHub REST API with retries and rate-limiting | Add multi-page pagination cursor support for exhaustive repository scans ($>30$ results). |
| **Company Verification** | Heuristic match on GitHub Org `company` and `blog` fields | Integrate Clearbit, Crunchbase, or LinkedIn enrichment APIs to corroborate revenue and verified headcount. |
| **Storage / State** | In-memory execution trace with CSV/JSON export | Add persistent database (e.g. SQLite / PostgreSQL) for job queuing and historical lead tracking. |
| **Concurrency** | Synchronous bounded ReAct loop | Async tool execution (`httpx.AsyncClient`) for parallel repository inspection. |
| **LLM Fallback** | Normalized `LLMClient` protocol with OpenAI-compatible cascade | Automatic multi-provider failover if primary provider returns persistent 503/429. |
