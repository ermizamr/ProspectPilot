# ProspectPilot

> **Agentic Python Research & Qualification System for Developer Outreach**  
> *Engineered for Evaboot's Technical Assessment — Built with typed, framework-free Python.*

---

## Overview

**ProspectPilot** is a production-grade, bounded agentic CLI system that discovers, researches, and qualifies B2B software companies for personalized developer outreach campaigns.

Unlike toy agents that disguise a hardcoded chain of steps or rely on opaque frameworks (such as LangChain or CrewAI), ProspectPilot implements a **pure Python ReAct control loop** with strict budget boundaries, Pydantic v2 argument validation, deterministic scoring rules, and comprehensive observability.

---

## Key Engineering Decisions

| Requirement | Implementation Decision | Engineering Justification |
| :--- | :--- | :--- |
| **No Framework Baggage** | Pure Python Bounded Loop (`controller.py`) | Production data scrapers and lead gen systems require full transparency, deterministic execution budgets, and zero dependency churn. |
| **Defensible Evaluation** | Deterministic Python Rubric (`models.py`) | The LLM retrieves and interprets evidence, but **Python calculates the mathematical score** (0–100) and enforces qualification thresholds. This eliminates hallucinated scores. |
| **Grounded Evidence** | Verifiable Attribute Citations (`EvidenceItem`) | Every claim links directly to a retrieved dataset field. Missing or incomplete data is explicitly flagged as `UNKNOWN` / `INSUFFICIENT_DATA`. |
| **Safe Tool Execution** | Pydantic Schema Sandboxing (`tools/base.py`) | Malformed arguments return structured validation feedback to the model rather than crashing the Python process. |
| **Adaptive Behavior** | Dynamic Observation-Driven Decisions | The agent selects subsequent queries and profile lookups based on preceding tool observations. |
| **Offline Reproducibility** | `MockLLMClient` & Offline Test Suite | 22 automated tests run in **0.05 seconds** with **zero network access and no API keys**. |

---

## Project Architecture

```
User Goal & Criteria
         │
         ▼
┌────────────────────────────────────────────────────────┐
│             AgentController (controller.py)            │
│  - Enforces max_iterations (default: 6)                │
│  - Enforces max_tool_calls (default: 10)               │
│  - Step timing, token accounting, and audit trace      │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
       Tool Call Request        Observation Feedback
                │                        │
                ▼                        │
┌───────────────────────────────┐        │
│   Tool Registry (base.py)     │        │
│   - Pydantic schema validation│        │
│   - Runtime exception trap    │        │
└───────────────┬───────────────┘        │
                │                        │
       Validated Execution               │
                ▼                        │
┌───────────────────────────────┐        │
│ Local Dataset (dataset.py)    │────────┘
│   - search_companies          │
│   - get_company_profile       │
└───────────────────────────────┘
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

# Install dependencies in editable mode
pip install -e ".[dev]"
```

### 2. Run Offline Demo (Zero API Key Needed)
Run the deterministic adaptive mock demonstrating dynamic research, tool execution, and qualification:
```bash
python -m prospectpilot.cli --mock
```

To test adaptive query behavior with a different goal:
```bash
python -m prospectpilot.cli --goal "Find security developer tools" --mock
```

To export the structured Pydantic report as JSON:
```bash
python -m prospectpilot.cli --mock --json
```

### 3. Run Test Suite (0.05s execution)
```bash
pytest -v
```

---

## Live Provider Setup (Optional)

ProspectPilot works with any OpenAI-compatible provider. You can verify your live connection before running:

```bash
# Verify connection & tool calling
python smoke_test_provider.py
```

### Supported Providers
Configure your `.env` file (see `.env.example`):
- **Groq** (Fast & free tier): `export GROQ_API_KEY="gsk_..."`
- **Google Gemini** (Free tier via OpenAI endpoint): `export GEMINI_API_KEY="AIzaSy..."`
- **OpenAI** (Low cost `gpt-4o-mini`): `export OPENAI_API_KEY="sk-..."`
- **Local Ollama** (100% free offline): `export OPENAI_BASE_URL="http://localhost:11434/v1"`

Run the live agent:
```bash
python -m prospectpilot.cli --goal "Find developer tools for backend engineers"
```

---

## Deterministic Scoring Rubric

To ensure defensibility, qualification scoring is mathematically calculated by Python:

1. **Developer Audience Alignment (Max 40 pts)**:
   - Target audience contains software engineers, developers, or SREs: `+25 pts`
   - Modern developer tech stack (Go, Rust, Python, TypeScript, Docker, etc.): `+3 pts` per technology (up to `+15 pts`)
2. **Company Scale & Stage Fit (Max 30 pts)**:
   - Team size within target range (5–150 employees): `+20 pts`
   - Verified funding stage: `+10 pts`
3. **Outreach Engagement Signals (Max 30 pts)**:
   - Public GitHub repository: `+10 pts`
   - Active engineering blog: `+10 pts`
   - Active hiring for developer/engineering roles: `+10 pts`

**Qualification Status:**
- `QUALIFIED`: Total score $\ge$ 70
- `DISQUALIFIED`: Total score < 50
- `INSUFFICIENT_DATA`: Missing $\ge$ 2 critical data attributes and score < 70

---

## Five-Minute Walkthrough Guide

Use this script to structure your 5-minute technical assessment video:

### Minute 1: Problem Statement & Architectural Stance
- **Show**: Directory structure (`prospectpilot/`).
- **Explain**: *"Rather than using black-box agent frameworks like LangChain, which obscure loop state and make budget enforcement difficult, I designed ProspectPilot with a custom Bounded ReAct Loop in pure, typed Python."*

### Minute 2: Sandboxed Tools & Schema Validation
- **Show**: `prospectpilot/tools/base.py` and `company_tools.py`.
- **Explain**: *"Every tool uses Pydantic v2 schemas for argument validation. If a model passes invalid arguments or requests a nonexistent company, Python intercepts the error and returns it as a structured observation rather than crashing the process."*

### Minute 3: Deterministic Scoring vs. Model Hallucination
- **Show**: `prospectpilot/models.py` (`evaluate_company_deterministic`).
- **Explain**: *"In B2B data workflows like Evaboot's, data groundedness is non-negotiable. The model discovers and gathers evidence, but Python computes the qualification score using a transparent rubric. Any missing attribute is explicitly flagged as uncertainty."*

### Minute 4: Live Demonstration & Adaptive Trace
- **Show**: Run `python -m prospectpilot.cli --mock` in terminal.
- **Explain**: *"Here is the live trace. In Step 1, the agent searches for database tools. In Step 2, observing the returned ID `comp-002`, it adaptively calls `get_company_profile('comp-002')`. It then produces a validated summary table with explicit evidence citations."*

### Minute 5: Automated Testing & Budget Enforcement
- **Show**: Run `pytest -v` (22 tests passing).
- **Explain**: *"We have 22 deterministic tests covering budget exhaustion (`max_iterations`), malformed arguments, missing data handling, and an adaptive behavior test proving that contrasting observations produce contrasting actions."*

---

## Demonstration Dataset Disclaimer

The data located in `data/companies.json` consists of **synthetic demonstration records** designed specifically to test developer tooling qualification, non-technical disqualification, and edge cases with missing information.
