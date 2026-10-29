#!/usr/bin/env python3
"""Minimal smoke test for live LLM provider tool-calling.

Run this script to verify that your configured provider supports tool calling:
    python smoke_test_provider.py
"""

import os
import sys
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()


def main() -> int:
    print("=" * 70)
    print(" ProspectPilot: Live Provider Tool-Calling Smoke Test")
    print("=" * 70)

    # Detect provider
    key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )

    if not key and not os.getenv("OPENAI_BASE_URL"):
        print("\n[!] No live LLM provider configured.")
        print("\nYou can run the entire agent offline using:")
        print("    python -m prospectpilot.cli --mock")
        print("    pytest\n")
        print("To enable a live LLM, choose any of these practical/free options:")
        print("  1. Groq (FREE tier, very fast):")
        print("     export GROQ_API_KEY=\"gsk_...\"")
        print("  2. Google Gemini (FREE tier via OpenAI endpoint):")
        print("     export GEMINI_API_KEY=\"AIzaSy...\"")
        print("  3. OpenAI (Very low cost ~0.02 cents/run for gpt-4o-mini):")
        print("     export OPENAI_API_KEY=\"sk-...\"")
        print("  4. Local Ollama (100% free, runs locally):")
        print("     export OPENAI_BASE_URL=\"http://localhost:11434/v1\"")
        print("     export OPENAI_API_KEY=\"ollama\"")
        print("\n(See .env.example for details)")
        return 0

    try:
        from prospectpilot.agent.llm import OpenAICompatibleClient

        client = OpenAICompatibleClient()
        print(f"[*] Detected Provider Endpoint: {client.base_url or 'https://api.openai.com/v1'}")
        print(f"[*] Configured Model: {client.model}")
        print("[*] Sending test prompt with tool schema to provider...")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_company",
                    "description": "Look up a company profile by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_id": {
                                "type": "string",
                                "description": "Company ID, e.g. comp-001",
                            }
                        },
                        "required": ["company_id"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "system",
                "content": "You are a research agent. When asked about a company, you MUST call lookup_company.",
            },
            {
                "role": "user",
                "content": "Look up company comp-001 in the dataset.",
            },
        ]

        decision = client.generate_decision(messages=messages, tools=tools)

        if decision.is_tool_call:
            print("\n[✓] SUCCESS: Model returned a validated tool call!")
            print(f"    - Tool Name: {decision.tool_name}")
            print(f"    - Arguments: {decision.tool_arguments}")
            print(f"    - Tokens Used: prompt={decision.prompt_tokens}, completion={decision.completion_tokens}")
            print("\nYour provider is verified and ready for live ProspectPilot runs.")
            return 0
        else:
            print("\n[!] WARNING: Model responded with text instead of a tool call:")
            print(f"    Content: {decision.content}")
            print("    Check if your configured model supports function/tool calling.")
            return 1

    except Exception as exc:
        print(f"\n[✗] ERROR contacting provider: {exc}")
        print("Check your API key, model name, and network connection.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
