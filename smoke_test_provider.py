#!/usr/bin/env python3
"""Minimal smoke test for live LLM provider tool-calling.

Run this script to verify that your configured provider supports tool calling:
    python smoke_test_provider.py
"""

import json
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
        print("\nYou can run the deterministic test suite offline using:")
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
                    "name": "search_github_repos",
                    "description": "Search GitHub repositories by query string",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query, e.g. topic:devops",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "system",
                "content": "You are a research agent. When asked about repositories, you MUST call search_github_repos.",
            },
            {
                "role": "user",
                "content": "Search for repositories with topic:devops.",
            },
        ]

        decision = client.generate_decision(messages=messages, tools=tools)

        if not decision.is_tool_call:
            print("\n[!] WARNING: Model responded with text instead of a tool call:")
            print(f"    Content: {decision.content}")
            print("    Check if your configured model supports function/tool calling.")
            return 1

        print("\n[✓] STEP 1: Model returned a tool call:")
        print(f"    - Tool Name: {decision.tool_name}")
        print(f"    - Tool Call ID: {decision.tool_call_id}")
        print(f"    - Arguments: {decision.tool_arguments}")

        # Simulate tool execution and feedback
        print("\n[*] STEP 2: Executing tool and returning observation to model...")
        tool_call_id = decision.tool_call_id or "call_test_001"
        observation = '{"total_count": 1, "repos": [{"full_name": "hashicorp/terraform", "owner_login": "hashicorp", "stars": 42000}]}'

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
            "content": observation,
        })

        decision2 = client.generate_decision(messages=messages, tools=tools)

        if decision2.is_tool_call:
            print(f"\n[!] Model made another tool call: {decision2.tool_name}")
        else:
            print("\n[✓] STEP 3: Model processed tool observation and returned final answer:")
            print(f"    Summary: {decision2.content[:200]}...")

        total_tokens = decision.prompt_tokens + decision.completion_tokens + decision2.prompt_tokens + decision2.completion_tokens
        print(f"\n[✓] FULL ROUNDTRIP SUCCESSFUL (Total tokens used: ~{total_tokens})")
        print("Your provider supports end-to-end tool calling and observation roundtrips.")
        return 0

    except Exception as exc:
        print(f"\n[✗] ERROR contacting provider: {exc}")
        print("Check your API key, model name, and network connection.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
