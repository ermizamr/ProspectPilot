"""LLM provider abstraction with OpenAI-compatible live provider and deterministic mock."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Protocol
from pydantic import BaseModel, Field


class ModelDecision(BaseModel):
    """Normalized response from any language model provider."""
    is_tool_call: bool
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    content: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    """Protocol defining the minimal interface required for agent loop execution."""

    def generate_decision(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> ModelDecision:
        """Inspects messages, reasons over available tools, and returns a normalized decision."""
        ...


class OpenAICompatibleClient:
    """Works with OpenAI, Groq, OpenRouter, Google Gemini (OpenAI endpoint), or local Ollama."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        # Read from explicit parameter or env vars with fallback cascade
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, "
                "or GEMINI_API_KEY in your environment, or configure OPENAI_BASE_URL."
            )

        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or (
                "https://api.groq.com/openai/v1"
                if os.getenv("GROQ_API_KEY")
                else None
            )
            or (
                "https://openrouter.ai/api/v1"
                if os.getenv("OPENROUTER_API_KEY")
                else None
            )
            or (
                "https://generativelanguage.googleapis.com/v1beta/openai/"
                if os.getenv("GEMINI_API_KEY")
                else None
            )
        )

        self.model = (
            model
            or os.getenv("PROSPECTPILOT_MODEL")
            or ("llama-3.3-70b-versatile" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini")
        )

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_decision(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> ModelDecision:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                err_str = str(exc)
                # 413 / Request too large / TPM exceeded: compact message history aggressively and retry
                if ("413" in err_str or "too large" in err_str.lower() or "tpm" in err_str.lower()) and attempt < max_retries:
                    import time
                    time.sleep(1.0)
                    from prospectpilot.agent.controller import AgentController
                    kwargs["messages"] = AgentController._compact_messages(kwargs["messages"], max_chars=4000)
                    continue
                # 429: Rate limit exceeded: backoff and retry
                elif "429" in err_str and attempt < max_retries:
                    import time
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise
        choice = response.choices[0]
        message = choice.message

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        # Handle tool call
        if message.tool_calls:
            call = message.tool_calls[0]
            fn_name = call.function.name
            try:
                fn_args = json.loads(call.function.arguments or "{}")
            except Exception:
                fn_args = {"_raw_unparsed": call.function.arguments}

            return ModelDecision(
                is_tool_call=True,
                tool_call_id=call.id,
                tool_name=fn_name,
                tool_arguments=fn_args,
                content=message.content,
                finish_reason="tool_calls",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        # Handle text / final decision
        return ModelDecision(
            is_tool_call=False,
            tool_name=None,
            tool_arguments={},
            content=message.content or "",
            finish_reason=choice.finish_reason or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class MockLLMClient:
    """Deterministic scriptable mock client for offline tests and predictable walkthroughs.

    Supports dynamic scripted handlers that react adaptively to previous observations.
    """

    def __init__(
        self,
        decisions: Optional[List[ModelDecision]] = None,
        dynamic_handler: Optional[Callable[[List[Dict[str, Any]]], ModelDecision]] = None,
    ) -> None:
        self.decisions = list(decisions) if decisions else []
        self.dynamic_handler = dynamic_handler
        self.call_history: List[List[Dict[str, Any]]] = []

    def generate_decision(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> ModelDecision:
        self.call_history.append(messages)

        # 1. Use dynamic handler if configured (for adaptive testing)
        if self.dynamic_handler:
            return self.dynamic_handler(messages)

        # 2. Pop next scripted decision
        if self.decisions:
            return self.decisions.pop(0)

        # 3. Default fallback: end research with empty/simple content
        return ModelDecision(
            is_tool_call=False,
            content="Research completed with available information.",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=10,
        )
