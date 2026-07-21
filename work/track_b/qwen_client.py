"""Minimal OpenAI-compatible chat client for Qwen3.5-35B-A3B.

vLLM and sglang both expose `/v1/chat/completions` with the OpenAI tool-call
schema, so a thin `requests` wrapper suffices and avoids a hard openai-SDK
dependency. The client supports two modes:

    real    — POST to OPENAI_BASE_URL/chat/completions
    stub    — call a programmable callable (used by tests and offline dev)

Switching between them is automatic: if `policy` is provided, stub mode
runs; otherwise the client reads `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
and `QWEN_MODEL` from the env.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests


@dataclass
class QwenConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_s: float = 60.0
    retries: int = 1

    @classmethod
    def from_env(cls) -> "QwenConfig | None":
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            return None
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-35B-A3B"),
            temperature=float(os.environ.get("QWEN_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("QWEN_MAX_TOKENS", "8192")),
            timeout_s=float(os.environ.get("QWEN_TIMEOUT_S", "60")),
            retries=int(os.environ.get("QWEN_RETRIES", "1")),
        )


@dataclass
class ToolCall:
    """Representation of a single function call requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    role: str
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    reasoning_content: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ---- Stub policy type ------------------------------------------------------

# A stub policy receives the full message list and the tool specs, and
# returns the next ChatResponse. Tests use this to drive deterministic
# multi-turn agent flows without a Qwen endpoint.
StubPolicy = Callable[[list[dict], list[dict]], ChatResponse]


class QwenClient:
    def __init__(
        self,
        config: QwenConfig | None = None,
        *,
        policy: StubPolicy | None = None,
    ):
        if policy is None and config is None:
            raise ValueError(
                "QwenClient requires either a real config or a stub policy"
            )
        self.config = config
        self.policy = policy

    @property
    def is_stub(self) -> bool:
        return self.policy is not None

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        tool_choice: str = "auto",
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        if self.policy is not None:
            # Stub policies that care about seed/temp can read them via the
            # 5-arg signature; older policies that take (messages, tools)
            # still work via the try/except.
            try:
                return self.policy(messages, tools or [], temperature, seed)  # type: ignore[arg-type]
            except TypeError:
                return self.policy(messages, tools or [])
        return self._chat_real(
            messages, tools or [],
            tool_choice=tool_choice,
            temperature=temperature,
            seed=seed,
        )

    # ---- real backend ----

    def _chat_real(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        assert self.config is not None
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        eff_temperature = self.config.temperature if temperature is None else temperature
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": eff_temperature,
            "max_tokens": self.config.max_tokens,
            # Belt-and-suspenders thinking-mode disable. The vLLM serve
            # script also strips <think> blocks via --reasoning-parser.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if "openrouter.ai" in self.config.base_url:
            # OpenRouter's hosted Qwen variants can emit reasoning-only
            # responses with `content=null` unless reasoning is explicitly
            # disabled in the provider-specific request shape.
            payload["reasoning"] = {"exclude": True}
            payload["include_reasoning"] = False
        if seed is not None:
            payload["seed"] = int(seed)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        last_exc: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                r = requests.post(
                    url, headers=headers, json=payload,
                    timeout=self.config.timeout_s,
                )
                r.raise_for_status()
                return _parse_openai_response(r.json())
            except (requests.RequestException, ValueError) as e:
                last_exc = e
                if attempt < self.config.retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"qwen chat exhausted retries: {last_exc}")


def _parse_openai_response(body: dict) -> ChatResponse:
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    raw_calls = msg.get("tool_calls") or []
    parsed_calls: list[ToolCall] = []
    for c in raw_calls:
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        parsed_calls.append(ToolCall(
            id=str(c.get("id", "")),
            name=str(fn.get("name", "")),
            arguments=args if isinstance(args, dict) else {},
        ))
    rc = msg.get("reasoning_content")
    if rc is None:
        rc = msg.get("reasoning")
    if rc is None:
        # OpenRouter occasionally returns reasoning_details=[{type: reasoning.text, text: "..."}]
        # without populating `reasoning` directly. Extract from there as a fallback.
        rd = msg.get("reasoning_details") or []
        rd_text = "".join(d.get("text", "") for d in rd if isinstance(d, dict))
        if rd_text:
            rc = rd_text
    content = msg.get("content")
    # OpenRouter (DeepInfra Qwen3.5-35B-A3B) routinely emits content=null with the
    # actual answer buried at the end of reasoning. Recover it.
    if content is None and rc:
        content = str(rc)
    return ChatResponse(
        role=str(msg.get("role", "assistant")),
        content=content,
        tool_calls=parsed_calls,
        finish_reason=str(choice.get("finish_reason", "")),
        reasoning_content=(str(rc) if rc is not None else None),
    )
