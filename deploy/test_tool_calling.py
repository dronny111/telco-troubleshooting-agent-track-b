"""End-to-end smoke test against a running vLLM server.

Verifies:
    1. /v1/models lists the configured Qwen3.5 model.
    2. A plain chat completion returns text.
    3. A tool-aware chat completion returns a structured tool_call.
    4. The thinking-mode-disabled chat_template_kwargs takes effect (no
       `<think>` blocks leak into the assistant response).

Reads the same env vars the agent runtime uses, so passing here means
`work/run_agent_demo.py` will work with no further configuration.

Run AFTER `vllm_serve.sh` is up and reporting `Application startup complete`.
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests


BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-35B-A3B")
TIMEOUT = float(os.environ.get("QWEN_TIMEOUT_S", "120"))


HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"  ok:   {msg}")


def http_get(path: str) -> tuple[int, dict]:
    r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=TIMEOUT)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {}


def http_post(path: str, body: dict) -> tuple[int, dict]:
    r = requests.post(f"{BASE_URL}{path}", headers=HEADERS, json=body, timeout=TIMEOUT)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"raw_text": r.text}


def test_models_endpoint() -> bool:
    section("/v1/models lists the model")
    code, body = http_get("/models")
    if code != 200:
        fail(f"status {code}: {body}")
        return False
    ids = [m.get("id") for m in body.get("data", [])]
    if MODEL not in ids:
        fail(f"{MODEL!r} not present. Saw: {ids}")
        return False
    ok(f"{MODEL} present in /v1/models")
    return True


def test_plain_chat() -> bool:
    section("plain chat completion")
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a network operations assistant. Answer briefly."},
            {"role": "user", "content": "Reply with the single word 'ready' and nothing else."},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
        # Belt-and-suspenders thinking-disable.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    code, resp = http_post("/chat/completions", body)
    if code != 200:
        fail(f"status {code}: {resp}")
        return False
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        fail(f"empty content: {resp}")
        return False
    if "<think>" in content or "</think>" in content:
        fail(f"thinking block leaked into content: {content!r}")
        return False
    ok(f"got content: {content!r}")
    return True


def test_tool_call_round_trip() -> bool:
    section("tool-call round trip (function-call schema)")
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Run a CLI command on a network device and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_name": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["device_name", "command"],
            },
        },
    }]
    messages = [
        {"role": "system", "content": (
            "You are NetOps-Agent. To answer questions about a device, "
            "call the tool. Do not answer from memory; you must inspect "
            "the device first."
        )},
        {"role": "user", "content": (
            "Show me the ARP table on Beta-Aegis-01."
        )},
    ]
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    code, resp = http_post("/chat/completions", body)
    if code != 200:
        fail(f"status {code}: {resp}")
        return False
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    tcs = msg.get("tool_calls") or []
    if not tcs:
        fail(f"no tool_calls in response. Got content: {msg.get('content')!r}")
        return False
    tc = tcs[0]
    fn = (tc.get("function") or {})
    args_raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError as e:
        fail(f"tool_calls[0].function.arguments not valid JSON: {args_raw!r} ({e})")
        return False
    if fn.get("name") != "execute_command":
        fail(f"unexpected tool name: {fn.get('name')!r}")
        return False
    for required in ("device_name", "command"):
        if required not in args:
            fail(f"missing required arg {required}: {args}")
            return False
    ok(f"tool_call → {fn['name']}({args})")
    # Now stuff a tool result back and verify the model can produce a
    # final answer that respects schema (no leading think block).
    messages.append({
        "role": "assistant",
        "content": msg.get("content") or "",
        "tool_calls": tcs,
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc.get("id", "call_test"),
        "name": fn["name"],
        "content": json.dumps({
            "status": "success",
            "device_name": args["device_name"],
            "command": args["command"],
            "result": "IP ADDRESS  MAC ADDRESS    EXPIRE  TYPE  VLAN  INTERFACE\n10.0.0.1   00aa-bb-cc-dd-ee  120  D    Vlanif100",
        }),
    })
    body2 = dict(body)
    body2["messages"] = messages
    code, resp = http_post("/chat/completions", body2)
    if code != 200:
        fail(f"second turn status {code}: {resp}")
        return False
    msg2 = (resp.get("choices") or [{}])[0].get("message") or {}
    content2 = (msg2.get("content") or "").strip()
    if "<think>" in content2 or "</think>" in content2:
        fail(f"thinking block in tool-result follow-up: {content2!r}")
        return False
    if not content2 and not msg2.get("tool_calls"):
        fail(f"empty second-turn content and no further tool_calls: {resp}")
        return False
    ok(f"second-turn output ok (len={len(content2)}): {content2[:120]!r}")
    return True


def main() -> int:
    print(f"target: {BASE_URL} model={MODEL}")
    print("waiting up to 30s for the server to be reachable...")
    for i in range(30):
        try:
            r = requests.get(f"{BASE_URL}/models", headers=HEADERS, timeout=2)
            if r.status_code in (200, 401):
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        print("server unreachable after 30s; is vllm_serve.sh running?")
        return 2

    failures = 0
    failures += not test_models_endpoint()
    failures += not test_plain_chat()
    failures += not test_tool_call_round_trip()

    print()
    if failures == 0:
        print("ALL CHECKS PASSED. The agent runtime is cleared to use this server.")
        return 0
    print(f"FAIL — {failures} check(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
