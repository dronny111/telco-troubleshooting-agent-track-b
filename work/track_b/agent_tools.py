"""Tool specs (OpenAI function-call format) and the Agent Tool Server client.

The four tools mirror the openclaw skill taxonomy so Qwen can pick a
narrow command family per call:

    infra_maintenance — current-configuration, logbuffer, alarm, memory, lldp
    l2_link           — interface, eth-trunk, vlan, mac-address, stp
    l3_route          — ip routing-table, arp, ipv6 neighbor, ospf, bgp
    adv_tunnel        — vxlan, vrrp, bfd, ip pool, srv6, segment-routing

All four wrap the same underlying HTTP endpoint
`POST /api/agent/execute` on the Agent Tool Server. Splitting them in
the schema does not change the API contract — it only narrows the
description Qwen sees per slot, which empirically reduces wrong-family
command picks.

Endpoint and auth come from env vars so the same code targets the
Chinese-region ELB, the Hong Kong ECS, or a local sandbox without
modification.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_TIMEOUT_S = 25.0
DEFAULT_RETRIES = 2
DEFAULT_RETRY_BACKOFF_S = 1.5


@dataclass(frozen=True)
class AgentToolConfig:
    base_url: str
    token: str
    timeout_s: float = DEFAULT_TIMEOUT_S
    retries: int = DEFAULT_RETRIES
    verify_tls: bool = True

    @classmethod
    def from_env(cls) -> "AgentToolConfig":
        return cls(
            base_url=os.environ.get(
                "AGENT_TOOL_SERVER_URL",
                "http://localhost:7860/api/agent/execute",
            ),
            token=os.environ.get("AGENT_TOOL_SERVER_TOKEN", ""),
            timeout_s=float(os.environ.get("AGENT_TOOL_SERVER_TIMEOUT", DEFAULT_TIMEOUT_S)),
            retries=int(os.environ.get("AGENT_TOOL_SERVER_RETRIES", DEFAULT_RETRIES)),
            verify_tls=os.environ.get("AGENT_TOOL_SERVER_VERIFY_TLS", "1") not in ("0", "false", "False"),
        )


# ---- OpenAI function-call tool specs ---------------------------------------

def _tool(
    *,
    name: str,
    description: str,
    command_examples: tuple[str, ...],
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                description
                + "\n\nExamples of allowed commands for this skill: "
                + ", ".join(command_examples)
                + ".\nUse Huawei-style commands by default; the simulator also accepts the equivalent Cisco/H3C wording."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_name": {
                        "type": "string",
                        "description": "Target device hostname, e.g. Beta-Aegis-01 or Core_SW_01.",
                    },
                    "command": {
                        "type": "string",
                        "description": "The exact CLI command to execute (must match the simulator's regex whitelist; do not invent free-form commands).",
                    },
                },
                "required": ["device_name", "command"],
            },
        },
    }


TOOL_SPECS: tuple[dict, ...] = (
    _tool(
        name="infra_maintenance",
        description="Inspect device configuration, logs, alarms, memory, and link-layer discovery.",
        command_examples=(
            "display current-configuration",
            "display current-configuration | include ip route-static",
            "display current-configuration | include nat",
            "display logbuffer",
            "display alarm active",
            "display memory",
            "display lldp neighbor brief",
        ),
    ),
    _tool(
        name="l2_link",
        description="Inspect L2 interfaces, aggregation, VLANs, MAC tables, and Spanning Tree.",
        command_examples=(
            "display interface brief",
            "display interface description",
            "display ip interface brief",
            "display eth-trunk",
            "display vlan",
            "display mac-address",
            "display stp brief",
            "display stp interface brief",
        ),
    ),
    _tool(
        name="l3_route",
        description="Inspect L3 routing — IP/ARP, OSPF, BGP, and VPN-instance routing.",
        command_examples=(
            "display ip routing-table",
            "display arp",
            "display arp all",
            "display ipv6 neighbors",
            "display ip vpn-instance",
            "display ospf peer",
            "display ospf routing",
            "display ospf interface",
            "display bgp peer",
            "display bgp routing-table",
            "display bgp evpn all routing-table",
            "display bgp vpnv4 all routing-table",
        ),
    ),
    _tool(
        name="adv_tunnel",
        description="Inspect advanced features — VXLAN tunnels, VRRP groups, BFD, DHCP pools, SRv6 policies.",
        command_examples=(
            "display vxlan tunnel",
            "display vxlan troubleshooting",
            "display vrrp verbose",
            "display bfd session all",
            "display ip pool",
            "display srv6-te policy",
            "display srv6-te policy status",
            "display segment-routing ipv6 local-sid end forwarding",
        ),
    ),
)

TOOL_NAMES: frozenset[str] = frozenset(t["function"]["name"] for t in TOOL_SPECS)


# ---- Agent Tool Server client ---------------------------------------------

class AgentToolError(RuntimeError):
    pass


@dataclass
class CommandResult:
    status_code: int
    status: str  # "success" | "execution_failed" | "error"
    device_name: str
    command: str
    vendor: str | None
    result_text: str
    raw: dict[str, Any]


def execute_command(
    *,
    device_name: str,
    command: str,
    question_number: int | str,
    config: AgentToolConfig,
) -> CommandResult:
    """Call POST /api/agent/execute with retry on transient errors.

    Returns a `CommandResult` regardless of the simulator's status; failures
    surface as `status="execution_failed"` so the agent loop can decide
    whether to revise its plan rather than crashing.
    """
    payload = {
        "device_name": device_name,
        "command": command,
        "question_number": str(question_number),
    }
    headers = {"Content-Type": "application/json"}
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"

    last_exc: Exception | None = None
    for attempt in range(config.retries + 1):
        try:
            r = requests.post(
                config.base_url,
                headers=headers,
                json=payload,
                timeout=config.timeout_s,
                verify=config.verify_tls,
            )
        except requests.RequestException as e:
            last_exc = e
            if attempt < config.retries:
                time.sleep(DEFAULT_RETRY_BACKOFF_S * (attempt + 1))
                continue
            raise AgentToolError(f"network error: {e}") from e
        try:
            body = r.json()
        except json.JSONDecodeError:
            body = {"raw_text": r.text}
        # Prefer informative result fields when present
        return CommandResult(
            status_code=r.status_code,
            status=str(body.get("status", "success" if r.ok else "error")),
            device_name=device_name,
            command=command,
            vendor=body.get("vendor"),
            result_text=str(body.get("result", body.get("raw_text", body))),
            raw=body if isinstance(body, dict) else {"raw": body},
        )
    if last_exc is not None:
        raise AgentToolError(str(last_exc))
    raise AgentToolError("execute_command exhausted retries with no exception")


def dispatch_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    question_number: int | str,
    config: AgentToolConfig,
) -> CommandResult:
    """Resolve a Qwen tool_call to an AgentToolServer call.

    All four skill tools route to the same endpoint; the tool name is a
    routing hint to the LLM, not the server.
    """
    if tool_name not in TOOL_NAMES:
        raise AgentToolError(f"unknown tool {tool_name!r}; expected one of {sorted(TOOL_NAMES)}")
    device = arguments.get("device_name")
    command = arguments.get("command")
    if not device or not command:
        raise AgentToolError(
            f"tool {tool_name} requires device_name and command, got {arguments!r}"
        )
    return execute_command(
        device_name=str(device),
        command=str(command),
        question_number=question_number,
        config=config,
    )
