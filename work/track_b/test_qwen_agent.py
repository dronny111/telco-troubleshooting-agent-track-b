"""Smoke test for the Qwen agent scaffold.

Drives the full agent runtime end-to-end against a programmable Qwen stub
and a mocked Agent Tool Server, verifying:

    - tool specs are well-formed (4 tools, OpenAI-shape, required args)
    - QwenClient stub policy is honoured
    - run_scenario routes a tool_call to the dispatcher and feeds the
      tool result back into the conversation
    - validator integration: a constraint-violating draft answer is
      rejected and re-emitted; a clean draft is accepted
    - tool-call budget cap halts the loop with `budget_exhausted`
    - The OpenAI response parser correctly extracts tool_calls
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.agent_runtime import AgentLimits, run_scenario
from track_b.agent_tools import (
    AgentToolConfig,
    CommandResult,
    TOOL_NAMES,
    TOOL_SPECS,
    dispatch_tool_call,
)
from track_b.prompt_context import RankedRow
from track_b.qwen_client import (
    ChatResponse,
    QwenClient,
    ToolCall,
    _parse_openai_response,
)


REAL_LIMITS = Path(__file__).resolve().parents[2] / "telco_data" / "Track B" / "question_limits_config.json"


# Minimal Phase-2 vocab (subset; the real harness extracts per-question)
ROUTING_VOCAB = (
    "blackhole route", "missing static route", "static route error",
    "ARP configuration error", "BGP configuration error",
    "OSPF configuration error",
)
PORT_VOCAB = ("shutdown", "interface IP error")


# A complete fault question (with format spec + symptom) needed for the
# vocab extractor to recognise it.
SYNTH_QUESTION = (
    "This question involves two categories of faults: routing faults and port faults. "
    "Routing fault output format: fault-node;destination-IP;fault-reason. "
    "Fault reasons include: (1) blackhole route; (2) missing static route; "
    "(3) static route error; (4) ARP configuration error; (5) BGP configuration error; "
    "(6) OSPF configuration error. "
    "Port fault output format: fault-node;fault-port;fault-reason. "
    "Fault reasons include: (1) shutdown; (2) interface IP error. "
    "Routing fault examples:\n"
    "Beta-Axis-01;192.168.1.1;blackhole route\n"
    "Port fault examples:\n"
    "Beta-Node-01;GE1/0/2;interface IP error\n"
    "From SRC, accessing 10.0.0.1 failed."
)

PATH_QUESTION = (
    "Path from SH_STO_PC01 to SZ_Server_Cluster3(10.3.30.1). "
    "Format requirements: use the -> symbol to connect nodes, output all node names on the path "
    "(including L2 path nodes) and the node's physical outbound interface names, where the node "
    "name and physical outbound interface name are connected using the '_' symbol; the destination "
    "node only outputs the node name, without the physical outbound interface name."
)

TOPOLOGY_QUESTION = (
    "Supplement the topology links for Core_SW_01. "
    "Format: LocalNodeName(LocalPortNumber)->RemoteNodeName(RemotePortNumber)."
)


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def assert_eq(a, b, label: str) -> bool:
    ok = a == b
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: a={a!r} b={b!r}")
    return ok


def main() -> int:
    failures = 0

    section("tool specs are well-formed")
    failures += not assert_eq(len(TOOL_SPECS), 4, "4 tool specs")
    for spec in TOOL_SPECS:
        fn = spec.get("function") or {}
        params = fn.get("parameters") or {}
        required = set(params.get("required") or [])
        failures += not assert_eq(spec.get("type"), "function", f"{fn.get('name')}: type")
        failures += not assert_eq("device_name" in required, True, f"{fn.get('name')}: device_name required")
        failures += not assert_eq("command" in required, True, f"{fn.get('name')}: command required")
    failures += not assert_eq(
        TOOL_NAMES,
        frozenset({"infra_maintenance", "l2_link", "l3_route", "adv_tunnel"}),
        "tool names",
    )

    section("OpenAI response parser")
    body = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "l3_route",
                        "arguments": json.dumps({"device_name": "Core_SW_01", "command": "display ip routing-table"}),
                    },
                }],
            },
        }]
    }
    parsed = _parse_openai_response(body)
    failures += not assert_eq(parsed.has_tool_calls, True, "has_tool_calls")
    failures += not assert_eq(parsed.tool_calls[0].name, "l3_route", "tool_call name")
    failures += not assert_eq(
        parsed.tool_calls[0].arguments,
        {"device_name": "Core_SW_01", "command": "display ip routing-table"},
        "tool_call arguments",
    )

    section("agent runtime — tool call → tool result → final answer")
    # Stub Qwen: first turn requests a routing-table on Core_SW_01;
    # second turn emits a clean fault line.
    turn = {"i": 0}
    def policy(messages, tools):
        turn["i"] += 1
        if turn["i"] == 1:
            return ChatResponse(
                role="assistant", content="", finish_reason="tool_calls",
                tool_calls=[ToolCall(
                    id="call_1", name="l3_route",
                    arguments={"device_name": "Core_SW_01", "command": "display ip routing-table"},
                )],
            )
        return ChatResponse(
            role="assistant",
            content="Core_SW_01;10.0.0.1;blackhole route",
            finish_reason="stop",
        )

    qwen = QwenClient(policy=policy)

    # Stub the Agent Tool Server: dispatch_tool_call gets monkey-patched
    # in agent_runtime by patching the symbol it imported.
    def _fake_dispatch(*, tool_name, arguments, question_number, config):
        return CommandResult(
            status_code=200,
            status="success",
            device_name=str(arguments.get("device_name")),
            command=str(arguments.get("command")),
            vendor="huawei",
            result_text="Destination/Mask  Proto  ...  10.0.0.0/8 NULL0",
            raw={},
        )

    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_fake_dispatch):
        trace = run_scenario(
            scenario_id="t1",
            question_number=1,
            phase=1,
            question_text=SYNTH_QUESTION,
            candidates=[RankedRow("t1", "Core_SW_01", "blackhole route", "routing", 5.0)],
            graph_features={"Core_SW_01": {"on_parsed_path": "1"}},
            anomaly_evidence={
                ("t1", "Core_SW_01", "blackhole route"): None,  # presence of key is enough
            },  # type: ignore[arg-type]
            qwen=qwen,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=5, max_tool_calls=10),
        )

    failures += not assert_eq(trace.final_action, "accept", "final_action")
    failures += not assert_eq(trace.tool_calls_made, 1, "tool_calls_made")
    failures += not assert_eq(trace.iterations, 2, "iterations")
    failures += not assert_eq(trace.final_answer, "Core_SW_01;10.0.0.1;blackhole route", "final_answer")

    section("validator-driven re-emit on constraint violation")
    # First draft proposes the blacklisted node; second draft is clean.
    turn2 = {"i": 0}
    def policy2(messages, tools):
        turn2["i"] += 1
        if turn2["i"] == 1:
            return ChatResponse(role="assistant",
                                content="FW_02;10.0.0.1;blackhole route",
                                finish_reason="stop")
        return ChatResponse(role="assistant",
                            content="Core_SW_01;10.0.0.1;blackhole route",
                            finish_reason="stop")
    qwen2 = QwenClient(policy=policy2)
    blacklisted_question = SYNTH_QUESTION + "\nLimitation: Do not look for faults on FW_02."
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_fake_dispatch):
        trace = run_scenario(
            scenario_id="t2",
            question_number=1,
            phase=1,
            question_text=blacklisted_question,
            candidates=[RankedRow("t2", "Core_SW_01", "blackhole route", "routing", 5.0)],
            graph_features={"Core_SW_01": {"on_parsed_path": "1"},
                            "FW_02": {"on_parsed_path": "0"}},
            anomaly_evidence={("t2", "Core_SW_01", "blackhole route"): None},  # type: ignore[arg-type]
            qwen=qwen2,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=5, max_tool_calls=10),
        )
    failures += not assert_eq(trace.constraint_rejections, 1, "constraint rejection counted")
    failures += not assert_eq(trace.final_action, "accept", "accepted on second draft")
    failures += not assert_eq(trace.final_answer, "Core_SW_01;10.0.0.1;blackhole route", "final answer clean")

    section("duplicate tool calls are cached and do not burn budget")
    # Qwen keeps requesting the same tool call; runtime should cache it.
    def loop_policy(messages, tools):
        return ChatResponse(
            role="assistant", content="", finish_reason="tool_calls",
            tool_calls=[ToolCall(
                id=f"call_{len(messages)}", name="infra_maintenance",
                arguments={"device_name": "Core_SW_01", "command": "display current-configuration"},
            )],
        )
    qwen3 = QwenClient(policy=loop_policy)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_fake_dispatch):
        trace = run_scenario(
            scenario_id="t3",
            question_number=1,
            phase=1,
            question_text=SYNTH_QUESTION,
            candidates=[],
            graph_features={"Core_SW_01": {}},
            anomaly_evidence={},
            qwen=qwen3,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=20, max_tool_calls=3),
        )
    failures += not assert_eq(trace.final_action, "incomplete", "loop ends by iteration cap, not budget")
    failures += not assert_eq(trace.tool_calls_made, 1, "duplicate calls not recounted")

    section("path runtime forces a final no-tools answer after length stop")
    turn3 = {"i": 0}
    def path_policy(messages, tools, temperature=None, seed=None):
        turn3["i"] += 1
        saw_tool = any(m.get("role") == "tool" for m in messages)
        if tools and not saw_tool:
            return ChatResponse(
                role="assistant",
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(
                    id="call_path_1",
                    name="l3_route",
                    arguments={"device_name": "SH_AR", "command": "display ip routing-table 10.3.30.1"},
                )],
            )
        if tools:
            return ChatResponse(role="assistant", content="", finish_reason="length")
        return ChatResponse(
            role="assistant",
            content="SH_STO_PC01_GE0/0/1->SH_AR_GE0/0/2->SZ_AR_GE0/0/3->SZ_Server_Cluster3",
            finish_reason="stop",
        )
    qwen4 = QwenClient(policy=path_policy)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_fake_dispatch):
        trace = run_scenario(
            scenario_id="t4",
            question_number=66,
            phase=2,
            question_text=PATH_QUESTION,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen4,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=5),
        )
    failures += not assert_eq(trace.final_action, "accept", "path final_action")
    failures += not assert_eq(
        trace.final_answer,
        "SH_STO_PC01_GE0/0/1->SH_AR_GE0/0/2->SZ_AR_GE0/0/3->SZ_Server_Cluster3",
        "path final_answer",
    )
    failures += not assert_eq(trace.tool_calls_made >= 1, True, "path tool_calls_made")
    failures += not assert_eq(
        any(
            str(rec.get("validator_action", "")).startswith("forced_finalize:")
            for rec in trace.transcript
        ),
        True,
        "path transcript records forced finalization",
    )

    section("deterministic prepass solves underscore-style path without LLM")
    underscore_question = (
        "Path from DEV_A to DEV_C(10.1.1.1). "
        "Format requirements: use the -> symbol to connect nodes, output all node names on the path "
        "(including L2 path nodes) and the node's physical outbound interface names, where the node "
        "name and physical outbound interface name are connected using the '_' symbol; the destination "
        "node only outputs the node name, without the physical outbound interface name."
    )
    calls7 = {"n": 0}
    def should_not_call_qwen(messages, tools, temperature=None, seed=None):
        calls7["n"] += 1
        return ChatResponse(role="assistant", content="", finish_reason="stop")
    def det_dispatch(*, tool_name, arguments, question_number, config):
        device = str(arguments.get("device_name"))
        command = str(arguments.get("command"))
        result = ""
        if command == "display lldp neighbor brief":
            if device == "DEV_A":
                result = "GE1/0/1 120 GE1/0/2 DEV_B"
            elif device == "DEV_B":
                result = "GE1/0/3 120 GE1/0/4 DEV_C"
        return CommandResult(
            status_code=200,
            status="success",
            device_name=device,
            command=command,
            vendor="huawei",
            result_text=result,
            raw={},
        )
    qwen7 = QwenClient(policy=should_not_call_qwen)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=det_dispatch):
        trace = run_scenario(
            scenario_id="t7",
            question_number=79,
            phase=2,
            question_text=underscore_question,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen7,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=8),
        )
    failures += not assert_eq(trace.final_action, "accept", "det path underscore final_action")
    failures += not assert_eq(trace.final_answer, "DEV_A_GE1/0/1->DEV_B_GE1/0/3->DEV_C",
                              "det path underscore final_answer")
    failures += not assert_eq(calls7["n"], 0, "det path underscore avoided LLM loop")

    section("deterministic prepass solves bank22 style with inbound/outbound ports")
    bank22_question = (
        "In network bank22, restore all L2 forwarding paths from device DEV-BL-01 to device DEV-PE-03 "
        "with destination IP 2.2.7.1. Output format: For each path, provide the full connection "
        "information of the physical outbound port and physical inbound port of each node on the path. "
        "Use the full name GigabitEthernet instead of GE for physical port names. The node output format "
        "is as follows: node-name(physical-port-name). Nodes are connected using the -> symbol, with no "
        "extra whitespace characters in the middle, at the beginning, or at the end. Each path is output "
        "on one line; if there are multiple paths, output them on separate lines. The output format for a "
        "single path example: start-node(outbound-port)->intermediate-node0(inbound-port)->"
        "intermediate-node0(outbound-port)->intermediate-node1(inbound-port)->"
        "intermediate-node1(outbound-port)->end-node(inbound-port)"
    )
    calls8 = {"n": 0}
    def should_not_call_qwen2(messages, tools, temperature=None, seed=None):
        calls8["n"] += 1
        return ChatResponse(role="assistant", content="", finish_reason="stop")
    def bank_dispatch(*, tool_name, arguments, question_number, config):
        device = str(arguments.get("device_name"))
        command = str(arguments.get("command"))
        result = ""
        if command == "display lldp neighbor brief":
            if device == "DEV-BL-01":
                result = "GE1/0/1 120 GE1/0/2 DEV-SP-01"
            elif device == "DEV-SP-01":
                result = "GE1/0/3 120 GE1/0/4 DEV-PE-03"
        return CommandResult(
            status_code=200,
            status="success",
            device_name=device,
            command=command,
            vendor="huawei",
            result_text=result,
            raw={},
        )
    qwen8 = QwenClient(policy=should_not_call_qwen2)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=bank_dispatch):
        trace = run_scenario(
            scenario_id="t8",
            question_number=71,
            phase=2,
            question_text=bank22_question,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen8,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=10),
        )
    failures += not assert_eq(trace.final_action, "accept", "det bank22 final_action")
    failures += not assert_eq(
        trace.final_answer,
        "DEV-BL-01(GigabitEthernet1/0/1)->DEV-SP-01(GigabitEthernet1/0/2)->"
        "DEV-SP-01(GigabitEthernet1/0/3)->DEV-PE-03(GigabitEthernet1/0/4)",
        "det bank22 final_answer",
    )
    failures += not assert_eq(calls8["n"], 0, "det bank22 avoided LLM loop")

    section("deterministic prepass proxies bank22 VM source to edge leaf")
    bank22_vm_question = (
        "In network bank22, restore all L2 forwarding paths from device DEV-VM-02 to device DEV-PE-01 "
        "with destination IP 10.101.1.2. Output format: For each path, provide the full connection "
        "information of the physical outbound port and physical inbound port of each node on the path. "
        "Use the full name GigabitEthernet instead of GE for physical port names. The node output format "
        "is as follows: node-name(physical-port-name). Nodes are connected using the -> symbol, with no "
        "extra whitespace characters in the middle, at the beginning, or at the end. Each path is output "
        "on one line; if there are multiple paths, output them on separate lines. The output format for a "
        "single path example: start-node(outbound-port)->intermediate-node0(inbound-port)->"
        "intermediate-node0(outbound-port)->intermediate-node1(inbound-port)->"
        "intermediate-node1(outbound-port)->end-node(inbound-port)"
    )
    calls8b = {"n": 0, "devices": []}
    def should_not_call_qwen2b(messages, tools, temperature=None, seed=None):
        calls8b["n"] += 1
        return ChatResponse(role="assistant", content="", finish_reason="stop")
    def bank_vm_dispatch(*, tool_name, arguments, question_number, config):
        device = str(arguments.get("device_name"))
        command = str(arguments.get("command"))
        calls8b["devices"].append((device, command))
        if device == "DEV-VM-02":
            return CommandResult(
                status_code=422,
                status="error",
                device_name=device,
                command=command,
                vendor="linux",
                result_text="/bin/sh: display lldp neighbor brief: command not found",
                raw={},
            )
        result = ""
        if command == "display lldp neighbor brief":
            if device == "DEV-BL-01":
                result = "GE1/0/2 120 GE1/0/0 DEV-PE-01"
        return CommandResult(
            status_code=200,
            status="success",
            device_name=device,
            command=command,
            vendor="huawei",
            result_text=result,
            raw={},
        )
    qwen8b = QwenClient(policy=should_not_call_qwen2b)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=bank_vm_dispatch):
        trace = run_scenario(
            scenario_id="t8b",
            question_number=73,
            phase=2,
            question_text=bank22_vm_question,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen8b,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=10),
        )
    failures += not assert_eq(trace.final_action, "accept", "det bank22 vm proxy final_action")
    failures += not assert_eq(
        trace.final_answer,
        "DEV-BL-01(GigabitEthernet1/0/2)->DEV-PE-01(GigabitEthernet1/0/0)",
        "det bank22 vm proxy final_answer",
    )
    failures += not assert_eq(calls8b["n"], 0, "det bank22 vm proxy avoided LLM loop")
    failures += not assert_eq(
        ("DEV-BL-01", "display lldp neighbor brief") in calls8b["devices"],
        True,
        "det bank22 vm proxy seeded bank22 edge leaf",
    )

    section("deterministic prepass proxies HQ access endpoints to aggregation and firewall")
    hq_question = (
        "Path from HQ_FIN_PC01 to HQ_DNS_Server_01(10.1.60.1). "
        "Format requirements: use the -> symbol to connect nodes, output all node names on the path "
        "(including L2 path nodes) and the node's physical outbound interface names, where the node "
        "name and physical outbound interface name are connected using the '_' symbol; the destination "
        "node only outputs the node name, without the physical outbound interface name; the spelling of "
        "the physical outbound interface should follow the interface name in the node's configuration, "
        "for example, Huawei devices use the interface name from the display current-configuration command output."
    )
    calls8c = {"n": 0, "devices": []}
    def should_not_call_qwen2c(messages, tools, temperature=None, seed=None):
        calls8c["n"] += 1
        return ChatResponse(role="assistant", content="", finish_reason="stop")
    def hq_dispatch(*, tool_name, arguments, question_number, config):
        device = str(arguments.get("device_name"))
        command = str(arguments.get("command"))
        calls8c["devices"].append((device, command))
        result = ""
        if command == "display lldp neighbor brief":
            if device == "Core_SW_01":
                result = (
                    "GE1/0/4 120 HundredGigE1/0/1 AGG_SW_02\n"
                    "GE1/0/8 120 GigabitEthernet1/0/1 FW_01\n"
                )
            elif device == "Core_SW_02":
                result = (
                    "GE1/0/4 120 HundredGigE1/0/1 AGG_SW_02\n"
                    "GE1/0/8 120 GigabitEthernet1/0/1 FW_02\n"
                )
        elif command == "display current-configuration":
            if device == "AGG_SW_02":
                result = "interface HundredGigE1/0/1\n description uplink\n#"
            elif device == "FW_01":
                result = "interface GigabitEthernet1/0/1\n#"
            elif device == "Core_SW_01":
                result = "interface GE1/0/4\n#\ninterface GE1/0/8\n#"
        return CommandResult(
            status_code=200,
            status="success",
            device_name=device,
            command=command,
            vendor="huawei",
            result_text=result,
            raw={},
        )
    qwen8c = QwenClient(policy=should_not_call_qwen2c)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=hq_dispatch):
        trace = run_scenario(
            scenario_id="t8c",
            question_number=80,
            phase=2,
            question_text=hq_question,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen8c,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=10),
        )
    failures += not assert_eq(trace.final_action, "accept", "det HQ proxy final_action")
    failures += not assert_eq(
        trace.final_answer,
        "AGG_SW_02_HundredGigE1/0/1->Core_SW_01_GE1/0/8->FW_01",
        "det HQ proxy final_answer",
    )
    failures += not assert_eq(calls8c["n"], 0, "det HQ proxy avoided LLM loop")
    failures += not assert_eq(
        ("AGG_SW_02", "display current-configuration") in calls8c["devices"],
        True,
        "det HQ proxy enriched aggregation port naming",
    )

    section("deterministic prepass enriches underscore ports from current config")
    config_pref_question = (
        "Path from DEV_A to DEV_C(10.1.1.1). "
        "Format requirements: use the -> symbol to connect nodes, output all node names on the path "
        "(including L2 path nodes) and the node's physical outbound interface names, where the node "
        "name and physical outbound interface name are connected using the '_' symbol; the destination "
        "node only outputs the node name, without the physical outbound interface name; the spelling of "
        "the physical outbound interface should follow the interface name in the node's configuration, "
        "for example, Huawei devices use the interface name from the display current-configuration command output."
    )
    calls9 = {"n": 0}
    def should_not_call_qwen3(messages, tools, temperature=None, seed=None):
        calls9["n"] += 1
        return ChatResponse(role="assistant", content="", finish_reason="stop")
    def cfg_dispatch(*, tool_name, arguments, question_number, config):
        device = str(arguments.get("device_name"))
        command = str(arguments.get("command"))
        result = ""
        if command == "display lldp neighbor brief":
            if device == "DEV_A":
                result = "GE1/0/1 120 GE1/0/2 DEV_B"
            elif device == "DEV_B":
                result = "GE1/0/3 120 GE1/0/4 DEV_C"
        elif command == "display current-configuration":
            if device == "DEV_A":
                result = "interface GigabitEthernet1/0/1\n description to DEV_B\n#"
            elif device == "DEV_B":
                result = "interface GigabitEthernet1/0/2\n#\ninterface GigabitEthernet1/0/3\n#"
        return CommandResult(
            status_code=200,
            status="success",
            device_name=device,
            command=command,
            vendor="huawei",
            result_text=result,
            raw={},
        )
    qwen9 = QwenClient(policy=should_not_call_qwen3)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=cfg_dispatch):
        trace = run_scenario(
            scenario_id="t9",
            question_number=79,
            phase=2,
            question_text=config_pref_question,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen9,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=10),
        )
    failures += not assert_eq(trace.final_action, "accept", "config enrichment final_action")
    failures += not assert_eq(
        trace.final_answer,
        "DEV_A_GigabitEthernet1/0/1->DEV_B_GigabitEthernet1/0/3->DEV_C",
        "config enrichment final_answer",
    )
    failures += not assert_eq(calls9["n"], 0, "config enrichment avoided LLM loop")

    section("deterministic prepass falls back to interface brief for port spelling")
    calls10 = {"n": 0}
    def should_not_call_qwen4(messages, tools, temperature=None, seed=None):
        calls10["n"] += 1
        return ChatResponse(role="assistant", content="", finish_reason="stop")
    def ifbrief_dispatch(*, tool_name, arguments, question_number, config):
        device = str(arguments.get("device_name"))
        command = str(arguments.get("command"))
        result = ""
        if command == "display lldp neighbor brief":
            if device == "DEV_A":
                result = "GE1/0/1 120 GE1/0/2 DEV_B"
            elif device == "DEV_B":
                result = "GE1/0/3 120 GE1/0/4 DEV_C"
        elif command == "display current-configuration":
            result = ""
        elif command == "display interface brief":
            if device == "DEV_A":
                result = "GigabitEthernet1/0/1 up up --\n"
            elif device == "DEV_B":
                result = "GigabitEthernet1/0/2 up up --\nGigabitEthernet1/0/3 up up --\n"
        return CommandResult(
            status_code=200,
            status="success",
            device_name=device,
            command=command,
            vendor="huawei",
            result_text=result,
            raw={},
        )
    qwen10 = QwenClient(policy=should_not_call_qwen4)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=ifbrief_dispatch):
        trace = run_scenario(
            scenario_id="t10",
            question_number=79,
            phase=2,
            question_text=config_pref_question,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen10,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=12),
        )
    failures += not assert_eq(trace.final_action, "accept", "interface brief fallback final_action")
    failures += not assert_eq(
        trace.final_answer,
        "DEV_A_GigabitEthernet1/0/1->DEV_B_GigabitEthernet1/0/3->DEV_C",
        "interface brief fallback final_answer",
    )
    failures += not assert_eq(calls10["n"], 0, "interface brief fallback avoided LLM loop")

    section("path solver returns bounded shortest multipaths")
    from track_b.path_solver import PathEvidenceGraph, parse_path_question_spec
    from track_b.topology import parse_lldp_brief
    natural_spec = parse_path_question_spec(
        "Path from Shanghai branch Sales Department employee SH_STO_PC01 to "
        "Shenzhen data center SZ_Server_Cluster3(10.3.30.1)."
    )
    failures += not assert_eq(natural_spec.source_node, "SH_STO_PC01", "natural language source parse")
    failures += not assert_eq(natural_spec.destination_node, "SZ_Server_Cluster3", "natural language dest parse")
    failures += not assert_eq(
        parse_lldp_brief(
            "<SH_AR> display lldp neighbor brief\n"
            "Local Intf                     Neighbor Dev         Neighbor Intf        Exptime (sec)\n"
            "--------------------------------------------------------------------------------------\n"
            "Ethernet1/0/0                  SH_Core              Ethernet1/0/1        120\n"
            "Ethernet1/0/1                  PE2                  Ethernet1/0/0        120\n"
        ),
        [("Ethernet1/0/0", "Ethernet1/0/1", "SH_Core"), ("Ethernet1/0/1", "Ethernet1/0/0", "PE2")],
        "parse live Huawei AR LLDP format",
    )
    failures += not assert_eq(
        parse_lldp_brief(
            "SZ_Core# show lldp neighbors\n"
            "Device ID                         Local Intf          Hold-time   Capability      Port ID\n"
            "SH_AR                             Gi0/0/0             120         R               Gi0/0/1\n"
        ),
        [("Gi0/0/0", "Gi0/0/1", "SH_AR")],
        "parse Cisco show lldp neighbors format",
    )
    g = PathEvidenceGraph()
    g.add_link(src_dev="A", src_port="GE1/0/1", dst_dev="B", dst_port="GE1/0/2")
    g.add_link(src_dev="A", src_port="GE1/0/3", dst_dev="C", dst_port="GE1/0/4")
    g.add_link(src_dev="B", src_port="GE1/0/5", dst_dev="D", dst_port="GE1/0/6")
    g.add_link(src_dev="C", src_port="GE1/0/7", dst_dev="D", dst_port="GE1/0/8")
    g.add_link(src_dev="B", src_port="GE1/0/9", dst_dev="C", dst_port="GE1/0/10")
    paths = g.find_paths(source="A", destination="D", max_paths=6)
    failures += not assert_eq(paths, [["A", "B", "D"], ["A", "C", "D"]], "shortest multipaths only")
    rendered = g.render_paths(paths=paths, spec=parse_path_question_spec(underscore_question.replace("DEV_A", "A").replace("DEV_C", "D")))
    failures += not assert_eq(
        rendered,
        "A_GE1/0/1->B_GE1/0/5->D\nA_GE1/0/3->C_GE1/0/7->D",
        "rendered multipaths",
    )
    proxy_spec = parse_path_question_spec(
        "Path from GUEST_WIFI_CLIENT01 to BaiduWebServer01(8.8.8.8). "
        "Format requirements: use the -> symbol to connect nodes, output all node names on the path "
        "(including L2 path nodes) and the node's physical outbound interface names, where the node "
        "name and physical outbound interface name are connected using the '_' symbol; the destination "
        "node only outputs the node name, without the physical outbound interface name."
    )
    g2 = PathEvidenceGraph()
    g2.add_link(src_dev="Core_SW_01", src_port="GE1/0/8", dst_dev="FW_01", dst_port="GigabitEthernet1/0/1")
    g2.add_link(src_dev="FW_01", src_port="GigabitEthernet1/0/3", dst_dev="BJHQ_CSR1000V_GW_01", dst_port="Gi2")
    proxy_rendered = g2.render_paths(
        paths=g2.find_paths(source="Core_SW_01", destination="BJHQ_CSR1000V_GW_01"),
        spec=proxy_spec,
    )
    failures += not assert_eq(
        proxy_rendered,
        "Core_SW_01_GE1/0/8->FW_01_GigabitEthernet1/0/3->BJHQ_CSR1000V_GW_01",
        "rendered proxy path ignores trivial same-node pair",
    )

    section("deterministic prepass uses site seeds for endpoint questions")
    calls11 = {"qwen": 0, "devices": []}
    def should_not_call_qwen5(messages, tools, temperature=None, seed=None):
        calls11["qwen"] += 1
        return ChatResponse(role="assistant", content="", finish_reason="stop")
    def seeded_dispatch(*, tool_name, arguments, question_number, config):
        device = str(arguments.get("device_name"))
        command = str(arguments.get("command"))
        calls11["devices"].append((device, command))
        result = ""
        status_code = 200
        status = "success"
        if device == "BJHQ_CSR1000V_GW_01" and command == "display lldp neighbor brief":
            status_code = 422
            result = "BJHQ_CSR1000V_GW_01# display lldp neighbor brief\n                     ^\n% Invalid input detected at '^' marker."
        elif device == "BJHQ_CSR1000V_GW_01" and command == "show lldp neighbors":
            result = (
                "BJHQ_CSR1000V_GW_01# show lldp neighbors\n"
                "Device ID                         Local Intf          Hold-time   Capability      Port ID\n"
                "PE2                               Gi0/0/0             120         R               Gi0/0/1\n"
            )
        elif device == "PE2" and command == "display lldp neighbor brief":
            result = "GE1/0/1 120 GE1/0/2 SH_AR\nGE1/0/3 120 GE1/0/4 PE3"
        elif device == "PE3" and command == "display lldp neighbor brief":
            result = "GE1/0/5 120 GE1/0/6 PE2\nGE1/0/7 120 GE1/0/8 SZ_AR"
        elif device == "SH_AR" and command == "display lldp neighbor brief":
            result = (
                "<SH_AR> display lldp neighbor brief\n"
                "Local Intf                     Neighbor Dev         Neighbor Intf        Exptime (sec)\n"
                "--------------------------------------------------------------------------------------\n"
                "Ethernet1/0/0                  PE2                  Ethernet1/0/1        120\n"
            )
        elif device == "SZ_AR" and command == "display lldp neighbor brief":
            result = (
                "<SZ_AR> display lldp neighbor brief\n"
                "Local Intf                     Neighbor Dev         Neighbor Intf        Exptime (sec)\n"
                "--------------------------------------------------------------------------------------\n"
                "Ethernet1/0/0                  PE3                  Ethernet1/0/7        120\n"
            )
        return CommandResult(
            status_code=status_code,
            status=status,
            device_name=device,
            command=command,
            vendor="mixed",
            result_text=result,
            raw={},
        )
    qwen11 = QwenClient(policy=should_not_call_qwen5)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=seeded_dispatch):
        trace = run_scenario(
            scenario_id="t11",
            question_number=67,
            phase=2,
            question_text=PATH_QUESTION,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen11,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=8),
        )
    failures += not assert_eq(trace.final_action, "accept", "seeded fallback final_action")
    failures += not assert_eq(
        trace.final_answer,
        "SH_AR_Ethernet1/0/0->PE2_GE1/0/3->PE3_GE1/0/7->SZ_AR",
        "seeded fallback final_answer",
    )
    failures += not assert_eq(calls11["qwen"], 0, "seeded fallback avoided LLM loop")
    failures += not assert_eq(
        any(device in {"SH_STO_PC01", "SZ_Server_Cluster3"} for device, _command in calls11["devices"]),
        False,
        "seeded fallback skipped endpoint probes",
    )

    section("deterministic prepass falls back to Cisco LLDP command")
    calls12 = {"qwen": 0, "devices": []}
    cisco_question = (
        "Path from DEV_A to DEV_C(10.1.1.1). "
        "Format requirements: use the -> symbol to connect nodes, output all node names on the path "
        "(including L2 path nodes) and the node's physical outbound interface names, where the node "
        "name and physical outbound interface name are connected using the '_' symbol; the destination "
        "node only outputs the node name, without the physical outbound interface name."
    )
    def should_not_call_qwen6(messages, tools, temperature=None, seed=None):
        calls12["qwen"] += 1
        return ChatResponse(role="assistant", content="", finish_reason="stop")
    def cisco_dispatch(*, tool_name, arguments, question_number, config):
        device = str(arguments.get("device_name"))
        command = str(arguments.get("command"))
        calls12["devices"].append((device, command))
        if device == "DEV_A" and command == "display lldp neighbor brief":
            return CommandResult(
                status_code=422,
                status="error",
                device_name=device,
                command=command,
                vendor="cisco",
                result_text="% Invalid input detected at '^' marker.",
                raw={},
            )
        if device == "DEV_A" and command == "show lldp neighbors":
            return CommandResult(
                status_code=200,
                status="success",
                device_name=device,
                command=command,
                vendor="cisco",
                result_text=(
                    "Device ID                         Local Intf          Hold-time   Capability      Port ID\n"
                    "DEV_B                             Gi0/0/0             120         R               Gi0/0/1\n"
                ),
                raw={},
            )
        if device == "DEV_B" and command == "display lldp neighbor brief":
            return CommandResult(
                status_code=200,
                status="success",
                device_name=device,
                command=command,
                vendor="huawei",
                result_text="GE1/0/3 120 GE1/0/4 DEV_C",
                raw={},
            )
        return CommandResult(
            status_code=200,
            status="success",
            device_name=device,
            command=command,
            vendor="mixed",
            result_text="",
            raw={},
        )
    qwen12 = QwenClient(policy=should_not_call_qwen6)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=cisco_dispatch):
        trace = run_scenario(
            scenario_id="t12",
            question_number=79,
            phase=2,
            question_text=cisco_question,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen12,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=8),
        )
    failures += not assert_eq(trace.final_action, "accept", "Cisco fallback final_action")
    failures += not assert_eq(trace.final_answer, "DEV_A_Gi0/0/0->DEV_B_GE1/0/3->DEV_C", "Cisco fallback final_answer")
    failures += not assert_eq(calls12["qwen"], 0, "Cisco fallback avoided LLM loop")
    failures += not assert_eq(
        ("DEV_A", "show lldp neighbors") in calls12["devices"],
        True,
        "Cisco fallback tried alternate LLDP command",
    )

    section("agent runtime skips known denied tool calls")
    def denied_policy(messages, tools):
        if not any(m.get("role") == "tool" for m in messages):
            return ChatResponse(
                role="assistant",
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(
                    id="call_denied_1",
                    name="infra_maintenance",
                    arguments={"device_name": "Gamma-Axis-02", "command": "display lldp neighbor brief"},
                )],
            )
        return ChatResponse(
            role="assistant",
            content="Core_SW_01;10.0.0.1;blackhole route",
            finish_reason="stop",
        )
    qwen5 = QwenClient(policy=denied_policy)
    dispatch_hits = {"n": 0}
    def _should_not_dispatch(*, tool_name, arguments, question_number, config):
        dispatch_hits["n"] += 1
        return _fake_dispatch(
            tool_name=tool_name,
            arguments=arguments,
            question_number=question_number,
            config=config,
        )
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_should_not_dispatch):
        trace = run_scenario(
            scenario_id="t5",
            question_number=2,
            phase=1,
            question_text=SYNTH_QUESTION,
            candidates=[],
            graph_features={"Core_SW_01": {}},
            anomaly_evidence={("t5", "Core_SW_01", "blackhole route"): None},  # type: ignore[arg-type]
            qwen=qwen5,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=5),
        )
    failures += not assert_eq(dispatch_hits["n"], 0, "denied pair not dispatched")
    failures += not assert_eq(trace.tool_calls_made, 0, "denied pair not counted as API call")
    failures += not assert_eq(trace.final_action, "accept", "answer can recover after skipped denied call")

    section("strict path validation rejects missing intermediate interfaces")
    from track_b.format_guard import validate_path
    rep = validate_path(
        "SH_STO_PC01->SH_AR_GE0/0/2->SZ_Server_Cluster3",
        require_intermediate_interfaces=True,
        forbid_final_interface=True,
    )
    failures += not assert_eq(rep.is_valid, False, "strict path requires outbound interface")
    rep = validate_path(
        "SH_STO_PC01_GE0/0/1->SH_AR_GE0/0/2->SZ_Server_Cluster3_GE0/0/3",
        require_intermediate_interfaces=True,
        forbid_final_interface=True,
    )
    failures += not assert_eq(rep.is_valid, False, "strict path forbids destination interface")
    rep = validate_path(
        "Core_SW_01_GE1/0/8->FW_01_GigabitEthernet1/0/3->BJHQ_CSR1000V_GW_01_Gi4->PE1_Ethernet1/0/1->PE2_Ethernet1/0/0->SH_AR",
        require_intermediate_interfaces=True,
        forbid_final_interface=True,
    )
    failures += not assert_eq(rep.is_valid, True, "strict path accepts Cisco short Gi ports")

    section("topology runtime accepts topology answer")
    def topo_policy(messages, tools, temperature=None, seed=None):
        if tools and not any(m.get("role") == "tool" for m in messages):
            return ChatResponse(
                role="assistant",
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(
                    id="call_topo_1",
                    name="infra_maintenance",
                    arguments={"device_name": "Core_SW_01", "command": "display lldp neighbor brief"},
                )],
            )
        return ChatResponse(
            role="assistant",
            content="Core_SW_01(GE1/0/1)->AGG_SW_01(GE1/0/2)",
            finish_reason="stop",
        )
    qwen6 = QwenClient(policy=topo_policy)
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_fake_dispatch):
        trace = run_scenario(
            scenario_id="t6",
            question_number=1,
            phase=2,
            question_text=TOPOLOGY_QUESTION,
            candidates=[],
            graph_features={},
            anomaly_evidence={},
            qwen=qwen6,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            limits=AgentLimits(max_iterations=3, max_tool_calls=5),
        )
    failures += not assert_eq(trace.final_action, "accept", "topology final_action")
    failures += not assert_eq(
        trace.final_answer,
        "Core_SW_01(GE1/0/1)->AGG_SW_01(GE1/0/2)",
        "topology final_answer",
    )

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
