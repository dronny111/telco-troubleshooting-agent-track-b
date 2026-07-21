"""Per-scenario Qwen agent loop.

Drives the full runtime stack for one Track B scenario:

    1. Parse the question -> ParsedConstraints, vocab, task family.
    2. Pre-build the deterministic ranker prompt context (Step 9).
    3. Construct the Qwen system prompt + initial user message.
    4. Tool-call loop:
        a. Call Qwen with messages + 4 skill-domain tools.
        b. If model returns tool_calls: execute each via the Agent Tool
           Server, append a `tool` message with the result, continue.
        c. If model returns final text: validate via Step 10 validator.
            - accept              -> return.
            - reemit_format       -> append corrective hint, loop.
            - reemit_constraint   -> append violation list, loop.
            - fetch_evidence      -> execute the single proposed
                                     follow-up command (counts toward
                                     the per-question budget) and loop.
    5. Cap on iteration count; cap on total Agent Tool calls.

The runtime is stateless across scenarios; one agent run per scenario.
The Step 11 eval harness can drive it via `run_scenario(...)` to swap
out `_stub_predict` for the real agent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .agent_tools import (
    AgentToolConfig,
    AgentToolError,
    CommandResult,
    TOOL_SPECS,
    dispatch_tool_call,
)
from .answer_validator import ValidatorDecision, validate_answer
from .constraint_parser import ParsedConstraints, parse as parse_constraints
from .format_guard import validate_path, validate_topology
from .path_solver import PathEvidenceGraph, PathQuestionSpec, parse_path_question_spec
from .permission_pruner import denied_pairs as load_denied_pairs
from .prompt_context import (
    PHASE_2_DEVICES,
    AnomalyEvidence,
    RankedRow,
    build_context,
    build_system_prompt,
    _FAULT_EXEMPLAR,
    _PATH_EXEMPLAR,
    _TOPOLOGY_EXEMPLAR,
)
from .qwen_client import ChatResponse, QwenClient
from .task_classifier import classify
from .topology import parse_lldp_brief
from .vocab_extractor import extract_fault_vocab


@dataclass
class AgentLimits:
    max_iterations: int = 10        # full LLM turns (model invocations)
    max_tool_calls: int = 80        # Agent Tool Server calls per scenario
    soft_call_warning: int = 60     # warn if approaching the per-scenario 500 cap


@dataclass
class AgentTrace:
    scenario_id: str
    final_answer: str = ""
    iterations: int = 0
    tool_calls_made: int = 0
    follow_ups_triggered: int = 0
    format_rejections: int = 0
    constraint_rejections: int = 0
    final_action: str = "incomplete"  # 'accept' | 'incomplete' | 'budget_exhausted'
    transcript: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _allowed_reason_hint(
    routing_vocab: tuple[str, ...],
    port_vocab: tuple[str, ...],
) -> str:
    return (
        "Allowed routing reasons: "
        + ", ".join(routing_vocab)
        + "\nAllowed port reasons: "
        + ", ".join(port_vocab)
        + "\nDo not output symptom labels or intermediate diagnoses; output only one of the allowed final fault reasons."
    )


def _system_prompt_for_scenario(
    *,
    scenario_id: str,
    question_text: str,
    parsed: ParsedConstraints,
    candidates: list[RankedRow],
    anomaly_evidence: dict[tuple[str, str, str], AnomalyEvidence],
    denied: set[tuple[str, str]],
    phase: int,
) -> str:
    family = classify(question_text)
    ctx = build_context(
        task_family=family,
        parsed=parsed,
        candidates=candidates,
        anomaly_evidence=anomaly_evidence,
        denied_pairs=denied,
    )
    exemplar = _FAULT_EXEMPLAR if family == "fault" else _TOPOLOGY_EXEMPLAR
    return build_system_prompt(
        context=ctx,
        few_shot_exemplars=(exemplar,),
        include_phase_2_device_list=(phase == 2),
    )


def run_scenario(
    *,
    scenario_id: str,
    question_number: int,
    phase: int,
    question_text: str,
    candidates: list[RankedRow],
    graph_features: dict[str, dict],
    anomaly_evidence: dict[tuple[str, str, str], AnomalyEvidence],
    qwen: QwenClient,
    tool_config: AgentToolConfig,
    limits_path: str,
    limits: AgentLimits = AgentLimits(),
    calibrated_scores: dict[tuple[str, str, str], tuple[float, float]] | None = None,
    fetched_seed: set[tuple[str, str]] | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> AgentTrace:
    """Execute the agent loop for one scenario and return the trace."""
    parsed = parse_constraints(question_text)
    family = classify(question_text)

    if family == "path":
        return _run_path_scenario(
            scenario_id=scenario_id,
            question_number=question_number,
            phase=phase,
            question_text=question_text,
            parsed=parsed,
            qwen=qwen,
            tool_config=tool_config,
            limits_path=limits_path,
            limits=limits,
            fetched_seed=fetched_seed,
            temperature=temperature,
            seed=seed,
        )
    if family == "topology":
        return _run_topology_scenario(
            scenario_id=scenario_id,
            question_number=question_number,
            phase=phase,
            question_text=question_text,
            parsed=parsed,
            qwen=qwen,
            tool_config=tool_config,
            limits_path=limits_path,
            limits=limits,
            fetched_seed=fetched_seed,
            temperature=temperature,
            seed=seed,
        )

    trace = AgentTrace(scenario_id=scenario_id)
    routing_vocab, port_vocab = extract_fault_vocab(question_text)
    if family != "fault" or not (routing_vocab and port_vocab):
        trace.errors.append(
            f"non-fault task family={family!r} or empty vocab; fault runtime requires both vocabs"
        )
        return trace

    denied = load_denied_pairs(question_number, limits_path)
    fetched: set[tuple[str, str]] = set(fetched_seed or set())
    command_cache: dict[tuple[str, str], str] = {}
    anomaly_set: set[tuple[str, str, str]] = set(anomaly_evidence.keys())

    sys_prompt = _system_prompt_for_scenario(
        scenario_id=scenario_id,
        question_text=question_text,
        parsed=parsed,
        candidates=candidates,
        anomaly_evidence=anomaly_evidence,
        denied=denied,
        phase=phase,
    )
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_text},
    ]
    trace.transcript.append({"role": "system", "content_len": len(sys_prompt)})
    trace.transcript.append({"role": "user", "content_len": len(question_text)})

    for it in range(limits.max_iterations):
        trace.iterations = it + 1
        try:
            response = qwen.chat(
                messages,
                tools=list(TOOL_SPECS),
                tool_choice="auto",
                temperature=temperature,
                seed=seed,
            )
        except Exception as e:
            trace.errors.append(f"qwen call failed at it={it}: {e}")
            trace.final_action = "incomplete"
            return trace

        if response.has_tool_calls:
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            })
            trace.transcript.append({
                "role": "assistant",
                "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in response.tool_calls],
            })
            for tc in response.tool_calls:
                if trace.tool_calls_made >= limits.max_tool_calls:
                    trace.errors.append("tool call budget exhausted")
                    trace.final_action = "budget_exhausted"
                    return trace
                body_text = _execute_or_skip_tool_call(
                    tc=tc,
                    question_number=question_number,
                    tool_config=tool_config,
                    denied=denied,
                    fetched=fetched,
                    command_cache=command_cache,
                    trace=trace,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": body_text,
                })
                trace.transcript.append({
                    "role": "tool",
                    "name": tc.name,
                    "content_len": len(body_text),
                })
            continue

        # Final assistant message — validate it.
        draft = response.content or ""
        decision = validate_answer(
            draft_answer=draft,
            scenario_id=scenario_id,
            parsed=parsed,
            routing_vocab=routing_vocab,
            port_vocab=port_vocab,
            graph_features=graph_features,
            anomaly_evidence=anomaly_set,
            denied_pairs=denied,
            calibrated_scores=calibrated_scores,
            fetched_commands=fetched,
        )
        rc = response.reasoning_content
        trace.transcript.append({
            "role": "assistant",
            "draft_answer": draft,
            "validator_action": decision.action,
            "finish_reason": response.finish_reason,
            "reasoning_content_len": (len(rc) if rc else 0),
            "reasoning_content_head": (rc[:600] if rc else ""),
        })

        if decision.action == "accept":
            trace.final_answer = decision.normalised_answer
            trace.final_action = "accept"
            return trace

        if decision.action == "reemit_format":
            trace.format_rejections += 1
            messages.append({"role": "assistant", "content": draft})
            messages.append({
                "role": "user",
                "content": (
                    "Validator: "
                    + decision.format_hint
                    + "\n"
                    + _allowed_reason_hint(routing_vocab, port_vocab)
                ),
            })
            continue

        if decision.action == "reemit_constraint":
            trace.constraint_rejections += 1
            messages.append({"role": "assistant", "content": draft})
            messages.append({
                "role": "user",
                "content": (
                    "Validator: constraint violations:\n- "
                    + "\n- ".join(decision.constraint_violations)
                    + "\nRevise the answer accordingly. Do not propose faults on blacklisted nodes."
                ),
            })
            continue

        if decision.action == "fetch_evidence" and decision.follow_up:
            device, command, for_reason = decision.follow_up
            if trace.tool_calls_made >= limits.max_tool_calls:
                trace.errors.append("tool call budget exhausted before follow-up")
                trace.final_action = "budget_exhausted"
                return trace
            body_text = _execute_or_skip_command(
                tool_name=_pick_tool_for_command(command),
                device=device,
                command=command,
                question_number=question_number,
                tool_config=tool_config,
                denied=denied,
                fetched=fetched,
                command_cache=command_cache,
                trace=trace,
                error_prefix="follow-up error",
            )
            if (device, command) in fetched:
                trace.follow_ups_triggered += 1
            messages.append({"role": "assistant", "content": draft})
            messages.append({
                "role": "user",
                "content": (
                    f"Validator: confidence too low for {device}/{for_reason}; "
                    f"executed targeted follow-up {command!r}; new evidence below.\n"
                    + body_text
                    + "\nRevise the answer based on this new evidence."
                ),
            })
            continue

    trace.final_action = "incomplete"
    return trace


def _build_path_system_prompt(
    *,
    parsed: ParsedConstraints,
    denied: set[tuple[str, str]],
    phase: int,
) -> str:
    src = parsed.source_endpoint or "(unspecified)"
    dest_ip = parsed.target_destination_ip or ""
    dest_host = parsed.target_destination_host or parsed.target_destination_node or ""
    dest = (f"{dest_host} ({dest_ip})" if dest_host and dest_ip
            else dest_host or dest_ip or "(unspecified)")
    parts: list[str] = [
        "You are NetOps-Agent, a path-discovery agent on a multi-vendor campus network."
        " Discover the service path hop by hop using the four skill tools."
        " Output ONLY the final answer in the exact schema the question demands —"
        " no commentary, no markdown, no explanation.",
        "Path discovery playbook:\n"
        "1. Resolve the destination prefix with `display ip routing-table <dest_ip>` on the source-side gateway.\n"
        "2. Follow next-hops through each device using `display ip routing-table` and `display arp`.\n"
        "3. Confirm the outgoing physical interface name with `display current-configuration` or `display interface brief`.\n"
        "4. Include every L2 hop between L3 devices (use `display mac-address` / `display lldp neighbor` if available).\n"
        "5. The destination node is emitted WITHOUT an outbound interface — the path ends at the destination name.\n"
        "6. Re-read the question for the exact join character between node and interface ('_' or '(...)' — use what the question says).",
        f"Endpoints: source={src!r}  destination={dest!r}",
    ]
    if phase == 2:
        parts.append(
            "Phase 2 known device vocabulary (use only these names; do not invent):\n"
            + ", ".join(PHASE_2_DEVICES)
        )
    if denied:
        parts.append(
            "Hard-blocked (device, command) pairs — do NOT call these:\n"
            + "\n".join(f"  - {d} : {c}" for d, c in sorted(denied))
        )
    parts.append("Example:\n" + _PATH_EXEMPLAR.strip())
    return "\n\n".join(parts)


def _force_path_final_answer(
    *,
    qwen: QwenClient,
    messages: list[dict],
    trace: AgentTrace,
    label: str,
    temperature: float | None,
    seed: int | None,
    require_intermediate_interfaces: bool,
    forbid_final_interface: bool,
) -> str:
    compact_evidence: list[str] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        try:
            payload = json.loads(str(msg.get("content") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        device = str(payload.get("device_name") or "")
        command = str(payload.get("command") or "")
        result = str(payload.get("result") or payload.get("content") or "")[:500]
        compact_evidence.append(
            f"- {device} | {command}\n{result}"
        )
    compact_evidence = compact_evidence[-12:]
    prompt_messages = [
        messages[0],
        messages[1],
        {
            "role": "user",
            "content": (
                "Collected evidence (most recent first may be most relevant):\n"
                + "\n\n".join(compact_evidence)
                + "\n\nStop calling tools. Based only on this evidence, emit the final path answer now.\n"
                "Use ONLY the schema the question specified. No prose, no markdown, no surrounding text. "
                "One path per line. Use '->' to join hops. The destination node must appear last with no trailing interface."
            ),
        },
    ]
    response = qwen.chat(
        prompt_messages,
        tools=[],
        temperature=temperature,
        seed=seed,
    )
    draft = response.content or ""
    rep = validate_path(
        draft,
        require_intermediate_interfaces=require_intermediate_interfaces,
        forbid_final_interface=forbid_final_interface,
    )
    trace.transcript.append({
        "role": "assistant",
        "draft_answer": draft,
        "validator_action": f"{label}:{'accept' if rep.is_valid else 'reemit_format'}",
        "finish_reason": response.finish_reason,
    })
    return rep.normalised if rep.is_valid else ""


def _is_deterministic_path_target(*, phase: int, question_number: int) -> bool:
    return phase == 2 and 67 <= int(question_number) <= 100


_PATH_BACKBONE_SEEDS: tuple[str, ...] = (
    "Core_SW_01",
    "Core_SW_02",
    "SH_Core",
    "SH_AR",
    "SZ_Core",
    "SZ_AR",
    "BJHQ_CSR1000V_GW_01",
    "FW_01",
)

_LLDP_COMMANDS: tuple[str, ...] = (
    "display lldp neighbor brief",
    "show lldp neighbors",
)


def _path_forbids_final_interface(question_text: str) -> bool:
    q = question_text.lower()
    if "end-node(inbound-port)" in q or "end node(inbound-port)" in q:
        return False
    return _path_requires_interfaces(question_text)


def _looks_like_endpoint(node_name: str) -> bool:
    n = node_name.lower()
    return any(tok in n for tok in ("client", "pc", "server", "internet", "wifi", "vm"))


def _phase2_path_seed_hints(node_name: str) -> list[str]:
    if not node_name:
        return []
    if node_name in {"BaiduWebServer01", "GoogleWebServer01"}:
        return ["BJHQ_CSR1000V_GW_01", "FW_01", "FW_02"]
    if node_name.startswith(("HQ_MKT_PC01", "HQ_MKT_Client01", "HQ_MKT_AP01")):
        return ["AGG_SW_01", "Core_SW_01", "Core_SW_02", "BJHQ_CSR1000V_GW_01"]
    if node_name.startswith(("HQ_FIN_PC01", "HQ_FIN_Client01")):
        return ["AGG_SW_02", "Core_SW_01", "Core_SW_02", "BJHQ_CSR1000V_GW_01"]
    if node_name.startswith(("HQ_HR_PC01", "HQ_HR_AP01")):
        return ["AGG_SW_03", "Core_SW_01", "Core_SW_02", "BJHQ_CSR1000V_GW_01"]
    if node_name.startswith(("HQ_PROC_PC01", "HQ_PROC_AP01")):
        return ["AGG_SW_04", "Core_SW_01", "Core_SW_02", "BJHQ_CSR1000V_GW_01"]
    if node_name.startswith("HQ_DNS_Server_01"):
        return ["FW_01", "FW_02", "Core_SW_01", "Core_SW_02"]
    if node_name.startswith("DEV-VM-02"):
        return ["DEV-BL-01", "DEV-SP-01"]
    if node_name.startswith("DEV-PC-01"):
        return ["DEV-SL-01", "DEV-SP-01"]
    if node_name.startswith("DEV-PC-02"):
        return ["DEV-SL-02", "DEV-SP-02"]
    if node_name.startswith("SH_"):
        return ["SH_AR", "SH_Core"]
    if node_name.startswith("SZ_"):
        return ["SZ_AR", "SZ_Core"]
    if (
        node_name.startswith(("BJHQ_", "EMPLOYEE_WIFI_", "GUEST_WIFI_", "Core_SW_", "AGG_SW_", "FW_"))
        or node_name in {"BaiduWebServer01", "GoogleWebServer01"}
    ):
        return ["Core_SW_01", "BJHQ_CSR1000V_GW_01", "FW_01"]
    return []


def _primary_endpoint_proxy(node_name: str) -> str | None:
    if not node_name:
        return None
    if node_name.startswith(("HQ_MKT_PC01", "HQ_MKT_Client01", "HQ_MKT_AP01")):
        return "AGG_SW_01"
    if node_name.startswith(("HQ_FIN_PC01", "HQ_FIN_Client01")):
        return "AGG_SW_02"
    if node_name.startswith(("HQ_HR_PC01", "HQ_HR_AP01")):
        return "AGG_SW_03"
    if node_name.startswith(("HQ_PROC_PC01", "HQ_PROC_AP01")):
        return "AGG_SW_04"
    if node_name.startswith("HQ_DNS_Server_01"):
        return "FW_01"
    if node_name.startswith("DEV-VM-02"):
        return "DEV-BL-01"
    if node_name.startswith("DEV-PC-01"):
        return "DEV-SL-01"
    if node_name.startswith("DEV-PC-02"):
        return "DEV-SL-02"
    if node_name.startswith("SH_"):
        return "SH_AR"
    if node_name.startswith("SZ_"):
        return "SZ_AR"
    if node_name.startswith(("EMPLOYEE_WIFI_", "GUEST_WIFI_", "BJHQ_")):
        return "Core_SW_01"
    if node_name in {"BaiduWebServer01", "GoogleWebServer01"} or "internet" in node_name.lower():
        return "BJHQ_CSR1000V_GW_01"
    return None


def _candidate_path_search_pairs(spec: PathQuestionSpec) -> list[tuple[str, str]]:
    if not spec.source_node or not spec.destination_node:
        return []
    src_candidates = [spec.source_node]
    dst_candidates = [spec.destination_node]
    if _looks_like_endpoint(spec.source_node):
        proxy = _primary_endpoint_proxy(spec.source_node)
        if proxy:
            src_candidates.append(proxy)
        src_candidates.extend(_phase2_path_seed_hints(spec.source_node))
    if _looks_like_endpoint(spec.destination_node):
        proxy = _primary_endpoint_proxy(spec.destination_node)
        if proxy:
            dst_candidates.append(proxy)
        dst_candidates.extend(_phase2_path_seed_hints(spec.destination_node))
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for src in src_candidates:
        for dst in dst_candidates:
            pair = (src, dst)
            if src and dst and pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _initial_path_probe_nodes(spec: PathQuestionSpec, question_text: str) -> list[str]:
    is_bank22 = "network bank22" in question_text.lower()
    ordered: list[str] = []
    hinted_any = False
    for node in (spec.source_node, spec.destination_node):
        if not node:
            continue
        if is_bank22 or not _looks_like_endpoint(node):
            ordered.append(node)
        hints = _phase2_path_seed_hints(node)
        if hints:
            hinted_any = True
        ordered.extend(hints)
    if (
        (not is_bank22)
        and (not hinted_any)
        and any(_looks_like_endpoint(node or "") for node in (spec.source_node, spec.destination_node))
    ):
        ordered.extend(_PATH_BACKBONE_SEEDS[:2])
    queued: list[str] = []
    seen: set[str] = set()
    for node in ordered:
        if node and node not in seen:
            seen.add(node)
            queued.append(node)
    return queued


def _is_transit_priority_node(node_name: str, spec: PathQuestionSpec) -> bool:
    if node_name.startswith("PE"):
        return True
    return node_name in {
        proxy
        for proxy in (
            _primary_endpoint_proxy(spec.source_node or ""),
            _primary_endpoint_proxy(spec.destination_node or ""),
        )
        if proxy
    }


def _lldp_commands_for_device(device: str) -> tuple[str, ...]:
    if device.startswith("AGG_SW_"):
        return ()
    if device in {"BJHQ_CSR1000V_GW_01", "SZ_Core"}:
        return ("show lldp neighbors",)
    return _LLDP_COMMANDS


def _is_unsupported_lldp_output(payload: dict) -> bool:
    result = str(payload.get("result") or "").lower()
    status = int(payload.get("status_code") or 0) if str(payload.get("status_code") or "").isdigit() else 0
    if status == 422:
        return True
    return any(
        marker in result
        for marker in (
            "invalid input",
            "unrecognized command",
            "unrecognized command found",
            "incomplete command",
            "command not found",
            "error: unrecognized command",
        )
    )


def _ingest_cached_lldp_rows(command: str, result: str) -> list[tuple[str, str, str]]:
    cmd = command.lower()
    if cmd.startswith("display lldp neighbor brief") or cmd.startswith("show lldp neighbors"):
        return parse_lldp_brief(result)
    return []


def _probe_lldp_neighbors(
    *,
    device: str,
    question_number: int,
    tool_config: AgentToolConfig,
    denied: set[tuple[str, str]],
    fetched: set[tuple[str, str]],
    command_cache: dict[tuple[str, str], str],
    trace: AgentTrace,
    remaining_budget: int,
) -> tuple[list[tuple[str, str, str]], int]:
    spent = 0
    for command in _lldp_commands_for_device(device):
        if spent >= remaining_budget:
            break
        body_text = _execute_or_skip_command(
            tool_name="infra_maintenance",
            device=device,
            command=command,
            question_number=question_number,
            tool_config=tool_config,
            denied=denied,
            fetched=fetched,
            command_cache=command_cache,
            trace=trace,
            error_prefix="deterministic prepass error",
        )
        trace.transcript.append({
            "role": "tool",
            "name": "infra_maintenance",
            "deterministic_prepass": True,
            "content_len": len(body_text),
        })
        spent += 1
        payload = _parse_tool_body_json(body_text)
        rows = _ingest_cached_lldp_rows(str(payload.get("command") or ""), str(payload.get("result") or ""))
        if rows:
            return rows, spent
        if not _is_unsupported_lldp_output(payload):
            return [], spent
    return [], spent


def _parse_tool_body_json(body_text: str) -> dict:
    try:
        payload = json.loads(body_text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _question_prefers_config_port_spelling(question_text: str) -> bool:
    q = question_text.lower()
    return (
        "follow the interface name in the node's configuration" in q
        or "follow the interface name from the display current-configuration" in q
        or "full name gigabitethernet instead of ge" in q
    )


def _render_has_abbreviated_ports(rendered: str) -> bool:
    return bool(
        rendered
        and (
            "_GE" in rendered
            or "(GE" in rendered
            or "_10GE" in rendered
            or "(10GE" in rendered
        )
    )


def _build_deterministic_path_state(
    *,
    question_text: str,
    command_cache: dict[tuple[str, str], str],
) -> tuple[PathQuestionSpec, PathEvidenceGraph, list[list[str]]]:
    spec = parse_path_question_spec(question_text)
    solver = PathEvidenceGraph()
    for body in command_cache.values():
        payload = _parse_tool_body_json(body)
        command = str(payload.get("command") or "")
        device = str(payload.get("device_name") or "")
        result = str(payload.get("result") or "")
        if not device or not result:
            continue
        cmd = command.lower()
        if _ingest_cached_lldp_rows(command, result):
            solver.ingest_lldp(local_device=device, lldp_output=result)
        elif cmd.startswith("display current-configuration"):
            solver.ingest_port_inventory(device=device, content=result, source="current_config")
        elif cmd.startswith("display interface brief"):
            solver.ingest_port_inventory(device=device, content=result, source="interface_brief")
    paths: list[list[str]] = []
    best_pair_index: int | None = None
    best_len: int | None = None
    for pair_index, (source, destination) in enumerate(_candidate_path_search_pairs(spec)):
        candidate_paths = solver.find_paths(source=source, destination=destination)
        candidate_paths = [path for path in candidate_paths if len(path) >= 2]
        if not candidate_paths:
            continue
        candidate_len = len(candidate_paths[0])
        if (
            best_pair_index is None
            or pair_index < best_pair_index
            or (pair_index == best_pair_index and (best_len is None or candidate_len < best_len))
        ):
            paths = candidate_paths
            best_pair_index = pair_index
            best_len = candidate_len
    return spec, solver, paths


def _deterministic_path_from_cache(
    *,
    question_text: str,
    command_cache: dict[tuple[str, str], str],
    require_intermediate_interfaces: bool,
    forbid_final_interface: bool,
) -> str:
    spec, solver, paths = _build_deterministic_path_state(
        question_text=question_text,
        command_cache=command_cache,
    )
    if not paths:
        return ""
    rendered = solver.render_paths(paths=paths, spec=spec)
    if not rendered:
        return ""
    rep = validate_path(
        rendered,
        require_intermediate_interfaces=require_intermediate_interfaces,
        forbid_final_interface=forbid_final_interface,
    )
    return rep.normalised if rep.is_valid else ""


def _enrich_deterministic_path_port_names(
    *,
    current_rendered: str,
    phase: int,
    question_number: int,
    question_text: str,
    limits: AgentLimits,
    tool_config: AgentToolConfig,
    denied: set[tuple[str, str]],
    fetched: set[tuple[str, str]],
    command_cache: dict[tuple[str, str], str],
    trace: AgentTrace,
    require_intermediate_interfaces: bool,
    forbid_final_interface: bool,
) -> str:
    if not _question_prefers_config_port_spelling(question_text):
        return current_rendered
    spec, _solver, paths = _build_deterministic_path_state(
        question_text=question_text,
        command_cache=command_cache,
    )
    if not paths:
        return current_rendered
    rendered = current_rendered
    if not _render_has_abbreviated_ports(rendered):
        return rendered
    remaining = max(0, limits.max_tool_calls - trace.tool_calls_made)
    if remaining == 0:
        return rendered
    nodes_to_probe: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for idx, node in enumerate(path):
            if idx == len(path) - 1 and not spec.require_final_interface:
                continue
            if node not in seen:
                seen.add(node)
                nodes_to_probe.append(node)
    for node in nodes_to_probe[: min(4, remaining)]:
        body_text = _execute_or_skip_command(
            tool_name="infra_maintenance",
            device=node,
            command="display current-configuration",
            question_number=question_number,
            tool_config=tool_config,
            denied=denied,
            fetched=fetched,
            command_cache=command_cache,
            trace=trace,
            error_prefix="deterministic config enrichment error",
        )
        trace.transcript.append({
            "role": "tool",
            "name": "infra_maintenance",
            "deterministic_config_enrichment": True,
            "content_len": len(body_text),
        })
    rendered = _deterministic_path_from_cache(
        question_text=question_text,
        command_cache=command_cache,
        require_intermediate_interfaces=require_intermediate_interfaces,
        forbid_final_interface=forbid_final_interface,
    )
    if not _render_has_abbreviated_ports(rendered):
        return rendered
    remaining = max(0, limits.max_tool_calls - trace.tool_calls_made)
    if remaining == 0:
        return rendered
    for node in nodes_to_probe[: min(4, remaining)]:
        body_text = _execute_or_skip_command(
            tool_name="l2_link",
            device=node,
            command="display interface brief",
            question_number=question_number,
            tool_config=tool_config,
            denied=denied,
            fetched=fetched,
            command_cache=command_cache,
            trace=trace,
            error_prefix="deterministic interface enrichment error",
        )
        trace.transcript.append({
            "role": "tool",
            "name": "l2_link",
            "deterministic_interface_enrichment": True,
            "content_len": len(body_text),
        })
    return _deterministic_path_from_cache(
        question_text=question_text,
        command_cache=command_cache,
        require_intermediate_interfaces=require_intermediate_interfaces,
        forbid_final_interface=forbid_final_interface,
    )


def _run_deterministic_path_prepass(
    *,
    phase: int,
    question_number: int,
    question_text: str,
    limits: AgentLimits,
    tool_config: AgentToolConfig,
    denied: set[tuple[str, str]],
    fetched: set[tuple[str, str]],
    command_cache: dict[tuple[str, str], str],
    trace: AgentTrace,
    require_intermediate_interfaces: bool,
    forbid_final_interface: bool,
) -> str:
    if not _is_deterministic_path_target(phase=phase, question_number=question_number):
        return ""
    spec: PathQuestionSpec = parse_path_question_spec(question_text)
    if not spec.source_node or not spec.destination_node:
        return ""

    is_bank22 = "network bank22" in question_text.lower()
    queued = _initial_path_probe_nodes(spec, question_text)

    queried: set[str] = set()
    prepass_cap = 6 if is_bank22 else (6 if any(_looks_like_endpoint(node or "") for node in (spec.source_node, spec.destination_node)) else 4)
    prepass_budget = min(prepass_cap, max(0, limits.max_tool_calls - trace.tool_calls_made))

    while queued and prepass_budget > 0:
        device = queued.pop(0)
        if device in queried:
            continue
        queried.add(device)
        rows, spent = _probe_lldp_neighbors(
            device=device,
            question_number=question_number,
            tool_config=tool_config,
            denied=denied,
            fetched=fetched,
            command_cache=command_cache,
            trace=trace,
            remaining_budget=prepass_budget,
        )
        prepass_budget -= spent
        for _local_if, _nbr_if, nbr_dev in rows:
            if nbr_dev not in queried and nbr_dev not in queued:
                if _is_transit_priority_node(nbr_dev, spec):
                    queued.insert(0, nbr_dev)
                else:
                    queued.append(nbr_dev)
        deterministic = _deterministic_path_from_cache(
            question_text=question_text,
            command_cache=command_cache,
            require_intermediate_interfaces=require_intermediate_interfaces,
            forbid_final_interface=forbid_final_interface,
        )
        if deterministic:
            deterministic = _enrich_deterministic_path_port_names(
                current_rendered=deterministic,
                phase=phase,
                question_number=question_number,
                question_text=question_text,
                limits=limits,
                tool_config=tool_config,
                denied=denied,
                fetched=fetched,
                command_cache=command_cache,
                trace=trace,
                require_intermediate_interfaces=require_intermediate_interfaces,
                forbid_final_interface=forbid_final_interface,
            )
            trace.transcript.append({
                "role": "assistant",
                "validator_action": "accept:deterministic_prepass",
                "draft_answer": deterministic,
            })
            return deterministic

    return ""


def _run_path_scenario(
    *,
    scenario_id: str,
    question_number: int,
    phase: int,
    question_text: str,
    parsed: ParsedConstraints,
    qwen: QwenClient,
    tool_config: AgentToolConfig,
    limits_path: str,
    limits: AgentLimits,
    fetched_seed: set[tuple[str, str]] | None,
    temperature: float | None,
    seed: int | None,
) -> AgentTrace:
    """Path-family agent loop. Format-only validator (no closed reason vocab)."""
    trace = AgentTrace(scenario_id=scenario_id)
    denied = load_denied_pairs(question_number, limits_path)
    fetched: set[tuple[str, str]] = set(fetched_seed or set())
    command_cache: dict[tuple[str, str], str] = {}
    strict_path_schema = _path_requires_interfaces(question_text)
    forbid_final_interface = _path_forbids_final_interface(question_text)

    sys_prompt = _build_path_system_prompt(parsed=parsed, denied=denied, phase=phase)
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_text},
    ]
    trace.transcript.append({"role": "system", "content_len": len(sys_prompt)})
    trace.transcript.append({"role": "user", "content_len": len(question_text)})

    deterministic = _run_deterministic_path_prepass(
        phase=phase,
        question_number=question_number,
        question_text=question_text,
        limits=limits,
        tool_config=tool_config,
        denied=denied,
        fetched=fetched,
        command_cache=command_cache,
        trace=trace,
        require_intermediate_interfaces=strict_path_schema,
        forbid_final_interface=forbid_final_interface,
    )
    if deterministic:
        trace.final_answer = deterministic
        trace.final_action = "accept"
        return trace

    for it in range(limits.max_iterations):
        trace.iterations = it + 1
        try:
            response = qwen.chat(
                messages,
                tools=list(TOOL_SPECS),
                tool_choice="auto",
                temperature=temperature,
                seed=seed,
            )
        except Exception as e:
            trace.errors.append(f"qwen call failed at it={it}: {e}")
            trace.final_action = "incomplete"
            return trace

        if response.has_tool_calls:
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            })
            trace.transcript.append({
                "role": "assistant",
                "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in response.tool_calls],
            })
            for tc in response.tool_calls:
                if trace.tool_calls_made >= limits.max_tool_calls:
                    trace.errors.append("tool call budget exhausted")
                    trace.final_action = "budget_exhausted"
                    return trace
                body_text = _execute_or_skip_tool_call(
                    tc=tc,
                    question_number=question_number,
                    tool_config=tool_config,
                    denied=denied,
                    fetched=fetched,
                    command_cache=command_cache,
                    trace=trace,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": body_text,
                })
                trace.transcript.append({
                    "role": "tool",
                    "name": tc.name,
                    "content_len": len(body_text),
                })
                deterministic = _deterministic_path_from_cache(
                    question_text=question_text,
                    command_cache=command_cache,
                    require_intermediate_interfaces=strict_path_schema,
                    forbid_final_interface=forbid_final_interface,
                )
                if deterministic:
                    deterministic = _enrich_deterministic_path_port_names(
                        current_rendered=deterministic,
                        phase=phase,
                        question_number=question_number,
                        question_text=question_text,
                        limits=limits,
                        tool_config=tool_config,
                        denied=denied,
                        fetched=fetched,
                        command_cache=command_cache,
                        trace=trace,
                        require_intermediate_interfaces=strict_path_schema,
                        forbid_final_interface=forbid_final_interface,
                    )
                    trace.final_answer = deterministic
                    trace.final_action = "accept"
                    return trace
            continue

        draft = response.content or ""
        rep = validate_path(
            draft,
            require_intermediate_interfaces=strict_path_schema,
            forbid_final_interface=forbid_final_interface,
        )
        rc = response.reasoning_content
        trace.transcript.append({
            "role": "assistant",
            "draft_answer": draft,
            "validator_action": "accept" if rep.is_valid else "reemit_format",
            "finish_reason": response.finish_reason,
            "reasoning_content_len": (len(rc) if rc else 0),
            "reasoning_content_head": (rc[:600] if rc else ""),
        })
        if rep.is_valid:
            trace.final_answer = rep.normalised
            trace.final_action = "accept"
            return trace
        if response.finish_reason == "length":
            forced = _force_path_final_answer(
                qwen=qwen,
                messages=messages,
                trace=trace,
                label="forced_finalize:length",
                temperature=temperature,
                seed=seed,
                require_intermediate_interfaces=strict_path_schema,
                forbid_final_interface=forbid_final_interface,
            )
            if forced:
                trace.final_answer = forced
                trace.final_action = "accept"
                return trace
        trace.format_rejections += 1
        messages.append({"role": "assistant", "content": draft})
        messages.append({
            "role": "user",
            "content": (
                "Validator: " + rep.hint_for_reemit()
                + "\nRe-emit the path answer using ONLY the schema the question specified."
                " No prose, no markdown, no surrounding text. One path per line."
                " Use '->' to join hops. The destination node must appear last with no trailing interface."
            ),
        })

    forced = _force_path_final_answer(
        qwen=qwen,
        messages=messages,
        trace=trace,
        label="forced_finalize:terminal",
        temperature=temperature,
        seed=seed,
        require_intermediate_interfaces=strict_path_schema,
        forbid_final_interface=forbid_final_interface,
    )
    if forced:
        trace.final_answer = forced
        trace.final_action = "accept"
        return trace
    trace.final_action = "incomplete"
    return trace


def _build_topology_system_prompt(
    *,
    parsed: ParsedConstraints,
    denied: set[tuple[str, str]],
    phase: int,
) -> str:
    parts: list[str] = [
        "You are NetOps-Agent, a topology-reconstruction agent on a multi-vendor network."
        " Discover missing physical links using the four skill tools."
        " Output ONLY topology links in the exact schema requested by the question:"
        " local_node(local_port)->remote_node(remote_port). No prose or markdown.",
        "Topology playbook:\n"
        "1. Use `display lldp neighbor brief` on candidate devices to find directly connected peers.\n"
        "2. Confirm port names with `display current-configuration` or `display interface brief`.\n"
        "3. Use only real interface names from command output; strip rate/bandwidth annotations.\n"
        "4. Emit one link per line. Do not duplicate a bidirectional link in both directions unless the question asks for both.",
    ]
    if parsed.fault_candidate_nodes:
        parts.append("Candidate nodes from question: " + ", ".join(parsed.fault_candidate_nodes))
    if phase == 2:
        parts.append(
            "Phase 2 known device vocabulary (use only these names; do not invent):\n"
            + ", ".join(PHASE_2_DEVICES)
        )
    if denied:
        parts.append(
            "Hard-blocked (device, command) pairs — do NOT call these:\n"
            + "\n".join(f"  - {d} : {c}" for d, c in sorted(denied))
        )
    return "\n\n".join(parts)


def _force_topology_final_answer(
    *,
    qwen: QwenClient,
    messages: list[dict],
    trace: AgentTrace,
    label: str,
    temperature: float | None,
    seed: int | None,
) -> str:
    compact_evidence: list[str] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        try:
            payload = json.loads(str(msg.get("content") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        device = str(payload.get("device_name") or "")
        command = str(payload.get("command") or "")
        result = str(payload.get("result") or payload.get("content") or "")[:500]
        compact_evidence.append(f"- {device} | {command}\n{result}")
    response = qwen.chat(
        [
            messages[0],
            messages[1],
            {
                "role": "user",
                "content": (
                    "Collected evidence:\n"
                    + "\n\n".join(compact_evidence[-12:])
                    + "\n\nStop calling tools. Emit ONLY topology links now, one per line, "
                    "as local_node(local_port)->remote_node(remote_port)."
                ),
            },
        ],
        tools=[],
        temperature=temperature,
        seed=seed,
    )
    draft = response.content or ""
    rep = validate_topology(draft)
    trace.transcript.append({
        "role": "assistant",
        "draft_answer": draft,
        "validator_action": f"{label}:{'accept' if rep.is_valid else 'reemit_format'}",
        "finish_reason": response.finish_reason,
    })
    return rep.normalised if rep.is_valid else ""


def _run_topology_scenario(
    *,
    scenario_id: str,
    question_number: int,
    phase: int,
    question_text: str,
    parsed: ParsedConstraints,
    qwen: QwenClient,
    tool_config: AgentToolConfig,
    limits_path: str,
    limits: AgentLimits,
    fetched_seed: set[tuple[str, str]] | None,
    temperature: float | None,
    seed: int | None,
) -> AgentTrace:
    trace = AgentTrace(scenario_id=scenario_id)
    denied = load_denied_pairs(question_number, limits_path)
    fetched: set[tuple[str, str]] = set(fetched_seed or set())
    command_cache: dict[tuple[str, str], str] = {}
    sys_prompt = _build_topology_system_prompt(parsed=parsed, denied=denied, phase=phase)
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_text},
    ]
    trace.transcript.append({"role": "system", "content_len": len(sys_prompt)})
    trace.transcript.append({"role": "user", "content_len": len(question_text)})

    for it in range(limits.max_iterations):
        trace.iterations = it + 1
        try:
            response = qwen.chat(
                messages,
                tools=list(TOOL_SPECS),
                tool_choice="auto",
                temperature=temperature,
                seed=seed,
            )
        except Exception as e:
            trace.errors.append(f"qwen call failed at it={it}: {e}")
            trace.final_action = "incomplete"
            return trace

        if response.has_tool_calls:
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            })
            trace.transcript.append({
                "role": "assistant",
                "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in response.tool_calls],
            })
            for tc in response.tool_calls:
                if trace.tool_calls_made >= limits.max_tool_calls:
                    trace.errors.append("tool call budget exhausted")
                    trace.final_action = "budget_exhausted"
                    return trace
                body_text = _execute_or_skip_tool_call(
                    tc=tc,
                    question_number=question_number,
                    tool_config=tool_config,
                    denied=denied,
                    fetched=fetched,
                    command_cache=command_cache,
                    trace=trace,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": body_text,
                })
                trace.transcript.append({
                    "role": "tool",
                    "name": tc.name,
                    "content_len": len(body_text),
                })
            continue

        draft = response.content or ""
        rep = validate_topology(draft)
        rc = response.reasoning_content
        trace.transcript.append({
            "role": "assistant",
            "draft_answer": draft,
            "validator_action": "accept" if rep.is_valid else "reemit_format",
            "finish_reason": response.finish_reason,
            "reasoning_content_len": (len(rc) if rc else 0),
            "reasoning_content_head": (rc[:600] if rc else ""),
        })
        if rep.is_valid:
            trace.final_answer = rep.normalised
            trace.final_action = "accept"
            return trace
        if response.finish_reason == "length":
            forced = _force_topology_final_answer(
                qwen=qwen,
                messages=messages,
                trace=trace,
                label="forced_finalize:length",
                temperature=temperature,
                seed=seed,
            )
            if forced:
                trace.final_answer = forced
                trace.final_action = "accept"
                return trace
        trace.format_rejections += 1
        messages.append({"role": "assistant", "content": draft})
        messages.append({
            "role": "user",
            "content": (
                "Validator: " + rep.hint_for_reemit()
                + "\nRe-emit ONLY topology links as local_node(local_port)->remote_node(remote_port). "
                "One link per line. No prose, markdown, or surrounding text."
            ),
        })

    forced = _force_topology_final_answer(
        qwen=qwen,
        messages=messages,
        trace=trace,
        label="forced_finalize:terminal",
        temperature=temperature,
        seed=seed,
    )
    if forced:
        trace.final_answer = forced
        trace.final_action = "accept"
        return trace
    trace.final_action = "incomplete"
    return trace


def _path_requires_interfaces(question_text: str) -> bool:
    q = question_text.lower()
    return (
        "outbound interface" in q
        or "outbound-interface" in q
        or "outbound port" in q
        or "outbound-port" in q
        or "physical outbound interface" in q
    )


def _is_permission_denied(result: CommandResult) -> bool:
    body = " ".join(
        str(x)
        for x in (
            result.status,
            result.status_code,
            result.result_text,
            result.raw.get("error") if isinstance(result.raw, dict) else "",
            result.raw.get("message") if isinstance(result.raw, dict) else "",
        )
    ).lower()
    return result.status_code == 403 or "no permission" in body or "permission denied" in body


def _denied_tool_body(device: str, command: str) -> str:
    return json.dumps({
        "status": "skipped_denied",
        "device_name": device,
        "command": command,
        "result": "Skipped locally because this (device, command) pair is denied for the scenario.",
    }, ensure_ascii=False)


def _execute_or_skip_command(
    *,
    tool_name: str,
    device: str,
    command: str,
    question_number: int,
    tool_config: AgentToolConfig,
    denied: set[tuple[str, str]],
    fetched: set[tuple[str, str]],
    command_cache: dict[tuple[str, str], str] | None,
    trace: AgentTrace,
    error_prefix: str,
) -> str:
    pair = (device, command)
    if command_cache is not None and pair in command_cache:
        return command_cache[pair]
    if pair in denied:
        trace.errors.append(f"skipped denied tool call: {device} | {command}")
        body = _denied_tool_body(device, command)
        if command_cache is not None:
            command_cache[pair] = body
        return body
    try:
        result = dispatch_tool_call(
            tool_name=tool_name,
            arguments={"device_name": device, "command": command},
            question_number=question_number,
            config=tool_config,
        )
        trace.tool_calls_made += 1
        fetched.add(pair)
        if _is_permission_denied(result):
            denied.add(pair)
            trace.errors.append(f"learned denied tool call: {device} | {command}")
        body = _format_tool_result(result)
        if command_cache is not None:
            command_cache[pair] = body
        return body
    except AgentToolError as e:
        trace.errors.append(f"{error_prefix}: {e}")
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)


def _execute_or_skip_tool_call(
    *,
    tc,
    question_number: int,
    tool_config: AgentToolConfig,
    denied: set[tuple[str, str]],
    fetched: set[tuple[str, str]],
    command_cache: dict[tuple[str, str], str] | None,
    trace: AgentTrace,
) -> str:
    device = str(tc.arguments.get("device_name", ""))
    command = str(tc.arguments.get("command", ""))
    return _execute_or_skip_command(
        tool_name=tc.name,
        device=device,
        command=command,
        question_number=question_number,
        tool_config=tool_config,
        denied=denied,
        fetched=fetched,
        command_cache=command_cache,
        trace=trace,
        error_prefix=f"tool error on {tc.name}",
    )


def _format_tool_result(result: CommandResult) -> str:
    return json.dumps({
        "status": result.status,
        "device_name": result.device_name,
        "vendor": result.vendor,
        "command": result.command,
        "result": result.result_text[:6000],  # safety cap on prompt growth
    }, ensure_ascii=False)


# ---- Tool routing helper for follow-ups -----------------------------------

# Validator-emitted follow-ups don't carry a tool name; we map command
# prefixes back to the matching skill domain so the conversation stays
# tagged consistently with the rest of the flow.
_PREFIX_TO_TOOL: tuple[tuple[str, str], ...] = (
    ("display current-configuration", "infra_maintenance"),
    ("display logbuffer", "infra_maintenance"),
    ("display alarm", "infra_maintenance"),
    ("display memory", "infra_maintenance"),
    ("display lldp", "infra_maintenance"),
    ("display interface", "l2_link"),
    ("display ip interface", "l2_link"),
    ("display eth-trunk", "l2_link"),
    ("display vlan", "l2_link"),
    ("display mac-address", "l2_link"),
    ("display stp", "l2_link"),
    ("display ip routing-table", "l3_route"),
    ("display arp", "l3_route"),
    ("display ipv6", "l3_route"),
    ("display ip vpn-instance", "l3_route"),
    ("display ospf", "l3_route"),
    ("display bgp", "l3_route"),
    ("display vxlan", "adv_tunnel"),
    ("display vrrp", "adv_tunnel"),
    ("display bfd", "adv_tunnel"),
    ("display ip pool", "adv_tunnel"),
    ("display srv6", "adv_tunnel"),
    ("display segment-routing", "adv_tunnel"),
)


def _pick_tool_for_command(command: str) -> str:
    cl = command.lower()
    for prefix, tool in _PREFIX_TO_TOOL:
        if cl.startswith(prefix.lower()):
            return tool
    return "infra_maintenance"  # safest default
