"""Build the compact JSON context block embedded in Qwen's system prompt.

The block is the only place where graph- and ranker-derived signals enter
the LLM. The plan caps the block at ~10 lines / ≤500 tokens so the rest
of the prompt budget can hold the few-shot exemplars (Step 1) and the
question text. Graph state is NEVER dumped — only the distilled signals.

Top-K selection prefers the XGBoost calibrated score and uncertainty when
they are present; otherwise falls back to the deterministic ranker's
combined_score. The block also surfaces:
    - parsed constraints (source, destination, blacklisted, suspected
      protocols, disclosed categories)
    - top-3 candidate (node, fault_reason) hypotheses with one-line
      evidence pointers from the anomaly miner
    - top-5 next-best diagnostic commands per playbook
    - hard-blocked (device, command) pairs from question_limits_config

`build_system_prompt` wraps the JSON in a deterministic preamble carrying
the three Track B Update rules ("most specific and closest fault cause",
path nodes including L2, port-name source) and the Phase 2 device
vocabulary so Qwen doesn't hallucinate device identifiers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from .constraint_parser import ParsedConstraints
from .playbook import lookup as playbook_lookup

__all__ = [
    "AnomalyEvidence",
    "PHASE_2_DEVICES",
    "RankedRow",
    "build_context",
    "build_system_prompt",
    "approx_token_count",
    "_FAULT_EXEMPLAR",
    "_PATH_EXEMPLAR",
    "_TOPOLOGY_EXEMPLAR",
]


# Phase 2 known device vocabulary (from TRACK_UPDATES.md). Embedded in the
# system prompt to suppress device-name hallucinations.
PHASE_2_DEVICES: tuple[str, ...] = (
    "AGG_SW_01", "AGG_SW_02", "AGG_SW_03", "AGG_SW_04",
    "BJHQ_CSR1000V_GW_01", "BaiduWebServer01", "ChinaUnicom_SW",
    "Core_SW_01", "Core_SW_02",
    "EMPLOYEE_WIFI_CLIENT01", "EMPLOYEE_WIFI_CLIENT02", "EMPLOYEE_WIFI_CLIENT03",
    "FW_01", "FW_02",
    "GUEST_WIFI_CLIENT01", "GUEST_WIFI_CLIENT02", "GUEST_WIFI_CLIENT03",
    "GoogleWebServer01", "HQ-DHCP-Server", "HQ_DNS_Server_01",
    "HQ_FIN_Client01", "HQ_FIN_PC01", "HQ_FTP_Server_01",
    "HQ_HR_AP01", "HQ_HR_PC01", "HQ_HTTP_Server_01",
    "HQ_MKT_AP01", "HQ_MKT_Client01", "HQ_MKT_PC01",
    "HQ_PROC_AP01", "HQ_PROC_PC01", "Internet_PC01",
    "Outside_FTP_Client01", "PE1", "PE2", "PE3",
    "SH_AR", "SH_Core", "SH_FAC_PC01", "SH_SAL_PC01", "SH_STO_PC01",
    "SW-DMZ-ACC-01", "SZ_AR", "SZ_Core",
    "SZ_Server_Cluster1", "SZ_Server_Cluster2", "SZ_Server_Cluster3",
)


@dataclass(frozen=True)
class RankedRow:
    """Minimal subset the context builder consumes per candidate."""
    scenario_id: str
    node: str
    fault_reason: str
    category: str
    combined_score: float
    calibrated_score: float | None = None
    uncertainty: float | None = None


@dataclass(frozen=True)
class AnomalyEvidence:
    scenario_id: str
    node: str
    fault_reason: str
    sample_evidence: str
    signatures_fired: str = ""


def _select_top_hypotheses(
    candidates: Sequence[RankedRow],
    *,
    k: int,
) -> list[RankedRow]:
    """Top-K by calibrated_score if any candidate carries it, else combined_score.

    Deduplicates by (node, fault_reason). Stable across reruns: secondary
    sort by node then fault_reason for deterministic ties.
    """
    if not candidates:
        return []
    use_calibrated = any(c.calibrated_score is not None for c in candidates)
    def key(c: RankedRow) -> tuple:
        primary = -(c.calibrated_score if use_calibrated and c.calibrated_score is not None else c.combined_score)
        # Secondary: tighter uncertainty wins ties when calibrated
        unc = c.uncertainty if c.uncertainty is not None else 0.0
        return (primary, unc, c.node, c.fault_reason)
    seen: set[tuple[str, str]] = set()
    out: list[RankedRow] = []
    for c in sorted(candidates, key=key):
        if (c.node, c.fault_reason) in seen:
            continue
        seen.add((c.node, c.fault_reason))
        out.append(c)
        if len(out) >= k:
            break
    return out


def _next_best_commands(
    hypotheses: Iterable[RankedRow],
    *,
    blacklisted_nodes: Iterable[str],
    denied_pairs: Iterable[tuple[str, str]] = (),
    vendor: str = "huawei",
    k: int = 5,
) -> list[dict]:
    """For each top hypothesis, list the playbook's primary diagnostic
    command for the device. Skip blacklisted or denied (device, command)
    pairs.
    """
    bl = set(blacklisted_nodes)
    denied = set(denied_pairs)
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for h in hypotheses:
        if h.node in bl:
            continue
        e = playbook_lookup(h.fault_reason, h.category)
        if not e:
            continue
        for cmd in e.commands.get(vendor, ()):
            pair = (h.node, cmd)
            if pair in seen or pair in denied:
                continue
            seen.add(pair)
            out.append({"device": h.node, "command": cmd, "for_reason": h.fault_reason})
            if len(out) >= k:
                return out
    return out


def build_context(
    *,
    task_family: str,
    parsed: ParsedConstraints,
    candidates: Sequence[RankedRow],
    anomaly_evidence: dict[tuple[str, str, str], AnomalyEvidence] | None = None,
    denied_pairs: Iterable[tuple[str, str]] = (),
    top_hypotheses_k: int = 3,
    next_best_commands_k: int = 5,
    vendor: str = "huawei",
) -> dict:
    """Return the compact context block as a plain dict (JSON-ready).

    Caller serialises with `json.dumps(..., separators=(',', ':'))` for the
    Qwen prompt; pretty-printed JSON is reserved for inspection only.
    """
    anomaly_evidence = anomaly_evidence or {}
    top = _select_top_hypotheses(candidates, k=top_hypotheses_k)
    blocks: list[dict] = []
    for h in top:
        ev = anomaly_evidence.get((h.scenario_id, h.node, h.fault_reason))
        block: dict = {
            "node": h.node,
            "reason": h.fault_reason,
            "category": h.category,
        }
        if h.calibrated_score is not None:
            block["score"] = round(h.calibrated_score, 3)
            if h.uncertainty is not None:
                block["unc"] = round(h.uncertainty, 3)
        else:
            block["score"] = round(h.combined_score, 3)
        if ev and ev.sample_evidence:
            block["evidence"] = ev.sample_evidence[:160]
        blocks.append(block)

    next_cmds = _next_best_commands(
        top,
        blacklisted_nodes=parsed.blacklisted_nodes,
        denied_pairs=denied_pairs,
        vendor=vendor,
        k=next_best_commands_k,
    )

    constraints = {
        "source": parsed.source_endpoint or None,
        "destination_ip": parsed.target_destination_ip or None,
        "destination_host": parsed.target_destination_host or None,
        "destination_node": parsed.target_destination_node or None,
        "blacklisted_nodes": list(parsed.blacklisted_nodes),
        "disclosed_categories": list(parsed.disclosed_fault_categories),
        "suspected_protocols": list(parsed.suspected_protocol_families),
        "fault_candidate_nodes": list(parsed.fault_candidate_nodes),
    }

    hard_blocked = [
        {"device": d, "command": c} for d, c in sorted(set(denied_pairs))
    ]

    return {
        "task_family": task_family,
        "constraints": constraints,
        "top_hypotheses": blocks,
        "next_best_commands": next_cmds,
        "hard_blocked": hard_blocked,
    }


# ---- System prompt assembly -----------------------------------------------

# ---- Few-shot exemplars -------------------------------------------------------
# One per task family. Kept compact (< 200 tokens each) to stay within the
# prompt budget. These are derived from known-correct live-run answers on the
# Phase 2 network; tool sequences reflect the actual diagnostic playbook.

_FAULT_EXEMPLAR = """Task family: fault
Scenario: HQ employee cannot reach internet (8.8.8.8). Routing + port faults possible.
Diagnostic steps taken:
  l3_route(Core_SW_01, "display ip routing-table 8.8.8.8")  → next-hop 10.0.0.1 via GE1/0/5 (FW_01 side)
  l3_route(FW_01,      "display ip routing-table 8.8.8.8")  → default route 0.0.0.0/0 to ISP ✓ (routing OK)
  infra_maintenance(FW_01, "display current-configuration") → security-policy permits source 10.2.0.0/16 only;
      HQ_MKT subnet 10.1.0.0/16 is NOT matched → traffic silently dropped
Final answer:
FW_01;8.8.8.8;security policy rule not permitting corresponding users"""

_PATH_EXEMPLAR = """Task family: path
Scenario: Path from HQ_MKT_PC01 to BaiduWebServer01. Format: node_outbound-interface->...->destination-node (no trailing interface on destination).
Diagnostic steps taken:
  infra_maintenance(AGG_SW_01, "display lldp neighbor brief")       → uplink to Core_SW_01 via GE0/0/1
  infra_maintenance(Core_SW_01, "display lldp neighbor brief")      → uplink to BJHQ_CSR1000V_GW_01 via GE1/0/24
  infra_maintenance(BJHQ_CSR1000V_GW_01, "display lldp neighbor brief") → uplink to FW_01 via GigabitEthernet0/0/0
  l3_route(FW_01, "display ip routing-table 114.114.114.114")       → default route → ISP egress
Final answer:
HQ_MKT_PC01_GE0/0/1->AGG_SW_01_GE0/0/1->Core_SW_01_GE1/0/24->BJHQ_CSR1000V_GW_01_GigabitEthernet0/0/0->FW_01_GE1/0/0->BaiduWebServer01"""

_TOPOLOGY_EXEMPLAR = """Task family: topology
Scenario: Supplement the UP link connections of Core_SW_01. Format: local_node(local_port)->remote_node(remote_port)
Diagnostic steps taken:
  infra_maintenance(Core_SW_01, "display lldp neighbor brief") → GE1/0/1↔AGG_SW_01 GE0/0/1, GE1/0/5↔FW_01 GE0/0/2
Final answer:
Core_SW_01(GE1/0/1)->AGG_SW_01(GE0/0/1)
Core_SW_01(GE1/0/5)->FW_01(GE0/0/2)"""


_TRACK_B_RULES = (
    "Track B output rules:\n"
    "1. Topology link tasks: use the interface name from `display current-configuration`."
    " If unavailable, use the interface name from `display interface brief`. Strip"
    " trailing rate/bandwidth annotations.\n"
    "2. Path tasks: include every node on the path (including L2-only hops). One"
    " path per line; use `->` to connect node names; no extra whitespace.\n"
    "3. Fault tasks: pick the MOST SPECIFIC and CLOSEST fault cause. Output schema"
    " `node;destination-IP;reason` (routing) or `node;port;reason` (port). One"
    " fault per line. No blank lines, no extra whitespace, English ASCII only.\n"
    "4. Use only commands on the simulator's regex whitelist. Avoid duplicate"
    " queries. Stop and emit the answer once evidence is sufficient."
)


def build_system_prompt(
    *,
    context: dict,
    few_shot_exemplars: Sequence[str] = (),
    include_phase_2_device_list: bool = False,
) -> str:
    """Assemble the full Qwen system prompt.

    Layout:
        1. Role + Track B rules.
        2. (optional) Phase 2 device vocabulary — for Phase 2 inference.
        3. Compact JSON context (the Step 9 product).
        4. (optional) Few-shot exemplars from the past competition zip.
    """
    parts: list[str] = [
        "You are NetOps-Agent, an evidence-driven network-troubleshooting agent."
        " Answer using only data collected via the tools. Do not reveal reasoning."
        " Output only the final answer in the exact required schema.",
        _TRACK_B_RULES,
    ]
    if include_phase_2_device_list:
        parts.append(
            "Phase 2 known device vocabulary (use only these names; do not invent):\n"
            + ", ".join(PHASE_2_DEVICES)
        )
    parts.append(
        "Compact context (top hypotheses, parsed constraints, recommended next"
        " commands, hard-denied pairs):\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
    for ex in few_shot_exemplars:
        parts.append("Example:\n" + ex.strip())
    return "\n\n".join(parts)


# ---- Token-budget helpers --------------------------------------------------

def approx_token_count(text: str) -> int:
    """A rough ~4-chars-per-token estimate. Qwen's tokeniser will be more
    accurate; this is just for prompt-budget guards in eval reporting."""
    return max(1, len(text) // 4)
