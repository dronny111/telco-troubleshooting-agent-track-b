"""Answer-feasibility validator + uncertainty-gated follow-up planner.

Sits after the LLM drafts an answer. Returns one of four decisions:

    accept              — schema, graph-feasibility, evidence, and (when
                          XGBoost is in) confidence all pass. Ship the
                          normalised answer.

    reemit_format       — format guard rejected (schema / vocab / whitespace
                          / ASCII). Same draft re-prompted with the guard's
                          corrective hint. NO API call.

    reemit_constraint   — answer is well-formed but violates a parsed
                          constraint (proposed fault on a blacklisted or
                          unknown node, vendor mismatch like HRP on a
                          non-firewall). LLM is re-prompted with the
                          violation list. NO API call.

    fetch_evidence      — answer is well-formed and constraint-clean but
                          lacks anomaly evidence for the top fault, OR the
                          XGBoost calibrated score is too low / uncertainty
                          too high. Issue exactly ONE targeted follow-up
                          command (the playbook's primary diagnostic for
                          the top hypothesis), then re-prompt.

Strict policy: the validator triggers AT MOST ONE follow-up call per
question. Repeated re-emit cycles within the format/constraint loop are
allowed (they cost no API calls) but are capped by the agent harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .constraint_parser import ParsedConstraints
from .format_guard import ValidationReport, validate_fault
from .playbook import lookup as playbook_lookup


_FAULT_LINE = re.compile(r"^([^;]+);([^;]+);([^;]+)$")


@dataclass(frozen=True)
class FaultLine:
    raw: str
    node: str
    middle: str  # IPv4 (routing) or port id (port)
    reason: str
    category: str  # 'routing' | 'port'


@dataclass
class ValidatorDecision:
    action: str  # 'accept' | 'reemit_format' | 'reemit_constraint' | 'fetch_evidence'
    accepted_lines: list[FaultLine] = field(default_factory=list)
    format_report: ValidationReport | None = None
    format_hint: str = ""
    constraint_violations: list[str] = field(default_factory=list)
    follow_up: tuple[str, str, str] | None = None  # (device, command, for_reason)
    follow_up_rationale: str = ""
    normalised_answer: str = ""

    @property
    def accept(self) -> bool:
        return self.action == "accept"


# ---- Helpers --------------------------------------------------------------

def _is_routing_middle(s: str) -> bool:
    parts = s.split(".")
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(p) <= 255 for p in parts)
    except ValueError:
        return False


def _parse_lines(
    text: str,
    routing_vocab: frozenset[str],
    port_vocab: frozenset[str],
) -> list[FaultLine]:
    out: list[FaultLine] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = _FAULT_LINE.match(raw)
        if not m:
            continue
        node, mid, reason = m.group(1), m.group(2), m.group(3)
        if reason in routing_vocab:
            cat = "routing"
        elif reason in port_vocab:
            cat = "port"
        else:
            continue
        out.append(FaultLine(raw=raw, node=node, middle=mid, reason=reason, category=cat))
    return out


_NOT_FIREWALL_TOKENS: tuple[str, ...] = (
    "_sw_", "-sw-", "switch",
    "spine", "leaf",
    "router",
    "_agg_", "-agg-",
    "core_sw", "core-sw",
    "_csr", "-csr",
    "_ar_", "-ar-",
    "_dns_", "-dns-",
    "_dhcp_", "-dhcp-",
    "_ftp_", "-ftp-",
    "_http_", "-http-",
    "wifi_client",
    "webserver",
    "_pc0", "-pc0",
    "server_cluster", "server-cluster",
)


def _device_is_firewall(graph_features: dict, node: str) -> bool | None:
    """Best-effort firewall detection from graph features.

    Returns None when neither name pattern nor an explicit flag in
    graph_features is conclusive; callers should treat None as 'unknown'.
    """
    feats = (graph_features or {}).get(node, {}) or {}
    flag = feats.get("is_firewall")
    if flag in (1, True, "1", "True", "true"):
        return True
    if flag in (0, False, "0", "False", "false"):
        return False
    name_low = node.lower()
    if (
        name_low.startswith("fw")
        or name_low.startswith("usg")
        or "firewall" in name_low
    ):
        return True
    for tok in _NOT_FIREWALL_TOKENS:
        if tok in name_low:
            return False
    return None


def _pick_followup_command(
    *,
    line: FaultLine,
    denied_pairs: set[tuple[str, str]],
    vendor: str = "huawei",
) -> tuple[str, str] | None:
    e = playbook_lookup(line.reason, line.category)
    if not e:
        return None
    for cmd in e.commands.get(vendor, ()):
        if (line.node, cmd) in denied_pairs:
            continue
        return (line.node, cmd)
    return None


# ---- Public API ----------------------------------------------------------

def validate_answer(
    *,
    draft_answer: str,
    scenario_id: str,
    parsed: ParsedConstraints,
    routing_vocab: tuple[str, ...],
    port_vocab: tuple[str, ...],
    graph_features: dict[str, dict] | None = None,
    anomaly_evidence: set[tuple[str, str, str]] | None = None,
    denied_pairs: set[tuple[str, str]] | None = None,
    calibrated_scores: dict[tuple[str, str, str], tuple[float, float]] | None = None,
    tau_score: float = 0.5,
    tau_unc: float = 0.2,
    fetched_commands: set[tuple[str, str]] | None = None,
    vendor: str = "huawei",
) -> ValidatorDecision:
    """Decide whether to accept, re-emit, or fetch one follow-up command.

    Parameters
    ----------
    fetched_commands
        Set of (device, command) pairs already executed for this scenario.
        Used to ensure the proposed follow-up is not a repeat.
    calibrated_scores
        Optional XGBoost output keyed by (scenario_id, node, fault_reason)
        with (calibrated_score, uncertainty). Triggers a follow-up only
        when this dict is non-empty (Step 8 promotion gate path).
    """
    rv = frozenset(routing_vocab)
    pv = frozenset(port_vocab)
    denied_pairs = denied_pairs or set()
    anomaly_evidence = anomaly_evidence or set()
    fetched_commands = fetched_commands or set()
    graph_features = graph_features or {}

    # 1. Format guard
    fr = validate_fault(draft_answer, rv, pv)
    if not fr.is_valid:
        return ValidatorDecision(
            action="reemit_format",
            format_report=fr,
            format_hint=fr.hint_for_reemit(),
            normalised_answer=fr.normalised,
        )
    lines = _parse_lines(fr.normalised, rv, pv)
    if not lines:
        return ValidatorDecision(
            action="reemit_format",
            format_report=fr,
            format_hint="Output parsed as empty — emit at least one fault line.",
            normalised_answer=fr.normalised,
        )

    # 2. Per-line constraint feasibility
    constraint_violations: list[str] = []
    blacklist = set(parsed.blacklisted_nodes)
    known_nodes = set(graph_features.keys()) if graph_features else set()
    for ln in lines:
        if ln.node in blacklist:
            constraint_violations.append(
                f"{ln.node} is blacklisted by 'Limitation: Do not look for faults on {ln.node}'"
            )
            continue
        if known_nodes and ln.node not in known_nodes:
            constraint_violations.append(
                f"{ln.node} is not a known device in this scenario's network"
            )
            continue
        # Routing-line middle must be IPv4; port-line middle must NOT be IPv4
        is_ip = _is_routing_middle(ln.middle)
        if ln.category == "routing" and not is_ip:
            constraint_violations.append(
                f"line {ln.raw!r}: routing-fault reason but middle field is not an IPv4 address"
            )
            continue
        if ln.category == "port" and is_ip:
            constraint_violations.append(
                f"line {ln.raw!r}: port-fault reason but middle field is an IPv4 address"
            )
            continue
        # Vendor-specific coherence: HRP only on firewalls
        if ln.reason == "global HRP hot redundancy protocol not enabled":
            is_fw = _device_is_firewall(graph_features, ln.node)
            if is_fw is False:
                constraint_violations.append(
                    f"{ln.node}: HRP fault claimed on a non-firewall device"
                )

    if constraint_violations:
        return ValidatorDecision(
            action="reemit_constraint",
            accepted_lines=lines,
            constraint_violations=constraint_violations,
            normalised_answer=fr.normalised,
        )

    # 3. Evidence + confidence checks (single follow-up budget)
    follow_up: tuple[str, str, str] | None = None
    rationale = ""
    for ln in lines:
        key = (scenario_id, ln.node, ln.reason)
        # XGBoost confidence gate (only when calibrated_scores are populated)
        if calibrated_scores:
            cs = calibrated_scores.get(key)
            if cs is not None:
                calibrated, unc = cs
                if calibrated < tau_score or unc > tau_unc:
                    pick = _pick_followup_command(
                        line=ln, denied_pairs=denied_pairs, vendor=vendor,
                    )
                    if pick and pick not in fetched_commands:
                        follow_up = (pick[0], pick[1], ln.reason)
                        rationale = (
                            f"calibrated_score={calibrated:.2f}<{tau_score} "
                            f"or uncertainty={unc:.2f}>{tau_unc} on {ln.node}/{ln.reason}"
                        )
                        break
        # Evidence-missing gate (always-on)
        has_anom = key in anomaly_evidence
        playbook_entry = playbook_lookup(ln.reason, ln.category)
        primary_cmd: str | None = None
        if playbook_entry:
            cmds = playbook_entry.commands.get(vendor, ())
            if cmds:
                primary_cmd = cmds[0]
        cmd_already_run = (
            primary_cmd is not None
            and (ln.node, primary_cmd) in fetched_commands
        )
        if not has_anom and not cmd_already_run:
            pick = _pick_followup_command(
                line=ln, denied_pairs=denied_pairs, vendor=vendor,
            )
            if pick and pick not in fetched_commands:
                follow_up = (pick[0], pick[1], ln.reason)
                rationale = (
                    f"no anomaly evidence and primary diagnostic for "
                    f"{ln.node}/{ln.reason} not yet executed"
                )
                break

    if follow_up is not None:
        return ValidatorDecision(
            action="fetch_evidence",
            accepted_lines=lines,
            follow_up=follow_up,
            follow_up_rationale=rationale,
            normalised_answer=fr.normalised,
        )

    # 4. Accept
    return ValidatorDecision(
        action="accept",
        accepted_lines=lines,
        normalised_answer=fr.normalised,
    )
