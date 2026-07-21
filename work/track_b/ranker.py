"""Deterministic weighted ranker for (device, fault_reason) candidates.

Combines the upstream signals from Steps 2–5 into per-candidate component
scores, an aggregated combined_score, and a within-scenario rank. Each
component is emitted as `{entity, score, score_norm}` per the
`mutation_discovery/gnn/driver_scorer.py` template so Step 8's XGBoost
layer can consume the same matrix without re-engineering features.

Component scores
----------------

    graph_centrality      — device's L2 betweenness rank percentile
    path_relevance        — on parsed source-destination shortest path,
                             plus inverse-hop bonus
    protocol_match        — fault_reason aligns with the suspected
                             protocol families parsed from the question
    vendor_compat         — playbook has at least one command for the
                             device's vendor (default Huawei)
    anomaly_prior         — anomaly miner emitted this (node, reason)
                             with given evidence_strength
    permission_survivor   — primary diagnostic command for this reason is
                             not on the per-question denial list
    disclosed_match       — reason is a disclosed_fault_category, or
                             node is a disclosed_fault_node, or node is
                             in fault_candidate_nodes
    contradiction_penalty — node is in `blacklisted_nodes`

The combined score is a weighted sum of the normalised components. The
weights are exposed as a `RankerWeights` dataclass and tunable per
ablation. Phase 2 and Phase 3 should run with the same weights unless
local eval shows otherwise (per the plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .constraint_parser import ParsedConstraints
from .playbook import lookup as playbook_lookup, all_entries as playbook_all


# ---- Fault-reason → protocol-family map -----------------------------------

# Each canonical Phase 2 reason is associated with the protocol families
# whose mention in the question text counts as evidence the reason is
# relevant. Multi-family entries (NAT involves both NAT and the firewall
# zone model) are explicit. A reason with no family stays neutral.
_REASON_PROTOCOLS: dict[str, tuple[str, ...]] = {
    "BGP configuration error": ("BGP", "MP-BGP"),
    "OSPF configuration error": ("OSPF",),
    "ISIS configuration error": ("ISIS",),
    "VXLAN configuration error": ("VXLAN",),
    "L3VPN configuration error": ("L3VPN", "MP-BGP"),
    "L2VPN configuration error": ("L2VPN", "EVPN"),
    "SRV6-Policy tunnel planning error": ("SRV6",),
    "ARP configuration error": ("ARP",),
    "Layer 3 loop": ("OSPF", "BGP"),
    "blackhole route": (),
    "missing static route": (),
    "static route error": (),
    "loopback interface IP configuration conflict": ("OSPF", "ISIS", "BGP"),
    "NAT external interface attribute configuration error or configuration missing": ("NAT",),
    "NAT internal interface attribute configuration error or missing": ("NAT",),
    "global STP not enabled": ("STP",),
    "IP address prefix list missing corresponding user source IP address": (),
    "global HRP hot redundancy protocol not enabled": ("HRP",),
    "security policy rule not permitting corresponding users": (),
    "shutdown": (),
    "interface IP error": (),
    "traffic occupying port bandwidth": (),
    "MAC address configuration error": (),
    "VPN configuration missing": ("L3VPN", "L2VPN"),
    "MTU value configuration error": (),
    "host information collection function missing": ("DHCP",),
    "interface VLAN configuration error": ("VLAN",),
    "port STP not enabled": ("STP",),
}


@dataclass(frozen=True)
class Candidate:
    scenario_id: str
    question_number: int
    phase: int
    node: str
    fault_reason: str
    category: str  # 'routing' | 'port'


@dataclass
class ComponentScores:
    graph_centrality: float = 0.0
    path_relevance: float = 0.0
    protocol_match: float = 0.0
    vendor_compat: float = 0.0
    anomaly_prior: float = 0.0
    permission_survivor: float = 1.0
    disclosed_match: float = 0.0
    contradiction_penalty: float = 0.0


@dataclass
class RankerWeights:
    graph_centrality: float = 0.5
    path_relevance: float = 1.5
    protocol_match: float = 1.0
    vendor_compat: float = 0.5
    anomaly_prior: float = 2.0
    permission_survivor: float = 0.5
    disclosed_match: float = 1.5
    contradiction_penalty: float = 3.0  # multiplied by negative penalty


_DEFAULT_WEIGHTS = RankerWeights()


@dataclass
class ScoredCandidate:
    candidate: Candidate
    raw: ComponentScores
    norm: ComponentScores
    combined_score: float = 0.0
    rank: int = -1


# ---- Per-component scorers (raw scores) ------------------------------------

def _score_graph_centrality(*, betweenness_norm: float | None) -> float:
    if betweenness_norm is None:
        return 0.0
    return max(0.0, min(1.0, float(betweenness_norm)))


def _score_path_relevance(
    *,
    on_path: int,
    hop_src: int,
    hop_dst: int,
) -> float:
    """1.0 if on shortest path; tapered bonus for nearness to either end."""
    if on_path:
        return 1.0
    score = 0.0
    if hop_src is not None and hop_src >= 0:
        score = max(score, max(0.0, 1.0 - 0.25 * hop_src))
    if hop_dst is not None and hop_dst >= 0:
        score = max(score, max(0.0, 1.0 - 0.25 * hop_dst))
    return score


def _score_protocol_match(
    *,
    fault_reason: str,
    suspected_protocols: tuple[str, ...],
) -> float:
    if not suspected_protocols:
        return 0.0
    expected = _REASON_PROTOCOLS.get(fault_reason, ())
    if not expected:
        return 0.0
    return 1.0 if any(p in suspected_protocols for p in expected) else 0.0


def _score_vendor_compat(
    *,
    fault_reason: str,
    category: str,
    vendor: str = "huawei",
) -> float:
    e = playbook_lookup(fault_reason, category)
    if e is None:
        return 0.0
    return 1.0 if e.commands.get(vendor.lower()) else 0.0


def _score_anomaly_prior(
    *,
    evidence_strength_num: int,
) -> float:
    """0/1/2/3 → 0.0/0.5/0.75/1.0 (low/medium/high)."""
    if evidence_strength_num <= 0:
        return 0.0
    if evidence_strength_num >= 3:
        return 1.0
    if evidence_strength_num == 2:
        return 0.75
    return 0.5


def _score_permission_survivor(
    *,
    denied_command_count: int,
) -> float:
    """1.0 when no denials touch this device; tapered down by denial count.

    The pruner upstream filters explicit (device, command) denials per call.
    This component is a soft signal: a device with many per-question
    denials is partially-blind, weakening confidence in any reason
    pinned on it.
    """
    return 1.0 / (1.0 + 0.5 * max(0, denied_command_count))


def _score_disclosed_match(
    *,
    node: str,
    fault_reason: str,
    parsed: ParsedConstraints,
) -> float:
    score = 0.0
    if fault_reason in parsed.disclosed_fault_categories:
        score = max(score, 1.0)
    if node in parsed.disclosed_fault_nodes:
        score = max(score, 1.0)
    if node in parsed.fault_candidate_nodes:
        score = max(score, 0.75)
    if fault_reason == "VRRP-dual-master" and "VRRP-dual-master" in parsed.disclosed_fault_categories:
        # Map the disclosed VRRP cue to several candidate reasons that
        # cause dual-master symptoms (interface VLAN config error or
        # global STP not enabled, depending on the diagnosis).
        score = max(score, 1.0)
    return score


def _score_contradiction_penalty(
    *,
    node: str,
    parsed: ParsedConstraints,
) -> float:
    """Returns a non-negative magnitude; combine_scores subtracts it."""
    return 1.0 if node in parsed.blacklisted_nodes else 0.0


# ---- Normalisation (within-scenario rank percentile) ----------------------

def _rank_norm_field(scored: list[ScoredCandidate], field_name: str) -> None:
    """Set the `norm.<field>` of each scored candidate to its rank percentile.

    Ties get the average rank. Computes in-place. If all values are equal
    (degenerate scenario) the percentile collapses to 0.5 for everyone.
    """
    pairs = [(i, getattr(s.raw, field_name)) for i, s in enumerate(scored)]
    pairs.sort(key=lambda kv: kv[1])
    n = len(pairs)
    if n == 0:
        return
    if n == 1:
        setattr(scored[0].norm, field_name, 0.5)
        return
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        pct = avg_rank / (n - 1)
        for k in range(i, j + 1):
            idx = pairs[k][0]
            setattr(scored[idx].norm, field_name, pct)
        i = j + 1


# ---- Candidate pool builder -----------------------------------------------

def build_candidate_pool(
    *,
    scenario_id: str,
    question_number: int,
    phase: int,
    devices: Iterable[str],
    parsed: ParsedConstraints,
    on_path_devices: set[str],
    anomaly_set: set[tuple[str, str, str]],
    fault_vocab: tuple[tuple[str, ...], tuple[str, ...]],
    extra_relevance_devices: set[str] | None = None,
    centrality_top_k: int = 12,
    centrality_norm: dict[str, float] | None = None,
) -> list[Candidate]:
    """Construct the (node, fault_reason) candidate pool for one scenario.

    The pool merges:
        - every (device, reason) the anomaly miner emitted
        - every reason crossed with on-parsed-path devices, disclosed nodes,
          fault-candidate nodes, blacklisted nodes (kept so contradiction
          penalty applies; XGBoost can still learn to demote them), and the
          top-k centrality devices.

    Permission denials at the diagnostic-command level do NOT prune the
    pool here; the `permission_survivor` component represents that signal
    via score, leaving the candidate visible to the LLM and to ablations.
    """
    routing_reasons, port_reasons = fault_vocab
    relevance_devices: set[str] = set()
    relevance_devices.update(on_path_devices)
    relevance_devices.update(parsed.disclosed_fault_nodes)
    relevance_devices.update(parsed.fault_candidate_nodes)
    relevance_devices.update(parsed.blacklisted_nodes)
    if extra_relevance_devices:
        relevance_devices.update(extra_relevance_devices)
    if centrality_norm:
        ordered = sorted(centrality_norm.items(), key=lambda kv: -kv[1])
        for name, _bn in ordered[:centrality_top_k]:
            relevance_devices.add(name)

    seen: set[tuple[str, str, str]] = set()
    out: list[Candidate] = []

    def add(node: str, reason: str, category: str) -> None:
        key = (node, reason, category)
        if key in seen:
            return
        seen.add(key)
        out.append(
            Candidate(
                scenario_id=scenario_id,
                question_number=question_number,
                phase=phase,
                node=node,
                fault_reason=reason,
                category=category,
            )
        )

    # 1. Anomaly miner candidates
    for s_id, node, reason in anomaly_set:
        if s_id != scenario_id:
            continue
        e = playbook_lookup(reason)
        cat = e.category if e else "routing"
        add(node, reason, cat)

    # 2. Relevance-set × full vocab
    for node in relevance_devices:
        if node not in set(devices):
            continue
        for r in routing_reasons:
            add(node, r, "routing")
        for r in port_reasons:
            add(node, r, "port")
    return out


# ---- Per-candidate scoring loop -------------------------------------------

def score_candidates(
    candidates: list[Candidate],
    *,
    parsed: ParsedConstraints,
    graph_features: dict[str, dict],  # node -> dict of features
    anomaly_priors: dict[tuple[str, str, str], int],  # (sid, node, reason) -> strength_num
    weights: RankerWeights = _DEFAULT_WEIGHTS,
    vendor: str = "huawei",
) -> list[ScoredCandidate]:
    scored: list[ScoredCandidate] = []
    for c in candidates:
        gf = graph_features.get(c.node, {})
        raw = ComponentScores(
            graph_centrality=_score_graph_centrality(
                betweenness_norm=gf.get("betweenness_norm"),
            ),
            path_relevance=_score_path_relevance(
                on_path=int(gf.get("on_parsed_path", 0)),
                hop_src=int(gf.get("hop_distance_source", -1)),
                hop_dst=int(gf.get("hop_distance_dest", -1)),
            ),
            protocol_match=_score_protocol_match(
                fault_reason=c.fault_reason,
                suspected_protocols=parsed.suspected_protocol_families,
            ),
            vendor_compat=_score_vendor_compat(
                fault_reason=c.fault_reason,
                category=c.category,
                vendor=vendor,
            ),
            anomaly_prior=_score_anomaly_prior(
                evidence_strength_num=anomaly_priors.get(
                    (c.scenario_id, c.node, c.fault_reason), 0
                ),
            ),
            permission_survivor=_score_permission_survivor(
                denied_command_count=int(gf.get("denied_command_count", 0)),
            ),
            disclosed_match=_score_disclosed_match(
                node=c.node, fault_reason=c.fault_reason, parsed=parsed,
            ),
            contradiction_penalty=_score_contradiction_penalty(
                node=c.node, parsed=parsed,
            ),
        )
        scored.append(
            ScoredCandidate(candidate=c, raw=raw, norm=ComponentScores(), combined_score=0.0)
        )
    if not scored:
        return scored
    for fld in (
        "graph_centrality",
        "path_relevance",
        "protocol_match",
        "vendor_compat",
        "anomaly_prior",
        "permission_survivor",
        "disclosed_match",
        "contradiction_penalty",
    ):
        _rank_norm_field(scored, fld)
    # Combine using normalised components for transferability across networks
    for s in scored:
        s.combined_score = (
            weights.graph_centrality * s.norm.graph_centrality
            + weights.path_relevance * s.norm.path_relevance
            + weights.protocol_match * s.norm.protocol_match
            + weights.vendor_compat * s.norm.vendor_compat
            + weights.anomaly_prior * s.norm.anomaly_prior
            + weights.permission_survivor * s.norm.permission_survivor
            + weights.disclosed_match * s.norm.disclosed_match
            - weights.contradiction_penalty * s.raw.contradiction_penalty
        )
    scored.sort(key=lambda s: -s.combined_score)
    for i, s in enumerate(scored):
        s.rank = i
    return scored


# ---- Output schema --------------------------------------------------------

def scored_to_row(s: ScoredCandidate) -> dict:
    out: dict = {
        "scenario_id": s.candidate.scenario_id,
        "question_number": s.candidate.question_number,
        "phase": s.candidate.phase,
        "node": s.candidate.node,
        "fault_reason": s.candidate.fault_reason,
        "category": s.candidate.category,
        "combined_score": s.combined_score,
        "rank": s.rank,
    }
    for fld in (
        "graph_centrality",
        "path_relevance",
        "protocol_match",
        "vendor_compat",
        "anomaly_prior",
        "permission_survivor",
        "disclosed_match",
        "contradiction_penalty",
    ):
        out[f"{fld}_raw"] = getattr(s.raw, fld)
        out[f"{fld}_norm"] = getattr(s.norm, fld)
    return out


SCORED_FIELDS: tuple[str, ...] = (
    "scenario_id", "question_number", "phase", "node", "fault_reason", "category",
    "combined_score", "rank",
    "graph_centrality_raw", "graph_centrality_norm",
    "path_relevance_raw", "path_relevance_norm",
    "protocol_match_raw", "protocol_match_norm",
    "vendor_compat_raw", "vendor_compat_norm",
    "anomaly_prior_raw", "anomaly_prior_norm",
    "permission_survivor_raw", "permission_survivor_norm",
    "disclosed_match_raw", "disclosed_match_norm",
    "contradiction_penalty_raw", "contradiction_penalty_norm",
)
