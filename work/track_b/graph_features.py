"""Per-(scenario, device) graph-feature extractor.

Consumes a `ScenarioGraph` and a `ParsedConstraints` (from Step 2) and
emits the following features per device, one row per (scenario_id, device):

    degree                — undirected L2 device-degree
    degree_norm           — within-scenario rank percentile (0..1)
    betweenness           — networkx betweenness on the L2 device subgraph
    betweenness_norm      — within-scenario rank percentile
    n_l3_ifaces           — count of interfaces with an IPv4 address
    has_loopback_ip       — 0/1
    n_vrrp_groups         — count of VRRP groups the device participates in
    n_vrf                 — count of VPN-instances on the device
    on_parsed_path        — 0/1, was the device identified as on the parsed
                             source→destination path
    hop_distance_source   — shortest-path hops from parsed source device,
                             -1 if unreachable or unresolved
    hop_distance_dest     — shortest-path hops from parsed destination
                             device, -1 if unreachable or unresolved
    is_blacklisted        — 0/1, device was named in `Limitation: Do not
                             look for faults on X.`
    is_disclosed_fault    — 0/1, device named in an explicit fault disclosure
                             (e.g., VRRP dual-master)
    denied_command_count  — count of (device, command) denials in
                             question_limits_config.json for this device

Features are scenario-relative (rank percentiles, not absolute values)
where applicable to transfer across networks (Phase 1 → Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import networkx as nx

from .topology import ScenarioGraph, device_l2_subgraph
from .constraint_parser import ParsedConstraints
from .permission_pruner import denied_pairs


@dataclass
class DeviceFeatures:
    scenario_id: str
    question_number: int
    phase: int
    node: str
    degree: int = 0
    degree_norm: float = 0.0
    betweenness: float = 0.0
    betweenness_norm: float = 0.0
    n_l3_ifaces: int = 0
    has_loopback_ip: int = 0
    n_vrrp_groups: int = 0
    n_vrf: int = 0
    on_parsed_path: int = 0
    hop_distance_source: int = -1
    hop_distance_dest: int = -1
    is_blacklisted: int = 0
    is_disclosed_fault: int = 0
    denied_command_count: int = 0


def _resolve_device_for_endpoint(
    name_or_ip: str | None,
    sg: ScenarioGraph,
) -> str | None:
    """Best-effort: map a parsed endpoint (device name OR IP OR host alias)
    to a device node in the graph.

    Tries exact device-name match, then IP→device map, then no resolution.
    Symbolic host aliases (e.g., GUEST_WIFI_CLIENT01, GoogleWebServer01)
    are NOT in the device graph; they resolve to None and downstream
    features fall back to "unresolved" sentinels.
    """
    if not name_or_ip:
        return None
    if name_or_ip in sg.graph:
        return name_or_ip
    # Try device by name
    for n, data in sg.graph.nodes(data=True):
        if data.get("type") == "device" and data.get("name") == name_or_ip:
            return data["name"]
    # IP lookup
    return sg.ip_to_device.get(name_or_ip)


def _rank_norm(values: dict[str, float]) -> dict[str, float]:
    """Within-set rank percentile in [0, 1]; ties get the average rank."""
    if not values:
        return {}
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    out: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        pct = avg_rank / max(n - 1, 1)
        for k in range(i, j + 1):
            out[items[k][0]] = pct
        i = j + 1
    return out


def extract_device_features(
    sg: ScenarioGraph,
    parsed: ParsedConstraints,
    *,
    question_limits_path: Path,
) -> list[DeviceFeatures]:
    """Compute features for every device in the scenario graph."""
    l2 = device_l2_subgraph(sg.graph)
    devices = list(l2.nodes)

    degree_raw = {d: l2.degree(d) for d in devices}
    if l2.number_of_nodes() > 1 and l2.number_of_edges() > 0:
        bw_raw = nx.betweenness_centrality(l2, normalized=True)
    else:
        bw_raw = {d: 0.0 for d in devices}
    degree_norm = _rank_norm({d: float(v) for d, v in degree_raw.items()})
    bw_norm = _rank_norm(bw_raw)

    src_dev = _resolve_device_for_endpoint(parsed.source_endpoint, sg)
    dst_dev = _resolve_device_for_endpoint(parsed.target_destination_node, sg)
    if dst_dev is None and parsed.target_destination_ip:
        dst_dev = sg.ip_to_device.get(parsed.target_destination_ip)
    if dst_dev is None and parsed.target_destination_host:
        dst_dev = _resolve_device_for_endpoint(parsed.target_destination_host, sg)

    src_dist: dict[str, int] = {}
    dst_dist: dict[str, int] = {}
    if src_dev and src_dev in l2:
        src_dist = nx.single_source_shortest_path_length(l2, src_dev)
    if dst_dev and dst_dev in l2:
        dst_dist = nx.single_source_shortest_path_length(l2, dst_dev)

    on_path: set[str] = set()
    if src_dev and dst_dev and src_dev in l2 and dst_dev in l2:
        try:
            for path in nx.all_shortest_paths(l2, src_dev, dst_dev):
                on_path.update(path)
        except nx.NetworkXNoPath:
            pass

    blacklist = set(parsed.blacklisted_nodes)
    disclosed = set(parsed.disclosed_fault_nodes)

    denials = denied_pairs(sg.question_number, question_limits_path)
    denials_per_device: dict[str, int] = {}
    for device, _command in denials:
        denials_per_device[device] = denials_per_device.get(device, 0) + 1

    out: list[DeviceFeatures] = []
    for device in devices:
        summary = sg.device_summaries.get(device, {})
        out.append(
            DeviceFeatures(
                scenario_id=sg.scenario_id,
                question_number=sg.question_number,
                phase=sg.phase,
                node=device,
                degree=degree_raw.get(device, 0),
                degree_norm=degree_norm.get(device, 0.0),
                betweenness=bw_raw.get(device, 0.0),
                betweenness_norm=bw_norm.get(device, 0.0),
                n_l3_ifaces=summary.get("n_l3_ifaces", 0),
                has_loopback_ip=int(summary.get("has_loopback_ip", False)),
                n_vrrp_groups=len(summary.get("vrrp_groups", [])),
                n_vrf=len(summary.get("vrf_names", [])),
                on_parsed_path=int(device in on_path),
                hop_distance_source=src_dist.get(device, -1),
                hop_distance_dest=dst_dist.get(device, -1),
                is_blacklisted=int(device in blacklist),
                is_disclosed_fault=int(device in disclosed),
                denied_command_count=denials_per_device.get(device, 0),
            )
        )
    return out


FEATURE_FIELDS = tuple(
    f for f in DeviceFeatures.__dataclass_fields__.keys()
)


def feature_to_row(f: DeviceFeatures) -> dict:
    return asdict(f)
