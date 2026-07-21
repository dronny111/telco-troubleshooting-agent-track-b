"""Typed offline-graph builder for Track B.

Walks `devices_outputs/{question_number}/{device}/*.txt` for one scenario
and constructs a networkx graph with typed nodes and edges:

    Nodes
        device       — one per device folder
        interface    — one per device-interface pair (e.g., (Beta-Aegis-01, GE1/0/0))
        vlan         — one per VLAN id observed in display_vlan
        vrrp_group   — one per VRRP group id observed in display_vrrp_verbose
        vrf          — one per vpn-instance observed in display_ip_vpn-instance

    Edges
        device -has_interface-> interface
        interface -l2_adjacent-> interface     (LLDP brief)
        device -has_vrrp-> vrrp_group          (vrrp_verbose)
        device -has_vrf-> vrf                  (vpn-instance)
        interface -in_vlan-> vlan              (vlan-id parsed from interface stanza)

The graph also carries auxiliary mappings the feature extractor and live
agent both consume:

    ip_to_device   : str -> str   IPv4 → owning device, derived from
                                  display_ip_interface_brief
    ip_to_interface: str -> tuple (device, interface) for the same source

The same parser code is intended to run on either offline static files or
online API responses: every parsing function accepts a string of CLI output
and returns structured data, with no I/O dependency.

Phase scoping: only Phase 1 has local bundles. Phase 2 builds the same
graph from API responses at inference time using the same parsers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import networkx as nx

from .anomaly_miner import command_to_filename


# ---- LLDP brief ------------------------------------------------------------

# Sample lines:
#   GE1/0/0                       101  GE1/0/4                       Beta-Portal-01
#   Ethernet1/0/0                SH_Core  Ethernet1/0/1            120
#   SZ_AR                        Gi0/0/0  120       R              Gi0/0/1
# Whitespace-aligned columns vary by platform, so parse line-by-line.
_LLDP_LINE = re.compile(r"^\s*(\S+)\s+\d+\s+(\S+)\s+(\S+)\s*$")
_LLDP_SPLIT = re.compile(r"\s{2,}")


def parse_lldp_brief(content: str) -> list[tuple[str, str, str]]:
    """Return [(local_iface, neighbor_iface, neighbor_device), ...]."""
    out: list[tuple[str, str, str]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if (
            lower.startswith(("<", "#", "capability codes", "total entries"))
            or "local intf" in lower
            or "neighbor dev" in lower
            or "device id" in lower
            or "port id" in lower
        ):
            continue
        if set(line) == {"-"}:
            continue

        local = nbr_iface = nbr_dev = ""
        m = _LLDP_LINE.match(line)
        if m:
            local, nbr_iface, nbr_dev = m.group(1), m.group(2), m.group(3)
        else:
            cols = [part.strip() for part in _LLDP_SPLIT.split(line) if part.strip()]
            if len(cols) < 4:
                continue
            if cols[-1].isdigit():
                # Huawei AR live output: local | neighbor dev | neighbor intf | exptime
                local, nbr_dev, nbr_iface = cols[0], cols[1], cols[2]
            elif cols[2].isdigit():
                # Cisco-like output: neighbor dev | local intf | hold-time | ... | port id
                nbr_dev, local, nbr_iface = cols[0], cols[1], cols[-1]
            else:
                continue

        if local.lower() in {"local", "interface"} or nbr_dev.lower() in {"device", "exptime(s)"}:
            continue
        if any(set(token) == {"-"} for token in (local, nbr_iface, nbr_dev)):
            continue
        out.append((local, nbr_iface, nbr_dev))
    return out


# ---- IP interface brief ---------------------------------------------------

# Sample line:
#   GE1/0/0                     192.168.65.178/30  up       up       --
#   MEth0/0/0                   unassigned         up       down     --
_IFACE_BRIEF_LINE = re.compile(
    r"^\s*(\S+)\s+(\S+)\s+(?:\*?down|up|administratively\s+down|\!down|\^down)\b",
    re.MULTILINE,
)
_IPV4_MASK = re.compile(r"^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})/(\d{1,2})$")


@dataclass(frozen=True)
class InterfaceRecord:
    name: str
    ip: str | None  # plain IPv4 without mask, or None if unassigned
    mask: int | None
    physical_state: str | None = None  # "up" | "down" | "*down" | etc.


def parse_ip_interface_brief(content: str) -> list[InterfaceRecord]:
    out: list[InterfaceRecord] = []
    for line in content.splitlines():
        m = re.match(r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*", line)
        if not m:
            continue
        name, addr, phys, _proto = m.group(1), m.group(2), m.group(3), m.group(4)
        if name.lower() in {"interface", "the", "*down:", "!down:", "^down:", "(l):", "(s):", "(d):", "(ed):"}:
            continue
        if addr.lower() in {"ip", "address/mask"}:
            continue
        if addr.lower() == "unassigned":
            out.append(InterfaceRecord(name=name, ip=None, mask=None, physical_state=phys))
            continue
        m2 = _IPV4_MASK.match(addr)
        if m2:
            out.append(
                InterfaceRecord(
                    name=name,
                    ip=m2.group(1),
                    mask=int(m2.group(2)),
                    physical_state=phys,
                )
            )
    return out


# ---- VRRP verbose ---------------------------------------------------------

_VRRP_GROUP_HEADER = re.compile(
    r"^\s*(\S+)\s*\|\s*Virtual\s+Router\s*(\d+)",
    re.MULTILINE | re.IGNORECASE,
)
# Tolerant fallback: "VRID : 1" or "Virtual Router 1"
_VRID_LINE = re.compile(r"\bVRID\s*:\s*(\d+)\b|\bVirtual\s+Router\s+(\d+)\b")
_VRRP_STATE = re.compile(r"\bState\s*:\s*(MASTER|BACKUP|INITIALIZE|UP|DOWN|Master|Backup|Initialize)\b")


@dataclass(frozen=True)
class VrrpRecord:
    group_id: int
    state: str | None  # MASTER / BACKUP / INITIALIZE / unknown


def parse_vrrp_verbose(content: str) -> list[VrrpRecord]:
    if "VRRP does not exist" in content:
        return []
    out: list[VrrpRecord] = []
    # Split into per-group blocks by VRID lines; keep nearest state.
    blocks: list[tuple[int, str]] = []
    last_pos = 0
    last_id: int | None = None
    for m in _VRID_LINE.finditer(content):
        if last_id is not None:
            blocks.append((last_id, content[last_pos:m.start()]))
        gid = int(m.group(1) or m.group(2))
        last_id = gid
        last_pos = m.start()
    if last_id is not None:
        blocks.append((last_id, content[last_pos:]))
    seen: set[int] = set()
    for gid, blk in blocks:
        if gid in seen:
            continue
        seen.add(gid)
        sm = _VRRP_STATE.search(blk)
        state = sm.group(1).upper() if sm else None
        out.append(VrrpRecord(group_id=gid, state=state))
    return out


# ---- IP VPN-instance ------------------------------------------------------

_VPN_INSTANCE_NAME = re.compile(
    r"\bVPN-Instance\s+(?:Name(?:\s+and\s+ID)?\s*:\s*)?(\S+)",
    re.IGNORECASE,
)


def parse_vpn_instances(content: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for m in _VPN_INSTANCE_NAME.finditer(content):
        n = m.group(1).strip().rstrip(",")
        if n and n.lower() not in {"name", "id"} and n not in seen:
            seen.add(n)
            names.append(n)
    return names


# ---- Graph construction ---------------------------------------------------

def _read(device_dir: Path, command: str) -> str | None:
    f = device_dir / command_to_filename(command)
    if not f.is_file():
        return None
    try:
        return f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def add_device_to_graph(
    g: nx.MultiDiGraph,
    device: str,
    device_dir: Path,
    *,
    ip_to_device: dict[str, str],
    ip_to_interface: dict[str, tuple[str, str]],
) -> dict:
    """Add one device's nodes and intra-device edges. Inter-device edges
    (LLDP) are added in a second pass once every device is known.

    Returns a small per-device summary used by the feature extractor and
    the prompt context.
    """
    g.add_node(("device", device), type="device", name=device)
    summary: dict = {
        "interfaces": [],
        "lldp_neighbors": [],
        "vrrp_groups": [],
        "vrf_names": [],
        "n_l3_ifaces": 0,
        "has_loopback_ip": False,
    }

    iface_brief = _read(device_dir, "display ip interface brief")
    if iface_brief:
        records = parse_ip_interface_brief(iface_brief)
        for r in records:
            iface_node = ("interface", device, r.name)
            g.add_node(iface_node, type="interface", device=device, name=r.name,
                       ip=r.ip, mask=r.mask, physical_state=r.physical_state)
            g.add_edge(("device", device), iface_node, type="has_interface")
            summary["interfaces"].append(r)
            if r.ip:
                ip_to_device.setdefault(r.ip, device)
                ip_to_interface.setdefault(r.ip, (device, r.name))
                summary["n_l3_ifaces"] += 1
                if r.name.lower().startswith("loopback"):
                    summary["has_loopback_ip"] = True

    vrrp_content = _read(device_dir, "display vrrp verbose")
    if vrrp_content:
        for v in parse_vrrp_verbose(vrrp_content):
            grp_node = ("vrrp_group", v.group_id)
            g.add_node(grp_node, type="vrrp_group", group_id=v.group_id)
            g.add_edge(("device", device), grp_node, type="has_vrrp", state=v.state)
            summary["vrrp_groups"].append(v)

    vpn_content = _read(device_dir, "display ip vpn-instance")
    if vpn_content:
        for name in parse_vpn_instances(vpn_content):
            vrf_node = ("vrf", name)
            g.add_node(vrf_node, type="vrf", name=name)
            g.add_edge(("device", device), vrf_node, type="has_vrf")
            summary["vrf_names"].append(name)

    lldp = _read(device_dir, "display lldp neighbor brief")
    if lldp:
        summary["lldp_neighbors"] = parse_lldp_brief(lldp)

    return summary


def add_lldp_edges(
    g: nx.MultiDiGraph,
    device: str,
    neighbors: list[tuple[str, str, str]],
) -> None:
    """Second pass: connect interfaces across devices via L2 adjacency.

    A LLDP neighbor row says "my local interface L sees neighbor device D
    on its interface I". We add an edge (device, L) -[l2_adjacent]-> (D, I).
    The corresponding device may not have its own LLDP block (asymmetry);
    we still add the edge so the graph reflects observed state.
    """
    for local, nbr_iface, nbr_device in neighbors:
        local_node = ("interface", device, local)
        nbr_node = ("interface", nbr_device, nbr_iface)
        if local_node not in g:
            g.add_node(local_node, type="interface", device=device, name=local)
            g.add_edge(("device", device), local_node, type="has_interface")
        if nbr_node not in g:
            g.add_node(nbr_node, type="interface", device=nbr_device, name=nbr_iface)
            if ("device", nbr_device) not in g:
                g.add_node(("device", nbr_device), type="device", name=nbr_device)
            g.add_edge(("device", nbr_device), nbr_node, type="has_interface")
        g.add_edge(local_node, nbr_node, type="l2_adjacent")


@dataclass
class ScenarioGraph:
    scenario_id: str
    question_number: int
    phase: int
    graph: nx.MultiDiGraph
    ip_to_device: dict[str, str] = field(default_factory=dict)
    ip_to_interface: dict[str, tuple[str, str]] = field(default_factory=dict)
    device_summaries: dict[str, dict] = field(default_factory=dict)


def build_scenario_graph(
    *,
    scenario_id: str,
    question_number: int,
    phase: int,
    devices_outputs_root: Path,
) -> ScenarioGraph | None:
    """Build a typed graph for one scenario. Returns None if no bundle."""
    bundle = devices_outputs_root / str(question_number)
    if not bundle.is_dir():
        return None
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    ip_to_device: dict[str, str] = {}
    ip_to_interface: dict[str, tuple[str, str]] = {}
    device_summaries: dict[str, dict] = {}
    for device_dir in sorted(bundle.iterdir()):
        if not device_dir.is_dir():
            continue
        device = device_dir.name
        device_summaries[device] = add_device_to_graph(
            g, device, device_dir,
            ip_to_device=ip_to_device,
            ip_to_interface=ip_to_interface,
        )
    for device, summ in device_summaries.items():
        add_lldp_edges(g, device, summ["lldp_neighbors"])
    return ScenarioGraph(
        scenario_id=scenario_id,
        question_number=question_number,
        phase=phase,
        graph=g,
        ip_to_device=ip_to_device,
        ip_to_interface=ip_to_interface,
        device_summaries=device_summaries,
    )


# ---- Device-level views over the typed graph -------------------------------

def device_l2_subgraph(g: nx.MultiDiGraph) -> nx.Graph:
    """Project the multi-typed graph to a simple device→device L2 graph
    suitable for centrality / shortest-path queries.

    For every l2_adjacent interface edge u_iface -> v_iface, add an
    undirected device-device edge u_dev <-> v_dev (deduplicated).
    """
    out = nx.Graph()
    for n, data in g.nodes(data=True):
        if data.get("type") == "device":
            out.add_node(data["name"])
    for u, v, data in g.edges(data=True):
        if data.get("type") != "l2_adjacent":
            continue
        if u[0] != "interface" or v[0] != "interface":
            continue
        u_dev = u[1]
        v_dev = v[1]
        if u_dev != v_dev:
            out.add_edge(u_dev, v_dev)
    return out
