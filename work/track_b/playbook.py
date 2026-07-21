"""Fault-catalog command playbook for Track B.

For each of the closed-vocabulary fault reasons (19 routing + 12 port for
Phase 2), this module precomputes:

    - The 2–3 highest-yield diagnostic commands per vendor.
    - Regex signatures that detect the fault from command output.
    - A coarse `evidence_strength` rating used by the deterministic ranker
      and the anomaly miner.

Signatures carry an `evidence_type`:

    "fault"            — match (or non-match for negative polarity) is direct
                         evidence of the fault. Anomaly miner treats these
                         as candidate-emitting.
    "feature_presence" — match indicates the device has the feature
                         configured. Useful for prompt context (telling the
                         LLM which devices to look at) but does NOT by
                         itself imply a fault. Anomaly miner ignores.

Negative-polarity signatures (e.g., "global STP not enabled") fire when the
regex does NOT match in any of the target command outputs. They must
declare `target_commands` and should also declare `device_filter_regex` so
the miner only runs them on devices where the feature is expected (e.g.,
HRP only on firewalls).

Phase 1 fault-vocab variants ("routing loop" vs "Layer 3 loop", "IS-IS"
vs "ISIS", "traffic congestion" vs "traffic occupying", etc.) map through
`PHASE_1_ALIASES` so a single playbook entry serves both phases.

Commands are aligned with the regex whitelist in
`telco_data/Track B/data/Phase_2/README.md`. Commands not on the whitelist
will be rejected by the simulator and waste calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Signature:
    name: str
    regex: str
    description: str
    target_commands: tuple[str, ...] = ()
    polarity: str = "positive"  # "positive" | "negative"
    evidence_type: str = "fault"  # "fault" | "feature_presence"
    device_filter_regex: str = ""  # if set, only run on devices whose
                                   # display current-configuration matches


@dataclass(frozen=True)
class FaultEntry:
    fault_reason: str
    category: str  # 'routing' | 'port'
    description: str
    commands: dict[str, tuple[str, ...]]  # vendor -> ordered list (highest yield first)
    signatures: tuple[Signature, ...]
    evidence_strength: str  # 'high' | 'medium' | 'low'


# ---- helper -----------------------------------------------------------------

_R = "routing"
_P = "port"


def _entry(
    *,
    reason: str,
    category: str,
    description: str,
    huawei: Iterable[str],
    cisco: Iterable[str],
    h3c: Iterable[str],
    signatures: Iterable[Signature],
    strength: str,
) -> FaultEntry:
    return FaultEntry(
        fault_reason=reason,
        category=category,
        description=description,
        commands={
            "huawei": tuple(huawei),
            "cisco": tuple(cisco),
            "h3c": tuple(h3c),
        },
        signatures=tuple(signatures),
        evidence_strength=strength,
    )


_F = "fault"
_FP = "feature_presence"


# ---- Phase 2 canonical playbook (19 routing + 12 port) ---------------------

_PLAYBOOK_LIST: tuple[FaultEntry, ...] = (
    # ---- ROUTING (19) ----
    _entry(
        reason="blackhole route",
        category=_R,
        description="A route exists but forwards traffic to NULL0 / Black Hole / REJECT, dropping packets silently.",
        huawei=("display ip routing-table", "display current-configuration | include ip route-static"),
        cisco=("show ip route", "show running-config"),
        h3c=("display ip routing-table", "display current-configuration"),
        signatures=(
            Signature("null0_next_hop", r"\bNULL0\b", "next-hop NULL0 in routing table",
                      target_commands=("display ip routing-table",), evidence_type=_F),
            Signature("reject_route", r"\bREJECT\b", "REJECT flag in routing table",
                      target_commands=("display ip routing-table",), evidence_type=_F),
            Signature("static_blackhole",
                      r"ip\s+route-static\s+\S+\s+\S+\s+(?:NULL0|Null0|null0)",
                      "static config blackholes the destination",
                      target_commands=("display current-configuration",), evidence_type=_F),
        ),
        strength="high",
    ),
    _entry(
        reason="missing static route",
        category=_R,
        description="No route exists for the destination prefix; the device drops with 'destination unreachable' or has no FIB entry.",
        huawei=("display ip routing-table", "display current-configuration | include ip route-static"),
        cisco=("show ip route", "show running-config"),
        h3c=("display ip routing-table", "display current-configuration"),
        signatures=(
            Signature("destination_unreachable",
                      r"(?:Destination|destination)\s+(?:net\s+)?unreachable",
                      "lookup yields no route",
                      target_commands=("display ip routing-table",), evidence_type=_F),
        ),
        strength="medium",
    ),
    _entry(
        reason="static route error",
        category=_R,
        description="A static route exists but with a wrong next-hop, wrong out-interface, or wrong destination prefix.",
        huawei=("display ip routing-table", "display current-configuration | include ip route-static"),
        cisco=("show ip route", "show running-config"),
        h3c=("display ip routing-table", "display current-configuration"),
        signatures=(
            Signature("static_route_pattern",
                      r"ip\s+route-static\s+(\S+)\s+(\S+)\s+(\S+)",
                      "candidate static-route line — verify next-hop reachability against neighbors",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="ARP configuration error",
        category=_R,
        description="ARP entry missing, incomplete, or maps the destination IP to a wrong MAC; static-ARP misconfigurations also fall here.",
        huawei=("display arp", "display arp all"),
        cisco=("show ip arp",),
        h3c=("display arp all",),
        signatures=(
            Signature("incomplete_arp", r"\b(?:Incomplete|incomplete|INCOMPLETE)\b",
                      "ARP entry incomplete",
                      target_commands=("display arp", "display arp all", "show ip arp"),
                      evidence_type=_F),
            Signature("arp_static_config",
                      r"arp\s+static\s+(\S+)\s+([0-9a-fA-F.:-]+)",
                      "static-ARP configuration entry",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="Layer 3 loop",
        category=_R,
        description="Routing loop at L3: traffic bounces between two or more devices because of misconfigured static/dynamic routes.",
        huawei=("display ip routing-table",),
        cisco=("show ip route",),
        h3c=("display ip routing-table",),
        signatures=(
            Signature("ttl_exceeded", r"\bTTL\s+(?:exceeded|expired)\b",
                      "TTL expired symptom of a loop", evidence_type=_F),
        ),
        strength="low",
    ),
    _entry(
        reason="BGP configuration error",
        category=_R,
        description="BGP peer is not Established, AS number mismatch, missing 'network' statement, route-policy filter, or wrong route-reflector relation.",
        huawei=("display bgp peer", "display bgp routing-table", "display current-configuration"),
        cisco=("show ip bgp summary", "show ip bgp", "show running-config"),
        h3c=("display bgp peer", "display bgp routing-table", "display current-configuration"),
        signatures=(
            Signature("peer_not_established",
                      r"^\s*\d+\.\d+\.\d+\.\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\S+\s+(?:Idle|Active|Connect|OpenSent|OpenConfirm)\b",
                      "peer state not Established (display bgp peer table row)",
                      target_commands=("display bgp peer",), evidence_type=_F),
            Signature("as_mismatch", r"\bAS\s+number\s+mismatch\b",
                      "AS number mismatch", evidence_type=_F),
            Signature("bgp_block", r"^\s*bgp\s+(\d+)\s*$",
                      "BGP process configuration block — verify AS",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="high",
    ),
    _entry(
        reason="OSPF configuration error",
        category=_R,
        description="OSPF neighbor not Full, area mismatch, missing 'network' statement, authentication failure, or area-type mismatch.",
        huawei=("display ospf peer", "display ospf routing", "display ospf interface", "display current-configuration"),
        cisco=("show ip ospf neighbor", "show ip route ospf", "show running-config"),
        h3c=("display ospf peer", "display ospf routing", "display current-configuration"),
        signatures=(
            Signature("neighbor_not_full",
                      r"\b(?:Down|Init|2-?Way|ExStart|Exchange|Loading)\b",
                      "OSPF neighbor stuck below Full state",
                      target_commands=("display ospf peer", "show ip ospf neighbor"),
                      evidence_type=_F),
            Signature("area_mismatch", r"\b[Aa]rea\s+mismatch\b",
                      "OSPF area mismatch", evidence_type=_F),
            Signature("ospf_block", r"^\s*ospf\s+(\d+)\s*$",
                      "OSPF process configuration block — verify network/area",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="high",
    ),
    _entry(
        reason="loopback interface IP configuration conflict",
        category=_R,
        description="Two devices configured with overlapping or identical loopback addresses, causing route-instability or RID collision.",
        huawei=("display ip interface brief", "display current-configuration"),
        cisco=("show ip interface brief", "show running-config"),
        h3c=("display ip interface brief", "display current-configuration"),
        signatures=(
            Signature("loopback_iface_block",
                      r"interface\s+(?:LoopBack|Loopback|loopback)\s*(\d+)",
                      "loopback interface declaration — compare IPs across devices",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="VXLAN configuration error",
        category=_R,
        description="VXLAN tunnel not up, VNI mismatch, or NVE peer/source missing.",
        huawei=("display vxlan tunnel", "display vxlan troubleshooting", "display current-configuration"),
        cisco=("show nve peers", "show nve vni", "show running-config"),
        h3c=("display vxlan tunnel", "display vxlan troubleshooting", "display current-configuration"),
        signatures=(
            Signature("tunnel_state_down",
                      r"\b(?:DOWN|down)\b",
                      "VXLAN tunnel state DOWN",
                      target_commands=("display vxlan tunnel",), evidence_type=_F),
            Signature("nve_block", r"\bnve\b",
                      "NVE configuration block presence",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="L3VPN configuration error",
        category=_R,
        description="VPN-instance RD/RT mismatch, missing import/export, or VPNv4 BGP family not enabled.",
        huawei=("display ip vpn-instance", "display ip vpn-instance verbose", "display bgp vpnv4 all routing-table"),
        cisco=("show ip vrf", "show bgp vpnv4 unicast all"),
        h3c=("display ip vpn-instance", "display bgp vpnv4 all routing-table"),
        signatures=(
            Signature("rt_export_import",
                      r"vpn-target\s+\S+\s+(?:export-extcommunity|import-extcommunity|both)",
                      "VPN-target import/export RT — verify symmetry",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="L2VPN configuration error",
        category=_R,
        description="EVPN/PWE3 service mis-bound: missing or wrong RD/RT, missing service instance, mismatched VC ID, or missing EVPN address-family activation.",
        huawei=("display bgp evpn all routing-table", "display current-configuration"),
        cisco=("show bgp l2vpn evpn", "show running-config"),
        h3c=("display bgp l2vpn evpn", "display current-configuration"),
        signatures=(
            Signature("evpn_block", r"\bevpn\b",
                      "EVPN configuration block presence",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="low",
    ),
    _entry(
        reason="ISIS configuration error",
        category=_R,
        description="ISIS adjacency stuck, area-id mismatch, level mismatch, or wrong network-entity-title (NET).",
        huawei=("display current-configuration",),
        cisco=("show running-config",),
        h3c=("display current-configuration",),
        signatures=(
            Signature("isis_block", r"^\s*isis\s+\d*\s*$",
                      "ISIS process declaration",
                      target_commands=("display current-configuration",), evidence_type=_FP),
            Signature("net_statement", r"\bnet\s+(\d+\.[0-9.]+\.\d+)",
                      "ISIS NET statement",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="SRV6-Policy tunnel planning error",
        category=_R,
        description="SRv6 policy color/endpoint mismatch, missing candidate-path SID list, or local SID not advertised.",
        huawei=("display srv6-te policy", "display srv6-te policy status", "display segment-routing ipv6 local-sid end forwarding"),
        cisco=("show segment-routing srv6 policy",),
        h3c=("display segment-routing ipv6 te policy", "display segment-routing ipv6 local-sid"),
        signatures=(
            Signature("policy_state_down",
                      r"\b(?:Down|DOWN|down)\b",
                      "SRv6 policy candidate-path DOWN",
                      target_commands=("display srv6-te policy",
                                       "display srv6-te policy status",
                                       "display segment-routing ipv6 te policy"),
                      evidence_type=_F),
        ),
        strength="low",
    ),
    _entry(
        reason="NAT external interface attribute configuration error or configuration missing",
        category=_R,
        description="NAT outbound rule missing, wrong zone, or wrong direction; the public-facing interface is not flagged as the NAT external/untrust attribute.",
        huawei=("display nat policy", "display nat session", "display current-configuration | include nat", "display zone"),
        cisco=("show running-config",),
        h3c=("display current-configuration",),
        signatures=(
            Signature("nat_outbound", r"nat\s+outbound\b",
                      "NAT outbound rule presence",
                      target_commands=("display current-configuration",), evidence_type=_FP),
            Signature("untrust_zone", r"firewall\s+zone\s+untrust\b",
                      "untrust zone definition",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="NAT internal interface attribute configuration error or missing",
        category=_R,
        description="The internal/trust-side interface is not bound to the NAT zone or not added to the inside-network ACL.",
        huawei=("display nat policy", "display current-configuration | include nat", "display zone"),
        cisco=("show running-config",),
        h3c=("display current-configuration",),
        signatures=(
            Signature("trust_zone", r"firewall\s+zone\s+trust\b",
                      "trust zone definition",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="global STP not enabled",
        category=_R,
        description="Global Spanning Tree is administratively disabled at the switch level — bridging loops and broadcast storms become possible.",
        huawei=("display stp", "display current-configuration"),
        cisco=("show spanning-tree summary", "show running-config"),
        h3c=("display stp", "display current-configuration"),
        signatures=(
            Signature("stp_disabled_global", r"\bstp\s+disable\b",
                      "explicit global STP disable",
                      target_commands=("display current-configuration",), evidence_type=_F),
            # Negative match on `stp enable` was too noisy: Huawei devices
            # default-enable STP globally, so absence of an explicit `stp
            # enable` line is NOT evidence of disablement. Rely on the
            # positive `stp disable` match instead.
        ),
        strength="high",
    ),
    _entry(
        reason="IP address prefix list missing corresponding user source IP address",
        category=_R,
        description="Prefix-list / ip-prefix used to gate user traffic does not include the user's source IP, causing the policy to reject the user.",
        huawei=("display current-configuration",),
        cisco=("show running-config",),
        h3c=("display current-configuration",),
        signatures=(
            Signature("prefix_list_block",
                      r"ip\s+ip-prefix\s+(\S+)\s+index\s+\d+\s+permit\s+(\S+)",
                      "ip-prefix list permit entry",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="global HRP hot redundancy protocol not enabled",
        category=_R,
        description="Huawei firewall HRP (Hot Redundancy Protocol) is not enabled, so session sync and peer takeover do not work.",
        huawei=("display current-configuration",),
        cisco=(),
        h3c=(),
        signatures=(
            Signature("hrp_enable", r"\bhrp\s+enable\b",
                      "absence of 'hrp enable' on a firewall device indicates HRP not enabled",
                      target_commands=("display current-configuration",),
                      polarity="negative", evidence_type=_F,
                      # HRP only applies to USG-series firewalls.
                      device_filter_regex=r"(?i)\b(?:firewall|USG\d+)\b"),
        ),
        strength="high",
    ),
    _entry(
        reason="security policy rule not permitting corresponding users",
        category=_R,
        description="Firewall security-policy rule denies (or fails to permit) the user-to-destination flow.",
        huawei=("display security-policy", "display zone", "display current-configuration"),
        cisco=("show running-config",),
        h3c=("display current-configuration",),
        signatures=(
            Signature("action_deny", r"\baction\s+deny\b",
                      "security-policy explicit deny",
                      target_commands=("display security-policy", "display current-configuration"),
                      evidence_type=_F),
            Signature("security_policy_rule", r"\bsecurity-policy\b",
                      "security-policy section presence",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="high",
    ),
    # ---- PORT (12) ----
    _entry(
        reason="shutdown",
        category=_P,
        description="Interface is administratively shut down, reported as DOWN/DOWN with admin DOWN.",
        huawei=("display interface brief", "display interface description"),
        cisco=("show interface brief", "show ip interface brief"),
        h3c=("display interface brief",),
        signatures=(
            # Many interfaces are legitimately admin-down or shut in healthy
            # configs (unused ports, routing-only loopbacks, etc.). Fire as
            # feature_presence so the LLM/ranker can pair them with a parsed
            # symptom (e.g., target on this interface) instead of the miner
            # blanket-emitting candidates.
            Signature("admin_down",
                      r"(?:administratively\s+down|admin(?:istrative)?\s+DOWN|admin\s+down)",
                      "interface admin DOWN observed in brief output",
                      target_commands=("display interface brief", "display interface description",
                                       "display ip interface brief", "show interface brief",
                                       "show ip interface brief"),
                      evidence_type=_FP),
            Signature("shutdown_in_config",
                      r"^\s*shutdown\s*$",
                      "'shutdown' keyword inside an interface stanza",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="high",
    ),
    _entry(
        reason="interface IP error",
        category=_P,
        description="Interface IP/mask is wrong: subnet mismatch with the peer, address conflict, or missing IP entirely.",
        huawei=("display ip interface brief", "display interface description", "display current-configuration"),
        cisco=("show ip interface brief", "show running-config"),
        h3c=("display ip interface brief", "display current-configuration"),
        signatures=(
            Signature("ip_address_block",
                      r"\bip\s+address\s+(\S+)\s+(\S+)",
                      "ip address line — verify subnet match with neighbor",
                      target_commands=("display current-configuration",), evidence_type=_FP),
            # L2 access ports legitimately have no IP; without symptom
            # context this is not fault evidence. Demote to presence.
            Signature("unassigned_ip", r"\bunassigned\b",
                      "interface has no IP — only meaningful for L3-routed ports",
                      target_commands=("display ip interface brief", "show ip interface brief"),
                      evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="traffic occupying port bandwidth",
        category=_P,
        description="Interface utilisation saturates the link bandwidth, causing intermittent loss; visible as high input/output rate or QoS drops.",
        huawei=("display interface description", "display traffic-policy"),
        cisco=("show interface brief",),
        h3c=("display interface brief",),
        signatures=(
            Signature("high_utilization",
                      r"(?:input|output)\s+rate\s+\d+\s+bits/sec.*(?:99|9[5-9])%",
                      "near-saturation utilisation",
                      target_commands=("display interface description",), evidence_type=_F),
        ),
        strength="low",
    ),
    _entry(
        reason="MAC address configuration error",
        category=_P,
        description="Static MAC entry maps to a wrong port or wrong VLAN, or the MAC is missing from the table where it should be learned.",
        huawei=("display mac-address",),
        cisco=("show mac address-table",),
        h3c=("display mac-address",),
        signatures=(
            Signature("mac_address_static_config",
                      r"mac-address\s+static\s+([0-9a-fA-F.-]+)",
                      "static MAC config entry",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="VPN configuration missing",
        category=_P,
        description="Interface is not bound to the expected vpn-instance, breaking VRF isolation.",
        huawei=("display ip vpn-instance", "display current-configuration"),
        cisco=("show ip vrf", "show running-config"),
        h3c=("display ip vpn-instance", "display current-configuration"),
        signatures=(
            Signature("binding_block",
                      r"\bip\s+binding\s+vpn-instance\s+(\S+)",
                      "vpn-instance binding on the interface",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="OSPF configuration error",  # port-side OSPF
        category=_P,
        description="Interface is not enabled in the expected OSPF area or has authentication misconfigured.",
        huawei=("display ospf interface", "display current-configuration"),
        cisco=("show ip ospf interface", "show running-config"),
        h3c=("display current-configuration",),
        signatures=(
            Signature("ospf_enable_iface", r"\bospf\s+enable\b",
                      "interface-level OSPF enable",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="MTU value configuration error",
        category=_P,
        description="Interface MTU is mismatched between the two ends, dropping large frames silently.",
        huawei=("display interface description", "display current-configuration"),
        cisco=("show interface brief", "show running-config"),
        h3c=("display interface brief", "display current-configuration"),
        signatures=(
            Signature("mtu_setting", r"\bmtu\s+(\d+)",
                      "explicit MTU value — verify both ends",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="host information collection function missing",
        category=_P,
        description="Host-info collection (e.g., DHCP-snooping or IP-source-guard) is not enabled on the access port, so user identity cannot be associated.",
        huawei=("display current-configuration",),
        cisco=("show running-config",),
        h3c=("display current-configuration",),
        signatures=(
            Signature("dhcp_snooping", r"\bdhcp\s+snooping\s+enable\b",
                      "DHCP-snooping enable line",
                      target_commands=("display current-configuration",), evidence_type=_FP),
            Signature("user_collect",
                      r"\b(?:user-collect|host-collect|ip\s+source\s+check)\b",
                      "host-information-collection feature toggle",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="low",
    ),
    _entry(
        reason="interface VLAN configuration error",
        category=_P,
        description="Interface placed in a wrong access VLAN, missing a trunk-allowed-VLAN, or PVID mismatched with the peer.",
        huawei=("display vlan", "display current-configuration"),
        cisco=("show vlan", "show running-config"),
        h3c=("display vlan", "display current-configuration"),
        signatures=(
            Signature("port_default_vlan",
                      r"\bport\s+default\s+vlan\s+(\d+)\b",
                      "access VLAN setting",
                      target_commands=("display current-configuration",), evidence_type=_FP),
            Signature("trunk_allow",
                      r"\bport\s+trunk\s+allow-pass\s+vlan\s+",
                      "trunk allow-pass list",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="NAT external interface attribute configuration error or configuration missing",  # port-level
        category=_P,
        description="Outbound interface lacks `nat outbound` configuration, so traffic leaving the firewall is not translated.",
        huawei=("display current-configuration | include nat", "display nat policy"),
        cisco=("show running-config",),
        h3c=("display current-configuration",),
        signatures=(
            Signature("nat_outbound_iface",
                      r"\bnat\s+outbound\s+(\d+)",
                      "interface-level nat outbound rule reference",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="NAT internal interface attribute configuration error or missing",
        category=_P,
        description="Internal-side interface missing the inside-NAT binding (or wrong zone), breaking outbound translation.",
        huawei=("display current-configuration | include nat",),
        cisco=("show running-config",),
        h3c=("display current-configuration",),
        signatures=(
            Signature("trust_zone_iface", r"\bzone\s+trust\b",
                      "trust-zone interface binding",
                      target_commands=("display current-configuration",), evidence_type=_FP),
        ),
        strength="medium",
    ),
    _entry(
        reason="port STP not enabled",
        category=_P,
        description="Spanning Tree disabled on a specific port, allowing local L2 loops.",
        huawei=("display stp interface brief", "display current-configuration"),
        cisco=("show spanning-tree brief",),
        h3c=("display stp interface brief", "display current-configuration"),
        signatures=(
            Signature("stp_iface_disable",
                      r"^\s*stp\s+disable\s*$",
                      "interface stanza disables STP",
                      target_commands=("display current-configuration",), evidence_type=_F),
        ),
        strength="high",
    ),
)


# ---- Phase 1 vocabulary aliases (lowercased keys mapped to Phase 2 names) --

PHASE_1_ALIASES: dict[str, str] = {
    "routing loop": "Layer 3 loop",
    "loopback IP configuration conflict": "loopback interface IP configuration conflict",
    "IS-IS configuration error": "ISIS configuration error",
    "traffic congestion on port bandwidth": "traffic occupying port bandwidth",
}


PLAYBOOK_BY_KEY: dict[tuple[str, str], FaultEntry] = {
    (e.fault_reason, e.category): e for e in _PLAYBOOK_LIST
}
PLAYBOOK_BY_REASON: dict[str, list[FaultEntry]] = {}
for _entry_obj in _PLAYBOOK_LIST:
    PLAYBOOK_BY_REASON.setdefault(_entry_obj.fault_reason, []).append(_entry_obj)


def lookup(reason: str, category: str | None = None) -> FaultEntry | None:
    canonical = reason
    if reason not in PLAYBOOK_BY_REASON:
        canonical = PHASE_1_ALIASES.get(reason, reason)
    entries = PLAYBOOK_BY_REASON.get(canonical)
    if not entries:
        return None
    if category is None:
        return entries[0]
    for e in entries:
        if e.category == category:
            return e
    return entries[0]


def vendor_commands(reason: str, vendor: str, category: str | None = None) -> tuple[str, ...]:
    e = lookup(reason, category)
    if not e:
        return ()
    return e.commands.get(vendor.lower(), ())


# ---- Compiled regex cache --------------------------------------------------

_COMPILED_CACHE: dict[tuple[str, str, str], re.Pattern[str]] = {}


def compile_signatures() -> dict[tuple[str, str, str], re.Pattern[str]]:
    if _COMPILED_CACHE:
        return _COMPILED_CACHE
    for e in _PLAYBOOK_LIST:
        for s in e.signatures:
            key = (e.fault_reason, e.category, s.name)
            _COMPILED_CACHE[key] = re.compile(s.regex, re.MULTILINE)
    return _COMPILED_CACHE


# ---- Prompt-context exporter ----------------------------------------------

def to_json_summary(reasons: Iterable[str], vendors: Iterable[str] = ("huawei",)) -> str:
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in reasons:
        for e in PLAYBOOK_BY_REASON.get(r, ()):
            if (e.fault_reason, e.category) in seen:
                continue
            seen.add((e.fault_reason, e.category))
            entry = {
                "fault_reason": e.fault_reason,
                "category": e.category,
                "description": e.description,
                "commands": {v: list(e.commands.get(v, ())) for v in vendors},
                "signatures": [
                    {"name": s.name, "regex": s.regex, "description": s.description,
                     "evidence_type": s.evidence_type, "polarity": s.polarity}
                    for s in e.signatures
                ],
                "evidence_strength": e.evidence_strength,
            }
            out.append(entry)
    return json.dumps(out, ensure_ascii=False, indent=2)


def all_reasons() -> tuple[str, ...]:
    return tuple(e.fault_reason for e in _PLAYBOOK_LIST)


def all_entries() -> tuple[FaultEntry, ...]:
    return _PLAYBOOK_LIST
