"""Closed-vocabulary fault reasons for Track B.

Every fault line in the final answer must use exactly one of these strings as
the third (reason) field. Wording matches the question's authoritative list
verbatim; the format guard rejects anything not in these sets.
"""

from __future__ import annotations

# Output format A: fault-node;destination-IP;fault-reason
ROUTING_FAULT_REASONS: tuple[str, ...] = (
    "blackhole route",
    "missing static route",
    "static route error",
    "ARP configuration error",
    "Layer 3 loop",
    "BGP configuration error",
    "OSPF configuration error",
    "loopback interface IP configuration conflict",
    "VXLAN configuration error",
    "L3VPN configuration error",
    "L2VPN configuration error",
    "ISIS configuration error",
    "SRV6-Policy tunnel planning error",
    "NAT external interface attribute configuration error or configuration missing",
    "NAT internal interface attribute configuration error or missing",
    "global STP not enabled",
    "IP address prefix list missing corresponding user source IP address",
    "global HRP hot redundancy protocol not enabled",
    "security policy rule not permitting corresponding users",
)

# Output format B: fault-node;fault-port;fault-reason
PORT_FAULT_REASONS: tuple[str, ...] = (
    "shutdown",
    "interface IP error",
    "traffic occupying port bandwidth",
    "MAC address configuration error",
    "VPN configuration missing",
    "OSPF configuration error",
    "MTU value configuration error",
    "host information collection function missing",
    "interface VLAN configuration error",
    "NAT external interface attribute configuration error or configuration missing",
    "NAT internal interface attribute configuration error or missing",
    "port STP not enabled",
)

ALL_FAULT_REASONS: frozenset[str] = frozenset(ROUTING_FAULT_REASONS) | frozenset(PORT_FAULT_REASONS)
ROUTING_REASON_SET: frozenset[str] = frozenset(ROUTING_FAULT_REASONS)
PORT_REASON_SET: frozenset[str] = frozenset(PORT_FAULT_REASONS)

PROTOCOL_FAMILIES: tuple[str, ...] = (
    "BGP",
    "OSPF",
    "ISIS",
    "VXLAN",
    "VRRP",
    "STP",
    "MP-BGP",
    "L3VPN",
    "L2VPN",
    "SRV6",
    "EVPN",
    "NAT",
    "ARP",
    "VLAN",
    "HRP",
    "MPLS",
    "BFD",
    "DHCP",
    "LLDP",
)
