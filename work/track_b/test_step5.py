"""Smoke test for the typed graph + permission pruner + feature extractor.

Builds a synthetic 4-device ring fixture, runs every parser, and asserts:
    - LLDP brief produces the expected adjacency edges
    - IP-interface-brief parses unassigned + IPv4/mask correctly
    - VRRP `Info: The VRRP does not exist` returns empty
    - VRRP with two MASTER groups produces the expected records
    - device_l2_subgraph collapses interface-level edges to device-level
    - extract_device_features computes degree + on_parsed_path correctly
    - permission_pruner returns the expected denied set against a real config
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.anomaly_miner import command_to_filename
from track_b.constraint_parser import parse as parse_constraints
from track_b.graph_features import extract_device_features
from track_b.permission_pruner import denied_pairs, is_denied
from track_b.topology import (
    build_scenario_graph,
    device_l2_subgraph,
    parse_ip_interface_brief,
    parse_lldp_brief,
    parse_vrrp_verbose,
)


REAL_LIMITS = Path(__file__).resolve().parents[2] / "telco_data" / "Track B" / "question_limits_config.json"


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def assert_eq(a, b, label: str) -> bool:
    ok = a == b
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: a={a!r} b={b!r}")
    return ok


def main() -> int:
    failures = 0

    section("LLDP brief parser")
    sample = """display lldp neighbor brief

Local Interface         Exptime(s) Neighbor Interface            Neighbor Device

-------------------------------------------------------------------------------------

GE1/0/0                       101  GE1/0/4                       Beta-Portal-01

GE1/0/1                       103  GE1/0/4                       Beta-Portal-02

GE1/0/2                       112  GE1/0/2                       Beta-Aegis-02
"""
    rows = parse_lldp_brief(sample)
    failures += not assert_eq(len(rows), 3, "lldp row count")
    failures += not assert_eq(rows[0], ("GE1/0/0", "GE1/0/4", "Beta-Portal-01"), "lldp row 0")
    failures += not assert_eq(rows[2][2], "Beta-Aegis-02", "lldp row 2 neighbor")

    section("IP interface brief parser")
    iface_sample = """display ip interface brief
*down: administratively down
The number of interface that is UP in Physical is 6
Interface                   IP Address/Mask    Physical Protocol VPN
GE1/0/0                     192.168.65.178/30  up       up       --
GE1/0/1                     192.168.65.194/30  up       up       --
LoopBack1                   192.168.66.254/32  up       up(s)    --
MEth0/0/0                   unassigned         up       down     --
NULL0                       unassigned         up       up(s)    --
"""
    ifs = parse_ip_interface_brief(iface_sample)
    by_name = {i.name: i for i in ifs}
    failures += not assert_eq(by_name["GE1/0/0"].ip, "192.168.65.178", "GE1/0/0 ip")
    failures += not assert_eq(by_name["GE1/0/0"].mask, 30, "GE1/0/0 mask")
    failures += not assert_eq(by_name["LoopBack1"].ip, "192.168.66.254", "LoopBack1 ip")
    failures += not assert_eq(by_name["MEth0/0/0"].ip, None, "MEth0/0/0 unassigned")

    section("VRRP verbose parser — empty case")
    failures += not assert_eq(parse_vrrp_verbose("Info: The VRRP does not exist."), [], "VRRP empty")

    section("VRRP verbose parser — two-group case")
    vrrp_sample = """display vrrp verbose

Vlanif100 | Virtual Router 1
   State : MASTER
   Virtual IP : 192.168.1.1

Vlanif200 | Virtual Router 2
   State : BACKUP
   Virtual IP : 192.168.2.1
"""
    vs = parse_vrrp_verbose(vrrp_sample)
    failures += not assert_eq(len(vs), 2, "VRRP records count")
    failures += not assert_eq(vs[0].group_id, 1, "VRRP[0].group_id")
    failures += not assert_eq(vs[0].state, "MASTER", "VRRP[0].state")
    failures += not assert_eq(vs[1].state, "BACKUP", "VRRP[1].state")

    section("synthetic 4-device ring graph")
    with tempfile.TemporaryDirectory() as tmpd:
        root = Path(tmpd) / "devices_outputs"
        scen = root / "999"
        # ring: DA — DB — DC — DD — DA. Multi-char names so the constraint
        # parser's source regex (requires [A-Z][A-Za-z0-9_]+) matches.
        ring = [("DA", "DB"), ("DB", "DC"), ("DC", "DD"), ("DD", "DA")]
        idx = {"DA": 1, "DB": 2, "DC": 3, "DD": 4}
        for dev_a, dev_b in ring:
            for d, nbr in ((dev_a, dev_b), (dev_b, dev_a)):
                ddir = scen / d
                ddir.mkdir(parents=True, exist_ok=True)
                lldp_path = ddir / command_to_filename("display lldp neighbor brief")
                local_iface = f"GE1/0/{idx[nbr]-1}"
                nbr_iface = f"GE1/0/{idx[d]-1}"
                line = f"{local_iface}                       100  {nbr_iface}                       {nbr}\n"
                with open(lldp_path, "a", encoding="utf-8") as f:
                    if not lldp_path.exists() or lldp_path.stat().st_size == 0:
                        f.write("display lldp neighbor brief\n\n"
                                "Local Interface         Exptime(s) Neighbor Interface            Neighbor Device\n"
                                "-------------------------------------------------------------------------------\n")
                    f.write(line)
                ipb_path = ddir / command_to_filename("display ip interface brief")
                with open(ipb_path, "a", encoding="utf-8") as f:
                    if not ipb_path.exists() or ipb_path.stat().st_size == 0:
                        f.write("display ip interface brief\n"
                                "Interface                   IP Address/Mask    Physical Protocol VPN\n")
                    octet = idx[d]
                    f.write(f"{local_iface}                     10.0.{octet}.1/30      up       up       --\n")

        sg = build_scenario_graph(
            scenario_id="ring-DA",
            question_number=999,
            phase=1,
            devices_outputs_root=root,
        )
        assert sg is not None
        l2 = device_l2_subgraph(sg.graph)
        failures += not assert_eq(sorted(l2.nodes), ["DA", "DB", "DC", "DD"], "ring devices")
        failures += not assert_eq(l2.number_of_edges(), 4, "ring edge count")
        for d in ("DA", "DB", "DC", "DD"):
            failures += not assert_eq(l2.degree(d), 2, f"ring degree({d})")

        # Synthesize a constraint mentioning DA and IP on DC; expect
        # on_parsed_path to include DA and DC.
        question = (
            "...interface IP error\n"
            "From DA, accessing 10.0.3.1 failed.\n"
        )
        parsed = parse_constraints(question)
        feats = extract_device_features(sg, parsed, question_limits_path=REAL_LIMITS)
        by_node = {f.node: f for f in feats}
        failures += not assert_eq(by_node["DA"].on_parsed_path, 1, "DA on_parsed_path")
        failures += not assert_eq(by_node["DC"].on_parsed_path, 1, "DC on_parsed_path")
        failures += not assert_eq(by_node["DA"].hop_distance_source, 0, "DA hop_dist_source")
        failures += not assert_eq(by_node["DC"].hop_distance_dest, 0, "DC hop_dist_dest")

    section("permission pruner against real config")
    # question_2 denies (Gamma-Axis-02, "display lldp neighbor brief")
    pairs = denied_pairs(2, REAL_LIMITS)
    failures += not assert_eq(("Gamma-Axis-02", "display lldp neighbor brief") in pairs, True, "q2 denies Gamma-Axis-02")
    failures += not assert_eq(
        is_denied(question_number=2, device="Gamma-Axis-02",
                  command="display lldp neighbor brief", config_path=REAL_LIMITS),
        True,
        "is_denied positive case",
    )
    failures += not assert_eq(
        is_denied(question_number=2, device="Gamma-Axis-02",
                  command="display arp", config_path=REAL_LIMITS),
        False,
        "is_denied negative case (different command)",
    )
    failures += not assert_eq(denied_pairs(1, REAL_LIMITS), set(), "q1 has no denials")

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
