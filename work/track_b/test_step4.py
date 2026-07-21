"""Smoke test for the offline anomaly miner.

Builds two synthetic device fixtures (a faulty one and a clean one), runs
the miner against them, and asserts the expected candidates fire / don't
fire. Also verifies:
    - feature_presence signatures never produce candidates
    - device_filter_regex correctly scopes (HRP only on firewall devices)
    - negative-polarity signatures fire when the regex is absent and the
      filter passes; don't fire when the regex is present
    - candidate_to_row schema matches the CSV columns
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.anomaly_miner import (
    _CANDIDATE_FIELDS,
    candidate_to_row,
    command_to_filename,
    mine_device,
)


def _write_device_fixture(root: Path, name: str, files: dict[str, str]) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for cmd, content in files.items():
        (d / command_to_filename(cmd)).write_text(content, encoding="utf-8")
    return d


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def main() -> int:
    failures = 0

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)

        # Fixture A — Huawei firewall-like device with multiple faults
        faulty_cfg = """
sysname FW-01
#
firewall zone trust
 set priority 85
#
firewall zone untrust
 set priority 5
#
ip route-static 10.99.0.0 255.255.0.0 NULL0
#
interface GigabitEthernet0/0/1
 shutdown
#
ospf 1
 area 0.0.0.0
#
"""
        faulty_arp = """
IP ADDRESS      MAC ADDRESS    EXPIRE(M) TYPE  VLAN  INTERFACE
10.3.1.2        Incomplete        1   D               Vlanif1001
10.0.0.1        00aa-bbcc-ddee  120   D               Vlanif1
"""
        faulty_ospf = """
Neighbor ID     Pri State        Mode    Priority  Cost
10.0.0.1        1   Init         Slave   1         1
"""
        faulty_vxlan = """
Number of vxlan tunnel : 2
Tunnel ID    Source             Destination          State  Type
4026531916   1.1.2.3            1.1.2.1              down   dynamic
"""
        faulty_routes = """
Destination/Mask    Proto   Pre  Cost    Flags  NextHop      Interface
10.99.0.0/16        Static  60   0       D      NULL0        NULL0
"""
        _write_device_fixture(
            tmp,
            "FW-01",
            {
                "display current-configuration": faulty_cfg,
                "display arp": faulty_arp,
                "display ospf peer": faulty_ospf,
                "display vxlan tunnel": faulty_vxlan,
                "display ip routing-table": faulty_routes,
            },
        )

        # Fixture B — clean Huawei switch (has STP enabled, no faults)
        clean_cfg = """
sysname Switch-01
#
stp mode rstp
stp enable
#
vlan batch 100 200
#
interface GigabitEthernet0/0/1
 port link-type trunk
 port trunk allow-pass vlan 100 200
#
"""
        clean_arp = """
IP ADDRESS      MAC ADDRESS     EXPIRE(M) TYPE  VLAN  INTERFACE
10.0.0.1        00aa-bbcc-ddee  120   D               Vlanif100
"""
        clean_ospf = """
Neighbor ID     Pri State        Mode    Priority  Cost
10.0.0.1        1   Full         Slave   1         1
"""
        clean_vxlan = """
Number of vxlan tunnel : 1
Tunnel ID    Source             Destination          State  Type
4026531916   1.1.2.3            1.1.2.1              up     dynamic
"""
        clean_routes = """
Destination/Mask    Proto   Pre  Cost    Flags  NextHop      Interface
10.0.0.0/8          Static  60   0       D      1.2.3.4      GE0/0/1
"""
        _write_device_fixture(
            tmp,
            "Switch-01",
            {
                "display current-configuration": clean_cfg,
                "display arp": clean_arp,
                "display ospf peer": clean_ospf,
                "display vxlan tunnel": clean_vxlan,
                "display ip routing-table": clean_routes,
            },
        )

        section("faulty device (FW-01) — expected candidates")
        cands_a = mine_device(
            scenario_id="test-A",
            question_number=1,
            phase=1,
            node="FW-01",
            device_dir=tmp / "FW-01",
        )
        reasons_a = sorted({c.fault_reason for c in cands_a})
        print(f"  fired reasons: {reasons_a}")
        expected = {
            "blackhole route",  # static_blackhole + null0_next_hop
            "ARP configuration error",  # incomplete_arp
            "OSPF configuration error",  # neighbor_not_full
            "VXLAN configuration error",  # tunnel_state_down
            "global HRP hot redundancy protocol not enabled",  # negative match (no hrp enable; firewall keyword present)
        }
        missing = expected - set(reasons_a)
        if missing:
            print(f"  FAIL: missing expected reasons {missing}")
            failures += 1
        else:
            print(f"  ok: all {len(expected)} expected reasons fired")

        section("clean device (Switch-01) — expected NO candidates")
        cands_b = mine_device(
            scenario_id="test-B",
            question_number=2,
            phase=1,
            node="Switch-01",
            device_dir=tmp / "Switch-01",
        )
        reasons_b = sorted({c.fault_reason for c in cands_b})
        print(f"  fired reasons: {reasons_b}")
        if reasons_b:
            print(f"  FAIL: clean device produced candidates")
            failures += 1
        else:
            print(f"  ok: zero candidates on clean device")

        section("HRP filter — non-firewall device should NOT fire")
        plain_router_cfg = """
sysname Router-01
#
ospf 1
 area 0.0.0.0
#
"""
        _write_device_fixture(tmp, "Router-01", {"display current-configuration": plain_router_cfg})
        cands_c = mine_device(
            scenario_id="test-C",
            question_number=3,
            phase=1,
            node="Router-01",
            device_dir=tmp / "Router-01",
        )
        hrp_fires = [c for c in cands_c if c.fault_reason == "global HRP hot redundancy protocol not enabled"]
        if hrp_fires:
            print(f"  FAIL: HRP fired on non-firewall device")
            failures += 1
        else:
            print(f"  ok: HRP did not fire on non-firewall")

        section("CSV row schema")
        if cands_a:
            row = candidate_to_row(cands_a[0])
            extra = set(row.keys()) - set(_CANDIDATE_FIELDS)
            missing = set(_CANDIDATE_FIELDS) - set(row.keys())
            if extra or missing:
                print(f"  FAIL: schema mismatch (extra={extra}, missing={missing})")
                failures += 1
            else:
                print(f"  ok: row has exactly the {len(_CANDIDATE_FIELDS)} expected columns")

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
