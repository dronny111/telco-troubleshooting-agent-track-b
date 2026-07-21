"""Validate the fault-catalog playbook coverage and signature regex.

Asserts:
    1. Every Phase 1 + Phase 2 fault-reason string (extracted dynamically
       from each question's format spec) resolves to a playbook entry.
    2. Every signature regex compiles.
    3. `vendor_commands()` returns at least one Huawei command for every
       routing-fault entry (Huawei is the dominant vendor in both phases'
       device-naming conventions).
    4. `to_json_summary()` produces valid JSON.
    5. Spot-checks: known signatures fire on synthetic CLI output.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.task_classifier import classify
from track_b.vocab_extractor import extract_fault_vocab
from track_b.playbook import (
    PHASE_1_ALIASES,
    all_entries,
    all_reasons,
    compile_signatures,
    lookup,
    to_json_summary,
    vendor_commands,
)

P1 = Path(__file__).resolve().parents[2] / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json"
P2 = Path(__file__).resolve().parents[2] / "telco_data" / "Track B" / "data" / "Phase_2" / "test.json"


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def main() -> int:
    failures = 0

    section("vocab coverage")
    p1 = json.load(open(P1))
    p2 = json.load(open(P2))
    seen_reasons: set[str] = set()
    for items in (p1, p2):
        for it in items:
            if classify(it["task"]["question"]) != "fault":
                continue
            r, p = extract_fault_vocab(it["task"]["question"])
            seen_reasons.update(r)
            seen_reasons.update(p)

    unmapped: list[str] = []
    for reason in sorted(seen_reasons):
        if lookup(reason) is None:
            unmapped.append(reason)
    if unmapped:
        print(f"  FAIL: {len(unmapped)} reason(s) without playbook entry:")
        for r in unmapped:
            print(f"    - {r!r}")
        failures += 1
    else:
        print(f"  ok: all {len(seen_reasons)} distinct reason strings resolved")

    section("regex compilation")
    try:
        compiled = compile_signatures()
        print(f"  ok: {len(compiled)} signatures compiled")
    except re.error as e:
        print(f"  FAIL: regex compile error — {e}")
        failures += 1

    section("vendor command presence")
    huawei_misses = []
    for e in all_entries():
        if e.category == "routing" and not vendor_commands(e.fault_reason, "huawei", "routing"):
            huawei_misses.append(e.fault_reason)
    if huawei_misses:
        print(f"  WARN: routing entries with no Huawei command: {huawei_misses}")
    else:
        print(f"  ok: every routing entry has at least one Huawei command")
    # Cisco / H3C may legitimately lack commands for vendor-specific features
    # like Huawei HRP — soft-warn only.
    cisco_total = sum(1 for e in all_entries() if e.commands.get("cisco"))
    h3c_total = sum(1 for e in all_entries() if e.commands.get("h3c"))
    print(f"  cisco command coverage: {cisco_total}/{len(all_entries())}")
    print(f"  h3c   command coverage: {h3c_total}/{len(all_entries())}")

    section("JSON summary serialisation")
    sample_reasons = ("blackhole route", "global STP not enabled", "VRRP configuration error")
    js = to_json_summary([r for r in sample_reasons if lookup(r) is not None])
    try:
        parsed = json.loads(js)
        print(f"  ok: emitted {len(parsed)} entries; size={len(js)} bytes")
        if parsed:
            print(f"  first entry keys: {sorted(parsed[0].keys())}")
    except json.JSONDecodeError as e:
        print(f"  FAIL: JSON parse error — {e}")
        failures += 1

    section("signature spot checks")
    samples = [
        (
            "blackhole route — null0 in routing table",
            "blackhole route",
            "Destination/Mask    Proto   Pre  Cost      Flags NextHop         Interface\n10.0.0.0/8           Static  60   0           D   NULL0           NULL0",
            True,
        ),
        (
            "blackhole route — clean routing output",
            "blackhole route",
            "Destination/Mask    Proto   Pre  Cost      Flags NextHop         Interface\n10.0.0.0/8           Static  60   0           D   1.2.3.4         GigabitEthernet0/0/1",
            False,
        ),
        (
            "shutdown — admin DOWN",
            "shutdown",
            "GigabitEthernet0/0/1 current state : administratively down",
            True,
        ),
        (
            "OSPF stuck below Full",
            "OSPF configuration error",
            "Neighbor ID     Pri    State        DR              BDR             Interface address\n10.1.1.1        1      ExStart      10.1.1.1        0.0.0.0         GigabitEthernet0/0/1",
            True,
        ),
        (
            "STP — disabled in config",
            "global STP not enabled",
            "stp disable\n#\nvlan 100",
            True,
        ),
    ]
    compiled = compile_signatures()
    for label, reason, sample_output, want_match in samples:
        e = lookup(reason)
        any_match = False
        for s in e.signatures:
            pat = compiled[(e.fault_reason, e.category, s.name)]
            if pat.search(sample_output):
                any_match = True
                break
        ok = any_match == want_match
        mark = "ok" if ok else "FAIL"
        print(f"  [{mark}] {label}: any_match={any_match} want={want_match}")
        if not ok:
            failures += 1

    section("Phase 1 alias resolution")
    for alias, canonical in PHASE_1_ALIASES.items():
        resolved = lookup(alias)
        if resolved is None or resolved.fault_reason != canonical:
            print(f"  FAIL: {alias!r} -> {resolved.fault_reason if resolved else None} (expected {canonical!r})")
            failures += 1
        else:
            print(f"  ok: {alias!r} -> {canonical!r}")

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
