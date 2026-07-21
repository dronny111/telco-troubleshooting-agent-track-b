"""Smoke test for Step 2 deliverables: classifier, vocab extractor,
constraint parser, and format guard.

Runs against the actual Phase 1 + Phase 2 questions and a small set of
synthetic answer strings (good + adversarial) to confirm:
    - Every question is classified into a known family.
    - Every fault question yields a non-empty (routing, port) vocab.
    - Constraint parser extracts the obvious symptom suffix fields.
    - Format guard accepts well-formed answers and rejects schema violations,
      out-of-vocabulary reasons, non-ASCII separators, and whitespace bugs.

Reports per-section pass/fail counts. Exits 0 iff every assertion holds.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # work/

from track_b.task_classifier import classify
from track_b.vocab_extractor import extract_fault_vocab
from track_b.constraint_parser import parse
from track_b.format_guard import validate_fault, validate_path, validate_topology


P1 = Path(__file__).resolve().parents[2] / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json"
P2 = Path(__file__).resolve().parents[2] / "telco_data" / "Track B" / "data" / "Phase_2" / "test.json"


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def assert_eq(actual, expected, label: str) -> bool:
    ok = actual == expected
    mark = "ok" if ok else "FAIL"
    print(f"  [{mark}] {label}: actual={actual!r}, expected={expected!r}")
    return ok


def main() -> int:
    failures = 0

    section("classifier coverage")
    p1 = json.load(open(P1))
    p2 = json.load(open(P2))
    p1_fams = Counter(classify(it["task"]["question"]) for it in p1)
    p2_fams = Counter(classify(it["task"]["question"]) for it in p2)
    print(f"  Phase 1 families: {dict(p1_fams)} (total {sum(p1_fams.values())})")
    print(f"  Phase 2 families: {dict(p2_fams)} (total {sum(p2_fams.values())})")
    if p1_fams.get("other", 0) > 0 or p2_fams.get("other", 0) > 0:
        print("  WARN: 'other' bucket non-empty — review classifier signatures")

    section("vocab extraction coverage on fault questions")
    misses = []
    for it in p1 + p2:
        if classify(it["task"]["question"]) != "fault":
            continue
        r, pv = extract_fault_vocab(it["task"]["question"])
        if not (r and pv):
            misses.append((it["task"]["id"], len(r), len(pv)))
    if misses:
        print(f"  FAIL: {len(misses)} extraction misses: {misses[:5]}")
        failures += 1
    else:
        fault_total = sum(1 for it in p1 + p2 if classify(it["task"]["question"]) == "fault")
        print(f"  ok: extracted vocab for all {fault_total} fault questions")

    section("constraint parser — known cases")
    # P2 task 1: VRRP dual-master + Limitation
    p2_t1 = next(it for it in p2 if it["task"]["id"] == 1)
    c1 = parse(p2_t1["task"]["question"])
    failures += not assert_eq(c1.disclosed_fault_nodes, ("Core_SW_01", "Core_SW_02"), "p2_t1 disclosed VRRP nodes")
    failures += not assert_eq(c1.disclosed_vlanif, "Vlanif120", "p2_t1 disclosed_vlanif")
    failures += not assert_eq(c1.blacklisted_nodes, ("Core_SW_02",), "p2_t1 blacklisted_nodes")
    failures += not assert_eq("VRRP-dual-master" in c1.disclosed_fault_categories, True, "p2_t1 disclosed VRRP-dual-master category")

    # P2 task 3: From X, accessing IP failed; Limitation: Do not look for faults on FW_02.
    p2_t3 = next(it for it in p2 if it["task"]["id"] == 3)
    c3 = parse(p2_t3["task"]["question"])
    failures += not assert_eq(c3.source_endpoint, "GUEST_WIFI_CLIENT01", "p2_t3 source_endpoint")
    failures += not assert_eq(c3.target_destination_ip, "10.1.60.2", "p2_t3 target_destination_ip")
    failures += not assert_eq(c3.blacklisted_nodes, ("FW_02",), "p2_t3 blacklisted_nodes")

    # P2 task 4: From X, accessing data center server NAME(IP) failed
    p2_t4 = next(it for it in p2 if it["task"]["id"] == 4)
    c4 = parse(p2_t4["task"]["question"])
    failures += not assert_eq(c4.source_endpoint, "GUEST_WIFI_CLIENT01", "p2_t4 source_endpoint")
    failures += not assert_eq(c4.target_destination_host, "SZ_Server_Cluster2", "p2_t4 target_destination_host")
    failures += not assert_eq(c4.target_destination_ip, "10.3.20.1", "p2_t4 target_destination_ip")

    # P2 task 10: parenthetical note on source
    p2_t10 = next(it for it in p2 if it["task"]["id"] == 10)
    c10 = parse(p2_t10["task"]["question"])
    failures += not assert_eq(c10.source_endpoint, "EMPLOYEE_WIFI_CLIENT01", "p2_t10 source with paren note")
    failures += not assert_eq(c10.target_destination_host, "GoogleWebServer01", "p2_t10 dest host")

    # P1 task 17: fault candidate list
    p1_t17 = next(it for it in p1 if it["task"]["id"] == 17)
    c17 = parse(p1_t17["task"]["question"])
    failures += not assert_eq(
        c17.fault_candidate_nodes,
        ("Beta-Axis-02", "Beta-Portal-02", "Alpha-Center-02"),
        "p1_t17 fault_candidate_nodes",
    )

    section("format guard — accept paths")
    # Use the actual P2 vocab (extracted from question 1) so the guard reflects
    # what the production answer schema requires.
    rv, pv_vocab = extract_fault_vocab(p2_t1["task"]["question"])
    good_fault = "\n".join(
        [
            "Core_SW_01;Vlanif120;interface VLAN configuration error",
            "Core_SW_01;192.168.1.5;blackhole route",
        ]
    )
    rep = validate_fault(good_fault, rv, pv_vocab)
    failures += not assert_eq(rep.is_valid, True, "good fault accepted")
    failures += not assert_eq(rep.line_count, 2, "good fault line_count")
    failures += not assert_eq(rep.error_count, 0, "good fault error_count")

    good_path = "Beta-Node-02(GE1/0/1)->Beta-Axis-01(GE1/0/2)\nBeta-Node-02(GE1/0/2)->Beta-Axis-02(GE1/0/2)"
    rep = validate_path(good_path)
    failures += not assert_eq(rep.is_valid, True, "good path accepted")

    good_topo = "Beta-Node-02(GE1/0/1)->Beta-Axis-01(GE1/0/2)"
    rep = validate_topology(good_topo)
    failures += not assert_eq(rep.is_valid, True, "good topology accepted")

    section("format guard — reject adversarial")
    bad_cases: list[tuple[str, str]] = [
        ("trailing whitespace", "Core_SW_01;192.168.1.5;blackhole route  "),
        ("leading whitespace", "  Core_SW_01;192.168.1.5;blackhole route"),
        ("blank line in middle", "Core_SW_01;192.168.1.5;blackhole route\n\nCore_SW_01;192.168.1.6;blackhole route"),
        ("non-ASCII semicolon", "Core_SW_01；192.168.1.5；blackhole route"),
        ("oov reason", "Core_SW_01;192.168.1.5;cosmic ray"),
        ("malformed schema 2 fields", "Core_SW_01;blackhole route"),
        ("malformed schema 4 fields", "Core_SW_01;192.168.1.5;blackhole route;extra"),
        ("invalid ip and not port", "Core_SW_01;999.999.999.999;blackhole route"),
        ("port reason on ip line", "Core_SW_01;192.168.1.5;shutdown"),
        ("routing reason on port line", "Core_SW_01;GE1/0/1;blackhole route"),
    ]
    for label, raw in bad_cases:
        rep = validate_fault(raw, rv, pv_vocab)
        ok = (not rep.is_valid) and rep.error_count >= 1
        mark = "ok" if ok else "FAIL"
        print(f"  [{mark}] {label}: is_valid={rep.is_valid} errs={rep.error_count} sample_err={rep.errors[:1]}")
        if not ok:
            failures += 1

    section("format guard — re-emit hint")
    rep = validate_fault("Core_SW_01;cosmic;ray", rv, pv_vocab)
    hint = rep.hint_for_reemit()
    print(f"  hint sample: {hint[:200]}")
    failures += not assert_eq(bool(hint), True, "non-empty hint on rejection")

    section("summary")
    if failures == 0:
        print("  PASS — all assertions held")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
