"""Smoke test for the answer-feasibility validator + follow-up planner.

Asserts the four decision paths:

    1. Format guard rejects                     → action='reemit_format'
    2. Blacklisted node                         → action='reemit_constraint'
    3. Unknown node (not in scenario graph)     → action='reemit_constraint'
    4. Routing middle is not an IPv4            → action='reemit_constraint'
    5. Port middle is an IPv4                   → action='reemit_constraint'
    6. HRP claimed on a non-firewall device     → action='reemit_constraint'
    7. Well-formed but no anomaly evidence and
       primary diagnostic not yet executed      → action='fetch_evidence'
    8. Well-formed + anomaly evidence present   → action='accept'
    9. Well-formed + diagnostic already run     → action='accept'
   10. XGBoost calibrated_score < tau_score
       triggers fetch_evidence                  → action='fetch_evidence'
   11. Follow-up never repeats a command
       already in `fetched_commands`            → action='accept' or different cmd
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.answer_validator import validate_answer
from track_b.constraint_parser import ParsedConstraints


# Phase 2-style vocab (matches the canonical 19+12 list used in run_ranker)
ROUTING_VOCAB = (
    "blackhole route", "missing static route", "static route error",
    "ARP configuration error", "Layer 3 loop", "BGP configuration error",
    "OSPF configuration error", "loopback interface IP configuration conflict",
    "VXLAN configuration error", "L3VPN configuration error",
    "L2VPN configuration error", "ISIS configuration error",
    "SRV6-Policy tunnel planning error",
    "NAT external interface attribute configuration error or configuration missing",
    "NAT internal interface attribute configuration error or missing",
    "global STP not enabled",
    "IP address prefix list missing corresponding user source IP address",
    "global HRP hot redundancy protocol not enabled",
    "security policy rule not permitting corresponding users",
)
PORT_VOCAB = (
    "shutdown", "interface IP error", "traffic occupying port bandwidth",
    "MAC address configuration error", "VPN configuration missing",
    "OSPF configuration error", "MTU value configuration error",
    "host information collection function missing",
    "interface VLAN configuration error",
    "NAT external interface attribute configuration error or configuration missing",
    "NAT internal interface attribute configuration error or missing",
    "port STP not enabled",
)


# Synthetic graph features for "Core_SW_01", "PE1", "FW_01", "Random"
GF: dict[str, dict] = {
    "Core_SW_01": {"on_parsed_path": "1"},
    "PE1": {"on_parsed_path": "1"},
    "FW_01": {"on_parsed_path": "1"},
    "AGG_SW_01": {"on_parsed_path": "0"},
}


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def assert_eq(a, b, label: str) -> bool:
    ok = a == b
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: a={a!r} b={b!r}")
    return ok


def main() -> int:
    failures = 0

    section("1. format guard rejects (trailing whitespace)")
    d = validate_answer(
        draft_answer="Core_SW_01;192.168.1.5;blackhole route  ",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features=GF,
    )
    failures += not assert_eq(d.action, "reemit_format", "action")
    failures += not assert_eq(bool(d.format_hint), True, "non-empty format_hint")

    section("2. blacklisted node")
    d = validate_answer(
        draft_answer="FW_02;192.168.1.5;blackhole route",
        scenario_id="scn",
        parsed=ParsedConstraints(blacklisted_nodes=("FW_02",)),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features={"FW_02": {}, "Core_SW_01": {}},
    )
    failures += not assert_eq(d.action, "reemit_constraint", "action")
    failures += not assert_eq(any("blacklist" in v for v in d.constraint_violations),
                              True, "violation mentions blacklist")

    section("3. unknown node")
    d = validate_answer(
        draft_answer="Ghost_Node;192.168.1.5;blackhole route",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features=GF,
    )
    failures += not assert_eq(d.action, "reemit_constraint", "action")
    failures += not assert_eq(any("not a known device" in v for v in d.constraint_violations),
                              True, "violation mentions unknown device")

    section("4. routing reason but middle is not an IPv4")
    d = validate_answer(
        draft_answer="Core_SW_01;GE1/0/1;blackhole route",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features=GF,
    )
    # NOTE: format_guard already rejects "routing reason on port line" before
    # we reach the constraint-violation layer. The test exists to confirm
    # the layered guard catches this.
    failures += not assert_eq(d.action in ("reemit_format", "reemit_constraint"), True,
                              "rejected by either format or constraint layer")

    section("5. port reason but middle IS an IPv4")
    d = validate_answer(
        draft_answer="Core_SW_01;192.168.1.5;shutdown",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features=GF,
    )
    failures += not assert_eq(d.action in ("reemit_format", "reemit_constraint"), True,
                              "rejected (format guard already catches this)")

    section("6. HRP claimed on a non-firewall device")
    d = validate_answer(
        draft_answer="Core_SW_01;192.168.1.5;global HRP hot redundancy protocol not enabled",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features={"Core_SW_01": {}},
    )
    failures += not assert_eq(d.action, "reemit_constraint", "action")
    failures += not assert_eq(any("HRP" in v for v in d.constraint_violations),
                              True, "violation mentions HRP")

    section("7. well-formed, no anomaly evidence, primary not yet executed → fetch_evidence")
    d = validate_answer(
        draft_answer="Core_SW_01;192.168.1.5;blackhole route",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features={"Core_SW_01": {}},
        anomaly_evidence=set(),
        denied_pairs=set(),
        fetched_commands=set(),
    )
    failures += not assert_eq(d.action, "fetch_evidence", "action")
    failures += not assert_eq(d.follow_up[0], "Core_SW_01", "follow-up device")
    failures += not assert_eq(
        d.follow_up[1] in ("display ip routing-table", "display current-configuration | include ip route-static"),
        True,
        "follow-up command from playbook",
    )
    failures += not assert_eq(d.follow_up[2], "blackhole route", "follow-up reason")

    section("8. well-formed + anomaly evidence present → accept")
    d = validate_answer(
        draft_answer="Core_SW_01;192.168.1.5;blackhole route",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features={"Core_SW_01": {}},
        anomaly_evidence={("scn", "Core_SW_01", "blackhole route")},
    )
    failures += not assert_eq(d.action, "accept", "action")
    failures += not assert_eq(len(d.accepted_lines), 1, "1 accepted line")

    section("9. well-formed, diagnostic already executed → accept (no repeat fetch)")
    d = validate_answer(
        draft_answer="Core_SW_01;192.168.1.5;blackhole route",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features={"Core_SW_01": {}},
        anomaly_evidence=set(),
        fetched_commands={("Core_SW_01", "display ip routing-table")},
    )
    failures += not assert_eq(d.action, "accept", "action — no repeat fetch when primary cmd already run")

    section("10. XGBoost low calibrated_score triggers fetch_evidence")
    d = validate_answer(
        draft_answer="Core_SW_01;192.168.1.5;blackhole route",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features={"Core_SW_01": {}},
        anomaly_evidence={("scn", "Core_SW_01", "blackhole route")},  # has anomaly
        calibrated_scores={("scn", "Core_SW_01", "blackhole route"): (0.30, 0.05)},
        tau_score=0.5,
        tau_unc=0.2,
    )
    failures += not assert_eq(d.action, "fetch_evidence", "action")
    failures += not assert_eq(
        "calibrated_score" in d.follow_up_rationale,
        True,
        "rationale mentions calibrated_score",
    )

    section("11. high uncertainty triggers fetch_evidence")
    d = validate_answer(
        draft_answer="Core_SW_01;192.168.1.5;blackhole route",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features={"Core_SW_01": {}},
        anomaly_evidence={("scn", "Core_SW_01", "blackhole route")},
        calibrated_scores={("scn", "Core_SW_01", "blackhole route"): (0.90, 0.40)},
        tau_score=0.5,
        tau_unc=0.2,
    )
    failures += not assert_eq(d.action, "fetch_evidence", "action")

    section("12. follow-up command never repeats")
    d = validate_answer(
        draft_answer="Core_SW_01;192.168.1.5;blackhole route",
        scenario_id="scn",
        parsed=ParsedConstraints(),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        graph_features={"Core_SW_01": {}},
        anomaly_evidence=set(),
        fetched_commands={
            ("Core_SW_01", "display ip routing-table"),
            ("Core_SW_01", "display current-configuration | include ip route-static"),
        },
    )
    # All playbook commands for this reason already executed; validator
    # has no follow-up to propose, so it accepts (or the harness uses this
    # signal to escalate). With no remaining commands the validator
    # accepts the well-formed answer.
    failures += not assert_eq(d.action, "accept",
                              "action — accepts when no fresh follow-up is available")

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
