"""Smoke test for the prompt-context builder.

Asserts:
    - build_context returns the expected top-level keys.
    - Top-K selection prefers calibrated_score when present.
    - blacklisted nodes are excluded from next_best_commands but still
      surfaced in constraints.
    - Denied (device, command) pairs land in hard_blocked and are NOT
      proposed as next-best commands.
    - Anomaly evidence lines are attached when available.
    - build_system_prompt embeds the JSON, the rules, and (optionally)
      the Phase 2 device vocabulary; estimated token count is bounded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.constraint_parser import ParsedConstraints
from track_b.prompt_context import (
    AnomalyEvidence,
    PHASE_2_DEVICES,
    RankedRow,
    approx_token_count,
    build_context,
    build_system_prompt,
)


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def assert_eq(a, b, label: str) -> bool:
    ok = a == b
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: a={a!r} b={b!r}")
    return ok


def main() -> int:
    failures = 0

    section("build_context — basic shape")
    parsed = ParsedConstraints(
        source_endpoint="GUEST_WIFI_CLIENT01",
        target_destination_ip="10.1.60.2",
        blacklisted_nodes=("FW_02",),
        suspected_protocol_families=("BGP",),
    )
    cands = [
        RankedRow("scn", "PE1", "BGP configuration error", "routing", 4.5),
        RankedRow("scn", "Core_SW_01", "ARP configuration error", "routing", 3.0),
        RankedRow("scn", "FW_02", "security policy rule not permitting corresponding users", "routing", 2.5),
        RankedRow("scn", "AGG_SW_01", "VXLAN configuration error", "routing", 2.0),
    ]
    ctx = build_context(
        task_family="fault",
        parsed=parsed,
        candidates=cands,
        denied_pairs=[("PE1", "display bgp routing-table")],
    )
    failures += not assert_eq(
        sorted(ctx.keys()),
        ["constraints", "hard_blocked", "next_best_commands", "task_family", "top_hypotheses"],
        "top-level keys",
    )
    failures += not assert_eq(ctx["task_family"], "fault", "task_family")
    failures += not assert_eq(ctx["constraints"]["source"], "GUEST_WIFI_CLIENT01", "constraint source")
    failures += not assert_eq(ctx["constraints"]["blacklisted_nodes"], ["FW_02"], "blacklist propagated")
    failures += not assert_eq(len(ctx["top_hypotheses"]), 3, "top-3 hypotheses")
    nodes = [h["node"] for h in ctx["top_hypotheses"]]
    failures += not assert_eq(nodes[0], "PE1", "top-1 by combined_score")
    failures += not assert_eq(
        ("FW_02" in nodes),
        True,
        "blacklisted node present in top hypotheses (XGBoost will demote later)",
    )

    section("denied + blacklist exclusion in next_best_commands")
    cmd_pairs = [(c["device"], c["command"]) for c in ctx["next_best_commands"]]
    failures += not assert_eq(
        ("PE1", "display bgp routing-table") in cmd_pairs,
        False,
        "denied pair excluded from next_best",
    )
    failures += not assert_eq(
        any(d == "FW_02" for d, _c in cmd_pairs),
        False,
        "blacklisted node excluded from next_best",
    )
    failures += not assert_eq(
        ("PE1", "display bgp routing-table") in [(b["device"], b["command"]) for b in ctx["hard_blocked"]],
        True,
        "denied pair surfaced in hard_blocked",
    )

    section("calibrated_score wins over combined_score")
    cands_calib = [
        RankedRow("scn", "PE1", "BGP configuration error", "routing", 4.5,
                  calibrated_score=0.40, uncertainty=0.10),
        RankedRow("scn", "Core_SW_01", "ARP configuration error", "routing", 3.0,
                  calibrated_score=0.85, uncertainty=0.05),
    ]
    ctx2 = build_context(
        task_family="fault",
        parsed=ParsedConstraints(),
        candidates=cands_calib,
    )
    failures += not assert_eq(
        ctx2["top_hypotheses"][0]["node"],
        "Core_SW_01",
        "calibrated_score=0.85 ranks above 0.40",
    )
    failures += not assert_eq(
        "score" in ctx2["top_hypotheses"][0] and "unc" in ctx2["top_hypotheses"][0],
        True,
        "calibrated path emits score+unc",
    )

    section("anomaly evidence attached")
    ev = AnomalyEvidence(
        scenario_id="scn",
        node="Core_SW_01",
        fault_reason="ARP configuration error",
        sample_evidence="10.3.1.2 Incomplete 1 D Vlanif1001",
    )
    ctx3 = build_context(
        task_family="fault",
        parsed=ParsedConstraints(),
        candidates=[RankedRow("scn", "Core_SW_01", "ARP configuration error", "routing", 3.0)],
        anomaly_evidence={("scn", "Core_SW_01", "ARP configuration error"): ev},
    )
    failures += not assert_eq(
        ctx3["top_hypotheses"][0]["evidence"],
        "10.3.1.2 Incomplete 1 D Vlanif1001",
        "evidence line surfaced",
    )

    section("system prompt assembly")
    prompt_p1 = build_system_prompt(context=ctx, include_phase_2_device_list=False)
    failures += not assert_eq("Track B output rules" in prompt_p1, True, "rules embedded")
    failures += not assert_eq('"task_family":"fault"' in prompt_p1, True, "compact JSON embedded")
    failures += not assert_eq("Phase 2 known device vocabulary" in prompt_p1, False,
                              "P1 prompt omits P2 device list")

    prompt_p2 = build_system_prompt(context=ctx, include_phase_2_device_list=True)
    failures += not assert_eq("Phase 2 known device vocabulary" in prompt_p2, True,
                              "P2 prompt includes P2 device list")
    failures += not assert_eq(
        all(d in prompt_p2 for d in ("PE1", "FW_01", "Core_SW_01")),
        True,
        "P2 prompt enumerates concrete devices",
    )

    section("token-budget guard")
    n_tok = approx_token_count(prompt_p2)
    failures += not assert_eq(n_tok < 1500, True, f"P2 prompt under 1500 tokens (got {n_tok})")

    section("Phase 2 vocab is non-empty and unique")
    failures += not assert_eq(len(set(PHASE_2_DEVICES)) == len(PHASE_2_DEVICES), True,
                              "device vocab unique")
    failures += not assert_eq(len(PHASE_2_DEVICES) >= 40, True, "device vocab covers >=40 devices")

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
