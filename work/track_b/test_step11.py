"""Smoke test for the Step 11 eval harness.

Asserts:
    - Variant configs are present and progressive (V0 ⊂ V1 ⊂ ... ⊂ V4 in
      the feature toggles).
    - run_eval returns one VariantReport per variant with all metrics.
    - Stub V0 produces empty answers → format_error_rate = 1.0,
      accuracy = 0.0.
    - V3 / V4 trigger validator follow-ups when no anomaly evidence is
      available for the top pick (mean_calls > 0).
    - ablate_one_signal returns a VariantReport whose name carries the
      `-no-<component>` suffix.
    - variance_across_reruns returns std=0 for the deterministic stub.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.eval_harness import (
    ABLATABLE_COMPONENTS,
    ScenarioInputs,
    VARIANTS,
    ablate_one_signal,
    run_eval,
    run_variant_on_scenario,
    variance_across_reruns,
)


# Canonical Phase-2 vocab subset for testing
ROUTING_VOCAB = (
    "blackhole route", "missing static route", "BGP configuration error",
    "OSPF configuration error", "ARP configuration error",
    "global HRP hot redundancy protocol not enabled",
)
PORT_VOCAB = ("shutdown", "interface IP error")


def _make_scenario(
    *,
    sid: str,
    silver: set[tuple[str, str]],
    candidates_det: list[dict] | None = None,
    candidates_xgb: list[dict] | None = None,
    anomaly_set: set[tuple[str, str, str]] | None = None,
) -> ScenarioInputs:
    return ScenarioInputs(
        scenario_id=sid,
        question_number=1,
        phase=1,
        question_text=(
            "...interface IP error\n"
            "From SRC, accessing 10.0.0.1 failed."
        ),
        routing_vocab=ROUTING_VOCAB,
        port_vocab=PORT_VOCAB,
        candidates_deterministic=candidates_det or [],
        candidates_xgb=candidates_xgb or [],
        graph_features={
            "Core_SW_01": {"on_parsed_path": "1"},
            "PE1": {"on_parsed_path": "1"},
        },
        anomaly_evidence=anomaly_set or set(),
        denied_pairs=set(),
        silver_positives=silver,
    )


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def assert_eq(a, b, label: str) -> bool:
    ok = a == b
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: a={a!r} b={b!r}")
    return ok


def main() -> int:
    failures = 0

    section("variant catalogue is progressive")
    for stronger, weaker in (("V1", "V0"), ("V2", "V1"), ("V3", "V2"), ("V4", "V3")):
        cs = VARIANTS[stronger]
        cw = VARIANTS[weaker]
        # Every toggle on in the weaker variant must remain on in the stronger.
        toggles = (
            "use_constraint_parser", "use_playbook", "use_graph",
            "use_permission_pruner", "use_deterministic_ranker",
            "use_anomaly_miner", "use_answer_validator", "use_xgboost",
        )
        progressed = all(getattr(cs, t) >= getattr(cw, t) for t in toggles)
        failures += not assert_eq(progressed, True, f"{stronger} ⊇ {weaker}")

    section("V0 — empty answer, all scenarios fail format")
    s = _make_scenario(sid="t1", silver={("Core_SW_01", "blackhole route")})
    [r0] = run_eval([VARIANTS["V0"]], [s])
    failures += not assert_eq(r0.format_error_rate, 1.0, "V0 format_error_rate")
    failures += not assert_eq(r0.n_exact, 0, "V0 exact match")

    section("V2 — top-1 from deterministic ranker scores correctly")
    cands_det = [
        {"node": "Core_SW_01", "fault_reason": "blackhole route", "category": "routing",
         "combined_score": 5.0, "graph_centrality_norm": 0.9, "path_relevance_norm": 1.0,
         "protocol_match_norm": 0.5, "anomaly_prior_norm": 1.0,
         "permission_survivor_norm": 1.0, "disclosed_match_norm": 0.0,
         "contradiction_penalty_raw": 0.0},
        {"node": "PE1", "fault_reason": "BGP configuration error", "category": "routing",
         "combined_score": 3.0, "graph_centrality_norm": 0.5, "path_relevance_norm": 0.5,
         "protocol_match_norm": 0.0, "anomaly_prior_norm": 0.0,
         "permission_survivor_norm": 1.0, "disclosed_match_norm": 0.0,
         "contradiction_penalty_raw": 0.0},
    ]
    s_v2 = _make_scenario(
        sid="t2",
        silver={("Core_SW_01", "blackhole route")},
        candidates_det=cands_det,
        anomaly_set={("t2", "Core_SW_01", "blackhole route")},
    )
    [r2] = run_eval([VARIANTS["V2"]], [s_v2])
    failures += not assert_eq(r2.n_top1, 1, "V2 picks Core_SW_01 (top-1)")
    failures += not assert_eq(r2.format_error_rate, 0.0, "V2 format clean")

    section("V3 — validator triggers a follow-up when no anomaly evidence")
    s_v3 = _make_scenario(
        sid="t3",
        silver={("Core_SW_01", "blackhole route")},
        candidates_det=cands_det,
        anomaly_set=set(),  # no evidence
    )
    trace = run_variant_on_scenario(variant=VARIANTS["V3"], inputs=s_v3)
    failures += not assert_eq(trace.n_follow_ups >= 1, True, "V3 fires at least one follow-up")
    failures += not assert_eq(trace.n_calls == trace.n_follow_ups, True,
                              "n_calls equals n_follow_ups for the stub")

    section("V4 — XGBoost calibrated_score below threshold triggers follow-up")
    cands_xgb = [
        {"node": "Core_SW_01", "fault_reason": "blackhole route", "category": "routing",
         "calibrated_score": 0.20, "uncertainty": 0.05, "combined_score": 5.0},
    ]
    s_v4 = _make_scenario(
        sid="t4",
        silver={("Core_SW_01", "blackhole route")},
        candidates_det=cands_det,
        candidates_xgb=cands_xgb,
        anomaly_set={("t4", "Core_SW_01", "blackhole route")},
    )
    trace4 = run_variant_on_scenario(variant=VARIANTS["V4"], inputs=s_v4)
    failures += not assert_eq(
        trace4.n_follow_ups >= 1,
        True,
        "V4 fires follow-up because calibrated_score=0.20 < tau_score=0.5",
    )

    section("ablate_one_signal returns labelled report")
    rep = ablate_one_signal(
        base_variant=VARIANTS["V2"], scenarios=[s_v2], component="anomaly_prior_norm",
    )
    failures += not assert_eq(
        rep.variant, "V2-no-anomaly_prior_norm", "ablation report variant label",
    )

    section("variance_across_reruns is 0 for deterministic stub")
    var = variance_across_reruns(VARIANTS["V2"], [s_v2], n_reruns=3)
    failures += not assert_eq(var["std_accuracy"], 0.0, "deterministic stub variance")

    section("ABLATABLE_COMPONENTS covers the ranker's score columns")
    must_include = {"path_relevance_norm", "anomaly_prior_norm", "disclosed_match_norm"}
    failures += not assert_eq(
        must_include.issubset(set(ABLATABLE_COMPONENTS)),
        True,
        "key components are ablatable",
    )

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
