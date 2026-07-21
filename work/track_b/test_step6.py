"""Smoke test for the deterministic ranker.

Asserts:
    - Component scorers return expected boundaries (e.g. on-path → 1.0).
    - rank_norm produces percentiles in [0, 1] with ties handled.
    - Candidate pool contains all anomaly candidates plus on-path × vocab.
    - Synthetic scenario where one device is on-path AND has anomaly
      evidence ranks #1 over a disjoint device with no signals.
    - Blacklisted node receives a contradiction penalty and falls below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.constraint_parser import ParsedConstraints
from track_b.ranker import (
    Candidate,
    RankerWeights,
    _rank_norm_field,
    _score_anomaly_prior,
    _score_path_relevance,
    _score_protocol_match,
    build_candidate_pool,
    score_candidates,
)


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def assert_eq(a, b, label: str) -> bool:
    ok = a == b
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: a={a!r} b={b!r}")
    return ok


def main() -> int:
    failures = 0

    section("component scorers")
    failures += not assert_eq(_score_path_relevance(on_path=1, hop_src=-1, hop_dst=-1), 1.0, "on_path=1 → 1.0")
    failures += not assert_eq(_score_path_relevance(on_path=0, hop_src=2, hop_dst=-1), 0.5, "hop_src=2 → 0.5")
    failures += not assert_eq(_score_path_relevance(on_path=0, hop_src=-1, hop_dst=-1), 0.0, "no info → 0.0")
    failures += not assert_eq(_score_anomaly_prior(evidence_strength_num=3), 1.0, "high → 1.0")
    failures += not assert_eq(_score_anomaly_prior(evidence_strength_num=2), 0.75, "medium → 0.75")
    failures += not assert_eq(_score_anomaly_prior(evidence_strength_num=0), 0.0, "absent → 0.0")
    failures += not assert_eq(
        _score_protocol_match(fault_reason="BGP configuration error", suspected_protocols=("BGP",)),
        1.0,
        "BGP suspected matches BGP fault",
    )
    failures += not assert_eq(
        _score_protocol_match(fault_reason="OSPF configuration error", suspected_protocols=("BGP",)),
        0.0,
        "OSPF fault, BGP suspected → no match",
    )

    section("rank-norm with ties")
    from dataclasses import dataclass

    # Use the ranker's internal helper directly via a small struct
    from track_b.ranker import ScoredCandidate, ComponentScores

    items = [
        ScoredCandidate(candidate=Candidate("s", 1, 1, f"D{i}", "x", "routing"),
                        raw=ComponentScores(graph_centrality=v),
                        norm=ComponentScores())
        for i, v in enumerate([0.0, 0.5, 0.5, 1.0])
    ]
    _rank_norm_field(items, "graph_centrality")
    norms = [s.norm.graph_centrality for s in items]
    failures += not assert_eq(norms[0], 0.0, "min → 0.0")
    failures += not assert_eq(norms[3], 1.0, "max → 1.0")
    failures += not assert_eq(norms[1], norms[2], "ties get same percentile")

    section("synthetic ranking — on-path + anomaly wins, blacklist drops")
    parsed = ParsedConstraints(
        source_endpoint="HOST_A",
        target_destination_ip="10.0.0.1",
        blacklisted_nodes=("BadNode",),
        suspected_protocol_families=("BGP",),
    )
    devices = ["GoodNode", "OkNode", "BadNode", "BackgroundNode"]
    on_path = {"GoodNode", "OkNode"}
    anomaly_set: set[tuple[str, str, str]] = {
        ("scen", "GoodNode", "BGP configuration error"),
    }
    centrality_norm = {"GoodNode": 0.9, "OkNode": 0.5, "BadNode": 0.4, "BackgroundNode": 0.1}
    fault_vocab = (("BGP configuration error", "OSPF configuration error"), ("shutdown",))

    candidates = build_candidate_pool(
        scenario_id="scen",
        question_number=1,
        phase=1,
        devices=devices,
        parsed=parsed,
        on_path_devices=on_path,
        anomaly_set=anomaly_set,
        fault_vocab=fault_vocab,
        centrality_norm=centrality_norm,
    )
    have = {(c.node, c.fault_reason) for c in candidates}
    failures += not assert_eq(("GoodNode", "BGP configuration error") in have,
                              True, "anomaly candidate present")
    failures += not assert_eq(("OkNode", "BGP configuration error") in have,
                              True, "on-path candidate present")
    failures += not assert_eq(("BadNode", "BGP configuration error") in have,
                              True, "blacklisted node included (penalty applies later)")

    graph_features = {
        "GoodNode":       {"betweenness_norm": 0.9, "on_parsed_path": "1",
                           "hop_distance_source": "1", "hop_distance_dest": "2",
                           "denied_command_count": "0"},
        "OkNode":         {"betweenness_norm": 0.5, "on_parsed_path": "1",
                           "hop_distance_source": "2", "hop_distance_dest": "1",
                           "denied_command_count": "0"},
        "BadNode":        {"betweenness_norm": 0.4, "on_parsed_path": "0",
                           "hop_distance_source": "-1", "hop_distance_dest": "-1",
                           "denied_command_count": "0"},
        "BackgroundNode": {"betweenness_norm": 0.1, "on_parsed_path": "0",
                           "hop_distance_source": "-1", "hop_distance_dest": "-1",
                           "denied_command_count": "0"},
    }
    anom_strength = {("scen", "GoodNode", "BGP configuration error"): 3}

    scored = score_candidates(
        candidates,
        parsed=parsed,
        graph_features=graph_features,
        anomaly_priors=anom_strength,
    )
    by_key = {(s.candidate.node, s.candidate.fault_reason): s for s in scored}
    s_good = by_key[("GoodNode", "BGP configuration error")]
    s_bad = by_key[("BadNode", "BGP configuration error")]
    failures += not assert_eq(s_good.rank, 0, "anomaly + on-path + protocol match ranks #1")
    failures += not assert_eq(s_good.combined_score > s_bad.combined_score, True,
                              "good > bad")
    failures += not assert_eq(s_bad.raw.contradiction_penalty, 1.0,
                              "blacklisted node has contradiction_penalty=1")

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
