"""Run the deterministic ranker over every Phase 1 scenario and emit
`work/ranked_candidates.csv` for Step 8 / Step 9 to consume.

Joins:
    - work/scenario_manifest.csv               (scenario_id, question_number)
    - work/anomaly_candidates.csv              (anomaly priors)
    - work/graph_features.csv                  (per-device graph features)
    - data/Phase_1/test.json + Phase_2         (question text → constraints
                                                + per-question fault vocab)

For Phase 2 scenarios (no offline bundle) the driver still emits a
sentinel row with `offline_bundle_missing=1` so Step 8's join is uniform.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.constraint_parser import parse as parse_constraints
from track_b.prompt_context import PHASE_2_DEVICES
from track_b.ranker import (
    Candidate,
    SCORED_FIELDS,
    build_candidate_pool,
    score_candidates,
    scored_to_row,
)
from track_b.task_classifier import classify
from track_b.vocab_extractor import extract_fault_vocab

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "work" / "scenario_manifest.csv"
ANOMALY = ROOT / "work" / "anomaly_candidates.csv"
GRAPH = ROOT / "work" / "graph_features.csv"
P1 = ROOT / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json"
P2 = ROOT / "telco_data" / "Track B" / "data" / "Phase_2" / "test.json"
OUT = ROOT / "work" / "ranked_candidates.csv"

OUT_FIELDS = SCORED_FIELDS + ("offline_bundle_missing",)

# Phase 2 seed-pool infrastructure. Without graph features, the seed pool
# would otherwise only contain devices the question text substring-matches
# (almost always the client endpoint); the actual fault is along the
# routing path. Always seed the Phase 2 infrastructure backbone so the
# LLM sees plausible fault-location hypotheses regardless of question
# wording, minus any device the question blacklists.
_PHASE_2_INFRASTRUCTURE: frozenset[str] = frozenset({
    "FW_01", "FW_02",
    "Core_SW_01", "Core_SW_02",
    "AGG_SW_01", "AGG_SW_02", "AGG_SW_03", "AGG_SW_04",
    "PE1", "PE2", "PE3",
    "BJHQ_CSR1000V_GW_01",
    "SH_AR", "SH_Core", "SZ_AR", "SZ_Core",
    "SW-DMZ-ACC-01", "ChinaUnicom_SW",
})

# Per-device-class affinity boosts. Without graph features every
# (device, reason) seed scores identically, so stable-sort puts
# alphabetically-first reasons on top and `_select_top_hypotheses(k=3)`
# never sees device-archetypal reasons. These boosts push each class's
# archetypal reasons above the tie-cluster.
#
# Boost magnitudes are staggered so cross-class ties resolve by class
# specificity (firewall-only reasons > L2-switch reasons > PE > branch).
# This guarantees HRP / security-policy land in the LLM-visible top-3 of
# any Phase 2 fault scenario whose routing path crosses a firewall.
#
# NAT-related reasons are deliberately omitted from the firewall list
# per the Phase 2 prompt hint "the Huawei firewall ... is not planned
# to deploy NAT functionality".
_DEVICE_CLASS_AFFINITY: tuple[tuple[frozenset[str], frozenset[str], float], ...] = (
    # Firewalls — highest specificity; HRP and security-policy have no
    # plausible alternative device class
    (frozenset({"FW_01", "FW_02"}), frozenset({
        "global HRP hot redundancy protocol not enabled",
        "security policy rule not permitting corresponding users",
        "IP address prefix list missing corresponding user source IP address",
        "host information collection function missing",
    }), 1.0),
    # Layer-2 / aggregation switches
    (frozenset({
        "Core_SW_01", "Core_SW_02",
        "AGG_SW_01", "AGG_SW_02", "AGG_SW_03", "AGG_SW_04",
        "SW-DMZ-ACC-01",
    }), frozenset({
        "global STP not enabled",
        "port STP not enabled",
        "interface VLAN configuration error",
        "MAC address configuration error",
    }), 0.6),
    # Provider-edge / WAN routers (overlay + L2/L3VPN territory)
    (frozenset({"PE1", "PE2", "PE3", "BJHQ_CSR1000V_GW_01"}), frozenset({
        "L3VPN configuration error",
        "L2VPN configuration error",
        "ISIS configuration error",
        "SRV6-Policy tunnel planning error",
        "BGP configuration error",
        "VPN configuration missing",
    }), 0.6),
    # Branch access routers / cores
    (frozenset({"SH_AR", "SZ_AR", "SH_Core", "SZ_Core"}), frozenset({
        "BGP configuration error",
        "OSPF configuration error",
        "ISIS configuration error",
    }), 0.4),
)


def _affinity_boost(node: str, fault_reason: str) -> float:
    for device_set, reason_set, magnitude in _DEVICE_CLASS_AFFINITY:
        if node in device_set and fault_reason in reason_set:
            return magnitude
    return 0.0


def _load_questions() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in (P1, P2):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            out[item["scenario_id"]] = item["task"]["question"]
    return out


def _load_anomaly() -> tuple[set[tuple[str, str, str]], dict[tuple[str, str, str], int]]:
    anom_set: set[tuple[str, str, str]] = set()
    anom_strength: dict[tuple[str, str, str], int] = {}
    with open(ANOMALY, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["offline_bundle_missing"] == "1":
                continue
            key = (r["scenario_id"], r["node"], r["fault_reason"])
            anom_set.add(key)
            try:
                anom_strength[key] = int(r["evidence_strength_num"])
            except (TypeError, ValueError):
                anom_strength[key] = 0
    return anom_set, anom_strength


def _load_graph_features() -> dict[str, dict[str, dict]]:
    """scenario_id -> {node -> feature dict}."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(GRAPH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["offline_bundle_missing"] == "1":
                continue
            sid = r["scenario_id"]
            node = r["node"]
            if not node:
                continue
            out[sid][node] = r
    return out


def main() -> int:
    if not MANIFEST.is_file() or not ANOMALY.is_file() or not GRAPH.is_file():
        print("missing prerequisite artifact; run earlier steps first")
        return 1
    questions = _load_questions()
    anom_set, anom_strength = _load_anomaly()
    graph = _load_graph_features()

    rows_out: list[dict] = []
    bundled = 0
    missing = 0
    nonfault = 0
    p2_seeded = 0
    p2_seed_rows = 0
    t0 = time.perf_counter()

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    for r in manifest_rows:
        scenario_id = r["scenario_id"]
        question_number = int(r["question_number"])
        phase = int(r["phase"])
        has_bundle = r["has_static_bundle"].lower() == "true"
        question_text = questions.get(scenario_id, "")

        if not has_bundle:
            rows_out.append({k: "" for k in OUT_FIELDS})
            sentinel = rows_out[-1]
            sentinel["scenario_id"] = scenario_id
            sentinel["question_number"] = question_number
            sentinel["phase"] = phase
            sentinel["node"] = ""
            sentinel["fault_reason"] = ""
            sentinel["category"] = ""
            sentinel["combined_score"] = 0.0
            sentinel["rank"] = -1
            sentinel["offline_bundle_missing"] = 1
            for fld in (
                "graph_centrality_raw", "graph_centrality_norm",
                "path_relevance_raw", "path_relevance_norm",
                "protocol_match_raw", "protocol_match_norm",
                "vendor_compat_raw", "vendor_compat_norm",
                "anomaly_prior_raw", "anomaly_prior_norm",
                "permission_survivor_raw", "permission_survivor_norm",
                "disclosed_match_raw", "disclosed_match_norm",
                "contradiction_penalty_raw", "contradiction_penalty_norm",
            ):
                sentinel[fld] = 0.0
            missing += 1

            # Phase 2 seed pool: without static bundles the candidate pool
            # would otherwise be empty, leaving the LLM to invent (node, reason)
            # pairs unaided. Seed it with the question-mentioned Phase 2
            # devices crossed with the full per-question vocab, so the LLM
            # sees concrete hypotheses (notably HRP on firewalls) in
            # prompt_context. The sentinel row above is preserved for Step 8
            # join uniformity; seed rows carry offline_bundle_missing=0 so
            # run_submission._load_ranked_with_scores ingests them as
            # RankedRow objects.
            if phase != 2 or classify(question_text) != "fault":
                continue
            fault_vocab = extract_fault_vocab(question_text)
            routing_reasons, port_reasons = fault_vocab
            if not (routing_reasons or port_reasons):
                continue
            parsed = parse_constraints(question_text)
            mentioned = [d for d in PHASE_2_DEVICES if d in question_text]
            # Always include the Phase 2 infrastructure backbone — the actual
            # fault is along the routing path, not on the named client.
            mentioned_set: set[str] = set(mentioned) | set(_PHASE_2_INFRASTRUCTURE)
            # Also union in parsed.disclosed_fault_nodes / fault_candidate_nodes
            # in case the constraint parser caught names the substring match
            # missed.
            mentioned_set.update(parsed.disclosed_fault_nodes)
            mentioned_set.update(parsed.fault_candidate_nodes)
            mentioned_set = {n for n in mentioned_set if n in PHASE_2_DEVICES}
            # Subtract devices the question explicitly excludes via a
            # "Limitation: Do not look for faults on X" clause. Phase 2 q3,
            # q38–40 use this to exclude FW_02; seeding (FW_02, *) would
            # give the LLM a hypothesis the question forbids.
            mentioned_set -= set(parsed.blacklisted_nodes)
            if not mentioned_set:
                continue
            seed_candidates: list[Candidate] = []
            for node in sorted(mentioned_set):
                for reason in routing_reasons:
                    seed_candidates.append(Candidate(
                        scenario_id=scenario_id,
                        question_number=question_number,
                        phase=phase,
                        node=node,
                        fault_reason=reason,
                        category="routing",
                    ))
                for reason in port_reasons:
                    seed_candidates.append(Candidate(
                        scenario_id=scenario_id,
                        question_number=question_number,
                        phase=phase,
                        node=node,
                        fault_reason=reason,
                        category="port",
                    ))
            # Score with empty graph_features and no anomaly priors. Most
            # components will be 0; protocol_match, vendor_compat, and
            # disclosed_match still fire from constraint parsing alone.
            seed_scored = score_candidates(
                seed_candidates,
                parsed=parsed,
                graph_features={},
                anomaly_priors={},
            )
            for s in seed_scored:
                s.combined_score += _affinity_boost(s.candidate.node, s.candidate.fault_reason)
                row = scored_to_row(s)
                row["offline_bundle_missing"] = 0
                rows_out.append(row)
                p2_seed_rows += 1
            p2_seeded += 1
            continue

        if classify(question_text) != "fault":
            # The ranker is fault-classification-specific; non-fault tasks
            # (path/topology) need their own ranker stack. Skip for now.
            nonfault += 1
            continue

        parsed = parse_constraints(question_text)
        fault_vocab = extract_fault_vocab(question_text)
        gf = graph.get(scenario_id, {})
        devices = list(gf.keys())
        if not devices:
            continue

        # On-path devices come from graph_features
        on_path = {n for n, d in gf.items() if d.get("on_parsed_path") == "1"}
        centrality_norm = {
            n: float(d.get("betweenness_norm") or 0.0) for n, d in gf.items()
        }

        candidates = build_candidate_pool(
            scenario_id=scenario_id,
            question_number=question_number,
            phase=phase,
            devices=devices,
            parsed=parsed,
            on_path_devices=on_path,
            anomaly_set=anom_set,
            fault_vocab=fault_vocab,
            centrality_norm=centrality_norm,
        )

        scored = score_candidates(
            candidates,
            parsed=parsed,
            graph_features=gf,
            anomaly_priors=anom_strength,
        )
        for s in scored:
            row = scored_to_row(s)
            row["offline_bundle_missing"] = 0
            rows_out.append(row)
        bundled += 1

    dt = time.perf_counter() - t0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for row in rows_out:
            w.writerow({k: row.get(k, "") for k in OUT_FIELDS})

    print(f"wrote {OUT} ({len(rows_out)} rows in {dt:.1f}s)")
    print(f"  scenarios bundled+fault:   {bundled}")
    print(f"  scenarios bundled non-fault (skipped): {nonfault}")
    print(f"  scenarios missing bundle:  {missing}")
    print(f"  Phase 2 scenarios seeded:  {p2_seeded}  (seed rows: {p2_seed_rows})")
    fault_rows = [r for r in rows_out if r.get("offline_bundle_missing", 0) == 0 and r.get("node")]
    if fault_rows:
        per_scen: dict[str, int] = defaultdict(int)
        for r in fault_rows:
            per_scen[r["scenario_id"]] += 1
        sizes = list(per_scen.values())
        sizes.sort()
        print(f"  candidates per fault scenario: avg={sum(sizes)/len(sizes):.1f} "
              f"min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")
        # Top-1 disclosed_match=1 hit-rate as a smoke gauge
        top1_disclosed = 0
        n_with_disclosed = 0
        for sid, _n in per_scen.items():
            scen_rows = [r for r in fault_rows if r["scenario_id"] == sid]
            scen_rows.sort(key=lambda r: int(r["rank"]))
            if any(float(r["disclosed_match_raw"]) > 0 for r in scen_rows):
                n_with_disclosed += 1
                if scen_rows and float(scen_rows[0]["disclosed_match_raw"]) > 0:
                    top1_disclosed += 1
        print(f"  scenarios with any disclosed_match>0: {n_with_disclosed}")
        print(f"  scenarios where top-1 has disclosed_match>0: {top1_disclosed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
