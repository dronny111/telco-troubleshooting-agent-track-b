"""Drive the offline anomaly miner over the scenario manifest.

Reads `work/scenario_manifest.csv`, mines every Phase 1 scenario with a
local bundle, and emits candidate rows for Phase 2 too (with
`offline_bundle_missing=1`). Output: `work/anomaly_candidates.csv`.

Run after `build_scenario_manifest.py`.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.anomaly_miner import (
    Candidate,
    emit_missing_bundle_row,
    mine_scenario,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "work" / "scenario_manifest.csv"
DEVICES_OUTPUTS = ROOT / "telco_data" / "Track B" / "devices_outputs"
OUT = ROOT / "work" / "anomaly_candidates.csv"


def main() -> int:
    if not MANIFEST.is_file():
        print(f"manifest missing: {MANIFEST}; run work/build_scenario_manifest.py first")
        return 1
    rows: list[dict] = []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    candidates: list[Candidate] = []
    bundled = 0
    missing = 0
    t0 = time.perf_counter()
    for r in rows:
        scenario_id = r["scenario_id"]
        question_number = int(r["question_number"])
        phase = int(r["phase"])
        has_bundle = r["has_static_bundle"].lower() == "true"
        if has_bundle:
            cands = mine_scenario(
                scenario_id=scenario_id,
                question_number=question_number,
                phase=phase,
                devices_outputs_root=DEVICES_OUTPUTS,
            )
            candidates.extend(cands)
            bundled += 1
        else:
            candidates.append(
                emit_missing_bundle_row(
                    scenario_id=scenario_id,
                    question_number=question_number,
                    phase=phase,
                )
            )
            missing += 1
    dt = time.perf_counter() - t0

    write_csv(candidates, OUT)

    print(f"wrote {OUT} ({len(candidates)} rows in {dt:.1f}s)")
    print(f"  scenarios mined:           {bundled}")
    print(f"  scenarios bundle-missing:  {missing}")
    print()
    fault_cands = [c for c in candidates if not c.offline_bundle_missing]
    by_phase = Counter(c.phase for c in fault_cands)
    print(f"  candidates by phase:       {dict(by_phase)}")
    by_reason = Counter(c.fault_reason for c in fault_cands)
    print(f"  top 10 fault_reasons fired:")
    for r, n in by_reason.most_common(10):
        print(f"    {n:>4}  {r}")
    by_category = Counter(c.category for c in fault_cands)
    print(f"  by category:               {dict(by_category)}")
    by_strength = Counter(c.evidence_strength for c in fault_cands)
    print(f"  by evidence_strength:      {dict(by_strength)}")
    nodes_per_scenario = Counter()
    for c in fault_cands:
        nodes_per_scenario[c.scenario_id] += 1
    if nodes_per_scenario:
        avg_per_scenario = sum(nodes_per_scenario.values()) / len(nodes_per_scenario)
        print(f"  candidates/scenario (Phase 1 only): avg={avg_per_scenario:.1f} max={max(nodes_per_scenario.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
