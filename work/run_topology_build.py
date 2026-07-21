"""Build typed graphs + per-device features for every Phase 1 scenario.

Reads `work/scenario_manifest.csv`, builds a `ScenarioGraph` for each
Phase 1 scenario with a local bundle, parses the question text via
the constraint parser, and writes per-device features to
`work/graph_features.csv`. For Phase 2 scenarios (no local bundle) it
emits sentinel rows with `offline_bundle_missing=1` and zeroed features.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.constraint_parser import parse as parse_constraints
from track_b.graph_features import (
    DeviceFeatures,
    FEATURE_FIELDS,
    extract_device_features,
    feature_to_row,
)
from track_b.topology import build_scenario_graph

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "work" / "scenario_manifest.csv"
DEVICES = ROOT / "telco_data" / "Track B" / "devices_outputs"
LIMITS = ROOT / "telco_data" / "Track B" / "question_limits_config.json"
P1 = ROOT / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json"
P2 = ROOT / "telco_data" / "Track B" / "data" / "Phase_2" / "test.json"
OUT = ROOT / "work" / "graph_features.csv"

OUT_FIELDS = tuple(FEATURE_FIELDS) + ("offline_bundle_missing",)


def _load_questions() -> dict[str, str]:
    """Return scenario_id → question text for both phases."""
    out: dict[str, str] = {}
    for path in (P1, P2):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            out[item["scenario_id"]] = item["task"]["question"]
    return out


def main() -> int:
    if not MANIFEST.is_file():
        print(f"manifest missing: {MANIFEST}; run work/build_scenario_manifest.py first")
        return 1
    questions = _load_questions()

    rows_in: list[dict] = []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))

    out_rows: list[dict] = []
    bundled = 0
    missing = 0
    no_question = 0
    t0 = time.perf_counter()

    for r in rows_in:
        scenario_id = r["scenario_id"]
        question_number = int(r["question_number"])
        phase = int(r["phase"])
        has_bundle = r["has_static_bundle"].lower() == "true"
        question_text = questions.get(scenario_id, "")
        if not question_text:
            no_question += 1

        if has_bundle:
            sg = build_scenario_graph(
                scenario_id=scenario_id,
                question_number=question_number,
                phase=phase,
                devices_outputs_root=DEVICES,
            )
            if sg is None:
                missing += 1
                continue
            parsed = parse_constraints(question_text)
            feats = extract_device_features(sg, parsed, question_limits_path=LIMITS)
            for f in feats:
                row = feature_to_row(f)
                row["offline_bundle_missing"] = 0
                out_rows.append(row)
            bundled += 1
        else:
            sentinel = DeviceFeatures(
                scenario_id=scenario_id,
                question_number=question_number,
                phase=phase,
                node="",
            )
            row = feature_to_row(sentinel)
            row["offline_bundle_missing"] = 1
            out_rows.append(row)
            missing += 1

    dt = time.perf_counter() - t0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"wrote {OUT} ({len(out_rows)} rows in {dt:.1f}s)")
    print(f"  scenarios bundled:  {bundled}")
    print(f"  scenarios missing:  {missing}")
    if no_question:
        print(f"  WARN scenarios without question text: {no_question}")
    feat_rows = [r for r in out_rows if int(r["offline_bundle_missing"]) == 0]
    print(f"  device-feature rows (Phase 1): {len(feat_rows)}")
    if feat_rows:
        on_path = sum(int(r["on_parsed_path"]) for r in feat_rows)
        blk = sum(int(r["is_blacklisted"]) for r in feat_rows)
        disc = sum(int(r["is_disclosed_fault"]) for r in feat_rows)
        denied_pos = sum(1 for r in feat_rows if int(r["denied_command_count"]) > 0)
        srcs_resolved = sum(1 for r in feat_rows if int(r["hop_distance_source"]) >= 0)
        dsts_resolved = sum(1 for r in feat_rows if int(r["hop_distance_dest"]) >= 0)
        print(f"  on_parsed_path=1:                 {on_path}")
        print(f"  is_blacklisted=1:                 {blk}")
        print(f"  is_disclosed_fault=1:             {disc}")
        print(f"  denied_command_count>0:           {denied_pos}")
        print(f"  rows w/ hop_distance_source>=0:   {srcs_resolved}")
        print(f"  rows w/ hop_distance_dest>=0:     {dsts_resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
