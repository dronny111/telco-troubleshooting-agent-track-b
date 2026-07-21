"""Build scenario manifest for Track B Phase 1 + Phase 2.

Establishes the question_number <-> scenario_id linkage and records, per scenario,
whether a static devices_outputs/ bundle is available locally and whether
question_limits_config.json defines per-question permission restrictions.

This is Step 0 of the refined Phase 2 plan. Steps 4-5 (offline anomaly miner +
typed graph) are conditional on `has_static_bundle == True`. For scenarios without
a bundle, the live execution path emits offline_bundle_missing=1 features rather
than fabricating evidence.

Linkage rule (verified from server.py lines 332, 362):
    API parameter `question_number` == test.json `task.id`.
    Local server resolves devices_outputs/{question_number}/{device}/...txt
    or falls back to devices_outputs/others/ (which does not exist locally).

Phase-scoping rule (verified from sample data):
    Local devices_outputs/ folders 1..50 are Phase 1 only — device naming
    (Gamma/Beta/Delta/Atlas/Janus/Aegis Greek-mythology family) and the
    32+22-node Phase 1 networks match. Phase 2 uses a different 40-node
    campus network with names like Core_SW_01, Test-Zone1-Spine-01, FW_02,
    GUEST_WIFI_CLIENT01. Therefore Phase 2 has NO local static bundles
    even though task_ids 1..50 collide numerically with the Phase 1 dirs.
    The same applies to question_limits_config.json (Phase 1 device names
    only).

Outputs:
    work/scenario_manifest.csv  — primary tabular artifact
    work/scenario_manifest.json — same content, dict-of-records form
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK_B = ROOT / "telco_data" / "Track B"
PHASE_1_TEST = TRACK_B / "data" / "Phase_1" / "test.json"
PHASE_2_TEST = TRACK_B / "data" / "Phase_2" / "test.json"
DEVICES_OUTPUTS = TRACK_B / "devices_outputs"
QUESTION_LIMITS = TRACK_B / "question_limits_config.json"

OUT_DIR = ROOT / "work"
OUT_CSV = OUT_DIR / "scenario_manifest.csv"
OUT_JSON = OUT_DIR / "scenario_manifest.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_bundle_devices(question_number: int) -> list[str]:
    bundle = DEVICES_OUTPUTS / str(question_number)
    if not bundle.is_dir():
        return []
    return sorted(d.name for d in bundle.iterdir() if d.is_dir())


def list_bundle_command_files(question_number: int) -> int:
    bundle = DEVICES_OUTPUTS / str(question_number)
    if not bundle.is_dir():
        return 0
    n = 0
    for device_dir in bundle.iterdir():
        if not device_dir.is_dir():
            continue
        n += sum(1 for f in device_dir.iterdir() if f.suffix == ".txt")
    return n


def question_limit_summary(question_number: int, limits: dict) -> tuple[bool, int, list[str]]:
    """Return (has_restrictions, denied_command_count, denied_command_list)."""
    key = f"question_{question_number}"
    entry = limits.get(key, {})
    no_perm = entry.get("no_permission", {})
    denied_commands = sorted(no_perm.keys())
    total_denied_pairs = sum(len(devs) for devs in no_perm.values())
    return bool(no_perm), total_denied_pairs, denied_commands


def build_rows(phase: int, test_path: Path, limits: dict) -> list[dict]:
    items = load_json(test_path)
    rows: list[dict] = []
    # Local devices_outputs/ + question_limits_config.json describe Phase 1 only.
    # For Phase 2, all offline-derived columns are forced to "missing" regardless
    # of any numeric task_id collision with Phase 1 directory names.
    phase_has_local_bundles = phase == 1
    for item in items:
        task_id = int(item["task"]["id"])
        question_number = task_id  # confirmed linkage
        scenario_id = item["scenario_id"]
        if phase_has_local_bundles:
            devices = list_bundle_devices(question_number)
            cmd_files = list_bundle_command_files(question_number)
            has_lim, denied_pairs, denied_cmds = question_limit_summary(question_number, limits)
        else:
            devices = []
            cmd_files = 0
            has_lim, denied_pairs, denied_cmds = False, 0, []
        has_bundle = bool(devices)
        rows.append(
            {
                "phase": phase,
                "scenario_id": scenario_id,
                "question_number": question_number,
                "task_id": task_id,
                "has_static_bundle": has_bundle,
                "bundle_device_count": len(devices),
                "bundle_command_file_count": cmd_files,
                "has_question_limits": has_lim,
                "denied_device_command_pairs": denied_pairs,
                "denied_commands": ";".join(denied_cmds),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    limits = load_json(QUESTION_LIMITS)

    rows: list[dict] = []
    rows.extend(build_rows(1, PHASE_1_TEST, limits))
    rows.extend(build_rows(2, PHASE_2_TEST, limits))

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    p1 = [r for r in rows if r["phase"] == 1]
    p2 = [r for r in rows if r["phase"] == 2]
    p1_with_bundle = sum(r["has_static_bundle"] for r in p1)
    p2_with_bundle = sum(r["has_static_bundle"] for r in p2)
    p1_with_limits = sum(r["has_question_limits"] for r in p1)
    p2_with_limits = sum(r["has_question_limits"] for r in p2)

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_JSON}")
    print()
    print("=== summary ===")
    print(f"phase 1 scenarios:                  {len(p1)}")
    print(f"  with static bundle:               {p1_with_bundle}")
    print(f"  with question_limits restriction: {p1_with_limits}")
    print(f"phase 2 scenarios:                  {len(p2)}")
    print(f"  with static bundle:               {p2_with_bundle}")
    print(f"  with question_limits restriction: {p2_with_limits}")
    print()
    print("=== unmapped IDs ===")
    p1_unmapped = [r["task_id"] for r in p1 if not r["has_static_bundle"]]
    p2_unmapped_count = sum(1 for r in p2 if not r["has_static_bundle"])
    print(f"phase 1 task_ids without bundle:    {p1_unmapped}")
    print(f"phase 2 task_ids without bundle:    all {p2_unmapped_count} (no local Phase 2 bundles)")


if __name__ == "__main__":
    main()
