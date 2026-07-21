"""Build the compact prompt context per scenario and persist for review.

Reads `work/ranked_candidates.csv` + `work/anomaly_candidates.csv` and the
question text, runs the constraint parser, and emits one line per scenario
to `work/prompt_contexts.jsonl` with the context dict and an estimated
token count for the assembled system prompt. Used for eyeballing prompt
budget and verifying signal quality before wiring up Qwen.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.constraint_parser import parse as parse_constraints
from track_b.permission_pruner import denied_pairs
from track_b.prompt_context import (
    AnomalyEvidence,
    RankedRow,
    approx_token_count,
    build_context,
    build_system_prompt,
)
from track_b.task_classifier import classify

ROOT = Path(__file__).resolve().parents[1]
RANKED = ROOT / "work" / "ranked_candidates.csv"
ANOMALY = ROOT / "work" / "anomaly_candidates.csv"
LIMITS = ROOT / "telco_data" / "Track B" / "question_limits_config.json"
P1 = ROOT / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json"
P2 = ROOT / "telco_data" / "Track B" / "data" / "Phase_2" / "test.json"
OUT = ROOT / "work" / "prompt_contexts.jsonl"


def _load_questions() -> dict[str, tuple[int, int, str]]:
    """scenario_id -> (phase, task_id, question_text)."""
    out: dict[str, tuple[int, int, str]] = {}
    for phase, path in ((1, P1), (2, P2)):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            out[item["scenario_id"]] = (
                phase,
                int(item["task"]["id"]),
                item["task"]["question"],
            )
    return out


def _load_ranked() -> dict[str, list[RankedRow]]:
    out: dict[str, list[RankedRow]] = defaultdict(list)
    with open(RANKED, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("offline_bundle_missing") == "1":
                continue
            if not r["node"]:
                continue
            out[r["scenario_id"]].append(
                RankedRow(
                    scenario_id=r["scenario_id"],
                    node=r["node"],
                    fault_reason=r["fault_reason"],
                    category=r["category"],
                    combined_score=float(r["combined_score"]),
                    calibrated_score=None,
                    uncertainty=None,
                )
            )
    return out


def _load_anomaly_evidence() -> dict[tuple[str, str, str], AnomalyEvidence]:
    out: dict[tuple[str, str, str], AnomalyEvidence] = {}
    with open(ANOMALY, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("offline_bundle_missing") == "1":
                continue
            key = (r["scenario_id"], r["node"], r["fault_reason"])
            out[key] = AnomalyEvidence(
                scenario_id=r["scenario_id"],
                node=r["node"],
                fault_reason=r["fault_reason"],
                sample_evidence=r["sample_evidence"],
                signatures_fired=r["signatures_fired"],
            )
    return out


def main() -> int:
    questions = _load_questions()
    ranked = _load_ranked()
    anomaly_ev = _load_anomaly_evidence()

    rows = []
    sizes = []
    skipped_nonfault = 0
    skipped_no_candidates = 0
    for scenario_id, (phase, task_id, qtext) in questions.items():
        family = classify(qtext)
        if family != "fault":
            skipped_nonfault += 1
            continue
        cands = ranked.get(scenario_id)
        if not cands:
            skipped_no_candidates += 1
            continue
        parsed = parse_constraints(qtext)
        denied = denied_pairs(task_id, LIMITS)
        ctx = build_context(
            task_family=family,
            parsed=parsed,
            candidates=cands,
            anomaly_evidence=anomaly_ev,
            denied_pairs=denied,
        )
        prompt = build_system_prompt(
            context=ctx,
            few_shot_exemplars=(),
            include_phase_2_device_list=(phase == 2),
        )
        n_tok = approx_token_count(prompt)
        rows.append(
            {
                "scenario_id": scenario_id,
                "phase": phase,
                "task_id": task_id,
                "context": ctx,
                "approx_token_count": n_tok,
            }
        )
        sizes.append(n_tok)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {OUT} ({len(rows)} contexts)")
    print(f"  non-fault scenarios skipped:    {skipped_nonfault}")
    print(f"  scenarios w/o candidates:       {skipped_no_candidates}")
    if sizes:
        sizes.sort()
        print(f"  approx system-prompt tokens:    "
              f"min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")
        # bucketed
        under_1k = sum(1 for s in sizes if s < 1000)
        under_2k = sum(1 for s in sizes if s < 2000)
        print(f"  under 1k tokens:               {under_1k}/{len(sizes)}")
        print(f"  under 2k tokens:               {under_2k}/{len(sizes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
