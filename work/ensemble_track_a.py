"""Ensemble two or more Track A result.csv files into a single submission.

Strategy (IoU-aware, majority vote):
  For each scenario, tally votes per option across all input runs.
  Options that appear in >= `--threshold` runs are kept (default: ceil(N/2)).

  Tie-break cascade when nothing reaches threshold:
    1. Intersection of all runs that have the scenario (if non-empty)
    2. First (primary) run's answer

Usage — 2-run intersection:
    python work/ensemble_track_a.py \
        --inputs "telco_data/Track A/results_phase2_fewshot_graph/result.csv" \
                 "telco_data/Track A/results_phase2_fewshot/result.csv" \
        --out work/ensemble_fsg_x_fs.csv

Usage — 3-run majority vote (2/3):
    python work/ensemble_track_a.py \
        --inputs "telco_data/Track A/results_phase2_fewshot_graph/result.csv" \
                 "telco_data/Track A/results_phase2_fewshot/result.csv" \
                 "telco_data/Track A/results_phase2_full/result.csv" \
        --out work/ensemble_3way.csv

Writes a CSV with columns: scenario_id,prediction
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if not r or r[0] == "scenario_id":
                continue
            rows[r[0]] = r[1].split("|")
    return rows


def _sort_opts(opts: set[str]) -> list[str]:
    return sorted(opts, key=lambda x: int(x[1:]))


def ensemble(
    runs: list[dict[str, list[str]]],
    threshold: int,
) -> tuple[dict[str, str], dict]:
    all_ids: set[str] = set()
    for r in runs:
        all_ids |= set(r)

    out: dict[str, str] = {}
    stats: dict[str, int] = {
        "unanimous": 0,
        "majority": 0,
        "fallback_intersect": 0,
        "fallback_primary": 0,
        "missing_some": 0,
    }

    n = len(runs)

    for sid in all_ids:
        present = [r[sid] for r in runs if sid in r]
        if len(present) < n:
            stats["missing_some"] += 1

        # Tally votes per option
        counts: Counter = Counter()
        for ans in present:
            for opt in ans:
                counts[opt] += 1

        majority_opts = {opt for opt, cnt in counts.items() if cnt >= threshold}

        if majority_opts:
            if all(counts[opt] == n for opt in majority_opts):
                stats["unanimous"] += 1
            else:
                stats["majority"] += 1
            out[sid] = "|".join(_sort_opts(majority_opts))
        else:
            # Fallback 1: intersection of all present runs
            intersect = set(present[0])
            for ans in present[1:]:
                intersect &= set(ans)
            if intersect:
                stats["fallback_intersect"] += 1
                out[sid] = "|".join(_sort_opts(intersect))
            else:
                # Fallback 2: primary (first run) answer
                stats["fallback_primary"] += 1
                out[sid] = "|".join(present[0])

    return out, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True,
                   help="Result CSV files in priority order (first = primary/fallback)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=int, default=0,
                   help="Min votes to include an option (default: ceil(N/2))")
    args = p.parse_args()

    runs = [load(path) for path in args.inputs]
    n = len(runs)
    threshold = args.threshold if args.threshold > 0 else (n // 2) + 1

    print(f"Runs: {n}  |  Threshold: {threshold}/{n}")
    for i, path in enumerate(args.inputs):
        print(f"  [{i}] {path} ({len(runs[i])} scenarios)")

    result, stats = ensemble(runs, threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "prediction"])
        for sid, pred in sorted(result.items()):
            w.writerow([sid, pred])

    total = len(result)
    print(f"\nEnsembled {total} scenarios → {args.out}")
    print(f"  Unanimous ({n}/{n}):       {stats['unanimous']:>4}  ({stats['unanimous']/total*100:.1f}%)")
    print(f"  Majority  ({threshold}/{n}):       {stats['majority']:>4}  ({stats['majority']/total*100:.1f}%)")
    print(f"  Fallback intersect:    {stats['fallback_intersect']:>4}  ({stats['fallback_intersect']/total*100:.1f}%)")
    print(f"  Fallback primary:      {stats['fallback_primary']:>4}  ({stats['fallback_primary']/total*100:.1f}%)")
    if stats["missing_some"]:
        print(f"  Missing from some run: {stats['missing_some']:>4}")

    dist = Counter(len(v.split("|")) for v in result.values())
    print(f"\nAnswer count distribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k} option(s): {v} scenarios")


if __name__ == "__main__":
    main()
