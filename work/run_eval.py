"""Run the V0–V4 ablation suite end-to-end.

Reads the pipeline artifacts from Steps 4–8, evaluates each variant
against the synthetic silver labels, runs the per-signal ablations on V2,
and dumps results to `work/eval_report.json` plus a console table.

When the real Qwen agent runtime is available, the only change is to
swap `_stub_predict` in eval_harness.py for the agent's per-scenario
`run(scenario)` call.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.eval_harness import (
    ABLATABLE_COMPONENTS,
    SWEEP_FAMILIES,
    VARIANTS,
    _primary_score_col,
    ablate_one_signal,
    auto_tau_grid,
    best_row,
    load_scenarios,
    run_eval,
    run_threshold_sweep_one,
    variance_across_reruns,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "manifest_path": ROOT / "work" / "scenario_manifest.csv",
    "ranked_path": ROOT / "work" / "ranked_candidates.csv",
    "ranked_xgb_path": ROOT / "work" / "ranked_candidates_xgb.csv",
    "anomaly_path": ROOT / "work" / "anomaly_candidates.csv",
    "graph_path": ROOT / "work" / "graph_features.csv",
    "silver_labels_path": ROOT / "work" / "xgb_silver_labels.csv",
    "questions_phase1": ROOT / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json",
    "questions_phase2": ROOT / "telco_data" / "Track B" / "data" / "Phase_2" / "test.json",
    "limits_path": ROOT / "telco_data" / "Track B" / "question_limits_config.json",
}
OUT = ROOT / "work" / "eval_report.json"


def _print_table(reports: list[dict], title: str) -> None:
    print(f"\n{title}")
    headers = [
        "variant", "n_scen", "exact", "top1", "P", "R", "F1",
        "calls/scen", "calls/correct", "fmt_err", "f_up_rate",
    ]
    keys = [
        "variant", "n_scenarios",
        "accuracy_exact", "accuracy_top1",
        "precision", "recall", "f1",
        "mean_calls", "calls_per_correct",
        "format_error_rate", "follow_up_rate",
    ]
    widths = [max(len(h), 8) for h in headers]
    print("  " + "  ".join(f"{h:>{w}}" for h, w in zip(headers, widths)))
    for r in reports:
        row = []
        for k, w in zip(keys, widths):
            v = r[k]
            if isinstance(v, float):
                row.append(f"{v:>{w}.4f}")
            else:
                row.append(f"{str(v):>{w}}")
        print("  " + "  ".join(row))


def _print_sweep_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    headers = [
        "variant", "family", "score", "tau", "k", "n",
        "emit", "abstain", "exact", "P", "R", "F1", "fmt_err",
    ]
    keys = [
        "variant", "family", "score_col", "tau", "k_max", "n_scenarios",
        "emit_rate", "abstain_rate", "accuracy_exact",
        "precision", "recall", "f1", "format_error_rate",
    ]
    widths = [max(len(h), 8) for h in headers]
    print("  " + "  ".join(f"{h:>{w}}" for h, w in zip(headers, widths)))
    for r in rows:
        row = []
        for k, w in zip(keys, widths):
            v = r.get(k, "")
            if isinstance(v, float):
                row.append(f"{v:>{w}.4f}")
            else:
                row.append(f"{str(v):>{w}}")
        print("  " + "  ".join(row))


def main() -> int:
    print("==> loading scenarios from on-disk artifacts")
    scenarios = load_scenarios(**ARTIFACTS)
    print(f"  scenarios with full pipeline output: {len(scenarios)}")
    if not scenarios:
        print("  no scenarios available; run earlier steps first")
        return 1
    n_with_silver = sum(1 for s in scenarios if s.silver_positives)
    print(f"  scenarios with silver positives:     {n_with_silver}")

    print("\n==> running V0..V4")
    reports = run_eval(VARIANTS.values(), scenarios)
    main_table = [r.as_dict() for r in reports]
    _print_table(main_table, "main variant results")

    print("\n==> per-signal ablation on V2 (drop one component, recompute combined_score)")
    ablation_rows: list[dict] = []
    for component in ABLATABLE_COMPONENTS:
        rep = ablate_one_signal(
            base_variant=VARIANTS["V2"], scenarios=scenarios, component=component,
        )
        ablation_rows.append(rep.as_dict())
    _print_table(ablation_rows, "V2 ablations")

    print("\n==> per-family threshold sweep (skip-emit at tau)")
    # Sweep variants that actually produce confidence scores; V0/V1 have
    # no usable score column. K_MAX_VALUES > 1 lets multi-line answers be
    # emitted for high-confidence multi-fault scenarios. When
    # USE_AGENT_RUNTIME is True the sweep routes emission through
    # agent_runtime.run_scenario (Qwen stub policy, full validator loop)
    # instead of the shallow _stub_predict path.
    #
    # Env-var overrides (smoke-test the real Qwen path without editing
    # this file):
    #   EVAL_SWEEP_VARIANTS=V4           (comma-sep)
    #   EVAL_SWEEP_KMAX=1                (comma-sep ints)
    #   EVAL_SWEEP_TAU_POINTS=3
    #   EVAL_USE_AGENT_RUNTIME=0|1
    def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        return tuple(x.strip() for x in raw.split(",") if x.strip())

    SWEEP_VARIANTS = _csv_env("EVAL_SWEEP_VARIANTS", ("V2", "V3", "V4"))
    K_MAX_VALUES = tuple(int(x) for x in _csv_env("EVAL_SWEEP_KMAX", ("1", "2", "3")))
    TAU_POINTS = int(os.environ.get("EVAL_SWEEP_TAU_POINTS", "9"))
    USE_AGENT_RUNTIME = os.environ.get("EVAL_USE_AGENT_RUNTIME", "1") not in ("0", "false", "False")
    print(f"  variants={SWEEP_VARIANTS}  k_max={K_MAX_VALUES}  tau_points={TAU_POINTS}  "
          f"agent_runtime={USE_AGENT_RUNTIME}  qwen_base_url={os.environ.get('OPENAI_BASE_URL') or 'STUB'}")
    sweep_rows: list[dict] = []
    best_by_pair: dict[tuple[str, str], dict] = {}
    for vname in SWEEP_VARIANTS:
        v = VARIANTS[vname]
        score_col = _primary_score_col(v)
        taus = auto_tau_grid(scenarios, score_col=score_col, n_points=TAU_POINTS)
        print(f"  {vname:<3} score_col={score_col}  tau_grid={taus}")
        for family in SWEEP_FAMILIES:
            for k_max in K_MAX_VALUES:
                rows = run_threshold_sweep_one(
                    variant=v, family=family, scenarios=scenarios,
                    taus=taus, k_max=k_max,
                    use_agent_runtime=USE_AGENT_RUNTIME,
                )
                for r in rows:
                    sweep_rows.append(r.as_dict())
                top = best_row(rows, objective="accuracy_exact")
                if top is not None and top.n_scenarios > 0:
                    best_by_pair[(vname, family)] = {
                        **top.as_dict(),
                        "objective": "accuracy_exact",
                    }
    _print_sweep_table(sweep_rows, "threshold sweep (per variant × family × tau × k_max)")
    print("\n  best (variant, family) → tau:")
    for (vname, family), row in best_by_pair.items():
        print(f"    {vname}/{family:<8} tau={row['tau']:.4f} k_max={row['k_max']} "
              f"exact={row['accuracy_exact']:.4f} abstain={row['abstain_rate']:.4f} "
              f"F1={row['f1']:.4f}")

    print("\n==> variance across 3 reruns (Phase-3 Pass@1 proxy)")
    variance_rows = []
    for v in VARIANTS.values():
        var = variance_across_reruns(v, scenarios, n_reruns=3)
        variance_rows.append(var)
        print(f"  {var['variant']:<5}  mean={var['mean_accuracy']:.4f}  "
              f"std={var['std_accuracy']:.4f}  per_run={var['per_rerun']}")

    out = {
        "main": main_table,
        "ablations_v2": ablation_rows,
        "threshold_sweep": sweep_rows,
        "threshold_sweep_best": {f"{k[0]}/{k[1]}": v for k, v in best_by_pair.items()},
        "variance": variance_rows,
        "n_scenarios": len(scenarios),
        "n_with_silver_positives": n_with_silver,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
