"""Drive the Step 7 oracle silver-label generation.

Loops `agent_runtime.run_scenario` over every Phase 1 fault scenario with
a local bundle, with N seeds, and writes:

    work/oracle_silver_labels.csv  — drop-in replacement for
                                     work/xgb_silver_labels.csv (synthetic).
                                     Has the exact columns Step 8's
                                     `train_kfold` reads.

    work/oracle_run_traces.jsonl   — per-scenario per-seed trace dump
                                     for offline inspection and error
                                     analysis.

Backend selection:
    - With OPENAI_BASE_URL set in env, calls the real Qwen endpoint.
    - Without it, falls back to a deterministic stub policy that
      varies its answer slightly per seed so the consensus path can be
      exercised end-to-end before Qwen is up.

Once the silver-label CSV is in place, refit Step 8 by running:

    LABELS_PATH=work/oracle_silver_labels.csv python3 work/run_xgb_train.py

(The trainer's label loader treats either path identically.)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.agent_tools import AgentToolConfig
from track_b.oracle_run import (
    OracleConfig,
    ScenarioOracleResult,
    labels_for_candidates,
    run_oracle_for_scenario,
    write_oracle_traces,
    write_silver_labels_csv,
)
from track_b.prompt_context import AnomalyEvidence, RankedRow
from track_b.qwen_client import (
    ChatResponse,
    QwenClient,
    QwenConfig,
    ToolCall,
)
from track_b.task_classifier import classify
from unittest.mock import patch
from track_b.agent_tools import CommandResult


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "work" / "scenario_manifest.csv"
RANKED = ROOT / "work" / "ranked_candidates.csv"
ANOMALY = ROOT / "work" / "anomaly_candidates.csv"
GRAPH = ROOT / "work" / "graph_features.csv"
P1 = ROOT / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json"
LIMITS = ROOT / "telco_data" / "Track B" / "question_limits_config.json"
OUT_LABELS = ROOT / "work" / "oracle_silver_labels.csv"
OUT_TRACES = ROOT / "work" / "oracle_run_traces.jsonl"


def _load_phase1_questions() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(P1, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        out[item["scenario_id"]] = {
            "task_id": int(item["task"]["id"]),
            "text": item["task"]["question"],
        }
    return out


def _load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_top_candidates(scenario_id: str, ranked_rows: list[dict], k: int = 10) -> list[RankedRow]:
    rows = [r for r in ranked_rows
            if r["scenario_id"] == scenario_id
            and r.get("offline_bundle_missing") == "0"
            and r.get("node")]
    rows.sort(key=lambda r: -float(r.get("combined_score", 0.0)))
    out: list[RankedRow] = []
    seen: set[tuple[str, str]] = set()
    for r in rows[:k * 2]:
        key = (r["node"], r["fault_reason"])
        if key in seen:
            continue
        seen.add(key)
        out.append(RankedRow(
            scenario_id=scenario_id,
            node=r["node"],
            fault_reason=r["fault_reason"],
            category=r["category"],
            combined_score=float(r["combined_score"]),
        ))
        if len(out) >= k:
            break
    return out


def _stub_qwen() -> QwenClient:
    """Programmable stub Qwen — emits slightly different answers per seed
    so the consensus path is exercised. Calls a tool once, then answers
    based on temperature: low temp gives the highest-score candidate;
    higher temps shuffle slightly."""
    state: dict = {"i": 0}

    def policy(messages, tools, temperature=None, seed=None):
        state["i"] += 1
        # First call always requests a config dump so the tool path is hit
        if state["i"] == 1:
            return ChatResponse(
                role="assistant", content="", finish_reason="tool_calls",
                tool_calls=[ToolCall(
                    id=f"call_{state['i']}",
                    name="infra_maintenance",
                    arguments={
                        "device_name": "Beta-Aegis-01",
                        "command": "display current-configuration",
                    },
                )],
            )
        # On the answer turn, emit a fault line keyed by temperature so the
        # 3 seeds produce slightly different answers (testing the
        # consensus path).
        suffix = "blackhole route"
        if temperature and temperature > 0.5:
            suffix = "missing static route"
        return ChatResponse(
            role="assistant",
            content=f"Beta-Aegis-01;192.168.1.5;{suffix}",
            finish_reason="stop",
        )

    return QwenClient(policy=policy)


def _stub_dispatch(*, tool_name, arguments, question_number, config):
    return CommandResult(
        status_code=200, status="success",
        device_name=str(arguments.get("device_name")),
        command=str(arguments.get("command")),
        vendor="huawei",
        result_text="stub config output",
        raw={},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="If set, run only the first N target scenarios (smoke test). "
             "Outputs are written with a .smoke suffix so full-run artifacts aren't clobbered.",
    )
    args = parser.parse_args()
    out_labels = OUT_LABELS
    out_traces = OUT_TRACES
    if args.limit is not None:
        out_labels = OUT_LABELS.with_suffix(".smoke.csv")
        out_traces = OUT_TRACES.with_suffix(".smoke.jsonl")

    if not MANIFEST.is_file() or not RANKED.is_file():
        print("missing prerequisite artifacts; run earlier pipeline steps first")
        return 1

    questions = _load_phase1_questions()
    manifest_rows = _load_csv(MANIFEST)
    ranked_rows = _load_csv(RANKED)
    anomaly_rows = _load_csv(ANOMALY)
    graph_rows = _load_csv(GRAPH)

    # Build per-scenario evidence dicts
    anomaly_by_sid: dict[str, dict[tuple[str, str, str], AnomalyEvidence]] = defaultdict(dict)
    for r in anomaly_rows:
        if r.get("offline_bundle_missing") == "1" or not r.get("node"):
            continue
        key = (r["scenario_id"], r["node"], r["fault_reason"])
        anomaly_by_sid[r["scenario_id"]][key] = AnomalyEvidence(
            scenario_id=r["scenario_id"],
            node=r["node"],
            fault_reason=r["fault_reason"],
            sample_evidence=r.get("sample_evidence", ""),
            signatures_fired=r.get("signatures_fired", ""),
        )

    graph_by_sid: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in graph_rows:
        if r.get("offline_bundle_missing") == "1" or not r.get("node"):
            continue
        graph_by_sid[r["scenario_id"]][r["node"]] = r

    # Pick QwenClient backend
    qwen_config = QwenConfig.from_env()
    using_stub = qwen_config is None
    if using_stub:
        print("OPENAI_BASE_URL unset — running with STUB Qwen (offline dry-run)")
        qwen = _stub_qwen()
    else:
        print(f"using REAL Qwen at {qwen_config.base_url} model={qwen_config.model}")
        qwen = QwenClient(qwen_config)

    tool_config = AgentToolConfig.from_env()
    if using_stub:
        print("AGENT_TOOL_SERVER stubbed — no real API calls will be made")
    else:
        print(f"agent tool server: {tool_config.base_url} (token set: {bool(tool_config.token)})")

    # Filter to Phase 1 fault scenarios with bundles
    targets: list[tuple[str, int]] = []
    for r in manifest_rows:
        if int(r["phase"]) != 1:
            continue
        if r["has_static_bundle"].lower() != "true":
            continue
        sid = r["scenario_id"]
        q = questions.get(sid)
        if not q:
            continue
        if classify(q["text"]) != "fault":
            continue
        targets.append((sid, int(r["question_number"])))

    if args.limit is not None:
        targets = targets[: args.limit]
        print(f"--limit {args.limit} applied; outputs will go to {out_labels.name} / {out_traces.name}")

    print(f"==> oracle run on {len(targets)} Phase 1 fault scenarios")

    config = OracleConfig()
    results: list[ScenarioOracleResult] = []
    t0 = time.perf_counter()
    for sid, qn in targets:
        q = questions[sid]
        cands = _load_top_candidates(sid, ranked_rows)
        anomaly = anomaly_by_sid.get(sid, {})
        gf = graph_by_sid.get(sid, {})
        if using_stub:
            with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_stub_dispatch):
                r = run_oracle_for_scenario(
                    scenario_id=sid,
                    question_number=qn,
                    phase=1,
                    question_text=q["text"],
                    candidates=cands,
                    graph_features=gf,
                    anomaly_evidence=anomaly,
                    qwen=qwen,
                    tool_config=tool_config,
                    limits_path=str(LIMITS),
                    config=config,
                )
        else:
            r = run_oracle_for_scenario(
                scenario_id=sid,
                question_number=qn,
                phase=1,
                question_text=q["text"],
                candidates=cands,
                graph_features=gf,
                anomaly_evidence=anomaly,
                qwen=qwen,
                tool_config=tool_config,
                limits_path=str(LIMITS),
                config=config,
            )
        results.append(r)
        n_pos = len(r.silver_positives)
        flag = "HIGH" if r.is_high_confidence else "low"
        print(f"  q={qn:>3}  sid={sid[:8]}  positives={n_pos:>2}  conf={flag}  notes={r.notes!r}")
    dt = time.perf_counter() - t0

    # Filter ranked_rows to Phase 1 with bundle for label generation
    phase1_ranked = [
        r for r in ranked_rows
        if r.get("offline_bundle_missing") == "0"
        and r.get("node")
        and int(r.get("phase", 0)) == 1
    ]
    oracle_by_sid = {r.scenario_id: r for r in results}
    label_rows = labels_for_candidates(
        candidate_rows=phase1_ranked, oracle_results=oracle_by_sid,
    )
    write_silver_labels_csv(label_rows, out_labels)
    write_oracle_traces(results, out_traces)

    n_high = sum(1 for r in results if r.is_high_confidence)
    n_usable = sum(1 for r in results if r.usable)
    n_pos_total = sum(int(row["relevance"]) for row in label_rows)
    n_rows = len(label_rows)
    print()
    print(f"==> wrote {out_labels} ({n_rows} rows)")
    print(f"    wrote {out_traces} ({len(results)} scenarios)")
    print(f"    elapsed:                 {dt:.1f}s")
    print(f"    scenarios run:           {len(results)}")
    print(f"    usable (any positives):  {n_usable}")
    print(f"    high-confidence:         {n_high}")
    print(f"    label rows total:        {n_rows}")
    print(f"    label rows positive:     {n_pos_total}")
    if using_stub:
        print()
        print("DRY RUN: re-run with OPENAI_BASE_URL set to write real silver labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
