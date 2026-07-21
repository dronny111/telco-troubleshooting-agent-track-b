"""One-off probe: run a single Phase 1 fault scenario through run_scenario
and dump the full transcript so we can see why the agent isn't converging.
Not a permanent driver — safe to delete after debugging."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.agent_runtime import AgentLimits, run_scenario
from track_b.agent_tools import AgentToolConfig
from track_b.prompt_context import RankedRow
from track_b.qwen_client import QwenClient, QwenConfig


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json"
LIMITS = ROOT / "telco_data" / "Track B" / "question_limits_config.json"
RANKED = ROOT / "work" / "ranked_candidates.csv"


def main() -> int:
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    max_it = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_tc = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    with open(P1) as f:
        data = json.load(f)
    item = next(x for x in data if int(x["task"]["id"]) == task_id)
    sid = item["scenario_id"]
    qtext = item["task"]["question"]

    cands_raw = []
    with open(RANKED) as f:
        for r in csv.DictReader(f):
            if r["scenario_id"] != sid:
                continue
            if r.get("offline_bundle_missing") == "1":
                continue
            if not r.get("node"):
                continue
            cands_raw.append((float(r.get("combined_score") or 0.0), r))
    cands_raw.sort(key=lambda x: -x[0])
    ranked = [
        RankedRow(
            scenario_id=sid,
            node=r["node"],
            fault_reason=r["fault_reason"],
            category=r["category"],
            combined_score=s,
        )
        for s, r in cands_raw[:5]
    ]
    print(f"sid={sid}")
    print(f"task_id={task_id}  limits: max_iterations={max_it}  max_tool_calls={max_tc}")
    print(f"question tail: ...{qtext[-200:]!r}")
    print(f"top candidates: {[(c.node, c.fault_reason) for c in ranked]}")
    print()

    qcfg = QwenConfig.from_env()
    assert qcfg is not None, "OPENAI_BASE_URL not set"
    q = QwenClient(qcfg)
    tcfg = AgentToolConfig.from_env()

    # Patch dispatch_tool_call to also stash the full result text on each entry
    # so we can see whether the tool server is returning real data or errors.
    import track_b.agent_runtime as ar
    orig_dispatch = ar.dispatch_tool_call
    captured: list[dict] = []

    def wrap(*, tool_name, arguments, question_number, config):
        try:
            res = orig_dispatch(
                tool_name=tool_name,
                arguments=arguments,
                question_number=question_number,
                config=config,
            )
            captured.append({
                "tool": tool_name,
                "args": dict(arguments),
                "status_code": res.status_code,
                "status": res.status,
                "result_text_head": (res.result_text or "")[:300],
            })
            return res
        except Exception as e:
            captured.append({
                "tool": tool_name,
                "args": dict(arguments),
                "exception": f"{type(e).__name__}: {e}",
            })
            raise

    ar.dispatch_tool_call = wrap
    try:
        trace = run_scenario(
            scenario_id=sid,
            question_number=task_id,
            phase=1,
            question_text=qtext,
            candidates=ranked,
            graph_features={},
            anomaly_evidence={},
            qwen=q,
            tool_config=tcfg,
            limits_path=str(LIMITS),
            limits=AgentLimits(max_iterations=max_it, max_tool_calls=max_tc),
            temperature=0.0,
            seed=42,
        )
    finally:
        ar.dispatch_tool_call = orig_dispatch

    print(f"final_action       : {trace.final_action}")
    print(f"iterations         : {trace.iterations}")
    print(f"tool_calls_made    : {trace.tool_calls_made}")
    print(f"follow_ups         : {trace.follow_ups_triggered}")
    print(f"format_rejections  : {trace.format_rejections}")
    print(f"constraint_rejects : {trace.constraint_rejections}")
    print(f"final_answer       : {trace.final_answer!r}")
    if trace.errors:
        print("errors:")
        for e in trace.errors:
            print(f"  - {e}")
    print()

    print("=== captured tool dispatch ===")
    for i, c in enumerate(captured):
        print(f"[{i}] {c.get('tool')}  args={c.get('args')}")
        if "exception" in c:
            print(f"    EXC: {c['exception']}")
        else:
            print(f"    status_code={c.get('status_code')}  status={c.get('status')}")
            print(f"    result_text_head: {c.get('result_text_head')!r}")
    print()

    print("=== transcript ===")
    for i, m in enumerate(trace.transcript):
        role = m.get("role", "?")
        print(f"[{i}] role={role}")
        if "content_len" in m:
            print(f"    content_len={m['content_len']}")
        if "tool_calls" in m:
            for tc in m["tool_calls"]:
                print(f"    tool_call: {tc.get('name')}({tc.get('args')})")
        if "name" in m and role == "tool":
            print(f"    tool_name(result): {m.get('name')}  result_len={m.get('content_len','?')}")
        if "draft_answer" in m:
            d = m["draft_answer"] or ""
            trunc = "... TRUNC" if len(d) > 800 else ""
            print(f"    draft_answer: {d[:800]}{trunc}")
        if "validator_action" in m:
            print(f"    validator_action: {m['validator_action']}")

    # Dump the captured dispatch results to file for inspection
    out = ROOT / "work" / "_probe_dispatch_dump.json"
    with open(out, "w") as f:
        json.dump({"task_id": task_id, "sid": sid, "captured": captured}, f, indent=2)
    print()
    print(f"wrote dispatch dump -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
