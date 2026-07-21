"""Offline demo: drive the agent runtime against a programmable Qwen stub
and a stub Agent Tool Server. Useful for sanity-checking the loop on a
real Phase 1 question without standing up Qwen/vLLM yet.

When OPENAI_BASE_URL is set in the environment, the demo calls the real
endpoint instead of the stub policy — same code path. Set
AGENT_TOOL_SERVER_URL to point at the actual simulator (Chinese ELB,
Hong Kong ECS, or local sandbox).

Usage:
    # Offline (default): stub Qwen + stub tool server
    python work/run_agent_demo.py 17

    # Wired up to a real vLLM/sglang server and the simulator:
    OPENAI_BASE_URL=http://localhost:8000/v1 \\
    OPENAI_API_KEY=EMPTY \\
    QWEN_MODEL=Qwen/Qwen3.5-35B-A3B \\
    AGENT_TOOL_SERVER_URL=https://trackB.organizer.example/ip/api/agent/execute \\
    AGENT_TOOL_SERVER_TOKEN=ip-xxx \\
    python work/run_agent_demo.py 17
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.agent_runtime import AgentLimits, run_scenario
from track_b.agent_tools import AgentToolConfig, CommandResult
from track_b.prompt_context import RankedRow
from track_b.qwen_client import (
    ChatResponse,
    QwenClient,
    QwenConfig,
    ToolCall,
)

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json"
RANKED = ROOT / "work" / "ranked_candidates.csv"
LIMITS = ROOT / "telco_data" / "Track B" / "question_limits_config.json"


def _load_question(task_id: int) -> tuple[str, str]:
    """Return (scenario_id, question_text) for a Phase 1 task_id."""
    with open(P1, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if int(item["task"]["id"]) == task_id:
            return item["scenario_id"], item["task"]["question"]
    raise SystemExit(f"task_id {task_id} not found in Phase 1 test.json")


def _load_top_candidates(scenario_id: str, k: int = 5) -> list[RankedRow]:
    import csv
    if not RANKED.is_file():
        return []
    rows = []
    with open(RANKED, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario_id"] != scenario_id:
                continue
            if r.get("offline_bundle_missing") == "1":
                continue
            if not r.get("node"):
                continue
            rows.append(r)
    rows.sort(key=lambda r: -float(r.get("combined_score", 0.0)))
    out: list[RankedRow] = []
    for r in rows[:k]:
        out.append(RankedRow(
            scenario_id=scenario_id,
            node=r["node"],
            fault_reason=r["fault_reason"],
            category=r["category"],
            combined_score=float(r["combined_score"]),
        ))
    return out


def _stub_qwen_policy() -> "callable":
    """Programmable stub: tool-call once, then emit a fault answer."""
    state = {"i": 0}
    def policy(messages, tools):
        state["i"] += 1
        if state["i"] == 1:
            return ChatResponse(
                role="assistant", content="", finish_reason="tool_calls",
                tool_calls=[ToolCall(
                    id="call_demo_1",
                    name="infra_maintenance",
                    arguments={"device_name": "Beta-Aegis-01",
                               "command": "display current-configuration"},
                )],
            )
        return ChatResponse(
            role="assistant",
            content="Beta-Aegis-01;192.168.1.5;blackhole route",
            finish_reason="stop",
        )
    return policy


def _stub_dispatch(*, tool_name, arguments, question_number, config):
    return CommandResult(
        status_code=200,
        status="success",
        device_name=str(arguments.get("device_name")),
        command=str(arguments.get("command")),
        vendor="huawei",
        result_text=(
            "stub output — would contain real CLI output in production "
            "(e.g. `display current-configuration` lines)"
        ),
        raw={},
    )


def main() -> int:
    if len(sys.argv) < 2:
        task_id = 17  # Phase 1 fault question with disclosed candidate list
    else:
        task_id = int(sys.argv[1])

    sid, qtext = _load_question(task_id)
    candidates = _load_top_candidates(sid)
    print(f"scenario_id: {sid}")
    print(f"question (suffix): ...{qtext[-200:]!r}")
    print(f"loaded {len(candidates)} ranker candidates")

    qwen_config = QwenConfig.from_env()
    if qwen_config is None:
        print("OPENAI_BASE_URL unset — running with STUB Qwen policy + STUB tool server")
        qwen = QwenClient(policy=_stub_qwen_policy())
        dispatch_target = _stub_dispatch
    else:
        print(f"using REAL Qwen at {qwen_config.base_url} model={qwen_config.model}")
        qwen = QwenClient(qwen_config)
        dispatch_target = None  # use the real dispatcher

    tool_config = AgentToolConfig.from_env()
    print(f"agent tool server: {tool_config.base_url} (token set: {bool(tool_config.token)})")

    if dispatch_target is not None:
        with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=dispatch_target):
            trace = run_scenario(
                scenario_id=sid,
                question_number=task_id,
                phase=1,
                question_text=qtext,
                candidates=candidates,
                graph_features={},
                anomaly_evidence={},
                qwen=qwen,
                tool_config=tool_config,
                limits_path=str(LIMITS),
                limits=AgentLimits(max_iterations=8, max_tool_calls=20),
            )
    else:
        trace = run_scenario(
            scenario_id=sid,
            question_number=task_id,
            phase=1,
            question_text=qtext,
            candidates=candidates,
            graph_features={},
            anomaly_evidence={},
            qwen=qwen,
            tool_config=tool_config,
            limits_path=str(LIMITS),
            limits=AgentLimits(max_iterations=8, max_tool_calls=20),
        )

    print()
    print(f"final_action       : {trace.final_action}")
    print(f"iterations         : {trace.iterations}")
    print(f"tool_calls_made    : {trace.tool_calls_made}")
    print(f"follow_ups         : {trace.follow_ups_triggered}")
    print(f"format_rejections  : {trace.format_rejections}")
    print(f"constraint_rejects : {trace.constraint_rejections}")
    print(f"final_answer       : {trace.final_answer!r}")
    if trace.errors:
        print(f"errors             :")
        for e in trace.errors:
            print(f"  - {e}")
    return 0 if trace.final_action == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
