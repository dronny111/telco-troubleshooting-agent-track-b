"""Smoke test for the Step 7 oracle silver-label generator.

Asserts:
    - aggregate_consensus picks the majority lines and sets the
      high-confidence flag correctly across edge cases (full agreement,
      partial agreement, no acceptance, single accepted run).
    - run_oracle_for_scenario plumbs (temperature, seed) through the
      stub Qwen policy so conditional audits can produce different answers.
    - High-confidence path emits sample_weight=1.0 when the primary run
      succeeds cleanly; drops to 0.5 when audit runs disagree.
    - labels_for_candidates emits exactly the schema feature_pipeline
      uses (so the trainer needs no path change).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.agent_tools import AgentToolConfig, CommandResult
from track_b.oracle_run import (
    OracleConfig,
    SeedResult,
    aggregate_consensus,
    labels_for_candidates,
    run_oracle_for_scenario,
)
from track_b.prompt_context import RankedRow
from track_b.qwen_client import ChatResponse, QwenClient, ToolCall


REAL_LIMITS = Path(__file__).resolve().parents[2] / "telco_data" / "Track B" / "question_limits_config.json"


# Reuse the question shape from the agent test
SYNTH_QUESTION = (
    "This question involves two categories of faults: routing faults and port faults. "
    "Routing fault output format: fault-node;destination-IP;fault-reason. "
    "Fault reasons include: (1) blackhole route; (2) missing static route. "
    "Port fault output format: fault-node;fault-port;fault-reason. "
    "Fault reasons include: (1) shutdown. "
    "Routing fault examples:\n"
    "Beta-Axis-01;192.168.1.1;blackhole route\n"
    "Port fault examples:\n"
    "Beta-Node-01;GE1/0/2;interface IP error\n"
    "From SRC, accessing 10.0.0.1 failed."
)


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def assert_eq(a, b, label: str) -> bool:
    ok = a == b
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: a={a!r} b={b!r}")
    return ok


def _seed(idx, action, lines):
    return SeedResult(
        seed_index=idx, final_action=action, final_answer="", answer_lines=set(lines),
    )


def main() -> int:
    failures = 0

    section("aggregate_consensus — full agreement → high-conf")
    seeds = [
        _seed(0, "accept", {("A", "blackhole route")}),
        _seed(1, "accept", {("A", "blackhole route")}),
        _seed(2, "accept", {("A", "blackhole route")}),
    ]
    silver, hc = aggregate_consensus(seeds, n_seeds=3)
    failures += not assert_eq(silver, {("A", "blackhole route")}, "silver set")
    failures += not assert_eq(hc, True, "high_confidence")

    section("aggregate_consensus — 2/3 majority → high-conf")
    seeds = [
        _seed(0, "accept", {("A", "blackhole route")}),
        _seed(1, "accept", {("A", "blackhole route")}),
        _seed(2, "accept", {("A", "blackhole route")}),  # all agree on A
    ]
    silver, hc = aggregate_consensus(seeds, n_seeds=3)
    failures += not assert_eq(hc, True, "majority high_conf")

    section("aggregate_consensus — split (1/1/1 disjoint) → low-conf via union")
    seeds = [
        _seed(0, "accept", {("A", "blackhole route")}),
        _seed(1, "accept", {("B", "missing static route")}),
        _seed(2, "accept", {("C", "blackhole route")}),
    ]
    silver, hc = aggregate_consensus(seeds, n_seeds=3)
    failures += not assert_eq(hc, False, "split → not high_conf")
    failures += not assert_eq(silver, {
        ("A", "blackhole route"),
        ("B", "missing static route"),
        ("C", "blackhole route"),
    }, "silver = union when no majority")

    section("aggregate_consensus — partial agreement → silver=union, low-conf")
    seeds = [
        _seed(0, "accept", {("A", "blackhole route"), ("B", "shutdown")}),
        _seed(1, "accept", {("A", "blackhole route")}),
        _seed(2, "accept", {("A", "blackhole route")}),
    ]
    silver, hc = aggregate_consensus(seeds, n_seeds=3)
    failures += not assert_eq(hc, False, "spurious one-off → not high_conf")
    failures += not assert_eq(
        silver, {("A", "blackhole route"), ("B", "shutdown")},
        "silver = union (down-weight rather than discard)",
    )

    section("aggregate_consensus — no accepted seeds → empty + low-conf")
    seeds = [
        _seed(0, "incomplete", set()),
        _seed(1, "budget_exhausted", set()),
    ]
    silver, hc = aggregate_consensus(seeds, n_seeds=2)
    failures += not assert_eq(silver, set(), "no accepted seed → empty silver")
    failures += not assert_eq(hc, False, "no accepted seed → not high_conf")

    section("aggregate_consensus — single accepted run → high-conf")
    seeds = [_seed(0, "accept", {("A", "blackhole route")})]
    silver, hc = aggregate_consensus(seeds, n_seeds=1)
    failures += not assert_eq(silver, {("A", "blackhole route")}, "single run silver")
    failures += not assert_eq(hc, True, "single run high_conf")

    section("run_oracle_for_scenario — primary accepted cleanly, no audits")
    state = {"per_seed_calls": {}}

    def primary_only_policy(messages, tools, temperature=None, seed=None):
        state["per_seed_calls"][seed] = state["per_seed_calls"].get(seed, 0) + 1
        if state["per_seed_calls"][seed] == 1:
            return ChatResponse(
                role="assistant", content="", finish_reason="tool_calls",
                tool_calls=[ToolCall(
                    id=f"call_{seed}",
                    name="infra_maintenance",
                    arguments={"device_name": "Core_SW_01",
                               "command": "display current-configuration"},
                )],
            )
        return ChatResponse(
            role="assistant",
            content="Core_SW_01;192.168.1.5;blackhole route",
            finish_reason="stop",
        )

    qwen = QwenClient(policy=primary_only_policy)

    def fake_dispatch(*, tool_name, arguments, question_number, config):
        return CommandResult(
            status_code=200, status="success",
            device_name=str(arguments.get("device_name")),
            command=str(arguments.get("command")),
            vendor="huawei",
            result_text="stub",
            raw={},
        )

    cfg = OracleConfig(
        audit_temperatures=(0.3, 0.6),
        max_audit_runs=2,
        low_margin_gap=0.05,
    )
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=fake_dispatch):
        result = run_oracle_for_scenario(
            scenario_id="t-oracle",
            question_number=1,
            phase=1,
            question_text=SYNTH_QUESTION,
            candidates=[
                RankedRow(
                    scenario_id="t-oracle",
                    node="Core_SW_01",
                    fault_reason="blackhole route",
                    category="routing",
                    combined_score=5.0,
                ),
                RankedRow(
                    scenario_id="t-oracle",
                    node="Other_Node",
                    fault_reason="missing static route",
                    category="routing",
                    combined_score=4.0,
                ),
            ],
            graph_features={"Core_SW_01": {}},
            anomaly_evidence={},
            qwen=qwen,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            config=cfg,
        )

    failures += not assert_eq(len(result.seeds), 1, "only primary run executed")
    failures += not assert_eq(result.is_high_confidence, True, "primary-only path high_conf")
    failures += not assert_eq(result.sample_weight, 1.0, "primary-only sample_weight")
    failures += not assert_eq(result.notes, "primary accepted without audit", "primary-only note")

    section("run_oracle_for_scenario — low-margin primary triggers audits")
    state = {"per_seed_calls": {}}

    def audit_policy(messages, tools, temperature=None, seed=None):
        state["per_seed_calls"][seed] = state["per_seed_calls"].get(seed, 0) + 1
        if state["per_seed_calls"][seed] == 1:
            return ChatResponse(
                role="assistant", content="", finish_reason="tool_calls",
                tool_calls=[ToolCall(
                    id=f"call_{seed}",
                    name="infra_maintenance",
                    arguments={"device_name": "Core_SW_01",
                               "command": "display current-configuration"},
                )],
            )
        reason = "blackhole route" if (temperature or 0) <= 0.3 else "missing static route"
        return ChatResponse(
            role="assistant",
            content=f"Core_SW_01;192.168.1.5;{reason}",
            finish_reason="stop",
        )

    qwen = QwenClient(policy=audit_policy)
    cfg = OracleConfig(
        audit_temperatures=(0.3, 0.6),
        max_audit_runs=2,
        low_margin_gap=0.2,
    )
    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=fake_dispatch):
        result = run_oracle_for_scenario(
            scenario_id="t-oracle",
            question_number=1,
            phase=1,
            question_text=SYNTH_QUESTION,
            candidates=[
                RankedRow(
                    scenario_id="t-oracle",
                    node="Core_SW_01",
                    fault_reason="blackhole route",
                    category="routing",
                    combined_score=5.0,
                ),
                RankedRow(
                    scenario_id="t-oracle",
                    node="PE1",
                    fault_reason="missing static route",
                    category="routing",
                    combined_score=4.9,
                ),
            ],
            graph_features={"Core_SW_01": {}},
            anomaly_evidence={},
            qwen=qwen,
            tool_config=AgentToolConfig(base_url="http://stub", token=""),
            limits_path=str(REAL_LIMITS),
            config=cfg,
        )

    failures += not assert_eq(len(result.seeds), 3, "primary + 2 audits executed")
    failures += not assert_eq(
        ("Core_SW_01", "blackhole route") in result.silver_positives,
        True,
        "majority line in silver positives",
    )
    failures += not assert_eq(
        result.is_high_confidence,
        False,
        "spurious one-off line → low_conf",
    )
    failures += not assert_eq(
        result.sample_weight,
        0.5,
        "low_conf sample_weight",
    )

    section("labels_for_candidates — emits trainer-compatible schema")
    candidate_rows = [
        {"scenario_id": "t-oracle", "question_number": 1, "phase": 1,
         "node": "Core_SW_01", "fault_reason": "blackhole route", "category": "routing"},
        {"scenario_id": "t-oracle", "question_number": 1, "phase": 1,
         "node": "Other_Node", "fault_reason": "shutdown", "category": "port"},
    ]
    label_rows = labels_for_candidates(
        candidate_rows=candidate_rows,
        oracle_results={"t-oracle": result},
    )
    by_node = {r["node"]: r for r in label_rows}
    failures += not assert_eq(by_node["Core_SW_01"]["relevance"], 1, "positive labelled")
    failures += not assert_eq(by_node["Other_Node"]["relevance"], 0, "negative labelled")
    expected_cols = {
        "scenario_id", "question_number", "phase",
        "node", "fault_reason", "category",
        "relevance", "sample_weight",
    }
    failures += not assert_eq(
        set(label_rows[0].keys()), expected_cols,
        "schema parity with feature_pipeline.synthetic_relevance",
    )

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
