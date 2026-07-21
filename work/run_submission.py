"""Generate a Track B submission CSV and execution traces.

Runs the existing `track_b.agent_runtime.run_scenario()` loop over a batch
JSON file (Phase 2 by default) and writes:

    work/submission/result.csv
    work/submission/eval_detail.jsonl

The CSV matches the organiser's legacy evaluator contract: columns
`id,prediction`, where `id` is `task.id`.

Config resolution order:
    1. Existing process env (`OPENAI_*`, `QWEN_MODEL`, `AGENT_TOOL_SERVER_*`)
    2. `.env` in the repository root:
         AGENT_MODEL_URL   -> OPENAI_BASE_URL
         AGENT_API_KEY     -> OPENAI_API_KEY
         AGENT_MODEL_NAME  -> QWEN_MODEL
         ZINDI_BEARER_TOKEN_B1/B2 -> fallback Track B tool tokens

If no explicit Track B tool URL is provided, the script defaults to the
official Hong Kong Phase 2 endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.agent_runtime import (
    AgentLimits,
    AgentTrace,
    _build_path_system_prompt,
    _build_topology_system_prompt,
    _path_requires_interfaces,
    run_scenario,
)
from track_b.agent_tools import AgentToolConfig, CommandResult
from track_b.constraint_parser import parse as parse_constraints
from track_b.format_guard import validate_fault, validate_path, validate_topology
from track_b.permission_pruner import denied_pairs as load_denied_pairs
from track_b.prompt_context import AnomalyEvidence, RankedRow, build_context, build_system_prompt
from track_b.qwen_client import ChatResponse, QwenClient, QwenConfig, ToolCall
from track_b.task_classifier import classify
from track_b.vocab_extractor import extract_fault_vocab


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "telco_data" / "Track B" / "data" / "Phase_2" / "test.json"
DEFAULT_OUT_DIR = ROOT / "work" / "submission"
LIMITS = ROOT / "telco_data" / "Track B" / "question_limits_config.json"
RANKED = ROOT / "work" / "ranked_candidates.csv"
RANKED_XGB = ROOT / "work" / "ranked_candidates_xgb.csv"
ANOMALY = ROOT / "work" / "anomaly_candidates.csv"
GRAPH = ROOT / "work" / "graph_features.csv"
DOTENV = ROOT / ".env"

DEFAULT_TRACK_B_TOOL_URL = "https://trackA.organizer.example/ip/api/agent/execute"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def _parse_args() -> argparse.Namespace:
    """Parse CLI args, with env-var fallbacks for every flag.

    Env-var precedence is `CLI > env > built-in default`. This mirrors
    the SWEEP_* gating in run_eval.py so a single block of `export ...`
    drives both scripts uniformly:

        SUBMISSION_INPUT_JSON, SUBMISSION_OUT_DIR,
        SUBMISSION_LIMIT, SUBMISSION_IDS, SUBMISSION_RESUME,
        SUBMISSION_STUB, SUBMISSION_TOP_K,
        SUBMISSION_MAX_ITERATIONS, SUBMISSION_MAX_TOOL_CALLS,
        SUBMISSION_WORKERS
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", type=Path,
                   default=Path(os.environ.get("SUBMISSION_INPUT_JSON", str(DEFAULT_INPUT))))
    p.add_argument("--out-dir", type=Path,
                   default=Path(os.environ.get("SUBMISSION_OUT_DIR", str(DEFAULT_OUT_DIR))))
    p.add_argument("--limit", type=int,
                   default=int(os.environ.get("SUBMISSION_LIMIT", "0")),
                   help="Run only the first N questions (0 = all)")
    p.add_argument("--ids", type=str,
                   default=os.environ.get("SUBMISSION_IDS", ""),
                   help="Comma-separated task ids/ranges to run, e.g. 1,67-100")
    p.add_argument("--resume", action="store_true",
                   default=_env_bool("SUBMISSION_RESUME"),
                   help="Skip ids already present in result.csv")
    p.add_argument("--stub", action="store_true",
                   default=_env_bool("SUBMISSION_STUB"),
                   help="Use the stubbed Qwen/tool path for offline dry-runs")
    p.add_argument("--top-k", type=int,
                   default=int(os.environ.get("SUBMISSION_TOP_K", "10")),
                   help="Top ranked candidates to pass into the prompt context")
    p.add_argument("--max-iterations", type=int,
                   default=int(os.environ.get("SUBMISSION_MAX_ITERATIONS", "10")))
    p.add_argument("--max-tool-calls", type=int,
                   default=int(os.environ.get("SUBMISSION_MAX_TOOL_CALLS", "80")))
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SUBMISSION_WORKERS", "1")),
                   help="Parallel worker threads (default 1 = sequential)")
    return p.parse_args()


def _parse_id_selector(raw: str) -> set[int]:
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _resolved_env(dotenv: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    if not env.get("OPENAI_BASE_URL") and dotenv.get("AGENT_MODEL_URL"):
        env["OPENAI_BASE_URL"] = dotenv["AGENT_MODEL_URL"]
    if not env.get("OPENAI_API_KEY") and dotenv.get("AGENT_API_KEY"):
        env["OPENAI_API_KEY"] = dotenv["AGENT_API_KEY"]
    if not env.get("QWEN_MODEL") and dotenv.get("AGENT_MODEL_NAME"):
        env["QWEN_MODEL"] = dotenv["AGENT_MODEL_NAME"]
    if not env.get("AGENT_TOOL_SERVER_URL"):
        env["AGENT_TOOL_SERVER_URL"] = DEFAULT_TRACK_B_TOOL_URL
    if "AGENT_TOOL_SERVER_VERIFY_TLS" not in env and env["AGENT_TOOL_SERVER_URL"].startswith("https://"):
        env["AGENT_TOOL_SERVER_VERIFY_TLS"] = "0"
    return env


def _qwen_from_env(env: dict[str, str]) -> QwenClient:
    config = QwenConfig(
        base_url=env["OPENAI_BASE_URL"].rstrip("/"),
        api_key=env.get("OPENAI_API_KEY", "EMPTY"),
        model=env.get("QWEN_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        temperature=float(env.get("QWEN_TEMPERATURE", "0.0")),
        max_tokens=int(env.get("QWEN_MAX_TOKENS", "8192")),
        timeout_s=float(env.get("QWEN_TIMEOUT_S", "120")),
        retries=int(env.get("QWEN_RETRIES", "1")),
    )
    return QwenClient(config)


def _available_tool_tokens(dotenv: dict[str, str], env: dict[str, str]) -> list[tuple[str, str]]:
    if env.get("AGENT_TOOL_SERVER_TOKEN"):
        return [("env", env["AGENT_TOOL_SERVER_TOKEN"])]
    out: list[tuple[str, str]] = []
    for key in ("ZINDI_BEARER_TOKEN_B1", "ZINDI_BEARER_TOKEN_B2"):
        if dotenv.get(key):
            out.append((key, dotenv[key]))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for label, token in out:
        if token in seen:
            continue
        seen.add(token)
        deduped.append((label, token))
    return deduped


def _tool_config_for_index(env: dict[str, str], tokens: list[tuple[str, str]], index: int) -> tuple[str, AgentToolConfig]:
    label, token = ("none", "")
    if tokens:
        label, token = tokens[index % len(tokens)]
    return label, AgentToolConfig(
        base_url=env["AGENT_TOOL_SERVER_URL"],
        token=token,
        timeout_s=float(env.get("AGENT_TOOL_SERVER_TIMEOUT", "25")),
        retries=int(env.get("AGENT_TOOL_SERVER_RETRIES", "2")),
        verify_tls=env.get("AGENT_TOOL_SERVER_VERIFY_TLS", "1") not in ("0", "false", "False"),
    )


def _load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_ranked_with_scores() -> dict[str, list[RankedRow]]:
    xgb_rows = _load_csv(RANKED_XGB)
    xgb_scores: dict[tuple[str, str, str], tuple[float, float]] = {}
    for r in xgb_rows:
        if r.get("offline_bundle_missing") == "1" or not r.get("node"):
            continue
        xgb_scores[(r["scenario_id"], r["node"], r["fault_reason"])] = (
            float(r.get("calibrated_score") or 0.0),
            float(r.get("uncertainty") or 0.0),
        )

    out: dict[str, list[RankedRow]] = defaultdict(list)
    for r in _load_csv(RANKED):
        if r.get("offline_bundle_missing") == "1" or not r.get("node"):
            continue
        key = (r["scenario_id"], r["node"], r["fault_reason"])
        score = xgb_scores.get(key)
        out[r["scenario_id"]].append(
            RankedRow(
                scenario_id=r["scenario_id"],
                node=r["node"],
                fault_reason=r["fault_reason"],
                category=r["category"],
                combined_score=float(r.get("combined_score") or 0.0),
                calibrated_score=(score[0] if score else None),
                uncertainty=(score[1] if score else None),
            )
        )
    return out


def _load_calibrated_scores() -> dict[str, dict[tuple[str, str, str], tuple[float, float]]]:
    out: dict[str, dict[tuple[str, str, str], tuple[float, float]]] = defaultdict(dict)
    for r in _load_csv(RANKED_XGB):
        if not r.get("node"):
            continue
        out[r["scenario_id"]][(r["scenario_id"], r["node"], r["fault_reason"])] = (
            float(r.get("calibrated_score") or 0.0),
            float(r.get("uncertainty") or 0.0),
        )
    return out


def _load_anomaly_by_sid() -> dict[str, dict[tuple[str, str, str], AnomalyEvidence]]:
    out: dict[str, dict[tuple[str, str, str], AnomalyEvidence]] = defaultdict(dict)
    for r in _load_csv(ANOMALY):
        if r.get("offline_bundle_missing") == "1" or not r.get("node"):
            continue
        key = (r["scenario_id"], r["node"], r["fault_reason"])
        out[r["scenario_id"]][key] = AnomalyEvidence(
            scenario_id=r["scenario_id"],
            node=r["node"],
            fault_reason=r["fault_reason"],
            sample_evidence=r.get("sample_evidence", ""),
            signatures_fired=r.get("signatures_fired", ""),
        )
    return out


def _load_graph_by_sid() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in _load_csv(GRAPH):
        if r.get("offline_bundle_missing") == "1" or not r.get("node"):
            continue
        out[r["scenario_id"]][r["node"]] = r
    return out


def _top_candidates(rows: Iterable[RankedRow], k: int) -> list[RankedRow]:
    seen: set[tuple[str, str]] = set()
    ordered = sorted(
        rows,
        key=lambda r: (
            -(r.calibrated_score if r.calibrated_score is not None else r.combined_score),
            r.uncertainty if r.uncertainty is not None else 0.0,
            r.node,
            r.fault_reason,
        ),
    )
    out: list[RankedRow] = []
    for row in ordered:
        key = (row.node, row.fault_reason)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= k:
            break
    return out


def _last_valid_fault_answer(trace: AgentTrace, question_text: str) -> str:
    routing_vocab, port_vocab = extract_fault_vocab(question_text)
    for rec in reversed(trace.transcript):
        draft = rec.get("draft_answer")
        if not draft:
            continue
        rep = validate_fault(draft, routing_vocab, port_vocab)
        if rep.is_valid:
            return rep.normalised
    return ""


def _last_valid_path_answer(trace: AgentTrace, question_text: str) -> str:
    strict = _path_requires_interfaces(question_text)
    for rec in reversed(trace.transcript):
        draft = rec.get("draft_answer")
        if not draft:
            continue
        rep = validate_path(
            draft,
            require_intermediate_interfaces=strict,
            forbid_final_interface=strict,
        )
        if rep.is_valid:
            return rep.normalised
    return ""


def _last_valid_topology_answer(trace: AgentTrace) -> str:
    for rec in reversed(trace.transcript):
        draft = rec.get("draft_answer")
        if not draft:
            continue
        rep = validate_topology(draft)
        if rep.is_valid:
            return rep.normalised
    return ""


def _no_tools_fallback_path_answer(
    *,
    qwen: QwenClient,
    question_number: int,
    phase: int,
    question_text: str,
) -> str:
    parsed = parse_constraints(question_text)
    denied = load_denied_pairs(question_number, LIMITS)
    sys_prompt = _build_path_system_prompt(parsed=parsed, denied=denied, phase=phase)
    response = qwen.chat(
        [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    question_text
                    + "\n\nReturn ONLY the final path answer now. Do not call tools."
                    " Do not output prose, markdown, or any commentary."
                    " Use the exact schema the question demands."
                    " If multiple paths, one per line. No leading/trailing whitespace."
                ),
            },
        ],
        tools=[],
        temperature=0.0,
    )
    draft = response.content or ""
    strict = _path_requires_interfaces(question_text)
    rep = validate_path(
        draft,
        require_intermediate_interfaces=strict,
        forbid_final_interface=strict,
    )
    return rep.normalised if rep.is_valid else ""


def _no_tools_fallback_topology_answer(
    *,
    qwen: QwenClient,
    question_number: int,
    phase: int,
    question_text: str,
) -> str:
    parsed = parse_constraints(question_text)
    denied = load_denied_pairs(question_number, LIMITS)
    sys_prompt = _build_topology_system_prompt(parsed=parsed, denied=denied, phase=phase)
    response = qwen.chat(
        [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    question_text
                    + "\n\nReturn ONLY the final topology-link answer now. Do not call tools."
                    " Use local_node(local_port)->remote_node(remote_port), one link per line."
                    " Do not output prose, markdown, or any commentary."
                ),
            },
        ],
        tools=[],
        temperature=0.0,
    )
    draft = response.content or ""
    rep = validate_topology(draft)
    return rep.normalised if rep.is_valid else ""


def _completed_ids(result_csv: Path) -> set[int]:
    if not result_csv.is_file():
        return set()
    with open(result_csv, "r", encoding="utf-8") as f:
        return {int(r["id"]) for r in csv.DictReader(f) if r.get("id")}


def _stub_qwen() -> QwenClient:
    def policy(messages, tools, temperature=None, seed=None):
        if not any(m.get("role") == "tool" for m in messages):
            return ChatResponse(
                role="assistant",
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_submission_1",
                        name="infra_maintenance",
                        arguments={
                            "device_name": "Core_SW_01",
                            "command": "display current-configuration",
                        },
                    )
                ],
            )
        return ChatResponse(
            role="assistant",
            content="Core_SW_01;10.1.60.2;blackhole route",
            finish_reason="stop",
        )

    return QwenClient(policy=policy)


def _stub_dispatch(*, tool_name, arguments, question_number, config):
    return CommandResult(
        status_code=200,
        status="success",
        device_name=str(arguments.get("device_name")),
        command=str(arguments.get("command")),
        vendor="huawei",
        result_text="stub output",
        raw={},
    )


def _infer_phase(input_json: Path) -> int:
    path = str(input_json)
    if "Phase_1" in path:
        return 1
    return 2


def _no_tools_fallback_answer(
    *,
    qwen: QwenClient,
    scenario_id: str,
    question_number: int,
    phase: int,
    question_text: str,
    candidates: list[RankedRow],
) -> str:
    parsed = parse_constraints(question_text)
    denied = load_denied_pairs(question_number, LIMITS)
    routing_vocab, port_vocab = extract_fault_vocab(question_text)
    context = build_context(
        task_family="fault",
        parsed=parsed,
        candidates=candidates,
        anomaly_evidence={},
        denied_pairs=denied,
    )
    sys_prompt = build_system_prompt(
        context=context,
        few_shot_exemplars=(),
        include_phase_2_device_list=(phase == 2),
    )
    response = qwen.chat(
        [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    question_text
                    + "\n\nReturn only the final answer now. Do not call tools."
                    + "\nAllowed routing reasons: "
                    + ", ".join(routing_vocab)
                    + "\nAllowed port reasons: "
                    + ", ".join(port_vocab)
                    + "\nDo not output symptom labels; output only a final fault reason from the allowed lists."
                ),
            },
        ],
        tools=[],
        temperature=0.0,
    )
    draft = response.content or ""
    rep = validate_fault(draft, routing_vocab, port_vocab)
    return rep.normalised if rep.is_valid else ""


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.out_dir / "result.csv"
    trace_jsonl = args.out_dir / "eval_detail.jsonl"

    dotenv = _load_dotenv(DOTENV)
    env = _resolved_env(dotenv)
    tokens = _available_tool_tokens(dotenv, env)

    with open(args.input_json, "r", encoding="utf-8") as f:
        questions = json.load(f)
    selected_ids = _parse_id_selector(args.ids) if args.ids else set()
    if selected_ids:
        questions = [q for q in questions if int(q["task"]["id"]) in selected_ids]
    if args.limit > 0:
        questions = questions[:args.limit]

    ranked_by_sid = _load_ranked_with_scores()
    graph_by_sid = _load_graph_by_sid()
    anomaly_by_sid = _load_anomaly_by_sid()
    calibrated_by_sid = _load_calibrated_scores()
    phase = _infer_phase(args.input_json)

    qwen = _stub_qwen() if args.stub else _qwen_from_env(env)
    completed = _completed_ids(result_csv) if args.resume else set()

    backend = "STUB" if args.stub else (env.get("OPENAI_BASE_URL") or "STUB")
    model = "stub-policy" if args.stub else env.get("QWEN_MODEL", "Qwen/Qwen3.5-35B-A3B")
    tool_url = "stub" if args.stub else env.get("AGENT_TOOL_SERVER_URL", "")
    print(
        f"==> submission config\n"
        f"   input_json={args.input_json}\n"
        f"   out_dir={args.out_dir}\n"
        f"   n_questions={len(questions)}  ids_filter={args.ids or '(all)'}  limit={args.limit or '(no cap)'}\n"
        f"   qwen_base_url={backend}  model={model}\n"
        f"   agent_tool_server={tool_url}  tool_token_slots={len(tokens)}\n"
        f"   caps: max_iterations={args.max_iterations}  max_tool_calls={args.max_tool_calls}  workers={args.workers}\n"
        f"   resume={args.resume}  stub={args.stub}  phase={phase}"
    )

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    write_header = not (args.resume and result_csv.is_file())
    write_lock = threading.Lock()

    with open(result_csv, "a" if args.resume else "w", encoding="utf-8", newline="") as csv_f, \
            open(trace_jsonl, "a" if args.resume else "w", encoding="utf-8") as trace_f:
        csv_writer = csv.writer(csv_f)
        if write_header:
            csv_writer.writerow(["id", "prediction"])

        t0 = time.perf_counter()
        ran = 0

        def _process_item(idx_item):
            idx, item = idx_item
            task_id = int(item["task"]["id"])
            if task_id in completed:
                return None

            scenario_id = item["scenario_id"]
            question_text = item["task"]["question"]
            family = classify(question_text)
            token_label, tool_config = _tool_config_for_index(env, tokens, idx)
            prediction = ""
            trace_payload: dict[str, object] = {
                "question_id": task_id,
                "scenario_id": scenario_id,
                "task_family": family,
                "tool_token_slot": token_label,
            }
            limits = AgentLimits(
                max_iterations=args.max_iterations,
                max_tool_calls=args.max_tool_calls,
            )

            if family == "fault":
                candidates = _top_candidates(ranked_by_sid.get(scenario_id, []), args.top_k)
                graph_features = graph_by_sid.get(scenario_id, {})
                anomaly_evidence = anomaly_by_sid.get(scenario_id, {})
                calibrated_scores = calibrated_by_sid.get(scenario_id) or None
                if args.stub:
                    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_stub_dispatch):
                        trace = run_scenario(
                            scenario_id=scenario_id, question_number=task_id, phase=phase,
                            question_text=question_text, candidates=candidates,
                            graph_features=graph_features, anomaly_evidence=anomaly_evidence,
                            qwen=qwen, tool_config=tool_config, limits_path=str(LIMITS),
                            limits=limits, calibrated_scores=calibrated_scores,
                        )
                else:
                    trace = run_scenario(
                        scenario_id=scenario_id, question_number=task_id, phase=phase,
                        question_text=question_text, candidates=candidates,
                        graph_features=graph_features, anomaly_evidence=anomaly_evidence,
                        qwen=qwen, tool_config=tool_config, limits_path=str(LIMITS),
                        limits=limits, calibrated_scores=calibrated_scores,
                    )
                prediction = trace.final_answer if trace.final_action == "accept" else _last_valid_fault_answer(trace, question_text)
                used_no_tools_fallback = False
                if not prediction and not args.stub:
                    prediction = _no_tools_fallback_answer(
                        qwen=qwen, scenario_id=scenario_id, question_number=task_id,
                        phase=phase, question_text=question_text, candidates=candidates,
                    )
                    used_no_tools_fallback = bool(prediction)
                trace_payload.update({
                    "prediction": prediction,
                    "used_fallback_prediction": trace.final_action != "accept" and bool(prediction),
                    "used_no_tools_fallback": used_no_tools_fallback,
                    "final_action": trace.final_action, "iterations": trace.iterations,
                    "tool_calls_made": trace.tool_calls_made, "follow_ups_triggered": trace.follow_ups_triggered,
                    "format_rejections": trace.format_rejections, "constraint_rejections": trace.constraint_rejections,
                    "errors": trace.errors, "transcript": trace.transcript, "n_candidates": len(candidates),
                })

            elif family == "path":
                if args.stub:
                    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_stub_dispatch):
                        trace = run_scenario(
                            scenario_id=scenario_id, question_number=task_id, phase=phase,
                            question_text=question_text, candidates=[], graph_features={},
                            anomaly_evidence={}, qwen=qwen, tool_config=tool_config,
                            limits_path=str(LIMITS), limits=limits, calibrated_scores=None,
                        )
                else:
                    trace = run_scenario(
                        scenario_id=scenario_id, question_number=task_id, phase=phase,
                        question_text=question_text, candidates=[], graph_features={},
                        anomaly_evidence={}, qwen=qwen, tool_config=tool_config,
                        limits_path=str(LIMITS), limits=limits, calibrated_scores=None,
                    )
                prediction = trace.final_answer if trace.final_action == "accept" else _last_valid_path_answer(trace, question_text)
                used_no_tools_fallback = False
                if not prediction and not args.stub:
                    prediction = _no_tools_fallback_path_answer(
                        qwen=qwen, question_number=task_id, phase=phase, question_text=question_text,
                    )
                    used_no_tools_fallback = bool(prediction)
                trace_payload.update({
                    "prediction": prediction,
                    "used_fallback_prediction": trace.final_action != "accept" and bool(prediction),
                    "used_no_tools_fallback": used_no_tools_fallback,
                    "final_action": trace.final_action, "iterations": trace.iterations,
                    "tool_calls_made": trace.tool_calls_made, "follow_ups_triggered": trace.follow_ups_triggered,
                    "format_rejections": trace.format_rejections, "constraint_rejections": trace.constraint_rejections,
                    "errors": trace.errors, "transcript": trace.transcript, "n_candidates": 0,
                })

            elif family == "topology":
                if args.stub:
                    with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_stub_dispatch):
                        trace = run_scenario(
                            scenario_id=scenario_id, question_number=task_id, phase=phase,
                            question_text=question_text, candidates=[], graph_features={},
                            anomaly_evidence={}, qwen=qwen, tool_config=tool_config,
                            limits_path=str(LIMITS), limits=limits, calibrated_scores=None,
                        )
                else:
                    trace = run_scenario(
                        scenario_id=scenario_id, question_number=task_id, phase=phase,
                        question_text=question_text, candidates=[], graph_features={},
                        anomaly_evidence={}, qwen=qwen, tool_config=tool_config,
                        limits_path=str(LIMITS), limits=limits, calibrated_scores=None,
                    )
                prediction = trace.final_answer if trace.final_action == "accept" else _last_valid_topology_answer(trace)
                used_no_tools_fallback = False
                if not prediction and not args.stub:
                    prediction = _no_tools_fallback_topology_answer(
                        qwen=qwen, question_number=task_id, phase=phase, question_text=question_text,
                    )
                    used_no_tools_fallback = bool(prediction)
                trace_payload.update({
                    "prediction": prediction,
                    "used_fallback_prediction": trace.final_action != "accept" and bool(prediction),
                    "used_no_tools_fallback": used_no_tools_fallback,
                    "final_action": trace.final_action, "iterations": trace.iterations,
                    "tool_calls_made": trace.tool_calls_made, "follow_ups_triggered": trace.follow_ups_triggered,
                    "format_rejections": trace.format_rejections, "constraint_rejections": trace.constraint_rejections,
                    "errors": trace.errors, "transcript": trace.transcript, "n_candidates": 0,
                })

            else:
                trace_payload.update({
                    "prediction": "",
                    "final_action": "unsupported_task_family",
                    "errors": [f"task family {family!r} is unsupported by the current runtime"],
                    "transcript": [],
                })

            return task_id, scenario_id, prediction, trace_payload

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_item, (idx, item)): item
                for idx, item in enumerate(questions)
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[ERR] scenario failed: {exc}")
                    continue
                if result is None:
                    continue
                task_id, scenario_id, prediction, trace_payload = result
                with write_lock:
                    csv_writer.writerow([task_id, prediction])
                    csv_f.flush()
                    trace_f.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
                    trace_f.flush()
                    ran += 1
                    print(
                        f"[{ran:>3}] q={task_id:>3} sid={scenario_id[:8]} "
                        f"pred={'yes' if prediction else 'no '} action={trace_payload['final_action']}"
                    )

        dt = time.perf_counter() - t0

    print()
    print(f"wrote {result_csv}")
    print(f"wrote {trace_jsonl}")
    print(f"questions processed: {ran}")
    print(f"elapsed: {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
