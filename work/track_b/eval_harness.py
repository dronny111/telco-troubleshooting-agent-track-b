"""Local eval harness for the V0–V4 ablation suite.

Plumbing only — the LLM call is stubbed with a deterministic "top-K from
the variant's strongest score column" predictor so the harness can be
exercised before Qwen serving exists. Once the Qwen agent runtime is
ready, replace `_stub_predict()` with the real agent loop; everything
else (variant configs, scoring, metrics, ablations) is unchanged.

Variants (per the refined plan):

    V0  — bare LLM, no graph, no playbook, no validator.
    V1  — constraint parser + format guard + playbook.
    V2  — V1 + offline graph + permission pruner + deterministic ranker.
    V3  — V2 + anomaly miner + answer validator + 1-shot follow-up.
    V4  — V3 + XGBoost calibration + uncertainty-gated follow-up.

Metrics emitted per variant (as a `VariantReport`):
    accuracy_exact      — predicted set equals silver-positive set
    accuracy_top1       — top-1 line is a silver positive
    precision           — micro precision over fault lines
    recall              — micro recall over fault lines
    f1                  — harmonic mean
    mean_calls          — average API-style calls per scenario (0 for the
                          stub variants; non-zero only when V3/V4 trigger
                          the 1-shot follow-up)
    calls_per_correct   — total calls / number of exact-match correct
                          (Phase 2 tiebreaker proxy)
    format_error_rate   — fraction of answers rejected by format guard
    follow_up_rate      — fraction of scenarios where the validator
                          triggered fetch_evidence

Variance is computed by running the predictor n_reruns times; with the
deterministic stub the variance is 0, but the same call surface lets
the real Qwen runtime measure Pass@1 stability for free.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LIMITS_PATH = _ROOT / "telco_data" / "Track B" / "question_limits_config.json"

from unittest.mock import patch

from .agent_runtime import AgentLimits, AgentTrace, run_scenario
from .agent_tools import AgentToolConfig, CommandResult
from .answer_validator import validate_answer
from .constraint_parser import parse as parse_constraints
from .format_guard import validate_fault
from .permission_pruner import denied_pairs as load_denied_pairs
from .playbook import lookup as playbook_lookup
from .prompt_context import AnomalyEvidence, RankedRow
from .qwen_client import ChatResponse, QwenClient, QwenConfig
from .task_classifier import classify
from .vocab_extractor import extract_fault_vocab


# ---- Variant configuration -------------------------------------------------

@dataclass(frozen=True)
class VariantConfig:
    name: str
    use_constraint_parser: bool = False
    use_playbook: bool = False
    use_graph: bool = False
    use_permission_pruner: bool = False
    use_deterministic_ranker: bool = False
    use_anomaly_miner: bool = False
    use_answer_validator: bool = False
    use_xgboost: bool = False


VARIANTS: dict[str, VariantConfig] = {
    "V0": VariantConfig(
        name="V0",
    ),
    "V1": VariantConfig(
        name="V1",
        use_constraint_parser=True,
        use_playbook=True,
    ),
    "V2": VariantConfig(
        name="V2",
        use_constraint_parser=True,
        use_playbook=True,
        use_graph=True,
        use_permission_pruner=True,
        use_deterministic_ranker=True,
    ),
    "V3": VariantConfig(
        name="V3",
        use_constraint_parser=True,
        use_playbook=True,
        use_graph=True,
        use_permission_pruner=True,
        use_deterministic_ranker=True,
        use_anomaly_miner=True,
        use_answer_validator=True,
    ),
    "V4": VariantConfig(
        name="V4",
        use_constraint_parser=True,
        use_playbook=True,
        use_graph=True,
        use_permission_pruner=True,
        use_deterministic_ranker=True,
        use_anomaly_miner=True,
        use_answer_validator=True,
        use_xgboost=True,
    ),
}


# ---- Scenario inputs -------------------------------------------------------

@dataclass
class ScenarioInputs:
    scenario_id: str
    question_number: int
    phase: int
    question_text: str
    routing_vocab: tuple[str, ...]
    port_vocab: tuple[str, ...]
    candidates_deterministic: list[dict]   # ranker rows: {node, fault_reason, category, combined_score}
    candidates_xgb: list[dict]             # xgb rows: {... calibrated_score, uncertainty}
    graph_features: dict[str, dict]        # node -> dict
    anomaly_evidence: set[tuple[str, str, str]]
    denied_pairs: set[tuple[str, str]]
    silver_positives: set[tuple[str, str]] # {(node, fault_reason), ...}


# ---- Stub predictor --------------------------------------------------------

def _stub_predict_topk(
    cands: Sequence[dict],
    score_col: str,
    k: int,
    tau: float | None = None,
) -> list[dict]:
    """Pick top-k candidates (deduped by node, fault_reason).

    If `tau` is given, drop any candidate whose `score_col` is below tau
    *before* the top-k truncation. If every candidate is below tau the
    return is empty — callers should treat that as "abstain" (emit an
    empty answer, which the format guard rejects rather than scoring as
    a wrong answer).
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for c in sorted(cands, key=lambda r: -float(r.get(score_col, 0.0))):
        if tau is not None and float(c.get(score_col, 0.0)) < tau:
            break
        key = (c["node"], c["fault_reason"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= k:
            break
    return out


def _format_answer_lines(picks: Sequence[dict], inputs: ScenarioInputs) -> str:
    """Render top-K picks as a fault-format answer."""
    lines: list[str] = []
    parsed = parse_constraints(inputs.question_text)
    for p in picks:
        node = p["node"]
        cat = p.get("category", "routing")
        if cat == "routing":
            mid = parsed.target_destination_ip or "192.168.1.1"
        else:
            # Use a plausible-looking interface name; real Qwen would have
            # collected this from interface-brief output.
            mid = "GE1/0/1"
        lines.append(f"{node};{mid};{p['fault_reason']}")
    return "\n".join(lines)


def _stub_predict(
    variant: VariantConfig,
    inputs: ScenarioInputs,
    k: int = 1,
    tau: float | None = None,
) -> str:
    """Deterministic placeholder for the Qwen call.

    Each variant only sees the inputs its config enables; this keeps the
    ablation honest before the real LLM lands.

    `tau` is the abstention threshold for the variant's primary score
    column (calibrated_score for V4, combined_score otherwise). When
    every candidate falls below tau the function returns an empty
    answer, which downstream scoring treats as an abstention rather
    than a wrong guess.
    """
    if variant.use_xgboost and inputs.candidates_xgb:
        picks = _stub_predict_topk(inputs.candidates_xgb, "calibrated_score", k, tau=tau)
    elif variant.use_deterministic_ranker and inputs.candidates_deterministic:
        picks = _stub_predict_topk(inputs.candidates_deterministic, "combined_score", k, tau=tau)
    elif variant.use_constraint_parser and variant.use_playbook:
        # V1: pick a centrality-relevant device + first allowed fault reason
        # from the parsed protocol family. With no graph/ranker we have very
        # little to go on; fall back to deterministic top-1 if available
        # (the test mostly stresses the validator path).
        picks = _stub_predict_topk(inputs.candidates_deterministic or [], "combined_score", k, tau=tau)
    else:
        # V0: emit the question's first listed fault example to simulate
        # the no-context degenerate case (will likely fail format-vocab).
        picks = []
    if not picks:
        # Empty answer triggers a format error in scoring; that's a
        # realistic V0 outcome when there is no useful signal.
        return ""
    return _format_answer_lines(picks, inputs)


# ---- Agent-runtime predictor ----------------------------------------------

def _filter_candidates(
    cands: Sequence[dict],
    score_col: str,
    tau: float | None,
    k_max: int,
) -> list[dict]:
    """Sort by score_col desc, drop rows below tau, take top-k_max."""
    pool = sorted(cands, key=lambda r: -float(r.get(score_col, 0.0)))
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for c in pool:
        if tau is not None and float(c.get(score_col, 0.0)) < tau:
            break
        key = (c.get("node", ""), c.get("fault_reason", ""))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= k_max:
            break
    return out


def _to_ranked_rows(filtered: Sequence[dict], scenario_id: str) -> list[RankedRow]:
    out: list[RankedRow] = []
    for c in filtered:
        cal = c.get("calibrated_score")
        unc = c.get("uncertainty")
        out.append(RankedRow(
            scenario_id=scenario_id,
            node=c["node"],
            fault_reason=c["fault_reason"],
            category=c.get("category", "routing"),
            combined_score=float(c.get("combined_score", 0.0)),
            calibrated_score=float(cal) if cal not in (None, "", "None") else None,
            uncertainty=float(unc) if unc not in (None, "", "None") else None,
        ))
    return out


def _build_topk_stub_policy(picks: Sequence[RankedRow]) -> Callable:
    """Stub Qwen policy that emits a fault-shape answer from `picks`.

    The policy ignores tool offers and goes straight to a final answer;
    the validator may still trigger a re-emit if the format guard fails
    (e.g., reason not in vocab). When `picks` is empty the policy emits
    an empty answer — that exercises the harness's abstain path.
    """
    def policy(messages, tools, temperature=None, seed=None):
        if not picks:
            return ChatResponse(role="assistant", content="", finish_reason="stop")
        lines: list[str] = []
        for r in picks:
            mid = "192.168.1.1" if r.category == "routing" else "GE1/0/1"
            lines.append(f"{r.node};{mid};{r.fault_reason}")
        return ChatResponse(role="assistant", content="\n".join(lines), finish_reason="stop")
    return policy


def _stub_tool_dispatch(*, tool_name, arguments, question_number, config):
    return CommandResult(
        status_code=200, status="success",
        device_name=str(arguments.get("device_name", "")),
        command=str(arguments.get("command", "")),
        vendor="huawei", result_text="", raw={},
    )


def _agent_predict(
    variant: VariantConfig,
    inputs: ScenarioInputs,
    *,
    tau: float | None,
    k_max: int,
    limits_path: Path = _DEFAULT_LIMITS_PATH,
) -> AgentTrace:
    """Drive the full `run_scenario` loop with the configured Qwen client.

    Replaces the shallow `_stub_predict` path: the same threshold/k_max
    semantics still apply, but emission now flows through the format
    guard, validator, follow-up logic, and family dispatcher in
    `agent_runtime.run_scenario`.

    Backend selection mirrors `run_agent_demo.py`:
      * OPENAI_BASE_URL set  → real Qwen via QwenClient(QwenConfig.from_env)
                               and the real tool dispatcher (live simulator).
      * OPENAI_BASE_URL unset → stub Qwen policy + stub tool dispatcher
                                (deterministic, no network).
    Set OPENAI_BASE_URL to point at vLLM (see `deploy/env.example`).
    """
    family = classify(inputs.question_text)
    if family != "fault":
        # Path/topology have their own internal scenario branches and
        # no silver labels in the local artifacts. Emit empty so the
        # caller's scoring records this as abstain rather than wrong.
        return AgentTrace(scenario_id=inputs.scenario_id, final_action="skipped_family")

    score_col = _primary_score_col(variant)
    cands = inputs.candidates_xgb if variant.use_xgboost else inputs.candidates_deterministic
    filtered = _filter_candidates(cands, score_col=score_col, tau=tau, k_max=k_max)
    ranked = _to_ranked_rows(filtered, scenario_id=inputs.scenario_id)

    cal_scores: dict[tuple[str, str, str], tuple[float, float]] | None = None
    if variant.use_xgboost and inputs.candidates_xgb:
        cal_scores = {}
        for c in inputs.candidates_xgb:
            key = (inputs.scenario_id, c.get("node", ""), c.get("fault_reason", ""))
            cs = c.get("calibrated_score")
            unc = c.get("uncertainty")
            cal_scores[key] = (
                float(cs) if cs not in (None, "", "None") else 0.5,
                float(unc) if unc not in (None, "", "None") else 0.0,
            )

    anomaly_dict: dict[tuple[str, str, str], AnomalyEvidence] = {}
    if variant.use_anomaly_miner:
        for (sid, node, reason) in inputs.anomaly_evidence:
            anomaly_dict[(sid, node, reason)] = AnomalyEvidence(
                scenario_id=sid, node=node, fault_reason=reason, sample_evidence="",
            )

    qwen_config = QwenConfig.from_env()
    if qwen_config is None:
        # Stub path: Qwen policy + tool dispatcher patched out.
        qwen = QwenClient(policy=_build_topk_stub_policy(ranked))
        tool_config = AgentToolConfig(base_url="http://localhost:0", token="")
        eval_limits = AgentLimits(max_iterations=3, max_tool_calls=5)
        with patch("track_b.agent_runtime.dispatch_tool_call", side_effect=_stub_tool_dispatch):
            return run_scenario(
                scenario_id=inputs.scenario_id,
                question_number=inputs.question_number,
                phase=inputs.phase,
                question_text=inputs.question_text,
                candidates=ranked,
                graph_features=inputs.graph_features,
                anomaly_evidence=anomaly_dict,
                qwen=qwen,
                tool_config=tool_config,
                limits_path=str(limits_path),
                limits=eval_limits,
                calibrated_scores=cal_scores,
            )

    # Real Qwen + real Agent Tool Server. Caps are deliberately tighter
    # than the per-scenario 500 ceiling because the sweep multiplies
    # scenarios × taus × k_max; pick a budget you can afford to spend.
    qwen = QwenClient(qwen_config)
    tool_config = AgentToolConfig.from_env()
    eval_limits = AgentLimits(max_iterations=6, max_tool_calls=10)
    return run_scenario(
        scenario_id=inputs.scenario_id,
        question_number=inputs.question_number,
        phase=inputs.phase,
        question_text=inputs.question_text,
        candidates=ranked,
        graph_features=inputs.graph_features,
        anomaly_evidence=anomaly_dict,
        qwen=qwen,
        tool_config=tool_config,
        limits_path=str(limits_path),
        limits=eval_limits,
        calibrated_scores=cal_scores,
    )


def _primary_score_col(variant: VariantConfig) -> str:
    return "calibrated_score" if variant.use_xgboost else "combined_score"


# ---- Per-scenario evaluation ---------------------------------------------

@dataclass
class ScenarioTrace:
    scenario_id: str
    answer: str
    n_calls: int = 0
    n_format_errors: int = 0
    n_follow_ups: int = 0
    final_action: str = "accept"


def run_variant_on_scenario(
    *,
    variant: VariantConfig,
    inputs: ScenarioInputs,
    n_iters_max: int = 2,
    tau: float | None = None,
    k_max: int = 1,
    use_agent_runtime: bool = False,
) -> ScenarioTrace:
    """Drive the predictor through the validator loop.

    n_iters_max bounds the re-emit cycles per question; in production this
    matches the agent harness's retry cap. Set to 2 by default: one initial
    draft + one validator-driven re-emit.

    `tau` is an abstention threshold on the variant's primary score column.
    `k_max` lets multiple lines be emitted when several candidates clear tau.

    When `use_agent_runtime` is True, emission flows through
    `agent_runtime.run_scenario` instead of the shallow `_stub_predict`:
    same threshold/k_max semantics, but the full prompt-context + format
    guard + validator + follow-up loop runs. The Qwen client uses a
    deterministic stub policy so no API calls are made.
    """
    trace = ScenarioTrace(scenario_id=inputs.scenario_id, answer="")

    if use_agent_runtime:
        agent_trace = _agent_predict(variant, inputs, tau=tau, k_max=k_max)
        trace.answer = agent_trace.final_answer
        trace.n_calls = agent_trace.tool_calls_made
        trace.n_follow_ups = agent_trace.follow_ups_triggered
        trace.n_format_errors = agent_trace.format_rejections
        trace.final_action = agent_trace.final_action
        return trace

    if not variant.use_answer_validator:
        # V0–V2: skip the validator stack entirely.
        ans = _stub_predict(variant, inputs, k=k_max, tau=tau)
        rep = validate_fault(ans, inputs.routing_vocab, inputs.port_vocab)
        trace.answer = rep.normalised if rep.is_valid else ans
        if not rep.is_valid:
            trace.n_format_errors = 1
        return trace

    # V3 / V4 — validator-gated path.
    fetched: set[tuple[str, str]] = set()
    for it in range(n_iters_max):
        ans = _stub_predict(variant, inputs, k=k_max, tau=tau)
        cal_scores: dict | None = None
        if variant.use_xgboost and inputs.candidates_xgb:
            cal_scores = {
                (inputs.scenario_id, c["node"], c["fault_reason"]):
                    (float(c.get("calibrated_score", 0.5)), float(c.get("uncertainty", 0.0)))
                for c in inputs.candidates_xgb
            }
        decision = validate_answer(
            draft_answer=ans,
            scenario_id=inputs.scenario_id,
            parsed=parse_constraints(inputs.question_text),
            routing_vocab=inputs.routing_vocab,
            port_vocab=inputs.port_vocab,
            graph_features=inputs.graph_features,
            anomaly_evidence=inputs.anomaly_evidence if variant.use_anomaly_miner else set(),
            denied_pairs=inputs.denied_pairs if variant.use_permission_pruner else set(),
            calibrated_scores=cal_scores,
            fetched_commands=fetched,
        )
        if decision.action == "reemit_format":
            trace.n_format_errors += 1
            trace.answer = decision.normalised_answer
            continue
        if decision.action == "reemit_constraint":
            trace.answer = decision.normalised_answer
            continue
        if decision.action == "fetch_evidence":
            trace.n_follow_ups += 1
            trace.n_calls += 1
            if decision.follow_up:
                fetched.add((decision.follow_up[0], decision.follow_up[1]))
            trace.answer = decision.normalised_answer
            continue
        # accept
        trace.answer = decision.normalised_answer
        trace.final_action = "accept"
        return trace
    trace.final_action = "ran_out_of_iters"
    return trace


# ---- Scoring -------------------------------------------------------------

def _score_answer(
    answer_text: str,
    silver_positives: set[tuple[str, str]],
    routing_vocab: tuple[str, ...],
    port_vocab: tuple[str, ...],
) -> tuple[bool, bool, int, int, int]:
    """Return (exact_match, top1_match, tp, fp, fn).

    Predictions are extracted by parsing the answer with the format guard
    and pulling (node, reason) from each line. Format-rejected answers
    score (False, False, 0, 0, len(silver_positives)).
    """
    rep = validate_fault(answer_text, frozenset(routing_vocab), frozenset(port_vocab))
    if not rep.is_valid:
        return False, False, 0, 0, len(silver_positives)
    pairs: list[tuple[str, str]] = []
    for line in rep.normalised.splitlines():
        parts = line.split(";")
        if len(parts) == 3:
            pairs.append((parts[0], parts[2]))
    pred_set = set(pairs)
    silver = set(silver_positives)
    tp = len(pred_set & silver)
    fp = len(pred_set - silver)
    fn = len(silver - pred_set)
    exact = pred_set == silver and bool(silver)
    top1 = bool(pairs) and pairs[0] in silver
    return exact, top1, tp, fp, fn


# ---- Variant report -------------------------------------------------------

@dataclass
class VariantReport:
    variant: str
    n_scenarios: int = 0
    n_exact: int = 0
    n_top1: int = 0
    sum_tp: int = 0
    sum_fp: int = 0
    sum_fn: int = 0
    sum_calls: int = 0
    sum_format_errors: int = 0
    sum_follow_ups: int = 0

    @property
    def accuracy_exact(self) -> float:
        return self.n_exact / max(1, self.n_scenarios)

    @property
    def accuracy_top1(self) -> float:
        return self.n_top1 / max(1, self.n_scenarios)

    @property
    def precision(self) -> float:
        denom = self.sum_tp + self.sum_fp
        return self.sum_tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.sum_tp + self.sum_fn
        return self.sum_tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_calls(self) -> float:
        return self.sum_calls / max(1, self.n_scenarios)

    @property
    def calls_per_correct(self) -> float:
        return self.sum_calls / max(1, self.n_exact)

    @property
    def format_error_rate(self) -> float:
        return self.sum_format_errors / max(1, self.n_scenarios)

    @property
    def follow_up_rate(self) -> float:
        return self.sum_follow_ups / max(1, self.n_scenarios)

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "n_scenarios": self.n_scenarios,
            "accuracy_exact": round(self.accuracy_exact, 4),
            "accuracy_top1": round(self.accuracy_top1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_calls": round(self.mean_calls, 4),
            "calls_per_correct": round(self.calls_per_correct, 4),
            "format_error_rate": round(self.format_error_rate, 4),
            "follow_up_rate": round(self.follow_up_rate, 4),
        }


def run_eval(
    variants: Iterable[VariantConfig],
    scenarios: Iterable[ScenarioInputs],
) -> list[VariantReport]:
    scenarios = list(scenarios)
    reports: list[VariantReport] = []
    for v in variants:
        rep = VariantReport(variant=v.name, n_scenarios=len(scenarios))
        for s in scenarios:
            trace = run_variant_on_scenario(variant=v, inputs=s)
            exact, top1, tp, fp, fn = _score_answer(
                trace.answer, s.silver_positives, s.routing_vocab, s.port_vocab,
            )
            rep.n_exact += int(exact)
            rep.n_top1 += int(top1)
            rep.sum_tp += tp
            rep.sum_fp += fp
            rep.sum_fn += fn
            rep.sum_calls += trace.n_calls
            rep.sum_format_errors += trace.n_format_errors
            rep.sum_follow_ups += trace.n_follow_ups
        reports.append(rep)
    return reports


# ---- Per-signal ablation -------------------------------------------------

# Ablation toggles operate on the deterministic ranker's component scores.
# Dropping a signal means setting that component to 0 in the candidate
# rows used to seed the variant. This is a pure-Python ablation harness
# that never re-trains XGBoost.
ABLATABLE_COMPONENTS: tuple[str, ...] = (
    "graph_centrality_norm",
    "path_relevance_norm",
    "protocol_match_norm",
    "anomaly_prior_norm",
    "permission_survivor_norm",
    "disclosed_match_norm",
)


def ablate_one_signal(
    *,
    base_variant: VariantConfig,
    scenarios: Sequence[ScenarioInputs],
    component: str,
) -> VariantReport:
    """Drop `component` from every candidate row, recompute combined_score,
    and re-run the variant. Component must be one of ABLATABLE_COMPONENTS.
    """
    if component not in ABLATABLE_COMPONENTS:
        raise ValueError(f"unknown component {component!r}")
    # Approximate the ranker's combine: combined_score ≈ sum of weighted
    # normalised components (weights are the deterministic defaults from
    # ranker.RankerWeights). For ablation purposes we rebuild combined
    # from per-row component fields if present.
    weights = {
        "graph_centrality_norm": 0.5,
        "path_relevance_norm": 1.5,
        "protocol_match_norm": 1.0,
        "vendor_compat_norm": 0.5,
        "anomaly_prior_norm": 2.0,
        "permission_survivor_norm": 0.5,
        "disclosed_match_norm": 1.5,
    }
    ablated_scenarios: list[ScenarioInputs] = []
    for s in scenarios:
        new_dets = []
        for c in s.candidates_deterministic:
            recomputed = sum(
                weights[k] * float(c.get(k, 0.0))
                for k in weights
                if k != component and k in c
            )
            recomputed -= 3.0 * float(c.get("contradiction_penalty_raw", 0.0))
            new_dets.append({**c, "combined_score": recomputed})
        ablated_scenarios.append(
            ScenarioInputs(
                scenario_id=s.scenario_id,
                question_number=s.question_number,
                phase=s.phase,
                question_text=s.question_text,
                routing_vocab=s.routing_vocab,
                port_vocab=s.port_vocab,
                candidates_deterministic=new_dets,
                candidates_xgb=s.candidates_xgb,
                graph_features=s.graph_features,
                anomaly_evidence=s.anomaly_evidence,
                denied_pairs=s.denied_pairs,
                silver_positives=s.silver_positives,
            )
        )
    [report] = run_eval([base_variant], ablated_scenarios)
    report.variant = f"{base_variant.name}-no-{component}"
    return report


# ---- Per-family threshold sweep ------------------------------------------

# Three task families recognised by the format guard. Only `fault` has
# silver labels in the local artifacts; path/topology are reported as
# `no_silver` and need a live submission + reference to sweep against
# (use `local_eval.py self` for those).
SWEEP_FAMILIES: tuple[str, ...] = ("fault", "path", "topology")


@dataclass
class ThresholdSweepRow:
    variant: str
    family: str
    score_col: str
    tau: float
    k_max: int
    n_scenarios: int
    n_emitted: int = 0           # scenarios with non-empty draft
    n_abstained: int = 0         # scenarios where every candidate < tau
    n_exact: int = 0
    n_top1: int = 0
    sum_tp: int = 0
    sum_fp: int = 0
    sum_fn: int = 0
    sum_calls: int = 0
    sum_format_errors: int = 0

    @property
    def emit_rate(self) -> float:
        return self.n_emitted / max(1, self.n_scenarios)

    @property
    def abstain_rate(self) -> float:
        return self.n_abstained / max(1, self.n_scenarios)

    @property
    def accuracy_exact(self) -> float:
        return self.n_exact / max(1, self.n_scenarios)

    @property
    def precision(self) -> float:
        denom = self.sum_tp + self.sum_fp
        return self.sum_tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.sum_tp + self.sum_fn
        return self.sum_tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def calls_per_correct(self) -> float:
        return self.sum_calls / max(1, self.n_exact)

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "family": self.family,
            "score_col": self.score_col,
            "tau": round(self.tau, 4),
            "k_max": self.k_max,
            "n_scenarios": self.n_scenarios,
            "emit_rate": round(self.emit_rate, 4),
            "abstain_rate": round(self.abstain_rate, 4),
            "accuracy_exact": round(self.accuracy_exact, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "calls_per_correct": round(self.calls_per_correct, 4),
            "format_error_rate": round(self.sum_format_errors / max(1, self.n_scenarios), 4),
        }


def _filter_fault_scenarios(scenarios: Sequence[ScenarioInputs]) -> list[ScenarioInputs]:
    """Keep only scenarios for which we have any silver positives; the
    sweep is meaningless when ground-truth recall == 0 for every tau."""
    return [s for s in scenarios if s.silver_positives]


def run_threshold_sweep_one(
    *,
    variant: VariantConfig,
    family: str,
    scenarios: Sequence[ScenarioInputs],
    taus: Sequence[float],
    k_max: int = 1,
    use_agent_runtime: bool = False,
) -> list[ThresholdSweepRow]:
    """Sweep `taus` on one (variant, family) pair.

    Fault is the only family covered by local silver labels; path and
    topology return a single sentinel row with abstain_rate=1.0 and a
    `score_col=='no_silver'` marker so the report stays per-family
    uniform.
    """
    if family != "fault":
        return [ThresholdSweepRow(
            variant=variant.name, family=family,
            score_col="no_silver", tau=0.0, k_max=k_max,
            n_scenarios=0, n_abstained=0,
        )]

    score_col = _primary_score_col(variant)
    fault_scenarios = _filter_fault_scenarios(scenarios)
    rows: list[ThresholdSweepRow] = []
    for tau in taus:
        row = ThresholdSweepRow(
            variant=variant.name, family=family,
            score_col=score_col, tau=float(tau), k_max=k_max,
            n_scenarios=len(fault_scenarios),
        )
        for s in fault_scenarios:
            trace = run_variant_on_scenario(
                variant=variant, inputs=s, tau=float(tau), k_max=k_max,
                use_agent_runtime=use_agent_runtime,
            )
            if trace.answer.strip():
                row.n_emitted += 1
            else:
                row.n_abstained += 1
            exact, top1, tp, fp, fn = _score_answer(
                trace.answer, s.silver_positives, s.routing_vocab, s.port_vocab,
            )
            row.n_exact += int(exact)
            row.n_top1 += int(top1)
            row.sum_tp += tp
            row.sum_fp += fp
            row.sum_fn += fn
            row.sum_calls += trace.n_calls
            row.sum_format_errors += trace.n_format_errors
        rows.append(row)
    return rows


def auto_tau_grid(
    scenarios: Sequence[ScenarioInputs],
    score_col: str,
    n_points: int = 9,
) -> list[float]:
    """Return tau values spanning the observed score distribution.

    Uses the empirical quantiles of `score_col` over all candidate rows
    of every scenario, then adds a 0.0 floor (no abstention) so V0..V4's
    no-threshold baseline is always present in the sweep.
    """
    pool: list[float] = []
    for s in scenarios:
        cands = s.candidates_xgb if score_col == "calibrated_score" else s.candidates_deterministic
        for c in cands:
            try:
                pool.append(float(c.get(score_col, 0.0)))
            except (TypeError, ValueError):
                continue
    if not pool:
        return [0.0]
    pool.sort()
    n = len(pool)
    qs: list[float] = [0.0]
    for i in range(1, n_points + 1):
        idx = min(n - 1, int(round(i / (n_points + 1) * n)))
        qs.append(round(pool[idx], 4))
    # Deduplicate while preserving order; keep at most n_points+1 values.
    seen: set[float] = set()
    out: list[float] = []
    for q in qs:
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= n_points + 1:
            break
    return out


def best_row(rows: Sequence[ThresholdSweepRow], objective: str = "accuracy_exact") -> ThresholdSweepRow | None:
    """Return the row with the highest value for `objective`.

    Ties break toward lower abstain_rate (more answers given), then lower tau.
    """
    candidates = [r for r in rows if r.n_scenarios > 0]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: (
            getattr(r, objective),
            -r.abstain_rate,
            -r.tau,
        ),
    )


# ---- Variance across reruns ----------------------------------------------

def variance_across_reruns(
    variant: VariantConfig,
    scenarios: Sequence[ScenarioInputs],
    n_reruns: int = 3,
) -> dict:
    """Run the variant n_reruns times and report stability.

    With the deterministic stub all reruns produce identical answers, so
    `accuracy_std` is 0. When the real Qwen agent replaces the stub this
    function measures Pass@1 instability.
    """
    accs: list[float] = []
    for _ in range(n_reruns):
        [r] = run_eval([variant], scenarios)
        accs.append(r.accuracy_exact)
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    return {
        "variant": variant.name,
        "n_reruns": n_reruns,
        "mean_accuracy": round(mean, 4),
        "std_accuracy": round(var ** 0.5, 4),
        "per_rerun": [round(a, 4) for a in accs],
    }


# ---- Loaders --------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_scenarios(
    *,
    manifest_path: Path,
    ranked_path: Path,
    ranked_xgb_path: Path,
    anomaly_path: Path,
    graph_path: Path,
    silver_labels_path: Path,
    questions_phase1: Path,
    questions_phase2: Path,
    limits_path: Path,
) -> list[ScenarioInputs]:
    """Build ScenarioInputs from the on-disk pipeline artifacts."""
    questions: dict[str, dict] = {}
    for path, phase in ((questions_phase1, 1), (questions_phase2, 2)):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            questions[item["scenario_id"]] = {
                "phase": phase,
                "task_id": int(item["task"]["id"]),
                "text": item["task"]["question"],
            }

    ranked_rows = [r for r in _load_csv(ranked_path) if r.get("offline_bundle_missing") == "0" and r["node"]]
    ranked_xgb_rows = [r for r in _load_csv(ranked_xgb_path) if r.get("node")]
    anomaly_rows = [r for r in _load_csv(anomaly_path)
                    if r.get("offline_bundle_missing") == "0" and r["node"]]
    graph_rows = [r for r in _load_csv(graph_path)
                  if r.get("offline_bundle_missing") == "0" and r["node"]]
    silver_rows = _load_csv(silver_labels_path)

    by_sid_ranked: dict[str, list[dict]] = {}
    for r in ranked_rows:
        by_sid_ranked.setdefault(r["scenario_id"], []).append(r)
    by_sid_xgb: dict[str, list[dict]] = {}
    for r in ranked_xgb_rows:
        by_sid_xgb.setdefault(r["scenario_id"], []).append(r)
    anomaly_set: dict[str, set[tuple[str, str, str]]] = {}
    for r in anomaly_rows:
        anomaly_set.setdefault(r["scenario_id"], set()).add(
            (r["scenario_id"], r["node"], r["fault_reason"])
        )
    graph_features_by_sid: dict[str, dict[str, dict]] = {}
    for r in graph_rows:
        graph_features_by_sid.setdefault(r["scenario_id"], {})[r["node"]] = r
    silver_by_sid: dict[str, set[tuple[str, str]]] = {}
    for r in silver_rows:
        if int(r.get("relevance", 0)) >= 1:
            silver_by_sid.setdefault(r["scenario_id"], set()).add(
                (r["node"], r["fault_reason"])
            )

    out: list[ScenarioInputs] = []
    for sid, q in questions.items():
        if sid not in by_sid_ranked:
            continue
        rv, pv = extract_fault_vocab(q["text"])
        if not (rv and pv):
            continue
        out.append(ScenarioInputs(
            scenario_id=sid,
            question_number=q["task_id"],
            phase=q["phase"],
            question_text=q["text"],
            routing_vocab=rv,
            port_vocab=pv,
            candidates_deterministic=by_sid_ranked.get(sid, []),
            candidates_xgb=by_sid_xgb.get(sid, []),
            graph_features=graph_features_by_sid.get(sid, {}),
            anomaly_evidence=anomaly_set.get(sid, set()),
            denied_pairs=load_denied_pairs(q["task_id"], limits_path),
            silver_positives=silver_by_sid.get(sid, set()),
        ))
    return out
