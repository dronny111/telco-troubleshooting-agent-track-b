"""Step 7 — Oracle silver-label generation.

Runs the agent (Qwen + the full pipeline) on every Phase 1 fault scenario
once with a primary configuration, then conditionally launches up to two
audit reruns when the primary answer is invalid or low-confidence. The
aggregated result emits a labelled candidate file Step 8 consumes verbatim:

    work/oracle_silver_labels.csv
        scenario_id, question_number, phase, node, fault_reason, category,
        relevance, sample_weight

Confidence policy
-----------------
- Primary-only path: if the primary run is accepted and yields at least one
  parsed fault line, the scenario is treated as high-confidence and keeps
  `sample_weight = 1.0`.
- Audit path: when audits run, a scenario is "high confidence" only if every
  emitted line is supported by the executed-run majority; otherwise the silver
  set is the union of accepted runs with `sample_weight = 0.5`.
- Scenarios where no run produced a valid answer are dropped entirely (the
  loss leaves them out).

Schema parity with feature_pipeline.synthetic_relevance() means the
trainer reads either label source via the same parquet/CSV path.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .agent_runtime import AgentLimits, AgentTrace, run_scenario
from .agent_tools import AgentToolConfig
from .answer_validator import _parse_lines as _parse_fault_lines
from .prompt_context import AnomalyEvidence, RankedRow
from .qwen_client import QwenClient


@dataclass(frozen=True)
class OracleConfig:
    """Primary + conditional-audit settings for the silver-label run."""
    base_seed: int = 42
    primary_temperature: float = 0.0
    audit_temperatures: tuple[float, ...] = (0.3, 0.6)
    max_audit_runs: int = 2
    max_tool_calls_primary: int = 350
    max_tool_calls_audit: int = 75
    max_iterations_primary: int = 12
    max_iterations_audit: int = 8
    audit_top_k: int = 5
    low_margin_gap: float = 0.25

    def __post_init__(self) -> None:
        if self.max_audit_runs < 0:
            raise ValueError("max_audit_runs must be non-negative")
        if self.max_audit_runs > len(self.audit_temperatures):
            raise ValueError("max_audit_runs exceeds configured audit temperatures")
        total_budget = self.max_tool_calls_primary + self.max_audit_runs * self.max_tool_calls_audit
        if total_budget > 500:
            raise ValueError("oracle config exceeds the 500-call per-scenario ceiling")

    def seed_for(self, i: int) -> int:
        return self.base_seed + i


@dataclass
class SeedResult:
    seed_index: int
    final_action: str
    final_answer: str
    answer_lines: set[tuple[str, str]] = field(default_factory=set)
    n_iterations: int = 0
    n_tool_calls: int = 0
    n_follow_ups: int = 0

    @property
    def accepted_with_lines(self) -> bool:
        return self.final_action == "accept" and bool(self.answer_lines)


@dataclass
class ScenarioOracleResult:
    scenario_id: str
    question_number: int
    phase: int
    seeds: list[SeedResult] = field(default_factory=list)
    silver_positives: set[tuple[str, str]] = field(default_factory=set)
    is_high_confidence: bool = False
    sample_weight: float = 0.5
    notes: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.silver_positives)


def _consensus_lines(seeds: list[SeedResult], min_agreement: int) -> set[tuple[str, str]]:
    """Return lines emitted by at least `min_agreement` seeds."""
    counter: Counter[tuple[str, str]] = Counter()
    for s in seeds:
        for line in s.answer_lines:
            counter[line] += 1
    return {line for line, n in counter.items() if n >= min_agreement}


def aggregate_consensus(
    seeds: list[SeedResult],
    *,
    n_seeds: int,
) -> tuple[set[tuple[str, str]], bool]:
    """Return (silver_positives, is_high_confidence).

    high_confidence requires ≥2/3 (or majority) agreement on every emitted
    line AND at least one accepted seed. With fewer seeds we adapt the
    threshold: ceil(n_seeds / 2).
    """
    accepted = [s for s in seeds if s.accepted_with_lines]
    if not accepted:
        return set(), False
    if n_seeds <= 1:
        union: set[tuple[str, str]] = set().union(*(s.answer_lines for s in accepted))
        return union, bool(union)
    threshold = max(2, (n_seeds + 1) // 2)
    consensus = _consensus_lines(accepted, threshold)
    union: set[tuple[str, str]] = set().union(*(s.answer_lines for s in accepted))
    # silver_positives = union of every accepted seed's lines; the
    # confidence flag tells the trainer whether to weight them at 1.0
    # or 0.5. Dropping a one-off here would lose recall — better to
    # keep it and let sample_weight encode uncertainty.
    high_conf = bool(consensus) and consensus == union
    return union, high_conf


def _run_one_seed(
    *,
    seed_index: int,
    scenario_id: str,
    question_number: int,
    phase: int,
    question_text: str,
    candidates: list[RankedRow],
    graph_features: dict[str, dict],
    anomaly_evidence: dict[tuple[str, str, str], AnomalyEvidence],
    qwen: QwenClient,
    tool_config: AgentToolConfig,
    limits_path: str,
    max_iterations: int,
    max_tool_calls: int,
    temperature: float,
    seed: int,
) -> SeedResult:
    trace: AgentTrace = run_scenario(
        scenario_id=scenario_id,
        question_number=question_number,
        phase=phase,
        question_text=question_text,
        candidates=candidates,
        graph_features=graph_features,
        anomaly_evidence=anomaly_evidence,
        qwen=qwen,
        tool_config=tool_config,
        limits_path=limits_path,
        limits=AgentLimits(
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
        ),
        temperature=temperature,
        seed=seed,
    )
    result = SeedResult(
        seed_index=seed_index,
        final_action=trace.final_action,
        final_answer=trace.final_answer,
        n_iterations=trace.iterations,
        n_tool_calls=trace.tool_calls_made,
        n_follow_ups=trace.follow_ups_triggered,
    )
    if trace.final_action == "accept" and trace.final_answer:
        from .vocab_extractor import extract_fault_vocab

        rv, pv = extract_fault_vocab(question_text)
        result.answer_lines = {
            (fl.node, fl.reason)
            for fl in _parse_fault_lines(trace.final_answer, frozenset(rv), frozenset(pv))
        }
    return result


def _needs_audit(
    primary: SeedResult,
    *,
    candidates: list[RankedRow],
    top_k: int,
    low_margin_gap: float,
) -> bool:
    if not primary.accepted_with_lines:
        return True
    if len(candidates) < 2:
        return False

    ranked = sorted(
        candidates,
        key=lambda c: (-float(c.combined_score), c.node, c.fault_reason, c.category),
    )
    top_pairs = {(c.node, c.fault_reason) for c in ranked[: max(1, top_k)]}
    if not any(pair in top_pairs for pair in primary.answer_lines):
        return True

    score_gap = float(ranked[0].combined_score) - float(ranked[1].combined_score)
    return score_gap <= low_margin_gap


def run_oracle_for_scenario(
    *,
    scenario_id: str,
    question_number: int,
    phase: int,
    question_text: str,
    candidates: list[RankedRow],
    graph_features: dict[str, dict],
    anomaly_evidence: dict[tuple[str, str, str], AnomalyEvidence],
    qwen: QwenClient,
    tool_config: AgentToolConfig,
    limits_path: str,
    config: OracleConfig = OracleConfig(),
) -> ScenarioOracleResult:
    """Run the agent primary + conditional audits for one scenario."""
    result = ScenarioOracleResult(
        scenario_id=scenario_id,
        question_number=question_number,
        phase=phase,
    )
    primary = _run_one_seed(
        seed_index=0,
        scenario_id=scenario_id,
        question_number=question_number,
        phase=phase,
        question_text=question_text,
        candidates=candidates,
        graph_features=graph_features,
        anomaly_evidence=anomaly_evidence,
        qwen=qwen,
        tool_config=tool_config,
        limits_path=limits_path,
        max_iterations=config.max_iterations_primary,
        max_tool_calls=config.max_tool_calls_primary,
        temperature=config.primary_temperature,
        seed=config.seed_for(0),
    )
    result.seeds.append(primary)

    if _needs_audit(
        primary,
        candidates=candidates,
        top_k=config.audit_top_k,
        low_margin_gap=config.low_margin_gap,
    ):
        for i, temperature in enumerate(config.audit_temperatures[: config.max_audit_runs], start=1):
            result.seeds.append(
                _run_one_seed(
                    seed_index=i,
                    scenario_id=scenario_id,
                    question_number=question_number,
                    phase=phase,
                    question_text=question_text,
                    candidates=candidates,
                    graph_features=graph_features,
                    anomaly_evidence=anomaly_evidence,
                    qwen=qwen,
                    tool_config=tool_config,
                    limits_path=limits_path,
                    max_iterations=config.max_iterations_audit,
                    max_tool_calls=config.max_tool_calls_audit,
                    temperature=temperature,
                    seed=config.seed_for(i),
                )
            )

    silver, high_conf = aggregate_consensus(result.seeds, n_seeds=len(result.seeds))
    result.silver_positives = silver
    result.is_high_confidence = high_conf
    result.sample_weight = 1.0 if high_conf else 0.5
    if not silver:
        result.notes = "no executed run produced a valid answer"
    elif len(result.seeds) == 1 and high_conf:
        result.notes = "primary accepted without audit"
    elif not high_conf:
        result.notes = "audit runs disagreed; downweighted"
    return result


# ---- Label-CSV emitter (matches feature_pipeline.synthetic_relevance) -----

def labels_for_candidates(
    *,
    candidate_rows: Iterable[dict],
    oracle_results: dict[str, ScenarioOracleResult],
) -> list[dict]:
    """Annotate candidate rows with oracle-derived (relevance, sample_weight).

    `candidate_rows` is the list-of-dicts read from `ranked_candidates.csv`
    (one per `(scenario_id, node, fault_reason)`). Output mirrors the
    columns `xgb_silver_labels.csv` carries so the trainer needs no path
    change.

    If a silver positive (from oracle consensus) is NOT present in the
    candidate pool for its scenario, this function adds a synthetic row
    so the trainer never silently loses positives. This matters when
    the oracle's answer lands on a device the deterministic ranker did
    not surface (the pool builder is not exhaustive across all 31
    reasons × all devices).
    """
    candidate_rows = list(candidate_rows)
    by_sid: dict[str, dict] = {}
    seen_keys: dict[str, set[tuple[str, str]]] = {}
    for r in candidate_rows:
        sid = r["scenario_id"]
        seen_keys.setdefault(sid, set()).add((r["node"], r["fault_reason"]))
        by_sid.setdefault(sid, r)

    out: list[dict] = []
    for r in candidate_rows:
        sid = r["scenario_id"]
        node = r["node"]
        reason = r["fault_reason"]
        oracle = oracle_results.get(sid)
        if oracle is None or not oracle.usable:
            relevance = 0
            sw = 0.0
        else:
            is_pos = (node, reason) in oracle.silver_positives
            relevance = 1 if is_pos else 0
            sw = oracle.sample_weight if is_pos else 0.5
        out.append({
            "scenario_id": sid,
            "question_number": r.get("question_number", ""),
            "phase": r.get("phase", ""),
            "node": node,
            "fault_reason": reason,
            "category": r.get("category", ""),
            "relevance": relevance,
            "sample_weight": sw,
        })

    # Add synthetic rows for silver positives missing from the candidate pool.
    for sid, oracle in oracle_results.items():
        if not oracle.usable:
            continue
        existing = seen_keys.get(sid, set())
        ref = by_sid.get(sid, {})
        for node, reason in oracle.silver_positives:
            if (node, reason) in existing:
                continue
            out.append({
                "scenario_id": sid,
                "question_number": ref.get("question_number", oracle.question_number),
                "phase": ref.get("phase", oracle.phase),
                "node": node,
                "fault_reason": reason,
                "category": "",  # unknown without playbook lookup; trainer ignores
                "relevance": 1,
                "sample_weight": oracle.sample_weight,
            })
    return out


def write_silver_labels_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "scenario_id", "question_number", "phase",
        "node", "fault_reason", "category",
        "relevance", "sample_weight",
    )
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_oracle_traces(
    results: Iterable[ScenarioOracleResult],
    path: Path,
) -> None:
    """Per-scenario trace dump for offline inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            payload = {
                "scenario_id": r.scenario_id,
                "question_number": r.question_number,
                "phase": r.phase,
                "is_high_confidence": r.is_high_confidence,
                "sample_weight": r.sample_weight,
                "silver_positives": sorted(list(r.silver_positives)),
                "notes": r.notes,
                "seeds": [
                    {
                        "seed_index": s.seed_index,
                        "final_action": s.final_action,
                        "final_answer": s.final_answer,
                        "answer_lines": sorted(list(s.answer_lines)),
                        "n_iterations": s.n_iterations,
                        "n_tool_calls": s.n_tool_calls,
                        "n_follow_ups": s.n_follow_ups,
                    }
                    for s in r.seeds
                ],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
