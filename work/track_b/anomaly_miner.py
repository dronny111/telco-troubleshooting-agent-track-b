"""Offline config-anomaly miner for Track B.

Walks the static `devices_outputs/{question_number}/{device}/*.txt` bundles,
runs the playbook signatures across the relevant command outputs, and emits
candidate `(scenario_id, node, fault_reason, evidence)` rows. Output feeds:

    - Step 6 deterministic ranker — anomaly_prior signal per candidate.
    - Step 7 oracle silver-label generation — same candidate pool the agent
      considers.
    - Step 8 XGBoost feature matrix — binary detector flags + scalar
      evidence-strength columns (the per-candidate feature spine).

Phase scoping: Phase 1 has 50 local bundles; Phase 2 has none (different
network on the production server). The miner walks only scenarios whose
manifest entry has `has_static_bundle=True` and emits an explicit
`offline_bundle_missing=1` row per Phase 2 scenario so downstream consumers
see uniform candidate rows across phases.

Negative-polarity signatures (e.g., "global STP not enabled") fire when the
regex does NOT match in any of the target command outputs for that device.
Positive-polarity signatures fire when the regex matches at least once.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .playbook import (
    FaultEntry,
    Signature,
    all_entries,
    compile_signatures,
)


# Filename mapping mirrors server.py:
#   safe_filename = command.replace("/", "_").replace("\\", "_")
#                          .replace("..", "").replace(" ", "_") + ".txt"
# Example: "display ip routing-table" -> "display_ip_routing-table.txt"
def command_to_filename(command: str) -> str:
    safe = command.replace("/", "_").replace("\\", "_").replace("..", "").replace(" ", "_")
    return safe + ".txt"


_STRENGTH_TO_NUM = {"high": 3, "medium": 2, "low": 1}


@dataclass
class MatchedSignature:
    name: str
    polarity: str
    n_matches: int
    sample_line: str = ""


@dataclass
class Candidate:
    scenario_id: str
    question_number: int
    phase: int
    node: str
    fault_reason: str
    category: str
    evidence_strength: str
    evidence_strength_num: int
    n_signatures_fired: int
    signatures_fired: tuple[MatchedSignature, ...]
    sample_evidence: str
    offline_bundle_missing: bool


def _read_command_output(device_dir: Path, command: str) -> str | None:
    f = device_dir / command_to_filename(command)
    if not f.is_file():
        return None
    try:
        return f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


_DEVICE_FILTER_CACHE: dict[str, re.Pattern[str]] = {}


def _device_filter_pattern(regex: str) -> re.Pattern[str] | None:
    if not regex:
        return None
    if regex not in _DEVICE_FILTER_CACHE:
        _DEVICE_FILTER_CACHE[regex] = re.compile(regex, re.MULTILINE)
    return _DEVICE_FILTER_CACHE[regex]


def _device_passes_filter(device_dir: Path, regex: str) -> bool:
    """Confirm a device matches the signature's device_filter_regex.

    Filter is checked against display_current-configuration. If that file
    is absent the device cannot be confirmed and the signature is skipped
    rather than fired (precision-leaning).
    """
    pat = _device_filter_pattern(regex)
    if pat is None:
        return True
    content = _read_command_output(device_dir, "display current-configuration")
    if content is None:
        return False
    return pat.search(content) is not None


def _evaluate_signature(
    sig: Signature,
    device_dir: Path,
    pattern: re.Pattern[str],
) -> MatchedSignature | None:
    """Return MatchedSignature if the signature fires per its polarity, else None."""
    if sig.evidence_type != "fault":
        return None  # feature_presence signatures are prompt-only, not candidates
    if sig.device_filter_regex and not _device_passes_filter(device_dir, sig.device_filter_regex):
        return None
    targets = sig.target_commands or ()
    if sig.polarity == "negative":
        if not targets:
            return None
        any_target_present = False
        any_match = False
        for cmd in targets:
            content = _read_command_output(device_dir, cmd)
            if content is None:
                continue
            any_target_present = True
            if pattern.search(content):
                any_match = True
                break
        if not any_target_present:
            return None
        if any_match:
            return None
        return MatchedSignature(name=sig.name, polarity="negative", n_matches=0)
    # Positive polarity
    contents: list[tuple[str, str]] = []
    if targets:
        for cmd in targets:
            content = _read_command_output(device_dir, cmd)
            if content is not None:
                contents.append((cmd, content))
    else:
        for txt in device_dir.iterdir():
            if txt.suffix != ".txt" or not txt.is_file():
                continue
            try:
                contents.append((txt.stem, txt.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    n_matches = 0
    sample = ""
    for _cmd, content in contents:
        for m in pattern.finditer(content):
            n_matches += 1
            if not sample:
                line_start = content.rfind("\n", 0, m.start()) + 1
                line_end = content.find("\n", m.end())
                if line_end == -1:
                    line_end = len(content)
                sample = content[line_start:line_end].strip()[:200]
        if n_matches:
            break
    if n_matches == 0:
        return None
    return MatchedSignature(name=sig.name, polarity="positive", n_matches=n_matches, sample_line=sample)


def mine_device(
    *,
    scenario_id: str,
    question_number: int,
    phase: int,
    node: str,
    device_dir: Path,
    entries: Iterable[FaultEntry] = all_entries(),
    compiled: dict[tuple[str, str, str], re.Pattern[str]] | None = None,
) -> list[Candidate]:
    """Run all playbook signatures against one device's command outputs.

    Returns one Candidate per (fault_reason, category) pair where at least
    one signature fired.
    """
    compiled = compiled or compile_signatures()
    out: list[Candidate] = []
    for entry in entries:
        fired: list[MatchedSignature] = []
        sample = ""
        for sig in entry.signatures:
            pat = compiled.get((entry.fault_reason, entry.category, sig.name))
            if pat is None:
                continue
            res = _evaluate_signature(sig, device_dir, pat)
            if res is not None:
                fired.append(res)
                if not sample and res.sample_line:
                    sample = res.sample_line
        if not fired:
            continue
        out.append(
            Candidate(
                scenario_id=scenario_id,
                question_number=question_number,
                phase=phase,
                node=node,
                fault_reason=entry.fault_reason,
                category=entry.category,
                evidence_strength=entry.evidence_strength,
                evidence_strength_num=_STRENGTH_TO_NUM[entry.evidence_strength],
                n_signatures_fired=len(fired),
                signatures_fired=tuple(fired),
                sample_evidence=sample,
                offline_bundle_missing=False,
            )
        )
    return out


def mine_scenario(
    *,
    scenario_id: str,
    question_number: int,
    phase: int,
    devices_outputs_root: Path,
) -> list[Candidate]:
    """Mine every device folder under devices_outputs/{question_number}/."""
    bundle = devices_outputs_root / str(question_number)
    if not bundle.is_dir():
        return []
    compiled = compile_signatures()
    cands: list[Candidate] = []
    for device_dir in sorted(bundle.iterdir()):
        if not device_dir.is_dir():
            continue
        cands.extend(
            mine_device(
                scenario_id=scenario_id,
                question_number=question_number,
                phase=phase,
                node=device_dir.name,
                device_dir=device_dir,
                compiled=compiled,
            )
        )
    return cands


def emit_missing_bundle_row(
    *, scenario_id: str, question_number: int, phase: int
) -> Candidate:
    """Sentinel candidate row for scenarios without a local static bundle.

    Step 8 consumes a uniform schema across phases; this row carries the
    `offline_bundle_missing=1` flag so the XGBoost feature pipeline never
    silently treats Phase 2 candidates as "no anomaly" — they are
    "no offline data".
    """
    return Candidate(
        scenario_id=scenario_id,
        question_number=question_number,
        phase=phase,
        node="",
        fault_reason="",
        category="",
        evidence_strength="",
        evidence_strength_num=0,
        n_signatures_fired=0,
        signatures_fired=(),
        sample_evidence="",
        offline_bundle_missing=True,
    )


_CANDIDATE_FIELDS = (
    "scenario_id",
    "question_number",
    "phase",
    "node",
    "fault_reason",
    "category",
    "evidence_strength",
    "evidence_strength_num",
    "n_signatures_fired",
    "signatures_fired",
    "sample_evidence",
    "offline_bundle_missing",
)


def candidate_to_row(c: Candidate) -> dict:
    sigs = ";".join(
        f"{s.name}({s.polarity},{s.n_matches})" for s in c.signatures_fired
    )
    return {
        "scenario_id": c.scenario_id,
        "question_number": c.question_number,
        "phase": c.phase,
        "node": c.node,
        "fault_reason": c.fault_reason,
        "category": c.category,
        "evidence_strength": c.evidence_strength,
        "evidence_strength_num": c.evidence_strength_num,
        "n_signatures_fired": c.n_signatures_fired,
        "signatures_fired": sigs,
        "sample_evidence": c.sample_evidence,
        "offline_bundle_missing": int(c.offline_bundle_missing),
    }


def write_csv(candidates: Iterable[Candidate], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CANDIDATE_FIELDS)
        w.writeheader()
        for c in candidates:
            w.writerow(candidate_to_row(c))
