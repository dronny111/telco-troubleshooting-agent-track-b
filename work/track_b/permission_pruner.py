"""Permission-aware hard pruner for Track B API calls.

Reads `question_limits_config.json` and exposes a fast denied-pair lookup
for the ranker. Denied (device, command) pairs MUST be filtered before
ranking — never penalty-ranked — because every denied call wastes a
billable API call against the Phase 2 tiebreaker.

Coverage: only Phase 1 questions 2, 4, 5, 6, 11, 12, 13, 14, 15, 19, 20,
22, 30, 31, 32, 33 have explicit denials. For all other Phase 1 questions
and all Phase 2 questions the lookup returns an empty set; the agent must
discover denials live (the production server returns HTTP 403 with an
"Error: No permission to perform the operation" body for those).

Phase 2 implication: at inference time the agent should attempt commands
opportunistically and treat 403 responses as a learned-denial signal that
applies for the rest of the scenario, never within-scenario re-attempting.
"""

from __future__ import annotations

import json
from pathlib import Path


_LIMITS_CACHE: dict[Path, dict] = {}


def _load(config_path: Path) -> dict:
    if config_path not in _LIMITS_CACHE:
        with open(config_path, "r", encoding="utf-8") as f:
            _LIMITS_CACHE[config_path] = json.load(f)
    return _LIMITS_CACHE[config_path]


def denied_pairs(question_number: int, config_path: Path) -> set[tuple[str, str]]:
    """Return the set of denied (device, command) tuples for one question.

    Empty set if the question has no entry in the config (most do not).
    """
    cfg = _load(config_path)
    entry = cfg.get(f"question_{question_number}", {})
    no_perm = entry.get("no_permission", {})
    out: set[tuple[str, str]] = set()
    for command, devices in no_perm.items():
        for device in devices:
            out.add((device, command))
    return out


def is_denied(
    *,
    question_number: int,
    device: str,
    command: str,
    config_path: Path,
) -> bool:
    """O(1) per-call check used by the deterministic ranker before scoring."""
    cfg = _load(config_path)
    entry = cfg.get(f"question_{question_number}", {})
    devs = entry.get("no_permission", {}).get(command)
    if not devs:
        return False
    return device in devs


def denied_command_count(question_number: int, config_path: Path) -> int:
    """Total number of denied (device, command) pairs for the scenario.

    Used as an XGBoost feature column at the scenario level (Step 8).
    """
    return len(denied_pairs(question_number, config_path))
