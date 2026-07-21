"""Local evaluation scorer for ITU Telco Phase 2 (both tracks).

Mirrors the official competition metric to the extent it has been documented:

Track A (5G wireless multi-choice):
    Per-scenario IoU = |pred ∩ gt| / |pred ∪ gt|.
    Overall score = mean over scenarios.
    Time discount (if elapsed minutes supplied):
        elapsed < 5 → 1.00
        5 ≤ e < 10  → 0.80
        10 ≤ e < 15 → 0.60
        e ≥ 15      → 0.00

Track B (network fault / path / topology):
    Per-scenario answer is a multi-line string. Each non-empty line is one
    record. Order of lines does NOT matter (organiser confirmation, 11 May).
    Per-line matching after normalisation. Three task families:

      fault    "node;ip-or-port;reason"          (semicolon-separated, 3 fields)
      topology "local(local_port)->remote(remote_port)" (one link per line)
      path     "node[_port]->node[_port]->...->dest"    (no trailing port on dst)

    Scoring per scenario:
        precision = correct_lines / total_pred_lines
        recall    = correct_lines / total_gt_lines
        f1        = 2pr/(p+r)
        em        = 1 iff sets are exactly equal else 0

    The official metric on Track B has not been publicly disclosed beyond
    "format-error zeros the question" and "all independently faulty nodes
    must be listed". This scorer reports BOTH set-EM and F1 so either
    interpretation can be read off.

    Format-zero rule: any line that does not parse into one of the three
    family schemas causes the scenario to score 0 (regardless of any
    correct lines in the same answer).

Normalisation:
    * Strip surrounding whitespace and trailing punctuation.
    * Collapse internal whitespace to single spaces.
    * Lowercase the `reason` field for fault; keep node and IP case-sensitive
      because device names are case-sensitive in the simulator.
    * Interface aliases: GE↔GigabitEthernet, TE↔TenGigabitEthernet, etc.
      Bandwidth/rate suffixes after the interface name (e.g. "GE1/0/1(GE)" or
      trailing "1G") are stripped, per the official rule on topology output.
    * Sort path/topology endpoints? — no, paths and links are directional.

Modes:
    audit    : Format-compliance audit only (no GT required). Reports the
               family of every answer and flags any that fail parsing or
               contain disallowed characters.
    score    : Score a submission against a ground-truth CSV.
    self     : Score a candidate submission against an ensemble/reference
               submission as pseudo-GT. Useful when official GT is not
               available — gives a relative quality signal.

Outputs:
    JSON report at the path given by --out (or printed to stdout).

Limitations:
    * Without official GT, this scorer cannot prove an absolute score. The
      "self" mode is calibrated: when run against itself the score must be
      1.0, and the audit mode catches the only deterministic failure mode
      (format errors that zero a question on the leaderboard).
    * Interface alias map and reason vocab are best-effort; if the official
      grader uses stricter string match, expected/actual divergence will
      appear in the per-line trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Track A — multi-choice IoU
# ---------------------------------------------------------------------------

_OPT = re.compile(r"^C\d+$", re.IGNORECASE)


def parse_track_a(raw: str) -> set[str]:
    """Parse a Track A answer cell into a set of canonical 'C<n>' tokens."""
    if not raw:
        return set()
    out: set[str] = set()
    for part in re.split(r"[|,\s]+", raw.strip()):
        part = part.strip().upper()
        if _OPT.match(part):
            out.add(f"C{int(part[1:])}")
    return out


def iou(pred: set, gt: set) -> float:
    if not pred and not gt:
        return 1.0
    union = pred | gt
    if not union:
        return 0.0
    return len(pred & gt) / len(union)


def time_discount(elapsed_min: float | None) -> float:
    if elapsed_min is None:
        return 1.0
    if elapsed_min < 5:
        return 1.0
    if elapsed_min < 10:
        return 0.8
    if elapsed_min < 15:
        return 0.6
    return 0.0


def score_track_a_one(pred_raw: str, gt_raw: str, elapsed_min: float | None = None) -> dict:
    pred = parse_track_a(pred_raw)
    gt = parse_track_a(gt_raw)
    raw = iou(pred, gt)
    disc = time_discount(elapsed_min)
    return {
        "pred": sorted(pred, key=lambda x: int(x[1:])),
        "gt": sorted(gt, key=lambda x: int(x[1:])),
        "iou": round(raw, 4),
        "discount": disc,
        "score": round(raw * disc, 4),
    }


# ---------------------------------------------------------------------------
# Track B — per-family parsing and scoring
# ---------------------------------------------------------------------------

# Interface-name canonicalisation. Map short → long form; downstream we
# compare canonical strings (case-preserving in numeric/slash portion).
_INTERFACE_ALIASES: dict[str, str] = {
    "ge": "gigabitethernet",
    "fe": "fastethernet",
    "te": "tengigabitethernet",
    "xge": "tengigabitethernet",
    "fge": "fortygige",
    "hge": "hundredgige",
    "eth": "ethernet",
    "gi": "gigabitethernet",
}

# Strip trailing bandwidth/rate annotations like "(1G)", " 1Gbps", "(GE)" after
# the interface identifier — official rule for topology answers.
_RATE_SUFFIX = re.compile(r"[\s(\[][^)\]]*[)\]]\s*$|\s+\d+\s*[GgMmKk](?:bps)?\s*$")

_INTERFACE_TOKEN = re.compile(r"^([A-Za-z]+)(\d.*)$")


def canon_interface(port: str) -> str:
    """Canonicalise an interface name: alias → long form, strip rate tag."""
    if not port:
        return ""
    p = port.strip()
    p = _RATE_SUFFIX.sub("", p).strip()
    m = _INTERFACE_TOKEN.match(p)
    if not m:
        return p.lower()
    head = m.group(1).lower()
    tail = m.group(2)
    head = _INTERFACE_ALIASES.get(head, head)
    return f"{head}{tail}".lower()


def canon_node(node: str) -> str:
    """Device names are case-sensitive in the simulator; just strip ws."""
    return node.strip()


def canon_ip_or_port(token: str) -> str:
    """Field 2 of a fault answer is either an IP or an interface name."""
    t = token.strip()
    # Heuristic: looks like an IPv4? leave as-is.
    if re.match(r"^\d+\.\d+\.\d+\.\d+(/\d+)?$", t):
        return t
    # Otherwise treat as interface name
    return canon_interface(t)


def canon_reason(reason: str) -> str:
    """Reason vocab is closed (19 routing + 12 port). Normalise whitespace
    and case so 'Layer 3 loop' == 'layer 3 loop' == 'layer  3 loop'."""
    return " ".join(reason.split()).lower()


# --- Family detection --------------------------------------------------------

_FAULT_LINE = re.compile(r"^[^;]+;[^;]+;[^;]+$")
_TOPOLOGY_LINE = re.compile(r"^[^()]+\([^()]+\)->[^()]+\([^()]+\)\s*$")
_PATH_LINE = re.compile(r"^[^;()]+->[^;]+$")  # falls through after topology


def detect_family(lines: list[str]) -> str:
    """Detect task family from the answer lines. Returns 'fault' | 'path' |
    'topology' | 'mixed' (if multiple) | 'unknown' (if nothing parses)."""
    families: set[str] = set()
    for line in lines:
        if not line:
            continue
        if _FAULT_LINE.match(line):
            families.add("fault")
        elif _TOPOLOGY_LINE.match(line):
            families.add("topology")
        elif "->" in line:
            families.add("path")
        else:
            families.add("unknown")
    if not families:
        return "unknown"
    if len(families) == 1:
        return next(iter(families))
    return "mixed"


# --- Per-family canonical-record extraction ---------------------------------

def parse_fault_line(line: str) -> tuple[str, str, str] | None:
    parts = [p.strip() for p in line.split(";")]
    if len(parts) != 3:
        return None
    node, ip_or_port, reason = parts
    return (canon_node(node), canon_ip_or_port(ip_or_port), canon_reason(reason))


def parse_topology_line(line: str) -> tuple[str, str, str, str] | None:
    m = re.match(r"^([^()]+)\(([^()]+)\)->([^()]+)\(([^()]+)\)\s*$", line)
    if not m:
        return None
    a_node, a_port, b_node, b_port = (s.strip() for s in m.groups())
    return (canon_node(a_node), canon_interface(a_port),
            canon_node(b_node), canon_interface(b_port))


def parse_path_line(line: str) -> tuple[str, ...] | None:
    if "->" not in line:
        return None
    hops = [h.strip() for h in line.split("->") if h.strip()]
    if len(hops) < 2:
        return None
    # Each hop may be "node_port" (underscore-separated) or "node(port)" or "node"
    canon: list[str] = []
    for hop in hops:
        m = re.match(r"^([^()]+)\(([^()]+)\)$", hop)
        if m:
            canon.append(f"{canon_node(m.group(1))}_{canon_interface(m.group(2))}")
            continue
        # Try split on last underscore for "node_GE1/0/1" pattern.
        # Only treat the trailing token as a port if it looks like an
        # interface (starts with letters then digits/slashes).
        idx = hop.rfind("_")
        if idx != -1:
            head, tail = hop[:idx], hop[idx + 1:]
            if _INTERFACE_TOKEN.match(tail):
                canon.append(f"{canon_node(head)}_{canon_interface(tail)}")
                continue
        canon.append(canon_node(hop))
    return tuple(canon)


# --- Top-level Track B per-scenario scorer ----------------------------------

@dataclass
class TrackBScenarioResult:
    scenario_id: str
    family_pred: str
    family_gt: str
    pred_lines: int
    gt_lines: int
    matched_lines: int
    precision: float
    recall: float
    f1: float
    exact_match: int  # 0 / 1
    format_error: bool
    bad_lines: list[str] = field(default_factory=list)


def _parse_lines(raw: str) -> list[str]:
    if not raw:
        return []
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return [ln.strip() for ln in raw.split("\n") if ln.strip()]


def _canon_records(lines: list[str], family: str) -> tuple[list, list[str]]:
    """Return (canonical records, bad-raw-lines) for a given family.
    A line that doesn't parse for the assumed family is a 'bad line'."""
    out: list = []
    bad: list[str] = []
    parser = {
        "fault": parse_fault_line,
        "topology": parse_topology_line,
        "path": parse_path_line,
    }.get(family)
    if not parser:
        return [], lines[:]
    for ln in lines:
        rec = parser(ln)
        if rec is None:
            bad.append(ln)
        else:
            out.append(rec)
    return out, bad


def score_track_b_one(pred_raw: str, gt_raw: str, scenario_id: str = "") -> TrackBScenarioResult:
    pred_lines = _parse_lines(pred_raw)
    gt_lines = _parse_lines(gt_raw)

    fam_pred = detect_family(pred_lines)
    fam_gt = detect_family(gt_lines)

    # Family must match (else 0 by the official "format zero" rule)
    if fam_gt == "unknown":
        # Without parseable GT, we can't score this scenario meaningfully
        return TrackBScenarioResult(
            scenario_id=scenario_id, family_pred=fam_pred, family_gt=fam_gt,
            pred_lines=len(pred_lines), gt_lines=len(gt_lines), matched_lines=0,
            precision=0.0, recall=0.0, f1=0.0, exact_match=0,
            format_error=True, bad_lines=pred_lines if fam_pred == "unknown" else [],
        )

    if fam_pred != fam_gt:
        return TrackBScenarioResult(
            scenario_id=scenario_id, family_pred=fam_pred, family_gt=fam_gt,
            pred_lines=len(pred_lines), gt_lines=len(gt_lines), matched_lines=0,
            precision=0.0, recall=0.0, f1=0.0, exact_match=0,
            format_error=True, bad_lines=pred_lines,
        )

    pred_recs, pred_bad = _canon_records(pred_lines, fam_gt)
    gt_recs, _ = _canon_records(gt_lines, fam_gt)

    if pred_bad:
        # Any malformed line in pred → format zero
        return TrackBScenarioResult(
            scenario_id=scenario_id, family_pred=fam_pred, family_gt=fam_gt,
            pred_lines=len(pred_lines), gt_lines=len(gt_lines), matched_lines=0,
            precision=0.0, recall=0.0, f1=0.0, exact_match=0,
            format_error=True, bad_lines=pred_bad,
        )

    pred_set = set(pred_recs)
    gt_set = set(gt_recs)
    matched = len(pred_set & gt_set)
    p = matched / len(pred_set) if pred_set else 0.0
    r = matched / len(gt_set) if gt_set else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    em = 1 if pred_set == gt_set else 0
    return TrackBScenarioResult(
        scenario_id=scenario_id, family_pred=fam_pred, family_gt=fam_gt,
        pred_lines=len(pred_lines), gt_lines=len(gt_lines), matched_lines=matched,
        precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
        exact_match=em, format_error=False, bad_lines=[],
    )


# ---------------------------------------------------------------------------
# Audit mode — format compliance without GT
# ---------------------------------------------------------------------------

def audit_track_a(rows: list[tuple[str, str]]) -> dict:
    counts = Counter()
    bad: list[tuple[str, str]] = []
    sizes = Counter()
    for sid, ans in rows:
        if not ans.strip():
            counts["blank"] += 1
            continue
        opts = parse_track_a(ans)
        if not opts:
            bad.append((sid, ans))
            counts["unparseable"] += 1
            continue
        counts["ok"] += 1
        sizes[len(opts)] += 1
    return {
        "total": len(rows),
        "counts": dict(counts),
        "size_distribution": dict(sorted(sizes.items())),
        "unparseable_examples": bad[:10],
    }


def audit_track_b(rows: list[tuple[str, str]]) -> dict:
    counts = Counter()
    fam_counts = Counter()
    bad: list[tuple[str, str, list[str]]] = []
    line_counts = Counter()
    for sid, ans in rows:
        if not ans.strip():
            counts["blank"] += 1
            continue
        lines = _parse_lines(ans)
        fam = detect_family(lines)
        fam_counts[fam] += 1
        line_counts[len(lines)] += 1
        if fam in ("unknown", "mixed"):
            counts["format_error"] += 1
            bad.append((sid, ans, lines))
            continue
        recs, bad_lines = _canon_records(lines, fam)
        if bad_lines:
            counts["format_error"] += 1
            bad.append((sid, ans, bad_lines))
        else:
            counts["ok"] += 1
    return {
        "total": len(rows),
        "counts": dict(counts),
        "family_distribution": dict(fam_counts),
        "line_count_distribution": dict(sorted(line_counts.items())),
        "format_error_examples": [
            {"id": sid, "answer": ans[:160], "bad_lines": bl[:3]}
            for sid, ans, bl in bad[:10]
        ],
    }


# ---------------------------------------------------------------------------
# Submission loading
# ---------------------------------------------------------------------------

def load_submission(path: Path) -> dict[str, dict[str, str]]:
    """Load a CSV in either the official template shape (ID,Track A,Track B)
    or the legacy 'scenario_id,prediction' / 'id,prediction' shapes.
    Returns mapping: id → {"track_a": ..., "track_b": ...}.
    """
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    if not rows:
        return {}
    cols = set(rows[0].keys())
    out: dict[str, dict[str, str]] = {}
    if {"ID", "Track A", "Track B"} <= cols:
        for r in rows:
            out[str(r["ID"])] = {"track_a": r.get("Track A", "") or "",
                                 "track_b": r.get("Track B", "") or ""}
    elif "scenario_id" in cols:
        ans_col = "prediction" if "prediction" in cols else (
            "answers" if "answers" in cols else None)
        if not ans_col:
            raise ValueError(f"unknown answer column in {path}: {cols}")
        for r in rows:
            out[str(r["scenario_id"])] = {"track_a": r.get(ans_col, "") or "",
                                          "track_b": ""}
    elif "id" in cols and "prediction" in cols:
        # Track B id→prediction; treat id as the row key
        for r in rows:
            out[str(r["id"])] = {"track_a": "",
                                 "track_b": r.get("prediction", "") or ""}
    else:
        raise ValueError(f"unknown CSV shape in {path}: {cols}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_audit(args: argparse.Namespace) -> int:
    sub = load_submission(args.submission)
    a_rows = [(sid, v["track_a"]) for sid, v in sub.items() if v["track_a"] or any(v.values())]
    # Only include rows that look like Track A scenarios: a row is Track A if
    # its Track A cell is non-blank; otherwise check if Track B is non-blank.
    a_rows = [(sid, v["track_a"]) for sid, v in sub.items()]
    b_rows = [(sid, v["track_b"]) for sid, v in sub.items()]
    # Filter to only the populated track per row
    a_pop = [(s, v) for s, v in a_rows if v.strip()]
    b_pop = [(s, v) for s, v in b_rows if v.strip()]
    report = {
        "submission": str(args.submission),
        "track_a": audit_track_a(a_pop),
        "track_b": audit_track_b(b_pop),
    }
    _emit(report, args.out)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    sub = load_submission(args.submission)
    gt = load_submission(args.ground_truth)
    return _score_and_emit(sub, gt, args, label_pred="submission", label_gt="ground_truth")


def cmd_self(args: argparse.Namespace) -> int:
    sub = load_submission(args.submission)
    ref = load_submission(args.reference)
    return _score_and_emit(sub, ref, args, label_pred="submission", label_gt="reference")


def _score_and_emit(sub: dict, gt: dict, args, *, label_pred: str, label_gt: str) -> int:
    a_results: list[dict] = []
    b_results: list[TrackBScenarioResult] = []
    common = sorted(set(sub) & set(gt))
    missing_in_sub = sorted(set(gt) - set(sub))
    missing_in_gt = sorted(set(sub) - set(gt))

    for sid in common:
        pa, ga = sub[sid].get("track_a", ""), gt[sid].get("track_a", "")
        pb, gb = sub[sid].get("track_b", ""), gt[sid].get("track_b", "")
        if pa or ga:
            r = score_track_a_one(pa, ga)
            r["id"] = sid
            a_results.append(r)
        if pb or gb:
            b_results.append(score_track_b_one(pb, gb, scenario_id=sid))

    # Aggregate
    a_summary = _summarise_track_a(a_results)
    b_summary = _summarise_track_b(b_results)

    report = {
        f"{label_pred}": str(args.submission),
        f"{label_gt}": str(getattr(args, "ground_truth", None) or getattr(args, "reference")),
        "common_scenarios": len(common),
        "missing_in_submission": len(missing_in_sub),
        "missing_in_ground_truth": len(missing_in_gt),
        "track_a": a_summary,
        "track_b": b_summary,
    }
    if args.detail:
        report["track_a_detail"] = a_results
        report["track_b_detail"] = [asdict(r) for r in b_results]
    _emit(report, args.out)
    return 0


def _summarise_track_a(results: list[dict]) -> dict:
    if not results:
        return {"n": 0}
    mean_iou = sum(r["iou"] for r in results) / len(results)
    mean_score = sum(r["score"] for r in results) / len(results)
    perfect = sum(1 for r in results if r["iou"] == 1.0)
    zero = sum(1 for r in results if r["iou"] == 0.0)
    return {
        "n": len(results),
        "mean_iou": round(mean_iou, 4),
        "mean_score_time_discounted": round(mean_score, 4),
        "perfect_iou_1.0": perfect,
        "zero_iou": zero,
    }


def _summarise_track_b(results: list[TrackBScenarioResult]) -> dict:
    if not results:
        return {"n": 0}
    n = len(results)
    em = sum(r.exact_match for r in results)
    fmt_err = sum(1 for r in results if r.format_error)
    mean_f1 = sum(r.f1 for r in results) / n
    mean_p = sum(r.precision for r in results) / n
    mean_r = sum(r.recall for r in results) / n
    fam = Counter()
    em_by_fam = Counter()
    n_by_fam = Counter()
    for r in results:
        fam[r.family_gt] += 1
        n_by_fam[r.family_gt] += 1
        if r.exact_match:
            em_by_fam[r.family_gt] += 1
    return {
        "n": n,
        "exact_match": em,
        "exact_match_rate": round(em / n, 4),
        "format_errors": fmt_err,
        "mean_f1": round(mean_f1, 4),
        "mean_precision": round(mean_p, 4),
        "mean_recall": round(mean_r, 4),
        "family_distribution_gt": dict(fam),
        "em_by_family": {f: f"{em_by_fam[f]}/{n_by_fam[f]}" for f in n_by_fam},
    }


def _emit(report: dict, out: Path | None) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"wrote {out}")
        # Also echo the high-level summary to stdout
        summary = {k: v for k, v in report.items() if k not in ("track_a_detail", "track_b_detail")}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(payload)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("audit", help="Format-compliance audit (no GT)")
    a.add_argument("--submission", type=Path, required=True)
    a.add_argument("--out", type=Path, default=None)
    a.set_defaults(func=cmd_audit, detail=False)

    s = sub.add_parser("score", help="Score against an official ground-truth CSV")
    s.add_argument("--submission", type=Path, required=True)
    s.add_argument("--ground-truth", type=Path, required=True)
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--detail", action="store_true")
    s.set_defaults(func=cmd_score)

    f = sub.add_parser("self", help="Score against a reference submission as pseudo-GT")
    f.add_argument("--submission", type=Path, required=True)
    f.add_argument("--reference", type=Path, required=True)
    f.add_argument("--out", type=Path, default=None)
    f.add_argument("--detail", action="store_true")
    f.set_defaults(func=cmd_self)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
