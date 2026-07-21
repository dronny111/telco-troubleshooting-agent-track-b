#!/usr/bin/env python3
"""Build a Phase 2 combined submission with strict template shape.

Track A policy:
    use the best measured public-LB Track A source (`past_subs/results_07.csv`).
    Local Track A validation did not transfer to public LB, so measured public
    signal beats offline radio-XGB confidence.

Track B policy:
    use the best measured public-LB Track B source (`past_subs/results_07.csv`)
    as base, then apply only high-confidence overrides grounded in question
    text or source agreement.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "telco_data/submission/Phase_2/submission_example.csv"
OUT = ROOT / "telco_data/submission/Phase_2/results.csv"
REPORT = ROOT / "telco_data/submission/Phase_2/results_validation.json"

TRACK_A_TEST = ROOT / "telco_data/Track A/data/Phase_2/test.json"
TRACK_A_LLM = ROOT / "work/ensemble_3way.csv"
TRACK_A_XGB = ROOT / "work/track_a_ranker/phase2_xgb_radio_fallback.csv"
KNOWN_BEST_PUBLIC = ROOT / "past_subs/results_07.csv"

TRACK_B_TEST = ROOT / "telco_data/Track B/data/Phase_2/test.json"
TRACK_B_SOURCES = {
    "public07": ROOT / "past_subs/results_07.csv",
    "vm": ROOT / "work/submission_vm_b/result.csv",
    "full": ROOT / "work/submission_full_b/result.csv",
    "q1fix": ROOT / "work/submission_live_v2_merged_q1fix.csv",
}
TRACK_B_FALLBACK_SOURCE = "public07"
TRACK_B_EXPLICIT_SOURCE_BY_ID = {
    # The question text explicitly says Vlanif120 VRRP dual-master and bans
    # Core_SW_02. The closest allowed routing reason is Layer 3 loop.
    1: "full",
}
TRACK_B_CONSENSUS_SOURCES = (
    "public07",
    "vm",
    "full",
    "q1fix",
)


ANSWER_TOKEN = re.compile(r"C(\d+)$")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def normalize_track_a(answer: object) -> str:
    raw = str(answer or "").strip()
    if not raw:
        return ""
    tokens: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[|,\s]+", raw):
        part = part.strip().upper()
        m = ANSWER_TOKEN.match(part)
        if not m:
            continue
        token = f"C{int(m.group(1))}"
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    tokens.sort(key=lambda t: int(t[1:]))
    return "|".join(tokens)


def normalize_track_b(prediction: object) -> str:
    raw = str(prediction or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    return "\n".join(lines)


def make_track_a_map() -> tuple[dict[str, str], dict[str, str]]:
    known = pd.read_csv(KNOWN_BEST_PUBLIC, keep_default_na=False) if KNOWN_BEST_PUBLIC.is_file() else None
    llm = pd.read_csv(TRACK_A_LLM, keep_default_na=False)
    xgb = pd.read_csv(TRACK_A_XGB, keep_default_na=False)
    known_map = {}
    if known is not None and "ID" in known.columns and "Track A" in known.columns:
        known_map = {
            str(row["ID"]): normalize_track_a(row["Track A"])
            for _, row in known.iterrows()
            if normalize_track_a(row["Track A"])
        }
    # Support both 'answers' (legacy) and 'prediction' (ensemble) column names
    llm_ans_col = "prediction" if "prediction" in llm.columns else "answers"
    llm_map = {
        str(row.scenario_id): normalize_track_a(getattr(row, llm_ans_col))
        for row in llm.itertuples(index=False)
        if normalize_track_a(getattr(row, llm_ans_col))
    }
    xgb_map = {
        str(row.scenario_id): normalize_track_a(row.xgb_answer)
        for row in xgb.itertuples(index=False)
        if normalize_track_a(row.xgb_answer)
    }
    out: dict[str, str] = {}
    source: dict[str, str] = {}
    for scenario in _load_json(TRACK_A_TEST):
        sid = str(scenario["scenario_id"])
        if known_map.get(sid):
            out[sid] = known_map[sid]
            source[sid] = "public07"
        elif xgb_map.get(sid):
            out[sid] = xgb_map[sid]
            source[sid] = "radio_xgb_fallback"
        elif llm_map.get(sid):
            out[sid] = llm_map[sid]
            source[sid] = "llm_fallback"
        else:
            out[sid] = ""
            source[sid] = "blank"
    return out, source


def _load_track_b_sources() -> dict[str, dict[int, str]]:
    loaded: dict[str, dict[int, str]] = {}
    for name, path in TRACK_B_SOURCES.items():
        source_map: dict[int, str] = {}
        if path.is_file():
            df = pd.read_csv(path, keep_default_na=False)
            if name == "public07" and "Track B" in df.columns:
                for _, row in df.iloc[500:].iterrows():
                    # Track B rows appear in Phase 2 test order after Track A.
                    pred = normalize_track_b(row["Track B"])
                    if pred:
                        source_map[len(source_map) + 1] = pred
            elif "id" in df.columns and "prediction" in df.columns:
                for row in df.itertuples(index=False):
                    try:
                        task_id = int(getattr(row, "id"))
                    except Exception:
                        continue
                    pred = normalize_track_b(getattr(row, "prediction"))
                    if pred:
                        source_map[task_id] = pred
        loaded[name] = source_map
    return loaded


def _consensus_prediction(
    task_id: int,
    source_maps: dict[str, dict[int, str]],
) -> tuple[str, str]:
    votes: dict[str, list[str]] = {}
    for name in TRACK_B_CONSENSUS_SOURCES:
        pred = source_maps.get(name, {}).get(task_id, "")
        if pred:
            votes.setdefault(pred, []).append(name)
    if not votes:
        return "", ""
    pred, names = max(votes.items(), key=lambda item: (len(item[1]), item[1][0] == TRACK_B_FALLBACK_SOURCE))
    if len(names) >= 2:
        return pred, "consensus:" + "+".join(names)
    return "", ""


def make_track_b_map() -> tuple[dict[str, str], dict[str, str]]:
    b_test = _load_json(TRACK_B_TEST)
    id_to_sid = {int(item["task"]["id"]): str(item["scenario_id"]) for item in b_test}
    out = {sid: "" for sid in id_to_sid.values()}
    source = {sid: "blank" for sid in id_to_sid.values()}
    source_maps = _load_track_b_sources()

    for task_id, sid in id_to_sid.items():
        explicit_source = TRACK_B_EXPLICIT_SOURCE_BY_ID.get(task_id)
        if explicit_source:
            pred = source_maps.get(explicit_source, {}).get(task_id, "")
            if pred:
                out[sid] = pred
                source[sid] = f"override:{explicit_source}"
                continue

        pred, src = _consensus_prediction(task_id, source_maps)
        if pred:
            out[sid] = pred
            source[sid] = src
            continue

        pred = source_maps.get(TRACK_B_FALLBACK_SOURCE, {}).get(task_id, "")
        if pred:
            out[sid] = pred
            source[sid] = f"fallback:{TRACK_B_FALLBACK_SOURCE}"
    return out, source


def main() -> int:
    template = pd.read_csv(TEMPLATE, keep_default_na=False)
    if list(template.columns) != ["ID", "Track A", "Track B"]:
        raise ValueError(f"Unexpected template columns: {list(template.columns)}")

    a_ids = [str(item["scenario_id"]) for item in _load_json(TRACK_A_TEST)]
    b_ids = [str(item["scenario_id"]) for item in _load_json(TRACK_B_TEST)]
    expected_ids = a_ids + b_ids
    if template["ID"].astype(str).tolist() != expected_ids:
        raise ValueError("Template ID order does not match Track A + Track B Phase 2 test order")

    a_map, a_source = make_track_a_map()
    b_map, b_source = make_track_b_map()

    out = template[["ID"]].copy()
    out["Track A"] = ""
    out["Track B"] = ""
    out.loc[: len(a_ids) - 1, "Track A"] = [a_map.get(sid, "") for sid in a_ids]
    out.loc[len(a_ids):, "Track B"] = [b_map.get(sid, "") for sid in b_ids]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if OUT.exists():
        backup = OUT.with_name(f"{OUT.stem}.bak_{timestamp}{OUT.suffix}")
        shutil.copy2(OUT, backup)
    out.to_csv(OUT, index=False)

    report = {
        "output": str(OUT.relative_to(ROOT)),
        "rows": int(len(out)),
        "columns": list(out.columns),
        "id_unique": int(out["ID"].nunique()),
        "track_a": {
            "expected": len(a_ids),
            "filled": int((out.loc[: len(a_ids) - 1, "Track A"].astype(str).str.strip() != "").sum()),
            "blank": int((out.loc[: len(a_ids) - 1, "Track A"].astype(str).str.strip() == "").sum()),
            "source_counts": pd.Series([a_source.get(sid, "blank") for sid in a_ids]).value_counts().to_dict(),
        },
        "track_b": {
            "expected": len(b_ids),
            "filled": int((out.loc[len(a_ids):, "Track B"].astype(str).str.strip() != "").sum()),
            "blank": int((out.loc[len(a_ids):, "Track B"].astype(str).str.strip() == "").sum()),
            "source_counts": pd.Series([b_source.get(sid, "blank") for sid in b_ids]).value_counts().to_dict(),
        },
        "track_a_blank_ids": [sid for sid in a_ids if not a_map.get(sid)],
        "track_b_blank_ids": [sid for sid in b_ids if not b_map.get(sid)],
    }
    with REPORT.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
