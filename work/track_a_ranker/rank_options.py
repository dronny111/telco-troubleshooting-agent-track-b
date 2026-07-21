#!/usr/bin/env python3
"""Rank Track A candidate options with a trained sidecar model."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_xgb_ranker import build_rows, drop_option_id_features  # noqa: E402


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def score_rows(bundle: dict[str, Any], rows: pd.DataFrame):
    if bundle.get("metrics", {}).get("drop_option_id_feature"):
        rows = drop_option_id_features(rows)
    x_dict = bundle["dict_vec"].transform(rows["features"])
    text_vec = bundle.get("text_vec")
    if text_vec is None:
        x = x_dict
    else:
        x_text = text_vec.transform(rows["text"].astype(str))
        x = sparse.hstack([x_text, x_dict], format="csr")
    return bundle["model"].predict_proba(x)[:, 1]


def proposed_answer(group: pd.DataFrame, top_n_multi: int = 4) -> str:
    ranked = group.sort_values("score", ascending=False)
    if ranked["tag"].iloc[0] == "single-answer":
        return str(ranked["option_id"].iloc[0])
    ids = ranked["option_id"].head(top_n_multi).tolist()
    return "|".join(ids)


def rank(args: argparse.Namespace) -> None:
    with Path(args.model).open("rb") as fp:
        bundle = pickle.load(fp)

    scenarios = load_json(Path(args.scenarios))
    if args.scenario_id:
        scenarios = [s for s in scenarios if str(s.get("scenario_id")) == args.scenario_id]
        if not scenarios:
            raise SystemExit(f"Scenario not found: {args.scenario_id}")

    rows = build_rows(scenarios, labeled=False)
    rows = rows.copy()
    rows["score"] = score_rows(bundle, rows)
    rows = rows.sort_values(["scenario_id", "score"], ascending=[True, False])

    records: list[dict[str, Any]] = []
    for sid, group in rows.groupby("scenario_id", sort=False):
        top = group.head(args.top_k)
        records.append(
            {
                "scenario_id": sid,
                "tag": group["tag"].iloc[0],
                "proposed_answer": proposed_answer(group, top_n_multi=args.top_n_multi),
                "top_options": [
                    {
                        "id": row.option_id,
                        "score": round(float(row.score), 6),
                        "label": row.option_label,
                    }
                    for row in top.itertuples(index=False)
                ],
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "jsonl":
        with out_path.open("w", encoding="utf-8") as fp:
            for rec in records:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        pd.DataFrame(
            {
                "scenario_id": rec["scenario_id"],
                "tag": rec["tag"],
                "proposed_answer": rec["proposed_answer"],
                "top_options": json.dumps(rec["top_options"], ensure_ascii=False),
            }
            for rec in records
        ).to_csv(out_path, index=False)

    if args.scenario_id:
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
    else:
        print(f"Wrote {len(records)} ranked scenarios to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="work/track_a_ranker/tuning_radio/best_fallback/ranker.pkl")
    parser.add_argument("--scenarios", default="telco_data/Track A/data/Phase_2/test.json")
    parser.add_argument("--scenario-id")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-n-multi", type=int, default=4)
    parser.add_argument("--out", default="work/track_a_ranker/phase2_rankings.jsonl")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    args = parser.parse_args()
    rank(args)


if __name__ == "__main__":
    main()
