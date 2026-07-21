#!/usr/bin/env python3
"""Tune XGBoost parameters for the Track A option ranker."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_xgb_ranker import build_rows, drop_option_id_features, evaluate_rankings, load_json, make_matrix_with_text  # noqa: E402


def config_space(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    anchors = [
        {
            "name": "shortlist_baseline",
            "n_estimators": 80,
            "max_depth": 5,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "name": "shallow_baseline",
            "n_estimators": 140,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ]
    sampled = []
    for i in range(64):
        sampled.append(
            {
                "name": f"random_{i:02d}",
                "n_estimators": rng.choice([50, 70, 90, 120, 160, 220]),
                "max_depth": rng.choice([2, 3, 4, 5, 6]),
                "learning_rate": rng.choice([0.025, 0.035, 0.05, 0.07, 0.1]),
                "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
                "colsample_bytree": rng.choice([0.55, 0.7, 0.85, 1.0]),
                "min_child_weight": rng.choice([1.0, 2.0, 4.0, 8.0, 12.0]),
                "gamma": rng.choice([0.0, 0.05, 0.1, 0.25, 0.5]),
                "reg_alpha": rng.choice([0.0, 0.001, 0.01, 0.05, 0.1]),
                "reg_lambda": rng.choice([0.5, 1.0, 2.0, 4.0, 8.0]),
            }
        )
    return anchors + sampled


def model_from_config(config: dict[str, Any], scale_pos_weight: float, seed: int, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        min_child_weight=config["min_child_weight"],
        gamma=config["gamma"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )


def score_model(
    config: dict[str, Any],
    x_train,
    y_train: np.ndarray,
    x_valid,
    y_valid: np.ndarray,
    valid_df: pd.DataFrame,
    scale_pos_weight: float,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], XGBClassifier]:
    model = model_from_config(config, scale_pos_weight=scale_pos_weight, seed=seed, n_jobs=n_jobs)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    scored = valid_df.copy()
    scored["score"] = model.predict_proba(x_valid)[:, 1]
    ranking = evaluate_rankings(scored)
    metrics = {
        "name": config["name"],
        **{k: v for k, v in config.items() if k != "name"},
        "row_average_precision": float(average_precision_score(y_valid, scored["score"])),
        "row_roc_auc": float(roc_auc_score(y_valid, scored["score"])),
        **{k: v for k, v in ranking.items() if k != "detail"},
    }
    metrics["shortlist_score"] = (
        metrics["top5_full_recall"]
        + 0.50 * metrics["top8_full_recall"]
        + 0.25 * metrics["top3_full_recall"]
    )
    metrics["fallback_score"] = metrics["mean_iou_fixed_by_tag"] + 0.25 * metrics["top1_hit"]
    return metrics, model


def save_bundle(
    out_dir: Path,
    name: str,
    model: XGBClassifier,
    text_vec,
    dict_vec,
    metrics: dict[str, Any],
) -> None:
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    with (target / "ranker.pkl").open("wb") as fp:
        pickle.dump({"model": model, "text_vec": text_vec, "dict_vec": dict_vec, "metrics": metrics}, fp)
    with (target / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def tune(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_json(Path(args.train_json))
    scenario_ids = [str(s["scenario_id"]) for s in scenarios]
    train_ids, valid_ids = train_test_split(
        scenario_ids,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=[s.get("tag", "") for s in scenarios],
    )
    train_ids = set(train_ids)
    valid_ids = set(valid_ids)
    train_scenarios = [s for s in scenarios if str(s["scenario_id"]) in train_ids]
    valid_scenarios = [s for s in scenarios if str(s["scenario_id"]) in valid_ids]

    print("Building rows and feature matrices...")
    train_df = build_rows(train_scenarios)
    valid_df = build_rows(valid_scenarios)
    if args.drop_option_id_feature:
        train_df = drop_option_id_features(train_df)
        valid_df = drop_option_id_features(valid_df)
    x_train, x_valid, text_vec, dict_vec = make_matrix_with_text(
        train_df,
        valid_df,
        max_text_features=args.max_text_features,
        text_vectorizer=args.text_vectorizer,
        text_min_df=args.text_min_df,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
    )
    y_train = train_df["target"].astype(int).to_numpy()
    y_valid = valid_df["target"].astype(int).to_numpy()
    counts = Counter(y_train)
    scale_pos_weight = counts[0] / max(counts[1], 1)

    configs = config_space(args.seed)[: args.max_trials]
    results: list[dict[str, Any]] = []
    best_shortlist: tuple[float, dict[str, Any], XGBClassifier] | None = None
    best_fallback: tuple[float, dict[str, Any], XGBClassifier] | None = None

    for i, config in enumerate(configs, start=1):
        metrics, model = score_model(
            config,
            x_train,
            y_train,
            x_valid,
            y_valid,
            valid_df,
            scale_pos_weight=scale_pos_weight,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        metrics.update(
            {
                "trial": i,
                "train_scenarios": len(train_scenarios),
                "valid_scenarios": len(valid_scenarios),
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "positive_rate": float(y_train.mean()),
                "scale_pos_weight": float(scale_pos_weight),
                "max_text_features": args.max_text_features,
                "text_vectorizer": args.text_vectorizer,
                "text_min_df": args.text_min_df,
                "drop_option_id_feature": args.drop_option_id_feature,
                "bm25_k1": args.bm25_k1 if args.text_vectorizer == "bm25" else None,
                "bm25_b": args.bm25_b if args.text_vectorizer == "bm25" else None,
            }
        )
        results.append(metrics)
        pd.DataFrame(results).sort_values("shortlist_score", ascending=False).to_csv(
            out_dir / "tuning_results.csv", index=False
        )

        if best_shortlist is None or metrics["shortlist_score"] > best_shortlist[0]:
            best_shortlist = (metrics["shortlist_score"], metrics, model)
            save_bundle(out_dir, "best_shortlist", model, text_vec, dict_vec, metrics)
        if best_fallback is None or metrics["fallback_score"] > best_fallback[0]:
            best_fallback = (metrics["fallback_score"], metrics, model)
            save_bundle(out_dir, "best_fallback", model, text_vec, dict_vec, metrics)

        print(
            f"[{i:02d}/{len(configs):02d}] {metrics['name']} "
            f"top1={metrics['top1_hit']:.3f} "
            f"top3={metrics['top3_full_recall']:.3f} "
            f"top5={metrics['top5_full_recall']:.3f} "
            f"top8={metrics['top8_full_recall']:.3f} "
            f"iou={metrics['mean_iou_fixed_by_tag']:.3f}"
        )

    ranked = pd.DataFrame(results)
    ranked.to_csv(out_dir / "tuning_results.csv", index=False)
    summary = {
        "best_shortlist": best_shortlist[1] if best_shortlist else None,
        "best_fallback": best_fallback[1] if best_fallback else None,
    }
    with (out_dir / "tuning_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default="telco_data/Track A/data/Phase_1/train.json")
    parser.add_argument("--out-dir", default="work/track_a_ranker/tuning")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-features", type=int, default=8000)
    parser.add_argument("--text-vectorizer", choices=["tfidf", "bm25", "none"], default="tfidf")
    parser.add_argument("--text-min-df", type=int, default=2)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--drop-option-id-feature", action="store_true")
    parser.add_argument("--max-trials", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    tune(args)


if __name__ == "__main__":
    main()
