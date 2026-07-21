"""Smoke test for Step 8 (XGBoost LTR calibration layer).

Asserts:
    - ndcg_at_k returns expected values for ordered / reversed / random.
    - mean_ndcg averages over groups correctly.
    - train_kfold runs end-to-end on a small synthetic feature matrix and
      returns a 4-fold training ensemble plus an isotonic calibrator fitted
      on a reserved fold.
    - EnsembleModel.predict returns calibrated_score ∈ [0, 1] and
      non-negative uncertainty.
    - evaluate_promotion's delta sign tracks which score column wins.

The test is end-to-end on small in-memory data; it does NOT depend on the
full 6,728-row pipeline output. Runs in under 10 seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_b.xgb_ltr import (
    DEFAULT_SWEEP,
    FoldArtifact,
    _expand_labels_to_feature_keys,
    evaluate_promotion,
    mean_ndcg,
    ndcg_at_k,
    train_kfold,
)


def section(s: str) -> None:
    print(f"\n=== {s} ===")


def assert_eq(a, b, label: str) -> bool:
    ok = a == b
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: a={a!r} b={b!r}")
    return ok


def assert_close(a, b, eps, label: str) -> bool:
    ok = abs(a - b) <= eps
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: a={a!r} b={b!r} (eps={eps})")
    return ok


def main() -> int:
    failures = 0

    section("ndcg_at_k")
    rel = np.array([0, 0, 1, 0, 2])
    perfect_scores = np.array([0.0, 0.1, 0.5, 0.2, 1.0])     # picks 2,1 first
    failures += not assert_close(ndcg_at_k(rel, perfect_scores, k=5), 1.0, 1e-6, "perfect → 1.0")
    reversed_scores = -perfect_scores
    failures += not assert_eq(ndcg_at_k(rel, reversed_scores, k=5) < 1.0, True, "reversed < 1.0")
    failures += not assert_eq(ndcg_at_k(np.zeros(5), np.arange(5).astype(float), k=5), 0.0,
                              "no positives → 0.0")

    section("mean_ndcg over groups")
    df = pd.DataFrame({
        "scenario_id": ["a", "a", "a", "b", "b", "b"],
        "score":       [0.9,  0.5,  0.1,  0.1,  0.5,  0.9],
        "relevance":   [2,    1,    0,    2,    1,    0],
    })
    # group a: perfect order → 1.0
    # group b: reversed → < 1.0
    a_only = mean_ndcg(df[df["scenario_id"] == "a"], "score")
    b_only = mean_ndcg(df[df["scenario_id"] == "b"], "score")
    failures += not assert_close(a_only, 1.0, 1e-6, "group a perfect")
    failures += not assert_eq(b_only < 1.0, True, "group b imperfect")
    avg = mean_ndcg(df, "score")
    failures += not assert_close(avg, (a_only + b_only) / 2, 1e-6, "mean over groups")

    section("train_kfold end-to-end on small synthetic data")
    rng = np.random.default_rng(42)
    rows = []
    for sid in [f"s{i}" for i in range(20)]:
        for cid in range(8):
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            # synthetic relevance: positive if f1+f2 > 1
            rel = 2 if (f1 + f2) > 1.0 else (1 if (f1 + f2) > 0 else 0)
            rows.append({
                "scenario_id": sid,
                "node": f"node{cid}",
                "fault_reason": "OSPF configuration error" if cid % 2 == 0 else "synthetic_reason",
                "category": "routing" if cid % 2 == 0 else "port",
                "f1": f1,
                "f2": f2,
                "combined_score": f1 + 0.5 * f2,  # weak deterministic baseline
                "relevance": rel,
            })
    full_df = pd.DataFrame(rows)
    label_df = full_df[["scenario_id", "node", "fault_reason", "category", "relevance"]].copy()
    feat_df = full_df.drop(columns=["relevance"])

    model = train_kfold(
        feat_df, label_df,
        feature_cols=["f1", "f2", "combined_score"],
        n_splits=5,
        sweep=({"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100},),
        sample_weight_col=None,
    )
    failures += not assert_eq(len(model.folds), 4, "4 training folds returned")
    failures += not assert_eq(all(isinstance(f, FoldArtifact) for f in model.folds),
                              True, "fold artifacts have correct type")
    failures += not assert_eq(model.isotonic is not None, True, "isotonic calibrator fitted")
    failures += not assert_eq(len(model.reserved_scenario_ids) > 0, True, "reserved fold captured")
    reserved_set = set(model.reserved_scenario_ids)
    train_fold_ids = set().union(*(set(f.test_scenario_ids) for f in model.folds))
    failures += not assert_eq(reserved_set.isdisjoint(train_fold_ids), True,
                              "reserved scenarios excluded from training folds")

    section("predict returns calibrated_score in [0,1] and non-negative uncertainty")
    cal, unc, raw = model.predict(feat_df)
    failures += not assert_eq(cal.shape, (len(feat_df),), "calibrated shape")
    failures += not assert_eq(unc.shape, (len(feat_df),), "uncertainty shape")
    failures += not assert_eq(bool((cal >= 0).all() and (cal <= 1).all()), True,
                              "calibrated in [0, 1]")
    failures += not assert_eq(bool((unc >= 0).all()), True, "uncertainty non-negative")

    section("category-less labels expand to feature candidate keys")
    dup_feat = pd.DataFrame({
        "scenario_id": ["s-dup", "s-dup"],
        "node": ["Core_SW_01", "Core_SW_01"],
        "fault_reason": ["OSPF configuration error", "OSPF configuration error"],
        "category": ["routing", "port"],
    })
    sparse_labels = pd.DataFrame({
        "scenario_id": ["s-dup"],
        "node": ["Core_SW_01"],
        "fault_reason": ["OSPF configuration error"],
        "relevance": [1],
    })
    expanded = _expand_labels_to_feature_keys(dup_feat, sparse_labels)
    failures += not assert_eq(len(expanded), 2, "blank-category label expands to both categories")

    section("evaluate_promotion sign tracks the better score on reserved fold")
    eval_df = full_df[full_df["scenario_id"].isin(model.reserved_scenario_ids)].copy()
    eval_cal, _, _ = model.predict(eval_df.drop(columns=["relevance"]))
    eval_df["calibrated_score"] = eval_cal
    gate = evaluate_promotion(eval_df)
    print(f"  XGB NDCG@5={gate.xgb_ndcg5:.4f}  Det NDCG@5={gate.deterministic_ndcg5:.4f} "
           f"delta={gate.delta:+.4f}")
    failures += not assert_eq(
        gate.beats_deterministic == (gate.xgb_ndcg5 > gate.deterministic_ndcg5),
        True,
        "beats_deterministic flag matches sign",
    )

    section("summary")
    if failures == 0:
        print("  PASS")
        return 0
    print(f"  FAIL — {failures} assertion(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
