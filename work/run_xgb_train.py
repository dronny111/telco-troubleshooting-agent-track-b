"""Train the XGBoost LTR calibration layer end-to-end (Step 8 dry run).

Inputs (from Steps 4–6):
    work/ranked_candidates.csv
    work/anomaly_candidates.csv
    work/graph_features.csv

Outputs:
    work/xgb_features.parquet         — feature matrix for inspection
    work/xgb_silver_labels.parquet    — synthetic relevance + sample_weight
    work/xgb_summary.json             — per-fold sweep + overall NDCG@5 +
                                        promotion-gate result
    work/ranked_candidates_xgb.csv    — same schema as ranked_candidates plus
                                        calibrated_score, uncertainty (used
                                        by Step 9 prompt context and
                                        Step 10 validator)

Once Step 7's oracle silver labels are available, swap the synthetic
labels for the parquet emitted by the oracle pipeline; this driver's
contract does not change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_b.feature_pipeline import (
    FEATURE_COLS,
    KEY_COLS,
    build_feature_matrix,
    synthetic_relevance,
)
from track_b.xgb_ltr import (
    DEFAULT_SWEEP,
    evaluate_promotion,
    save_summary,
    train_kfold,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_FEATURES = ROOT / "work" / "xgb_features.parquet"
OUT_LABELS = ROOT / "work" / "xgb_silver_labels.parquet"
OUT_SUMMARY = ROOT / "work" / "xgb_summary.json"
OUT_RANKED = ROOT / "work" / "ranked_candidates_xgb.csv"


def _try_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # Fall back to CSV if no parquet engine is installed
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)
    print(f"  wrote {path}")


def main() -> int:
    print("==> building feature matrix")
    feats = build_feature_matrix()
    print(f"  rows: {len(feats)}, scenarios: {feats['scenario_id'].nunique()}")
    if feats.empty:
        print("  no candidate rows to train on; run earlier steps first")
        return 1
    _try_parquet(feats, OUT_FEATURES)

    print("==> generating synthetic silver labels (dry run)")
    labels = synthetic_relevance(feats)
    pos = (labels["relevance"] > 0).sum()
    print(f"  total rows: {len(labels)}; positives (rel>0): {pos}; "
          f"high-conf (rel=2): {(labels['relevance']==2).sum()}")
    _try_parquet(labels, OUT_LABELS)

    feature_cols = [c for c in FEATURE_COLS if c in feats.columns]
    print(f"==> training 5-fold XGBRanker ensemble across "
          f"{len(DEFAULT_SWEEP)} sweep points × n_folds")
    model = train_kfold(
        feats, labels,
        feature_cols=feature_cols,
        n_splits=5,
        sweep=DEFAULT_SWEEP,
        sample_weight_col="sample_weight",
    )
    print(f"  overall OOF NDCG@5: {model.overall_val_ndcg5:.4f}")
    for s in model.sweep_summary:
        print(f"  fold {s['fold']}  ndcg5={s['val_ndcg5']:.4f}  best={s['best_params']}")

    print("==> running inference on the full feature matrix")
    cal, unc, raw = model.predict(feats)
    feats_with_scores = feats.copy()
    feats_with_scores["calibrated_score"] = cal
    feats_with_scores["uncertainty"] = unc
    feats_with_scores["raw_xgb_score"] = raw

    print("==> evaluating promotion gate (XGBoost vs deterministic)")
    eval_df = feats_with_scores.merge(
        labels[["scenario_id", "node", "fault_reason", "relevance"]],
        on=["scenario_id", "node", "fault_reason"], how="left",
    )
    eval_df["relevance"] = eval_df["relevance"].fillna(0).astype(int)
    gate = evaluate_promotion(eval_df)
    print(f"  XGB NDCG@5         : {gate.xgb_ndcg5:.4f}")
    print(f"  Deterministic NDCG@5: {gate.deterministic_ndcg5:.4f}")
    print(f"  delta              : {gate.delta:+.4f}")
    print(f"  beats deterministic: {gate.beats_deterministic}")

    save_summary(model, OUT_SUMMARY, gate=gate)
    print(f"  wrote {OUT_SUMMARY}")

    print("==> writing ranked_candidates_xgb.csv (Step 9/10 input)")
    # Include the deterministic ranker's combined_score for downstream callers
    # that want to fall back when XGBoost is dropped from the submission.
    out_cols = list(KEY_COLS) + [
        "combined_score", "calibrated_score", "uncertainty",
    ]
    feats_with_scores[out_cols].to_csv(OUT_RANKED, index=False)
    print(f"  wrote {OUT_RANKED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
