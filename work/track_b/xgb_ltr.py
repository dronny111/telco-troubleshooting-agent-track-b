"""XGBoost LTR calibration layer (Step 8).

Reserves one disjoint `scenario_id` fold for calibration/promotion and
trains a GroupKFold ensemble on the remaining scenarios so the model
generalises across networks rather than memorising specific topologies.

Outputs at inference time per `(scenario_id, candidate)`:
    calibrated_score ∈ [0, 1]   — isotonic-regressed mean ensemble score
    uncertainty                 — std-dev across the training ensemble members

Promotion gate (Step 8 plan): XGBoost ships only if it beats deterministic-
only on held-out NDCG@5 AND on calls/correct in the local eval. The gate
function returns the comparison numbers for Step 11 to act on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBRanker


# ---- Hyperparameter sweep --------------------------------------------------

DEFAULT_SWEEP: tuple[dict, ...] = tuple(
    {"learning_rate": lr, "max_depth": d, "n_estimators": n}
    for lr in (0.05, 0.1)
    for d in (4, 6)
    for n in (200, 500)
)
FIXED_PARAMS = {
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "rank:pairwise",
    "tree_method": "hist",
    "verbosity": 0,
}


# ---- NDCG@K (no scipy dependency) -----------------------------------------

def _dcg(rel: np.ndarray) -> float:
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    gains = (np.power(2.0, rel) - 1.0)
    return float(np.sum(gains * discounts))


def ndcg_at_k(rel_true: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    """Compute NDCG@k for one query-group."""
    if rel_true.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ideal = np.sort(rel_true)[::-1]
    top_pred = rel_true[order][:k]
    top_ideal = ideal[:k]
    idcg = _dcg(top_ideal.astype(float))
    if idcg == 0:
        return 0.0
    return _dcg(top_pred.astype(float)) / idcg


def mean_ndcg(
    df: pd.DataFrame, score_col: str, label_col: str = "relevance",
    group_col: str = "scenario_id", k: int = 5,
) -> float:
    scores = df[score_col].to_numpy(dtype=float)
    labels = df[label_col].to_numpy(dtype=float)
    groups = df[group_col].to_numpy()
    out = []
    for sid in pd.unique(groups):
        mask = groups == sid
        out.append(ndcg_at_k(labels[mask], scores[mask], k=k))
    if not out:
        return 0.0
    return float(np.mean(out))


# ---- Trainer ---------------------------------------------------------------

@dataclass
class FoldArtifact:
    fold_index: int
    model: XGBRanker
    test_scenario_ids: list[str]
    val_ndcg5: float
    best_params: dict


@dataclass
class EnsembleModel:
    feature_cols: list[str]
    folds: list[FoldArtifact]
    isotonic: IsotonicRegression | None = None
    sweep_summary: list[dict] = field(default_factory=list)
    overall_val_ndcg5: float = 0.0
    reserved_scenario_ids: list[str] = field(default_factory=list)

    def predict_raw_per_fold(self, X: pd.DataFrame) -> np.ndarray:
        """Return shape=(n_folds, n_samples) of raw scores."""
        feats = X[self.feature_cols].to_numpy(dtype=float)
        out = np.stack([f.model.predict(feats) for f in self.folds], axis=0)
        return out

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (calibrated_score, uncertainty, raw_mean) for each row."""
        per_fold = self.predict_raw_per_fold(X)
        raw_mean = per_fold.mean(axis=0)
        unc = per_fold.std(axis=0, ddof=0)
        if self.isotonic is not None:
            calibrated = self.isotonic.predict(raw_mean)
            calibrated = np.clip(calibrated, 0.0, 1.0)
        else:
            calibrated = (raw_mean - raw_mean.min()) / max(
                1e-9, (raw_mean.max() - raw_mean.min())
            )
        return calibrated, unc, raw_mean


def _train_one_fold(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    sweep: Sequence[dict],
    sample_weight_col: str | None,
) -> tuple[XGBRanker, dict, float]:
    """Sweep hyperparams; return best model + chosen params + val NDCG@5.

    Note on sample weighting: xgboost's LTR objective treats `sample_weight`
    as one weight per query-group, not per row. Per-candidate confidence
    is therefore encoded in the relevance levels (0/1/2) themselves
    rather than via sample_weight. If `sample_weight_col` is provided, the
    per-row weights are aggregated to a per-group mean and passed as the
    group weight; this affects pairwise loss scaling at the scenario level.
    """
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["relevance"].to_numpy(dtype=int)
    train_groups = train_df.groupby("scenario_id", sort=False)
    g_train = train_groups.size().to_numpy()
    sw_train = None
    if sample_weight_col and sample_weight_col in train_df.columns:
        # One weight per group (mean of the group's per-row weights)
        sw_train = train_groups[sample_weight_col].mean().to_numpy(dtype=float)

    X_val = val_df[feature_cols].to_numpy(dtype=float)

    best_model = None
    best_params = {}
    best_ndcg = -1.0
    for params in sweep:
        model = XGBRanker(**FIXED_PARAMS, **params)
        fit_kwargs: dict = {"group": g_train}
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train
        model.fit(X_train, y_train, **fit_kwargs)
        val_df_local = val_df.copy()
        val_df_local["_pred"] = model.predict(X_val)
        ndcg = mean_ndcg(val_df_local, "_pred")
        if ndcg > best_ndcg:
            best_model = model
            best_params = dict(params)
            best_ndcg = ndcg
    return best_model, best_params, best_ndcg


def _expand_labels_to_feature_keys(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align labels to the feature candidate key.

    Features are keyed by `(scenario_id, node, fault_reason, category)`.
    Labels may omit `category` or leave it blank (synthetic oracle positives).
    In that case, expand the label to every matching feature-row category for
    the same `(scenario_id, node, fault_reason)` trio.
    """
    feature_key_cols = ["scenario_id", "node", "fault_reason", "category"]
    trio_cols = ["scenario_id", "node", "fault_reason"]

    dupes = features_df.duplicated(feature_key_cols)
    if dupes.any():
        examples = features_df.loc[dupes, feature_key_cols].head(5).to_dict("records")
        raise ValueError(f"features_df has duplicate candidate keys: {examples}")

    labels = labels_df.copy()
    if "category" not in labels.columns:
        labels["category"] = ""
    labels["category"] = labels["category"].fillna("").astype(str)

    exact = labels[labels["category"] != ""].copy()
    unresolved = labels[labels["category"] == ""].copy()

    if not unresolved.empty:
        category_map = features_df[feature_key_cols].drop_duplicates()
        unresolved = unresolved.drop(columns=["category"]).merge(
            category_map,
            on=trio_cols,
            how="inner",
        )
        labels = pd.concat([exact, unresolved], ignore_index=True)
    else:
        labels = exact

    if labels.empty:
        return labels

    agg_map = {"relevance": "max"}
    for col in labels.columns:
        if col in feature_key_cols or col == "relevance":
            continue
        agg_map[col] = "max" if col == "sample_weight" else "first"
    return labels.groupby(feature_key_cols, as_index=False).agg(agg_map)


def train_kfold(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    feature_cols: Iterable[str],
    n_splits: int = 5,
    sweep: Sequence[dict] = DEFAULT_SWEEP,
    sample_weight_col: str | None = "sample_weight",
    random_state: int = 0,
) -> EnsembleModel:
    """Train a reserved-fold-calibrated XGBRanker ensemble by scenario_id."""
    feature_cols = list(feature_cols)
    label_cols = ["scenario_id", "node", "fault_reason", "relevance"]
    if "category" in labels_df.columns:
        label_cols.append("category")
    if sample_weight_col:
        label_cols.append(sample_weight_col)
    labels_keyed = _expand_labels_to_feature_keys(features_df, labels_df[label_cols])

    df = features_df.merge(
        labels_keyed,
        on=["scenario_id", "node", "fault_reason", "category"],
        how="left",
    )
    df["relevance"] = df["relevance"].fillna(0).astype(int)
    if sample_weight_col:
        df[sample_weight_col] = df[sample_weight_col].fillna(0.5).astype(float)
    df = df.sort_values("scenario_id").reset_index(drop=True)

    groups = df["scenario_id"].to_numpy()
    unique_groups = pd.unique(groups)
    n_actual_splits = min(n_splits, max(2, len(unique_groups)))
    outer_gkf = GroupKFold(n_splits=n_actual_splits)
    outer_splits = list(outer_gkf.split(df, df["relevance"].to_numpy(), groups=groups))
    train_pool_idx, reserved_idx = outer_splits[-1]
    train_pool_df = df.iloc[train_pool_idx].sort_values("scenario_id").reset_index(drop=True)
    reserved_df = df.iloc[reserved_idx].reset_index(drop=True)
    reserved_scenario_ids = list(map(str, pd.unique(reserved_df["scenario_id"])))

    train_groups = train_pool_df["scenario_id"].to_numpy()
    train_unique_groups = pd.unique(train_groups)
    inner_splits = min(max(2, n_actual_splits - 1), len(train_unique_groups))
    inner_gkf = GroupKFold(n_splits=inner_splits)

    folds: list[FoldArtifact] = []
    sweep_summary: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        inner_gkf.split(
            train_pool_df,
            train_pool_df["relevance"].to_numpy(),
            groups=train_groups,
        )
    ):
        # Train data MUST be sorted by scenario_id for the `group=` arg.
        train_df = train_pool_df.iloc[train_idx].sort_values("scenario_id").reset_index(drop=True)
        val_df_orig = train_pool_df.iloc[val_idx].reset_index(drop=True)
        model, params, ndcg = _train_one_fold(
            train_df=train_df,
            val_df=val_df_orig,
            feature_cols=feature_cols,
            sweep=sweep,
            sample_weight_col=sample_weight_col,
        )
        folds.append(FoldArtifact(
            fold_index=fold_idx,
            model=model,
            test_scenario_ids=list(map(str, pd.unique(val_df_orig["scenario_id"]))),
            val_ndcg5=ndcg,
            best_params=params,
        ))
        sweep_summary.append({"fold": fold_idx, "best_params": params, "val_ndcg5": ndcg})

    reserved_feats = reserved_df[feature_cols].to_numpy(dtype=float)
    per_fold_reserved = np.stack(
        [f.model.predict(reserved_feats) for f in folds],
        axis=0,
    )
    reserved_raw_mean = per_fold_reserved.mean(axis=0)
    reserved_df = reserved_df.copy()
    reserved_df["_raw_mean"] = reserved_raw_mean
    overall_ndcg = mean_ndcg(reserved_df, "_raw_mean")

    # Isotonic on the reserved fold: relevance is 0/1/2; map to [0, 1].
    rel_norm = (
        reserved_df["relevance"] / max(reserved_df["relevance"].max(), 1)
    ).to_numpy(dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(reserved_raw_mean, rel_norm)

    return EnsembleModel(
        feature_cols=feature_cols,
        folds=folds,
        isotonic=iso,
        sweep_summary=sweep_summary,
        overall_val_ndcg5=overall_ndcg,
        reserved_scenario_ids=reserved_scenario_ids,
    )


# ---- Promotion gate -------------------------------------------------------

@dataclass
class PromotionGate:
    xgb_ndcg5: float
    deterministic_ndcg5: float
    delta: float
    beats_deterministic: bool


def evaluate_promotion(
    df: pd.DataFrame,
    *,
    xgb_score_col: str = "calibrated_score",
    deterministic_score_col: str = "combined_score",
    label_col: str = "relevance",
    k: int = 5,
) -> PromotionGate:
    xgb_ndcg = mean_ndcg(df, xgb_score_col, label_col=label_col, k=k)
    det_ndcg = mean_ndcg(df, deterministic_score_col, label_col=label_col, k=k)
    return PromotionGate(
        xgb_ndcg5=xgb_ndcg,
        deterministic_ndcg5=det_ndcg,
        delta=xgb_ndcg - det_ndcg,
        beats_deterministic=xgb_ndcg > det_ndcg,
    )


# ---- Persistence ----------------------------------------------------------

def save_summary(model: EnsembleModel, path: Path, gate: PromotionGate | None = None) -> None:
    summary = {
        "feature_cols": model.feature_cols,
        "n_folds": len(model.folds),
        "overall_val_ndcg5": model.overall_val_ndcg5,
        "reserved_scenario_ids": model.reserved_scenario_ids,
        "per_fold": model.sweep_summary,
    }
    if gate is not None:
        summary["promotion_gate"] = {
            "xgb_ndcg5": gate.xgb_ndcg5,
            "deterministic_ndcg5": gate.deterministic_ndcg5,
            "delta": gate.delta,
            "beats_deterministic": gate.beats_deterministic,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
