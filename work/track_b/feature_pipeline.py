"""Join Steps 4–6 artifacts into the XGBoost feature matrix.

One row per `(scenario_id, node, fault_reason)` candidate:
    - 8 deterministic-ranker component scores (Step 6, _raw + _norm).
    - Anomaly-miner aggregate signals (Step 4): n_signatures_fired,
      evidence_strength_num, has_anomaly (binary).
    - Graph features (Step 5): degree, betweenness, on_parsed_path,
      hop_distance_source/dest, is_blacklisted, is_disclosed_fault,
      n_l3_ifaces, has_loopback_ip, n_vrrp_groups, n_vrf,
      denied_command_count.
    - Constraint-parser flags (Step 2): blacklist/disclosed-category
      indicators per row already encoded in the ranker's
      contradiction_penalty / disclosed_match — no double-counting.

All features land in scenario-relative form (rank percentiles already in
the ranker `_norm` columns; node-name embedding-style features are
omitted by design per the plan).

Synthetic relevance labels are emitted for the Step 8 dry run; they get
overwritten by oracle silver labels when Step 7 lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RANKED = ROOT / "work" / "ranked_candidates.csv"
ANOMALY = ROOT / "work" / "anomaly_candidates.csv"
GRAPH = ROOT / "work" / "graph_features.csv"
P1 = ROOT / "telco_data" / "Track B" / "data" / "Phase_1" / "test.json"
P2 = ROOT / "telco_data" / "Track B" / "data" / "Phase_2" / "test.json"


# ---- Feature column lists --------------------------------------------------

RANKER_COMPONENT_COLS: tuple[str, ...] = (
    "graph_centrality_raw", "graph_centrality_norm",
    "path_relevance_raw", "path_relevance_norm",
    "protocol_match_raw", "protocol_match_norm",
    "vendor_compat_raw", "vendor_compat_norm",
    "anomaly_prior_raw", "anomaly_prior_norm",
    "permission_survivor_raw", "permission_survivor_norm",
    "disclosed_match_raw", "disclosed_match_norm",
    "contradiction_penalty_raw", "contradiction_penalty_norm",
    "combined_score",
)

ANOMALY_COLS: tuple[str, ...] = (
    "anom_has_evidence",          # 0/1
    "anom_strength_num",          # 0/1/2/3 (none/low/med/high)
    "anom_n_signatures",          # int
)

GRAPH_FEATURE_COLS: tuple[str, ...] = (
    "degree", "betweenness", "betweenness_norm",
    "n_l3_ifaces", "has_loopback_ip",
    "n_vrrp_groups", "n_vrf",
    "on_parsed_path",
    "hop_distance_source", "hop_distance_dest",
    "is_blacklisted", "is_disclosed_fault",
    "denied_command_count",
)

CANDIDATE_KEY_COLS = ("scenario_id", "question_number", "phase", "node", "fault_reason", "category")
GRAPH_JOIN_COLS = ("scenario_id", "question_number", "phase", "node")
KEY_COLS = CANDIDATE_KEY_COLS
FEATURE_COLS: tuple[str, ...] = RANKER_COMPONENT_COLS + ANOMALY_COLS + GRAPH_FEATURE_COLS


# ---- Loaders ---------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _assert_unique(df: pd.DataFrame, keys: tuple[str, ...], label: str) -> None:
    dupes = df.duplicated(list(keys))
    if dupes.any():
        examples = df.loc[dupes, list(keys)].head(5).to_dict("records")
        raise ValueError(f"{label} is not unique on {keys}: {examples}")


def _merge_without_growth(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: tuple[str, ...],
    how: str,
    label: str,
) -> pd.DataFrame:
    before = len(left)
    merged = left.merge(right, on=list(on), how=how)
    if len(merged) != before:
        raise ValueError(
            f"{label} changed row count from {before} to {len(merged)} on keys {on}"
        )
    return merged


def _load_questions() -> pd.DataFrame:
    """One row per scenario with task family info — used by the synthetic
    labeler to know which scenarios are fault tasks (path/topology have
    no fault-line schema)."""
    rows = []
    for path in (P1, P2):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            rows.append({
                "scenario_id": item["scenario_id"],
                "task_id": int(item["task"]["id"]),
                "question_text": item["task"]["question"],
            })
    return pd.DataFrame(rows)


def build_feature_matrix() -> pd.DataFrame:
    """Join the four upstream artifacts into the candidate-level feature DF."""
    ranked = _read(RANKED)
    ranked = ranked[ranked["offline_bundle_missing"].astype(int) == 0].copy()
    ranked = ranked[ranked["node"].astype(str) != ""]
    if ranked.empty:
        return pd.DataFrame(columns=list(KEY_COLS) + list(FEATURE_COLS))
    _assert_unique(ranked, CANDIDATE_KEY_COLS, "ranked_candidates")

    anomaly = _read(ANOMALY)
    anomaly = anomaly[anomaly["offline_bundle_missing"].astype(int) == 0].copy()
    anomaly = anomaly[anomaly["node"].astype(str) != ""]
    anomaly_keyed = (
        anomaly.assign(anom_has_evidence=1)
        .rename(columns={
            "evidence_strength_num": "anom_strength_num",
            "n_signatures_fired": "anom_n_signatures",
        })[
            [*CANDIDATE_KEY_COLS,
              "anom_has_evidence", "anom_strength_num", "anom_n_signatures"]
        ]
        .groupby(list(CANDIDATE_KEY_COLS), as_index=False)
        .agg({
            "anom_has_evidence": "max",
            "anom_strength_num": "max",
            "anom_n_signatures": "sum",
        })
    )
    _assert_unique(anomaly_keyed, CANDIDATE_KEY_COLS, "anomaly_candidates")

    graph = _read(GRAPH)
    graph = graph[graph["offline_bundle_missing"].astype(int) == 0].copy()
    graph = graph[graph["node"].astype(str) != ""]
    graph = graph[[*GRAPH_JOIN_COLS, *GRAPH_FEATURE_COLS]]
    _assert_unique(graph, GRAPH_JOIN_COLS, "graph_features")

    df = _merge_without_growth(
        ranked,
        anomaly_keyed,
        on=CANDIDATE_KEY_COLS,
        how="left",
        label="anomaly merge",
    )
    for col in ANOMALY_COLS:
        df[col] = df[col].fillna(0)
    df[ANOMALY_COLS[0]] = df[ANOMALY_COLS[0]].astype(int)
    df[ANOMALY_COLS[1]] = df[ANOMALY_COLS[1]].astype(int)
    df[ANOMALY_COLS[2]] = df[ANOMALY_COLS[2]].astype(int)

    df = _merge_without_growth(
        df,
        graph,
        on=GRAPH_JOIN_COLS,
        how="left",
        label="graph merge",
    )
    for col in GRAPH_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(-1).infer_objects(copy=False)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    keep = list(KEY_COLS) + feat_cols
    return df[keep].reset_index(drop=True)


# ---- Synthetic relevance labels (dry run) ---------------------------------

def synthetic_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """Emit weak labels for the Step 8 dry run.

    Strategy — these labels approximate what an oracle silver-label run
    would produce on Phase 1, without needing Qwen yet:

        relevance=2  if anomaly evidence is present (`anom_has_evidence=1`)
                     OR the node is in the parsed `fault_candidate_nodes`
                     OR the node is in `disclosed_fault_nodes`
                     (these are the candidates an oracle would most likely
                      land on first).

        relevance=1  if `on_parsed_path=1` and `combined_score` is in the
                     top 5% of the scenario.

        relevance=0  otherwise.

    The XGBoost trainer learns from these noisy weak labels; once Step 7
    produces real oracle silver labels they replace this stub directly.
    """
    out = df[list(KEY_COLS)].copy()
    out["relevance"] = 0
    out["sample_weight"] = 0.5  # weak labels carry low weight by default

    out.loc[df["anom_has_evidence"].astype(int) == 1, "relevance"] = 2
    out.loc[df["is_disclosed_fault"].astype(int) == 1, "relevance"] = 2

    # Within-scenario top-5%: rank by combined_score descending
    df = df.copy()
    df["_rank"] = df.groupby("scenario_id")["combined_score"].rank(method="dense", ascending=False)
    df["_n"] = df.groupby("scenario_id")["combined_score"].transform("count")
    cutoff = (df["_n"] * 0.05).clip(lower=1).round()
    on_top_path = (df["_rank"] <= cutoff) & (df["on_parsed_path"].astype(int) == 1)
    out.loc[on_top_path & (out["relevance"] == 0), "relevance"] = 1

    # Heavier sample weight on confident positives
    out.loc[out["relevance"] == 2, "sample_weight"] = 1.0
    out.loc[out["relevance"] == 1, "sample_weight"] = 0.75
    return out
