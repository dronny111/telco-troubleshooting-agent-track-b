#!/usr/bin/env python3
"""Train an option-level Track A ranker.

The label C7 is only meaningful inside a scenario, so this script trains on
scenario-option rows:

    scenario features + option text/features -> option_is_correct
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from text_features import BM25Vectorizer


CELL_TOKEN = re.compile(r"\b(\d{6,7})_(\d+)\b")

ACTION_PATTERNS = {
    "neighbor": r"\bneighbor relationship\b|\bneighbor\b",
    "tx_power_inc": r"\bincrease transmission power\b",
    "tx_power_dec": r"\bdecrease transmission power\b",
    "tilt_down": r"\bpress down\b|\bdown tilt\b",
    "tilt_up": r"\blift the tilt\b|\blift\b",
    "azimuth": r"\bazimuth\b",
    "a3_offset_inc": r"\bincrease a3 offset\b",
    "a3_offset_dec": r"\bdecrease a3 offset\b",
    "coverage_threshold": r"\bcovinterfreq|a2rsrp|a5rsrp\b",
    "pdcch": r"\bpdcchoccupiedsymbolnum|pdcch\b",
    "server_transport": r"\bserver\b|\btransmission issues\b",
}

NUMERIC_KEYWORDS = (
    "throughput",
    "rsrp",
    "sinr",
    "pci",
    "arfcn",
    "bler",
    "mcs",
    "rank",
    "grant",
    "rb",
    "cce",
    "longitude",
    "latitude",
    "azimuth",
    "tilt",
    "power",
    "offset",
    "threshold",
    "pathloss",
    "gain",
    "distance",
    "angle",
)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def answer_set(answer: str) -> set[str]:
    return {part.strip() for part in str(answer).split("|") if part.strip().startswith("C")}


def clean_name(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return value[:max_len] or "empty"


def parse_pipe_table(text: Any) -> pd.DataFrame | None:
    if not isinstance(text, str) or "|" not in text or "\n" not in text:
        return None
    try:
        df = pd.read_csv(io.StringIO(text), sep="|", low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def to_num(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        if isinstance(value, str) and value.strip() in {"", "-", "nan", "None"}:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def cell_key(gnodeb: Any, cell: Any) -> str | None:
    g = to_num(gnodeb)
    c = to_num(cell)
    if np.isnan(g) or np.isnan(c):
        return None
    return f"{int(g)}_{int(c)}"


def extract_cell_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g, c in CELL_TOKEN.findall(str(text)):
        key = f"{g}_{c}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def add_prefixed_stats(out: dict[str, float], prefix: str, values: list[float]) -> None:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    out[f"{prefix}__n"] = float(arr.size)
    if arr.size == 0:
        return
    out[f"{prefix}__mean"] = float(arr.mean())
    out[f"{prefix}__min"] = float(arr.min())
    out[f"{prefix}__max"] = float(arr.max())
    out[f"{prefix}__std"] = float(arr.std(ddof=0))


def _mean_col(df: pd.DataFrame, col: str) -> float:
    s = numeric_series(df, col)
    return float(s.mean()) if s.notna().any() else np.nan


def _min_col(df: pd.DataFrame, col: str) -> float:
    s = numeric_series(df, col)
    return float(s.min()) if s.notna().any() else np.nan


def _max_col(df: pd.DataFrame, col: str) -> float:
    s = numeric_series(df, col)
    return float(s.max()) if s.notna().any() else np.nan


def _cell_feature(cell_features: dict[str, dict[str, float]], key: str, name: str) -> float:
    return float(cell_features.get(key, {}).get(name, np.nan))


def build_radio_context(scenario: dict[str, Any]) -> dict[str, Any]:
    """Build a lightweight typed radio graph context.

    Nodes are cells keyed as `gNodeBID_CellID`. PCI/ARFCN observations from
    drive-test and MR tables are joined back to config cells, which gives the
    option-level ranker features like "target cell is serving" or "target cell
    is top neighbor".
    """
    data = scenario.get("data", {})
    config = parse_pipe_table(data.get("network_configuration_data")) if isinstance(data, dict) else None
    user = parse_pipe_table(data.get("user_plane_data")) if isinstance(data, dict) else None
    mr = parse_pipe_table(data.get("mr_data")) if isinstance(data, dict) else None
    traffic = parse_pipe_table(data.get("traffic_data")) if isinstance(data, dict) else None

    cell_features: dict[str, dict[str, float]] = {}
    pci_arfcn_to_cells: dict[tuple[int, int], set[str]] = {}
    neighbors: dict[str, set[str]] = {}

    if config is not None:
        for _, row in config.iterrows():
            key = cell_key(row.get("gNodeB ID"), row.get("Cell ID"))
            if not key:
                continue
            f = cell_features.setdefault(key, {})
            f["in_config"] = 1.0
            for src, dst in (
                ("PCI", "config_pci"),
                ("ARFCN", "config_arfcn"),
                ("DL ARFCN", "config_dl_arfcn"),
                ("Transmission Power", "config_tx_power"),
                ("Max Transmit Power", "config_max_tx_power"),
                ("Mechanical Azimuth", "config_mech_azimuth"),
                ("Digital Azimuth", "config_digital_azimuth"),
                ("Mechanical Downtilt", "config_mech_downtilt"),
                ("Digital Tilt", "config_digital_tilt"),
                ("Height", "config_height"),
                ("CovInterFreqA2RsrpThld [dBm]", "config_a2_rsrp"),
                ("CovInterFreqA5RsrpThld1 [dBm]", "config_a5_rsrp1"),
                ("CovInterFreqA5RsrpThld2 [dBm]", "config_a5_rsrp2"),
                ("IntraFreqHoA3Offset [0.5dB]", "config_a3_offset"),
                ("IntraFreqHoA3Hyst [0.5dB]", "config_a3_hyst"),
            ):
                val = to_num(row.get(src))
                if not np.isnan(val):
                    f[dst] = val
            pdcch = str(row.get("PdcchOccupiedSymbolNum", ""))
            m = re.search(r"(\d+)", pdcch)
            if m:
                f["config_pdcch_symbols"] = float(m.group(1))
            pci = to_num(row.get("PCI"))
            arfcn = to_num(row.get("ARFCN"))
            if not np.isnan(pci) and not np.isnan(arfcn):
                pci_arfcn_to_cells.setdefault((int(pci), int(arfcn)), set()).add(key)
            raw_neighbors = str(row.get("PCell Neighbor Cell (gNodeBID_ARFCN_PCI)", ""))
            for g, arf, pci_s in re.findall(r"(\d{6,7})_(\d+)_(\d+)", raw_neighbors):
                for nbr_key in pci_arfcn_to_cells.get((int(pci_s), int(arf)), set()):
                    neighbors.setdefault(key, set()).add(nbr_key)
            # Also retain unresolved neighbor triples as weak count evidence.
            f["config_neighbor_decl_count"] = float(len(re.findall(r"\d{6,7}_\d+_\d+", raw_neighbors)))
        # Second pass after every PCI/ARFCN mapping exists. This avoids
        # order-dependent neighbor misses when the referenced cell appears
        # later in the configuration table.
        for _, row in config.iterrows():
            key = cell_key(row.get("gNodeB ID"), row.get("Cell ID"))
            if not key:
                continue
            raw_neighbors = str(row.get("PCell Neighbor Cell (gNodeBID_ARFCN_PCI)", ""))
            for _g, arf, pci_s in re.findall(r"(\d{6,7})_(\d+)_(\d+)", raw_neighbors):
                for nbr_key in pci_arfcn_to_cells.get((int(pci_s), int(arf)), set()):
                    neighbors.setdefault(key, set()).add(nbr_key)

    def add_observation(
        key: str,
        prefix: str,
        values: dict[str, float],
    ) -> None:
        f = cell_features.setdefault(key, {})
        f[f"{prefix}_seen"] = 1.0
        f[f"{prefix}_count"] = f.get(f"{prefix}_count", 0.0) + 1.0
        for name, val in values.items():
            if np.isnan(val):
                continue
            vals_key = f"__vals__{prefix}_{name}"
            f.setdefault(vals_key, []).append(float(val))

    if user is not None:
        total_rows = max(len(user), 1)
        for _, row in user.iterrows():
            serving_pci = to_num(row.get("5G KPI PCell RF Serving PCI"))
            serving_arfcn = to_num(row.get("5G KPI PCell RF Serving ARFCN"))
            serving_cells = pci_arfcn_to_cells.get((int(serving_pci), int(serving_arfcn)), set()) if not np.isnan(serving_pci) and not np.isnan(serving_arfcn) else set()
            for key in serving_cells:
                add_observation(key, "user_serving", {
                    "rsrp": to_num(row.get("5G KPI PCell RF Serving SS-RSRP [dBm]")),
                    "sinr": to_num(row.get("5G KPI PCell RF Serving SS-SINR [dB]")),
                    "throughput": to_num(row.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]")),
                    "cce_fail": to_num(row.get("CCE Fail Rate")),
                    "mcs": to_num(row.get("Avg MCS")),
                    "rb_num": to_num(row.get("5G KPI PCell Layer1 DL RB Num (Including 0)")),
                    "initial_bler": to_num(row.get("Initial BLER(%)")),
                    "residual_bler": to_num(row.get("Residual BLER(%)")),
                })
            for rank in range(1, 6):
                pci_col = f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {rank} PCI"
                arf_col = f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {rank} ARFCN"
                rsrp_col = f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {rank} Filtered Tx BRSRP [dBm]"
                pci = to_num(row.get(pci_col))
                arfcn = to_num(row.get(arf_col))
                for key in pci_arfcn_to_cells.get((int(pci), int(arfcn)), set()) if not np.isnan(pci) and not np.isnan(arfcn) else set():
                    add_observation(key, "user_neighbor", {
                        "rank": float(rank),
                        "rsrp": to_num(row.get(rsrp_col)),
                    })
        for f in cell_features.values():
            for prefix in ("user_serving", "user_neighbor"):
                if f.get(f"{prefix}_count"):
                    f[f"{prefix}_share"] = f[f"{prefix}_count"] / total_rows

    if mr is not None:
        total_rows = max(len(mr), 1)
        for _, row in mr.iterrows():
            pci = to_num(row.get("Serving PCI"))
            arfcn = to_num(row.get("Serving ARFCN"))
            for key in pci_arfcn_to_cells.get((int(pci), int(arfcn)), set()) if not np.isnan(pci) and not np.isnan(arfcn) else set():
                add_observation(key, "mr_serving", {
                    "rsrp": to_num(row.get("Serving RSRP(dBm)")),
                    "throughput": to_num(row.get("Throughput(Mbps)")),
                })
            for rank in range(1, 4):
                pci = to_num(row.get(f"Neighbor {rank} PCI"))
                arfcn = to_num(row.get(f"Neighbor {rank} ARFCN"))
                for key in pci_arfcn_to_cells.get((int(pci), int(arfcn)), set()) if not np.isnan(pci) and not np.isnan(arfcn) else set():
                    add_observation(key, "mr_neighbor", {
                        "rank": float(rank),
                        "rsrp": to_num(row.get(f"Neighbor {rank} RSRP(dBm)")),
                    })
        for f in cell_features.values():
            for prefix in ("mr_serving", "mr_neighbor"):
                if f.get(f"{prefix}_count"):
                    f[f"{prefix}_share"] = f[f"{prefix}_count"] / total_rows

    if traffic is not None:
        for key, group in traffic.groupby([
            traffic.get("gNodeB_ID", pd.Series(dtype=object)),
            traffic.get("Cell_ID", pd.Series(dtype=object)),
        ], dropna=True):
            ckey = cell_key(key[0], key[1])
            if not ckey:
                continue
            f = cell_features.setdefault(ckey, {})
            f["traffic_seen"] = 1.0
            for col, name in (
                ("Uplink PRB utilization(%)", "traffic_ul_prb"),
                ("Downlink PRB utilization(%)", "traffic_dl_prb"),
                ("Uplink PRB Interference(dBm)", "traffic_ul_interference"),
                ("User Uplink Throughput(Mbps)", "traffic_ul_tput"),
                ("User Downlink Throughput(Mbps)", "traffic_dl_tput"),
                ("Downlink Weak Coversge Ratio", "traffic_weak_coverage"),
                ("TA>1KM Ratio", "traffic_ta_gt_1km"),
                ("Uplink CCE utilization(%)", "traffic_ul_cce_util"),
                ("Downlink CCE utilization(%)", "traffic_dl_cce_util"),
                ("Uplink CCE Allocation Success Rate(%)", "traffic_ul_cce_success"),
                ("Downlink CCE Allocation Success Rate(%)", "traffic_dl_cce_success"),
            ):
                val = _mean_col(group, col)
                if not np.isnan(val):
                    f[name] = val

    # Collapse list-valued observation features.
    for f in cell_features.values():
        list_keys = [k for k in f if k.startswith("__vals__")]
        for key in list_keys:
            vals = f.pop(key)
            base = key.replace("__vals__", "")
            arr = np.array(vals, dtype=float)
            if arr.size:
                f[f"{base}_mean"] = float(arr.mean())
                f[f"{base}_min"] = float(arr.min())
                f[f"{base}_max"] = float(arr.max())
                f[f"{base}_std"] = float(arr.std(ddof=0))

    return {
        "cell_features": cell_features,
        "neighbors": neighbors,
        "n_cells": len(cell_features),
    }


def summarize_table(prefix: str, text: Any) -> dict[str, float]:
    df = parse_pipe_table(text)
    if df is None:
        return {}

    feats: dict[str, float] = {
        f"{prefix}__rows": float(len(df)),
        f"{prefix}__cols": float(len(df.columns)),
    }

    selected = []
    for col in df.columns:
        low = str(col).lower()
        if any(key in low for key in NUMERIC_KEYWORDS):
            selected.append(col)

    for col in selected[:80]:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        name = f"{prefix}__{clean_name(str(col), 55)}"
        feats[f"{name}__mean"] = float(series.mean())
        feats[f"{name}__min"] = float(series.min())
        feats[f"{name}__max"] = float(series.max())
        feats[f"{name}__std"] = float(series.std(ddof=0))
        feats[f"{name}__na"] = float(series.isna().mean())

    return feats


def scenario_features(scenario: dict[str, Any]) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats[f"tag__{scenario.get('tag', 'unknown')}"] = 1.0

    data = scenario.get("data", {})
    if isinstance(data, dict):
        for key, value in data.items():
            feats.update(summarize_table(clean_name(key), value))

    context = scenario.get("context", {})
    if isinstance(context, dict):
        wireless = context.get("wireless_network_information")
        feats.update(summarize_table("wireless_network_information", wireless))

    return feats


def option_features(option: dict[str, Any]) -> dict[str, float]:
    label = str(option.get("label", ""))
    low = label.lower()
    feats: dict[str, float] = {}
    feats[f"option_id__{option.get('id')}"] = 1.0
    for name, pattern in ACTION_PATTERNS.items():
        feats[f"action__{name}"] = 1.0 if re.search(pattern, low) else 0.0

    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", label)]
    feats["option_num_count"] = float(len(nums))
    if nums:
        feats["option_num_abs_mean"] = float(np.mean(np.abs(nums)))
        feats["option_num_abs_max"] = float(np.max(np.abs(nums)))
    return feats


def radio_option_features(option: dict[str, Any], radio: dict[str, Any]) -> dict[str, float]:
    """Candidate-specific graph features for a Track A option.

    The option label identifies target cells such as `3267908_4`. We aggregate
    observed/configured graph-role features for those cells and add pairwise
    relationship flags for neighbor-relation options.
    """
    label = str(option.get("label", ""))
    low = label.lower()
    targets = extract_cell_tokens(label)
    cell_features: dict[str, dict[str, float]] = radio.get("cell_features", {})
    neighbors: dict[str, set[str]] = radio.get("neighbors", {})

    feats: dict[str, float] = {
        "radio_target_count": float(len(targets)),
        "radio_target_unique_count": float(len(set(targets))),
        "radio_any_target_in_config": 0.0,
        "radio_all_targets_in_config": 0.0,
        "radio_pair_count": 0.0,
        "radio_pair_neighbor_relation_exists": 0.0,
        "radio_pair_same_arfcn": 0.0,
    }
    if not targets:
        return feats

    target_set = list(dict.fromkeys(targets))
    known = [t for t in target_set if t in cell_features]
    feats["radio_any_target_in_config"] = float(bool(known))
    feats["radio_all_targets_in_config"] = float(len(known) == len(target_set))
    feats["radio_known_target_fraction"] = float(len(known) / max(len(target_set), 1))
    feats["radio_total_cells_in_scenario"] = float(radio.get("n_cells", 0))

    feature_names = (
        "user_serving_share",
        "user_serving_rsrp_mean",
        "user_serving_rsrp_min",
        "user_serving_sinr_mean",
        "user_serving_throughput_mean",
        "user_serving_cce_fail_mean",
        "user_serving_mcs_mean",
        "user_serving_initial_bler_mean",
        "user_neighbor_share",
        "user_neighbor_rank_mean",
        "user_neighbor_rank_min",
        "user_neighbor_rsrp_mean",
        "user_neighbor_rsrp_max",
        "mr_serving_share",
        "mr_serving_rsrp_mean",
        "mr_serving_throughput_mean",
        "mr_neighbor_share",
        "mr_neighbor_rank_mean",
        "mr_neighbor_rank_min",
        "mr_neighbor_rsrp_mean",
        "mr_neighbor_rsrp_max",
        "traffic_dl_prb",
        "traffic_ul_interference",
        "traffic_dl_tput",
        "traffic_weak_coverage",
        "traffic_dl_cce_util",
        "traffic_dl_cce_success",
        "config_tx_power",
        "config_max_tx_power",
        "config_mech_azimuth",
        "config_digital_azimuth",
        "config_mech_downtilt",
        "config_digital_tilt",
        "config_a2_rsrp",
        "config_a5_rsrp1",
        "config_a5_rsrp2",
        "config_a3_offset",
        "config_pdcch_symbols",
        "config_neighbor_decl_count",
    )
    for name in feature_names:
        vals = [_cell_feature(cell_features, t, name) for t in target_set]
        add_prefixed_stats(feats, f"radio_target_{name}", vals)

    # Relative serving vs neighbor signal for the target cells.
    for src_a, src_b, out_name in (
        ("user_neighbor_rsrp_mean", "user_serving_rsrp_mean", "user_neighbor_minus_serving_rsrp"),
        ("mr_neighbor_rsrp_mean", "mr_serving_rsrp_mean", "mr_neighbor_minus_serving_rsrp"),
    ):
        vals = []
        for t in target_set:
            a = _cell_feature(cell_features, t, src_a)
            b = _cell_feature(cell_features, t, src_b)
            vals.append(a - b if not np.isnan(a) and not np.isnan(b) else np.nan)
        add_prefixed_stats(feats, f"radio_target_{out_name}", vals)

    if len(target_set) >= 2:
        pairs = [(target_set[i], target_set[j]) for i in range(len(target_set)) for j in range(i + 1, len(target_set))]
        feats["radio_pair_count"] = float(len(pairs))
        relation_hits = 0
        same_arfcn = 0
        for a, b in pairs:
            if b in neighbors.get(a, set()) or a in neighbors.get(b, set()):
                relation_hits += 1
            a_arfcn = _cell_feature(cell_features, a, "config_arfcn")
            b_arfcn = _cell_feature(cell_features, b, "config_arfcn")
            if not np.isnan(a_arfcn) and not np.isnan(b_arfcn) and int(a_arfcn) == int(b_arfcn):
                same_arfcn += 1
        feats["radio_pair_neighbor_relation_exists"] = float(relation_hits > 0)
        feats["radio_pair_neighbor_relation_fraction"] = float(relation_hits / max(len(pairs), 1))
        feats["radio_pair_same_arfcn"] = float(same_arfcn > 0)
        feats["radio_pair_same_arfcn_fraction"] = float(same_arfcn / max(len(pairs), 1))

    # Action × role interactions. These are intentionally simple and readable:
    # XGBoost can decide whether the sign/threshold is useful.
    action = option_features(option)
    serving_strength = feats.get("radio_target_user_serving_share__max", 0.0)
    neighbor_strength = feats.get("radio_target_user_neighbor_share__max", 0.0)
    cce_fail = feats.get("radio_target_user_serving_cce_fail_mean__max", 0.0)
    weak_cov = feats.get("radio_target_traffic_weak_coverage__max", 0.0)
    sinr = feats.get("radio_target_user_serving_sinr_mean__mean", 0.0)
    rsrp = feats.get("radio_target_user_serving_rsrp_mean__mean", 0.0)
    for action_name in ACTION_PATTERNS:
        flag = action.get(f"action__{action_name}", 0.0)
        feats[f"radio_interact_{action_name}_serving"] = flag * serving_strength
        feats[f"radio_interact_{action_name}_neighbor"] = flag * neighbor_strength
    feats["radio_interact_pdcch_cce_fail"] = action.get("action__pdcch", 0.0) * cce_fail
    feats["radio_interact_threshold_neighbor"] = action.get("action__coverage_threshold", 0.0) * neighbor_strength
    feats["radio_interact_tilt_weak_coverage"] = max(action.get("action__tilt_down", 0.0), action.get("action__tilt_up", 0.0)) * weak_cov
    feats["radio_interact_power_low_rsrp"] = max(action.get("action__tx_power_inc", 0.0), action.get("action__tx_power_dec", 0.0)) * (-rsrp if rsrp else 0.0)
    feats["radio_interact_azimuth_low_sinr"] = action.get("action__azimuth", 0.0) * (-sinr if sinr else 0.0)
    feats["radio_interact_neighbor_relation_pair_exists"] = action.get("action__neighbor", 0.0) * feats["radio_pair_neighbor_relation_exists"]
    feats["radio_interact_server_radio_looks_ok"] = action.get("action__server_transport", 0.0) * float((sinr or 0) > 10 and (rsrp or -999) > -90)

    # Presence flags help trees distinguish true zeroes from missing values.
    feats["radio_target_seen_as_serving"] = float(serving_strength > 0)
    feats["radio_target_seen_as_neighbor"] = float(neighbor_strength > 0)
    feats["radio_target_seen_in_mr"] = float(
        feats.get("radio_target_mr_serving_share__max", 0.0) > 0
        or feats.get("radio_target_mr_neighbor_share__max", 0.0) > 0
    )
    feats["radio_option_mentions_insufficient_data"] = float("insufficient data" in low or "more data is needed" in low)
    return feats


def option_text(scenario: dict[str, Any], option: dict[str, Any]) -> str:
    task = scenario.get("task", {})
    context = scenario.get("context", {})
    bits = [
        str(scenario.get("tag", "")),
        str(task.get("description", "")),
        str(context.get("description", "")) if isinstance(context, dict) else "",
        str(option.get("id", "")),
        str(option.get("label", "")),
    ]
    return "\n".join(bits)


def build_rows(scenarios: list[dict[str, Any]], labeled: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        sid = str(scenario.get("scenario_id"))
        positives = answer_set(str(scenario.get("answer", "")))
        s_feats = scenario_features(scenario)
        radio = build_radio_context(scenario)
        options = scenario.get("task", {}).get("options", [])
        for option in options:
            oid = str(option.get("id"))
            feats = dict(s_feats)
            feats.update(option_features(option))
            feats.update(radio_option_features(option, radio))
            rows.append(
                {
                    "scenario_id": sid,
                    "tag": scenario.get("tag", ""),
                    "option_id": oid,
                    "option_label": option.get("label", ""),
                    "text": option_text(scenario, option),
                    "features": feats,
                    "target": int(oid in positives) if labeled else -1,
                    "answer": "|".join(sorted(positives, key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)),
                }
            )
    return pd.DataFrame(rows)


def drop_option_id_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["features"] = [
        {k: v for k, v in features.items() if not str(k).startswith("option_id__")}
        for features in df["features"]
    ]
    return df


def make_matrix(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    max_text_features: int,
) -> tuple[Any, Any, Any, DictVectorizer]:
    return make_matrix_with_text(
        train_df,
        valid_df,
        max_text_features=max_text_features,
        text_vectorizer="tfidf",
    )


def make_text_vectorizer(
    text_vectorizer: str,
    *,
    max_text_features: int,
    min_df: int,
    bm25_k1: float,
    bm25_b: float,
) -> Any | None:
    if text_vectorizer == "none":
        return None
    common = {
        "lowercase": True,
        "ngram_range": (1, 2),
        "min_df": min_df,
        "max_features": max_text_features,
        "strip_accents": "unicode",
    }
    if text_vectorizer == "tfidf":
        return TfidfVectorizer(**common)
    if text_vectorizer == "bm25":
        return BM25Vectorizer(**common, k1=bm25_k1, b=bm25_b)
    raise ValueError(f"Unknown text vectorizer: {text_vectorizer}")


def make_matrix_with_text(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    max_text_features: int,
    text_vectorizer: str = "tfidf",
    text_min_df: int = 2,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> tuple[Any, Any, Any | None, DictVectorizer]:
    text_vec = make_text_vectorizer(
        text_vectorizer,
        max_text_features=max_text_features,
        min_df=text_min_df,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    dict_vec = DictVectorizer(sparse=True)

    x_train_dict = dict_vec.fit_transform(train_df["features"])
    x_valid_dict = dict_vec.transform(valid_df["features"])
    if text_vec is None:
        x_train = x_train_dict
        x_valid = x_valid_dict
    else:
        x_train_text = text_vec.fit_transform(train_df["text"].astype(str))
        x_valid_text = text_vec.transform(valid_df["text"].astype(str))
        x_train = sparse.hstack([x_train_text, x_train_dict], format="csr")
        x_valid = sparse.hstack([x_valid_text, x_valid_dict], format="csr")
    return (
        x_train,
        x_valid,
        text_vec,
        dict_vec,
    )


def iou(pred: set[str], truth: set[str]) -> float:
    if not pred and not truth:
        return 1.0
    union = pred | truth
    return len(pred & truth) / len(union) if union else 0.0


def evaluate_rankings(df: pd.DataFrame) -> dict[str, Any]:
    by_scenario = []
    for sid, group in df.groupby("scenario_id", sort=False):
        truth = answer_set(group["answer"].iloc[0])
        ranked = group.sort_values("score", ascending=False)
        top_ids = ranked["option_id"].tolist()
        tag = group["tag"].iloc[0]
        fixed_n = 1 if tag == "single-answer" else min(4, len(top_ids))
        pred_fixed = set(top_ids[:fixed_n])
        pred_threshold = set(ranked.loc[ranked["score"] >= 0.35, "option_id"].head(4))
        if not pred_threshold:
            pred_threshold = {top_ids[0]}
        by_scenario.append(
            {
                "scenario_id": sid,
                "tag": tag,
                "truth": "|".join(sorted(truth)),
                "top1": top_ids[0],
                "top3": "|".join(top_ids[:3]),
                "top5": "|".join(top_ids[:5]),
                "top1_hit": float(bool(truth & set(top_ids[:1]))),
                "top3_recall_all": float(truth <= set(top_ids[:3])),
                "top5_recall_all": float(truth <= set(top_ids[:5])),
                "top8_recall_all": float(truth <= set(top_ids[:8])),
                "iou_fixed": iou(pred_fixed, truth),
                "iou_threshold": iou(pred_threshold, truth),
            }
        )
    detail = pd.DataFrame(by_scenario)
    return {
        "scenario_count": int(len(detail)),
        "top1_hit": float(detail["top1_hit"].mean()),
        "top3_full_recall": float(detail["top3_recall_all"].mean()),
        "top5_full_recall": float(detail["top5_recall_all"].mean()),
        "top8_full_recall": float(detail["top8_recall_all"].mean()),
        "mean_iou_fixed_by_tag": float(detail["iou_fixed"].mean()),
        "mean_iou_threshold_035": float(detail["iou_threshold"].mean()),
        "detail": detail,
    }


def train(args: argparse.Namespace) -> None:
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
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    valid_df = valid_df.copy()
    valid_df["score"] = model.predict_proba(x_valid)[:, 1]
    ranking_metrics = evaluate_rankings(valid_df)
    metrics = {
        "train_scenarios": len(train_scenarios),
        "valid_scenarios": len(valid_scenarios),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "positive_rate": float(y_train.mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "text_vectorizer": args.text_vectorizer,
        "text_min_df": args.text_min_df,
        "drop_option_id_feature": args.drop_option_id_feature,
        "bm25_k1": args.bm25_k1 if args.text_vectorizer == "bm25" else None,
        "bm25_b": args.bm25_b if args.text_vectorizer == "bm25" else None,
        "row_average_precision": float(average_precision_score(y_valid, valid_df["score"])),
        "row_roc_auc": float(roc_auc_score(y_valid, valid_df["score"])),
        **{k: v for k, v in ranking_metrics.items() if k != "detail"},
    }

    valid_out = ranking_metrics["detail"].merge(
        valid_df[["scenario_id", "option_id", "option_label", "score", "target"]],
        on="scenario_id",
        how="left",
    )
    valid_out.to_csv(out_dir / "validation_rankings.csv", index=False)
    valid_df.to_csv(out_dir / "validation_option_scores.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    with (out_dir / "ranker.pkl").open("wb") as fp:
        pickle.dump({"model": model, "text_vec": text_vec, "dict_vec": dict_vec, "metrics": metrics}, fp)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default="telco_data/Track A/data/Phase_1/train.json")
    parser.add_argument("--out-dir", default="work/track_a_ranker")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-features", type=int, default=20000)
    parser.add_argument("--text-vectorizer", choices=["tfidf", "bm25", "none"], default="tfidf")
    parser.add_argument("--text-min-df", type=int, default=2)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--drop-option-id-feature", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
