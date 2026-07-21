# Track A XGBoost Option Ranker

This is an offline sidecar experiment for Track A. It treats each scenario option
as a binary classification row:

```text
scenario features + option text/features -> option_is_correct
```

This avoids learning that a global label like `C7` is always correct. In Track A,
option IDs are scenario-local and must be interpreted together with their option
text.

## Artifacts

- `train_xgb_ranker.py`: trains and evaluates an option-level XGBoost ranker.
- `rank_options.py`: loads a trained model and ranks options for new scenarios.
- `xgb_shortlist/ranker.pkl`: best current shortlist model.
- `xgb_shortlist/metrics.json`: validation metrics for the shortlist model.
- `xgb_shallow/ranker.pkl`: shallower model with better top-1/fixed-answer IoU.
- `tuning/tuning_results.csv`: hyperparameter tuning leaderboard.
- `tuning/best_shortlist/ranker.pkl`: tuned model optimized for broad top-k recall.
- `tuning/best_fallback/ranker.pkl`: tuned model optimized for direct fallback IoU.
- `tuning_radio/best_shortlist/ranker.pkl`: radio-graph enhanced tuned model.
- `tuning_radio/best_fallback/ranker.pkl`: radio-graph enhanced tuned model;
  currently the default used by `rank_options.py`.
- `phase2_rankings.jsonl`: top-8 ranked options for all Phase 2 scenarios.
- `phase2_xgb_proposals.csv`: compact Phase 2 proposed answers and top-8 lists.
- `phase2_rankings_tuned_shortlist.jsonl`: Phase 2 top-8 rankings from tuned shortlist model.
- `phase2_rankings_tuned_fallback.jsonl`: Phase 2 top-8 rankings from tuned fallback model.
- `phase2_xgb_tuned_shortlist.csv`: compact tuned shortlist proposals.
- `phase2_xgb_tuned_fallback.csv`: compact tuned fallback proposals.
- `phase2_rankings_radio_shortlist.jsonl`: Phase 2 rankings from the radio-graph model.
- `phase2_rankings_radio_fallback.jsonl`: Phase 2 fallback rankings from the radio-graph model.
- `phase2_xgb_radio_shortlist.csv`: compact radio-graph shortlist proposals.
- `phase2_xgb_radio_fallback.csv`: compact radio-graph fallback proposals.

## Current Validation Signal

The shortlist model uses a scenario-level 80/20 split of Phase 1 train:

```text
top1_hit: 0.3300
top3_full_recall: 0.5375
top5_full_recall: 0.7775
top8_full_recall: 0.8500
mean_iou_fixed_by_tag: 0.3239
```

The shallow model is better as a direct fallback but worse as a broad shortlist:

```text
top1_hit: 0.3850
top3_full_recall: 0.6525
top5_full_recall: 0.7300
top8_full_recall: 0.7400
mean_iou_fixed_by_tag: 0.3889
```

## Tuned Models

`tune_xgb_ranker.py` ran 24 fixed-split trials over tree depth, learning rate,
sampling, child weight, gamma, and regularization.

Best tuned shortlist model:

```text
n_estimators: 70
max_depth: 6
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 1.0
min_child_weight: 8.0
gamma: 0.25
reg_alpha: 0.001
reg_lambda: 2.0
top1_hit: 0.2825
top3_full_recall: 0.5500
top5_full_recall: 0.8250
top8_full_recall: 0.8500
mean_iou_fixed_by_tag: 0.2764
```

Best tuned fallback model:

```text
n_estimators: 70
max_depth: 4
learning_rate: 0.025
subsample: 0.8
colsample_bytree: 0.55
min_child_weight: 4.0
gamma: 0.25
reg_alpha: 0.01
reg_lambda: 0.5
top1_hit: 0.3875
top3_full_recall: 0.6775
top5_full_recall: 0.7500
top8_full_recall: 0.8500
mean_iou_fixed_by_tag: 0.3910
```

Use `tuning/best_shortlist/ranker.pkl` when the LLM needs a broad candidate
list. Use `tuning/best_fallback/ranker.pkl` when filling blanks or creating a
non-LLM backup answer.

## Radio-Graph Features

The ranker now adapts the Track B graph idea to Track A by building a typed
radio context per scenario. Instead of device/interface topology, Track A uses
cell-level graph roles:

```text
option -> target cell(s)
cell -> PCI/ARFCN
cell -> configured neighbor cells
drive-test rows -> serving cell / neighbor cell roles
MR rows -> serving cell / neighbor cell roles
traffic rows -> cell load / coverage / CCE statistics
```

Option labels are parsed for cell tokens such as `3267908_4`. Those target
cells are joined to observed PCI/ARFCN roles from `user_plane_data` and
`mr_data`, plus configuration and traffic stats. This adds candidate-specific
features such as:

```text
radio_target_seen_as_serving
radio_target_seen_as_neighbor
radio_pair_neighbor_relation_exists
radio_target_user_neighbor_share__mean
radio_target_user_neighbor_rank_mean__mean
radio_target_mr_serving_throughput_mean__min
radio_interact_pdcch_cce_fail
radio_interact_threshold_neighbor
radio_interact_tilt_weak_coverage
radio_interact_power_low_rsrp
```

This is the biggest performance jump so far. With the same fixed Phase 1
scenario-level 80/20 split, the best radio-graph model achieved:

```text
n_estimators: 220
max_depth: 2
learning_rate: 0.1
subsample: 0.9
colsample_bytree: 0.85
min_child_weight: 1.0
gamma: 0.1
reg_alpha: 0.05
reg_lambda: 1.0
row_average_precision: 0.7848
row_roc_auc: 0.9753
top1_hit: 0.9250
top3_full_recall: 0.8950
top5_full_recall: 0.9125
top8_full_recall: 0.9575
mean_iou_fixed_by_tag: 0.8953
```

The top importances are dominated by radio graph features such as target-cell
neighbor share/rank, MR serving throughput, MR neighbor share, traffic PRB, and
PDCCH/CCE interactions, which is a useful sanity check that the lift is coming
from the intended features.

## Usage

Train:

```bash
python3 work/track_a_ranker/train_xgb_ranker.py \
  --out-dir work/track_a_ranker/xgb_shortlist \
  --n-estimators 80 \
  --max-text-features 8000 \
  --n-jobs 4
```

Text-vectorizer ablations:

```bash
python3 work/track_a_ranker/train_xgb_ranker.py \
  --out-dir work/track_a_ranker/ablation_bm25_fixed \
  --text-vectorizer bm25 \
  --max-text-features 8000 \
  --n-estimators 220 \
  --max-depth 2 \
  --learning-rate 0.1 \
  --n-jobs 6
```

The ranker currently supports `--text-vectorizer tfidf`, `bm25`, and `none`.
Use `--drop-option-id-feature` when checking whether validation is relying on
the stable `C*` option ids rather than option semantics.

On the fixed radio-graph XGBoost configuration used for the current skill,
TF-IDF beat BM25 on top-1 and fixed-answer IoU. BM25 improved top5/top8 recall,
but was materially worse as a direct fallback answerer:

```text
run                  top1   top5   top8   fixed_iou
tfidf_fixed          .873   .905   .908   .877
bm25_fixed           .828  1.000  1.000   .770
none_fixed          1.000  1.000  1.000   .944
tfidf_no_option_id   .873   .905   .908   .882
bm25_no_option_id    .710  1.000  1.000   .634
none_no_option_id   1.000  1.000  1.000   .944
```

Full results are in `work/track_a_ranker/text_ablation_summary.csv`.

Rank one scenario:

```bash
python3 work/track_a_ranker/rank_options.py \
  --scenario-id 40573780-92ac-4436-97c7-efca33b2a839 \
  --top-k 8
```

Rank all Phase 2 scenarios:

```bash
python3 work/track_a_ranker/rank_options.py \
  --model work/track_a_ranker/xgb_shortlist/ranker.pkl \
  --scenarios "telco_data/Track A/data/Phase_2/test.json" \
  --top-k 8 \
  --out work/track_a_ranker/phase2_rankings.jsonl
```

## Recommended Integration

Expose `rank_options.py` as a local/HTTP tool that the LLM can call with a
`scenario_id`. The LLM should use it as a shortlist and prior, not as the sole
decision-maker.
