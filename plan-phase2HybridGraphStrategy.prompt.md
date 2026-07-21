## Plan: Phase 2 Hybrid Graph Strategy (Refined, Win-Optimized)

**Thesis.** The agent must be **Qwen3.5-35B-A3B** (organizer-confirmed, mandatory all phases). Phase 2 problems are **closed-vocabulary fault classification** (19 routing reasons + 12 port reasons) with a rigid `node;ip-or-port;reason` output schema. Optimize for: (a) standing up a compliant Qwen tool-calling agent first, (b) exploiting the closed fault catalog with deterministic detectors and a fault→command playbook, (c) a typed offline graph and permission-aware deterministic ranker that compresses API calls, **but only for scenarios with verified static bundles and identifier mapping**, (d) an **XGBoost `rank:pairwise` calibration layer** trained on silver labels from a budget-disciplined oracle Qwen run on Phase 1 (sits on top of the deterministic ranker, gated by a Day-5 ablation on Phase-2-shaped validation — drops out of Sub #1 if it does not beat deterministic-only), and (e) hard format/feasibility guards on the LLM's draft answer. **No GNN work in Phase 2.** Three submissions, 8 days (deadline 2026-05-18), 500-call/scenario hard ceiling.

**Hard constraints driving every choice**
- Inference engine: Qwen3.5-35B-A3B only; the existing `agent/openclaw_config/`, `agent/claude_competition_prompt_short.md`, and `agent/evaluate_openclaw.py` Claude/openclaw stack is **non-compliant** and is treated as design reference only.
- Submissions: 3. Allocate as floor / improvement / polish.
- API budget: 500 calls per scenario; tiebreaker on calls used on **correctly answered** problems.
- Output schema: `fault-node;destination-IP;fault-reason` or `fault-node;fault-port;fault-reason`, English symbols only, no whitespace, no blank lines, fault reasons drawn from the closed list of 19+12.
- Phase 3 forward compatibility: Pass@1 over 4 trials + TTS speed buckets + 15-min per-task cutoff on a *different* network. The Phase 2 architecture must be deterministic at inference, fast, and not overfit to specific node names.

**Steps**

0. **Day 1 — Dataset / identifier contract validation (blocker before offline features).** Build a manifest `scenario_manifest.parquet` that records, for every Phase 1 / Phase 2 item: `scenario_id`, API-facing `question_number`, presence/absence of a static `devices_outputs/` bundle, and presence/absence of `question_limits_config.json` restrictions. Confirm whether the current `devices_outputs/` directory (50 bundles) maps only to Phase 1 or to a subset of Phase 2, and unzip / locate any missing archive before assuming broader coverage. Establish the **question_number ↔ scenario_id** linkage from server-side filenames, traces, or loader code; if a complete mapping cannot be recovered on Day 1, then Steps 4–5 become **conditional accelerators** only, keyed off the manifest, and the live execution path must rely on API `question_number` plus parser/playbook/ranker features that do not require static bundles. This step also defines the fallback contract: scenarios without a static bundle get sparse "offline-missing" features and skip graph/anomaly evidence rather than failing closed.

1. **Day 1 — Stand up the Qwen3.5-35B-A3B agent (compliance blocker).** Deploy Qwen3.5-35B-A3B locally (vLLM or sglang serving), wire a function-calling loop using the model's native tool-call format (Qwen function-call schema; Hermes-style only if Qwen-native is unstable). Port the four openclaw skill domains (`infra_maintenance`, `l2_link`, `l3_route`, `adv_tunnel`) to Qwen tool specs that wrap the Agent Tool Server `POST /api/agent/execute` endpoint. Select **3–5 few-shot exemplars** from the past competition archive at `/Users/ronnypolle/Desktop/telco_itu/the-ai-telco-troubleshooting-challenge20260120-9768-ainojt.zip` (`train.csv` + `phase_1_test_truth.csv`); pick ones that resemble structured fault diagnosis and embed them as in-context examples in the Qwen system prompt. Keep total system prompt ≤ 4K tokens. Validate end-to-end on one Phase 2 question before doing anything else. Until this works, every other step is wasted.

2. **Day 1–2 — Question-text constraint parser + output-format guard.** Build a deterministic preprocessor that extracts from the question text: target destination IP/host, source endpoint, blacklisted nodes (`Limitation: Do not look for faults on X`), explicitly disclosed fault category (e.g., *"VRRP dual-master on Vlanif120"*), and suspected protocol family. Build a post-LLM regex validator that hard-enforces the schema, rejects malformed or out-of-vocabulary lines, and triggers exactly one re-emit. Format errors zero out a question — this is the cheapest accuracy save. Parsed flags become XGBoost features in Step 8.

3. **Day 2–3 — Fault-catalog command playbook (largest accuracy lever).** For each of the 19 routing + 12 port fault reasons, precompute the 2–3 highest-yield diagnostic commands and the per-vendor variants (Huawei / Cisco / H3C). Encode the diagnostic *signature* per category (e.g., VRRP dual-master → two MASTER states in `display vrrp verbose` for the same group; "global STP not enabled" → absence of `stp enable` in `display current-configuration`; "blackhole route" → null next-hop in `display ip routing-table` for the destination). The LLM consults the playbook rather than inventing diagnostic paths.

4. **Day 2–4 — Offline config-anomaly miner over verified static bundles only.** For scenarios whose `scenario_manifest` entry confirms a `devices_outputs/` bundle, deterministic detectors run at **zero API cost**. For each closed fault category, write a regex/structural detector over `display current-configuration`, routing tables, VRRP/STP/BGP outputs. Emit candidate `(scenario_id, node, fault_reason, evidence_dict)` tuples. Many of the 19+12 categories are literal config matches; this is the highest expected-accuracy step that does not depend on the LLM at all. The evidence dict (binary detector flags + evidence-strength scalars) becomes the XGBoost feature spine. For scenarios without a static bundle, emit explicit `offline_bundle_missing=1` features and skip this evidence path rather than fabricating negatives.

5. **Day 3–4 — Typed offline graph + permission-aware hard pruner.** Build the device → protocol → interface typed graph only for scenarios with a verified static bundle in `scenario_manifest`: nodes for devices/interfaces/VLANs/VRRP groups/VRFs/tunnels, typed edges for L2 adjacency, L3 next-hop, VRRP membership, route propagation. Vendor and command-availability metadata on nodes. **Hard-prune** denied (device, command) pairs from `question_limits_config.json` *before* ranking — never penalty-rank, since denied calls waste the tiebreaker for free — but only after the Day-1 manifest establishes the `question_number ↔ scenario_id` mapping. Granularity locked to device-protocol-interface (transfer-friendly to Phase 3's different network). Graph features (degree, betweenness, on-parsed-path flag, hop distance from source/destination) feed Step 8. If a scenario lacks either a static bundle or a verified permission mapping, fall back to parser/playbook-only ranking with explicit missing-feature indicators.

6. **Day 4–5 — Deterministic weighted ranker (top of the existing template).** Reuse the multi-signal pattern from `mutation_discovery/gnn/driver_scorer.py`. Signals (component scores transparent and ablatable):
   - graph centrality / path relevance (constrained by parsed source/destination)
   - protocol-family match to the question text
   - vendor-command compatibility
   - anomaly-miner candidate priors (Step 4)
   - permission survivors only (post-pruning)
   - contradiction penalties (e.g., proposed fault on a blacklisted node)
   - evidence-coverage bonus (does the answer explain the symptom?)
   Output per question: ranked candidate (node, command) pairs and ranked candidate (node, fault_reason) answer hypotheses. Each component score is emitted as `{entity, score, score_norm}` (port the `driver_scorer.py` DataFrame pattern) and persisted so Step 6.5 can consume the same matrix.

7. **Day 3–4 — Oracle Qwen silver-label generation for XGBoost training.** Start this only after Steps 4–5 pass their baseline checks on a 10-scenario pilot and the manifest confirms the Phase 1 static-bundle coverage. For each of the **50 Phase 1 scenarios** (`data/Phase_1/test.json`):
   - Run the Qwen agent in **oracle mode** with a **total per-scenario budget capped at 500 API calls across all retries/seeds**. Use one primary run capped at 350 calls, then launch up to two audit reruns capped at 75 calls each **only** when the primary answer is format-invalid, feasibility-invalid, or low-margin. No unconditional 3-seed reruns.
   - Capture the agent's final answer (the silver label). Mark a scenario as confidently labeled if the primary run is valid and any audit rerun that was triggered agrees on every fault line; otherwise flag as low-confidence and down-weight in training.
   - For each scenario, build the candidate pool: (anomaly-miner outputs ∪ graph-plausible nodes within parsed path) × the 19+12 reason vocabulary, filtered by question type and permission pruner. Typical pool size 100–300 candidates per scenario.
   - Label candidates matching the silver answer as `relevance=1`, others as `relevance=0`. Multi-fault scenarios → multiple positives.
   - Persist as `data/silver_labels_phase1.parquet` with columns `scenario_id, candidate_id, node, fault_reason, relevance, sample_weight, <feature_columns...>`.

8. **Day 4–5 — XGBoost LTR calibration layer on top of the deterministic ranker.** Trainable substitute for the cut GNN step.
   - **Feature matrix per `(scenario_id, candidate)`:**
     - All Step 6 component scores: `graph_centrality_norm, path_relevance_norm, protocol_match, vendor_compat, anomaly_prior, contradiction_penalty, evidence_coverage`.
     - Anomaly-miner evidence indicators (binary per detector category) and evidence-strength scalars (Step 4).
     - Graph features: degree, betweenness, on-parsed-path flag, hop distance from source/destination (Step 5).
     - Constraint-parser flags: `target_ip_match, blacklisted_node, denied_command_count, disclosed_fault_category_match` (Step 2).
     - **All features computed scenario-relative** (within-scenario rank percentiles or z-scores), never absolute, to transfer across networks. Node-name and absolute-graph features banned.
   - **Training:** `xgboost.XGBRanker` with `objective="rank:pairwise"`, group sizes from `scenario_id`. Reserve **Fold 5** as an untouched calibration / threshold-tuning fold. Train a **4-fold ensemble** over Folds 1–4 using `GroupKFold` by `scenario_id`. Sweep `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {4, 6}`, `n_estimators ∈ {200, 500}`, fixed `min_child_weight=5`, `reg_lambda=1.0`. Early-stop on validation NDCG@5 inside the 4-fold loop. Sample-weight low-confidence silver labels at 0.5 and high-confidence at 1.0 (Step 7).
   - **Calibration:** isotonic regression on reserved Fold 5 maps raw ranker output to a probability-like `calibrated_score ∈ [0, 1]`.
   - **Uncertainty:** 4-fold ensemble; `uncertainty = std` of per-fold predictions for the same candidate.
   - **Inference outputs:** `(calibrated_score, uncertainty)` per `(scenario, candidate)`.
   - **Promotion gate (Day 5, mandatory):** XGBoost ships only if it beats deterministic-only on reserved-fold NDCG@5 **and** on calls/correct in the Phase-2-shaped local eval (Step 11). Otherwise drop XGBoost from Sub #1 entirely; reconsider for Sub #2 after error analysis.

9. **Day 5 — Compact graph/ranker context for Qwen.** Inject ≤10 lines of structured JSON into the system prompt: parsed constraints, top-3 candidate fault hypotheses with evidence pointers and (if XGBoost passes the gate) `calibrated_score` + `uncertainty` per candidate, top-5 next-best commands, hard-blacklisted nodes/commands. Top-K selection is by `calibrated_score` if XGBoost is in; otherwise by deterministic combined score. **Do not** dump graph state. Tune for Qwen's prompt-bandwidth behavior — re-test prompt length tradeoffs since this differs from Claude.

10. **Day 5–6 — Answer-feasibility validator + uncertainty-gated single follow-up.** After Qwen drafts an answer, validate against the offline graph + anomaly-miner **when available**: does the proposed fault node lie on the parsed path? Is the fault category compatible with the symptom? Does evidence support it? **Trigger exactly one targeted follow-up** when (a) format guard rejects, OR (b) graph-feasibility check rejects, OR (c) chosen candidate's `uncertainty > τ_unc` OR `calibrated_score < τ_score` (XGBoost path only). Tune `τ_unc` and `τ_score` on reserved Fold 5, then sanity-check them on a small Phase 2 pilot to keep follow-up rate near 10%. No broad re-exploration.

11. **Day 6 — Local eval loop and ablations.** Variants on a held-out subset of `data/Phase_1` (using oracle silver labels as proxy ground truth) and a Phase-2-shaped validation slice: prioritize Phase 2 scenarios with verified static bundles, then add a small live-API pilot for scenarios without bundles so the fallback path is exercised:
   - V0: bare Qwen agent (no graph, no playbook, no validator)
   - V1: Qwen + parser + format guard + playbook
   - V2: V1 + offline graph + permission pruning + deterministic ranker context
   - V3: V2 + anomaly miner + answer validator + 1-shot follow-up
   - **V4: V3 + XGBoost calibration layer + uncertainty-gated follow-up**
   Metrics: accuracy, calls per correct answer, calls per scenario distribution, format-error rate, follow-up-trigger rate, variance across 3 reruns (Phase-3 Pass@1 proxy). Ablate ranker signals one at a time. **XGBoost promotion gate evaluated here.**

12. **Day 7 — Submission #1 (floor).** Lock the lowest-variance variant that beats V0 on both accuracy and calls (V3 or V4 depending on Day-5 gate result). Freeze prompts, graph version, ranker weights, XGBoost model + calibrator + thresholds, playbook, dependency pins, random seeds. Reproducibility package per organizer requirements (result.csv + execution trace).

13. **Day 7–8 — Submissions #2 and #3 (conditional).** Promotion rules:
    - Sub #2: ship the **alternate** of V3↔V4 (whichever was not chosen for Sub #1) if Day-5 eval gap was small (<2pp), otherwise ship a tuned variant of Sub #1's winner. Promote only if Δaccuracy ≥ +2pp OR Δcalls ≤ −15% with zero accuracy regression vs Sub #1.
    - Sub #3 only on top of a confirmed-good Sub #2; reserve for prompt polish, threshold retuning from Sub #2 traces, retry policy, or a single high-confidence playbook fix. Do not ship #3 in the final 2 hours without margin.

**Submission allocation table**

| Slot | Day | Variant | Promotion rule |
|------|-----|---------|----------------|
| #1 | Day 7 | Best-tested V3 or V4 | Lowest-variance variant beating V0; XGBoost in only if it passed the Day-5 gate |
| #2 | Day 8 AM | Alternate (V3↔V4 swap) or tuned winner | Δacc ≥ +2pp OR Δcalls ≤ −15%, no regression |
| #3 | Day 8 PM | Polish on #2 | Only with ≥2h margin and a clean diff |

**Relevant files**
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/README.md` — phase rules, scoring, vendor scope
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/data/Phase_2/test.json` — 100 Phase-2 questions; structured fault-classification schema
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/data/Phase_2/README.md` — API endpoints, payload, 500-call/scenario cap, supported command whitelist per vendor
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/data/Phase_1/test.json` — 50 scenarios used for oracle silver labeling (Step 7)
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/question_limits_config.json` — `question_number`-keyed denied (device, command) pairs; requires Day-1 mapping before it can prune `scenario_id`-indexed offline artifacts
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/server.py` — local simulator, vendor regex whitelist, cache behavior
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/devices_outputs/` — currently 50 static CLI-output bundles; Day-1 manifest determines which scenarios they cover before Steps 4–5 assume availability
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/examples/traces.json` — 3 reasoning trajectories; useful as constraint-parser sanity checks, not training labels
- `/Users/ronnypolle/Desktop/telco_itu/the-ai-telco-troubleshooting-challenge20260120-9768-ainojt.zip` — past-competition archive (`train.csv`, `phase_1_test_truth.csv`); **few-shot in-context exemplars only** (Step 1). Schema differs from Track B; not used for LoRA or LTR labels.
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/agent/openclaw_config/SOUL.md`, `claude_competition_prompt_short.md`, `claude_past_challenge_insights.md` — **design references only** (Claude-targeted, non-compliant for submission); port principles to Qwen system prompt
- `/Users/ronnypolle/Desktop/telco_itu/telco_data/Track B/agent/skills/{adv_tunnel,infra_maintenance,l2_link,l3_route}` — skill scaffolding to port to Qwen tool specs
- `/Users/ronnypolle/Desktop/mutation_discovery/gnn/driver_scorer.py` — template for transparent weighted multi-signal ranking with component breakdowns; reuse the per-scorer `pd.DataFrame` emission pattern (`{entity, <score>, <score>_norm}`) for the XGBoost feature matrix in Step 8

**Verification**
1. Day-1 manifest is complete for all Phase 1 / Phase 2 entries: each row has `scenario_id`, API `question_number` or an explicit "mapping unresolved" flag, static-bundle presence, and permission-rule presence.
2. Qwen agent runs end-to-end on one Phase 2 question with the past-competition exemplar prompt, produces a schema-valid answer, and uploads a trace in the organizer's expected format.
3. Format guard rejects every synthetic out-of-vocabulary or malformed line on a fuzz set; no false rejections of valid lines.
4. Offline anomaly miner reproduces faults on a held-out subset of `data/Phase_1` (oracle silver labels) with precision ≥ 0.7 on the categories it claims (precision-leaning; recall comes from the LLM), and emits `offline_bundle_missing=1` instead of empty evidence on uncovered scenarios.
5. Permission pruner confirms zero denied calls in execution traces across the local eval **for scenarios whose mapping is resolved**; unresolved scenarios fall back cleanly without referencing the pruner.
6. Oracle silver-label run completes on all 50 Phase 1 scenarios within the **500 total calls / scenario** budget by Day 4 EOD; `data/silver_labels_phase1.parquet` has ≥50 rows with `relevance=1` and a non-degenerate `sample_weight` distribution.
7. XGBoost training: reserved Fold 5 stays untouched until calibration / threshold tuning; calibrated scores fall in `[0, 1]`; uncertainty band has non-zero variance across the 4 training folds. Day-5 promotion gate evaluated and recorded on both reserved-fold ranking metrics and the Phase-2-shaped eval slice.
8. Ranker ablation: each signal's removal degrades either accuracy or calls/correct on the local eval (kills features that don't pay rent). XGBoost gate-result logged.
9. Reproducibility: same scenario rerun 3× yields identical answers and ±5% call count; XGBoost calibrated scores deterministic to the saved seed.
10. Phase-3 generalization sanity: graph extractor + anomaly miner + XGBoost feature pipeline run on a synthetically renamed copy of one scenario without code changes (no hard-coded node names; no absolute-graph features), and the agent can fall back to deterministic ranking if XGBoost artifacts fail to load.

**Decisions**
- Inference engine: Qwen3.5-35B-A3B, served locally (vLLM/sglang). Non-negotiable.
- Graph granularity: device-protocol-interface typed graph (transferable to Phase 3's different network).
- Insight injection surface: prompt-side structured summary (≤10 lines JSON), not a separate retrieval skill.
- Validator behavior on rejection: exactly one targeted follow-up, then commit. Trigger now includes XGBoost uncertainty/score thresholds when XGBoost is in.
- **In scope (gated): XGBoost `rank:pairwise` calibration layer** trained on oracle silver labels from Phase 1; ships to Sub #1 only if Day-5 ablation gate passes.
- **In scope (low-cost): past-competition data as 3–5 few-shot in-context exemplars** for Qwen (no LoRA, no LTR labels — schema mismatch).
- Offline graph / anomaly features are **conditional on manifest-confirmed static bundles**; fallback path for uncovered scenarios is parser + playbook + live-API ranker, not failure.
- **Out of Phase 2 scope**: GNN reranker, LoRA fine-tuning, end-to-end answer model, ensembling/consensus across ranker variants, generic "Phase 0 lock scope" planning step.

**Risks and mitigations**
- *Qwen tool-calling instability or unsupported function format*: fall back to Hermes-style ReAct format with parser; validate on Day 1, not Day 5.
- *Anomaly miner overfits to Phase-1/2 string patterns*: write detectors against config *structure* (sections, command presence/absence) not specific node names.
- *Format guard rejects valid edge-case answers*: keep rejection lenient (one retry, then accept the LLM output) so guard never zeros a correct answer.
- *Local hardware insufficient for 35B serving*: pre-flight on Day 1; if blocked, request remote GPU before Day 2, otherwise the entire plan stalls.
- *500-call/scenario cap hit during exploration*: ranker permission pruner + playbook should keep typical scenarios well under 100 calls; monitor and fail-safe early-stop at 400.
- *Silver labels noisy or biased toward the oracle's failure modes*: audit-rerun agreement filter (Step 7); sample-weight low-confidence labels at 0.5; cap XGBoost depth at 6; track NDCG@5 on held-out fold for distribution shift.
- *50 scenarios is a small training set for XGBoost*: shallow trees (`max_depth ≤ 6`), strong regularization (`reg_lambda ≥ 1`, `min_child_weight ≥ 5`), within-scenario rank features only, isotonic calibration on a clean fold.
- *Phase 1 → Phase 2 distribution shift (different network, different protocol mix — e.g., VRRP/MP-BGP only in Phase 2)*: features must be scenario-relative; node-name and absolute-graph features banned; Day-5 gate is the safety valve — XGBoost ships only if it actually generalizes on Phase-2-shaped held-out scenarios.
- *Identifier / coverage mismatch between `scenario_id`, `question_number`, `question_limits_config.json`, and `devices_outputs/`*: front-load the Day-1 manifest and treat offline features as conditional; unresolved mappings degrade to the live-only path rather than breaking execution.
- *Phase-3 runtime lacks a compatible XGBoost wheel / native library*: pin the version during local packaging tests and keep a deterministic-ranker fallback flag that disables XGBoost cleanly at startup.

**Phase 3 forward-looking notes (do not block Phase 2)**
- Pass@1 over 4 trials and 15-min per-task cutoff make determinism + latency mandatory; the deterministic ranker + XGBoost calibration is already the right architecture.
- Phase 3 uses a *different* network — the graph extractor, anomaly miner, and XGBoost feature pipeline must read from `devices_outputs/`-shaped inputs without node-name assumptions.
- Phase 3 introduces SRv6/EVPN/ISIS protocol families that may be sparse in Phase 1 oracle silver labels. Budget a feature-extraction sanity check on the first Phase 3 scenario; if novel-protocol features dominate the candidate scores with high uncertainty, fall back to the deterministic ranker for that scenario.
- Phase 3 organizer deploys Qwen3.5-35B-A3B and the Agent Tool Server on Huawei Cloud; package the agent code, prompt templates, ranker code, anomaly miner, and any optional XGBoost artifacts in a hermetic build, but ensure the code can start and solve scenarios with deterministic ranking only if the XGBoost dependency is unavailable.
