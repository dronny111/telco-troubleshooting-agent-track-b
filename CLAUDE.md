# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This repo is one participant's solution for **Track B (IP networking)** of the ITU "Telco Troubleshooting and Optimization Agentic Challenge". Phase 2 (deadline 2026-05-18) requires an agent built on **Qwen3.5-35B-A3B** (mandatory base model across all phases) that diagnoses faults on a 40-node multi-vendor campus network by calling the organiser's Agent Tool Server (`POST /api/agent/execute`).

The full strategy is `plan-phase2HybridGraphStrategy.prompt.md` (Steps 0–11). That plan is the source of truth — when in doubt, re-read it before changing pipeline shape. `.github/copilot-instructions.md` contains a parallel, more terse view of the same architecture and is also accurate.

**Hard constraints driving every code change:**
- Qwen3.5-35B-A3B only. The pre-existing `telco_data/Track B/agent/openclaw_config/` and `evaluate_openclaw.py` are Claude/openclaw-targeted and **non-compliant for submission** — treat as design reference only.
- 500 API calls / scenario hard ceiling; calls-on-correct is the Phase 2 tiebreaker. Never burn calls on denied (device, command) pairs (`question_limits_config.json`) — hard-prune, don't penalty-rank.
- The `format_guard` enforces three task-family schemas: `fault` (`node;ip-or-port;reason`, reason from the closed 19 routing + 12 port vocab), `path` (`node(port)->node(port)->...`), `topology` (`local_node(local_port)->remote_node(remote_port)`), all ASCII-only with no blank lines. Format errors zero a question.
- **Thinking mode must be disabled** on both the vLLM server (`--reasoning-parser deepseek_r1` in `deploy/vllm_serve.sh`) and the client (`chat_template_kwargs={"enable_thinking": False}` in `qwen_client.py`). Both layers are intentional belt-and-suspenders — do not remove either.

## Directory map

- `telco_data/Track B/` — competition data, simulator, official agent server.
  - `data/Phase_1/test.json` — 50 fault scenarios; comes with `devices_outputs/` static bundles.
  - `data/Phase_2/test.json` — 100 evaluation scenarios; **no static bundles** (different network, names like `Core_SW_01`, `BJHQ_CSR1000V_GW_01`).
  - `devices_outputs/` — Phase 1 only, indexed by `question_number == task.id`. Phase 2 task_ids 1..50 collide numerically but are not Phase 1 bundles.
  - `question_limits_config.json` — per-`question_number` denied (device, command) pairs (Phase 1 device names only).
  - `server.py` — local simulator (port 7860); unzip `devices_outputs.zip` first.
- `work/track_b/` — Python package with the runtime stack (constraint parser, format guard, playbook, anomaly miner, topology, ranker, prompt context, answer validator, oracle runner, XGBoost LTR, eval harness, Qwen client, agent runtime, path/topology solvers).
- `work/run_*.py`, `work/build_*.py` — step drivers that read/write CSV/JSON artifacts in `work/`. All use `ROOT = Path(__file__).resolve().parents[1]`; no hard-coded absolute paths.
- `deploy/` — vLLM serving kit for **a CUDA GPU box** (Apple Silicon is not supported). `vllm_serve.sh`, `smoke_test.sh`, `test_tool_calling.py`, `env.example`, plus its own README with hardware sizing.
- `the-ai-telco-troubleshooting-challenge20260120-9768-ainojt.zip` — past competition archive; 3–5 entries used as few-shot exemplars only (no LoRA, no LTR labels — schema mismatch).

## Pipeline (run in order; each step writes the inputs the next needs)

```
work/build_scenario_manifest.py          # Step 0  → scenario_manifest.csv/json
work/run_anomaly_miner.py                # Step 4  → anomaly_candidates.csv
work/run_topology_build.py               # Step 5  → graph_features.csv
work/run_ranker.py                       # Step 6  → ranked_candidates.csv
work/run_oracle_silver_labels.py         # Step 7  → oracle_silver_labels.csv, oracle_run_traces.jsonl
work/run_xgb_train.py                    # Step 8  → xgb_features.{parquet,csv}, xgb_silver_labels.{parquet,csv}, xgb_summary.json, ranked_candidates_xgb.csv
work/run_prompt_context.py               # Step 9  → prompt_contexts.jsonl
work/run_eval.py                         # Step 11 → eval_report.json (V0..V4 ablation suite)
work/run_agent_demo.py [task_id]         # End-to-end smoke test on one Phase 1 question
work/run_submission.py                   # Batch-run agent over a test.json, write result.csv + eval_detail.jsonl
work/build_phase2_submission.py          # Combine Track A + Track B answers into the organiser's submission template
```

Steps 1–3 of the plan have no driver — Step 1 is the Qwen deployment itself; Steps 2–3 are library modules (`constraint_parser.py`, `playbook.py`) consumed by later steps. Steps 4–5 and parts of 7 are **conditional on `has_static_bundle == True`** in the manifest. Phase 2 scenarios get sentinel rows with `offline_bundle_missing=1`; they must not be filtered out — Step 8's join expects them so feature shape stays uniform.

## Common commands

```bash
# Local Track B simulator (offline dev on Mac)
cd "telco_data/Track B" && python3 -m venv .venv && source .venv/bin/activate \
  && pip install -r requirements.txt && python server.py        # port 7860

# Qwen/vLLM on a CUDA box (NOT supported on Apple Silicon)
cd deploy && python3 -m venv .venv && source .venv/bin/activate \
  && pip install -r requirements.txt && ./vllm_serve.sh         # tees vllm.log
cd deploy && ./smoke_test.sh                                    # /v1/models + chat + tool-call

# Single end-to-end agent run for one task
python3 work/run_agent_demo.py 17

# Batch submission over a Phase 2 test.json
python3 work/run_submission.py --input-json telco_data/Track\ B/data/Phase_2/test.json \
  --out-dir work/submission_live

# Eval harness (V0..V4 ablation)
python3 work/run_eval.py

# Tests: no pytest config — each test file has main()/SystemExit
python3 work/track_b/test_qwen_agent.py        # single test
for f in work/track_b/test_*.py; do python3 "$f"; done   # full suite
```

Tests import `track_b` as a package; they prepend `Path(__file__).resolve().parents[1]` to `sys.path`. Match that style when adding coverage.

## Stub-vs-real backend switching

`run_agent_demo.py`, `run_submission.py`, and `run_oracle_silver_labels.py` all run identically against either:
- **Stub mode** (default): when `OPENAI_BASE_URL` is unset, `QwenClient` uses a programmable stub policy and the agent tool dispatcher is patched out. No real API calls. Use this for offline dev on the Mac.
- **Real mode**: with `OPENAI_BASE_URL` set, the same code path hits a live vLLM endpoint and the real Agent Tool Server.

The stub policy returns deterministic ChatResponses; tests in `work/track_b/test_*.py` rely on this contract — when extending `QwenClient.chat`, preserve the `(messages, tools, temperature, seed)` policy signature.

Pointing the runtime at a live server:
```
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=$VLLM_API_KEY
export QWEN_MODEL=Qwen/Qwen3.5-35B-A3B
export AGENT_TOOL_SERVER_URL=http://localhost:7860/api/agent/execute   # local sim
# or the production ELB / HK ECS — see deploy/env.example for both URLs
```

`run_submission.py` also reads `AGENT_MODEL_URL`, `AGENT_API_KEY`, `AGENT_MODEL_NAME`, and `ZINDI_BEARER_TOKEN_B1/B2` from a top-level `.env` as fallbacks.

## Architecture (the stuff that requires reading multiple files)

The pipeline splits into **offline artifact generation** and a **live per-scenario agent loop**.

1. **Manifest** (`work/build_scenario_manifest.py`): single source of truth for `question_number == task.id`, whether a static bundle exists, and which (device, command) pairs are denied. Phase 2 rows carry `offline_bundle_missing=1` sentinels rather than fabricated evidence.
2. **Offline features**: `run_topology_build.py` builds a NetworkX graph from `devices_outputs/` and emits scenario-relative graph features; `run_anomaly_miner.py` extracts anomaly priors. Both gate on `has_static_bundle`.
3. **Ranker** (`run_ranker.py` → `run_xgb_train.py`): merges manifest + graph features + anomalies + parsed constraints into `ranked_candidates.csv`, then XGBoost adds `calibrated_score` and `uncertainty` producing `ranked_candidates_xgb.csv`. XGBoost is **gated**: `xgb_summary.json` records the promotion-gate result, and the variant only ships if it beats the deterministic ranker on reserved-fold NDCG@5 *and* on calls/correct in the Phase-2-shaped slice. Keep a clean fallback path that disables XGBoost at startup if the artifact or wheel is missing.
4. **Live runtime** (`work/track_b/agent_runtime.py`): `run_scenario()` is the entry point. It parses constraints, extracts allowed fault vocabulary, builds the prompt context, sends to Qwen, executes any tool calls against the Agent Tool Server, then runs the answer validator. `run_scenario` dispatches by task family — `_run_path_scenario` and `_run_topology_scenario` are separate code paths from the fault branch; `format_guard.py`, `task_classifier.py`, and the ranker each support all three families, so when changing family behaviour mirror the change across all four sites.
5. **Validator** (`answer_validator.validate_answer`) drives the agent loop with four decisions: `accept`, `reemit_format`, `reemit_constraint`, `fetch_evidence` (exactly one targeted follow-up — never broad re-exploration). Edits here directly affect call budget and accuracy. The follow-up command path uses `_PREFIX_TO_TOOL` in `agent_runtime.py` to keep traces tagged consistently with the four tool specs.
6. **Two backends**: `qwen_client.py` talks to the OpenAI-compatible vLLM endpoint; `agent_tools.py` talks to the simulator's `/api/agent/execute`. The four tool names (`infra_maintenance`, `l2_link`, `l3_route`, `adv_tunnel`) mirror the openclaw skill taxonomy but **all route to the same endpoint** — splitting them only narrows the description Qwen sees per slot, which empirically reduces wrong-family command picks.

## Conventions worth knowing before editing

- **Features must be scenario-relative.** XGBoost trains on within-scenario rank percentiles or z-scores; absolute graph metrics and node-name features are banned because Phase 3 uses a different network. The `_norm` columns in the ranker output are the contract.
- **Denied (device, command) pairs** from `question_limits_config.json` flow through graph features, prompt context, and the validator. Preserve that thread when changing ranking, prompting, or follow-up logic — hard-prune, never penalty-rank.
- **Missing offline evidence is explicit**, not fabricated: sentinel rows with `offline_bundle_missing=1`. Do not invent fallback data for Phase 2 scenarios.
- **Vendor coverage**: default commands and playbooks are Huawei-style first, but the local simulator accepts Cisco and H3C equivalents through regex whitelists in `server.py`.
- **`TRACK_UPDATES.md` is operational, not reference**: its Track B output rules are embedded in the system prompt and its Phase 2 device list is copied into `prompt_context.py` to suppress hallucinated device names.

## Track B output format clarifications (from `TRACK_UPDATES.md`)

- Topology-link answers: port names follow the interface name in `display current-configuration`; if unavailable, use `display interface brief`. Strip trailing bandwidth/rate metadata.
- Path answers: include all nodes, including L2 path nodes. Multi-path answers go on separate lines.
- Fault-cause answers: pick the *most specific and closest* cause from the closed vocabulary.
