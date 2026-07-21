# Copilot instructions

## Repository focus

The active implementation work is in `work/track_b/`. Treat the rest of the repository as supporting inputs:

- `telco_data/Track B/` contains the competition data, `question_limits_config.json`, and the local Flask-based Agent Tool Server in `server.py`.
- `work/` contains driver scripts that build offline artifacts and exercise the Track B runtime.
- `deploy/` contains the GPU-side vLLM/Qwen serving kit used by the Track B agent runtime.
- `telco_data/Track A/` is dataset/documentation only in this workspace; there is no parallel Track A code pipeline here.

## Build, test, and run commands

| Purpose | Command |
| --- | --- |
| Start the local Track B simulator | `cd "telco_data/Track B" && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python server.py` |
| Start the Qwen/vLLM server on a CUDA machine | `cd deploy && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && ./vllm_serve.sh` |
| Smoke-test the running vLLM server | `cd deploy && ./smoke_test.sh` |
| Python tool-calling smoke test for vLLM | `cd deploy && python3 test_tool_calling.py` |
| Offline/real agent demo for one Track B task | `python3 work/run_agent_demo.py 17` |
| Build the scenario manifest | `python3 work/build_scenario_manifest.py` |
| Build graph-derived device features | `python3 work/run_topology_build.py` |
| Mine anomaly candidates from static bundles | `python3 work/run_anomaly_miner.py` |
| Build deterministic candidate rankings | `python3 work/run_ranker.py` |
| Train/apply the XGBoost calibration layer | `python3 work/run_xgb_train.py` |
| Run the V0-V4 local eval harness | `python3 work/run_eval.py` |
| Run the full Track B script-based test suite | `for f in work/track_b/test_*.py; do python3 "$f"; done` |
| Run a single Track B test script | `python3 work/track_b/test_qwen_agent.py` |

There is no checked-in lint command or central test runner; tests are standalone Python scripts under `work/track_b/test_*.py`.

## High-level architecture

The Track B pipeline is split between offline artifact generation and the live per-scenario agent loop.

1. `work/build_scenario_manifest.py` is the starting point. It establishes `question_number == task.id`, records whether a local static bundle exists, and records question-level permission restrictions. The local `devices_outputs/` bundles and `question_limits_config.json` are Phase 1 only.
2. `work/run_topology_build.py` and `work/run_anomaly_miner.py` consume Phase 1 static bundles from `telco_data/Track B/devices_outputs/` to create `work/graph_features.csv` and `work/anomaly_candidates.csv`. For Phase 2, they intentionally emit sentinel rows with `offline_bundle_missing=1` instead of inventing evidence.
3. `work/run_ranker.py` merges the manifest, graph features, anomaly priors, and parsed question constraints into `work/ranked_candidates.csv`. `work/run_xgb_train.py` adds `calibrated_score` and `uncertainty`, producing `work/ranked_candidates_xgb.csv` for later prompt-building and validation.
4. `work/track_b/agent_runtime.py` is the live runtime. It parses question constraints, extracts the allowed fault vocabulary, builds a compact prompt context from ranked candidates, sends the prompt to Qwen, executes model-selected tool calls, and then runs the answer validator before accepting or re-emitting the answer.
5. The runtime talks to two separate services: `work/track_b/qwen_client.py` calls the OpenAI-compatible vLLM endpoint from `deploy/`, while `work/track_b/agent_tools.py` calls the Track B simulator endpoint in `telco_data/Track B/server.py`.

## Key conventions

- Most scripts hard-code `ROOT = Path("/Users/ronnypolle/Desktop/telco_itu")` instead of resolving paths relative to the current working directory. Preserve that assumption unless you are deliberately refactoring all affected entry points.
- The current Track B runtime is fault-task-specific. `run_scenario()` rejects non-fault tasks, and the deterministic ranker also skips non-fault scenarios.
- Keep the fault answer schema exact: `node;destination-IP;reason` for routing faults or `node;port;reason` for port faults. The validator expects ASCII-only output with no extra prose.
- Do not remove the double thinking-mode guard. The deploy-side serve script strips `<think>` output, and the client in `qwen_client.py` also sends `chat_template_kwargs={"enable_thinking": False}`.
- The four tool names (`infra_maintenance`, `l2_link`, `l3_route`, `adv_tunnel`) are prompt-shaping categories only. They all route to the same `POST /api/agent/execute` simulator endpoint.
- Default commands and playbooks are Huawei-style first, but the local simulator accepts Cisco and H3C equivalents through regex whitelists in `server.py`.
- `TRACK_UPDATES.md` is not just reference material: its Track B output rules are embedded in the system prompt, and its Phase 2 device list is copied into `prompt_context.py` to suppress hallucinated device names.
- Question-level denied `(device, command)` pairs from `question_limits_config.json` flow through graph features, prompt context, and the validator. Preserve that thread when changing ranking, prompting, or follow-up logic.
- Missing offline evidence is represented explicitly with sentinel rows (`offline_bundle_missing=1`), not with fabricated graph/anomaly data.
- Tests are script-style smoke/behavior checks with `main()` functions and PASS/FAIL output, not `pytest` tests. Match that style when adding coverage next to the existing Track B tests.
