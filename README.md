# Telco Troubleshooting Agent — Track B (IP Networking)

Solution stack for **Track B** of the ITU *Telco Troubleshooting
and Optimization Agentic Challenge* (Phase 2). The agent diagnoses faults on a
40-node multi-vendor campus network by driving **Qwen3.5-35B-A3B** (the
competition's mandatory base model) through a tool-calling loop against the
organiser's Agent Tool Server.

This repository contains **code and design notes only**. The competition dataset
is **not** included — the organiser restricts its redistribution (see below).

## What's here

- `work/track_b/` — runtime stack: constraint parser, format guard, playbook,
  anomaly miner, topology graph, ranker, prompt context, answer validator,
  oracle runner, XGBoost LTR, eval harness, Qwen client, agent runtime, and
  path/topology solvers.
- `work/run_*.py`, `work/build_*.py` — step drivers for the offline pipeline
  (manifest → anomalies → graph features → ranker → silver labels → XGB →
  prompt context → eval).
- `work/track_a_ranker/` — Track A option-ranking (XGBoost) utilities.
- `deploy/` — vLLM serving kit for a CUDA GPU box (`vllm_serve.sh`,
  `smoke_test.sh`, `env.example`).
- `CLAUDE.md`, `plan-phase2HybridGraphStrategy.prompt.md` — architecture and
  strategy.
- `SOTA_RETROSPECTIVE.md` — post-mortem: why the result was capability-bound,
  and what the state-of-the-art move would have been.

## Not included (by design)

- **`telco_data/`** — the ITU benchmark dataset. Excluded per the organiser's
  redistribution restriction. Access it through the official competition
  channels.
- **`.env` and all credentials** — API keys and bearer tokens never leave the
  local machine. See `deploy/env.example` for the expected variable names.
- **Generated artifacts** (`*.csv`, `*.json`, `*.pkl`, submission files, logs) —
  reproducible by running the pipeline against the dataset.

Production server hostnames have been replaced with placeholders
(`trackA.organizer.example`, `trackB.organizer.example`); supply the real
endpoints via environment variables.

## Running

The pipeline steps are documented in [CLAUDE.md](CLAUDE.md). In brief: point the
runtime at a Qwen/vLLM endpoint and the Agent Tool Server via environment
variables, then run the `work/run_*.py` drivers in order. A stub backend runs
the same code paths offline when `OPENAI_BASE_URL` is unset.

## License

Code is the author's own work. The competition dataset is governed by the ITU
challenge terms and is not part of this repository.
