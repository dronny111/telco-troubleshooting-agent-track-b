# Phase 2 Learnings — Handoff for the Next Iteration

Deadline-driven knowledge dump after building submissions from `_07` (0.07385 LB) up through `v10` (0.090 LB, measured) and `v13a` (built but unshipped at the time of writing). Optimised for whoever picks this up next — assume you have the repo but not the conversational context.

---

## 1. The LB metric (CONFIRMED by submitted scores)

```
LB = mean(Track A IoU, Track B Accuracy)
```

Verified against v10 submission breakdown:
- v10 LB = 0.090016666
- v10 Track A IoU = 0.0700333
- v10 Track B Accuracy = 0.11
- (0.0700333 + 0.11) / 2 = 0.0900166665 ✓

**Track B is Accuracy, NOT F1.** This is the single most important fact and the one I was wrong about for most of the session. Multi-line emissions for "recall" do not help proportionally under Accuracy; if anything they make precision worse because the metric treats partial answers strictly.

**Track A IoU is harsh under cardinality mismatch**. v10 used Track A with avg cardinality ~6 (union of 5 sources, capped at 6 options per scenario), got TA = 0.07. The math suggests GT cardinality is small (avg 1–2 options): a 6-option prediction with 1 right option against 1-option GT gives IoU = 1/6 = 0.167, vs IoU = 1.0 if a single-pick is right. Big multi-pick predictions are punished hard.

**Both observations imply**: precision-focused, conservative strategies dominate. Submissions ranked by measured LB:

| File | LB | Notes |
|---|---|---|
| `past_subs/results_07.csv` | **0.07385** | Baseline. Track A = 296 single / 18 dbl / 4 tri / 182 quad. Track B = 28× missing-static-route + 26× security-policy + a few targeted |
| `telco_data/submission/Phase_2/results_heuristic_v10.csv` | **0.090016666** | Aggressive Track A union + multi-line Track B. +0.016 lift |
| `telco_data/submission/Phase_2/results_fresh_v13a.csv` | unmeasured | Built from fresh Qwen3.5-35B-A3B runs (Track A main.py + Track B run_submission.py). Hot candidate to test next |
| `past_subs/results_frozen_20260514_1025.csv` | 0.03123 | Earlier baseline, too many wrong fault emissions hurt precision |

---

## 2. Pipeline shape (load-bearing files only)

### Submission template / shape
- `telco_data/submission/Phase_2/submission_example.csv` — template with 600 rows (500 Track A scenario IDs then 100 Track B scenario IDs) and columns `ID, Track A, Track B`. ANY submission must match this row order byte-for-byte on `ID`.
- `work/build_phase2_submission.py` — assembles a template-shape submission from multiple Track A and Track B sources. Uses a consensus model (≥2 sources agree) with fallback to `public07` (= `_07`).

### Track A (5G drive-test multi-choice, 500 questions)
- **Test JSON**: `telco_data/Track A/data/Phase_2/test.json` — task description + C1-C30 options, but `data` field is mostly *placeholders* like "Please use the API to access the cloud server to retrieve data". You cannot do pure-text inference; the actual radio data must be fetched via tool calls.
- **Inference driver**: `telco_data/Track A/main.py` — orchestrates Qwen + tool calls against `https://trackA.organizer.example/no` (production Track A server, auth via `ZINDI_BEARER_TOKEN_A` in `.env`). Default flags: `--num_attempts 1 --radio_graph_prior_top_k 8 --max_tokens 16000`. Uses `openai` package (installed via `pip3 install --break-system-packages openai`).
- **Existing Qwen Track A runs** (all from May 14–18, all Qwen3.5-35B-A3B via OpenRouter):
  - `results_phase2_fewshot/result.csv` — fewshot prompt
  - `results_phase2_fewshot_graph/result.csv` — + graph context (most _07-aligned: 282 exact / 218 superset / 0 subset)
  - `results_phase2_full/result.csv` — free-mode
  - `results_phase2_full/result_hybrid_full.csv` — curated hybrid
  - `results_phase2_radio_prior_full/result.csv` — with radio prior
  - `results_phase2_hybrid_current/result.csv` — latest hybrid pre-fresh
  - `results_phase2_graph_only_full/result.csv` — graph-only (partial, 241 rows)
  - **`results_phase2_fresh_topk16/result.csv`** — NEW fresh run with `--radio_graph_prior_top_k 16`. 500/500. Cardinality {1:224, 2:83, 3:43, 4:150} — closer to _07's shape than other LLM sources. 464/500 diff from _07, 36 exact, 104 superset, 14 subset
- **XGBoost rankers** (`work/track_a_ranker/phase2_xgb_*.csv`) — provide `xgb_answer`, `xgb_top5/8`, `xgb_top1_score`. NONE match _07 picks. Confidence calibration is suspicious (uniform clusters). Untested on LB.

### Track B (network fault/path/topology, 100 questions)
- **Test JSON**: `telco_data/Track B/data/Phase_2/test.json`. Family split: 66 fault / 34 path / 0 topology.
- **Family classifier**: `work/track_b/task_classifier.py` — regex on embedded format-spec phrase. Don't replace with BERT (see learnings.md history if relevant).
- **Inference driver**: `work/run_submission.py`. Multi-turn agent loop with tool calls. Reads `OPENAI_BASE_URL` / `AGENT_TOOL_SERVER_URL` etc. from `.env` (or shell). Supports env-var gating (`SUBMISSION_IDS`, `SUBMISSION_LIMIT`, `SUBMISSION_MAX_TOOL_CALLS=15-30` for path scenarios, `SUBMISSION_WORKERS`).
- **Local simulator**: `cd "telco_data/Track B" && python server.py` (port 7860). **CAUTION**: this is Phase 1's simulator. When the agent calls `display interface brief` against a Phase 2 device, the sim returns Phase 1 device names (`DEV-BL-01`, `DEV-VM-02`, `DEV-PE-01`, `DEV-SP-02`). Qwen faithfully echoes them — DON'T trust path emissions that contain `DEV-*` names; they're hallucinations sourced from the wrong simulator.
- **Production tool server**: `https://trackB.organizer.example/api/agent/execute` with bearer token from `.env`. Authentic Phase 2 device responses. Burns Zindi quota.
- **Existing Track B sources**:
  - `_07` baseline (in `past_subs/results_07.csv`) — Track B emissions
  - `work/submission_full_b/result.csv` — older comprehensive run (path emissions are byte-identical to _07's path)
  - `work/submission_vm_b/result.csv` — IDENTICAL to _07 on Track B; no new signal
  - `work/submission_live_v2_*` — older runs; merged version has BLANK paths (a 34-row free win for any submission that fills them); `_07` already incorporates the path fills
  - **`work/submission_fresh_b/result.csv`** — NEW fresh inference via run_submission.py + OpenRouter Qwen + local sim. 100 unique qids (csv has 100 rows; `wc -l` may show more due to newlines inside quoted multi-line fields). 55 fault filled, 11 fault blank, 20 path filled (14 Phase 2-style usable, 6 Phase 1-style trash), 14 path blank.

---

## 3. What worked vs. what didn't

### Worked
- **Replacing `_07`'s generic `missing static route` fallbacks** (34/66 fault questions) with cluster-targeted picks. This is the core insight of v7→v10 and gave +0.016 LB measured.
- **Path emissions in `_07`** (already there via `submission_full_b/`'s answers). They contribute to TB Accuracy.
- **Running `telco_data/Track A/main.py`** with non-default `--radio_graph_prior_top_k 16` gave a fresh Track A with _07-shaped cardinality (mostly singles) and 67% different picks. Untested LB-wise but worth trying.

### Didn't work (and don't repeat)
- **Track B multi-line emissions for "recall"** under the Accuracy metric. v10 used 3-line emissions on 39 fault questions; lift over `_07` was small (+0.016 total, mostly Track A).
- **Track A multi-source UNION strategies** at high cardinality (avg 6). v10 expansion only got TA = 0.07. Single-pick or low-cardinality strategies likely dominate.
- **Local simulator for Track B path scenarios**. The agent emits Phase 1 device names that won't match Phase 2 GT. Either route paths through the production server, or fall back to `_07`'s existing path emissions.
- **OpenRouter `qwen/qwen3.5-35b-a3b` without `max_tokens ≥ 16k`**. The variant served via OpenRouter→DeepInfra ignores all thinking-disable flags (`enable_thinking=false`, `reasoning.exclude=true`, `include_reasoning=false`). At 8k tokens the model burns its budget on hidden reasoning and returns `content=null`. Solution applied: bump max_tokens AND patched `work/track_b/qwen_client.py` `_parse_openai_response` to extract from `reasoning_details[]` when `content` is null.
- **Heuristic LB estimates from F1-based priors**. My estimates were off by 3–4× because the metric is Accuracy not F1. Future estimates: discount any prior-driven projection accordingly, calibrate against `_07 = 0.07385` and `v10 = 0.090`.

### Mixed
- **Fresh Track B fault emissions**: 53 of 55 successful fault answers picked `IP address prefix list missing corresponding user source IP address` on FW_01 (96% concentration). Either Qwen-with-simulator converged on a real config issue OR mode-collapsed. The v13a submission is the test of this hypothesis. If 25–30% of those 53 are LB-correct, TB jumps from 0.11 to ~0.25.
- **Fresh Track A**: distinct from existing artifacts AND from `_07`. Could be much better or much worse. v13a tests this.

---

## 4. The v13a hot candidate (built but unshipped)

`telco_data/submission/Phase_2/results_fresh_v13a.csv`. 600 rows, template-shape valid, all 100 Track B rows format-validated.

Composition:
| Component | Source | Count |
|---|---|---|
| Track A | fresh Qwen (`results_phase2_fresh_topk16/`) | 500/500 |
| Track B fault | fresh Qwen (`submission_fresh_b/`) where valid | 55/66 |
| Track B fault fallback | `_07` baseline | 11/66 (where fresh was blank) |
| Track B path | fresh Qwen where Phase-2-style + valid | 14/34 |
| Track B path fallback | `_07` baseline | 20/34 (where fresh was blank or used Phase-1 names) |

Honest LB estimate range: **0.075–0.20**, midpoint ~0.13. Reach high end depends on the IP-prefix-list hypothesis landing.

Variants NOT YET BUILT (worth considering):
- **v13b** — `_07` Track A + fresh Track B (hedge: keeps proven Track A, tests Track B alone)
- **v13c** — fresh Track A + `_07` Track B (hedge: tests Track A alone)
- **v14** — XGBoost-confidence-gated Track A (use `xgb_top1_score` > τ for single picks, else `_07`)

---

## 5. Reproducing the inference (key commands)

Environment expected:
```bash
# .env at repo root has:
AGENT_MODEL_URL=https://openrouter.ai/api/v1
AGENT_API_KEY=<openrouter key>
AGENT_MODEL_NAME=qwen/qwen3.5-35b-a3b
ZINDI_BEARER_TOKEN_A=<zindi track A token>
ZINDI_BEARER_TOKEN_B1=<zindi track B token>
ZINDI_BEARER_TOKEN_B2=<zindi track B token, second>
TRACK_A_SERVER_URL=https://trackA.organizer.example/no
```

Local simulator (Phase 1 — only useful for fault, NOT path):
```bash
cd "telco_data/Track B" && python server.py   # binds :7860
```

Track A fresh inference (~3 hours for 500 scenarios at workers=4):
```bash
cd "telco_data/Track A" && \
  set -a && source ../../.env && set +a && export OPENAI_API_KEY="$AGENT_API_KEY" && \
  python3 main.py \
    --server_url "https://trackA.organizer.example/no" \
    --max_samples 500 --num_workers 4 \
    --save_dir ./results_phase2_fresh_topk16 \
    --num_attempts 1 --radio_graph_prior_top_k 16 \
    --save_freq 5 --resume
```

Track B fresh inference against PRODUCTION Phase 2 server (use this — local sim is Phase 1):
```bash
set -a && source .env && set +a && \
  export OPENAI_BASE_URL="$AGENT_MODEL_URL" OPENAI_API_KEY="$AGENT_API_KEY" \
         QWEN_MODEL="$AGENT_MODEL_NAME" \
         AGENT_TOOL_SERVER_URL="https://trackB.organizer.example/api/agent/execute" \
         AGENT_TOOL_SERVER_TOKEN="$ZINDI_BEARER_TOKEN_B1" \
         SUBMISSION_OUT_DIR="work/submission_phase2_prod_b" \
         SUBMISSION_WORKERS=2 SUBMISSION_MAX_TOOL_CALLS=30 SUBMISSION_MAX_ITERATIONS=10 && \
  python3 work/run_submission.py \
    --input-json "telco_data/Track B/data/Phase_2/test.json"
```

Build final submission combining fresh + fallback:
```bash
# See the v13a build script logic in the conversation history; key steps:
# 1. Read fresh Track A from telco_data/Track A/results_phase2_fresh_topk16/result.csv
# 2. Read fresh Track B from work/submission_fresh_b/result.csv (deduped automatically by csv.DictReader)
# 3. Read template from telco_data/submission/Phase_2/submission_example.csv
# 4. Fall back to past_subs/results_07.csv for blanks/Phase-1-name paths
# 5. Format-validate each Track B emission with work/track_b/format_guard.py
# 6. Write to telco_data/submission/Phase_2/results_fresh_v13a.csv (or v13b/c/d)
```

---

## 6. Hard-won facts that aren't documented elsewhere

1. **The `data` field in Track A test.json is placeholders**. Real radio data only comes via tool calls to the production server. Pure text Qwen calls will pick C16 "Insufficient data" by default.
2. **Path emissions from the local sim are Phase-1-polluted**. The sim returns `DEV-BL-01`, `DEV-VM-02`, `DEV-PE-01`, `DEV-SP-02`, `DEV-CUS-*` etc. — these are Phase 1 device names. Phase 2 uses `Core_SW_01`, `FW_01`, `PE1`, `BJHQ_CSR1000V_GW_01`, etc. Detection rule for "trash this emission": `"DEV-" in pred or "No valid path" in pred`.
3. **`wc -l` lies on the Track B CSV** because path/multi-line fault predictions contain `\n` inside their quoted CSV fields. Always count via `csv.DictReader`: `len(list(csv.DictReader(open(path))))`.
4. **`work/submission_vm_b/result.csv` adds zero new signal** — its Track B is byte-identical to `_07`. Don't waste a consensus slot on it.
5. **`_07`'s Track B `missing static route` × 28** are almost all wrong (generic fallback). Replacing them with cluster-targeted picks is the proven lever (the v7→v10 path).
6. **`SleepDisabled 1`** (set via `sudo pmset -a disablesleep 1`) is the only macOS knob that prevents lid-close sleep without an external display in clamshell. Required for overnight inference. `caffeinate` alone is not enough.
7. **OpenRouter Qwen returns `content=null` with reasoning text only** for many requests. `work/track_b/qwen_client.py` is patched to extract from `reasoning_details[]` when `content` is null; verify this is still in place before relying on Qwen outputs.

---

## 7. Suggested next moves (ranked by expected LB / cost)

1. **Ship v13a** as one of your remaining slots. The IP-prefix-list hypothesis is the single highest-value untested bet (53 fault questions × any non-zero hit rate = TB lift). Worst case: regression to ~0.08, best case: ~0.18–0.22.
2. **Re-run Track B against the PRODUCTION server** (`https://trackB.organizer.example/api/agent/execute` with `ZINDI_BEARER_TOKEN_B1`). The path emissions we got from the local sim were polluted with Phase 1 names. Production server returns Phase 2 device names. Burns Zindi quota but should unlock 20+ usable path emissions.
3. **Build v14 = XGB-confidence-gated Track A**: use `xgb_top1_score > 0.4` from `work/track_a_ranker/phase2_xgb_tuned_short.csv` as a single-pick override; fall back to fresh / _07 union otherwise. Untested but the confidence signal is there.
4. **Build v13b** (= `_07` Track A + fresh Track B) as a hedge submission. If v13a regresses, this attribution-isolates which side of v13a was the culprit.
5. **Investigate the q1 / q2 VRRP scenarios specifically**. Fresh Qwen picks different reasons each time (q1: "interface IP error", q2: "Layer 3 loop"). The team's `build_phase2_submission.py:46` had a manual override to "Layer 3 loop". Worth grounding via the production simulator to determine which is correct.
6. **Don't bother building bigger Track A ensembles**. Measured v10 showed that union expansion only gets +0.03 TA over `_07`. The lever is correctness per pick (try XGBoost confidence-gating or a different Qwen prompt), not breadth.

---

## 8. Files / artifacts worth preserving

- All `results_phase2_*/result.csv` Track A variants
- `past_subs/results_07.csv` — the LB-validated reference
- `telco_data/submission/Phase_2/results_heuristic_v10.csv` — measured at 0.090
- `telco_data/submission/Phase_2/results_fresh_v13a.csv` — built but unshipped
- `work/submission_fresh_b/result.csv` — fresh Track B emissions
- `/tmp/track_a_fresh.log` and `/tmp/track_b_fresh.log` — raw inference logs (may rotate)
- `work/INFERENCE_DONE.txt` — completion summary from the watcher

---

## 9. OpenRouter credit exhausted (as of 2026-05-18)

The OpenRouter account hit its $50 monthly cap during the fresh-inference runs. `GET /auth/key` returns:

```
limit: 50, usage: 50.037, limit_remaining: 0, expires_at: 2026-06-10
```

All further Qwen calls return **HTTP 403 Forbidden** until the user tops up credit or waits for limit reset. This means:
- No more `main.py` Track A runs
- No more `run_submission.py` Track B runs against any tool server
- Existing fresh artifacts in `telco_data/Track A/results_phase2_fresh_topk16/` and `work/submission_fresh_b/` are all the fresh data we'll get without a top-up

If you top up: the **production Track B server re-run** (`AGENT_TOOL_SERVER_URL=https://trackB.organizer.example/api/agent/execute` + `ZINDI_BEARER_TOKEN_B1`) is the single highest-value next inference step, since it would unlock 20+ correctly-named Phase 2 path emissions (vs the local sim's Phase 1-polluted ones).

---

## 10. Open / unresolved

- Why does fresh Qwen pick `IP address prefix list missing` for 53/55 fault questions? Genuine signal or mode collapse?
- What's the actual GT cardinality distribution on Track A? My math hints at avg ~1–2 but it's a guess from one measured submission.
- Is the production Track B simulator's per-question quota generous enough to re-run all 100 questions? `.env` has two bearer tokens (`_B1`, `_B2`) suggesting quota splitting.
- XGBoost rankers (`work/track_a_ranker/phase2_xgb_*.csv`) — untested on LB but their `xgb_top1_score` could power a confidence-gated strategy.
