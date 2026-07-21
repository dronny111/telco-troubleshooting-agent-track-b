# Qwen3.5-35B-A3B on vLLM — Deployment Kit

This is the GPU-side serving setup for the Track B Phase 2 agent. The
**local Mac is not a target** — vLLM does not support Apple Silicon. Run
these scripts on a CUDA GPU machine (your own box, a cloud VM, or the
Huawei Cloud instance the organiser provides for Phase 3).

---

## Model resolution

Hugging Face: **`Qwen/Qwen3.5-35B-A3B`** (Apache 2.0).

| Property | Value |
|---|---|
| Total params | 35 B |
| Activated MoE params | 3 B |
| Architecture | Hybrid Gated DeltaNet + sparse MoE (256 experts, 8 routed + 1 shared) |
| Native context | 262 144 tokens (extensible to ~1 M) |
| Default mode | `thinking` (emits `<think>...</think>` blocks) — must be disabled for Track B's strict output schema |

This is the model the competition rules mandate; do not substitute.

---

## Hardware sizing

Track B prompts are <2 K tokens each (verified in Step 9), so we cap
`--max-model-len` aggressively to shrink the KV cache. With that cap the
model fits on a single high-end GPU.

| GPU | Setup | Notes |
|---|---|---|
| 1× H100 80GB / H200 141GB | bf16, `--max-model-len 8192` | Comfortable, recommended for Phase 2 testing. |
| 1× A100 80GB | bf16, `--max-model-len 4096` | Tight but viable. fp8 KV cache helps. |
| 1× A100 40GB | **fp8 quant**, `--max-model-len 4096` | Use a community fp8 / AWQ checkpoint. |
| 2× A100 40GB / 2× A6000 48GB | bf16 + `--tensor-parallel-size 2` | Most accessible multi-GPU option. |
| 4× L40S 48GB | bf16 + `--tensor-parallel-size 4` | Cloud-friendly. |

For Phase 3 the organiser deploys the model on Huawei Cloud GPUs you
don't manage; this kit is for **local Phase 2 inference** and silver-label
generation (Step 7).

---

## Quick start

```bash
cd /path/to/telco_itu/deploy

# 1. Install vLLM (recommended: a fresh venv on the GPU box)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Launch the server (single-GPU bf16 default)
./vllm_serve.sh

# 3. From any other shell on the same machine, smoke test
./smoke_test.sh
```

If everything passes, point the agent runtime at the server:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
export QWEN_MODEL=Qwen/Qwen3.5-35B-A3B
python3 ../work/run_agent_demo.py 17
```

---

## Files

| File | Purpose |
|---|---|
| `vllm_serve.sh` | Main launch script. Arg-driven: GPU count, quantisation, max-len, dtype. |
| `ollama_serve.sh` | Local Ollama launcher with `OLLAMA_CONTEXT_LENGTH=64000`. |
| `requirements.txt` | Pinned vLLM + smoke-test deps. |
| `env.example` | Environment template for both the server and the agent client. |
| `smoke_test.sh` | End-to-end check: `/v1/models` → simple chat → tool-call round-trip. |
| `test_tool_calling.py` | Python tool-calling smoke test (the part the agent runtime relies on). |

---

## Critical configuration choices

### 1. Disable the thinking mode

Qwen3.5 emits `<think>...</think>` by default. The Track B answer schema
(`node;ip-or-port;reason`) does not allow extra prose, and the format
guard rejects anything wrapped in non-vocabulary text. The serve script
sets `--reasoning-parser deepseek_r1` (which strips `<think>` blocks
on the server side) **and** the agent runtime sets

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

via the OpenAI request payload. Both layers are intentional belt-and-
suspenders — do not remove either.

### 2. Tool calling

vLLM exposes Qwen3-series tool calling via OpenAI's function-call schema.
The serve script enables it with:

```
--enable-auto-tool-choice
--tool-call-parser hermes
```

`hermes` is the recommended parser for Qwen3-series chat templates
because Qwen3 follows Hermes-2-Pro tool-call conventions. If you see
malformed `tool_calls` in responses, switch to `--tool-call-parser
qwen3` once your vLLM version supports it (≥0.10.x).

### 3. Max model length

Default is `8192` here. We never go above ~2 K total prompt tokens
(per Step 9 measurements: median 539, max 591 in prompt + ~1 K headroom
for tool-call round-trips). Setting `--max-model-len 8192` shrinks the
KV cache by ~32× compared to the 262 K default, freeing roughly 30–40 GB
of VRAM.

### 4. Concurrency

Phase 2 has **no daily API quota** but the production simulator caps at
**1 concurrent request per token** (Phase_2/README.md). On the agent
side that means we run scenarios sequentially anyway. On the vLLM side
we still allow modest batching (`--max-num-seqs 16`) to make local eval
faster.

---

## Operating notes

- **First launch downloads ~70 GB** of weights from HF. Pre-download to
  a stable disk path and set `--download-dir` to avoid surprise pulls
  during competition runs.
- **GPU memory headroom**: leave at least 4 GB free for activations and
  CUDA workspace. If you see OOM, drop `--max-model-len` first.
- **Restart on weight update**: vLLM doesn't hot-reload. If you swap a
  quantised checkpoint, kill and re-run the serve script.
- **Logging**: the serve script tees stderr to `vllm.log` for post-mortem
  inspection; rotate it between scenarios if you're debugging.

---

## Ollama fallback

If OpenRouter is exhausted and you want to run through Ollama locally:

```bash
cd /path/to/telco_itu
./deploy/ollama_serve.sh
```

The launcher sets:

```bash
OLLAMA_CONTEXT_LENGTH=64000
OLLAMA_HOST=127.0.0.1:11434
```

Point the submission runner at Ollama's OpenAI-compatible endpoint:

```bash
export AGENT_MODEL_URL=http://localhost:11434/v1
export AGENT_MODEL_NAME=<your-ollama-model-name>
export AGENT_API_KEY=dummy
export QWEN_MAX_TOKENS=4096
```
