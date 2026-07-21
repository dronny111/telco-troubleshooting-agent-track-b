#!/usr/bin/env bash
# Launch vLLM serving Qwen3.5-35B-A3B with the flags Track B needs.
#
# Reads its config from env vars (see env.example). Override on the
# command line with positional KEY=VALUE pairs:
#
#   ./vllm_serve.sh TENSOR_PARALLEL=2 MAX_MODEL_LEN=4096 QUANTIZATION=fp8
#
# Logs to stdout AND to ./vllm.log. Ctrl-C to stop.

set -euo pipefail

# ---- Defaults (env can override) ------------------------------------------
: "${MODEL:=Qwen/Qwen3.5-35B-A3B}"
: "${VLLM_HOST:=0.0.0.0}"
: "${VLLM_PORT:=8000}"
: "${TENSOR_PARALLEL:=1}"
: "${MAX_MODEL_LEN:=8192}"
: "${DTYPE:=bfloat16}"
: "${QUANTIZATION:=}"
: "${VLLM_API_KEY:=local-dev-key}"
: "${MAX_NUM_SEQS:=16}"
: "${GPU_MEMORY_UTILIZATION:=0.9}"
: "${KV_CACHE_DTYPE:=auto}"      # set to fp8 to halve KV memory
: "${TOOL_CALL_PARSER:=hermes}"  # try `qwen3` on vLLM >=0.10.x

# ---- Parse positional KEY=VALUE overrides ---------------------------------
for arg in "$@"; do
    case "$arg" in
        *=*) export "$arg" ;;
        --help|-h)
            grep -E "^# " "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

# ---- Sanity ---------------------------------------------------------------
if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vllm CLI not found. Activate the venv where you ran 'pip install -r requirements.txt'." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi not found; vLLM will fail without CUDA-visible GPUs." >&2
fi

# ---- Build the command ----------------------------------------------------
ARGS=(
    serve "$MODEL"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --max-model-len "$MAX_MODEL_LEN"
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --api-key "$VLLM_API_KEY"
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
    # Strip <think>...</think> on the server. The client also disables
    # thinking via chat_template_kwargs; both layers are intentional.
    --reasoning-parser deepseek_r1
)

if [[ -n "$QUANTIZATION" ]]; then
    ARGS+=(--quantization "$QUANTIZATION")
fi
if [[ "$KV_CACHE_DTYPE" != "auto" ]]; then
    ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ -n "${HF_HUB_OFFLINE:-}" ]]; then
    ARGS+=(--load-format auto)
fi
if [[ -n "${DOWNLOAD_DIR:-}" ]]; then
    ARGS+=(--download-dir "$DOWNLOAD_DIR")
fi

echo "==> launching vLLM"
echo "    model:               $MODEL"
echo "    host:port:           $VLLM_HOST:$VLLM_PORT"
echo "    tensor parallel:     $TENSOR_PARALLEL"
echo "    max model len:       $MAX_MODEL_LEN"
echo "    dtype:               $DTYPE"
echo "    quantization:        ${QUANTIZATION:-none}"
echo "    kv-cache dtype:      $KV_CACHE_DTYPE"
echo "    tool-call parser:    $TOOL_CALL_PARSER"
echo "    GPU memory util:     $GPU_MEMORY_UTILIZATION"
echo "    max-num-seqs:        $MAX_NUM_SEQS"
echo "    cmd: vllm ${ARGS[*]}"
echo

# Tee both stdout and stderr so we keep a local log for post-mortems.
exec stdbuf -oL -eL vllm "${ARGS[@]}" 2>&1 | tee vllm.log
