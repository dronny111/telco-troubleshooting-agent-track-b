#!/usr/bin/env bash
# Run the deployment smoke test against a live vLLM server.
#
# Loads `env` if present (you can `cp env.example env` and tweak), then
# runs `test_tool_calling.py` which validates:
#   - /v1/models lists Qwen3.5-35B-A3B
#   - plain chat returns text
#   - tool_call round trip (function-call schema)
#   - thinking-mode disabled (no <think> blocks leak into responses)

set -euo pipefail

if [[ -f "$(dirname "$0")/env" ]]; then
    # shellcheck disable=SC1091
    source "$(dirname "$0")/env"
fi

: "${OPENAI_BASE_URL:=http://localhost:8000/v1}"
: "${OPENAI_API_KEY:=local-dev-key}"
: "${QWEN_MODEL:=Qwen/Qwen3.5-35B-A3B}"
export OPENAI_BASE_URL OPENAI_API_KEY QWEN_MODEL

echo "==> probing $OPENAI_BASE_URL/models"
HTTP_CODE=$(curl -s -o /tmp/vllm-models.json -w "%{http_code}" \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    "${OPENAI_BASE_URL%/}/models" || true)
if [[ "$HTTP_CODE" != "200" ]]; then
    echo "FAIL: GET /v1/models returned $HTTP_CODE"
    cat /tmp/vllm-models.json 2>/dev/null || true
    exit 1
fi
echo "    /v1/models reachable"

echo "==> running Python tool-call smoke test"
python3 "$(dirname "$0")/test_tool_calling.py"
