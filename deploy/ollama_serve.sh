#!/usr/bin/env bash
set -euo pipefail

# Ollama OpenAI-compatible endpoint for local Track B inference.
# Keep context moderate so local inference remains responsive.
: "${OLLAMA_HOST:=127.0.0.1:11434}"
: "${OLLAMA_CONTEXT_LENGTH:=16000}"

export OLLAMA_HOST
export OLLAMA_CONTEXT_LENGTH

echo "Starting Ollama on ${OLLAMA_HOST} with OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH}"
exec ollama serve
