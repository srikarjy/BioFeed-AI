#!/usr/bin/env bash
set -euo pipefail

# Launches vLLM's OpenAI-compatible server for the anomaly-explanation
# feature. Run on a rented GPU instance (A10G/L4 class, ~24GB VRAM covers a
# 7B AWQ model plus KV cache headroom at the concurrency below).
#
# Engine args, and why:
#   --quantization awq         Checkpoint is already AWQ-quantized; this tells
#                               vLLM how to load it, it does not requantize.
#   --max-model-len             Explanation prompts + output are short; capping
#                               this bounds KV cache memory per sequence.
#   --gpu-memory-utilization    Fraction of GPU memory vLLM may claim for
#                               weights + KV cache pool.
#   --max-num-seqs              Ceiling on concurrently-batched sequences.
#                               Continuous batching itself is vLLM's default
#                               scheduling behavior (PagedAttention) -- no flag
#                               turns it on, this only bounds how wide it gets.
#   --port                      Must match LLM_VLLM_BASE_URL in the backend's
#                               .env (http://<this-host>:<port>/v1).

MODEL_ID="${VLLM_MODEL:-TheBloke/Mistral-7B-Instruct-v0.2-AWQ}"
PORT="${VLLM_PORT:-8000}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --quantization awq \
  --dtype auto \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --port "${PORT}"
