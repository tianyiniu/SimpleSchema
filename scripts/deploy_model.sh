#!/usr/bin/env bash
# Deploy a vLLM server for one of the supported orchestrator/agent models.
#
# Usage:
#   ./deploy_qwen.sh -m <model_key> -d <cuda_devices> [-p <port>] [extra vllm args...]
#
#   -m <model_key>     short name (see table below)
#   -d <cuda_devices>  e.g. "2" or "0,1,2,3" (TP size = #devices)
#   -p <port>          override; defaults to the model's class port
#   -h                 show this help
#
# Default ports (match config.toml):
#   orchestrator-class models -> 7471
#   agent-class models        -> 7472
#   judge-class models        -> 7473
#
# Supported models:
#   orchestrator-class (default port 7471):
#     qwen3-4b      Qwen/Qwen3-4B
#     gemma-4-e2b   google/gemma-4-E2B
#   agent-class (default port 7472):
#     qwen3-14b              Qwen/Qwen3-14B
#     nemotron-30b           nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
#     gemma-4-31b            google/gemma-4-31B
#     gpt-oss-20b            openai/gpt-oss-20b
#   judge-class (default port 7473):
#     nemotron-super-120b    nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
#     gpt-oss-120b           openai/gpt-oss-120b
#
# Env overrides:
#   MAX_MODEL_LEN          default 32768
#   MAX_NUM_SEQS           default 64   (>= [llm.agent].max_workers in config.toml)
#   MAX_NUM_BATCHED_TOKENS default 16384
#   DOWNLOAD_DIR           default /nas-ssd2/tianyin4/cache/pretrained_models
#
# Examples:
#   ./deploy_qwen.sh -m qwen3-4b -d 0
#   ./deploy_qwen.sh -m qwen3-14b -d 1,2
#   ./deploy_qwen.sh -m nemotron-30b -d 0,1,2,3 -p 6668 --gpu-memory-utilization 0.92

set -euo pipefail

usage() { sed -n '2,32p' "$0"; }

MODEL_KEY=""
CUDA_DEVICES=""
PORT_OVERRIDE=""

while getopts ":m:d:p:h" opt; do
  case "$opt" in
    m) MODEL_KEY="$OPTARG" ;;
    d) CUDA_DEVICES="$OPTARG" ;;
    p) PORT_OVERRIDE="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown flag: -$OPTARG" >&2; usage; exit 1 ;;
    :)  echo "Flag -$OPTARG requires an argument" >&2; usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

if [[ -z "$MODEL_KEY" || -z "$CUDA_DEVICES" ]]; then
  echo "Both -m <model_key> and -d <cuda_devices> are required." >&2
  usage
  exit 1
fi

# --- Model registry ----------------------------------------------------------
# Each entry sets MODEL, TOOL_PARSER, REASONING_PARSER, DEFAULT_PORT, and
# optionally appends to EXTRA_VLLM_FLAGS for model-specific tuning (quant
# dtype, KV cache, mamba state, chunked prefill, plugin parsers, ...).
# REASONING_PARSER is empty for models that do not emit a separable reasoning
# trace; in that case the --reasoning-parser flag is omitted below.
TOOL_PARSER=""
REASONING_PARSER=""
EXTRA_VLLM_FLAGS=()
case "$MODEL_KEY" in
  qwen3-4b)
    MODEL="Qwen/Qwen3-4B"
    TOOL_PARSER="hermes"
    # Qwen3 emits <think>...</think> when enable_thinking=true; vLLM ships a
    # dedicated qwen3 reasoning parser for that format.
    REASONING_PARSER="qwen3"
    DEFAULT_PORT=7471
    ;;
  gemma-4-e2b)
    MODEL="google/gemma-4-E2B-it"
    TOOL_PARSER="gemma4"
    # Gemma 4 chat templates do not emit a separable reasoning channel.
    REASONING_PARSER="gemma4"
    DEFAULT_PORT=7471
    ;;
  qwen3-14b)
    MODEL="Qwen/Qwen3-14B"
    TOOL_PARSER="hermes"
    # Same <think>...</think> convention as qwen3-4b.
    REASONING_PARSER="qwen3"
    DEFAULT_PORT=7472
    ;;
  gemma-4-31b)
    MODEL="google/gemma-4-31B-it"
    TOOL_PARSER="gemma4"
    REASONING_PARSER="gemma4"
    DEFAULT_PORT=7472
    ;;
  nemotron-30b)
    # Built-in nemotron_v3 reasoning parser handles the Nemotron-3 family
    # (Nano and Super) trace format. The HF model card's references to
    # nano_v3 / super_v3 are docker-pipeline plugins; for stock vLLM the
    # unified built-in is correct.
    MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    TOOL_PARSER="qwen3_coder"
    REASONING_PARSER="nemotron_v3"
    DEFAULT_PORT=7472
    ;;
  nemotron-super-120b)
    MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    TOOL_PARSER="qwen3_coder"
    REASONING_PARSER="nemotron_v3"
    DEFAULT_PORT=7473
    EXTRA_VLLM_FLAGS+=(
      --dtype auto
      --kv-cache-dtype fp8
      --mamba-ssm-cache-dtype float16
      --enable-chunked-prefill
      --max-cudagraph-capture-size 128
      --async-scheduling
    )
    ;;
  gpt-oss-20b)
    MODEL="openai/gpt-oss-20b"
    TOOL_PARSER="openai"
    REASONING_PARSER="openai_gptoss"
    DEFAULT_PORT=7472
    ;;
  gpt-oss-120b)
    MODEL="openai/gpt-oss-120b"
    TOOL_PARSER="openai"
    REASONING_PARSER="openai_gptoss"
    DEFAULT_PORT=7473
    ;;
  *)
    echo "Unknown model_key: $MODEL_KEY" >&2
    exit 1
    ;;
esac

PORT="${PORT_OVERRIDE:-$DEFAULT_PORT}"
TP_SIZE=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Concurrency knobs: must accommodate the one-shot driver's peak in-flight
# count = [oneshot].task_workers * [oneshot].eval_workers * max_personas_per_round
# (default 2 * 4 * 5 = 40). MAX_NUM_SEQS=64 gives headroom; bump to 96+ if
# you raise task_workers or eval_workers in config.toml.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/nas-ssd2/tianyin4/cache/pretrained_models}"

echo "[deploy] model=$MODEL port=$PORT tp=$TP_SIZE max_len=$MAX_MODEL_LEN max_seqs=$MAX_NUM_SEQS max_batched_tokens=$MAX_NUM_BATCHED_TOKENS cuda=$CUDA_DEVICES tool_parser=$TOOL_PARSER reasoning_parser=${REASONING_PARSER:-<none>}"

REASONING_ARGS=()
if [[ -n "$REASONING_PARSER" ]]; then
  REASONING_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" VLLM_USE_V1=1 vllm serve "$MODEL" \
  --trust-remote-code \
  --host localhost \
  --port "$PORT" \
  --download-dir "$DOWNLOAD_DIR" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER" \
  "${REASONING_ARGS[@]}" \
  "${EXTRA_VLLM_FLAGS[@]}" \
  "${EXTRA_ARGS[@]}"
