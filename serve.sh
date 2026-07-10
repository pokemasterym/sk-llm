#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server for GLM-4.5-Air.
# All knobs come from env vars (see Dockerfile defaults). Extra CLI args pass through.
#
#   Basic:   serve.sh
#   Tweak:   MAX_MODEL_LEN=32768 serve.sh --max-num-seqs 8
#
# NOTE: --tool-call-parser / --reasoning-parser are for LibreChat tools + <think>
#       separation. If your vLLM build rejects the parser name, drop those two
#       lines — plain chat still works without them.
set -euo pipefail

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --served-model-name "${SERVED_NAME}" \
  --quantization "${QUANTIZATION}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --host 0.0.0.0 --port "${PORT}" \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-auto-tool-choice --tool-call-parser glm45 \
  --reasoning-parser glm45 \
  "$@"
