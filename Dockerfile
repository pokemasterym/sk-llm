# GLM-4.5-Air vLLM image for Flightbase (Jonathan)
# - Base: official vLLM OpenAI-compatible server (CUDA, H100 sm_90 OK)
# - Adds dev tooling so you can "get inside and develop" per Flightbase 개발도구
# - Defaults target GLM-4.5-Air AWQ on a single H100 80GB
#
# GLM-4.5 is MoE and needs a recent vLLM. Pin VLLM_TAG to a known-good tag
# if `latest` ever regresses:  --build-arg VLLM_TAG=v0.x.y
ARG VLLM_TAG=latest
FROM vllm/vllm-openai:${VLLM_TAG}

LABEL org.opencontainers.image.title="glm-vllm" \
      org.opencontainers.image.description="vLLM OpenAI server for GLM-4.5-Air (AWQ) on H100"

# --- dev conveniences (shell-in and iterate) ---
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl vim tmux jq htop ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# huggingface-cli for pre-downloading weights (useful on closed networks)
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# Flightbase convention: do work under /root/project (survives across sessions)
WORKDIR /root/project

# Serving defaults — override any of these via env at deploy time (워커 환경변수)
ENV MODEL_ID=QuantTrio/GLM-4.5-Air-AWQ-FP16Mix \
    SERVED_NAME=glm-4.5-air \
    QUANTIZATION=awq_marlin \
    TP_SIZE=1 \
    MAX_MODEL_LEN=65536 \
    GPU_MEM_UTIL=0.92 \
    PORT=8000 \
    HF_HOME=/root/project/.hf

COPY serve.sh /usr/local/bin/serve.sh
RUN chmod +x /usr/local/bin/serve.sh

# The base image ENTRYPOINTs straight into the API server. Clear it so this image
# can either drop into a shell (development) or run `serve.sh` (deployment).
ENTRYPOINT []
CMD ["/bin/bash"]
