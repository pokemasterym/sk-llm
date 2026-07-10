# glm-vllm — GLM-4.5-Air on vLLM for Flightbase

A vLLM (OpenAI-compatible) image for serving **GLM-4.5-Air (AWQ)** on a single
**H100 80GB**, packaged for the **Acryl Jonathan / Flightbase** MLOps platform and
wired to connect with **LibreChat**.

- MoE 106B / 12B active → speed of a 12B, knowledge of a 100B
- AWQ 4-bit weights (~53–60GB) fit one H100 with room for KV cache
- Native `/v1/chat/completions` (OpenAI-compatible) — LibreChat plugs straight in

## How the pieces fit

```
GitHub push ──► Actions builds image ──► ghcr.io/<owner>/<repo>:latest
                                              │
                        Flightbase → Docker 이미지 생성 → Pull → (paste URL)
                                              │
                     개발도구로 shell 진입해 개발  /  Nexus 배포로 워커 상주
                                              │
                          http://{kong-ip}/glm/v1  ◄── LibreChat custom endpoint
```

## 1. Push to GitHub (build the image)

```bash
git init && git add . && git commit -m "GLM-4.5-Air vLLM image"
gh repo create glm-vllm --private --source=. --push   # or create the repo in the UI
```

The `build-and-push` workflow runs on every push to `main` and publishes:

```
ghcr.io/<owner>/glm-vllm:latest
```

> **Make it pullable by Flightbase.** GHCR packages are **private** by default.
> Either mark the package **Public** (GitHub → your profile → Packages → glm-vllm →
> Package settings → Change visibility), or supply registry credentials to Flightbase
> if its Pull supports auth.

## 2. Flightbase — 도커 이미지 생성

- 생성 방식: **Pull**
- **Pull URL**: `ghcr.io/<owner>/glm-vllm:latest`
- 이름: `glm-vllm`  /  공개 범위: 워크스페이스 (팀 공유) 또는 전체

## 3. Develop / serve

Inside the container (Flightbase 개발도구, or `docker run -it ... bash`):

```bash
serve.sh                      # start the OpenAI server with the baked defaults
# or tweak on the fly:
MAX_MODEL_LEN=32768 serve.sh --max-num-seqs 8
```

Local GPU smoke test:

```bash
HF_TOKEN=hf_xxx docker compose up --build
curl localhost:8000/v1/models
```

## 4. Deploy as a worker (for LibreChat)

Deploy via **Nexus에서 불러오기 (custom container)** — *not* the 배포코드 path, so
vLLM's native OpenAI API is exposed unchanged. Ingress path e.g. `/glm`, **Dynamic
Scaling OFF** (LLM load time makes cold starts fatal). Run command = `serve.sh`.

## 5. LibreChat (add a custom endpoint)

```yaml
endpoints:
  custom:
    - name: "GLM-4.5-Air"
      apiKey: "dummy"
      baseURL: "http://{kong-ip}/glm/v1"
      models:
        default: ["glm-4.5-air"]     # matches SERVED_NAME
        fetch: true
      titleConvo: true
```

## Config (env vars, override at deploy)

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_ID` | `QuantTrio/GLM-4.5-Air-AWQ-FP16Mix` | or `cyankiwi/GLM-4.5-Air-AWQ-4bit` (lighter) |
| `SERVED_NAME` | `glm-4.5-air` | must match LibreChat `models.default` |
| `QUANTIZATION` | `awq_marlin` | AWQ kernel for H100 |
| `TP_SIZE` | `1` | GPUs per worker |
| `MAX_MODEL_LEN` | `65536` | lower if KV cache is tight |
| `GPU_MEM_UTIL` | `0.92` | |
| `HF_TOKEN` | — | set to download weights from HuggingFace |

> Weights are **not** baked into the image (too large for CI). They download on first
> run via `HF_TOKEN`, cached under `/root/project/.hf`. On a closed network,
> pre-download with `huggingface-cli download <MODEL_ID>` into that path.
