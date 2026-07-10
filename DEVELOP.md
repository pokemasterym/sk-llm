# GLM-4.5-Air 개발 가이드 (Flightbase / Jonathan 직접 접속)

MLOps 서버(Acryl Jonathan Flightbase)의 **개발도구**로 직접 들어가 GLM-4.5-Air를
vLLM으로 띄우고, 테스트한 뒤 배포 → LibreChat 연결까지 가는 실전 런북.

- 하드웨어: **H100 80GB × 1**
- 이미지(Pull URL): `ghcr.io/pokemasterym/sk-llm:latest`
- 목표: OpenAI 호환 `/v1/chat/completions` 엔드포인트 → LibreChat 연결

---

## 0. 전제 체크

- [ ] Flightbase에서 **도커 이미지 생성 → Pull** 로 위 Pull URL 등록 완료
      (안 했으면: 도커 이미지 목록 → 생성 → 방식 **Pull** → URL 붙여넣기 → 생성)
- [ ] GHCR 패키지가 **Public** (또는 Flightbase에 레지스트리 인증 등록)
- [ ] HuggingFace 토큰 준비 (`hf_...`) — 가중치 다운로드용. 폐쇄망이면 §3-B 참고

---

## 1. 개발도구로 접속하기

1. **학습 프로젝트 생성** → 도커 이미지 = 방금 등록한 `sk-llm` 선택
2. **인스턴스 할당**: vLLM 테스트를 하려면 개발도구에 **GPU 1장(H100)** 을 할당해야 함
   > 매뉴얼 기본 권장은 "개발도구엔 GPU 미할당"이지만, LLM을 실제로 띄워 테스트하려면
   > 여기서 GPU 1장이 필요함. **활성화된 동안 자원을 계속 점유**하므로 테스트 끝나면
   > 반드시 비활성화할 것.
3. **개발도구 활성화** → **VSCode / Jupyter Lab / Shell** 중 하나 실행
4. 작업은 반드시 **`/root/project`** 경로에서 진행
   > ⚠️ 비활성화 시 `/root/project/**` 외의 모든 데이터는 **소멸**됨. 패키지 설치 등
   > 바깥 경로 변경을 남기려면 **이미지 Commit**(§7) 필수. 모델 체크포인트/가중치는
   > `/root/project` 안에 저장.

---

## 2. 컨테이너 안 상태 확인

Shell(또는 VSCode 터미널)에서:

```bash
nvidia-smi                 # H100 인식 확인
python -c "import vllm; print(vllm.__version__)"   # vLLM 버전 확인 (GLM-4.5 MoE 지원 필요)
cat /usr/local/bin/serve.sh 2>/dev/null            # 기동 스크립트 있는지 (이미지에 포함됨)
echo "$MODEL_ID / $SERVED_NAME / $QUANTIZATION"    # 기본 env 값 확인
```

vLLM 버전이 너무 낮아 GLM-4.5를 못 올리면:
```bash
pip install -U vllm      # 최신으로 올린 뒤, 유지하려면 §7 Commit
```

---

## 3. 모델 가중치 준비

### A. 온라인 (HF에서 다운로드) — 기본

```bash
export HF_TOKEN=hf_xxxxxxxx
export HF_HOME=/root/project/.hf        # /root/project 안에 캐시 → 세션 유지
huggingface-cli download QuantTrio/GLM-4.5-Air-AWQ-FP16Mix \
  --local-dir /root/project/models/glm-4.5-air
```
> vLLM에 `--model` 로 로컬 경로를 주거나, 그냥 모델ID를 주면 첫 실행 때 자동 다운로드됨.
> 용량이 크므로(수십 GB) 반드시 `HF_HOME` 을 `/root/project` 하위로 둘 것.

### B. 폐쇄망 (인터넷 불가)

- 인터넷 되는 곳에서 위 `huggingface-cli download` 로 받은 폴더를 **데이터셋 업로드** 또는
  스토리지로 옮겨 `/root/project/models/glm-4.5-air` 에 배치
- 실행 시 `--model /root/project/models/glm-4.5-air` 로 로컬 경로 지정

---

## 4. vLLM 서버 실행 (개발 테스트)

이미지에 포함된 스크립트로:

```bash
serve.sh
```

또는 직접(내용 확인/수정용):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model QuantTrio/GLM-4.5-Air-AWQ-FP16Mix \
  --served-model-name glm-4.5-air \
  --quantization awq_marlin \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser glm45 \
  --reasoning-parser glm45
```

로딩에 수십 초~수 분 걸림. `Application startup complete` 뜨면 준비 완료.

---

## 5. 로컬 테스트 (같은 컨테이너 다른 터미널)

```bash
# 모델 목록
curl -s localhost:8000/v1/models | jq

# 채팅
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.5-air",
    "messages": [{"role":"user","content":"한국어로 간단히 자기소개 해줘"}],
    "max_tokens": 256
  }' | jq -r '.choices[0].message.content'
```

정상 응답 오면 개발 단계 성공.

---

## 6. 자주 겪는 문제

| 증상 | 원인 / 해결 |
|------|-------------|
| `CUDA out of memory` | `--max-model-len` 낮추기(예 32768), `--gpu-memory-utilization 0.90`, 동시요청 `--max-num-seqs 8` |
| `unknown quantization` / awq 로드 실패 | vLLM 최신화(`pip install -U vllm`), `--quantization awq_marlin` 확인 |
| `unknown parser 'glm45'` | 해당 vLLM이 파서 미지원 → `--tool-call-parser`/`--reasoning-parser` 두 줄 삭제(기본 채팅은 됨) |
| 모델 다운로드 안 됨 | `HF_TOKEN` 미설정, 또는 폐쇄망 → §3-B |
| MoE/GLM-4.5 미지원 에러 | vLLM 버전 낮음 → 최신화 후 Commit |
| 느린 첫 응답 | 가중치 다운로드+로딩 중. 두 번째부터 정상 |

---

## 7. 개발 환경 보존 (이미지 Commit)

`/root/project` 밖을 바꿨으면(예: `pip install -U vllm`) 비활성화 전 **이미지 Commit**:

- 개발도구 화면의 **이미지 커밋** 버튼 사용
- 이후 배포/재사용 시 그 커밋 이미지를 쓰면 환경이 그대로 유지됨
- 단, 대용량 모델 가중치는 이미지에 넣지 말고 `/root/project` 또는 스토리지에 둘 것

작업 끝나면 **개발도구 비활성화**로 GPU 반납.

---

## 8. 배포로 전환 (실서비스 워커)

개발/테스트가 끝나면 상주 서비스로 배포:

1. **모델배포 → 새 배포 → Nexus에서 불러오기(커스텀 컨테이너)**
   > ⚠️ **배포코드(학습에서 불러오기) 방식은 쓰지 말 것.** vLLM은 이미 OpenAI 규격
   > 서버라, 플랫폼 워커 포맷으로 감싸면 LibreChat 규격이 깨짐.
2. **Nexus URL** = 이미지 저장 주소, **Ingress 경로** = 예 `/glm`
   → 접속: `http://{kong-ip}/glm/v1/...`
3. **워커 설정**: 이미지 선택 + 환경변수(`HF_TOKEN`, 필요 시 `MODEL_ID` 등) + **실행 커맨드 = `serve.sh`**
4. **Dynamic Scaling OFF** (LLM 로딩이 길어 콜드스타트 치명적 → 워커 상주)
5. 워커 뜨면 테스트: `curl http://{kong-ip}/glm/v1/models`

---

## 9. LibreChat 연결 (운영서버 `librechat.yaml`)

```yaml
endpoints:
  custom:
    - name: "GLM-4.5-Air"
      apiKey: "dummy"                       # vLLM에 인증 안 걸었으면 아무 값
      baseURL: "http://{kong-ip}/glm/v1"    # 배포 Ingress 경로
      models:
        default: ["glm-4.5-air"]            # served-model-name과 일치
        fetch: true
      titleConvo: true
      modelDisplayLabel: "GLM-4.5-Air"
```
추가 후 LibreChat 재기동 → 모델 목록에 표시됨.

> 네트워크: **LibreChat 운영서버 → Flightbase kong IP** 방화벽이 열려 있어야 함.
> 폐쇄망이면 이 경로부터 확인.

---

## 10. 최종 체크리스트

- [ ] 개발도구 GPU(H100 1장) 할당 + 활성화
- [ ] `nvidia-smi`, `vllm` 버전 OK
- [ ] `HF_HOME=/root/project/.hf`, 가중치 다운로드 OK
- [ ] `serve.sh` 로 서버 기동, `Application startup complete`
- [ ] `curl .../v1/chat/completions` 한국어 응답 확인
- [ ] (환경 변경 시) 이미지 Commit
- [ ] Nexus 커스텀 컨테이너로 배포, Dynamic Scaling OFF
- [ ] LibreChat custom endpoint 추가 후 재기동
- [ ] 테스트 끝난 개발도구는 비활성화(GPU 반납)

---
막히면 해당 단계의 로그(서버 기동 로그 / `curl` 응답 / `nvidia-smi`)와 함께 문의.
