# MCP 서비스 자체 호스팅 (docker compose)

Modal 크레딧 소진과 Hugging Face Spaces 유료화(Docker Space는 PRO 필요)로, MCP 서버와 질의 모델을
**아무 리눅스 박스**에서 `docker compose`로 띄우는 구성임. 코퍼스 데이터만 HF 데이터셋 저장소(무료)에서 내려받음.

## Ⅰ. 구성

### 1. 컨테이너 2개 (`docker-compose.yml`, 프로필 `serve`)

| 서비스 | 폴더 | 포트 | 역할 | 메모리 |
|---|---|---|---|---|
| `encoders` | `services/encoders/` | 7860 | Nemotron 질의 인코더, BGE-M3, bge-reranker를 HTTP로 제공 | 약 12 GB 상주 |
| `mcp` | `services/mcp/` | 8000 | MCP 서버(streamable HTTP, OAuth 2.1 + 정적 토큰). 시작 시 코퍼스 다운로드 | 약 3.5 GB (풀링 비전 인덱스 2.4 GB 포함) |

- `mcp`는 `encoders`에 compose 내부 주소(`http://encoders:7860`)로 접속함
- 볼륨: `hf_cache`(모델 가중치 12 GB, 1회 다운로드), `corpus_data`(코퍼스 2.5 GB + `auth.json`)
- 환경 변수는 저장소 루트 `.env`에서 compose가 골라 넣음(필요한 키만 컨테이너에 전달)

### 2. 요구 사양

- RAM **16 GB** (8 GB로는 모델 3개 상주 불가), 디스크 여유 약 20 GB, x86_64 또는 aarch64
- 후보
  - Oracle Cloud Always Free ARM VM (Ampere A1, 4 OCPU와 24 GB, 무료)
  - 집 PC + Cloudflare Tunnel (공인 IP나 포트 개방 없이 https 노출)
- ARM 주의
  - 베이스 이미지 `python:3.11-slim`은 멀티아치라 그대로 빌드됨
  - torch는 PyTorch CPU 인덱스의 aarch64 휠(2.8.0) 사용. 해당 `pip` 줄이 실패하면 `--index-url`을 빼고 PyPI 휠 사용(aarch64 PyPI torch는 CPU 전용)
  - 개발 PC(x86)에서 빌드해 옮기지 말고 **대상 박스에서 빌드**할 것

### 3. 지연 시간 (정직한 기대치)

| 항목 | 예상 |
|---|---|
| 첫 기동: 가중치 12 GB 다운로드 + 로드 | 수 분 ~ 십수 분 (네트워크 의존). 그동안 POST는 503 |
| 재기동 (볼륨에 캐시 있음): 모델 로드 | 2 ~ 5분 |
| `search_pages` 1건 (Nemotron 인코드, 2 ~ 4 코어 CPU) | **30 ~ 60초** |
| `search_text` 1건 (BGE-M3 임베딩 + 리랭크) | 수 초 |
| `mcp` 기동: 코퍼스 2.5 GB 다운로드 + BM25 + 비전 인덱스 로드 | 첫 회 수 분, 이후 30초 내 |

## Ⅱ. 절차

### 1. 데이터셋 저장소 채우기 (개발 PC, 1회)

```bash
huggingface-cli login                                   # write 토큰
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/30_upload_corpus_data.py --repo sangwongim922/pharma-corpus-data --private
```

- 올리는 파일: `eval/corpus.json`, `data/embeddings/v2/text/text_chunks.jsonl`, `data/pdf/corpus/*.pdf`(27개),
  `data/embeddings/vision_index_pooled/*`(2.4 GB)
- 같은 크기의 파일은 건너뜀. 중단되면 다시 실행
- `--private`를 쓰면 `mcp` 컨테이너에 `HF_TOKEN`(read) 필요. PDF는 재배포 권리가 불명확하므로 비공개 권장

### 2. `.env` 채우기 (대상 박스의 저장소 루트)

| 키 | 값 |
|---|---|
| `ENCODER_TOKEN` | `python -c "import secrets; print(secrets.token_hex(16))"` |
| `MCP_TOKEN` | 위와 같은 방식으로 별도 생성. Claude Code용 정적 토큰 |
| `MCP_USER`, `MCP_PASSWORD` | claude.ai와 Claude Desktop 커넥터 로그인 폼 계정 |
| `MCP_PUBLIC_URL` | 외부에서 보이는 https 원본. 예: `https://mcp.example.com` (터널 호스트명). OAuth 리다이렉트와 메타데이터에 쓰임 |
| `QDRANT_CLOUD_URL`, `QDRANT_CLOUD_API_KEY` | 기존 Qdrant Cloud 클러스터(컬렉션 `pharma_text_v2`) |
| `DATA_REPO` | `sangwongim922/pharma-corpus-data` |
| `HF_TOKEN` | 데이터셋이 비공개일 때만 실제 read 토큰. **자리표시자 문자열이면 다운로드가 401로 실패하므로 비워 둘 것** |

- 개발 PC의 `.env`에도 `ENCODER_URL=https://<encoders 공개 주소>`와 `ENCODER_TOKEN`을 넣으면 로컬 stdio MCP 서버(`mcp_server.py`)가
  같은 인코더를 씀(`VISION_ENCODER=auto`는 RunPod 키 > `ENCODER_URL` > Modal 토큰 순으로 고름)

### 3. 빌드와 기동 (대상 박스)

```bash
git clone <repo> && cd pharma-vision-rag        # .env 작성 후
docker compose --profile serve up -d --build
docker compose logs -f encoders                 # "models loaded in NNNs" 까지 대기
docker compose logs -f mcp                      # "corpus data from ... ready", "Uvicorn running on ... 8000"
```

- `docker compose up -d qdrant`(개발용)는 프로필 밖이라 영향 없음
- 재빌드가 필요한 변경: `src/` 또는 `services/` 수정. 데이터만 바뀌면 `docker compose --profile serve restart mcp`

### 4. 스모크 테스트 (curl)

```bash
# encoders
curl -s http://localhost:7860/health                                      # {"status":"ok",...} 이어야 함
curl -s -X POST http://localhost:7860/text/embed -H "Authorization: Bearer $ENCODER_TOKEN" \
  -H "Content-Type: application/json" -d '{"queries":["Dupixent sales Q2 2025"]}' | head -c 200
curl -s -X POST http://localhost:7860/vision/encode -H "Authorization: Bearer $ENCODER_TOKEN" \
  -H "Content-Type: application/json" -d '{"queries":["Dupixent sales chart"]}' | head -c 200   # 30~60초
# mcp
curl -s http://localhost:8000/health
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/mcp                       # 401
curl -s -X POST http://localhost:8000/mcp -H "Authorization: Bearer $MCP_TOKEN" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
curl -s https://$MCP_PUBLIC_HOST/.well-known/oauth-authorization-server                          # 터널 뒤에서
```

- 개발 PC에서 전체 도구 호출 점검: `.env`의 `MCP_URL=https://<공개 주소>/mcp`, `MCP_TOKEN` 설정 후 `python serverless/modal_mcp.py`
  (스모크 함수는 호스팅과 무관하게 URL과 토큰만 씀)

### 5. 외부 노출 (Cloudflare Tunnel 예)

```bash
cloudflared tunnel login
cloudflared tunnel create pharma-mcp
cloudflared tunnel route dns pharma-mcp mcp.example.com
cloudflared tunnel run --url http://localhost:8000 pharma-mcp     # 또는 config.yml에 ingress 등록 후 서비스로
```

- `MCP_PUBLIC_URL=https://mcp.example.com`으로 맞추고 `mcp` 재기동
- `encoders`는 외부 노출 불필요(`mcp`가 내부 네트워크로 호출). 개발 PC에서도 쓰려면 별도 호스트명으로 터널 하나 더

### 6. 클라이언트 등록

- Claude Code
  ```bash
  claude mcp add --transport http pharma-corpus-remote https://mcp.example.com/mcp --header "Authorization: Bearer <MCP_TOKEN>"
  ```
- claude.ai 웹과 Claude Desktop (OAuth, 기존 Modal 배포와 동일한 흐름)
  - Settings > Connectors > Add custom connector. URL `https://mcp.example.com/mcp`, OAuth client id와 secret은 비움(동적 등록)
  - Connect 클릭 시 서버 로그인 폼이 열림. `MCP_USER`와 `MCP_PASSWORD` 입력
  - 액세스 토큰 30일, 리프레시 90일. 토큰은 `corpus_data` 볼륨의 `auth.json`에 있어 재기동에도 유지됨.
    볼륨을 지우면 재로그인 1회

## Ⅲ. 운영 메모

- 컨테이너 안에 RunPod 키와 Modal 토큰이 없으므로 `services/mcp/app.py`가 `VISION_ENCODER=http`를 기본으로 둠
- `encoders`가 아직 로딩 중이면 `search_pages`는 "HTTP 503 (models still loading...)"으로 실패하고 도구가 `search_text`를 권함
- Modal 경로(`serverless/modal_*.py`)와 RunPod 경로(`serverless/handler.py`)는 그대로 남아 있음. 결제 수단을 등록하면 다시 동작함
- 로컬 검증 범위(`tests/test_services.py`): 가짜 모델을 넣은 encoders 앱(인증, 검증, 503, 클라이언트 왕복), JsonStore 위 OAuth 체인,
  이 체크아웃의 코퍼스로 빌드한 mcp 앱(health, 401, initialize, tools/call). 실제 모델 로드와 Docker 빌드는 대상 박스에서 확인 필요
