# MCP 서비스 자체 호스팅 (docker compose)

Modal 크레딧 소진과 Hugging Face Spaces 유료화(Docker Space는 PRO 필요)로, MCP 서버와 질의 모델을
**아무 리눅스 박스**에서 `docker compose`로 띄우는 구성임. 코퍼스 데이터만 HF 데이터셋 저장소(무료)에서 내려받음.

## Ⅰ. 구성

### 1. 컨테이너 3개 (`docker-compose.yml`, 프로필 `serve`)

| 서비스 | 폴더 | 포트 | 역할 | 메모리 |
|---|---|---|---|---|
| `encoders` | `services/encoders/` | 7860 (내부만) | Nemotron 질의 인코더, BGE-M3, bge-reranker를 HTTP로 제공 | 약 12 GB 상주 |
| `mcp` | `services/mcp/` | 8000 (내부만) | MCP 서버(streamable HTTP, OAuth 2.1 + 정적 토큰). 시작 시 코퍼스 다운로드. 도구 호출 로그 `/data/logs/mcp_calls.jsonl` | 약 3.5 GB (풀링 비전 인덱스 2.4 GB 포함) |
| `caddy` | `services/caddy/` | 80, 443 | `https://$PUBLIC_HOST`를 `mcp:8000`으로 리버스 프록시. Let's Encrypt 자동 발급 | 수십 MB |

- `mcp`는 `encoders`에 compose 내부 주소(`http://encoders:7860`)로 접속함. 호스트에는 80과 443만 열림
- `PUBLIC_HOST`는 박스의 공인 IP로 풀리는 호스트명. AWS에서는 `<public-ip>.sslip.io`(sslip.io가 호스트명을 그 IP로 풀어 줌, DNS 설정 불필요)
  - Let's Encrypt 한도: 같은 호스트명 주당 5장, `sslip.io` 전체 주당 50장. IP가 바뀌면 새 호스트명이라 새 인증서
  - 인증서는 `caddy_data` 볼륨에 남으므로 `docker compose down -v`를 피할 것
- 볼륨: `hf_cache`(모델 가중치 12 GB, 1회 다운로드), `corpus_data`(코퍼스 2.5 GB + `auth.json` + `logs/`), `caddy_data`, `caddy_config`
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
| `PUBLIC_HOST` | Caddy가 인증서를 받을 호스트명. 예: `3.35.10.20.sslip.io`. `MCP_PUBLIC_URL`은 compose가 `https://$PUBLIC_HOST`로 만듦(OAuth 리다이렉트와 메타데이터) |
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
# encoders (호스트 포트 없음: 컨테이너 안에서)
docker compose exec encoders python -c "import urllib.request;print(urllib.request.urlopen('http://localhost:7860/health').read())"
# mcp (Caddy 경유)
curl -s https://$PUBLIC_HOST/health
curl -s -o /dev/null -w "%{http_code}\n" -X POST https://$PUBLIC_HOST/mcp                       # 401
curl -s -X POST https://$PUBLIC_HOST/mcp -H "Authorization: Bearer $MCP_TOKEN" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
curl -s https://$PUBLIC_HOST/.well-known/oauth-authorization-server
```

- 개발 PC에서 전체 도구 호출 점검: `.env`의 `MCP_URL=https://<공개 주소>/mcp`, `MCP_TOKEN` 설정 후 `python serverless/modal_mcp.py`
  (스모크 함수는 호스팅과 무관하게 URL과 토큰만 씀)

### 5. 외부 노출 (Cloudflare Tunnel 예, Caddy 없이 쓰던 구성)

- 이 경로는 Caddy가 인증서를 받을 수 없음(호스트명이 터널을 가리킴). `caddy` 서비스를 빼고 `mcp`에 `ports: 8000:8000`을 되살린 뒤 아래대로

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

## Ⅲ. AWS (한 달 실험)

### 1. 전제

- AWS Free Plan 계정 (크레딧 소진 시 계정 자동 종료. 한 달 실험용)
- IAM 사용자 1명, 정책 `AmazonEC2FullAccess` (비용 조회까지 원하면 `ce:GetCostAndUsage` 추가)
- `.env`에 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION=ap-northeast-2`
- `.env`에 Ⅱ.2의 키 전부 (`ENCODER_TOKEN`, `MCP_TOKEN`, `MCP_USER`, `MCP_PASSWORD`, `QDRANT_CLOUD_*`, `DATA_REPO`, 비공개면 `HF_TOKEN`)
- `pip install boto3 paramiko` (requirements.txt에 포함)

### 2. 명령 4개 (순서대로, 개발 PC에서)

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/31_aws_deploy.py up            # 키 페어, 보안 그룹, t4g.xlarge 40 GB, 공인 IP를 .env에 기록
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/31_aws_deploy.py deploy        # git clone, 원격 .env 업로드, compose up --build, https 헬스 대기
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/31_aws_deploy.py status        # 상태, compose ps, 이달 EC2 비용
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/31_aws_deploy.py down --yes    # 인스턴스 종료, EIP 해제, 보안 그룹과 키 페어 삭제
```

- `up` 옵션: `--instance-type`(기본 `t4g.xlarge`, arm64 4 vCPU 16 GB), `--spot`(일회성 스팟, 회수될 수 있음), `--eip`(고정 IP)
- `up`은 `Name=pharma-rag` 태그로 찾아 재사용함. 두 번 실행해도 인스턴스가 늘지 않음
- `deploy`는 `--branch`(기본 `claude/sharp-tesla-po09dt`)를 받음. 첫 빌드는 arm64 torch 설치로 수 분, 이후 모델 12 GB 다운로드
  - 원격 `.env`에는 compose가 읽는 키만 들어감. 값은 화면에 찍지 않고 키 이름만 표시
  - 헬스가 200이 될 때까지 최대 30분 폴링, 30초마다 `encoders` 로그 3줄 출력
- 중간 명령: `stop`(컴퓨트 과금 중단, EBS 40 GB는 월 약 $4 유지), `start`(IP가 바뀌면 `deploy` 재실행 필요. `--eip`였으면 유지),
  `logs`(도구 호출 로그 회수)
- 22번 포트는 `up`을 실행한 PC의 공인 IP만 허용. IP가 바뀌면 `up`을 다시 실행(규칙 추가)

### 3. 비용

| 항목 | 금액 |
|---|---|
| `t4g.xlarge` 온디맨드 | us-east-1 $0.134/h, ap-northeast-2는 약 10% 높음 (약 $0.15/h) |
| 24시간 가동 한 달 | 약 $100 ~ $110 |
| 밤에 `stop` (하루 12시간) | 절반, 약 $50 ~ $55 |
| gp3 40 GB | 월 약 $4 |
| Elastic IP | 인스턴스가 멈춘 시간 동안만 과금 (시간당 $0.005) |
| 스팟 (`--spot`) | 온디맨드의 약 30 ~ 40%, 회수 시 종료됨 |

- 실측은 `status`의 Cost Explorer 줄(하루 정도 지연) 또는 콘솔 Billing

### 4. 클라이언트 등록

- `deploy`가 끝나면 `.env`의 `MCP_URL=https://<ip>.sslip.io/mcp`, `MCP_PUBLIC_URL`이 채워져 있음
- Claude Code
  ```bash
  claude mcp add --transport http pharma-corpus-remote https://<ip>.sslip.io/mcp --header "Authorization: Bearer <MCP_TOKEN>"
  ```
- claude.ai 웹과 Claude Desktop: Settings > Connectors > Add custom connector, URL `https://<ip>.sslip.io/mcp`,
  Connect에서 `MCP_USER`와 `MCP_PASSWORD` 입력 (Ⅱ.6과 동일)
- IP가 바뀌면(재시작, 재생성) URL도 바뀌므로 두 곳 다시 등록

### 5. 사용 로그 회수 (글쓰기용)

- `mcp` 컨테이너가 도구 호출마다 `/data/logs/mcp_calls.jsonl`에 한 줄 기록(시각, 도구, 인자 200자, 지연 ms, 성공 여부, 검색 상위 3건의 문서와 페이지)
- 회수와 요약
  ```bash
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/31_aws_deploy.py logs         # docker cp + scp -> eval/results/mcp_calls.jsonl
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/32_mcp_usage_report.py        # 도구별 건수, 중앙값과 p90 지연, 오류율, 일별, 상위 문서 표
  ```
- 수동 회수: 박스에서 `docker cp pharma-rag-mcp:/data/logs/mcp_calls.jsonl .` 뒤 `scp -i ~/.ssh/pharma-rag.pem ubuntu@<ip>:mcp_calls.jsonl .`
- `down` 전에 반드시 회수. 볼륨은 인스턴스와 함께 사라짐

## Ⅳ. 운영 메모

- 컨테이너 안에 RunPod 키와 Modal 토큰이 없으므로 `services/mcp/app.py`가 `VISION_ENCODER=http`를 기본으로 둠
- `encoders`가 아직 로딩 중이면 `search_pages`는 "HTTP 503 (models still loading...)"으로 실패하고 도구가 `search_text`를 권함
- Modal 경로(`serverless/modal_*.py`)와 RunPod 경로(`serverless/handler.py`)는 그대로 남아 있음. 결제 수단을 등록하면 다시 동작함
- 로컬 검증 범위(`tests/test_services.py`): 가짜 모델을 넣은 encoders 앱(인증, 검증, 503, 클라이언트 왕복), JsonStore 위 OAuth 체인,
  이 체크아웃의 코퍼스로 빌드한 mcp 앱(health, 401, initialize, tools/call). 실제 모델 로드와 Docker 빌드는 대상 박스에서 확인 필요
