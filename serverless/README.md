# Nemotron 질의 인코더 (RunPod Serverless)

MCP 서버 `search_pages`의 질의 임베딩을 RunPod GPU 워커에서 계산하는 엔드포인트임.
개발 PC(여유 RAM 약 1 GB)에 3B 모델을 올리지 않기 위한 구성임.

## Ⅰ. 구성

### 1. 파일

| 파일 | 역할 |
|---|---|
| `handler.py` | 워커 본체. 기동 시 모델 1회 로드(CUDA, bf16), 질의를 fp16 멀티벡터로 반환 |
| `Dockerfile` | 이미지 정의. `BAKE_MODEL` 빌드 인자로 가중치 포함 여부 선택 |
| `../src/pharma_vision_rag/retriever/vision_remote.py` | 로컬 클라이언트. `/runsync` 호출 후 `LocalVisionIndex`가 받는 배열로 복원 |

### 2. 입출력

- 입력: `{"query": str}` 또는 `{"queries": [str, ...]}`
  - 제한: 질의 16개 이하, 질의당 2,000자 이하
- 출력: 질의별 `{"shape": [토큰 수, 3072], "dtype": "float16", "data": base64}`
  - 크기: 질의 1건 약 20 x 3072 fp16, 약 120 KB (base64 적용 시 약 160 KB)
- 오류: `{"error": "..."}` 반환, RunPod가 작업을 FAILED로 표시

## Ⅱ. 빌드와 푸시

### 1. 이미지 방식 선택

| 방식 | 빌드 인자 | 이미지 크기 | 콜드 스타트 | 비고 |
|---|---|---|---|---|
| 가중치 포함 | `BAKE_MODEL=1` (기본값) | 약 15 GB (추정) | 모델 로드만 | 권장 |
| 가중치 미포함 | `BAKE_MODEL=0` | 약 8 GB (추정) | 다운로드 + 로드 | 네트워크 볼륨 연결 시 `/runpod-volume/hf`에 1회 저장 |

- **라이선스 주의**: 모델이 NVIDIA 비상업 라이선스임. `BAKE_MODEL=1` 이미지는 **비공개 레지스트리에만** 푸시 필요

### 2. 로컬 빌드 후 푸시

```bash
cd serverless
docker build --platform linux/amd64 --build-arg BAKE_MODEL=1 -t <registry>/<user>/nemotron-query:v1 .
docker login <registry>
docker push <registry>/<user>/nemotron-query:v1
```

- 레지스트리가 비공개이면 RunPod 설정의 컨테이너 레지스트리 인증 정보 등록 필요
- 개발 PC는 메모리와 디스크 여유가 부족해 빌드 실패 가능성 있음. 이 경우 아래 2안 사용

### 3. GitHub 저장소에서 RunPod가 직접 빌드 (대안)

- RunPod Serverless는 GitHub 저장소 연결 후 Dockerfile 기준 빌드 지원
- 지정 값
  - Dockerfile 경로: `serverless/Dockerfile`
  - 빌드 컨텍스트: `serverless` (`COPY handler.py`가 이 디렉터리 기준임)
  - 빌드 인자: `BAKE_MODEL=1`
- 빌드 결과는 RunPod 내부 레지스트리에 보관됨
- 메뉴 이름과 위치는 바뀔 수 있으므로 현재 RunPod 문서 확인 필요

## Ⅲ. 엔드포인트 설정

| 항목 | 값 | 근거 |
|---|---|---|
| GPU | 16 GB 등급 | bf16 3B 모델 약 7 GB, 질의 인코딩만 수행 |
| 최소 워커 | 0 | 대기 중 과금 없음 |
| 최대 워커 | 1 | 단일 사용자, 동시 요청 적음 |
| 유휴 타임아웃 | 약 5초 | 워커 유지 시간만큼 과금 |
| FlashBoot | 사용 가능하면 켬 | 재기동 시간 단축 |
| 컨테이너 디스크 | 20 GB 이상 | `BAKE_MODEL=0`이고 볼륨이 없을 때 가중치 저장 공간 |
| CUDA 버전 필터 | 12.6 이상 | 기본 이미지가 CUDA 12.6 빌드임 |

- 배포 후 엔드포인트 ID와 API 키를 `.env`에 기입

```bash
RUNPOD_API_KEY=...        # RunPod 설정에서 발급, 커밋 금지
RUNPOD_ENDPOINT_ID=...    # 엔드포인트 화면의 ID
```

- 두 값이 모두 있으면 MCP `search_pages`가 원격 인코더 사용, 하나라도 비면 기존 로컬 CPU 인코더 사용
- MCP 서버 재시작 필요 (인코더는 첫 `search_pages` 호출 때 1회 결정됨)

## Ⅳ. 동작 확인

### 1. curl

```bash
# 키는 환경 변수에서만 읽음. 명령에 직접 쓰지 말 것
curl -s -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "Dupixent quarterly sales chart"}}'
```

- 정상 응답: `"status": "COMPLETED"`, `output.embeddings[0].shape`가 `[토큰 수, 3072]`
- 콜드 스타트가 길면 `IN_QUEUE` 또는 `IN_PROGRESS`와 작업 `id` 반환. `/status/{id}`로 재조회

### 2. 로컬 클라이언트

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe -c "from pharma_vision_rag.retriever.vision_remote import encoder_from_env; print(encoder_from_env()('Dupixent sales chart').shape)"
```

- 클라이언트 동작
  - 제한 시간: 120초 (콜드 스타트 포함)
  - 미완료 작업은 2초 간격으로 `/status/{id}` 재조회
  - 401/403은 API 키, 404는 엔드포인트 ID 확인 안내 메시지 출력

## Ⅴ. 비용

- 단가: 16 GB 등급 **시간당 $0.58** (RunPod 가격 페이지, 2026-09-22 기준)
  - 초당 약 $0.00016
- 과금 범위: 워커 실행 시간만 (콜드 스타트, 처리, 유휴 타임아웃 포함). 워커 0개일 때 과금 없음
- 질의 1건 비용 (추정, 실측 전)

| 상황 | 워커 시간 | 비용 |
|---|---|---|
| 워커 대기 중 호출 | 처리 1초 미만 + 유휴 5초 | 약 $0.001 |
| 콜드 스타트 후 호출 | 로드 30~60초 + 유휴 5초 | 약 $0.01 |

- 콜드 스타트 시간은 이미지 방식, FlashBoot, GPU 호스트에 따라 달라짐. 첫 배포 후 실측 필요
- 네트워크 볼륨 사용 시 볼륨 저장 요금 별도
