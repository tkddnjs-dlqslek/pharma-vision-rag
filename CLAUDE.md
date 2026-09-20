# CLAUDE.md — pharma-vision-rag 프로젝트 개요서

> 이 파일은 Claude Code가 세션 시작 시 자동으로 읽는다. 로컬(cmd)에서 작업을 이어갈 때
> 필요한 컨텍스트를 여기서 확보하고, 실험 세부는 `docs/EXPERIMENT_PLAN.md`를 따른다.
> 최종 갱신: 2026-09-20

## 한 줄 요약

Sanofi 2025 공시 PDF(차트·표 밀집)에 **한국어로 질문**하면 영문 페이지를 찾아 답하는
멀티모달 RAG를 **4가지 모드**로 구현하고 비교 벤치마크하는 포트폴리오 프로젝트.
핵심 주장: *텍스트 경로를 QT+HyDE로 최적화해도 Vision/Hybrid 경로의 우위가 남는가.*

## 현재 상태 (Phase 2 진행 중, 2026-09-20)

상세 현황표는 `docs/EXPERIMENT_PLAN.md` 0절. 요약:
- **코퍼스 v2**: 27개 문서 1,709p (`eval/corpus.json`이 기준). Sanofi 2024와 2025 전체, Novartis, Roche, AstraZeneca 슬라이드.
  v1(172p)은 긴 컨텍스트에 통째로 들어가 RAG를 정당화할 수 없어서 확대함.
- **평가셋**: 30문항 × 한/영, v2 문서 id로 이전됨. 60문항 확장과 A01, A03, A06 재설계가 남음.
- **평가 코드**: `eval/metrics.py`, `eval/runner.py` 완성. v1 텍스트 기준선은 dense R@5 0.67, 리랭커 0.72(차트 0.40).
- **인덱스**: `pharma_text`는 v1(172p)만 있음. v2 텍스트와 비전은 RunPod 실행 대기. 캡션은 Anthropic 키 대기.
- Phase 1의 4모드(text_only, vision_only, caption, hybrid)와 LangGraph HybridGraph 코드는 그대로 있음.

## 실제 사용 모델 (PRD/README 표기와 다른 곳 주의)

| 역할 | 실제 모델 ID | 코드 위치 | 비고 |
|---|---|---|---|
| 비전 임베딩 | `nvidia/llama-nemotron-colembed-vl-3b-v2` | `scripts/embed_pages_gpu.py` (RunPod) | PRD는 4B였으나 Colab 컴퓨트 소진으로 3B(ViDoRe V3 6위). T4에서 bf16 + float-only cast |
| 텍스트 임베딩 | `BAAI/bge-m3` dense 1024-d cosine | `retriever/docling_text.py` | ColBERT 모드 아님, dense만 사용 |
| 리랭커 | `BAAI/bge-reranker-v2-m3` | `rerank/zerank2.py` | **파일/클래스명은 "ZeRank2"지만 실제는 bge-reranker.** 대외 표기 시 반드시 실제 모델명 사용 |
| 캡션 | `claude-haiku-4-5` | `retriever/caption.py` | 페이지당 ~$0.003 |
| 답변 생성 | `claude-sonnet-4-6` | `generator/claude_vision.py` | 프롬프트 캐싱 사용(최소 2048 토큰 prefix 필요 → 3페이지 이상 넣어야 캐시 동작) |
| QT / HyDE 드래프트 | Haiku (`generator/claude_text.py`) | `retriever/text_qt.py`, `text_hyde.py` | |
| Nemotron 리랭커 | **미구현** (`rerank/nemotron_rerank.py` 없음) | — | plan.md에는 있으나 코드 없음 |

## 아키텍처 요점

```
질문(KR/EN)
 ├─ text_only : Docling 청크 → BGE-M3 → Qdrant(dense) → [reranker] → top-k 페이지 렌더 → Sonnet Vision
 ├─ vision_only: 페이지 이미지 → Nemotron(Colab) → Qdrant(multi-vector MAX_SIM) → Sonnet Vision
 ├─ caption   : 페이지 → Haiku 캡션 → BGE-M3 → Qdrant → Sonnet Vision
 └─ hybrid    : text ∥ vision 각 top-8 → RRF(키워드 휴리스틱 가중치) → top-3 → Sonnet Vision
                └─ HybridGraph(LangGraph)는 hybrid와 기능 동일, Langfuse span 분리용
```

- **hybrid의 `route()`는 LLM이 아닌 키워드 if문**이고 RRF 가중치만 바꾼다. 조건부 엣지 없음.
  → 이 프로젝트는 **agent가 아니라 결정론적 DAG 파이프라인**이다. 대외 설명 시 "agentic"이라 부르지 말 것.
- Qdrant 컬렉션: `pharma_text`(dense) / `pharma_vision`(multi-vector 3072-d, MAX_SIM) / `pharma_caption`(dense).
  포인트 ID는 `uuid5(source:page)`로 결정적 → 재인덱싱 idempotent.

## 레포 구조

```
src/pharma_vision_rag/
  retriever/   docling_text, text_baseline/qt/hyde, nemotron(HTTP client + Qdrant), caption
  rerank/      zerank2 (= bge-reranker-v2-m3)
  generator/   claude_vision (Sonnet), claude_text (Haiku)
  modes/       text_only, vision_only, caption, hybrid (route + rrf_merge 포함)
  router/      langgraph_router (HybridGraph)
  eval/        (비어 있음 — Phase 2에서 채움)
  utils/pdf.py pypdfium2 렌더 헬퍼 (render_page, iter_pages, page_count)
scripts/00~10  Phase 1 smoke/index 스크립트 (Q1.pdf 기준, 이력용)
scripts/11_build_corpus.py        raw PDF → data/pdf/corpus/ + eval/corpus.json
scripts/12_validate_questions.py  평가셋 검증 (스키마, 분포, gold 완전성, 차트 문항 정답 누출 경고)
scripts/13_score_vision_exact.py  RunPod 임베딩 → 전 질의 정확 MaxSim 순위 (벡터 DB 없음)
scripts/14_index_text_all.py      로컬 Docling 인덱싱 (페이지당 약 50초, 소량 전용) + 원본 블록 캐시
scripts/15_rebuild_text_index.py  블록 캐시에서 재청킹, 재임베딩 (Docling 없이 수 분)
scripts/16_make_runpod_bundle.py  RunPod 업로드용 zip 생성
scripts/embed_pages_gpu.py        RunPod 전용: Nemotron 페이지와 질의 임베딩 + Docling 블록 추출
src/.../retriever/chunking.py     블록 → 청크 규칙 (표 머리글 유지, 짧은 조각 병합, 문서 라벨 접두어). 자체 테스트 포함
notebooks/     01 Nemotron smoke, 02 Colab 터널 (deprecated)
docs/          EXPERIMENT_PLAN.md (현행 계획, 0절이 현황), CORPUS.md (코퍼스 구성과 근거), VARAG_REVIEW.md
data/pdf/      gitignore. raw/ = 다운로드 원본, corpus/ = 인덱싱 대상 27개 (11번 스크립트가 생성)
data/embeddings/  gitignore. text_blocks.jsonl (Docling 원본 블록), v2/ (RunPod 산출물), runpod_input.zip
eval/          corpus.json, questions.jsonl, page_inventory.csv (커밋), results/ (gitignore)
```

## 로컬 실행 규칙

- Windows 환경. 스크립트는 반드시 `PYTHONIOENCODING=utf-8 python scripts/XX.py` (cp949 깨짐 방지).
- venv는 `.venv` (anaconda 3.11 기반, 2026-09-16). `pip install -r requirements.txt`는 `PYTHONUTF8=1` 필요(주석의 `─`가 cp949 에러).
  설치 후 반드시 교체: `torch==2.8.0`, `torchvision==0.23.0` (`--index-url https://download.pytorch.org/whl/cpu`),
  `sentencepiece==0.2.0`. 기본 pip가 주는 torch 2.14는 c10.dll 초기화 실패(WinError 1114), sentencepiece 0.2.2는 segfault.
- `.env`는 `.env.example` 기준. `QDRANT_URL=http://localhost:6335`. **절대 커밋 금지.**
- Qdrant: `docker compose up -d qdrant`. 포트 **6335(REST) / 6336(gRPC)** — 6333은 다른 컨테이너(n8n)가 점유.
  멀티벡터 업서트는 반드시 gRPC(`prefer_grpc=True`). REST는 JSON 팽창으로 100MB 초과.
  `QDRANT__SERVICE__MAX_REQUEST_SIZE_MB`는 v1.12.4에서 actix 워커를 멈추므로 **설정하지 말 것**.
- Qdrant 저장소는 **named volume** (`qdrant_storage`). Windows 바인드 마운트는 비정상 종료 후 벡터가 전부 0이 되는 사고가 있었음(2026-09-19).
  Qdrant는 파생 인덱스로만 취급하고, 기준 데이터(`text_blocks.jsonl`, `data/embeddings/v2/`)에서 언제든 재구축한다.
- 개발 PC는 RAM 16GB. BGE-M3와 리랭커(각 2.3GB)를 동시에 올리지 말 것: `runner --mode text` 후 `--mode text_rerank`를 따로 실행(후보 캐시 사용).
  10분 넘는 작업은 PowerShell `Start-Process`로 분리 실행하고 로그 파일로 확인.
- v2 실행 순서: `11`(코퍼스) → `16`(번들) → RunPod `embed_pages_gpu.py` → zip들을 `data/embeddings/v2/`에 풀고 `text_blocks.jsonl`을
  `data/embeddings/`로 복사 → `15`(텍스트 인덱스) → `13`(비전 순위) → `python -m pharma_vision_rag.eval.runner --mode all`.
- 새 스크립트는 기존 패턴 유지: `ROOT/src`를 `sys.path`에 추가, `load_dotenv(ROOT/".env")`, 번호 접두사.

## 알려진 문제와 우회 (재발 방지용)

| 문제 | 원인 | 우회/해법 |
|---|---|---|
| 비전 인덱싱 페이지당 ~3분, 6/23p에서 중단 | 페이지당 `[N_patches, 3072]` fp16(~14MB)을 Cloudflare 무료 터널로 왕복. 30MB 초과 시 502 | `render_scale` 1.5→0.85로 임시 회피. **근본 해법: Colab 내부에서 전 페이지 임베딩 → `.npz` 1회 다운로드 → 로컬 로더로 Qdrant 업서트** (Phase 2 인프라 과제, `docs/EXPERIMENT_PLAN.md` §4) |
| Docling `std::bad_alloc`, OOM | CPU RAM 한계 | OCR 끔 + 5페이지 창 단위 변환(`14`)으로 해결, 누락 0. 대량은 RunPod GPU에서 |
| 텍스트 검색 R@5 0.40 (초기) | 표 조각에 머리글 없음, 40자 미만 청크 36%, gold 누락 | `retriever/chunking.py` 규칙 + gold 완전성 검사. 0.67로 상승 |
| Qdrant 벡터 전부 0 | Windows 바인드 마운트 + 비정상 종료 | named volume, 재구축 스크립트의 0벡터 assert |
| Nemotron dtype mismatch | 내부 ViT가 fp32 고정 | bf16 로드 후 float 파라미터/버퍼만 캐스팅 |
| Nemotron `model(**inputs)` None | 전용 API 필요 | `forward_images / forward_queries / get_scores` 사용 |
| 프롬프트 캐시 0 토큰 | Sonnet 캐시 최소 2048 토큰 | 페이지 3장 이상 컨텍스트 |
| ngrok 익명 차단 | 정책 변경 | cloudflared 사용 |
| Sonnet 답변 변동성 | 모델 출력 비결정성 | 생성 평가는 3회 반복 평균 (검색 지표는 결정적 → 1회) |

## 작업 원칙

- `data/pdf/`, `.env`, `data/qdrant_storage/`, `*.csv`(단 `eval/questions.jsonl`은 커밋)는 커밋하지 않는다.
- 문서 3종의 역할: `pharma-vision-rag-PRD.md`(원안, 수정 안 함) / `plan.md`(원 로드맵, 이력) /
  `docs/EXPERIMENT_PLAN.md`(**현행 실험 계획, 여기만 갱신**) / `summary.md`(진행 요약, Phase 끝마다 갱신).
- 대외 문서(README·블로그)에는 실제 모델 ID를 쓴다. "ZeRank2", "Nemotron 4B", "agentic"은 사실과 다르다.
- Phase 완료 시 태그: `phase-N-done`. 커밋 메시지 prefix: `feat(phase-N)`, `fix`, `docs`, `chore`.

## 다음 할 일

1. (사용자) RunPod에서 `embed_pages_gpu.py` 실행. 업로드 파일은 `data/embeddings/runpod_input.zip`. 먼저 `--smoke`로 패치 수와 예상 용량 확인.
2. 산출물을 받아 `15`(텍스트 인덱스 v2)와 `13`(비전 순위) 실행, `runner --mode all`로 v2 검색 벤치마크.
3. 평가셋을 60문항으로 확장: 연도 간, 회사 간 문항 추가, A01, A03, A06 재설계, gold 완전성 검사 통과.
4. (사용자) `.env`에 `ANTHROPIC_API_KEY` → 캡션 인덱스, QT, HyDE, 답변 생성과 judge, "통째로 넣기" 비교군.
5. E1, E2 실행 → `docs/REPORT.md`.
