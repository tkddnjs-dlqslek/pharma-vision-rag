# CLAUDE.md — pharma-vision-rag 프로젝트 개요서

> 이 파일은 Claude Code가 세션 시작 시 자동으로 읽는다. 로컬(cmd)에서 작업을 이어갈 때
> 필요한 컨텍스트를 여기서 확보하고, 실험 세부는 `docs/EXPERIMENT_PLAN.md`를 따른다.
> 최종 갱신: 2026-09-16

## 한 줄 요약

Sanofi 2025 공시 PDF(차트·표 밀집)에 **한국어로 질문**하면 영문 페이지를 찾아 답하는
멀티모달 RAG를 **4가지 모드**로 구현하고 비교 벤치마크하는 포트폴리오 프로젝트.
핵심 주장: *텍스트 경로를 QT+HyDE로 최적화해도 Vision/Hybrid 경로의 우위가 남는가.*

## 현재 상태 (Phase 1 완료, Phase 2 미착수)

- 4모드(text_only / vision_only / caption / hybrid) + LangGraph HybridGraph 모두 end-to-end 동작.
- 5-mode × 3-query smoke: hybrid·vision 3/3, text_only·caption 2/3. (`summary.md` 참조)
- 인덱스는 **Q1.pdf만, 그것도 부분**: 텍스트 20/23p(319청크), 비전 **6/23p**, 캡션 23/23p.
  Q2·Q3·20-F는 전부 미인덱싱. 비전 인덱싱 병목 원인과 해법은 아래 "알려진 문제" 참조.
- 평가셋(`eval/questions.jsonl`) 0개. `src/pharma_vision_rag/eval/`는 빈 디렉토리.

## 실제 사용 모델 (PRD/README 표기와 다른 곳 주의)

| 역할 | 실제 모델 ID | 코드 위치 | 비고 |
|---|---|---|---|
| 비전 임베딩 | `nvidia/llama-nemotron-colembed-vl-3b-v2` | `notebooks/02_nemotron_tunnel.ipynb` cell 6 | PRD는 4B였으나 Colab 컴퓨트 소진으로 3B(ViDoRe V3 6위). T4에서 bf16 + float-only cast |
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
scripts/00~10  단계별 smoke/index 스크립트 (번호 순서 = 의존 순서)
notebooks/     01 Nemotron smoke, 02 Colab FastAPI 터널 (현재 인덱싱 경로)
docs/          VARAG_REVIEW.md, EXPERIMENT_PLAN.md, CORPUS.md (코퍼스 발췌 근거·페이지 매핑)
data/pdf/      원본 PDF — gitignore. 원본 파일명 고정: Q1.pdf, Q2.pdf, Q3.pdf, "Form 20-F 2025 (Oct 2025).pdf",
               raw/Q{1,2,3}_deck_full.pdf. 인덱싱 코퍼스 7파일(172p): Q1~Q3.pdf, 20F_extract.pdf, Q{1,2,3}_deck.pdf
               (발췌본은 scripts/11_build_extracts.py로 생성. gold_pages는 발췌 후 페이지 번호)
```

## 로컬 실행 규칙

- Windows 환경. 스크립트는 반드시 `PYTHONIOENCODING=utf-8 python scripts/XX.py` (cp949 깨짐 방지).
- `.env`는 `.env.example` 기준. `QDRANT_URL=http://localhost:6335`. **절대 커밋 금지.**
- Qdrant: `docker compose up -d qdrant`. 포트 **6335(REST) / 6336(gRPC)** — 6333은 다른 컨테이너(n8n)가 점유.
  멀티벡터 업서트는 반드시 gRPC(`prefer_grpc=True`). REST는 JSON 팽창으로 100MB 초과.
  `QDRANT__SERVICE__MAX_REQUEST_SIZE_MB`는 v1.12.4에서 actix 워커를 멈추므로 **설정하지 말 것**.
- 비전 인덱싱(Phase 2): RunPod에서 `scripts/embed_pages_gpu.py` 실행 → `embeddings.zip` 받아
  `data/embeddings/`에 두고 `scripts/13_load_nemotron_npz.py <zip>`으로 Qdrant 업서트. 터널(`06`, `COLAB_EMBEDDING_URL`)은 deprecated.
- 실행 순서(의존): `11`(발췌본) → `03`(텍스트 인덱스) → `08`(캡션 인덱스) → `13`(비전 로더) → `10`(4모드 smoke).
- 새 스크립트는 기존 패턴 유지: `ROOT/src`를 `sys.path`에 추가, `load_dotenv(ROOT/".env")`, 번호 접두사.

## 알려진 문제와 우회 (재발 방지용)

| 문제 | 원인 | 우회/해법 |
|---|---|---|
| 비전 인덱싱 페이지당 ~3분, 6/23p에서 중단 | 페이지당 `[N_patches, 3072]` fp16(~14MB)을 Cloudflare 무료 터널로 왕복. 30MB 초과 시 502 | `render_scale` 1.5→0.85로 임시 회피. **근본 해법: Colab 내부에서 전 페이지 임베딩 → `.npz` 1회 다운로드 → 로컬 로더로 Qdrant 업서트** (Phase 2 인프라 과제, `docs/EXPERIMENT_PLAN.md` §4) |
| Docling `std::bad_alloc` (Q1 p16/17/21) | CPU RAM 한계 | 페이지별 배치 처리로 재시도. 텍스트 인덱스에 3페이지 누락 상태 |
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

`docs/EXPERIMENT_PLAN.md` §5 실행 순서를 따른다. 요약: 20-F 발췌 확정 → 페이지 인벤토리 →
질문 30×2 작성·레이블 → RunPod 배치 임베딩 + 로컬 텍스트·캡션 인덱스로 3개 인덱스 채움 → `eval/runner.py` → 벤치마크.
(2026-09-16 기준 1~3 완료, 4 코드 완료. 다음: RunPod 실행 → `13` 로더 → 순서 5.)
