# CLAUDE.md — pharma-vision-rag 프로젝트 개요서

> 이 파일은 Claude Code가 세션 시작 시 자동으로 읽는다. 로컬(cmd)에서 작업을 이어갈 때
> 필요한 컨텍스트를 여기서 확보하고, 실험 세부는 `docs/EXPERIMENT_PLAN.md`를 따른다.
> 최종 갱신: 2026-09-21

## 한 줄 요약

Sanofi 2025 공시 PDF(차트·표 밀집)에 **한국어로 질문**하면 영문 페이지를 찾아 답하는
멀티모달 RAG를 **4가지 모드**로 구현하고 비교 벤치마크하는 포트폴리오 프로젝트.
핵심 주장: *텍스트 경로를 QT+HyDE로 최적화해도 Vision/Hybrid 경로의 우위가 남는가.*

## 진행 중 작업 (핸드오프, 2026-09-21). 컨텍스트 압축 뒤에는 여기부터 읽을 것

**사용자 지시**: 답변은 짧게(결론 몇 줄, 표는 꼭 필요할 때 1개). 중간 알림은 문제 있을 때만. **`docs/REPORT.md` 같은 보고서 문서는 쓰지 말고, 끝나면 채팅으로 요약만.**
API 키와 크레딧이 없어서 **답변 생성과 채점은 Anthropic API 대신 서브에이전트(Agent 도구, model "sonnet" = Sonnet 5)로 수행**한다. 캡션 인덱스는 보류. RunPod Pod는 삭제됨.

**지금 하는 일: 답변 생성(blind) → 채점 → 방식별 정답률 요약**
1. `scripts/17_make_generation_tasks.py`가 text_rerank, vision, hybrid_rerank의 질문별 상위 3페이지로 작업 331건을 만들었음
   (`eval/results/gen_tasks/batch_00~27.json`, 12건씩, 정답과 방식 이름은 `key.json`에만 있음. 전부 gitignore).
2. **[완료 2026-09-21] 답변 생성 331/331건.** 묶음마다 서브에이전트 하나(프롬프트: "`gen_tasks/INSTRUCTIONS.md`를 읽고 그대로 따를 것, 묶음 번호 NN"),
   결과는 `eval/results/gen_answers/batch_NN.jsonl`. 한 번에 3개씩 실행, 묶음당 약 16만 토큰과 3분. 이미지 로드 실패 0건.
   "정보를 찾을 수 없습니다" 44건(13%). 재개할 일이 생기면 `ls eval/results/gen_answers/*.jsonl`로 없는 번호만 다시 실행.
3. **[완료 2026-09-21] 채점 331건.** `18_score_generation.py prepare`가 `judge_tasks/batch_00~08.json`(40건씩)을 만들고,
   기준은 `judge_tasks/INSTRUCTIONS.md`. 판정은 `eval/results/judge_verdicts/batch_NN.jsonl`에
   `{"task_id","verdict":"correct|partial|wrong","reason"}`. 미발견은 `wrong`으로 침(검색 실패가 파이프라인 실패이므로).
   **채점 에이전트가 사용량 한도로 전부 실패해서 메인 세션이 직접 판정함.** 다시 돌릴 일이 생기면 한도 여유를 보고 에이전트로.
   결과 표는 `18_score_generation.py report`로 재생성, 셀 단위 기록은 `eval/results/generation_scored.csv`.

   | 방식 | 전체 | 차트A | 표B | 서술C | 멀티홉D | 영어 | 한국어 | 명시기간 | 모호기간 |
   |---|---|---|---|---|---|---|---|---|---|
   | text_rerank | 0.60 | 0.65 | 0.49 | 0.90 | 0.42 | 0.66 | 0.54 | 0.59 | 0.62 |
   | vision | 0.79 | 0.86 | 0.85 | 0.80 | 0.53 | 0.81 | 0.78 | 0.83 | 0.57 |
   | hybrid_rerank | 0.80 | 0.81 | 0.85 | 0.95 | 0.50 | 0.81 | 0.78 | 0.81 | 0.70 |

   부호 검정: vision과 hybrid_rerank는 text_rerank보다 유의하게 높음(전체 p<0.01, 차트 p≈0.02). vision 대 hybrid_rerank는 차이 없음(p=0.57).
   검색 단계 우위가 생성 단계까지 남음. 한국어 손실은 text 경로에서만 큼(-0.12), 비전과 하이브리드는 -0.03.
4. **[완료 2026-09-21] agentic 모드(E4) 120질의.** 서브에이전트가 `scripts/19_agentic_tools.py`(= `modes/agentic.py` 도구 본문)를 셸로 호출.
   작업은 `20_make_agentic_tasks.py`(한국어 60건을 묶음 0~5, 영어를 6~11로 나눠 **같은 질문의 한영 쌍이 한 에이전트에 가지 않게** 함.
   처음엔 섞여 있어서 에이전트가 쌍둥이 질문을 앞 질문의 검색 결과로 답했음), 채점은 `21_score_agentic.py prepare/report`.
   평균 도구 2.7회, 인용 페이지가 gold에 맞은 비율 0.84.
5. **[완료 2026-09-21] 혼합 blind 재채점.** `scripts/22_blind_rejudge.py prepare/report`. 네 방식 답변 451건을 익명 id로 섞어
   서브에이전트 12명이 채점(`eval/results/rejudge_tasks/`, 역매핑 `map.json`은 채점자 열람 금지). **이게 최종 수치**:
   text_rerank 0.59, vision 0.79, hybrid_rerank 0.75, **agentic 0.87**(멀티홉 0.90, 차트 0.86, 본문 0.72).
   agentic 대 vision p=0.03, 멀티홉 12승 0패. 1차 채점과 일치율 0.88~0.94, agentic 편향은 셀당 +0.008로 거의 없었음. 상세는 계획서 0절.
6. **기준 정답 수정(렌더로 확인)**: A01과 B08의 기준 정답과 gold를 고침. 내용은 계획서 0절과 `eval/questions.jsonl`의 notes.
   재채점 묶음은 A01 수정 뒤, B08 수정 전에 만들어졌음.
7. **[완료 2026-09-21] E4b agentic + 비전 도구.** `agentic.py`에 `page_retriever="vision_precomputed"`(원래 질문 문구로만 조회, 테스트 추가),
   CLI는 `search_pages --source vision`. 작업은 `20_make_agentic_tasks.py --name agentic_vision`, 채점은 `22 prepare --round r2 --anchors 40`
   (새 답변 120 + 1차 기준점 40, `rejudge_r2_tasks/`). 결과 **0.93**(본문 0.95, 차트 0.90, 멀티홉 0.93). r2가 답변당 +0.025 후해서 약 0.90으로 읽을 것.
   E4 대비 본문 8승 1패(p=0.04), 차트는 vision과 동률. 비용과 지연 표는 계획서 0절 끝.
8. **[완료 2026-09-21] gold 완전성 점검.** `scripts/23_gold_candidates.py`가 정답 판정 답변이 인용한 gold 밖 페이지 92개를 모으고
   서브에이전트가 렌더 이미지로 판정(`eval/results/gold_check/`, 57건 해당), 메인 세션이 "그 페이지 하나로 질문을 다 답하나(멀티홉은 해당 홉)"로
   다시 걸러 **38페이지 추가**(27문항). 텍스트 교차 확인 전부 통과, 3건은 직접 렌더 확인. 검색 지표 재계산해 계획서 표 교체.
   결과: vision R@5 0.84에서 0.88, **멀티홉 0.42에서 0.62**, text 계열은 0.02~0.05 상승. "텍스트가 과소평가됐다"는 의심은 틀렸음.
   생성 단계 점수는 기준 정답으로 채점해서 영향 없음.
9. **[완료 2026-09-22] 반복 실행(2회차).** 같은 작업을 새 에이전트로 한 번 더 생성(`gen_tasks_rep2/`, `agentic_rep2_tasks/`,
   `agentic_vision_rep2_tasks/`, 답변은 `*_rep2` 폴더), r3 라운드로 blind 채점(기준점 40건 중 39건 일치, +0.013).
   `22_blind_rejudge.py report`가 두 회차 평균표와 회차 간 일치율을 출력함. **대표 수치는 두 회차 평균**:
   text_rerank 0.59, vision 0.79, hybrid_rerank 0.77, E4 0.84, **E4b 0.91**.
   **결론 변경**: E4 대 vision은 두 회차 평균으로 유의하지 않음(p=0.25, 1회차의 p=0.03은 재현 안 됨). E4b 대 vision은 유의(p=0.002).
   E4b의 본문 개선(1회차 p=0.04)도 재현 안 됨(에이전트가 비전 도구를 안 쓴 묶음이 있었음). 에이전트는 셀 일치율 0.80으로 고정 방식(0.89~0.94)보다 흔들림.
   실행 뒤 BM25 문서 필터 버그(상위 50개를 먼저 자르고 거름)를 고치고 테스트 추가(16개 통과). 두 회차 모두 버그 있는 도구로 실행됨.
10. **[완료 2026-09-22] 에이전트 3회차.** BM25 필터 수정과 도구 지연 개선 뒤 E4, E4b를 새 에이전트로 다시 생성
   (`agentic_rep3_*`, `agentic_vision_rep3_*`), r4 라운드로 blind 채점(기준점 37/40 일치, -0.013). **대표 수치는 전 회차 평균**
   (고정 방식 2회, 에이전트 3회): text_rerank 0.59, vision 0.79, hybrid_rerank 0.77, E4 0.84, **E4b 0.89**.
   E4b 대 vision은 여전히 유의(p=0.024)하지만 약해짐(3회차 E4b 0.85). E4b 대 E4는 p=0.020으로 유의해짐. E4 대 vision은 계속 유의하지 않음(p=0.27).
   E4b 1회차 대 3회차가 p=0.004로 달라서 에이전트 점수는 여러 회차 평균으로만 말할 것. 상세는 계획서 0절, FAILURE_ANALYSIS 8.5.
11. **남은 선택 과제**: D03 질문에 회사명 넣기(RunPod에서 페이지 임베딩 재생성 약 1시간 $2 + 해당 셀 재생성과 재채점),
   실제 API 경로 파일럿(키 필요), 기간 모호 문항의 채점 기준 확정(에이전트 손실의 40% 이상이 여기서 나옴).

**평가셋 결함 상태 (질문 문구를 바꾸면 RunPod에서 질의 임베딩 재계산 필요, 약 $2)**
- A02, A01, B08: 해결(렌더로 확인해 gold와 기준 정답 반영). D03: 질문에 회사 이름이 없음(v1 잔재), 미해결.
- gold 누락: 점검 완료(8번).
- **CLAUDE.md는 서브에이전트도 자동으로 읽는다.** 그래서 이 파일에는 문항별 정답 수치, gold 페이지 번호, 기준 정답 내용을 쓰지 말 것
  (2026-09-21까지 A01, A02, B08의 수치와 페이지가 여기 적혀 있었고, 생성 에이전트 하나가 이 파일 내용을 언급해 발견함). 그런 내용은 계획서나 FAILURE_ANALYSIS에.
- 교훈: gold는 텍스트 일치만으로 넣지 말고 반드시 렌더해서 확인(A05에서 틀린 적 있음). 서브에이전트 보고도 검증 후 반영.

## 현재 상태 (Phase 3 완료, 2026-09-21. 생성 단계와 E4 결과는 위 핸드오프 절과 계획서 0절)

상세 현황표는 `docs/EXPERIMENT_PLAN.md` 0절. 요약:
- **코퍼스 v2**: 27개 문서 1,709p (`eval/corpus.json`이 기준). Sanofi 2024와 2025 전체, Novartis, Roche, AstraZeneca 슬라이드.
  v1(172p)은 긴 컨텍스트에 통째로 들어가 RAG를 정당화할 수 없어서 확대함.
- **평가셋**: 60문항 × 한/영(A20, B20, C10, D10, 기간 모호 10). gold는 정답이 실린 모든 페이지, 멀티홉은 `gold_groups`. 외부 리뷰어 검토 미완.
- **v2 검색 벤치마크 완료**(120질의, R@5): text 0.66, text_rerank 0.70, text_bm25 0.67, **vision 0.84**, hybrid_rerank 0.86.
  차트는 text_rerank 0.72 대 vision 0.97(p=0.006). BM25는 효과 없음. 멀티홉은 전부 0.22~0.43. 상세는 계획서 0절, 실패 유형은 `docs/FAILURE_ANALYSIS.md`.
- **인덱스**: 텍스트와 비전은 RunPod에서 계산해 `data/embeddings/v2/`에 결과만 보관. `runner`는 그 파일을 읽으므로 로컬에 모델이 필요 없음.
  `pharma_text`(15,182청크)는 Qdrant에도 있음. 캡션은 Anthropic 키 대기.
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
  → 기존 4모드는 **agent가 아니라 결정론적 DAG 파이프라인**이다. "agentic"은 계획 중인 E4 모드(계획서 3.3절)에만 쓴다.
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
scripts/text_retrieval_gpu.py     RunPod 전용: 청킹, BGE-M3 임베딩, 질의별 정확 코사인 top-30, 리랭커 순위
src/.../eval/generate.py          검색 상위 3장 → Sonnet 답변 생성(3회 반복, 비용 기록, 이어하기, --dry-run)
src/.../eval/judge.py             Haiku 채점(정답/부분/오답), 유형과 언어별 집계
src/.../eval/pricing.py           토큰 단가 상수. **보고 전에 현재 가격과 대조할 것**
src/.../modes/agentic.py          E4: tool use 단일 agent(도구 6개, 최대 10회). **파일을 직접 실행할 것**(`-m`으로 돌리면 `modes/__init__`이 torch를 불러 76초)
tests/test_eval_generation.py     가짜 클라이언트로 14개 테스트(네트워크 없음)
src/.../retriever/bm25.py         Okapi BM25 (stdlib + numpy, 모델 없음). runner의 bm25, text_bm25, hybrid_bm25 변형
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
- v2 실행 순서: `11`(코퍼스) → `16`(번들) → RunPod에서 `embed_pages_gpu.py`, `13_score_vision_exact.py out`, `text_retrieval_gpu.py` →
  작은 결과 파일만 `data/embeddings/v2/`로 → `15 --precomputed data/embeddings/v2/text` → `python -m pharma_vision_rag.eval.runner --mode all`.
  RunPod 주의: Volume Disk가 `/workspace`에 실제로 마운트됐는지 `df -h`로 확인(30GB 컨테이너 디스크가 가득 찬 적 있음), `pip install hf_transfer` 필요.
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

생성 단계와 E4는 서브에이전트로 끝났음(핸드오프 절 5번이 최종 수치). 남은 건 전부 선택 과제:
1. (사용자, API 키 필요) 캡션 인덱스, QT, HyDE, "통째로 넣기" 비교군. 그리고 API 경로 파일럿:
   `generate.py`, `judge.py`, `modes/agentic.py` 루프는 **실제 API로 한 번도 돌려보지 않았음**.
   미검증 가정: tool_result 안의 이미지 블록, `tool_choice`로 final_answer 강제, 모델 id 유효성.
2. D03 질문에 회사명 추가(RunPod 질의 재임베딩), gold 누락 재검토, 외부 리뷰어 질문 검토.
3. `docs/REPORT.md`는 사용자 지시로 쓰지 않음(요약은 채팅으로).
