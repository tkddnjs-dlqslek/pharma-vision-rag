# pharma-vision-rag — 실행 계획 (plan.md)

**작성일**: 2026-04-21
**기반 문서**: [pharma-vision-rag-PRD.md](pharma-vision-rag-PRD.md)
**현재 상태**: Phase 0 진행 중. git 초기화, 원격 연결(`tkddnjs-dlqslek/pharma-vision-rag`), Sanofi PDF 4개 수집, 디렉토리 구조 완료. 설정 파일 및 Colab 노트북 생성 중.

---

## 0. 결정 사항 (확정 완료)

| # | 항목 | 결정 |
|---|---|---|
| D1 | 타임라인 | **Phase 0 후 재검토** (속도 보고 결정) |
| D2 | 공개 범위 | **완전 공개** (MIT, HF Spaces 데모 포함) |
| D3 | 블로그 플랫폼 | **Medium + 링크드인 병행** |
| D4 | Stretch goal | **전부 보류** (본편 v1 완주 후 분리) |
| D5 | GPU | **Colab** (로컬 GPU 불필요) |
| D6 | 평가셋 범위 | **Sanofi 한정 + 한국어 질문 자작** (ViDoRe 재활용 X) |
| D7 | Text 경로 최적화 | **Option A 채택** — baseline + QT + HyDE 3버전 비교 |
| D8 | 데이터 기간 정책 | **2025년으로 통일** — 20-F(FY2025 전체, Jan-Dec 2025) + Q1/Q2/Q3 2025 Press Release. 정정: 앞서 기간 혼합이라 기록했으나 실제 검증 시 20-F도 FY2025. |
| D9 | Granularity 혼합 | **의도적 활용** — 20-F는 연간 전체, Q1/Q2/Q3 PR은 분기별. 같은 정보가 양쪽에 중복 존재 가능 → "retriever가 연간/분기/누적을 구분하는가" 실험축. |
| D10 | Nemotron 모델 크기 | **4B 채택** (`nvidia/nemotron-colembed-vl-4b-v2`). ViDoRe V3 **3위** (1위 8B, 6위 3B). 선정 이유: (1) Colab Pro L4(24GB)에서 안정 동작, (2) top-3 SOTA로 포트폴리오 스토리 충분, (3) 인코딩 시간 8B 대비 절반 → 실험 반복성 확보. 필요 시 8B로 업그레이드는 MODEL_ID 한 줄 교체로 가능. |

---

## 1. Day 0 진행 현황 (환경 전 사전 작업)

- [x] **D1~D9 의사결정 완료** (§0 테이블)
- [x] GitHub 공개 레포 생성 — https://github.com/tkddnjs-dlqslek/pharma-vision-rag
- [x] git init + 원격 연결 (main 브랜치)
- [x] Sanofi PDF 4개 수집 (`data/pdf/` — .gitignore로 커밋 제외)
- [x] 디렉토리 구조 생성 (`src/pharma_vision_rag/`, `notebooks/`, `data/samples/`, `scripts/`, `docs/`)
- [x] `.gitignore` 작성 (모델·데이터·시크릿 제외)
- [ ] `plan.md`, PRD, 설정 파일들 초기 커밋 (Phase 0 끝에)
- [ ] Nemotron ColEmbed V2 **라이선스 확인** (HF 페이지에서 상업/재배포 조건)
- [x] **GPU 전략 확정: Colab Pro 사용** (구독 완료, 로컬 GPU 불필요)

---

## 2. Phase별 세부 체크리스트

PRD §6을 작업 가능 단위로 풀어씀. 각 Phase 끝에 git 커밋 + 태그(`phase-0-done` 등).

### Phase 0: 환경 셋업 (Day 1~2, 4주 기준 Week 1 초반)

- [ ] VARAG fork → `pharma-vision-rag`에 submodule 또는 복사 결정
- [ ] `requirements.txt` 초안 작성 (docling, qdrant-client, transformers, langgraph, langfuse, gradio, fastapi, anthropic)
- [ ] Qdrant Docker 컨테이너 기동 (`docker-compose.yml` 작성)
- [ ] Nemotron ColEmbed V2 3B HF 다운로드 + 더미 이미지로 임베딩 추출 smoke test
- [ ] Docling 설치 → 샘플 PDF 1개로 레이아웃·표 추출 확인
- [ ] VARAG 기본 4개 모드가 샘플 PDF로 돌아가는지 검증
- [ ] **VARAG 코드 리뷰 + 개선점 발굴** — 각 모드의 내부 로직 읽고 다음을 기록:
  - 청킹 전략(고정 길이? 문단?)을 더 나은 걸로 바꿀 여지
  - 프롬프트 템플릿 품질 (파마 도메인에 맞게 다시 써야 할 부분)
  - 에러 핸들링·로깅 누락 지점
  - 우리가 교체할 모듈 인터페이스가 제대로 추상화되어 있는지
  - `VARAG_REVIEW.md` 파일에 개선 리스트 작성 → Phase 1에 반영
- [ ] Langfuse 셋업 (기존 키 재사용 — SmartShift·dupixent-classifier 스택)
- [ ] `.env.example` + Anthropic / HF / Langfuse 키 문서화

**승인 기준**: 샘플 PDF 한 개에서 VARAG 기본 모드 돌아가고, Nemotron 임베딩 shape 확인됨.

### Phase 1: 최신 모델로 교체 (Day 3~7, Week 1 후반~Week 2 초반)

- [ ] `retriever/nemotron.py` — ColPali 인터페이스 맞춰 Nemotron wrapper 작성
- [ ] `retriever/docling_text.py` — OCR 자리를 Docling block 기반 추출로 교체 (text / table-as-markdown 블록만 BGE-M3로)
- [ ] **Text 경로 3버전 구현** (크로스링구얼 최적화 ablation — Option A)
  - `retriever/text_baseline.py` — 원 질문 그대로 BGE-M3 임베딩
  - `retriever/text_qt.py` — Query Translation: 한국어 Q → Claude Haiku로 영어 번역 → BGE-M3
  - `retriever/text_hyde.py` — HyDE: Claude Haiku가 가상 답변 생성 → 그 생성문을 BGE-M3 임베딩
  - 세 버전 모두 동일 인터페이스(`.search(query) -> top_k`) 유지
  - Phase 3에서 세 버전을 Text-only 모드 내부에서 ablation
- [ ] `rerank/` — **리랭커 2종 병렬 구현 후 A/B 비교** (Phase 3에서 실측):
  - `rerank/zerank2.py` — ZeRank2 (범용 cross-encoder)
  - `rerank/nemotron_rerank.py` — `llama-nemotron-rerank-vl-1b-v2` (리트리버와 동일 생태계)
  - 실험 설계:
    - 독립변수: reranker 종류
    - 통제변수: 쿼리셋·top-20 후보·생성 VLM·프롬프트 동일
    - 종속변수: NDCG@5, Recall@3, Judge score, 지연(ms), 비용($)
  - 결과 포맷: 2 리랭커 × 30 질문 × 6 지표 CSV
  - 최종 승자를 Hybrid 모드 기본값으로. 패자는 블로그 비교 재료로 유지
- [ ] `generator/claude_vision.py` — OpenAI 자리에 Claude Sonnet 4.7 Vision
- [ ] `modes/` 폴더에 4가지 모드 각각 엔트리포인트 분리:
  - `text_only.py`
  - `vision_only.py`
  - `caption.py`
  - `hybrid.py` (LangGraph 라우팅 그래프)
- [ ] 샘플 질문 3개 smoke test — 4 모드 모두 답변 생성 + Langfuse trace 확인

**승인 기준**: Langfuse 대시보드에 4 모드 × 3 질문 = 12 trace 보임.

### Phase 2: 평가셋 구축 (Day 8~14, Week 2 후반~Week 3 초반)

- [x] **Sanofi 공개 자료 수집 완료** (2026-04-21)
  - `data/pdf/Form 20-F 2025 (Oct 2025).pdf` — 300페이지, FY2025 연간 (Jan-Dec 2025)
  - `data/pdf/Q1.pdf` — 23페이지, 2025 Q1 Press Release
  - `data/pdf/Q2.pdf` — 26페이지, 2025 Q2 Press Release
  - `data/pdf/Q3.pdf` — 28페이지, 2025 Q3 Press Release
  - 총 377페이지. 밀도 높은 섹션 50~80p 추출 예정
  - **Granularity 주의**: 20-F 연간 vs PR 분기 — 질문 작성 시 "연간"/"Q1"/"Q2" 등 명시 필수
  - (보류) Dupilumab PMC 임상 논문 — 임상 유형 질문 부족 시 Phase 2 진행 중 추가
- [ ] **한국어 질문 30개 자작** (크로스링구얼: 한국어 Q × 영문 PDF)
  - 유형 분포: 차트 10 / 표 10 / 본문 5 / 멀티홉 5
  - 각 질문에 정답 + page_id + block_type(chart/table/text) 레이블
  - 인턴 경험 활용해 실제 업무에서 나올 법한 질문으로 작성 → 면접 토킹 포인트
  - **각 질문의 영어 대응쌍도 작성** → KR/EN 성능 비교용 (총 30 × 2 = 60개)
  - **기간 명시 규칙**:
    - 25개 기본 질문: 연도·분기 명시 ("FY2024 연간", "2025 Q2" 등) → gold_page 단일
    - 5개 실험 질문: 기간 애매 ("Dupixent 매출 추세") → multiple gold_pages 허용 (temporal disambiguation 테스트)
  - `eval/questions.jsonl` 포맷 확정 (필드: id, q_ko, q_en, answer, gold_pages[list], block_type, type, period_spec)
- [ ] **Eval 스크립트 자작** (ViDoRe 구조 참고, 전체 재활용은 아님)
  - 이유: ViDoRe는 순수 retrieval 중심, 우리는 크로스링구얼 + answer generation 평가까지 필요
  - 재활용 범위는 Recall@k·NDCG@k 공식 함수 수준
  - 평가 루프·judge 로직·리포팅은 직접 구현
- [ ] 외부 리뷰어 1명에게 한국어 질문 품질 검토 요청

**승인 기준**: `python -m eval.run --mode all` 한 방에 4 모드 × 30 질문(한국어 자작) × 2 언어 × 전 지표 CSV 생성.

### Phase 3: 벤치마크 실행 및 분석 (Day 15~21, Week 3)

- [ ] 4 모드 × 30 질문 × 2 언어 풀 실행 (인덱싱 1회 + 쿼리 240회)
- [ ] **Text-only 3버전 ablation** (baseline vs +QT vs +HyDE) — 30질문 × 2언어 × 3버전 = 180 runs
  - 블로그 킬러 주장 검증: "QT·HyDE로 텍스트 최적화해도 Vision 경로의 우위가 남는가"
- [ ] 결과 CSV → matplotlib/plotly로 표·그래프 생성
  - 모드별 Recall@5 / NDCG@5 / Answer F1 bar chart
  - 질문 유형(A/B/C/D)별 breakdown
  - **언어(KR/EN)별 breakdown** — "Vision 경로가 언어 격차를 메우는가" 가설 검증
  - **Text-only 3버전 breakdown** — QT/HyDE 효과 정량화
  - 비용 vs 정확도 scatter
- [ ] **Reranker A/B 실측** (Phase 1에서 구현한 ZeRank2 vs Nemotron Rerank)
  - 2 리랭커 × 30 질문 × 6 지표 CSV
  - 최종 승자 확정 후 Hybrid 기본값 고정
- [ ] Hybrid 모드 라우팅 결정 로그 분석 (몇 %가 vision으로, 몇 %가 text로 갔나)
- [ ] **질문 표현 최적화 파일럿** (소규모 실험, 0.5일)
  - 10개 질문 × 3가지 표현(키워드형/구체형/질문형) × Hybrid 모드만 = 30 runs
  - 결과 명확하면 블로그에 "효율적 질문법" 섹션, 애매하면 폐기
- [ ] 실패 케이스 **정성 분석 10개** — 각 case에 원인 태그 (OCR 실패 / 리트리버 miss / VLM 환각 / 언어 mismatch 등)
- [ ] `REPORT.md` 초안 (표 + 그래프 + 해석)

**승인 기준**: PRD §5.3 가설표 실측 버전 + 실패 10케이스 분석 완성.

### Phase 4: 문서화 및 공개 (Day 22~28, Week 4)

- [ ] `README.md` 한·영 병기 — 아키텍처 다이어그램, 빠른 시작, 벤치마크 요약
- [ ] `docker-compose up` 원클릭 실행 검증 (다른 머신에서)
- [ ] **Gradio UI 3종 구성**:
  - **A. 데모 페이지**: 질문 입력 → 답변 + top-3 근거 페이지 이미지 썸네일
  - **B. 배치 평가 대시보드**: 30개 질문 × 4모드 결과표, 모드별 Recall/NDCG 바차트, 실패 케이스 토글
  - **C. 페이지 뷰어**: PDF 페이지 좌측, 해당 페이지에 걸린 질문·정답·블록타입 메타데이터 우측. 디버깅용
- [ ] HuggingFace Spaces에 데모(A)만 배포, B·C는 로컬/docker용
- [ ] 데모 GIF 촬영 (Hybrid 모드가 차트 질문에 답변하는 장면)
- [ ] 한국어 블로그 포스트 초안 (링크드인용, ~2000자)
- [ ] 영어 블로그 포스트 초안 (Medium용, ~1500 words)
- [ ] 링크드인 포스트 게재 + 레포 star/watch 유도

**승인 기준**: 외부인이 링크드인 포스트만 보고 레포 clone → docker compose up → 데모까지 5분 이내.

---

## 3. 재활용 자산 체크리스트 (PRD §3.3)

이미 있는 것들 — 새로 짜지 말 것.

- [ ] `pharma-validator` 레포에서 PDF 전처리 유틸 copy
- [ ] `dupixent-classifier` eval harness 골격 copy → 파마 버전으로 수정
- [ ] LangGraph 교재 노트 → hybrid 라우팅 그래프 초안
- [ ] Langfuse 프로젝트: 기존 것 재사용 vs 신규 생성 결정

---

## 4. 주요 리스크 모니터링 포인트

PRD §7 중 실제 발생 가능성이 높은 것:

- **GPU 메모리**: Phase 0에서 먼저 smoke test. 막히면 즉시 Colab Pro로 피벗
- **VARAG 의존성 파손**: fork 시점 고정(commit SHA 명시). pip freeze로 lockfile
- **Hybrid가 Vision-only보다 나쁨**: Phase 3에서 드러나도 블로그 주제로 전환 (PRD §7 대응안). "왜 실패했는가" 회고가 오히려 차별화
- **Sanofi 업무 충돌**: 주 8시간 기준으로 Phase별 데드라인 재조정 기준값 → plan.md에 주 단위 점검 추가

---

## 5. Phase 0 완료 상태 (2026-04-22)

**전부 완료**:
- [x] 디렉토리 구조 + `.gitkeep` + `.gitignore`
- [x] GitHub 레포 + git (커밋 `phase-0-scaffold`)
- [x] Sanofi PDF 4개 수집 (`data/pdf/`)
- [x] `README.md`, `requirements.txt`, `docker-compose.yml`, `.env.example`
- [x] `.env` 로컬 키 6개 설정 (Anthropic, HF, Langfuse US, Qdrant 포트 6335)
- [x] **Colab Nemotron smoke test 통과** — T4 + 3B-v2 + bf16 + float-only cast. 임베딩 shape `[N, 3072]`, MaxSim 12.88
- [x] **로컬 Qdrant 기동** (docker-compose, 포트 6335로 분리) + single/multi-vector 연결 smoke test
- [x] **로컬 Docling smoke test** (`scripts/00_docling_smoke.py`) — Q1.pdf 23페이지 → 306 text + 19 table + 2 picture 추출. 첫 표가 마크다운 구조 보존.
- [x] `scripts/01_qdrant_smoke.py` — 단일/멀티벡터 컬렉션 생성·검색·삭제 검증

**Phase 0 산출물 총평**:
- ✅ 4-mode 비교 구조의 **모든 컴포넌트가 각자 smoke test 통과** (Docling, Qdrant multi-vector, Nemotron ColEmbed V2)
- ✅ `.gitignore`로 시크릿·모델·PDF 차단. 리포지토리는 공개 가능 상태
- ⚠️ 로컬 Docling은 일부 페이지에서 CPU RAM 한계로 `std::bad_alloc` — Phase 1에서 페이지별 배치 처리로 우회
- ⚠️ Colab Pro 컴퓨트 유닛 이번 달 소진 → 다음 달 재충전 시점까지 T4 무료 티어 or Kaggle 대기

## 6. Phase 1 진행 상황

**완료**:
- [x] VARAG 레포 참고용 clone (`c:/Users/user/Desktop/VARAG-ref/`)
- [x] VARAG 코드 리뷰 + `docs/VARAG_REVIEW.md` 작성
- [x] `generator/claude_vision.py` — Claude Sonnet 4.6 Vision + 프롬프트 캐싱 검증 (tag: phase-1-generator)
- [x] `generator/claude_text.py` — Claude Haiku 4.5 (translation, HyDE 드래프트)
- [x] `utils/pdf.py` — pypdfium2 기반 렌더 헬퍼 (Windows-friendly, no Poppler)
- [x] `retriever/docling_text.py` — Docling + BGE-M3 + Qdrant 인덱서·검색기 (tag: phase-1-text-retriever)
- [x] `retriever/text_baseline.py` / `text_qt.py` / `text_hyde.py` — Text 경로 3버전 (tag: phase-1-text-variants)
- [x] `rerank/zerank2.py` — BGE-reranker-v2-m3 cross-encoder (PRD의 "ZeRank2" 라벨 매핑)
- [x] `modes/text_only.py` — 첫 모드 + Langfuse @observe 통합

**남음 (Colab 의존)**:
- [ ] `retriever/nemotron.py` — Colab tunnel client (FastAPI + pypdfium2)
- [ ] `rerank/nemotron_rerank.py` — `llama-nemotron-rerank-vl-1b-v2` (Colab 노트북에 엔드포인트 추가)
- [ ] `modes/vision_only.py` — Nemotron retriever + Claude Vision (페이지 이미지)
- [ ] `modes/caption.py` — Haiku 캡션 → BGE-M3 인덱싱 → 텍스트 검색 → Sonnet
- [ ] `modes/hybrid.py` + `router/langgraph.py` — text + vision 라우팅 + 리랭크 합병
- [ ] 4-mode × 3-query smoke test — Langfuse에 12 trace 확인
