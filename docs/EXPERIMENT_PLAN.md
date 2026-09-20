# 실험 계획서 — Phase 2·3 (현행)

> `plan.md`의 Phase 2·3를 Phase 1 결과와 2026-09-16 검토를 반영해 다시 쓴 것. 이 문서가 현행이며
> `plan.md`는 이력으로만 남긴다. 상태 표기: ✅ 완료 · 🔶 부분 · ❌ 미착수

## 0. 현황 요약 (2026-09-20, 코퍼스 v2). 이 절이 아래 2.1~2.3절, 4절, 5절의 상태 표기보다 우선함

**코퍼스를 172p에서 1,709p로 확대** (27개 문서, `eval/corpus.json`이 기준, 상세 `docs/CORPUS.md` 0절).
172p는 긴 컨텍스트에 통째로 들어가서 RAG를 쓸 이유를 설명할 수 없었음. v2는 Sanofi 2024와 2025
(보도자료, 슬라이드, 20-F 전체)와 Novartis, Roche, AstraZeneca 2025 Q1~Q3 슬라이드. 분기, 연도, 회사만 다른
"쌍둥이 페이지"가 검색을 실제로 어렵게 만듦. 텍스트만 약 100만 토큰이라 컨텍스트에 들어가지 않음.

| 항목 | 상태 |
|---|---|
| 코퍼스 v2 수집과 매니페스트 | ✅ `scripts/11_build_corpus.py`, 1,709p |
| 평가셋 | ✅ 60문항. v2 문서 id와 전체본 페이지 번호로 이전. gold는 "정답이 실린 모든 페이지" 기준으로 보완, 멀티홉은 `gold_groups` |
| 평가 코드 | ✅ `eval/metrics.py`(그룹 단위 Recall@k, NDCG@k), `eval/runner.py`(text, text_rerank, vision, caption, hybrid) |
| 텍스트 인덱스 | 🔶 v1(172p)만 있음. v2는 RunPod에서 Docling 블록 추출 후 `scripts/15_rebuild_text_index.py` |
| 비전 | ❌ RunPod 실행 대기. `scripts/embed_pages_gpu.py` → `scripts/13_score_vision_exact.py` |
| 캡션 인덱스 | ❌ Anthropic 키 필요. 1,709p 약 $5 |
| 평가셋 확장 | ✅ 60문항(A20, B20, C10, D10, 모호 10). 경쟁사 차트, 회사 간 멀티홉 포함. A03과 A06은 Roche 차트 문항으로 교체. 외부 리뷰어 검토는 미완 |
| agentic 모드 (E4) | ❌ 계획 확정, 3.3절. E1 기준선 확보 후 착수 |

**v1(172p) 텍스트 경로 기준선** (30문항 × 한/영, `eval/results/`는 gitignore):

| 변형 | R@1 | R@5 | A 차트 R@5 | B 표 R@5 | C 본문 R@5 | D 멀티홉 R@5 |
|---|---|---|---|---|---|---|
| dense (BGE-M3) | 0.31 | 0.67 | 0.55 | 0.80 | 0.70 | 0.63 |
| dense + bge-reranker | 0.40 | 0.72 | 0.40 | 0.90 | 1.00 | 0.72 |

리랭커는 표와 본문을 끌어올리지만 차트는 오히려 떨어뜨림(텍스트가 적은 슬라이드를 뒤로 밀어냄). v2에서 다시 측정.

**비전 검색 설계 변경**: 페이지당 패치 벡터 전체를 Qdrant 멀티벡터로 넣으면 1,709p에 약 24GB(fp32). 벤치마크는
질의가 고정이므로 `.npy` 원본을 한 번 훑으며 전 질의의 MaxSim을 정확히 계산함(벡터 DB 없음, 근사 없음).
대화형 앱용 2단계 검색(페이지 평균 벡터로 후보 추림 후 MaxSim)은 이 정확한 순위를 기준으로 검증할 후속 과제.

**데이터 보존 원칙**: Qdrant는 파생 인덱스일 뿐. 기준 데이터는 `data/pdf/`, `data/embeddings/text_blocks.jsonl`,
`data/embeddings/v2/`(RunPod 산출물). 2026-09-19에 Windows 바인드 마운트 볼륨에서 벡터가 전부 0이 되는 사고가 있었고
compose를 named volume으로 변경함.

## 1. 목표와 핵심 주장

**연구 질문**: 차트·표 밀집 제약 공시 PDF에서, 한국어 질문 → 영문 문서 크로스링구얼 RAG를 할 때
텍스트 경로를 최대한 최적화(QT + HyDE + reranker)해도 **Vision / Hybrid 경로의 우위가 남는가?**

**부가 질문** (핵심 주장에 종속, 별도 축 아님):
- 질문 유형(차트/표/본문/멀티홉)별로 어느 모드가 이기는가
- KR 질문과 EN 질문의 격차가 모드별로 얼마나 다른가
- 기간이 애매한 질문에서 연간(20-F) vs 분기(PR) 페이지를 구분하는가

**실험 축 축소 결정 (2026-09-16)**: 30문항 벤치마크 위에 ablation 4축을 쌓으면 축 하나당 결론이
노이즈(1문항 = 3.3%p)에 묻힌다. 아래 §3의 핵심 매트릭스만 본편으로 하고 나머지는 부록/후속으로 뺀다.

## 2. 데이터 — 4개 층 (2.1~2.3은 v1 기록. 현재 구성은 0절과 `docs/CORPUS.md` 0절)

### 2.1 원본 PDF (`data/pdf/`, gitignore) — ✅ 확보됨 (2026-09-16 재다운로드, `docs/CORPUS.md` §5 URL)

| 파일명 (정확히) | 페이지 | 출처 |
|---|---|---|
| `Q1.pdf` | 23 | Sanofi IR · Q1 2025 Press Release |
| `Q2.pdf` | 26 | Sanofi IR · Q2 2025 Press Release |
| `Q3.pdf` | 28 | Sanofi IR · Q3 2025 Press Release |
| `Form 20-F 2025 (Oct 2025).pdf` | 300 | Sanofi IR · Form 20-F (FY2025, Jan–Dec) |
| `raw/Q1_deck_full.pdf` | 41 | Sanofi IR · Q1 2025 Results Presentation (**09-16 추가**) |
| `raw/Q2_deck_full.pdf` | 41 | Sanofi IR · Q2 2025 Results Presentation (**09-16 추가**) |
| `raw/Q3_deck_full.pdf` | 43 | Sanofi IR · Q3 2025 Results Presentation (**09-16 추가**) |

**덱 추가 사유 (09-16)**: 보도자료와 20-F 전 페이지 렌더 확인 결과 **차트 0개**(표와 텍스트만). A유형(차트) 10문항과
"차트·표 밀집" 전제가 성립하지 않아 같은 IR 페이지의 실적 발표 슬라이드에서 차트 페이지만 발췌해 추가. 상세 `docs/CORPUS.md` §1.

### 2.2 파생 데이터 — 🔶 발췌본 완료, 인벤토리 미착수

| 항목 | 요건 | 산출물 |
|---|---|---|
| **20-F 발췌본** ✅ | 300p 중 **50p** (서술 14 / 혼합 8 / 표 28). 재무제표·파이프라인 표와 서술 섹션을 섞어 특정 모드에 유리하지 않게 | `data/pdf/20F_extract.pdf` + `docs/CORPUS.md` §2 |
| **덱 발췌본** ✅ | 덱당 15p, 차트 주 콘텐츠 페이지 + 요약 표 3~4장. 표지·사진·약어표 제외 | `data/pdf/Q{1,2,3}_deck.pdf` + `docs/CORPUS.md` §3 |
| **페이지 인벤토리** ❌ | 코퍼스 전 페이지(172p)에 주 콘텐츠 유형 태깅: `chart` / `table` / `text` / `mixed`. 질문 유형 분포를 맞추고 gold_page를 검증하는 기준 | `eval/page_inventory.csv` (source, page, block_type, note) |

빌드: `scripts/11_build_extracts.py` (페이지 목록 상수, idempotent).

최종 코퍼스 = 23 + 26 + 28 + 50 + 15 × 3 = **172페이지** (7파일). 참고 벤치마크(ViDoRe task당 수백~1,000p,
FinanceBench 150문항) 대비 작은 편이지만 "파일럿 벤치마크"로 방어 가능. **여기서 더 늘리지 않는다.**

**정답 중복 규칙**: 덱 차트 수치가 같은 분기 보도자료 표에도 있으면 text_only가 그 페이지로 맞혀 A유형 검증이 무효.
A유형은 보도자료 텍스트에 없는 수치(세그먼트, 브릿지 항목, 추세)만 출제하고, `gold_pages`에 정답이 있는 페이지를 전부 기록. `docs/CORPUS.md` §4.

### 2.3 인덱스 (3 컬렉션 × 7 파일) — 🔶 21칸 중 7칸 (2026-09-17)

| | Q1 | Q2 | Q3 | 20-F 발췌 | Q1 덱 | Q2 덱 | Q3 덱 |
|---|---|---|---|---|---|---|---|
| `pharma_text` (BGE-M3) | ✅ 339 | ✅ 377 | ✅ 393 | ✅ 806 | ✅ 238 | ✅ 211 | ✅ 213 |
| `pharma_vision` (Nemotron 3B) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `pharma_caption` (Haiku→BGE-M3) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

`pharma_text`: 2,577청크, 172/172p 커버, 실패 페이지 0 (`scripts/14_index_text_all.py`, 약 70분 CPU).
청킹 규칙: Docling 텍스트 블록과 표(마크다운)를 **1,500자에서 줄 단위 분할**, BGE-M3 `max_seq_length=1024`.
분할 전에는 부록 표 한 청크가 8,929자까지 커서 CPU 임베딩이 병목이었음(15분/5p → 6분/5p).
Phase 1 `pharma_text`(Q1 20/23p, 319청크)와 `pharma_caption`(Q1 23p)은 예전 클론의 Qdrant 볼륨에 있어 소실. 전부 재구축.
비전은 §4 RunPod 경로로 채움. 캡션은 `08` 스크립트에 7파일 지정해 재실행(Anthropic 키 필요).

### 2.4 평가셋 `eval/questions.jsonl` — ✅ 30/30 (2026-09-16, `scripts/12_validate_questions.py` 통과)

```json
{"id": "A03", "type": "A", "block_type": "chart",
 "q_ko": "2024년 2분기 독감(Flu) 백신 매출은 얼마였나?",
 "q_en": "What were influenza (Flu) vaccine sales in Q2 2024?",
 "answer": "€115 million (Q2 2024). Q2 2025 was €141 million.",
 "answer_keys": ["115"],
 "gold_pages": [{"source": "Q2_deck.pdf", "page": 4}],
 "period_spec": "explicit", "needs_review": false, "visual_only": false,
 "notes": "Q2 deck vaccines stacked bar, prior-year segment. Press release gives only Q2 2025 value 141."}
```

| 필드 | 값 |
|---|---|
| `type` | A 차트 10 / B 표 10 / C 본문 5 / D 멀티홉 5 |
| `block_type` | 정답이 위치한 블록: `chart` / `table` / `text` |
| `period_spec` | `explicit` 25개 / `ambiguous` 5개(gold_pages 복수, 하나만 잡혀도 hit). ambiguous는 A09, A10, B09, B10, C05 |
| `gold_pages` | `page_inventory.csv`와 대조해 검증. 발췌본(20-F, 덱)은 **발췌 후 페이지 번호** |
| `answer_keys` | 정답의 핵심 문자열. 검증 스크립트가 gold page 텍스트 레이어에서 존재 확인 (judge 보조용) |
| `visual_only` | 정답이 텍스트 레이어에 없음(래스터 라벨, 막대 높이 읽기). A04, A08 |
| `needs_review` | 렌더로 확인 못 한 값. 현재 0개 |

**정답 중복 검사**: 검증 스크립트가 A유형 answer_keys를 보도자료 전 페이지에서 검색해 경고. 현재 경고는 전부 우연한
부분 문자열 일치(환율표의 98.637, 주식수 1,218.1 등)로 확인됨. 실제 중복이던 A07의 CER 매출 13,012(Q3.pdf p25)와
A04의 +52.5%(Q2.pdf p2)는 질문에서 제거.

작성 규칙:
- 25개 explicit 질문은 "FY2025 / Q2 2025"처럼 기간을 명시한다 (D8·D9: 연간 vs 분기 granularity 혼재).
- 5개 ambiguous 질문("Dupixent 매출 추세는?")은 temporal disambiguation 실험용. 정답은 복수 페이지 중 하나면 정답.
- 인턴 업무에서 실제 나올 법한 질문으로 (면접 토킹 포인트).
- 외부 리뷰어 1명이 한국어 질문 품질 검토 (plan.md 요건 유지).
- 정답 수치는 통화·단위·기간까지 verbatim (`€3.5 billion`, `+20.3% CER`).

## 3. 실험 매트릭스

### 3.1 본편 (핵심 주장 검증)

| 실험 | 구성 | 런 수 | 반복 | 비고 |
|---|---|---|---|---|
| **E1 4모드 비교** | text_only / vision_only / caption / hybrid × 60질문 | 240 | 검색 1회, 생성 3회 | 핵심 |
| **E2 텍스트 최적화 ablation** | text_only 내부 baseline / +QT / +HyDE (+reranker on/off) × 60질문 | 120~240 | 검색 1회 | "최적화해도 남는가"의 왼쪽 항 |
| **E3 언어·유형 breakdown** | E1·E2 결과를 KR/EN × A/B/C/D로 재집계 | 0 (재집계) | — | 추가 런 없음 |

**반복 원칙**: 검색 지표(Recall/NDCG)는 결정적이므로 1회. 생성 지표(정답 일치)만 Sonnet 변동성 때문에 3회.
→ Sonnet 호출 = 240 × 3 = **720회**. 프롬프트 캐싱 시 ~$20 예상. 예산 초과 시 3회 반복은 KR 30문항에만 적용.

### 3.2 부록 / 후속 (본편 결과 나온 뒤 결정)

| 실험 | 이유로 뺌 |
|---|---|
| 리랭커 A/B (bge-reranker vs Nemotron rerank) | Nemotron 리랭커 **미구현**. 30문항으로 두 리랭커 차이를 가리기 어려움 |
| 질문 표현 최적화 파일럿 (키워드/구체/질문형) | 축 하나 더 얹으면 "데이터는 작은데 매트릭스는 논문급"으로 읽힘 |
| Hybrid 라우팅 결정 로그 분석 | 라우터가 키워드 휴리스틱이라 분석 가치 낮음. 라우터를 LLM/조건부 엣지로 바꾼 뒤에 의미 있음 |
| Qwen3-VL 생성기 비교 | PRD stretch. D4에 따라 보류 |

### 3.3 E4: agentic 모드 (2026-09-20 추가, 기준선 확보 후 착수)

**왜**: 고정 파이프라인의 실패 유형이 "한 번 검색하고 끝"이라는 구조에서 나온다. v1 벤치마크에서 확인한 것:
1위 오답의 상당수가 분기나 연도만 다른 쌍둥이 페이지, 멀티홉(D)은 상위 5장에 모든 홉이 들어오지 않음,
기간이 모호한 질문은 아무 분기나 가져옴, hybrid의 라우터는 키워드 if문. 코퍼스 v2(1,709p, 4개사, 2개 연도)에서 더 심해진다.
**위치**: 기존 4모드를 대체하지 않고 **다섯 번째 비교 대상**으로 추가한다. 같은 60문항으로
"고정 파이프라인 대비 정답률이 얼마나 오르고 비용과 시간이 몇 배 드는가"를 측정한다.
이 모드에 한해서만 대외 문서에서 "agentic"이라는 표현을 쓴다.

**구조**: Claude의 tool use로 직접 구현(LangGraph 불필요). 모델은 답변 생성과 같은 Sonnet. 최대 도구 호출 10회.

| 도구 | 입력 | 동작 |
|---|---|---|
| `list_documents` | 없음 | `eval/corpus.json`의 문서 id, 회사, 기간, 종류, 페이지 수 |
| `search_text` | query, 선택적 문서 id 필터 | `pharma_text` dense 검색 후 리랭커, 상위 청크와 (문서, 페이지) |
| `search_pages` | query, 선택적 문서 id 필터 | `pharma_caption` 검색. 차트 페이지로 가는 다리 역할 |
| `open_page` | 문서 id, 페이지 | 페이지 이미지를 렌더해 모델에 보여줌. 차트는 여기서 읽음 |
| `calculate` | 수식 | 증감액, 비율. 환각 방지용 |
| `final_answer` | 답변, 인용 페이지 목록 | 종료 |

비전(Nemotron) 검색은 도구로 넣지 않는다. agent가 즉석에서 만든 검색어는 미리 임베딩할 수 없고, 3B 모델을 개발 PC의
CPU(RAM 16GB)로 돌릴 수 없기 때문. GPU 엔드포인트를 상시로 띄울 수 있게 되면 `search_vision`을 추가하는 것이 후속 과제.

**측정**:
- 정답 일치(Haiku judge, 3회 평균). 기존 모드와 같은 기준
- 인용 페이지의 gold 적중(`final_answer`의 인용 목록을 `eval/metrics.py`의 그룹 기준으로 채점)
- 질문당 도구 호출 수, 입력과 출력 토큰, 비용, 소요 시간
- 유형별(A, B, C, D), 기간 명시 여부별, 한/영별 재집계. 가설: D와 ambiguous에서 가장 크게 오르고 C는 차이 없음

**비용 추정**: 질의 120개 × 호출 5~10회 × 3회 반복. 프롬프트 캐싱 전제 약 $15~30. 실제 값은 10문항 파일럿으로 먼저 확인.
**선행 조건**: `ANTHROPIC_API_KEY`, `pharma_text`와 `pharma_caption`의 v2 인덱스, E1 기준선 결과.
**산출물**: `src/pharma_vision_rag/modes/agentic.py`, `eval/runner.py`의 생성 단계에 mode 추가, `docs/REPORT.md`의 비교 절.
**하지 않는 것**: 멀티 에이전트, 장기 메모리, 웹 검색. 단일 agent와 다섯 도구로 한정한다.

## 4. 인프라 선행 과제 — 비전 인덱싱 경로 교체 (🔶 코드 완료, 실행 대기 — 2026-09-16)

**문제**: 현재 `retriever/nemotron.py`는 Colab FastAPI 터널에 페이지 이미지를 보내고 `[N_patches, 3072]`
임베딩(~14MB/페이지)을 받아온다. Cloudflare 무료 터널이 30MB에서 502를 내고 throughput이 페이지당
~3분이라 172p 인덱싱이 사실상 불가능하다.

**해법**: 임베딩을 인터넷으로 왕복시키지 않는다.

1. ✅ `scripts/embed_pages_gpu.py` — **RunPod**(Colab 아님, 09-16 결정) GPU 박스에서 단독 실행. PDF 7개 →
   전 페이지 `render_scale=1.5` 렌더 → Nemotron 3B `forward_images` 배치 → `pages/<source>/<page>.npy`(fp16) +
   `queries/<id>_<ko|en>.npy` 60개 + `manifest.json` → `embeddings.zip`. 재실행 시 기존 `.npy` 건너뜀.
   (예상: 4090 약 10분, T4급 약 15분. zip 약 2.4GB)
2. ✅ `scripts/13_load_nemotron_npz.py` — zip 해제 → `NemotronVisionRetriever.qdrant.upsert` (gRPC)로 로컬 업서트.
   포인트 ID는 기존 `_page_id(source, page)` 재사용 → idempotent. 더미 임베딩으로 스모크 테스트 통과.
3. ✅ 쿼리 임베딩 60개도 같은 스크립트에서 뽑아 zip에 포함 → 벤치마크 중 GPU 의존 없음.
   `eval/runner.py`(순서 6)는 `data/embeddings/<zip이름>/queries/`를 읽는 오프라인 쿼리 클라이언트를 쓴다.

`06_index_nemotron.py`(터널 방식)는 docstring에 deprecated 표기. `02_nemotron_tunnel.ipynb`는 fallback으로만.

## 5. 실행 순서와 실행 환경

| 순서 | 작업 | 환경 | 산출물 |
|---|---|---|---|
| 1 ✅ | 20-F 300p + 덱 3개 페이지 유형 스캔 → 발췌 50p + 15p × 3 확정 | 로컬 (pypdfium2 렌더 + Claude 확인) | `scripts/11_build_extracts.py`, `data/pdf/20F_extract.pdf`, `Q{n}_deck.pdf`, `docs/CORPUS.md` |
| 2 ✅ | 코퍼스 172p 페이지 인벤토리 (chart 27 / table 73 / text 35 / mixed 37) | 로컬 | `eval/page_inventory.csv` |
| 3 ✅ | 질문 30 × 2 작성 + 레이블 + 검증 스크립트 (외부 리뷰어 검토는 미완) | 로컬 | `eval/questions.jsonl`, `scripts/12_validate_questions.py` |
| 4 🔶 | GPU 배치 임베딩 스크립트 + npz 로더 (코드 완료, RunPod 실행은 사용자) | RunPod + 로컬 | `scripts/embed_pages_gpu.py`, `scripts/13_load_nemotron_npz.py` |
| 5 | 텍스트·캡션 인덱스 7파일 전부 (Docling bad_alloc 페이지 배치 재처리 포함) | 로컬 (+API) | Qdrant 3컬렉션 채움 |
| 6 | `eval/metrics.py`(Recall@k, NDCG@k, 정답 일치 judge) + `eval/runner.py` | 로컬 | `python -m pharma_vision_rag.eval.runner --mode all` → CSV |
| 7 | E1·E2 실행 (검색 1회 → 생성 3회) | 로컬 (+API) | `eval/results/*.csv` (gitignore) |
| 8 | E3 재집계 + 그래프 + 실패 케이스 10개 정성 분석 | 로컬 | `docs/REPORT.md` |

1~3은 API·GPU 없이 PDF만 있으면 된다. 4는 RunPod GPU 필요. 5~7은 로컬 Qdrant + Anthropic 키 필요.

## 6. 지표와 보고 규칙

**검색**: Recall@1/3/5, NDCG@5 — gold_pages 기준. ambiguous 질문은 gold 중 하나라도 잡히면 hit.
**생성**: 정답 일치 (Haiku judge: 수치·단위·기간 일치 여부 3단계 — 정답/부분/오답). 3회 평균.
**비용**: 인덱싱 $/페이지, 쿼리 $/질문, 저장 용량 (모드별).

보고 규칙 (30문항 벤치마크의 정직성 확보):
1. 집계 %와 함께 **질문별 30행 결과표를 그대로 공개** — "어떤 질문에서 갈렸는가"가 진짜 콘텐츠.
2. 생성 지표는 **평균 ± 범위(3회)**로 표기.
3. **3문항(10%p) 이내 차이는 유의하지 않음**을 리포트 서두에 선언.
4. 실패 케이스 10개는 원인 태그: OCR/레이아웃 손실 · 리트리버 miss · 리랭커 강등 · VLM 환각 · 언어 mismatch · 기간 혼동.

## 7. 미결 사항

- [x] 20-F 발췌 50p의 구체 페이지 범위 → `docs/CORPUS.md` §2 (2026-09-16 확정)
- [ ] 리랭커를 E1 hybrid에 기본 포함할지 (Phase 1에서는 text_only에만 붙어 있음)
- [ ] 예산 초과 시 3회 반복을 KR 30문항으로 한정할지
- [ ] 임상 유형(C) 질문 5개가 20-F 서술부만으로 충분한지 → 부족하면 Dupilumab PMC 논문 1편 추가 (plan.md 보류 항목)
- [ ] README의 "Nemotron 4B / ZeRank2 / Claude Sonnet 4.7" 표기 → 실제 모델로 정정 (2026-09-16 일부 정정)
