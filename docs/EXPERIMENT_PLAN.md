# 실험 계획서 — Phase 2·3 (현행)

> `plan.md`의 Phase 2·3를 Phase 1 결과와 2026-09-16 검토를 반영해 다시 쓴 것. 이 문서가 현행이며
> `plan.md`는 이력으로만 남긴다. 상태 표기: ✅ 완료 · 🔶 부분 · ❌ 미착수

## 1. 목표와 핵심 주장

**연구 질문**: 차트·표 밀집 제약 공시 PDF에서, 한국어 질문 → 영문 문서 크로스링구얼 RAG를 할 때
텍스트 경로를 최대한 최적화(QT + HyDE + reranker)해도 **Vision / Hybrid 경로의 우위가 남는가?**

**부가 질문** (핵심 주장에 종속, 별도 축 아님):
- 질문 유형(차트/표/본문/멀티홉)별로 어느 모드가 이기는가
- KR 질문과 EN 질문의 격차가 모드별로 얼마나 다른가
- 기간이 애매한 질문에서 연간(20-F) vs 분기(PR) 페이지를 구분하는가

**실험 축 축소 결정 (2026-09-16)**: 30문항 벤치마크 위에 ablation 4축을 쌓으면 축 하나당 결론이
노이즈(1문항 = 3.3%p)에 묻힌다. 아래 §3의 핵심 매트릭스만 본편으로 하고 나머지는 부록/후속으로 뺀다.

## 2. 데이터 — 4개 층

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

### 2.3 인덱스 (3 컬렉션 × 7 파일) — 🔶 21칸 중 1칸

| | Q1 | Q2 | Q3 | 20-F 발췌 |
|---|---|---|---|---|
| `pharma_text` (BGE-M3) | 🔶 20/23p | ❌ | ❌ | ❌ |
| `pharma_vision` (Nemotron 3B) | 🔶 **6/23p** | ❌ | ❌ | ❌ |
| `pharma_caption` (Haiku→BGE-M3) | ✅ 23/23p | ❌ | ❌ | ❌ |

비전 인덱스는 §4의 배치 방식으로 전환한 뒤 전부 다시 채운다 (터널 방식은 `render_scale` 0.85로
품질이 깎여 있으므로 1.5로 재인덱싱).

### 2.4 평가셋 `eval/questions.jsonl` — ❌ 0/30

```json
{"id": "A03", "type": "A", "block_type": "chart",
 "q_ko": "2025년 2분기 Dupixent 매출은 전년 동기 대비 몇 % 성장했나?",
 "q_en": "How much did Dupixent sales grow year-over-year in Q2 2025?",
 "answer": "…% (CER 기준), €… ",
 "gold_pages": [{"source": "Q2.pdf", "page": 5}],
 "period_spec": "explicit",
 "notes": "차트 축 라벨에서만 읽힘. 본문에는 CER 성장률만 있음"}
```

| 필드 | 값 |
|---|---|
| `type` | A 차트 10 / B 표 10 / C 본문 5 / D 멀티홉 5 |
| `block_type` | 정답이 위치한 블록: `chart` / `table` / `text` (D는 복수 가능) |
| `period_spec` | `explicit` 25개(gold_pages 단일) / `ambiguous` 5개(gold_pages 복수 허용) |
| `gold_pages` | `page_inventory.csv`와 대조해 검증. 20-F 발췌본은 **발췌 후 페이지 번호** 사용 |

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

## 4. 인프라 선행 과제 — 비전 인덱싱 경로 교체 (❌, Phase 2 첫 작업)

**문제**: 현재 `retriever/nemotron.py`는 Colab FastAPI 터널에 페이지 이미지를 보내고 `[N_patches, 3072]`
임베딩(~14MB/페이지)을 받아온다. Cloudflare 무료 터널이 30MB에서 502를 내고 throughput이 페이지당
~3분이라 172p 인덱싱이 사실상 불가능하다.

**해법**: 임베딩을 인터넷으로 왕복시키지 않는다.

1. `notebooks/03_nemotron_batch_index.ipynb` — Colab에서 PDF 4개를 업로드/마운트 → 전 페이지
   `render_scale=1.5`로 렌더 → Nemotron 3B `forward_images` 배치 → 페이지당 fp16 배열을
   `{source}_{page}.npy`로 저장 → `embeddings.zip` 하나로 다운로드. (예상: 페이지당 ~5초 → 172p ≈ 15분)
2. `scripts/11_load_nemotron_npz.py` — zip 해제 → `NemotronVisionRetriever.qdrant.upsert` (gRPC)로 로컬 업서트.
   포인트 ID는 기존 `_page_id(source, page)` 재사용 → idempotent.
3. 쿼리 시점 임베딩(`embed_query`)은 여전히 Colab 터널이 필요 (질문 60개 × 텍스트라 가벼움).
   벤치마크 실행 전에 **60개 질문의 쿼리 임베딩도 같은 노트북에서 미리 뽑아 `.npz`로 받아두면** 벤치마크
   중 Colab 의존을 완전히 끊을 수 있다. 권장.

기존 `06_index_nemotron.py`(터널 방식)는 남겨두되 docstring에 deprecated 표기.

## 5. 실행 순서와 실행 환경

| 순서 | 작업 | 환경 | 산출물 |
|---|---|---|---|
| 1 ✅ | 20-F 300p + 덱 3개 페이지 유형 스캔 → 발췌 50p + 15p × 3 확정 | 로컬 (pypdfium2 렌더 + Claude 확인) | `scripts/11_build_extracts.py`, `data/pdf/20F_extract.pdf`, `Q{n}_deck.pdf`, `docs/CORPUS.md` |
| 2 | 코퍼스 172p 페이지 인벤토리 | 로컬 | `eval/page_inventory.csv` |
| 3 | 질문 30 × 2 작성 + 레이블 + 검증 스크립트 | 로컬 | `eval/questions.jsonl`, `scripts/12_validate_questions.py` |
| 4 | Colab 배치 인덱싱 노트북 + npz 로더 | Colab + 로컬 | `notebooks/03_*.ipynb`, `scripts/13_load_nemotron_npz.py` |
| 5 | 텍스트·캡션 인덱스 7파일 전부 (Docling bad_alloc 페이지 배치 재처리 포함) | 로컬 (+API) | Qdrant 3컬렉션 채움 |
| 6 | `eval/metrics.py`(Recall@k, NDCG@k, 정답 일치 judge) + `eval/runner.py` | 로컬 | `python -m pharma_vision_rag.eval.runner --mode all` → CSV |
| 7 | E1·E2 실행 (검색 1회 → 생성 3회) | 로컬 (+API) | `eval/results/*.csv` (gitignore) |
| 8 | E3 재집계 + 그래프 + 실패 케이스 10개 정성 분석 | 로컬 | `docs/REPORT.md` |

1~3은 API·GPU 없이 PDF만 있으면 된다. 4는 Colab 필요. 5~7은 로컬 Qdrant + Anthropic 키 필요.

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

- [ ] 20-F 발췌 50p의 구체 페이지 범위 (순서 1에서 확정)
- [ ] 리랭커를 E1 hybrid에 기본 포함할지 (Phase 1에서는 text_only에만 붙어 있음)
- [ ] 예산 초과 시 3회 반복을 KR 30문항으로 한정할지
- [ ] 임상 유형(C) 질문 5개가 20-F 서술부만으로 충분한지 → 부족하면 Dupilumab PMC 논문 1편 추가 (plan.md 보류 항목)
- [ ] README의 "Nemotron 4B / ZeRank2 / Claude Sonnet 4.7" 표기 → 실제 모델로 정정 (2026-09-16 일부 정정)
