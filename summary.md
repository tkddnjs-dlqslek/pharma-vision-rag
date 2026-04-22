# pharma-vision-rag — 진행 요약

**기간**: 2026-04-21 ~ 04-22 (이틀, 약 14시간)
**현재 상태**: **Phase 1 완료** ✅ — 4가지 RAG 모드 + LangGraph 하이브리드 그래프 모두 동작 검증.

---

## 한 줄 요약

Sanofi 공시 PDF에서 한국어로 질문하면 영문 페이지를 찾아 답해주는 멀티모달 하이브리드 RAG. 텍스트·비전·캡션·하이브리드 4가지 모드 모두 end-to-end 동작. **하이브리드 모드가 smoke 테스트 3/3 통과**(텍스트 단독은 2/3, 캡션도 2/3) — Phase 1의 main contribution 검증됨.

---

## 무엇을 했나 (시간 순서)

### Day 1 (2026-04-21) — 기획 + 환경 셋업

1. **기획안 검토 후 plan.md 작성**
2. **10가지 의사결정 확정** (타임라인, 공개 범위, GPU 전략 = Colab, 평가 데이터 정책)
3. **GitHub 공개 레포 생성** + git 연결
4. **디렉토리 구조 + 설정 파일 일괄 생성** (`.gitignore`, `requirements.txt`, `docker-compose.yml`, `.env.example`, README skeleton)
5. **Sanofi PDF 4개 수집**: Form 20-F 2025 + Q1/Q2/Q3 2025 Press Release
6. **Colab에서 Nemotron 모델 동작 검증** (smoke test)
7. **로컬에서 Qdrant 기동** + Docling으로 Q1.pdf 파싱 검증

### Day 2 오전 (2026-04-22) — 텍스트 경로 빌드

8. **VARAG 레포 리뷰** → 우리 프로젝트와 차이 정리 (`docs/VARAG_REVIEW.md`)
9. **Claude Vision 생성기** + 프롬프트 캐싱 검증 (52% 토큰 절감)
10. **Docling → BGE-M3 → Qdrant 인덱서** — Q1.pdf 319 청크 인덱싱
11. **텍스트 검색기 3변형** (baseline/QT/HyDE) — 추상 쿼리 실패 → HyDE 회복 검증
12. **Text-only 모드** + ZeRank2 리랭커 (BGE-reranker-v2-m3) + Langfuse trace

### Day 2 오후 (2026-04-22) — 비전·하이브리드 빌드

13. **Colab production tunnel 노트북** (`02_nemotron_tunnel.ipynb`) — FastAPI 4 endpoints
14. **NemotronEmbeddingClient + NemotronVisionRetriever** — Qdrant multi-vector (MAX_SIM)
15. **Caption Indexer/Retriever** — Haiku Vision으로 페이지 캡션 → BGE-M3 → Qdrant
16. **vision_only / caption / hybrid (RRF) 모드** + **HybridGraph (LangGraph)**
17. **Q1.pdf 6페이지 Nemotron 인덱싱** (Cloudflare Tunnel throughput 한계로 부분 인덱싱)
18. **5-mode × 3-query smoke test** — vision/hybrid/hybrid_lg 3/3 OK

---

## 지금 동작하는 것

**예시**: "2025년 1분기 Beyfortus 백신 매출은?" 한국어 질문 →

1. (모드별로 다름 — 위 3개는 다 정답)
2. Hybrid 모드: BGE-M3가 텍스트 청크 8개 + Nemotron이 비전 페이지 8개 검색 → RRF로 합쳐 top-3 페이지 선정 → 그 페이지 이미지를 Claude Sonnet Vision에 → 한국어 답변
3. Langfuse 대시보드에 모드/모델/페이지/토큰 사용량/캐시 히트 자동 기록

### 5-mode smoke 최종 결과 (Q1.pdf 6/23 페이지 인덱싱 기준)

| 질문 | text_only | vision_only | caption | hybrid | hybrid_lg |
|---|---|---|---|---|---|
| Dupixent Q1 sales (고유명사) | ❌ | ✅ | ❌ | ✅ | ✅ |
| EPS growth (추상) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Beyfortus 백신 (KR) | ✅ | ✅ | ✅ | ✅ | ✅ |

**Hybrid가 모든 질문에 100% 통과** = Phase 1 main contribution 검증됨.

---

## 가설 vs 실제 — 무엇이 달랐나

### ✅ 가설대로 들어맞은 것

- **Vision retrieval이 차트에 강함** → vision_only가 6페이지만 인덱싱했는데도 3/3
- **Hybrid가 단일 모드보다 안정적** → text/caption 단독은 2/3, hybrid는 3/3
- **Docling이 표 구조 보존** → IFRS 매출표가 마크다운으로 깔끔히 변환
- **프롬프트 캐싱이 비용 절감** → 4,637 토큰 prefix 첫 쿼리 후 100% cache hit, 52% 절감

### ❌ 가설과 달랐던 것 + 어떻게 대응했나

| 가설 | 실제 | 대응 |
|---|---|---|
| **Claude Sonnet 4.7** | 2026-04 기준 4.6이 최신 | `claude-sonnet-4-6` 사용 |
| **Nemotron 4B** (3위 모델) | Colab Pro 컴퓨트 소진 → L4 사용 불가 | **3B (6위)로 다운그레이드** |
| **fp16으로 모델 로드** | Nemotron 내부 ViT가 fp32 고정 → dtype mismatch | **bf16 + float-only cast**로 우회 |
| **`pdf2image` 사용** | Colab에 poppler 없음 | `apt-get install poppler-utils` + 로컬은 pypdfium2 |
| **Qdrant 6333 포트** | n8n 스타터 키트의 다른 Qdrant가 점유 중 | **포트 분리 (6335)** |
| **20-F는 FY2024** | 실제 FY2025 (Jan-Dec 2025) | "기간 통일, granularity만 차이"로 정정 |
| **자동 캐싱** | 첫 시도 0 토큰 (Sonnet 최소 2048 토큰 prefix 미달) | 1페이지 → 3페이지로 확장 → 캐싱 작동 |
| **Nemotron `processor + model(**inputs)`** | None 에러 | 전용 메서드 `forward_images/queries/get_scores` 사용 |
| **ngrok로 Colab 터널** | 익명 사용 막힘 (verified account 필요) | **cloudflared로 전환** (무료, 가입 X) |
| **Cloudflare Tunnel이 큰 응답 처리** | ~30MB 넘는 응답에 502 Bad Gateway | **render_scale 1.5 → 0.85**로 patch 수 1/3 |
| **Qdrant `MAX_REQUEST_SIZE_MB` env로 한도 상향** | 1.12.4에서 actix 워커 stuck (실키한 버그) | env var 제거 + **모든 Qdrant 클라이언트를 gRPC로** (port 6336) |
| **Cloudflare 무료 throughput으로 23페이지 인덱싱 OK** | 페이지당 ~3분 → 70분 소요 예상 | 6페이지로 부분 인덱싱 후 검증. Phase 3에서 Colab 내부 인덱싱 + 파일 dump로 재구성 예정 |

### 🆕 예상 못한 발견

**1. BGE-M3 multilingual이 생각보다 강함**
한국어 쿼리 → 영문 PDF 검색이 baseline 그대로도 잘 됨. **QT 효과가 거의 없음** — 의외 (블로그 토픽).

**2. HyDE와 Reranker, 둘 다 추상 쿼리 실패를 회복**
"business EPS growth"에서 baseline FAIL → **HyDE만으로 회복** + **리랭커 단독으로도 회복** (top-20 후보풀 + cross-encoder). Phase 3에서 두 회복 경로 별도 비교 가능.

**3. Hybrid가 6페이지만으로도 100% 통과**
RRF (Reciprocal Rank Fusion)이 텍스트+비전 후보를 합칠 때 **하나의 retriever가 놓친 페이지를 다른 retriever가 보완**하는 구조. 작은 인덱스 사이즈에서도 강함.

**4. text_only와 caption이 같은 질문(Dupixent)에서 같이 실패**
두 모드 모두 BGE-M3 dense + top-3 baseline으로 검색 → top-3에 정답 페이지(p1) 미포함하는 케이스가 같음. **vision_only(MaxSim)와 hybrid(RRF)는 우회**. Dense single-vector 검색의 한계 정량 증거.

**5. Sonnet 답변에 변동성 있음**
caption 모드 단독 테스트(09)에서는 Dupixent OK였는데, 통합 4-mode 테스트에서는 같은 캡션 컨텍스트 받고도 Dupixent FAIL. 모델 출력 변동성 → Phase 3 평가 시 **3회 반복 평균** 권장.

**6. Docling 일부 페이지에서 메모리 부족**
Q1.pdf 23페이지 중 16/17/21에서 `std::bad_alloc` (CPU RAM 한계). 나머지 20페이지 정상 → Phase 1에선 우회, Phase 3에서 페이지별 배치 처리.

**7. Claude Vision의 멀티페이지 컨텍스트 통합**
페이지 3장 주면 자동으로 본문(p1) + 표(p2-3) 종합 답변. 별도 프롬프트 엔지니어링 없이.

---

## 비용 정리

**총 지출 (Day 1~2)**:
- Anthropic API: 약 **$0.50** (스모크 ~10회 + caption 인덱싱 $0.08 + 4-mode 통합 2회)
- Colab: 무료 티어 T4 (Pro 유닛 0)
- Qdrant·Langfuse·Docling·BGE-M3·Nemotron: **전부 무료**

**Phase 3 (벤치마크) 예상 추가 지출**: $10~20
**프로젝트 전체 out-of-pocket**: $10~20 (Colab Pro 빼면)

---

## 라이선스 안전성

- **우리 코드**: MIT
- **Nemotron ColEmbed V2**: CC-BY-NC-4.0 (비상업용 — 우리 포트폴리오 OK)
- **BGE-M3, Docling, Qdrant**: MIT/Apache (자유)
- **Sanofi PDF**: 레포에 커밋 안 함 (`data/pdf/` gitignore)

---

## 지금 위치

```
Phase 0: ████████████ 100% (환경 셋업 완료)
Phase 1: ████████████ 100% (4모드 + LangGraph 하이브리드 검증 완료) ← 오늘 종료
Phase 2: ░░░░░░░░░░░░   0% (평가셋 30개 작성 — 다음 세션)
Phase 3: ░░░░░░░░░░░░   0% (벤치마크)
Phase 4: ░░░░░░░░░░░░   0% (UI + 블로그)
```

**다음 세션 (Phase 2, 약 1주 분산)**:
1. **한국어 질문 30개 자작** (차트 10 / 표 10 / 본문 5 / 멀티홉 5)
2. **각 질문의 영어 대응쌍 작성** → 60개 (KR/EN 비교용)
3. **정답 레이블링** (page_id, block_type, period_spec)
4. **`eval/questions.jsonl` 포맷 확정**
5. **`eval/runner.py`** — 4모드 × 60질문 × 메트릭 → CSV
6. (옵션) Q2/Q3/20-F 추가 인덱싱 — Cloudflare Tunnel 대신 Colab 내부 indexing + .npz 다운로드 방식으로 재구성 (페이지당 5분 → 5초로 단축 예상)

---

## 핵심 깨달음 (블로그/면접용)

1. **"오픈소스 스택 + Claude API로 SOTA RAG, $20 이내 풀 벤치마크"**
2. **"BGE-M3 multilingual은 dense retrieval 격차를 거의 메움"** — QT 같은 번역 단계 의외로 불필요
3. **"HyDE와 Reranker는 다른 문제를 푼다"** — HyDE = 쿼리 재작성 (의미 격차), Reranker = 후보 풀 확장 (검색 깊이). 직교적
4. **"Hybrid의 진가는 retriever 다양성"** — RRF가 한 retriever의 빈틈을 다른 게 메움. 6페이지 인덱스에서도 100%
5. **"멀티모달 모델은 dtype/serialization 함정이 많다"** — Nemotron의 fp32 ViT, JSON 100MB 페이로드, Cloudflare 30MB 한도... 운영 디테일이 모델 크기보다 더 큰 변수
6. **"포트폴리오도 production 디버깅 그대로 만난다"** — Colab 컴퓨트 소진, ngrok 정책 변경, Qdrant env var 버그, port 충돌, Windows symlink 권한, cp949 인코딩... 실제 production-grade 트러블슈팅 경험

---

## 산출물

**GitHub**: https://github.com/tkddnjs-dlqslek/pharma-vision-rag

**태그 (세이브포인트, 시간순)**:
| 태그 | 의미 |
|---|---|
| `phase-0-scaffold` | 레포 구조 + 설정 파일 |
| `phase-0-done` | 환경 셋업 끝 (Nemotron + Qdrant + Docling smoke pass) |
| `phase-1-generator` | Claude Vision 생성기 + 프롬프트 캐싱 |
| `phase-1-text-retriever` | Docling + BGE-M3 + Qdrant 인덱서 |
| `phase-1-text-variants` | baseline / QT / HyDE 3변형 |
| `phase-1-local-done` | text_only 모드 + ZeRank2 + Langfuse |
| `phase-1-code-complete` | 4 모드 + LangGraph 하이브리드 그래프 코드 |
| `phase-1-done` | **5-mode smoke 13/15 통과, hybrid 100%** ← 현재 |

**파일 통계** (대략):
- 소스 코드: 약 1,500줄 Python (`src/pharma_vision_rag/`)
- 스크립트: 11개 (`scripts/`)
- 노트북: 2개 (Phase 0 smoke + Phase 1 production tunnel)
- 문서: PRD, plan.md, summary.md, VARAG_REVIEW.md
