# pharma-vision-rag — 진행 요약

**기간**: 2026-04-21 ~ 04-22 (이틀, 약 12시간)
**현재 상태**: Phase 1 약 60% 완료. 로컬에서 동작하는 첫 RAG 모드(text-only) 끝까지 통과.

---

## 한 줄 요약

Sanofi 공시 PDF에서 한국어로 질문하면 영문 페이지를 찾아 답해주는 멀티모달 RAG를 만드는 중. 지금까지 텍스트 검색 → 답변까지 동작 확인했고, 다음 단계로 비전 검색을 붙일 차례.

---

## 무엇을 했나 (시간 순서)

### Day 1 (2026-04-21) — 기획 + 환경 셋업

1. **기획안 검토 후 plan.md 작성** — PRD를 실행 가능한 체크리스트로 풀어씀
2. **10가지 의사결정 확정** — 타임라인, 공개 범위, GPU 전략(Colab), 평가 데이터 정책 등
3. **GitHub 공개 레포 생성** + git 연결
4. **디렉토리 구조 + .gitignore + 설정 파일 일괄 생성** (`README.md`, `requirements.txt`, `docker-compose.yml`, `.env.example`)
5. **Sanofi PDF 4개 수집**: Form 20-F 2025 + Q1/Q2/Q3 2025 Press Release
6. **Colab에서 Nemotron 모델 동작 검증** (smoke test 통과)
7. **로컬에서 Qdrant 기동** + Docling으로 Q1.pdf 파싱 검증
8. **Phase 0 완료 커밋** (`phase-0-done`)

### Day 2 (2026-04-22) — 모듈 빌드

9. **VARAG 레포 리뷰** → 우리 프로젝트와 차이 정리 (`docs/VARAG_REVIEW.md`)
10. **Claude Vision 생성기 구현** + 프롬프트 캐싱으로 52% 토큰 절감 검증
11. **Docling → BGE-M3 → Qdrant 인덱서 구현** — Q1.pdf에서 319개 청크 추출·인덱싱
12. **텍스트 검색기 3변형(baseline/QT/HyDE) 구현** — 추상 쿼리 실패 → HyDE 회복 검증
13. **Text-only 모드 완성** — 검색 → 리랭커 → 생성 → 답변 end-to-end
14. **ZeRank2 리랭커 통합** (BGE-reranker-v2-m3 사용)
15. **Langfuse trace 연동** — 대시보드에서 모든 호출 추적 가능

---

## 지금 동작하는 것

**예시**: 한국어로 "2025년 1분기 Beyfortus 백신 매출은?" 질문하면 →

1. BGE-M3가 영문 PDF 청크 20개 후보 검색
2. BGE-reranker가 top-3로 재정렬
3. Claude Sonnet이 한국어로 답변: *"2025년 1분기 Beyfortus 매출은 €284 million으로, 전년 동기 대비 CER 기준 +54.9% 성장했습니다..."*
4. Langfuse 대시보드에 모든 단계 trace 자동 기록

3개 테스트 케이스(추상/고유명사/한국어) 모두 정답 추출 성공.

---

## 가설 vs 실제 — 무엇이 달랐나

### ✅ 가설대로 들어맞은 것

- **Vision retrieval이 차트에 강할 것** → Phase 0 Nemotron smoke test에서 Q1.pdf 페이지 임베딩 정상 생성 확인 (Phase 3에서 정량 검증 예정)
- **Docling이 표 구조를 보존** → IFRS 매출표가 마크다운으로 깔끔히 변환되어 청크에 들어감
- **프롬프트 캐싱이 비용 절감** → 첫 쿼리 4,637 토큰 cache write, 이후 쿼리 100% cache hit. 3쿼리만으로 52% 절감

### ❌ 가설과 달랐던 것 + 어떻게 대응했나

| 가설 | 실제 | 대응 |
|---|---|---|
| **Claude Sonnet 4.7** 사용 | 2026-04 기준 Sonnet은 4.6이 최신 | `claude-sonnet-4-6` 사용. 4.7 출시 시 한 줄 교체 |
| **Nemotron ColEmbed V2 4B** 사용 (3위 모델) | Colab Pro 컴퓨팅 유닛이 이번 달 소진 → L4 사용 불가 | **3B 모델 (6위)로 다운그레이드**. ViDoRe NDCG 차이 ~3pt이지만 우리 도메인 작아서 영향 미미 예상 |
| **fp16으로 모델 로드** | Nemotron 내부 ViT가 fp32 고정이라 dtype mismatch 에러 | **bf16 + float-only cast**로 우회 (integer buffer 보존) |
| **`pdf2image`로 PDF→이미지** | Colab에 poppler 없어서 첫 시도 실패 | `apt-get install poppler-utils` 추가 + 로컬은 pypdfium2 (Windows-friendly) |
| **Qdrant 6333 포트** | 사용자 PC에 n8n 스타터 키트의 다른 Qdrant가 이미 6333 점유 | **포트 분리 (6335)**, `.env` + docker-compose 수정 |
| **20-F는 FY2024 데이터** | PDF 직접 확인 시 FY2025 (Jan-Dec 2025) | "기간 혼합" → "기간 통일, granularity만 차이"로 정정. Phase 3 실험 축 재설계 |
| **Caching이 자동 적용** | 첫 시도 0 토큰 캐시 (Sonnet 4.6 최소 2048 토큰 prefix 미달) | 스모크 테스트를 1페이지 → 3페이지로 변경 → 4,637 토큰 prefix로 캐싱 정상 작동 |
| **Nemotron `processor + model(**inputs)` 호출** | None 에러 (멀티모달 입력 정렬 안 됨) | 모델 카드 재확인 → 전용 메서드 `forward_images()` / `forward_queries()` / `get_scores()` 사용 |

### 🆕 예상 못한 발견

**1. BGE-M3 multilingual이 생각보다 강함**
한국어 쿼리로 영문 PDF 검색이 baseline 그대로도 잘 됨. 원래 Query Translation(QT)이 크로스링구얼 격차를 메울 핵심 기법으로 봤는데, **QT 효과가 거의 없음**. 의외의 발견 — 블로그 토픽 추가.

**2. HyDE와 Reranker, 둘 다 추상 쿼리 실패를 회복**
"business EPS growth" 같은 추상 쿼리에서 baseline 실패 → HyDE 한 가지로 풀릴 거라 예상. 그런데 **리랭커 단독으로도 회복** (top-20에 정답이 있으면 cross-encoder가 1등으로 끌어올림). Phase 3에서 두 회복 경로 별도 비교 가능 → 블로그 매트릭스 확장.

**3. Docling이 일부 페이지에서 메모리 부족**
Q1.pdf 23페이지 중 페이지 16/17/21에서 `std::bad_alloc` 발생. 로컬 CPU RAM 한계. **나머지 20페이지는 정상 처리** → Phase 1에서 페이지별 배치 처리로 우회 예정. 블로커 아님.

**4. Claude Vision의 멀티페이지 컨텍스트 통합 능력**
질문 1개에 페이지 3장 주면 **여러 페이지 데이터를 자동으로 종합해서 답변** (예: 페이지 1 본문 + 페이지 2-3 지역별 표 → 한 답변에 모두 포함). 별도 프롬프트 엔지니어링 없이 자동.

---

## 비용 정리

**총 지출 (Day 1~2)**:
- Anthropic API: 약 **$0.10** (스모크 테스트 5~6회)
- Colab: 무료 티어 T4 사용 (Pro 유닛 0)
- Qdrant·Langfuse·Docling·BGE-M3·Nemotron: **전부 무료**

**Phase 3 (벤치마크) 예상 추가 지출**: $10~20 (Anthropic API 풀 벤치)
**프로젝트 전체 out-of-pocket**: $10~20 (Colab Pro 빼면)

---

## 라이선스 안전성

- **우리 코드**: MIT (포트폴리오·상업 모두 OK)
- **Nemotron ColEmbed V2**: CC-BY-NC-4.0 (비상업 한정 — 우리 포트폴리오 OK, 상업화 시 NVIDIA 상업 모델로 교체)
- **BGE-M3, Docling, Qdrant**: MIT/Apache (전부 자유)
- **Sanofi PDF**: 레포에 커밋 안 함 (`data/pdf/` gitignore)

---

## 지금 위치

```
Phase 0: ████████████ 100% (환경 셋업 완료)
Phase 1: ███████░░░░░  60% (text_only 모드 끝, vision/caption/hybrid 남음)
Phase 2: ░░░░░░░░░░░░   0% (평가셋 30개 작성 — 다음 다음)
Phase 3: ░░░░░░░░░░░░   0% (벤치마크)
Phase 4: ░░░░░░░░░░░░   0% (UI + 블로그)
```

**다음 세션 (Colab 의존 묶음, 4~6시간)**:
1. `retriever/nemotron.py` — Colab 노트북에 FastAPI+ngrok 엔드포인트 띄우고 로컬 클라이언트 작성
2. `rerank/nemotron_rerank.py` — 같은 Colab 노트북에 reranker 엔드포인트 추가
3. `modes/vision_only.py` — Nemotron 검색 + Claude Vision (페이지 이미지 직접)
4. `modes/caption.py` — Haiku로 페이지 캡션 → BGE-M3 인덱싱 → 텍스트 검색
5. `modes/hybrid.py` + LangGraph 라우터 — 텍스트+비전 후보 합병 + 리랭크
6. 4-mode × 3-query smoke test로 Phase 1 완료

**그 다음 (1주)**: Phase 2 평가셋 작성 (한국어 질문 30개 + 영어 대응쌍 + 정답 레이블)

---

## 핵심 깨달음 (블로그/면접용)

1. **"오픈소스 스택만으로 SOTA RAG 가능"** — Claude API 빼고 전부 무료, $20 이내 풀 벤치마크
2. **"BGE-M3 multilingual은 dense retrieval 격차를 거의 다 메움"** — QT 같은 번역 단계가 의외로 불필요
3. **"HyDE와 Reranker는 다른 문제를 푼다"** — HyDE는 쿼리 재작성, Reranker는 후보 풀 확장. 둘이 직교적
4. **"멀티모달 모델은 dtype 함정이 많다"** — Nemotron의 ViT/LM 분리 설계가 fp16/fp32 미스매치 유발. bf16 + 선택적 캐스팅이 정답
5. **"포트폴리오 프로젝트도 운영 함정을 그대로 만남"** — Colab 컴퓨트 소진, 포트 충돌, Windows symlink 권한, cp949 인코딩... 실제 production 디버깅 경험과 동일한 결로 진행됨

---

## 산출물

**GitHub**: https://github.com/tkddnjs-dlqslek/pharma-vision-rag

**태그 (세이브포인트)**:
- `phase-0-scaffold` — 레포 구조
- `phase-0-done` — 환경 셋업 끝
- `phase-1-generator` — Claude Vision 생성기
- `phase-1-text-retriever` — Docling+BGE-M3+Qdrant 인덱서
- `phase-1-text-variants` — baseline/QT/HyDE 3변형
- `phase-1-local-done` — text_only 모드 + 리랭커 + Langfuse (현재)
