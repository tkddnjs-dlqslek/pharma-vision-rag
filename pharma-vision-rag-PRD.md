# pharma-vision-rag — 프로젝트 기획안

**작성일**: 2026-04-21
**작성자**: 김상원
**버전**: v1.0
**대상**: LLM/AI Engineer 포트폴리오 용도

---

## 0. Executive Summary

파마 도메인 PDF 리포트(차트/표 밀집)에 최적화된 멀티모달 하이브리드 RAG 시스템을 구축하고, 2026년 최신 SOTA 모델들 간 벤치마크를 수행하는 프로젝트. VARAG 프레임워크를 scaffold로 사용하되 내부 리트리버를 최신 모델(Nemotron ColEmbed V2)로 교체. ViDoRe V3 벤치마크 메소돌로지 차용하여 파마 도메인 전용 평가셋을 구축함. 최종 산출물은 오픈소스 레포지토리, 기술 블로그 포스트, 그리고 정량 벤치마크 결과.

---

## 1. 문제 정의

### 1.1 왜 일반 RAG로 안 되는가

파마 도메인 PDF의 특성:
- IQVIA DDD 월간 리포트: 표 90%+, 본문 최소
- Channel Dynamics 자료: 차트 다수, 축·범례에 핵심 숫자 삽입
- Webinar 슬라이드: 그림·인포그래픽 밀집
- 논문·임상 리포트: 테이블·플롯·텍스트 혼재

전통 RAG 파이프라인(OCR → 텍스트 청킹 → 임베딩)은 이 유형에서 실패한다. 차트/표의 정보가 OCR 단계에서 구조적으로 깨지고, "3월 Dupixent 처방률 12% 상승" 같은 숫자가 그래프에만 있을 때 검색 자체가 불가능하다.

### 1.2 기존 해결책들의 한계

**VLM-Caption 방식** (각 페이지를 VLM으로 서술 후 텍스트 RAG):
- 페이지당 API 호출 비용 누적 (10,000페이지 → $100~$300)
- 서술 과정에서 구체 수치 손실

**ColPali 단독 Vision-only RAG**:
- 텍스트 위주 페이지에서 BGE-M3 같은 텍스트 임베딩보다 성능 낮음
- 저장 용량이 single-vector의 1000배

**필요한 것**: 페이지 유형에 따라 최적 리트리버를 동적으로 선택하는 하이브리드 구조.

---

## 2. 프로젝트 목표

### 2.1 기술적 목표

1. 파마 PDF에 대한 **4가지 RAG 모드** 구현 및 비교:
   - Text-only RAG (Docling + BGE-M3, baseline)
   - Vision-only RAG (Nemotron ColEmbed V2 + Claude Vision)
   - Caption RAG (페이지별 VLM 서술 후 텍스트 RAG)
   - **Hybrid RAG** (Text retrieval + Vision reranking, 본 프로젝트의 main contribution)

2. 파마 도메인 평가셋 구축:
   - 파마 PDF 50~100페이지 × 질문 30개
   - 질문 유형: 차트 기반 / 표 기반 / 본문 기반 / 멀티홉
   - ViDoRe V3 메소돌로지 참조하여 human-verified 정답 작성

3. 정량 벤치마크:
   - 정확도 지표: NDCG@5, Recall@10, Answer F1
   - 비용 지표: 인덱싱 비용 (페이지당), 쿼리 비용 (질문당), 저장 용량
   - 지연 지표: 인덱싱 throughput, 쿼리 레이턴시

### 2.2 커리어 목표

LLM/AI Engineer 채용 우대사항 중 다음을 정면 돌파:
- RAG 시스템 + 벡터 DB 운영 경험
- Multimodal / Vision Language Model 활용
- 평가 프레임워크 설계 (offline golden set, LLM-as-judge)
- 비용·레이턴시 최적화
- Agentic workflow (Hybrid 모드에 쓰임)
- 도메인 특화 적응

### 2.3 산출물

- GitHub 레포지토리 (MIT 라이선스, Docker 컴포즈 원클릭 실행)
- 기술 블로그 포스트 1~2개 (한국어 링크드인 + 영어 Medium)
- 벤치마크 결과 리포트 (표·그래프 포함)
- 재현 가능한 평가 스크립트 + 평가 데이터셋
- 데모 Gradio 앱

---

## 3. 기술 스택

### 3.1 2026년 SOTA 기준 선정

| 역할 | 선택 | 선정 이유 |
|---|---|---|
| **Visual retriever** | Nemotron ColEmbed V2 (3B) | 2026년 2월 기준 ViDoRe V3 1위. 3B 버전이 소비자 GPU에서 동작 |
| **Text retriever** | BGE-M3 (한국어 지원) | 다국어 + 긴 문맥. 국내 기업 JD 연계 |
| **Reranker** | ZeRank2 또는 Jina Reranker v2 | ViDoRe V3에서 상위권 |
| **Layout + OCR** | IBM Docling | 표·수식·레이아웃 통합 추출. 2024년 말 공개, 2025~2026 확산 |
| **PDF → 이미지** | pdf2image (pypdfium2 backend) | VARAG 기본값과 일치 |
| **생성 VLM** | Claude Sonnet 4.7 Vision (메인) + Qwen3-VL-8B (오픈 비교군) | 현재 Claude가 차트 해석 강점, Qwen3-VL로 비용 비교 |
| **벡터 DB** | Qdrant (멀티벡터 네이티브 지원) | ColBERT-스타일 late interaction 지원 |
| **오케스트레이션** | LangGraph | Hybrid 모드의 다단계 라우팅 표현 |
| **관찰성** | Langfuse | 기존 프로젝트(dupixent-classifier)와 스택 일관성 |
| **프레임워크 scaffold** | VARAG (fork) | 4가지 모드 비교 구조 이미 구현됨. 내부 모델만 교체 |
| **데모 UI** | Gradio | VARAG 기본 제공, 데모 제출용 |
| **API 서버** | FastAPI | 기존 스택 |
| **평가** | ViDoRe V3 evaluation code (차용) | 표준 메트릭 |

### 3.2 왜 VARAG를 scaffold로 쓰는가

VARAG는 2024년 ColPali 시대 프레임워크지만 구조적으로 4가지 RAG 모드를 나란히 비교할 수 있도록 설계되어 있음. 이 구조는 2026년에도 유효함. 단:

- 리트리버: ColPali → Nemotron ColEmbed V2로 교체
- 텍스트 RAG 부분: 단순 OCR → Docling으로 교체
- 리랭커: 추가 (원본 VARAG에 없음)
- 벤치마크 평가 루프: ViDoRe V3 스타일로 재작성

이 교체·추가 작업 자체가 포트폴리오 스토리가 됨.

### 3.3 기존 자산 재활용

- `pharma-validator` 코드: PDF 전처리 로직
- `dupixent-classifier` eval harness 패턴: 평가 프레임워크 골격
- LangGraph 교재: Hybrid 모드의 라우팅 그래프 설계
- Langfuse 셋업: 그대로 사용
- 파마 도메인 지식: 평가셋 질문 작성

---

## 4. 시스템 아키텍처

### 4.1 Hybrid RAG 흐름 (본 프로젝트의 main contribution)

```
[인덱싱 단계, 한 번]
PDF 입력
  │
  ├─► Docling ──► Layout 분석 ──► 블록별 추출 (텍스트/표/그림 블록)
  │                                   │
  │                                   ├─► 텍스트 블록 ──► BGE-M3 임베딩 ──► Qdrant (text collection)
  │                                   ├─► 표 블록 ──► 마크다운 변환 ──► BGE-M3 임베딩 ──► Qdrant (text collection)
  │                                   └─► 그림/차트 블록 ──► (스킵, 아래에서 처리)
  │
  └─► pdf2image ──► 페이지 이미지 ──► Nemotron ColEmbed V2 (multi-vector) ──► Qdrant (vision collection)

[검색 단계, 질문마다]
질문
  │
  ├─► BGE-M3 임베딩 ──► Qdrant text collection에서 Top-20 검색
  │
  └─► Nemotron ColEmbed V2 임베딩 ──► Qdrant vision collection에서 Top-20 검색
        │
        ▼
  [LangGraph Router 노드]
  ├─► 질문에 "그래프/차트/표/수치" 키워드 → vision Top-20 위주로 가중치
  └─► 질문이 개념/정의/설명 → text Top-20 위주로 가중치
        │
        ▼
  [Reranker: ZeRank2]
  └─► 합친 후보군을 재점수 → Top-3 선정

[답변 단계]
Top-3 (이미지 페이지 + 텍스트 블록 혼합) + 질문
  │
  └─► Claude Sonnet 4.7 Vision ──► 최종 답변 생성

[검증 단계]
생성된 답변
  │
  └─► LLM-as-judge (Claude Haiku) ──► 근거 없는 주장 검출
                                 ──► Langfuse 로깅
```

### 4.2 다른 3가지 모드 구현

**Text-only RAG (baseline)**:
- Docling → BGE-M3 → Qdrant → Claude Sonnet (텍스트만)

**Vision-only RAG**:
- pdf2image → Nemotron ColEmbed V2 → Qdrant → Claude Sonnet Vision

**Caption RAG**:
- pdf2image → 각 페이지 Claude Haiku Vision으로 서술 생성 → BGE-M3 임베딩 → Qdrant → Claude Sonnet

네 가지 모두 동일한 Qdrant 인스턴스, 동일한 질문셋, 동일한 평가 루프에서 돌림.

---

## 5. 평가 프레임워크

### 5.1 평가셋 구축

**문서 코퍼스** (50~100페이지):
- 공개 파마 연간 보고서 2~3개 (Sanofi, Pfizer, Merck 10-K)
- 오픈 액세스 임상 논문 10개 (PubMed Central)
- 파마 마켓 리서치 리포트 공개본 (가능한 경우)

이들은 저작권 리스크 없이 공개 깃헙 제출 가능.

**질문 세트 30개**:
- 유형 A (10개): 차트 독해 필수 — "2024년 Q3 대비 Q4 매출 변화율"
- 유형 B (10개): 표 독해 필수 — "임상 시험 III상 부작용 발생률"
- 유형 C (5개): 본문 이해 — "약물 작용 메커니즘"
- 유형 D (5개): 멀티홉 — "A약의 Q4 매출과 B약의 Q4 매출을 비교"

각 질문에 대해 정답 + 정답이 위치한 페이지 번호 + 필요 블록 타입(chart/table/text) 레이블.

### 5.2 지표

**검색 정확도**:
- Recall@5, Recall@10: 정답 페이지가 Top-k에 포함된 비율
- NDCG@5: 순위까지 고려한 점수

**답변 정확도**:
- Exact Match: 정답이 숫자일 때
- Claude-as-judge Score: 주관식 답변의 의미적 정확도 (1~5점)
- Hallucination Rate: 근거 없는 주장 비율

**비용·성능**:
- 인덱싱 비용: 100페이지 코퍼스 기준 총 비용 (GPU + API)
- 쿼리 비용: 질문 30개 처리 총 비용
- 저장 용량: 각 모드의 벡터 DB 크기
- 인덱싱 시간: 페이지당 초
- 쿼리 지연: Top-3 반환까지 초

### 5.3 예상 결과 (가설)

| 모드 | Recall@5 예상 | 비용 (100pg) | 적합 영역 |
|---|---|---|---|
| Text-only RAG | ~0.55 | $0.1 | 본문 기반 Q |
| Vision-only RAG | ~0.72 | $0.5 | 차트·표 기반 Q |
| Caption RAG | ~0.65 | $5.0 | 모든 Q, 단 비쌈 |
| **Hybrid RAG** | **~0.82** | **$0.6** | **모든 Q, 비용 효율** |

*숫자는 가설임. 실제 벤치마크로 검증.*

가설이 맞으면 "하이브리드가 Caption RAG 대비 8배 싸면서 17pt 정확도 상승" 같은 강한 스토리 나옴.

---

## 6. 개발 로드맵

### Phase 0: 환경 셋업 (Day 1)

- [ ] VARAG 레포 fork 및 로컬 실행 확인
- [ ] Gradio 데모 샘플 PDF로 동작 검증
- [ ] Nemotron ColEmbed V2 3B 모델 HuggingFace에서 다운로드 및 추론 테스트
- [ ] Docling 설치 및 샘플 파마 PDF 처리 테스트
- [ ] Qdrant Docker 컨테이너 기동

**승인 기준**: 샘플 PDF 하나에서 VARAG 기본 4개 모드가 모두 돌아가고, 내가 다운로드한 Nemotron 모델로도 임베딩이 추출됨.

### Phase 1: 최신 모델로 교체 (Day 2~4)

- [ ] VARAG의 ColPali 부분을 Nemotron ColEmbed V2로 교체
- [ ] 텍스트 RAG 부분 OCR을 Docling으로 교체
- [ ] ZeRank2 리랭커 통합
- [ ] Claude Sonnet Vision 연동 (VARAG 기본은 OpenAI)
- [ ] 네 가지 모드 모두 새 스택에서 동작 확인

**승인 기준**: 샘플 질문 3개에 네 가지 모드가 모두 답변을 생성하고 Langfuse에 로깅됨.

### Phase 2: 평가셋 구축 (Day 5~7)

- [ ] 파마 PDF 코퍼스 수집 (50~100페이지)
- [ ] 질문 30개 작성 + 정답 레이블링
- [ ] 질문 유형별 분포 확인
- [ ] 평가 루프 스크립트 작성 (ViDoRe V3 코드 차용)

**승인 기준**: 평가 스크립트가 네 가지 모드를 차례로 돌려서 모든 지표를 CSV로 출력함.

### Phase 3: 벤치마크 실행 및 분석 (Day 8~10)

- [ ] 전체 평가셋에서 네 가지 모드 풀 실행
- [ ] 결과 표/그래프 생성
- [ ] 질문 유형별 breakdown 분석
- [ ] 실패 케이스 정성 분석 (왜 틀렸는지 10개 케이스)
- [ ] Hybrid 모드의 라우팅 결정 분석 (어떤 질문에 vision, 어떤 질문에 text로 갔는지)

### Phase 4: 문서화 및 공개 (Day 11~14)

- [ ] README 작성 (한·영 병기)
- [ ] 블로그 포스트 한국어판 작성 (링크드인용)
- [ ] 블로그 포스트 영어판 작성 (Medium/Dev.to)
- [ ] 데모 GIF 촬영
- [ ] Gradio 데모 HuggingFace Spaces 배포
- [ ] 링크드인 포스트 게재

### Stretch Goals (이후)

- Nemotron V2 대신 **한국어 파마 PDF 전용 ColBERT-스타일 모델** 파인튜닝 시도
- 국내 제약사 공개 IR 자료로 확장
- 실제 LUCI webinar 자료에 적용 (Sanofi 내부 허가 하에)
- ViDoRe V3 리더보드에 새 파이프라인 제출

---

## 7. 리스크 & 대응

| 리스크 | 영향도 | 대응 |
|---|---|---|
| Nemotron ColEmbed V2 모델 크기(3B) GPU 메모리 부족 | 중 | 4bit 양자화 사용, 또는 Colab 프로 사용, 또는 API 기반 모델로 대체 |
| 파마 PDF 저작권 이슈 | 중 | 공개 데이터(10-K, PMC 논문)만 사용. 내부 자료는 로컬 실험만 하고 공개 레포에 제외 |
| Hybrid 모드가 vision-only보다 나쁜 결과 | 중 | 그래도 그 결과 자체가 블로그 소재. "왜 hybrid가 실패하는가" 회고 글. 부정적 결과도 가치 있음 |
| 평가셋 질문 품질이 낮아 결과 신뢰도 부족 | 중 | 외부 리뷰어 1명 이상에게 질문 검증 요청 (지인 가능) |
| VARAG 레포 의존성 이슈 | 낮음 | fork한 시점 고정. 필요 시 핵심 모듈만 재작성 |
| 시간 초과 (2주 넘어감) | 중 | Phase 4 건너뛰고 Phase 3까지만 해도 포트폴리오로 충분. 블로그는 나중에 |
| Sanofi 인턴 업무와 충돌 | 높음 | 평일 저녁 1~2시간, 주말 집중. 2주 대신 4주 계획으로 버퍼링 |

---

## 8. 이 프로젝트가 커버하는 JD 우대사항 (점검표)

- [x] RAG 시스템 구축 경험 (4가지 모드)
- [x] 벡터 DB 운영 (Qdrant, multi-vector)
- [x] Multimodal / VLM (Nemotron, Claude Vision)
- [x] Late interaction / ColBERT 계열 이해
- [x] Evaluation framework 설계 (offline golden set, LLM-as-judge)
- [x] 통계 리터러시 (NDCG, recall, 신뢰구간)
- [x] Prompt engineering (답변 생성 프롬프트 + judge 프롬프트)
- [x] Guardrails (hallucination 검출)
- [x] 비용·레이턴시 최적화 (4 모드 비용 비교)
- [x] Agentic workflow (LangGraph 라우팅)
- [x] Observability (Langfuse)
- [x] 도메인 특화 (파마, 네 차별화)
- [x] 오픈소스 기여 (VARAG fork, ViDoRe 리더보드 제출 가능성)
- [x] 한국어 지원 (BGE-M3, 국내 기업 어필)

---

## 9. 면접 토킹 포인트

이 프로젝트 하나로 다음 질문들에 구체적으로 답변 가능해짐:

1. "RAG 시스템 구축 경험 있나요?" → 4가지 모드 구현, 각각의 trade-off 정량 설명
2. "Multimodal에 대한 이해도?" → ColBERT late interaction → ColPali → Nemotron ColEmbed V2 진화 설명
3. "Evaluation 어떻게 하셨어요?" → ViDoRe V3 메소돌로지 차용, 도메인 평가셋 구축 과정
4. "왜 Hybrid가 나은가요?" → 비용 $X, 정확도 Y% 데이터로 설명
5. "비용 최적화 경험?" → Caption RAG 대비 8배 절감 사례
6. "Agentic하게 만들어보신 적?" → LangGraph로 질문 유형별 라우팅 구현
7. "한국어 처리 어떻게?" → BGE-M3 선택 이유, 한국 파마 문서 특성 설명
8. "자기 프로젝트 홍보해주세요" → 링크드인 포스트 Y개, 블로그 조회수 Z, 깃헙 스타 N개

---

## 10. 다음 단계 (오늘 체크리스트)

즉시 실행:

1. [ ] 이 기획안 리뷰 및 스코프 확정 (2주 vs 4주, 어디까지)
2. [ ] GitHub 레포 `pharma-vision-rag` 생성
3. [ ] VARAG fork 완료
4. [ ] Nemotron ColEmbed V2 3B 모델 로컬에 다운로드 시작 (용량 큼)
5. [ ] Docling `pip install docling` 설치
6. [ ] 평가셋용 공개 파마 PDF 5개 선정 및 다운로드

작성자 결정 필요:

- [ ] 타임라인 (2주 집중 vs 4주 여유)
- [ ] 공개 범위 (완전 공개 오픈소스 vs 부분 공개)
- [ ] 블로그 플랫폼 선택 (링크드인 단독 vs Medium 병행)
- [ ] Stretch goal 중 어디까지 할지 (ViDoRe 리더보드 제출 시도 여부)
