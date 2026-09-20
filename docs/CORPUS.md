# 코퍼스 구성 (Phase 2, 2026-09-16 확정)

## 0. 코퍼스 v2 (2026-09-20 확정): 27개 문서, 1,709페이지

기준 파일은 `eval/corpus.json`(문서 id, 회사, 기간, 종류, 라벨, 페이지 수)이고 `scripts/11_build_corpus.py`가
`data/pdf/raw/`와 `data/pdf/`의 원본을 `data/pdf/corpus/`에 통일된 이름으로 복사해 만든다.

| 묶음 | 문서 | 페이지 |
|---|---|---|
| Sanofi 2025: 보도자료 Q1~Q4, 슬라이드 Q1~Q4, 20-F FY2025 전체 | 9 | 579 |
| Sanofi 2024: 보도자료 Q1~Q4, 슬라이드 Q1~Q4, 20-F FY2024 전체 | 9 | 592 |
| Novartis 2025 Q1~Q3 슬라이드 | 3 | 231 |
| Roche 2025 Q1, 상반기, Q3 슬라이드 | 3 | 187 |
| AstraZeneca 2025 Q1~Q3 슬라이드 | 3 | 120 |

**왜 늘렸나**: v1의 172p는 긴 컨텍스트 창에 통째로 들어가 "그냥 다 넣으면 되지 않나"에 답할 수 없었다.
**무엇을 넣었나**: 아무 문서가 아니라 헷갈리는 문서. 같은 회사의 전년 같은 분기, 경쟁사의 같은 분기처럼 구조가
같고 숫자만 다른 페이지가 실제 업무의 검색 난이도를 만든다. 슬라이드 위주라 차트 밀도도 유지된다.
**넣지 않은 것**: 컨퍼런스콜 녹취록(텍스트뿐이라 텍스트 경로에만 유리), ESG 보고서 등 질문을 만들 수 없는 자료.
Eli Lilly 슬라이드는 IR 사이트의 봇 차단(HTTP 403)으로 받지 못했고 목표 분량을 이미 채워 제외했다.
**상한**: 비전 패치 임베딩이 페이지당 수 MB라 이 개발 PC(16GB RAM, 디스크 여유 58GB)에서는 이 규모가 현실적 최대치.

v1 문서 id와 발췌본 페이지 번호는 `scripts/11_build_corpus.py --migrate-v1`이 v2로 변환했다. 아래 2절과 3절의
발췌 매핑 표는 그 변환의 근거로 남긴다.

출처 URL 패턴: Sanofi 슬라이드 `sanofi.com/assets/dotcom/content-app/events/quaterly-results/<연도>/...`,
Novartis `novartis.com/sites/novartis_com/files/q<N>-2025-investor-presentation.pdf`,
AstraZeneca `astrazeneca.com/content/dam/az/PDF/2025/...`, Roche `assets.roche.com/f/176343/x/<해시>/irp<날짜>-a.pdf`
(Roche는 해시가 매번 달라 행사 페이지에서 다시 찾아야 함).

## v1 기록 (2026-09-16, 172페이지). 아래는 이력이며 현재 구성은 0절

인덱싱 대상 7파일, **172페이지**. 빌드: `PYTHONIOENCODING=utf-8 python scripts/11_build_extracts.py`
(페이지 목록은 스크립트 상수가 원본. 이 문서는 근거 기록).

| 인덱스 파일 | 페이지 | 원본 | 역할 |
|---|---|---|---|
| `Q1.pdf` | 23 | Sanofi IR Q1 2025 보도자료 (2025-04-24) | 표 + 서술 |
| `Q2.pdf` | 26 | Q2 2025 보도자료 (2025-07-31) | 표 + 서술 |
| `Q3.pdf` | 28 | Q3 2025 보도자료 (2025-10-24) | 표 + 서술 |
| `20F_extract.pdf` | 50 | `Form 20-F 2025 (Oct 2025).pdf` 300p 중 발췌 | 서술(임상, 시장) + 재무 표 |
| `Q1_deck.pdf` | 15 | `raw/Q1_deck_full.pdf` 41p 중 발췌 | **차트** |
| `Q2_deck.pdf` | 15 | `raw/Q2_deck_full.pdf` 41p 중 발췌 | **차트** |
| `Q3_deck.pdf` | 15 | `raw/Q3_deck_full.pdf` 43p 중 발췌 | **차트** |

## 1. 왜 실적 발표 슬라이드를 추가했나

2026-09-16 전 페이지 렌더 확인 결과 **보도자료 77p와 20-F 300p에 차트가 0개**였음. 텍스트와 표만 있고
그래픽은 ESG 등급 로고(Q2 p8, Q3 p10), 이사회 사진 표(20-F p92)뿐. 계획서의 A유형(차트 독해) 10문항과
"차트·표 밀집 PDF" 전제가 성립하지 않았음.

같은 IR 페이지의 분기 실적 발표 슬라이드(Results Presentation)에 막대, 누적 막대, 워터폴, 도넛, 임상 곡선이
있어 차트 페이지만 발췌해 추가. 원본 계획의 "더 늘리지 않는다"는 통계력 대비 인덱싱 비용 논리였는데,
차트 부재는 벤치마크 타당성 문제라 예외로 둠. 추가 인덱싱 비용: 비전 45p × 약 5초, 캡션 약 $0.13.

**차트 텍스트 추출 특성** (text_only 경로가 구조적으로 못 푸는 이유): 숫자 라벨은 텍스트 객체라 추출되지만
세그먼트·연도와의 매핑이 사라짐. 예: Q2 덱 백신 누적 막대 → `711 693 | 297 307 | 115 141 | Beyfortus Flu
Meningitis PPH`. Blueprint 추세 막대는 축 눈금만 추출되고 막대 값 없음.

## 2. 20-F 발췌 50p 선정 근거

원칙: 서술과 표를 섞어 특정 모드에 유리하지 않게. 20-F는 차트가 없으므로 A유형 출제 대상 아님.
C유형(본문 이해)과 B유형(표) 출처.

| 발췌 p | 원본 p | 유형 | 내용 |
|---|---|---|---|
| 1~4 | 25~28 | text | Item 4 회사 개요, 전략(Play to Win), 효율화 |
| 5~7 | 30~32 | text | Dupixent 적응증별 승인 이력(EoE, PN, CSU, COPD, BP) |
| 8~9 | 33~34 | text | 라이프사이클 관리, 희귀질환(Cerezyme, Cerdelga, Myozyme, Nexviazyme) |
| 10~11 | 39~40 | text | Tzield, 백신 개요 |
| 12~16 | 44~48 | table | 바이오파마 파이프라인 표 (immunology, rare, neuro, onco, vaccines, line extensions) |
| 17~19 | 49, 51~52 | text | 시장, 미국 약가·IRA, 유럽·중국 |
| 20 | 55 | table | 주요 제품 규제 독점권·특허 만료 표 (US/EU/JP) |
| 21~22 | 64~65 | table | R&D Appendix 파이프라인 (Phase 1/2/3, registration) |
| 23~25 | 68, 70~71 | mixed | 2025 재무 결과 개요, business net income 정의, IFRS 조정 표 |
| 26~31 | 77~82 | mixed | 손익 요약, 제품·지역별 순매출 표(원본 p79가 핵심), 제품별 서술, gross profit·R&D·SG&A |
| 32~34 | 85~87 | mixed | 현금흐름 요약, FCF 조정, 순부채 표 |
| 35 | 92 | table | 이사회 구성 표 (사진, 국적 아이콘 포함. 코퍼스 유일 이미지 표) |
| 36~38 | 117, 133, 145 | table | 이사 출석률, 이사 보수, 경영진·직원 보수 비율 |
| 39 | 175 | table | 환위험 파생상품 |
| 40~43 | 196~198, 202 | table | 연결 재무상태표(자산·부채), 손익계산서, 현금흐름표 |
| 44~45 | 231, 235 | table | 유형자산 변동, 무형자산 변동 |
| 46~47 | 254, 256 | table | 유로 채권 발행 내역, 금리별 부채 |
| 48~50 | 288, 290, 292 | table | 경영진 보수, 세그먼트 손익, 지역별 순매출 |

## 3. 덱 발췌 15p × 3 선정 근거

기준: 차트가 주 콘텐츠인 페이지 + 발표 요약 표 3~4장(멀티홉 재료). 표지, 사진, Q&A 안내, 파이프라인
약어표, 임상 목록 텍스트 페이지는 제외.

| 발췌 p | Q1 원본 p | Q2 원본 p | Q3 원본 p | 내용 |
|---|---|---|---|---|
| 1 | 5 | 5 | 5 | 사업부별 매출 막대 (Pharma launches, Vaccines, Dupixent, Other) |
| 2 | 6 | 6 | 6 | 신제품 매출 표 |
| 3 | 7 | 7 | 7 | Dupixent US/Outside US 누적 막대 |
| 4 | 9 | 8 | 8 | 백신 세부(Beyfortus, Flu, Meningitis, PPH) 누적 막대 |
| 5 | 12 | 9 | 9 | Q1: 손익 표 / Q2: Blueprint 추세 막대 / Q3: Flu·COVID 요약 |
| 6 | 14 | 10 | 10 | Q1: 2025 dynamics / Q2: 지속가능성 막대 / Q3: 접근성 막대 |
| 7 | 19 | 12 | 12 | Q1: Qfitlia 임상 막대 / Q2: 손익 표 / Q3: 매출 브릿지(환율) |
| 8 | 20 | 14 | 13 | Q1: 뉴스플로 로드맵 / Q2: Regeneron·Amcura 막대 / Q3: 손익 표 |
| 9 | 23 | 17 | 17 | Q1: 매출 표 / Q2: 파이프라인 하이라이트 / Q3: 피부과 임상 곡선 |
| 10 | 24 | 21 | 18 | Q1: 환율 영향 막대 / Q2: 뉴스플로 / Q3: 호흡기 임상 막대 |
| 11 | 25 | 24 | 24 | Q1: 통화 도넛 / Q2·Q3: 매출 표 |
| 12 | 26 | 25 | 25 | Q1: FCF 워터폴 / Q2·Q3: 환율 영향 막대 |
| 13 | 27 | 26 | 26 | Q1: 순부채 워터폴 / Q2·Q3: 통화 도넛 |
| 14 | 28 | 27 | 27 | Q1: 배당 추세 / Q2·Q3: FCF 워터폴 |
| 15 | 30 | 28 | 28 | Q1: 면역학 로드맵 / Q2·Q3: 순부채 워터폴 |

## 4. 정답 중복 규칙 (질문 작성 시)

덱 차트 수치가 같은 분기 보도자료 표에도 있으면 text_only가 보도자료 페이지로 정답을 맞혀 A유형 검증이
무효가 됨. 따라서:

- A유형은 **보도자료 텍스트에 없는 수치**만 출제 (백신 세부 분기값, FCF 브릿지 항목, 환율 영향 분기별,
  추세 막대 높이, 누적 막대 세그먼트).
- `gold_pages`에는 정답이 있는 페이지를 **모두** 기록. 검증 스크립트가 정답 문자열을 코퍼스 전 페이지
  텍스트에서 검색해 누락된 gold_page를 경고.

## 5. gitignore

`data/pdf/` 전체 gitignore. 원본은 아래에서 재다운로드 가능.

- 보도자료: `sanofi.com/assets/dotcom/pressreleases/2025/2025-{04-24,07-31,10-24}-05-30-00-*-en.pdf`
- 20-F: `sanofi.com/assets/dotcom/content-app/publications/annual-report-on-form-20-f/2025-01-01-form-20-f-2025-en.pdf`
- 덱: `sanofi.com/assets/dotcom/content-app/events/quaterly-results/2025/2025-q{n}-2025-results/2025_{MM_DD}_Sanofi_Q{n}_2025_Results.pdf`
