# 평가셋 외부 검토 (eval/questions.jsonl, 60문항)

검토일: 2026-09-27. 검토 방법: 60문항 전부의 첫 gold 페이지와 멀티홉(D) 각 gold 그룹의 첫 페이지를 렌더해 눈으로 확인(총 62장), 나머지 gold 페이지는 텍스트 레이어로 answer_keys 대조. 이미 알려진 사항(D03 회사명 누락, A01과 A02와 B08 수정 완료, 기간 모호 문항의 관대 채점, B06의 NEW LAUNCHES 블록)은 다시 적지 않음.

## 1. 요약

| 구분 | 문항 수 |
|---|---|
| 전체 | 60 |
| 이상 없음 | 42 |
| 문구 수정 권고 (q_ko 또는 q_en 변경, 질의 재임베딩 필요) | 5 (A07, B18 필수, A01, A16, C02 선택) |
| 기준 정답 또는 answer_keys 수정 권고 | 12 (A04, A05, A09, A12, A19, A20, B17, C06, C10, D07, D08, B18) |
| gold 또는 설계 수정 권고 | 1 (D02) |
| 참고 메모 (수정 불필요) | 1 (B05) |

- 심각도 high 1건: D02. 4홉 계산 문항인데 정답(4분기 Dupixent 4,246)이 gold 안의 한 페이지에 그대로 인쇄돼 있음
- 심각도 medium 7건: A07, B18, A19, A20, B17, D07, D08
- gold 페이지에서 기준 정답 수치를 찾지 못한 문항 없음. 확인한 값은 3절에 적음

## 2. 지적 문항 (18건)

각 줄: 유형 / 문제 / 근거(페이지와 실제로 본 값) / 권고

### 가. high

- **D02** (D, 표) / gold 문제와 트리비얼: notes에 "Q4는 코퍼스 어디에도 없음"이라 적혀 있으나 v2 코퍼스의 sanofi_2025Q4_pr.pdf p3(Dupixent 4,246, +32.2%)과 sanofi_2025Q4_deck.pdf p30에 4분기 값이 직접 인쇄됨. 게다가 Q4_pr p3은 D02의 gold_groups 0번(FY 홉)에 들어 있어서 그 한 장만 찾으면 계산 없이 답이 나옴. B13과 정답이 같음 / 근거: Q4_pr p3 "Dupixent sales were €4,246 million and increased by 32.2%", 같은 문단에 FY 15,714. 검산 3,480+3,832+4,156=11,468, 15,714-11,468=4,246 일치 / 권고 두 가지 중 택일. (1) 문구 유지, notes 수정, 채점 기준에 "Q4 직접 인용도 정답"을 명시하고 검색 지표는 이 문항을 멀티홉 집계에서 제외. (2) 문구 변경(재임베딩 필요): q_ko "2025년 3분기와 4분기 Dupixent 매출을 더한 하반기 매출은 얼마인가?", q_en "What were Dupixent sales for H2 2025, i.e. Q3 plus Q4 2025 combined?", 정답 4,156+4,246=€8,402m (하반기 합계는 코퍼스에 인쇄되지 않음), gold_groups는 [Q3 페이지들], [Q4_pr p3, Q4_deck p30]

### 나. medium

- **A07** (A, 차트) / 모호: 회사명이 q_ko와 q_en 모두에 없음. D03과 같은 결함. Pharma와 Vaccines 조합으로 Sanofi가 유추되지만 Roche도 Pharma 브릿지를 냄 / 근거: sanofi_2025Q3_deck p12, Pharma 1,150, Vaccines -295, 13,012 at CER, Forex -578, 12,434 확인 / 권고(재임베딩 필요): q_ko "Sanofi의 2025년 3분기 매출 브릿지에서 Pharma와 Vaccines가 각각 기여한 금액은 얼마였나?", q_en "In Sanofi's Q3 2025 sales bridge, how much did Pharma and Vaccines each contribute?"
- **B18** (B, 표) / 기준 모호: "전년 대비 변화율"이 reported인지 CER인지 없음. 세트의 다른 문항(B12 등)은 CER을 명시하는데 이 문항의 answer_keys는 reported 17.6%를 씀 / 근거: sanofi_2024Q1_pr p1, business EPS €1.78, -17.6% reported, -7.4% at CER / 권고: 문구 유지 시 answer_keys를 ["1.78", "17.6", "7.4"]로 바꾸고 둘 중 하나만 답해도 정답 처리. 문구 변경 시(재임베딩 필요) q_ko "Sanofi의 2024년 1분기 business EPS와 전년 대비 변화율은 reported 기준과 CER 기준으로 각각 얼마였나?", q_en "What was Sanofi's Q1 2024 business EPS, and its change versus prior year on a reported basis and at CER?"
- **A19** (A, 차트, 기간 모호) / 기준 정답 불완전: "다른 기간은 Q1과 Q3 덱에 있음"이라고만 적혀 있어 채점자가 값을 모름 / 근거: novartis_2025Q1_deck p20 Q1 2025 USD 3.4bn(Q1 2024 2.0, +66%), novartis_2025Q3_deck p21 9M 2025 USD 15.9bn(9M 2024 12.6, +26%), Q2 p21 H1 9.7 확인 / 권고: answer에 세 기간 값을 모두 적음. answer_keys에 "3.4", "15.9" 추가
- **A20** (A, 차트, 기간 모호) / 기준 정답 불완전: 6월 말 값만 있고 gold인 Q1과 Q3 덱의 값이 없음 / 근거: astrazeneca_2025Q1_deck p12 end March 2025 $26.1bn, astrazeneca_2025Q3_deck p12 end Sep 2025 $24.0bn, Q2 p13 $25.2bn, end 2024 $24.6bn 확인 / 권고: answer "End Mar 2025 $26.1bn; end Jun 2025 $25.2bn; end Sep 2025 $24.0bn (end 2024 $24.6bn)", answer_keys ["26.1", "25.2", "24.0"]
- **B17** (B, 표) / answer_keys 불일치: 질문은 매출과 CER 성장률을 묻는데 keys는 매출과 전년 매출 / 근거: roche_2025Q2_deck p39, United States 12,670, HY 2024 11,882, CER +10 / 권고: answer_keys ["12,670", "10"]
- **D07** (D, 차트) / 기준 정답이 gold의 한 페이지와 어긋남: Novartis gold 그룹에 든 novartis_2025Q1_deck p34 워터폴은 배당을 -5.3(순액, 스위스 원천세 2.5 차감)으로 표시하고 7.8은 각주에만 있음. 그 페이지를 인용해 5.3이라 답하면 기준 정답(7.8)과 달라 오답 처리될 수 있음 / 근거: Q1 p34 "Annual dividend -5.3", 각주 2 "gross dividend of USD 7.8 billion reduced by the USD 2.5 billion Swiss withholding tax". Q2 p37은 -7.8, Roche p31 dividends paid -7.9, AZ p13 dividend 3.4 확인 / 권고: answer에 "Novartis USD 7.8bn gross (USD 5.3bn net of withholding tax, Q1 deck)"를 적고 두 값 모두 정답 처리
- **D08** (D, 표) / answer_keys 불일치: 질문은 고정환율 성장률인데 keys는 매출 절대액 / 근거: novartis_2025Q2_deck p20 H1 +13% cc(27,287), roche_2025Q2_deck p20 +7 CER(30,944), astrazeneca_2025Q2_deck p10 11 CER(28,045) 확인 / 권고: answer_keys ["13%", "7%", "11%"]

### 다. low

- **A01** (A, 차트) / 한영 불일치와 회사명 없음: q_ko에만 괄호 부연(소아 접종 설명)이 있고 q_en에는 없음. 두 언어 모두 Sanofi를 적지 않음(PPH & Boosters는 Sanofi 고유 라벨이라 실질 문제는 작음) / 근거: sanofi_2025Q1_deck p9 Q1 2024 막대 637, sanofi_2024Q1_pr p5 636(-0.5%) 확인 / 권고(선택, 재임베딩 필요): q_ko "Sanofi의 2024년 1분기 PPH & Boosters 백신 매출은 얼마였나?", q_en "What were Sanofi's PPH & Boosters vaccine sales in Q1 2024?"
- **A04** (A, 차트) / 기준 정답 표현 오류: "five consecutive quarterly increases"는 틀림. 5개 분기이므로 증가는 4회. 또한 "매 분기 증가했나"가 참이면 최저와 최고 분기는 자동으로 첫 분기와 마지막 분기라 뒷 질문이 중복됨 / 근거: sanofi_2025Q2_deck p9 막대 약 115, 130, 145, 150, 175 ($m, 라벨 없음), +52.5% / 권고: answer를 "four consecutive increases over five quarters"로 고침. 문구는 유지 가능
- **A05** (A, 차트) / answer_keys 불완전: Q4 덱은 2026년을 0.7로 바꿨는데 keys는 "0.8"만 있음. answer 본문은 이미 설명함 / 근거: sanofi_2025Q2_deck p14 1.1, 0.8, 0 확인. sanofi_2025Q4_deck p35 텍스트에 0.7 / 권고: answer_keys ["1.1", "0.8", "0.7"]
- **A09** (A, 차트, 기간 모호) / answer_keys 불일치: 질문은 비중(%)인데 keys는 미국 매출 절대액. 차트에 비중은 인쇄돼 있지 않아 계산이 필요함 / 근거: sanofi_2025Q1_deck p7 US 2,476 / 3,480 = 71% / 권고: answer_keys ["71", "73", "74"] 또는 유지(채점은 answer 본문 기준이므로 영향 작음)
- **A12** (A, 차트) / 기준 정답 부호 혼동: "CFO -$7.1bn"이라 적혀 있으나 차트에서 CFO 7.1은 순부채를 줄이는 유입 / 근거: astrazeneca_2025Q2_deck p13, 24.6 → CFO 7.1 → CapEx 1.3 → Deal 2.3 → Dividend 3.4 → Other 0.8 → 25.2 / 권고: "CFO $7.1bn inflow (reduces net debt)"
- **A16** (A, 차트) / 한영 불일치: q_ko는 "Novartis Kesimpta", q_en은 "Kesimpta"만. 제품명이 고유해 답은 같음 / 근거: novartis_2025Q2_deck p7 Q2 2024 ex-US 244, US 555, 합계 799 확인 / 권고(선택, 재임베딩 필요): q_en "What were Novartis's Kesimpta ex-US sales in Q2 2024?"
- **B05** (B, 표) / 참고: "H1 2025" 열이지만 실제 집계는 클로징(4월 30일) 이후 5월 1일부터 6월 30일까지의 두 달. 기준 정답 notes가 이미 밝힘 / 근거: sanofi_2025Q2_pr p25, Net sales and other revenues 887, Net income 24, 각주 "With effect from May 1, 2025 ... equity method" / 권고: 수정 불필요. 채점 시 "두 달치"라는 단서를 붙인 답을 감점하지 말 것
- **C02** (C, 본문) / 모호: "소아 적응증 확대는 언제였나"에 규제기관이 없음. FDA(2024년 1월)와 EU(2024년 11월) 두 답이 가능. 기준 정답은 둘 다 적음 / 근거: sanofi_20F_FY2025 p31, "On May 20, 2022, the FDA approved ... aged 12 years and older", "In January 2024, Dupixent was approved by the FDA for ... aged one year or older, weighing at least 15 kilograms", "approved in the EU in November 2024" / 권고(선택, 재임베딩 필요): q_ko 끝을 "FDA의 소아 적응증 확대 승인은 언제였나?"로, q_en을 "and when did the FDA expand the pediatric indication?"로
- **C06** (C, 본문) / answer_keys 불일치: 두 번째 키 "Interim Chief Executive Officer"는 질문(후임과 취임 시점)의 답이 아님 / 근거: sanofi_20F_FY2025 p117, "appointed Belén Garijo to succeed him after the Shareholders' Meeting of April 29, 2026", Hudson 해임 2026년 2월 17일 종료 시점, Charmeil 임시 CEO 2월 18일부터 / 권고: answer_keys ["Belén Garijo", "April 29, 2026"]
- **C10** (C, 본문, 기간 모호) / answer_keys 형태: "Group sales growth +6%" 같은 문장형 키는 생성 답변에 그대로 나오지 않음. 유형 라벨은 text이나 gold 10장 중 7장이 차트 또는 표 페이지 / 근거: roche_2025Q1_deck p6 "Group sales growth +6%", Q2 p20 CER +7, Q3 p6 +7 / 권고: answer_keys ["+6%", "+7%"]. 유형 라벨은 유지해도 무방(기간 모호 문항이라 어느 페이지든 정답)

## 3. 이상 없음 (42건, 첫 gold 페이지 렌더로 값 확인)

- A02 (-201, -0.08), A03 (Q4'24 +9%, Q1'25 +6%), A06 (+2,492, -347), A08 (주 24: Q12W 18.1%, Q4W 15.2%), A10 (11.2 at Mar 31 2025), A11 (China 3,515 +5%, Ex-China 4,182 +19%), A13 (11,955 +16%), A14 (-1,194, 6,309), A15 (-21.0, -7.9), A17 (Kisqali +460, Entresto +459), A18 (-23.8, -7.8)
- B01 (2,476 +18.4%), B02 (53 +271.4%), B03 (261 +166.3%), B04 (13,200), B06 (July 2035, November 2032, March 2032), B07 (588 +28.5% CER, +24.8% reported), B08 (06/23/2025, 3.000%, 750), B09 (7,991), B10 (3,480), B11 (3,303 +29.2%), B12 (7.12 +4.1% CER, -1.8% reported), B13 (4,246 +32.2%), B14 (28,045, R&D 6,707 +16%), B15 (11.08, 12,010), B16 (14,054, 5,925, 42.2%), B19 (13,233 +15% cc), B20 (13,588)
- C01 (IL-4/IL-13, type 2 inflammation), C03 (61%, TrumpRx 약 70%, $35 인슐린 상한, 관세 면제), C04 (15%, £25k~£35k/QALY, 0.6% GDP by 2035), C05 (Q1 mid-to-high single-digit, Q2 high single-digit, Q3 unchanged), C07 (Chief Digital Officer), C08 ($80bn by 2030), C09 (702,562,700 Genussscheine → participation certificates CHF 0.001, AGM 2026)
- D01 (3,480 → 3,832 → 4,156, +676, +19.4%), D03 (11.2, 5.1, 11.1; Opella -10.7, M&A 10.4, 배당 4.8, 자사주 4.1), D04 (72%, 80.3%, 86.1%), D05 (284, 72, 739), D06 (5.1, -23.8, -21.0, 25.2), D09 (3,303 +29.2% → 3,832 +21.1%), D10 (91.8% vs 64%)

## 4. 확인하지 못한 것

- A04의 막대 값은 라벨이 없어 눈대중(약 115, 130, 145, 150, 175). 순서와 방향은 확실함
- 기간 모호 문항(A09, A10, A19, A20, B09, B10, B19, B20, C05, C10)은 첫 gold 페이지만 렌더했고 나머지는 텍스트 레이어로 키 존재만 확인함
- 20-F FY2024 쪽 gold 페이지(B06, C01, C02, C07)는 텍스트 레이어로만 확인함
- 기계 판독용 파일: eval/results/question_review.json
