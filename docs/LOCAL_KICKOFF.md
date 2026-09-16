# 로컬 Claude Code 킥오프 프롬프트

로컬(cmd)에서 `claude`를 실행한 뒤 아래 블록을 그대로 붙여넣는다. `CLAUDE.md`는 자동으로 읽히므로
프롬프트는 "무엇을, 어떤 순서로, 어디서 멈출지"만 지정한다.

---

```
docs/EXPERIMENT_PLAN.md를 먼저 읽고 시작해. CLAUDE.md의 실행 규칙(PYTHONIOENCODING=utf-8, Qdrant 6335/6336, data/pdf 파일명)을 지켜.

0. 환경 점검부터 해서 결과를 표로 보여줘 — 다음 단계로 넘어가기 전에:
   - data/pdf/ 에 Q1.pdf, Q2.pdf, Q3.pdf, "Form 20-F 2025 (Oct 2025).pdf" 4개가 있는지, 각 페이지 수 (pypdfium2로 실측)
   - .env 존재 여부 (값은 절대 출력하지 마)
   - docker compose ps 로 Qdrant 컨테이너 상태, 6335 응답 여부
   - 파이썬 venv 활성 여부와 requirements 설치 상태
   빠진 게 있으면 뭘 준비해야 하는지 알려주고 멈춰.

1. 환경이 되면 EXPERIMENT_PLAN §5의 순서 1~3만 진행해 (PDF만 있으면 되는 작업. API 키·Colab·Qdrant 불필요):
   1) 20-F 300페이지를 페이지별로 훑어 콘텐츠 유형을 분류하고, 발췌 50페이지 후보 구간을 표·차트·서술이 섞이도록 제안해. 내가 확정하면 data/pdf/20F_extract.pdf 생성 + docs/CORPUS.md에 선정 근거 기록.
   2) 코퍼스 전 페이지(Q1/Q2/Q3/20F_extract) 인벤토리 → eval/page_inventory.csv (source, page, block_type[chart/table/text/mixed], note).
   3) 인벤토리 기준으로 한국어 질문 30개 + 영어 대응쌍 → eval/questions.jsonl (§2.4 스키마와 유형 분포 A10/B10/C5/D5, explicit 25/ambiguous 5). 정답은 페이지를 렌더해서 직접 확인한 값만 넣고, 확인 못 한 건 "needs_review": true 로 표시해. 검증 스크립트 scripts/12_validate_questions.py도 만들어서 통과시켜.

2. 각 단계 끝날 때마다 커밋해 (prefix: feat(phase-2)). data/pdf/, .env, *.csv 는 커밋하지 마 — 단 eval/questions.jsonl과 eval/page_inventory.csv는 커밋 대상이야 (.gitignore 예외 추가 필요하면 해).

3. 순서 4(Colab 배치 인덱싱 노트북)부터는 내 확인 받고 시작해. 그 전엔 API 호출·Colab·Qdrant 쓰는 작업 하지 마.

막히거나 판단이 필요한 건 추측하지 말고 물어봐.
```

---

## 이후 세션용 짧은 프롬프트

```
docs/EXPERIMENT_PLAN.md §5 기준 현재 어디까지 됐는지 git log와 파일 존재 여부로 확인하고, 다음 미완 단계 하나만 진행해. 끝나면 커밋하고 멈춰.
```
