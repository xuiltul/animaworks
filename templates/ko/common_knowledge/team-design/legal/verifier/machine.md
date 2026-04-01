# Legal Verifier — machine 활용 패턴

## 기본 규칙

1. **계획서를 먼저 쓴다** — 인라인 짧은 지시 문자열로의 실행은 금지. 계획서 파일을 넘긴다
2. **출력은 드래프트** — machine 출력은 반드시 스스로 검증하고, `status: approved`로 한 뒤 다음 공정으로 넘긴다
3. **저장 위치**: `state/plans/{YYYY-MM-DD}_{개요}.{type}.md`(`/tmp/` 금지)
4. **레이트 제한**: chat 5회/session, heartbeat 2회
5. **machine은 인프라에 접근할 수 없다** — 기억·메시지·조직 정보는 계획서에 포함한다

---

## 개요

Verifier는 **machine에 검증 스캔을 위임하고, 그 스캔 결과의 정당성을 검증한다(메타 검증)**.

- 검증 관점의 설계 → Verifier 본인이 판단
- 차분 검출·carry-forward 대조 실행 → machine에 위임
- 검출 결과의 정당성 검증 → Verifier 본인이 판단
- 최악 시나리오 구성 → Verifier 본인이 판단

machine은 조항 차분 검출·carry-forward의 기계적 대조·리스크 평가의 전번 비교 등을 빠르게 수행할 수 있으나,
낙관적 편향 판정과 최악 시나리오 구성은 Verifier의 책임이다.

---

## 워크플로

### Step 1: 검증 계획서를 작성한다(Verifier 본인이 작성)

검증의 관점·대상·기준을 명확히 한 계획서를 만든다.

```bash
write_memory_file(path="state/plans/{date}_{사건명}.verification.md", content="...")
```

작성 전에 Anima 측에서 아래 정보를 준비한다:
- Director의 `audit-report.md`와 `analysis-plan.md`
- carry-forward tracker(전판 대비 비교용)
- 계약서 본문(수정 전후 양 버전)

### Step 2: machine에 검증 스캔을 던진다

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{verification계획서})" \
  -d /path/to/workspace
```

결과를 `state/plans/{date}_{사건명}.verification.md`에 덧붙이거나 덮어쓴다(`status: draft`).

### Step 3: 검증 결과를 메타 검증한다

Verifier가 verification.md를 읽고 아래를 확인한다:

- [ ] 지적 내용이 사실에 기초하는지(오검출이 없는지)
- [ ] carry-forward 대조 결과에 누락이 없는지
- [ ] Director의 「수용 가능」 판정에 대한 최악 시나리오 분석이 포함되어 있는지
- [ ] 리스크 평가의 전번 비교가 정확한지
- [ ] Verifier 본인의 관점에서 추가할 지적이 없는지

Verifier 본인이 수정·보충하고 `status: approved`로 변경한다.

### Step 4: 피드백

approved인 verification-report.md를 Director에게 송부한다.
지적이 있으면 구체적 반론 근거와 권장 수정을 명시한다.

---

## 검증 계획서 템플릿(verification.plan.md)

```markdown
# 검증 계획서: {검증 대상 개요}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: verification-plan

## 검증 관점

- [ ] 낙관적 편향 검출: 「수용 가능」 판정의 전 항목 재평가
- [ ] carry-forward 추적: 전번 지적의 전건이 이번 보고에 반영되어 있는지
- [ ] 리스크 평가의 전번 비교: 낮아진 평가에 충분한 근거가 있는지
- [ ] 법적 정확성: 인용 법령·판례의 정확성
- [ ] 조항 포괄성: 계약서의 전 조항이 분석 대상인지

## 대상

- audit-report.md: {경로}
- analysis-plan.md: {경로}
- carry-forward tracker: {경로}
- 계약서 본문: {경로 / 저장 위치}

## 차분 정보

{수정 전후 계약서의 주요 변경점. 전문이 아니라 차분에 한정}

## 출력 형식(필수)

아래 형식으로 검증 결과를 출력한다. **이 형식을 따르지 않은 출력은 무효로 한다.**

- **Critical**: 수정 필수 문제(낙관적 편향·silent drop·법적 오류)
- **Warning**: 수정 권장 문제(근거 부족·문구 리스크)
- **Info**: 정보 제공·개선 제안
```

## 검증 보고서 템플릿(verification-report.md)

```markdown
# 검증 보고서: {사건명}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: verification-report

## 종합 판정

{APPROVE / REQUEST_CHANGES / COMMENT}

## 낙관적 편향 검출

| # | 대상 조항 | Director 판정 | Verifier 소견 | 최악 시나리오 | 권장 |
|---|---------|-------------|-------------|------------|------|
| 1 | {조항} | {수용 가능} | {우려 내용} | {최악의 경우} | {수정안} |

## Carry-forward 누락 검출

| # | 전번 지적 | 전번 리스크 | 이번 보고에서의 취급 | 판정 |
|---|---------|----------|------------------|------|
| 1 | {지적} | {리스크} | {언급 있음 / silent drop} | {OK / NG} |

## 리스크 평가의 전번 비교

| # | 조항 | 전번 리스크 | 이번 리스크 | 변경 이유의 타당성 |
|---|------|----------|----------|---------------|
| 1 | {조항} | {전번} | {이번} | {타당 / 근거 부족} |

## 법적 정확성

| # | 지적 내용 | 중요도 |
|---|---------|--------|
| 1 | {내용} | {Critical/Warning/Info} |

## Verifier 소견

{Verifier 본인의 분석·추가 관찰·권장 사항}
```

---

## 제약 사항

- 검증 계획서(무엇을 관점으로 검증할지)는 MUST: Verifier 본인이 작성한다
- machine의 검증 결과를 그대로 Director에게 넘겨서는 안 된다(NEVER) — 반드시 Verifier가 메타 검증한다
- `status: approved`가 아닌 verification-report.md를 Director에게 피드백해서는 안 된다(NEVER)
- 「수용 가능」 판정에 대한 최악 시나리오 검토는 machine에 맡기지 않고 Verifier 본인이 수행한다(MUST)
