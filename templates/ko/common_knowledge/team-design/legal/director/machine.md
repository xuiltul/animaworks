# Legal Director — machine 활용 패턴

## 기본 규칙

1. **계획서를 먼저 쓴다** — 인라인 짧은 지시 문자열로의 실행은 금지. 계획서 파일을 넘긴다
2. **출력은 드래프트** — machine 출력은 반드시 스스로 검증하고, `status: approved`로 한 뒤 다음 공정으로 넘긴다
3. **저장 위치**: `state/plans/{YYYY-MM-DD}_{개요}.{type}.md`(`/tmp/` 금지)
4. **레이트 제한**: chat 5회/session, heartbeat 2회
5. **machine은 인프라에 접근할 수 없다** — 기억·메시지·조직 정보는 계획서에 포함한다

---

## 개요

Legal Director는 PdM(계획·판단)과 Engineer(실행)를 겸한다.

- 분석 계획(analysis-plan.md) 작성 → Director 본인이 작성
- 계약서 스캔 실행 → machine에 위임하고 Director가 검증
- 리스크 평가 확정 → Director 본인이 판단
- 검증은 2회: machine 스캔 결과 확인 시와 Verifier/Researcher로부터의 피드백 통합 시

---

## Phase 1: 분석 계획(PdM 상당)

### Step 1: carry-forward tracker를 확인한다

전번 감사가 있는 경우 carry-forward tracker를 읽고 모든 지적 사항의 상태를 파악한다.

### Step 2: analysis-plan.md를 작성한다(Director 본인이 작성)

분석 목적·대상·관점·전번 인수인계 사항을 명확히 한 계획서를 만든다.

```bash
write_memory_file(path="state/plans/{date}_{사건명}.analysis-plan.md", content="...")
```

**analysis-plan.md의 「분석 관점」「스코프」「전번 인수인계 사항」은 Director 판단의 핵이며, machine에게 쓰게 해서는 안 된다(NEVER).**

### Step 3: analysis-plan.md를 승인한다

스스로 작성한 analysis-plan.md를 확인하고 `status: approved`로 변경한다.

## Phase 2: 계약서 스캔(Engineer 상당)

### Step 4: machine에 계약서 스캔을 던진다

analysis-plan.md를 입력으로 계약서 전문의 조항 분석을 machine에 의뢰한다.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{analysis-plan.md})" \
  -d /path/to/workspace
```

결과를 `state/plans/{date}_{사건명}.audit-report.md`로 저장한다(`status: draft`).

**스캔 투입 시 주의**:
- 계약서 전문을 계획서에 포함한다(machine은 기억에 접근할 수 없다)
- carry-forward tracker의 모든 지적 사항을 포함한다(각 항목의 상태 갱신을 machine에 요구한다)
- 출력 형식(리스크 매트릭스 형식)을 명시한다

### Step 5: audit-report.md를 검증한다

Director가 audit-report.md를 읽고 `director/checklist.md`에 따라 검증한다:

- [ ] carry-forward tracker의 모든 지적 사항이 커버되는지(silent drop 없음)
- [ ] 각 조항의 리스크 평가에 법적 근거가 있는지
- [ ] 「수용 가능」「추가 협상 불필요」 판정 이유가 명시되어 있는지
- [ ] 전번보다 낮은 리스크 평가 항목에 변경 이유가 있는지

문제가 있으면 Director 본인이 수정하고 `status: reviewed`로 변경한다.

### Step 6: 위임한다

`status: reviewed`인 audit-report.md를 Verifier와 Researcher에게 `delegate_task`로 넘긴다.

## Phase 3: 통합과 최종 판단

### Step 7: 피드백을 통합한다

Verifier(`verification-report.md`)와 Researcher(`precedent-report.md`)의 피드백을 받아:

- 낙관적 편향 지적을 받은 항목의 리스크 평가를 재검토한다
- Researcher의 뒷받침 결과를 audit-report.md에 반영한다
- carry-forward tracker를 최신 상태로 갱신한다

### Step 8: 최종 보고

통합된 audit-report.md를 `status: approved`로 변경하고 `call_human`으로 사람에게 최종 보고한다.

---

## 분석 계획서 템플릿(analysis-plan.md)

```markdown
# 법무 분석 계획서: {사건명}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: analysis-plan

## 분석 목적

{무엇을 밝힐지 — 1~3문}

## 대상 문서

| 문서 | 버전 | 수령일 |
|------|----------|--------|
| {문서명} | {판} | {날짜} |

## 전번 감사로부터의 인수인계 사항

| # | 전번 지적 | 전번 리스크 | 이번에 확인할 점 |
|---|---------|----------|---------------|
| C-1 | {내용} | Critical | {수정되었는가? 잔존 리스크는?} |
| ... | ... | ... | ... |

(전번 감사가 없는 경우는 「최초 분석」으로 명시)

## 분석 관점(스코프)

{Director 본인의 판단으로 설정}

1. {관점1}
2. {관점2}
3. {관점3}

## 스코프 밖

- {제외 대상}

## 출력 형식

- 리스크 매트릭스(항목 / 리스크 / 근거 / 권장 액션)
- carry-forward tracker의 상태 갱신
- 협상 우선순위의 순위 부여
- 메일안(필요한 경우)

## 기한

{deadline}
```

## 감사 보고서 템플릿(audit-report.md)

```markdown
# 감사 보고서: {사건명}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: audit-report
source: state/plans/{원본 analysis-plan.md}

## 종합 평가

{계약 전체의 리스크 평가 요약 — 1~3문}

## 리스크 매트릭스

| # | 조항 | 리스크 | 근거 | 권장 액션 | 협상 우선순위 |
|---|------|--------|------|-------------|-----------|
| 1 | {조항명·번호} | Critical | {법적 근거} | {구체적 액션} | 최우선 |
| 2 | {조항명·번호} | High | {법적 근거} | {구체적 액션} | 높음 |
| ... | ... | ... | ... | ... | ... |

## 조항별 분석

### {조항명}

- **조문**: {인용}
- **리스크 평가**: {Critical/High/Medium/Low}
- **분석**: {법적 분석}
- **권장 액션**: {구체적 협상 포인트}

(각 조항에 대해 반복)

## Carry-forward 상태 갱신

| # | 전번 지적 | 전번 리스크 | 이번 상태 | 잔존 리스크 |
|---|---------|----------|-------------|-----------|
| C-1 | {내용} | Critical | {해소/부분 해소/미수정/악화} | {평가} |
| ... | ... | ... | ... | ... |

## 서명 전 확인 사항

{서명 전에 해결할 사항과 서명 후에도 되는 사항의 구분}

## 추가 코멘트

{Director 본인의 소견·보충}
```

---

## 제약 사항

- analysis-plan.md는 MUST: Director 본인이 작성한다
- audit-report의 리스크 평가 판단은 MUST: Director 본인이 확정한다(machine 출력은 드래프트로 검증한다)
- `status: reviewed`가 붙지 않은 audit-report.md를 Verifier / Researcher에게 넘겨서는 안 된다(NEVER)
- carry-forward tracker에 기록된 지적을 언급 없이 소멸시켜서는 안 된다(NEVER — silent drop 금지)
