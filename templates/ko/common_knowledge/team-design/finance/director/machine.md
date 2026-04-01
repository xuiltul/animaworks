# Finance Director — machine 활용 패턴

## 기본 규칙

1. **계획서를 먼저 쓴다** — 인라인 짧은 지시 문자열로의 실행은 금지. 계획서 파일을 넘긴다
2. **출력은 초안** — machine 출력은 반드시 스스로 검증하고, 다음 공정으로 넘기기 전에 `status: approved`로 둔다
3. **저장 위치**: `state/plans/{YYYY-MM-DD}_{개요}.{type}.md`(`/tmp/` 금지)
4. **레이트 제한**: chat 5회/session, heartbeat 2회
5. **machine은 인프라에 접근할 수 없다** — 기억·메시지·조직 정보는 계획서에 포함한다

---

## 개요

Finance Director는 PdM(계획·판단)과 Engineer(실행)를 겸한다.

- 분석 계획(analysis-plan.md) 작성 → Director 자신이 쓴다
- 분석 실행 → machine에 위임하고 Director가 검증한다
- 리스크 평가·해석 확정 → Director 자신이 판단한다
- 검증은 2회: machine 분석 결과 확인 시와 Auditor로부터의 피드백 통합 시

---

## Phase 1: 분석 계획(PdM 상당)

### Step 1: Variance Tracker를 확인한다

전회 분석이 있으면 Variance Tracker를 읽고, 전 차이의 상태를 파악한다.

### Step 2: analysis-plan.md를 작성한다(Director 자신이 쓴다)

분석 목적·대상·관점·전회 인계 사항을 명확히 한 계획서를 작성한다.

```bash
write_memory_file(path="state/plans/{date}_{대상명}.analysis-plan.md", content="...")
```

**analysis-plan.md의 「분석 관점」「스코프」「전회 인계 사항」은 Director 판단의 핵이며, machine에 쓰게 해서는 안 된다(NEVER).**

### Step 3: analysis-plan.md를 승인한다

스스로 작성한 analysis-plan.md를 확인하고 `status: approved`로 변경한다.

## Phase 2: 분석 실행(Engineer 상당)

### Step 4: machine에 분석 실행을 넘긴다

analysis-plan.md와 소스 데이터를 입력으로 분석을 machine에 의뢰한다.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{analysis-plan.md})" \
  -d /path/to/workspace
```

결과를 `state/plans/{date}_{대상명}.analysis-report.md`로 저장한다(`status: draft`).

**분석 투입 시 주의사항:**
- 소스 데이터(Analyst/Collector로부터 수령한 데이터)를 계획서에 포함한다(machine은 기억에 접근할 수 없다)
- Variance Tracker의 전 차이를 포함한다(각 항목의 상태 갱신을 machine에 요구한다)
- 출력 형식을 명시한다

### Step 5: analysis-report.md를 검증한다

Director가 analysis-report.md를 읽고 `director/checklist.md`에 따라 검증한다:

- [ ] Variance Tracker의 전 차이가 커버되는가(silent drop 없음)
- [ ] 모든 수치를 프로그램적으로 검증했는가(assert 등으로 항등식·일관성 확인)
- [ ] 해석·가정에 충분한 근거가 있는가
- [ ] 추정값에 「추정」 마커가 붙어 있는가

문제가 있으면 Director 자신이 수정하고 `status: reviewed`로 변경한다.

### Step 6: 위임한다

`status: reviewed`인 analysis-report.md를 Auditor에게 `delegate_task`로 넘긴다.

## Phase 3: 통합과 최종 판단

### Step 7: 피드백을 통합한다

Auditor(audit-report.md)의 피드백을 받으면:

- 가정에 대한 지적을 받은 항목을 과거 데이터로 재검증한다
- Data Lineage 지적을 받은 수치의 소스를 재확인한다
- 수치 정확성 지적이 있으면 재계산한다
- Variance Tracker를 최신 상태로 갱신한다

### Step 8: 최종 보고

통합된 analysis-report.md를 `status: approved`로 변경하고 `call_human`으로 사람에게 최종 보고한다.

---

## 분석 계획서 템플릿(analysis-plan.md)

```markdown
# 재무 분석 계획서: {대상명}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: analysis-plan

## 분석 목적

{무엇을 밝힐지 — 1~3문}

## 대상 기간·대상 법인

| 대상 | 기간 | 소스 데이터 |
|------|------|------------|
| {법인명/대상명} | {기간} | {데이터 파일} |

## 전회 분석으로부터의 인계 사항

| # | 전회 차이 | 전회 상태 | 이번에 확인할 점 |
|---|---------|----------|----------------|
| V-1 | {내용} | 지속 모니터링 | {해소했는가? 악화하지 않았는가?} |
| ... | ... | ... | ... |

(전회 분석이 없으면 「최초 분석」으로 명시)

## 분석 관점(스코프)

{Director 자신의 판단으로 설정}

1. {관점1}
2. {관점2}
3. {관점3}

## 스코프 밖

- {제외 대상}

## 출력 형식

- 주요 지표 요약
- 이상값·중요 차이 목록
- 계정과목별 분석(해당 시)
- Variance Tracker 상태 갱신
- 권장 액션

## 기한

{deadline}
```

## 분석 보고서 템플릿(analysis-report.md)

```markdown
# 재무 분석 보고서: {대상명}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: analysis-report
source: state/plans/{원본 analysis-plan.md}

## 종합 평가

{분석 전체 요약 — 1~3문}

## 주요 지표 요약

| 지표 | 당기 | 전기 | 변동률 | 평가 |
|------|------|------|--------|------|
| {지표명} | {값} | {값} | {%} | {정상/주의/이상} |

## 이상값·중요 차이 목록

| # | 항목 | 변동률 | 리스크 | 근거 | 권장 액션 |
|---|------|--------|--------|------|----------|
| 1 | {항목명} | {%} | Critical | {분석 근거} | {구체적 액션} |
| 2 | {항목명} | {%} | High | {분석 근거} | {구체적 액션} |

## 계정과목별 분석

### {과목명}

- **수치**: {당기값} / {전기값} / 변동 {%}
- **분석**: {변동 원인·배경}
- **가정**: {가정이 있으면 명시}
- **권장 액션**: {구체적 액션}

(각 과목에 대해 반복)

## Variance Tracker 상태 갱신

| # | 전회 차이 | 전회 상태 | 이번 상태 | 잔존 리스크 |
|---|---------|----------|----------|-----------|
| V-1 | {내용} | 지속 모니터링 | {해소/지속 모니터링/악화} | {평가} |

## 수치 검증 결과

{프로그램적 검증(assert 등) 실행 결과 요약}

## 권장 액션

{우선순위별로 정리한 구체적 액션}
```

---

## 제약 사항

- analysis-plan.md는 MUST: Director 자신이 쓴다
- analysis-report의 해석·판단은 MUST: Director 자신이 확정한다(machine 출력은 초안으로 검증한다)
- `status: reviewed`가 붙지 않은 analysis-report.md를 Auditor에게 넘겨서는 안 된다(NEVER)
- Variance Tracker에 기록된 차이를 언급 없이 소멸시켜서는 안 된다(NEVER — silent drop 금지)
- 모든 수치를 프로그램적으로 검증한 뒤 보고한다(MUST — LLM 암산을 믿지 않는다)
