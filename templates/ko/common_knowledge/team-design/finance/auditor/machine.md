# Financial Auditor — machine 활용 패턴

## 기본 규칙

1. **계획서를 먼저 쓴다** — 인라인 짧은 지시 문자열로의 실행은 금지. 계획서 파일을 넘긴다
2. **출력은 초안** — machine 출력은 반드시 스스로 검증하고, 다음 공정으로 넘기기 전에 `status: approved`로 둔다
3. **저장 위치**: `state/plans/{YYYY-MM-DD}_{개요}.{type}.md`(`/tmp/` 금지)
4. **레이트 제한**: chat 5회/session, heartbeat 2회
5. **machine은 인프라에 접근할 수 없다** — 기억·메시지·조직 정보는 계획서에 포함한다

---

## 개요

Auditor는 **machine에 검증 스캔을 위임하고, 그 스캔 결과의 정당성을 검증한다(메타 검증)**.

- 검증 관점 설계 → Auditor 자신이 판단한다
- 수치 재계산·Variance Tracker 대조·Data Lineage 추적 → machine에 위임한다
- 검출 결과의 정당성 검증 → Auditor 자신이 판단한다
- 가정의 타당성 검증(Assumption Challenge) → Auditor 자신이 판단한다

machine은 수치 재계산·차분 검출·Data Lineage의 기계적 추적을 빠르게 할 수 있으나,
가정의 타당성 판정이나 비관 시나리오 구성은 Auditor의 책무이다.

---

## 워크플로

### Step 1: 검증 계획서를 작성한다(Auditor 자신이 쓴다)

검증 관점·대상·기준을 명확히 한 계획서를 작성한다.

```bash
write_memory_file(path="state/plans/{date}_{대상명}.audit-plan.md", content="...")
```

작성 전에 Anima 측에서 다음 정보를 준비한다:
- Director의 `analysis-report.md`와 `analysis-plan.md`
- Variance Tracker(전회와의 비교용)
- 소스 데이터(독립 검증에 필요한 경우)

### Step 2: machine에 검증 스캔을 넘긴다

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{audit 계획서})" \
  -d /path/to/workspace
```

결과를 `state/plans/{date}_{대상명}.audit-report.md`에 추가하거나 덮어쓴다(`status: draft`).

### Step 3: 검증 결과를 메타 검증한다

Auditor가 audit-report.md를 읽고 다음을 확인한다:

- [ ] 지적 내용이 사실에 기반하는가(오검출이 없는가)
- [ ] Variance Tracker 대조 결과에 누락이 없는가
- [ ] Data Lineage 추적 결과가 정확한가
- [ ] Director 가정에 대한 Assumption Challenge가 포함되어 있는가
- [ ] Auditor 자신의 관점에서 추가할 지적이 없는가

Auditor 자신이 수정·보완하고 `status: approved`로 변경한다.

### Step 4: 피드백

approved인 audit-report.md를 Director에게 송부한다.
지적이 있으면 구체적 반증 데이터와 권장 수정을 명시한다.

---

## 검증 계획서 템플릿(audit-plan.md)

```markdown
# 검증 계획서: {검증 대상 개요}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: audit-plan

## 검증 관점

- [ ] Assumption Challenge: Director 가정·해석을 반증적 관점으로 검증
- [ ] Variance Tracker 추적: 전회 차이의 전건이 이번 보고에 반영되었는지
- [ ] Data Lineage: 모든 수치가 소스 데이터까지 소급 가능한지
- [ ] 수치 정확성: 주요 지표의 독립 재계산
- [ ] 회계 항등식: BS 등식·TB 균형 성립 확인

## 대상

- analysis-report.md: {경로}
- analysis-plan.md: {경로}
- Variance Tracker: {경로}
- 소스 데이터: {경로 / 보관 위치}

## 출력 형식(필수)

다음 형식으로 검증 결과를 출력할 것. **이 형식을 따르지 않은 출력은 무효로 한다.**

- **Critical**: 수정 필수 문제(가정 불비·silent drop·수치 불일치)
- **Warning**: 수정 권장 문제(근거 부족·Data Lineage 불완전)
- **Info**: 정보 제공·개선 제안
```

## 검증 보고서 템플릿(audit-report.md)

```markdown
# 검증 보고서: {대상명}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: audit-report

## 종합 판정

{APPROVE / REQUEST_CHANGES / COMMENT}

## Assumption Challenge 결과

| # | 대상 항목 | Director 판단 | Auditor 소견 | 반증 데이터 | 권장 |
|---|---------|---------------|-------------|----------|------|
| 1 | {항목} | {계절 요인} | {우려 내용} | {과거 데이터} | {수정안} |

## Variance Tracker 누락 검출

| # | 전회 차이 | 전회 상태 | 이번 보고에서의 취급 | 판정 |
|---|---------|----------|------------------|------|
| 1 | {차이} | {상태} | {언급 있음 / silent drop} | {OK / NG} |

## Data Lineage 검증

| # | 대상 수치 | 소스 정보 | 판정 |
|---|---------|----------|------|
| 1 | {수치·지표} | {추적 가능 / 소스 불명} | {OK / NG} |

## 수치 재계산 결과

| # | 지표 | Director 값 | Auditor 재계산값 | 차이 | 판정 |
|---|------|-------------|----------------|------|------|
| 1 | {지표} | {값} | {값} | {차이} | {OK / NG} |

## Auditor 소견

{Auditor 자신의 분석·추가 관찰·권장 사항}
```

---

## 제약 사항

- 검증 계획서(무엇을 관점으로 검증할지)는 MUST: Auditor 자신이 쓴다
- machine 검증 결과를 그대로 Director에게 넘겨서는 안 된다(NEVER) — 반드시 Auditor가 메타 검증한다
- `status: approved`가 아닌 audit-report.md를 Director에게 피드백해서는 안 된다(NEVER)
- 가정의 타당성 검증(Assumption Challenge)은 machine에 맡기지 않고 Auditor 자신이 수행한다(MUST)
