# 팀 설계 가이드 — 기본 원칙

## 개요

이 문서는 목적별로 Anima 팀을 설계할 때의 기본 원칙을 정의한다.
프레임워크의 조직 메커니즘(`organization/roles.md`, `organization/hierarchy-rules.md`)과는 별도로,
**「어떤 목표에 대해 어떤 역할 구성으로 팀을 짤 것인가」**를 설계하기 위한 가이드이다.

---

## 왜 기능을 나누는가

AI 에이전트에서 역할 분리는 인간 팀과 다른 이유가 있다:

| 이유 | 설명 |
|------|------|
| **컨텍스트 오염 방지** | 한 에이전트가 전 과정을 맡으면 컨텍스트 윈도우가 비대해지고 판단이 나빠진다. 역할로 범위를 한정한다. |
| **전문성 심화** | 역할별 지침·체크리스트·기억을 두면 범용 에이전트보다 품질을 높일 수 있다. |
| **병렬 실행** | 독립된 관점의 역할(예: 리뷰와 테스트)은 동시에 돌려 처리량을 올릴 수 있다. |
| **품질의 구조적 보장** | 「실행」과 「검증」을 나누면 셀프 리뷰의 사각을 줄인다. |

---

## 설계 원칙

### 1. 단일 책임(Single Responsibility)

한 역할에는 하나의 명확한 책무를 둔다. 모호한 역할은 판단 기준이 불명확해져 품질이 떨어진다.

**좋은 예**: 「코드 리뷰 수행 및 품질 판정」(Reviewer)  
**나쁜 예**: 「구현과 리뷰와 테스트」(올인원)

### 2. 실행과 검증의 분리

machine이 실행한 출력을 **같은 Anima가 검증**하는 구조로 둔다. machine 출력을 검증 없이 다음 단계로 넘기지 않는다.

```
Anima가 지시서 작성 → machine 실행 → Anima가 검증·승인
```

### 3. 문서 기반 핸드오프

역할 간 인계는 상태가 있는 Markdown으로 한다. 메시지 본문만으로 넘기면 정보가 빠진다.

```
plan.md (status: approved) → delegate_task로 Engineer에게
```

### 4. 병렬 실행 가능 설계

독립된 역할은 병렬로 돌릴 수 있게 설계한다. 의존이 없는 역할을 불필요하게 직렬로 두지 않는다.

---

## 팀 설계 프로세스

### 1단계: 목표 정의

팀이 달성해야 할 목표를 한 문장으로 명확히 한다.

예:
- 「계획부터 구현·검증까지 소프트웨어 개발 프로젝트를 일관되게 수행한다」
- 「고객 지원 1차 대응을 24시간 한다」

### 2단계: 역할 분해

필요한 기능을 나열하고 단일 책임에 따라 역할로 쪼갠다.

판단 기준:
- **판단이 필요한가** → 독립 역할 가치가 있음
- **다른 작업과 독립적으로 실행 가능한가** → 분리해 병렬 이점을 취함
- **전문 지식이 필요한가** → 전문 역할로 품질을 올림

### 3단계: 책임 경계 명확화

각 역할의 MUST / SHOULD / MAY를 정의한다. 특히 인접 역할 간 경계를 명확히 한다.

### 4단계: 핸드오프 체인 설계

문서 전달 순서와 병렬 실행 지점을 정한다.

### 5단계: 역할 템플릿 선택

프레임워크 `--role`(engineer, manager, writer, researcher, ops, general) 중 가장 가까운 것을 고르고,
`injection.md`에서 팀 설계로 덮어쓴다.

---

## 겸임 판단

소규모 팀이나 리소스 제약이 있으면 한 Anima가 여러 역할을 겸할 수 있다.

### 겸임해도 되는 경우

- **태스크 규모가 작음** — 한 사람이 전 과정을 돌려도 품질 유지 가능
- **관점이 가까움** — 예: PdM + Engineer(소규모 변경에서 계획·구현이 밀접)
- **비용** — 전용 Anima를 둘 만큼의 작업량이 없음

### 분리하는 편이 좋은 경우

- **실행과 검증이 동일 인물** — Engineer가 자기 코드를 리뷰하는 상태는 피함
- **컨텍스트 충돌** — 리뷰 관점과 구현 관점 전환이 잦음
- **병렬 이득이 큼** — 리뷰와 테스트를 동시에 돌리고 싶음

### 겸임 시

**역할 전환을 의식**한다. 「지금은 Reviewer로 판단」「지금은 Engineer로 구현」을 구분한다.

---

## 규모 스케일링

| 규모 | 구성 | 적용 시나리오 |
|------|------|---------------|
| **솔로** | 1 Anima가 전 역할 겸임 | 소규모 태스크, 프로토타입 |
| **페어** | PdM + Engineer(리뷰는 Engineer 겸임) | 중형 정형 태스크 |
| **풀 팀** | PdM + Engineer + Reviewer + Tester | 본격 프로젝트 |
| **스케일** | PdM + Engineer 여러 명 + Reviewer/Tester 1~2명씩 | 대규모·다중 모듈 |

스케일업 판단:
- 실패 비용이 크다 → 역할 분리 증가
- 병렬 가능한 모듈이 많다 → Engineer 증가
- 품질 요구가 높다 → Reviewer·Tester 독립

---

## 팀 템플릿 목록

| 템플릿 | 경로 | 개요 |
|--------|------|------|
| 개발 풀 팀 | `team-design/development/team.md` | PdM + Engineer + Reviewer + Tester 4역할 |
| 법무 풀 팀 | `team-design/legal/team.md` | Legal Director + Legal Verifier + Precedent Researcher 3역할 |
| 재무 풀 팀 | `team-design/finance/team.md` | Finance Director + Financial Auditor + Data Analyst + Market Data Collector 4역할 |
| 트레이딩 풀 팀 | `team-design/trading/team.md` | Strategy Director + Market Analyst + Trading Engineer + Risk Auditor 4역할 |
| 영업·마케팅 풀 팀 | `team-design/sales-marketing/team.md` | Director + Marketing Creator + SDR + Market Researcher 4역할 |

> 새 템플릿을 추가할 때는 같은 구조(`team.md` + 역할별 디렉터리)로 `team-design/{팀 이름}/`에 둔다.
