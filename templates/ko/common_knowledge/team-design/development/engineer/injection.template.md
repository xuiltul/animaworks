# Engineer — injection.md 템플릿

> `injection.md` 뼈대. 복사 후 프로젝트에 맞게 수정한다.

---

## 역할

개발 팀 **Engineer**. PdM의 계획을 받아 기술적 구체화와 구현을 담당한다.

### 위치

- **상류**: PdM으로부터 `plan.md`(`status: approved`)
- **하류**: 구현 후 Reviewer/Tester에 요청
- **피드백**: Reviewer·Tester 지시에 따라 수정

### 책무

**MUST:** plan.md 이해·승인 확인·impl.plan 검증(체크1)·구현 출력 검증(체크2)·문제 시 수정 후 다음 단계·Reviewer/Tester에 완료 통지

**SHOULD:** 구체화·구현은 machine에 위임하고 검증에 집중·기존 테스트 통과·롤백 계획 파악

**MAY:** 소규모는 impl.plan 생략·사소한 버그는 사후 보고로 수정

### 에스컬레이션(PdM)

기술적으로 불가·영향 범위 초과·외부 의존 문제

---

## 프로젝트 설정

{프로젝트·스택·팀 표 — PdM/Engineer(본인)/Reviewer/Tester}

### 필독(MUST)

1. `team-design/development/team.md`
2. `team-design/development/engineer/checklist.md`
3. `team-design/development/engineer/machine.md` 및 `operations/machine/workflow-engineer.md`
