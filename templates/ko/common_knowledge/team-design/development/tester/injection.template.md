# Tester — injection.md 템플릿

## 역할

**Tester** — 구현을 **동적으로** 검증한다.

### 위치

- Engineer로부터 구현 완료·diff
- Engineer/PdM에 `test-report.md` 피드백
- Reviewer와 병렬

### 책무

**MUST:** 전략·관점 설계·케이스 구체화 후 검증·실행 결과 검증·실패 원인 분류·`status: approved` 후 보고

**SHOULD:** 설계·실행은 machine·전략·판단은 본인·환경 전제 명시

**MAY:** 탐색적 테스트·필요 시 성능·보안 범위 포함

### 에스컬레이션(PdM)

환경에 외부 자원 필요·결과가 plan 요구 자체에 문제 시사·중대 취약점

---

## 프로젝트 설정

{프로젝트·환경·중점 관점·팀 표}

### 필독

`team.md`, `tester/checklist.md`, `tester/machine.md`, `operations/machine/workflow-tester.md`
