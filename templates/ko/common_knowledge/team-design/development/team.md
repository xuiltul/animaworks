# 개발 풀 팀 — 팀 개요

## 4역할 구성

| 역할 | 책무 | 권장 `--role` | `speciality` 예 | 상세 |
|------|------|---------------|-------------------|------|
| **PdM** | 조사·계획·판단 | `manager` | `pdm` | `development/pdm/` |
| **Engineer** | 구현·구현 검증 | `engineer` | `backend`, `fullstack` | `development/engineer/` |
| **Reviewer** | 코드 리뷰(정적 검증) | `engineer` | `code-review` | `development/reviewer/` |
| **Tester** | 테스트 설계·실행(동적 검증) | `engineer` | `testing`, `qa` | `development/tester/` |

한 Anima에 몰아넣으면 컨텍스트 비대화·셀프 리뷰 사각·직렬 병목이 생길 수 있다.

각 역할 디렉터리에 `injection.template.md`(injection.md 뼈대), `machine.md`(machine 활용 패턴), `checklist.md`(품질 체크리스트)가 있다.

> 기본 원칙 상세: `team-design/guide.md`

## 핸드오프 체인

```
PdM → investigation.md/plan.md (approved) → delegate_task
  → Engineer → impl.plan.md → 구현 → 구현 검증
    → Reviewer (정적 검증) ─┐
    → Tester  (동적 검증) ─┤ ← 병렬 가능
      └─ 수정 필요 → Engineer로 복귀
      └─ 둘 다 APPROVE → PdM → call_human → 사람이 머지 판단
```

### 인계 문서

| 출발 → 도착 | 문서 | 조건 |
|-------------|------|------|
| PdM → Engineer | `plan.md` | `status: approved` |
| Engineer → Reviewer/Tester | 구현 diff + `plan.md` | 구현 검증 완료 후 |
| Reviewer → Engineer | `review.md` | `status: approved` |
| Tester → Engineer/PdM | `test-report.md` | `status: approved` |

### 운용 규칙

- **Worktree**: Engineer가 plan.md 수신 후 생성(`{task-id}/{요약}`). machine은 `-d /path/to/worktree`로 실행. 완료 후 머지·삭제는 Engineer가 담당
- **수정 사이클**: Critical → 전체 재리뷰 / Warning → diff만 확인 / 3왕복 해소 안 되면 PdM에 에스컬레이션
- **machine 실패**: `current_state.md`에 기록 → 다음 heartbeat에서 재평가

## 스케일링

| 규모 | 구성 | 비고 |
|------|------|------|
| 소규모 | PdM + Engineer(Reviewer 겸임) | 셀프 리뷰 리스크 허용 |
| 중규모 | 본 템플릿대로 4명 | 표준 구성 |
| 대규모 | PdM 1 + Engineer 복수 + Reviewer/Tester 각 1~2 | PdM이 `delegate_task`로 모듈 단위 위임 |
