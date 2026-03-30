# machine 도구 이용 가이드 — 공통 원칙

## 개요

`animaworks-tool machine run`은 외부 에이전트(cursor-agent / claude / codex / gemini)에게
코드 변경·조사·분석·리뷰·테스트 등 무거운 작업을 위임하는 도구이다.

machine은 AnimaWorks 인프라에 접근할 수 없는 독립 실행 환경에서 동작한다.
기억·메시지·조직 정보를 쓸 수 없으므로 필요한 정보는 모두 계획서에 담는다.

## 메타 패턴 — 전 로ール 공통

모든 단계·모든 로ール에서 아래 4단계를 지킨다:

```
① Anima가 지시서(계획서)를 작성한다
② machine에 계획서를 넘겨 실행시킨다
③ machine 출력은 초안으로 취급한다
④ Anima가 출력을 검증하고 승인 또는 수정한다
```

**원칙:**
- machine에 무엇을 시킬지에 대한 최초 지시서는 **Anima가 직접 작성**(MUST)
- 중간 산출물 구체화(plan → impl.plan 등)는 machine에 맡겨도 됨
- machine 출력은 항상 초안 — **Anima 승인 없이 다음 단계로 넘기지 않음**(NEVER)
- 문제가 있으면 계획서를 고쳐 machine에 재위임하거나 Anima가 직접 수정

## 상태 관리

계획서·산출물의 상태는 문서 자체에 메타데이터로 둔다.
프레임워크가 상태를 자동 관리하지 않으며 — Anima가 전이를 스스로 판단한다.

각 문서 상단에 다음 메타데이터 블록을 둔다:

```markdown
status: draft | reviewed | approved
author: {anima 이름}
date: {YYYY-MM-DD}
type: investigation | plan | impl-plan | review | test-plan | test-report
```

전이 기준:
- `draft` → `reviewed`: Anima가 읽고 내용을 확인함
- `reviewed` → `approved`: Anima가 품질을 승인하고 다음 단계로 넘겨도 된다고 판단함
- 승인은 상위 역할(상사)이 할 수도 있음 — 조직 운용 규칙을 따른다

## 계획서 저장 위치

**모든 계획서는 `state/plans/`에 저장한다.** `/tmp/` 저장은 금지(재시작 시 소실).

파일 이름 규칙: `{YYYY-MM-DD}_{태스크 요약}.{type}.md`

| type | 용도 | 예 |
|------|------|-----|
| `investigation` | 조사 보고서 | `2026-03-27_login-bug.investigation.md` |
| `plan` | 구현 계획서 | `2026-03-27_fix-email-validation.plan.md` |
| `impl-plan` | 구현 상세 계획서 | `2026-03-27_fix-email-validation.impl-plan.md` |
| `review` | 리뷰 보고서 | `2026-03-27_review-PR-42.review.md` |
| `test-plan` | 테스트 계획서 | `2026-03-27_e2e-login-flow.test-plan.md` |
| `test-report` | 테스트 결과 보고서 | `2026-03-27_e2e-login-flow.test-report.md` |

`state/plans/`를 쓰는 이유:
- **영속성**: OS 재시작 후에도 사라지지 않음
- **추적 가능**: 나중에 “무엇을 지시했는지” 검증 가능
- **상사 열람**: supervisor가 `read_memory_file`로 부하의 계획서를 확인 가능

```
read_memory_file(path="../{부하 이름}/state/plans/2026-03-27_fix-email-validation.plan.md")
```

## machine 실행 명령

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{계획서 파일명})" \
  -d /path/to/worktree
```

엔진을 명시하는 경우:

```bash
animaworks-tool machine run -e cursor-agent \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{계획서 파일명})" \
  -d /path/to/worktree
```

백그라운드 실행(무거운 작업용):

```bash
animaworks-tool machine run --background \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{계획서 파일명})" \
  -d /path/to/worktree
```

## 레이트 제한

- 세션(chat): 세션당 5회
- Heartbeat: heartbeat당 2회
- `--background` 결과는 `state/background_notifications/`에서 확인

## 금지 사항

- **계획서 없이 `machine run` 실행 금지**(짧은 인라인 지시만으로는 실행 불가)
- 계획서에 “목표”와 “완료 조건”이 모두 없는 상태에서의 실행 금지
- **machine 출력을 검증하지 않고 다음 단계로 넘기는 것 금지**(NEVER)
- machine 출력을 검증하지 않고 커밋·푸시 금지

## machine 제약

- machine은 AnimaWorks 인프라에 접근할 수 없음(기억·메시지·조직 정보 없음)
- GitHub API 조작(diff 조회·코멘트 등)은 Anima 측에서 수행하고 결과를 계획서에 포함
- 장시간 작업은 `--background` 사용

## 팀 설계 템플릿을 쓰는 경우

`injection.md`에 `team-design/development/{role}/machine.md`가 지정되어 있으면
**본 파일보다 그쪽이 우선한다**. 각 역할의 machine.md는 기본 규칙·프롬프트 작성법을 포함해 자급자족한다.

본 파일은 팀 설계 템플릿을 쓰지 않는 단독 Anima용 공통 가이다.

## 로별 워크플로 가이드

자신의 역할에 해당하는 가이드를 참고한다:

| 역할 | 가이드 | 개요 |
|------|--------|------|
| PdM | `operations/machine/workflow-pdm.md` | 조사 → 계획서 작성 |
| Engineer | `operations/machine/workflow-engineer.md` | 구체화 → 구현 |
| Reviewer | `operations/machine/workflow-reviewer.md` | 코드 리뷰 → 메타 리뷰 |
| Tester | `operations/machine/workflow-tester.md` | 테스트 설계 → 실행 → 결과 검증 |

역할 할당은 `injection.md` 또는 `specialty_prompt.md`에서 확인한다.
복수 역할을 겸하면 해당하는 모든 가이드를 참고한다.
