# Engineer — machine 활용 패턴

## 기본 규칙

PdM·Reviewer·Tester와 동일(계획서 우선·출력 검증·`state/plans/`·레이트·인프라 비접근).

---

## 개요

Engineer는 **plan.md를 받아 구체화·구현을 machine에 맡기고 검증 체크포인트 2곳을 담당**한다.

- impl.plan.md 구체화 → machine, Engineer가 검증
- 코드 구현 → machine, Engineer가 검증

상세 단계·명령 템플릿은 `common_knowledge/operations/machine/workflow-engineer.md`를 참고한다(영문).

---

## 제약

- `status: approved` 없는 plan.md로 작업 시작 금지
- impl.plan.md 승인 없이 구현 machine 실행 금지
- 검증 없이 커밋·푸시 금지
