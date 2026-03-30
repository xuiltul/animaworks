# Reviewer — machine 활용 패턴

## 기본 규칙

계획서 우선·출력 검증·`state/plans/`·레이트·인프라 비접근.

## 개요

**machine에 리뷰를 맡기고, 결과의 타당성을 메타 리뷰**한다.  
diff·Issue·plan 요구는 Anima가 수집해 계획서에 넣는다(GitHub API는 machine 불가).

## 상세

워크플로·템플릿은 `operations/machine/workflow-reviewer.md` 참고(영문).

## 제약

- 리뷰 계획(무엇을 볼지)은 MUST: Reviewer 작성
- machine 출력을 검증 없이 Engineer에게 넘기면 안 됨
- `status: approved` 없는 review.md 전달 금지
