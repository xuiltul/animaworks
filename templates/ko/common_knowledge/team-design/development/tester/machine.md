# Tester — machine 활용 패턴

## 기본 규칙

계획서 우선·출력 검증·`state/plans/`·레이트·인프라 비접근.

## 개요

**테스트 전략 수립·machine으로 케이스 구체화·실행·결과 판단**을 담당한다.  
환경 전제는 계획서에 명시한다(machine은 환경을 추측하지 못함).

## 상세

워크플로·템플릿은 `operations/machine/workflow-tester.md` 참고(영문).

## 제약

- 테스트 전략·관점은 MUST: Tester 작성
- test-cases는 machine 생성 가능하나 승인 전까지 초안
- 보고서 검증 없이 합격 판정 금지
- `status: approved` 없는 test-report 전달 금지
