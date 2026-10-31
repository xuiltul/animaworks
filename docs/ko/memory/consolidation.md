<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/consolidation.md -->
<!-- i18n: source-sha256=a05463694132540a497b99625f9dc933a6b34890c941860e8c6bc5269d5e45a0 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 193a5e72

# 기억의 통합과 망각

기억의 통합은 Anima의 판단과 프레임워크가 수행하는 후보 수집·색인 유지로 구성된다. 시각은 설정 가능하며, 기본 일일 통합은 02:00, 주간 통합은 일요일 03:00이다. 주간 통합은 `weekly_enabled`이 유효한 경우에 실행된다.

## 일일 처리

`server/supervisor/_mgr_scheduler.py`이 일일 job을 등록하고, `core/lifecycle/system_consolidation.py`의 `_handle_daily_consolidation`가 대상 Anima를 선택한다. 비활동 기간이나 최근 24시간의 기록 수에 따라 실행을 건너뛸 수 있다.

Anima의 `run_consolidation`는 전날의 활동 로그를 날짜가 포함된 에피소드로 요약하고, 요약에서 atomic facts를 추출한다. 이후 framework post-processing은 다음 순서로 진행된다.

1. `ForgettingEngine.synaptic_downscaling`가 저활성 후보를 인덱스에서 표시한다.
2. `knowledge_self_correction`가 대상이 되는 지식·절차를 LLM으로 검토하고, 해결된 과제에서 절차를 작성한다.

이 보정은 `core/lifecycle/knowledge_correction.py`이 상한을 관리하면서 실행한다. 자세한 재고정 조건은 후술한다.

## 주간 처리

주간 Anima cycle은 `core/anima/lifecycle.py`의 `_run_weekly_consolidation`이 담당한다. `ConsolidationEngine`가 merge, fact 간의 conflict, 망각 후보를 준비하고, `hygiene.py`이 지식 파일의 형식이나 크기상의 확인 후보를 리포트로 만든다. 이것들은 주간 프롬프트에 포함되며, Anima가 내용을 판단하여 기억 파일을 업데이트·보관한다. hygiene scan 자체는 기억 파일을 이동·삭제하지 않는다.

주간 처리 후에는 `ProceduralDistiller.weekly_pattern_distill`가 활동 로그 중의 반복 패턴을 절차 후보로 증류한다. 또한 `skill_autolearn_enabled`가 유효하면, 적격한 절차에서 저위험 probation skill을 작성한다.

## 망각 후보와 임계값

`core/memory/maintenance/forgetting.py`의 일일 downscaling은 일반 지식·에피소드에 대해 최종 사용 후 90일을 초과하고 사용 횟수가 3회 미만인 경우 저활성으로 표시한다. 절차는 별도 기준으로, 180일을 초과하여 미사용이고 사용 합계가 3회 미만인 경우, 또는 실패가 3회 이상이고 효용이 0.3 미만인 경우 저활성으로 표시한다. 보호된 기억이나 재사용 중인 기억은 대상에서 제외된다.

주간 망각 후보에는 저활성 상태가 90일을 초과하고 사용 횟수가 2회 이하인 기억이 포함된다. 이것은 삭제 명령이 아니라, Anima가 본문이나 주변 정보를 보고 보관할지를 결정하기 위한 후보이다. 후보 제시의 대상이나 보호 규칙의 구현은 `ForgettingEngine`이 담당한다.

## 절차의 사용 추적과 재고정

절차를 사용한 후에는 `report_procedure_outcome`으로 성공 또는 실패를 기록한다. 결과는 frontmatter의 `success_count`, `failure_count`와 `confidence`의 업데이트에 사용된다.

`core/memory/maintenance/reconsolidation.py`는 절차의 `failure_count >= 1` **또는** `confidence < 0.6`을 충족하는 경우 재고정 후보로 한다. LLM이 개정한 경우에는 구버전을 보관하고, `version`을 진행하며 성공·실패 카운터와 신뢰도를 초기값으로 되돌린다.

## RAG 인덱스 정기 업데이트

RAG 인덱스 업데이트는 주간 통합과는 별개의 단일 스케줄이다. `server/supervisor/_mgr_scheduler.py`의 일일 indexing job이 기본 04:00에 메모리 파일, facts, 공유 데이터를 반영하고, BM25와 entity 인덱스도 업데이트한다. 주간 post-processing이 별도로 RAG 전체를 재구축하는 경로는 없다. 인덱스와 복구 경로는 [의도적 회상과 검색](retrieval.md)을 참조한다.
