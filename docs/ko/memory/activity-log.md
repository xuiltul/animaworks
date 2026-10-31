<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/activity-log.md -->
<!-- i18n: source-sha256=79f543a7c8e5b8a683d0ba6b2184927f6350431ed966bd2ffeea5a2991768cf6 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 193a5e72

# 활동 로그

`core/memory/activity/`은 Anima의 메시지, 실행, 메모리 작업 등을 시간 순서대로 기록한다. 공개 API는 `core/memory/activity/logger.py`의 `ActivityLogger`이며, 기록은 `{anima_dir}/activity_log/{date}.jsonl`에 추가된다.

## 항목

각 JSONL 줄은 하나의 이벤트이다. 기본 항목은 `ts`(시간), `type`(이벤트 유형), `content`, `summary`, `from`, `to`, `channel`, `tool`, `meta`, `origin`, `origin_chain`, `ctx`이다. 빈 값은 저장 시 생략된다. 내부에서는 `from_person`와 `to_person`을 사용하며, JSON에서는 각각 `from`와 `to`로 출력한다.

이벤트 유형에는 닫힌 열거형이 없으며, 호출 측이 용도에 따라 기록한다. 현재 대표적인 예는 다음과 같다.

| 분류 | 이벤트 예 |
|---|---|
| 대화·통신 | `message_received`, `response_sent`, `message_sent`, `channel_post`, `channel_read`, `human_notify`, `human_reply` |
| 실행·상태 | `tool_use`, `tool_result`, `error`, `heartbeat_start`, `heartbeat_end`, `heartbeat_reflection`, `cron_executed`, `inbox_processing_start`, `inbox_processing_end` |
| 메모리·통합 | `memory_write`, `issue_resolved`, `knowledge_outcome`, `knowledge_reconsolidated`, `procedure_reconsolidated`, `consolidation_start`, `consolidation_end` |
| 작업·스킬 | `task_created`, `task_updated`, `task_exec_start`, `task_exec_end`, `skill_auto_created`, `skill_autolearn_summary` |

오래된 `dm_sent`와 `dm_received`은 읽을 때 각각 `message_sent`, `message_received`로 처리된다.

## 스트리밍 저널

`core/memory/conversation/streaming_journal.py`의 `StreamingJournal`은 LLM 응답 스트리밍 중에 텍스트 조각과 도구의 시작·종료를 `shortterm/streaming_journal_{session_type}.jsonl`에 순차적으로 기록한다. 정상 종료 시에는 `done`을 기록하여 저널을 닫는다. 프로세스가 비정상 종료된 경우, 다음 시작 시 남은 기록에서 응답 텍스트나 도구 실행 상황을 복구할 수 있다. 쓰기는 500자 도달 시, 또는 최종 flush 후 1초 경과 시에 수행한다.

## 보존과 로테이션

`ActivityLogger`은 검색·타임라인 표시 등의 읽기도 제공한다. 로그의 보존량, 기간, 로테이션 시간은 `activity_log` 설정으로 관리하며, 실행은 supervisor가 담당한다. 설정 항목의 목록은 [설정 참조](../reference/config.md)를 참조한다.
