# 정본 태스크 아키텍처

## 하나의 영속 정본

LLM 태스크의 정본은 호스트가 관리하는 TaskStore입니다. 태스크 ID, 완전한 실행 입력, 의존 관계, 결과, 실행 시도, 위임 별칭과 영속 기동 알림을 필요한 단위로 함께 확정합니다. 별도의 파일 실행 큐와 상사 원장을 대조하는 구조가 아닙니다.

확인은 `list_tasks` / `task_tracker`, 변경은 `submit_tasks` / `delegate_task` / `update_task`를 사용하세요. DB나 태스크 파일을 직접 수정하지 마세요. `backlog_task`는 추적용 작업을 등록하며 실행 권한은 획득하지 않습니다.

## 실행 계약

1. 새 `submit_tasks`는 태스크와 완전한 입력을 원자적으로 공개합니다. 원래 지시, 제약, workspace, 완료 기준을 보존하세요.
2. 호스트가 의존 관계를 확인하고 고유한 시도 토큰으로 실행 권한을 획득합니다. `in_progress`는 호스트만 설정합니다.
3. 에이전트는 `update_task`로 `done`, `pending`, `cancelled`를 선언합니다. 오래된 시도는 새 시도의 완료나 수락된 결과를 덮어쓸 수 없습니다.
4. 의존 태스크 완료와 영속 기동은 호스트가 처리하며 정기 Heartbeat에 의존하지 않습니다. 취소, 오류, 중단에는 증거와 주의 사유를 남깁니다.
5. 중단 태스크를 무조건 재시도하지 않습니다. 이전 효과를 확인하고 원인을 해결한 뒤 `submit_tasks(..., tasks=[{"task_id": "ID", "resume": true}])`로 같은 미종료 태스크를 명시적으로 재개하세요. 입력과 이력을 보존하며 resume 없는 재전달은 멱등 처리됩니다.

상사의 위임 보기는 부하의 정본 태스크에 대한 별칭입니다. 별도의 가변 원장이나 Heartbeat 동기화 없이 양쪽 보기에 최신 상태가 반영됩니다. 의존 태스크의 종료가 성공을 뜻하지는 않습니다. 취소를 done으로 취급하여 후속 작업을 실행하지 마세요.

## 작업 맥락과 증거

`state/current_state.md`는 간결한 작업 맥락이며 태스크의 정본이 아닙니다. 관찰, 계획과 차단 요소를 기록하고 일반 세션 경계에서 유지됩니다. 영구 지식과 절차는 전용 기억 영역에 보관하세요.

TaskExec 결과 요약은 `state/task_results/{task_id}/{attempt_token}.md`에 저장하고 TaskStore가 수락한 결과를 선택합니다. 파일명이나 오래된 요약만으로 완료를 증명할 수 없습니다. 활동 로그와 원래 지시를 증거로 보존하세요.

## 기존 저장소와 명령 태스크

기존 `state/task_queue.jsonl`과 `state/pending/`은 마이그레이션·내보내기 증거로만 보존합니다. 마이그레이션은 기존 쓰기 작업을 멈추고 백업한 뒤 운영자가 명시적으로 수행합니다. 임의의 읽기로 실행 중인 기존 데이터를 가져오지 않습니다.

장시간 명령 도구는 별개입니다. `animaworks-tool submit`은 계속 `state/background_tasks/pending/`을 사용하며 BackgroundTaskManager가 명령 상태와 알림을 저장합니다. LLM 태스크 계약을 적용한다고 이 파일 경로를 제거하지 마세요.

도구 예제는 `reference/operations/task-management.md`, 명령 실행은 `operations/background-tasks.md`를 참조하세요.
