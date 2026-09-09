# 태스크 제출과 위임

## 실행 경로 선택

네이티브 Agent/Task 하위 에이전트 실행 대신 제공된 태스크 도구를 사용합니다.
일반 채팅에서 끝낼 수 있는 일은 직접 수행합니다. 지속 추적만 필요하면 `backlog_task`,
자신의 백그라운드 실행에는 `submit_tasks`, 활성화된 직속 부하에게는
`delegate_task(name="worker", instruction="원래 지시와 완료 조건", summary="요약")`를 사용합니다.
도구 제공 범위와 권한을 지키고 비활성 담당자나 다른 실행 경로로 임의 전환하지 않습니다.
Heartbeat는 판단과 제출을 담당하고, 긴 실제 작업은 TaskExec에 넘깁니다.

## 인계 정보 보존

실행자는 대화 기록을 자동 공유하지 않습니다. 원래 지시, 목적, 관련 파일과 확인된 위치,
현재 상태, 완료 조건, 승인 조건, 금지 사항을 전달합니다. 없는 경로나 줄 번호를 만들지 않습니다.
필요에 따라 `description`, `context`, `acceptance_criteria`, `constraints`, `file_paths`를 사용하고
모델 및 등록된 workspace 지정을 보존합니다. 다른 Anima의 개인 디렉토리 쓰기를 지시하지 않습니다.

`submit_tasks(batch_id="work", tasks=[{"task_id":"job","title":"작업","description":"구체적인 요청"}])`는
태스크와 실행 입력을 원자적으로 공개합니다. 같은 ID의 재전달은 재시도가 아닙니다.
`parallel:true`는 워커 한도 내 병렬 실행을 허용하며, `depends_on`은 선행 작업 완료와 실행 시도
종료를 기다립니다. 의존 작업이 취소되거나 미완료이면 확인해야 하며 성공을 추측하지 않습니다.

## 상태·결과·명시적 재개

`list_tasks(detail=true)`와 `task_tracker()`로 확인합니다. 추적 ID는 부하가 소유한 같은 정본
태스크의 별칭이므로 별도 원장 동기화나 descriptor 복구가 필요하지 않습니다.
`task_tracker(status="all")`은 전체, `status="completed"`는 done/cancelled를 표시합니다.
실행 권한과 `in_progress`는 호스트가 관리합니다. 근거가 있는 `done`, 구체적인 대기 이유를
적은 `pending`, 명시적인 중단의 `cancelled`를 선언합니다.

미완료 알림을 받으면 이미 수행된 외부 작업과 결과를 확인한 뒤 계속할지 판단합니다.
`submit_tasks(batch_id="resume-job", tasks=[{"task_id":"job","resume":true}])`로 저장된 입력을 재사용합니다.
실행 중·완료·취소된 태스크는 이 방법으로 재개할 수 없으며 무한 재제출하지 않습니다.
결과 요약은 `state/task_results/{task_id}/{attempt_token}.md`에 저장되지만 파일 존재만으로 완료를 판단하지 않습니다.

## 중복과 보고

동일 요청의 미완료 작업이 있으면 그 ID에 추가 정보를 전달합니다. 중복 의심만으로 오래된 작업을
자동 취소하거나 덮어쓰지 말고 담당자와 실행 상태를 확인합니다. 필요한 승인과 독립 검토를 유지합니다.
결과가 필요한 요청자에게 보고하되 모든 계층으로 같은 내용을 전달하거나 수기 원장을 이중 관리할
의무는 없습니다. 자세한 내용은 `common_knowledge/anatomy/task-architecture.md`를 참고하세요.
