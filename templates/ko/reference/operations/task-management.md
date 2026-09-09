# 태스크 관리 방법

## 하나의 정본

태스크는 `list_tasks(detail=true)` 또는 `animaworks-tool task list`로 확인합니다. 호스트가 관리하는 TaskStore에 지시, 의존 관계, 실행 시도, 결과, 위임 별칭을 함께 영속화합니다. 데이터베이스를 직접 수정하거나 실행 권한을 만들어 내거나 파일을 작성해 큐를 복구하지 마세요. 기존 `state/task_queue.jsonl`과 `state/pending/`은 마이그레이션·내보내기 증거이며 실행 중인 입력 경로가 아닙니다. 운영자의 마이그레이션을 위해 보존하세요.

일반 대화에서 처리 가능한 요청은 직접 대응해도 됩니다. 백그라운드 실행, 병렬 작업, 지속 추적이 필요할 때만 등록하세요. 사람의 요청을 최우선으로, 같은 우선순위에서는 상사의 요청을 동료보다 우선합니다. 인계할 때 원래 지시, 완료 기준, 제약과 필요한 맥락을 보존하세요.

## 실행 경로 선택

- Inbox는 메시지와 가벼운 답변을 처리합니다.
- Heartbeat는 의미 있는 변화를 확인하고 필요한 조치를 판단합니다. 긴 코딩이나 대량 도구 작업은 수행하지 말고, 자신의 TaskExec에는 `submit_tasks`, 직속 부하에게는 `delegate_task`를 사용하세요.
- TaskExec는 영속화된 태스크를 도구와 함께 실행합니다. 호스트가 실행 권한, 동시 실행, 의존 관계, 취소, 시도 복구를 관리하며 정기 Heartbeat에 의존하지 않습니다.
- Agent/Task 하위 에이전트 생성 도구는 비활성화되어 있습니다. 위의 제출·위임 도구를 사용하세요.

## 제출과 확인

`submit_tasks`는 부하가 아닌 **자신의 TaskExec**에서 실행됩니다.

```
submit_tasks(batch_id="report-build", tasks=[
  {"task_id": "collect", "title": "근거 수집", "description": "요청한 근거를 출처와 함께 수집한다.", "parallel": true},
  {"task_id": "report", "title": "보고서 작성", "description": "수집한 근거로 요청한 보고서를 작성한다.", "depends_on": ["collect"]}
])
list_tasks(detail=true)
```

새 태스크에는 `task_id`, `title`, `description`이 필요합니다. 선택 항목은 `context`, `acceptance_criteria`, `constraints`, `file_paths`, `workspace`, `parallel`, `depends_on`, `reply_to`, `model`입니다. `workspace`는 등록된 작업공간 별칭이며 모델 선택은 보통 런타임 설정을 따릅니다. 긴 원본 지시를 짧은 요약으로 대체하지 마세요.

제출 시 배치를 검증하고 태스크와 실행 입력을 원자적으로 저장합니다. 같은 제출의 재전달은 멱등 처리되며 재시도가 아닙니다. 선행 태스크가 정상 완료되기 전에는 후속 작업을 실행하지 않습니다. 파일이 없거나 상태가 `pending`이라는 이유만으로 실행 가능하다고 판단하지 마세요.

## 결과 선언과 명시적 재개

```
update_task(task_id="TASK_ID", status="done", summary="검증한 결과", result="근거와 산출물 위치")
update_task(task_id="TASK_ID", status="pending", summary="지정한 입력을 기다림")
update_task(task_id="TASK_ID", status="cancelled", summary="더 이상 필요하지 않은 이유")
```

`in_progress`는 호스트가 실행 권한을 획득할 때 설정하는 읽기 전용 상태입니다. `update_task`로 설정하지 마세요. 완료 선언 없이 시도가 끝나면 주의 사유가 있는 pending 상태가 될 수 있습니다. pending은 자동 재시도를 뜻하지 않습니다. 재개하려고 새 ID로 복제하지 마세요.

중단 원인을 해결한 뒤 같은 미종료 태스크를 명시적으로 재개합니다.

```
submit_tasks(batch_id="resume-report", tasks=[{"task_id": "TASK_ID", "resume": true}])
```

저장된 입력을 재사용하고 이력을 보존합니다. 실행 중인 시도나 완료·취소된 태스크는 이 방식으로 재개할 수 없습니다. 의존 태스크가 취소되거나 주의가 필요하면 세부 정보를 확인하고 요청자에게 묻거나 불필요해진 작업을 취소하세요. 성공을 꾸며내지 마세요.

진행할 수 없으면 사실, 시도한 내용, 부족한 권한·정보, 다음 조치를 요청자에게 알리고 같은 실패를 반복하지 마세요. 관련 지식은 필요할 때 검색하며 의무적인 의식으로 만들지 마세요. 기다리는 동안 다른 허가된 작업을 수행해도 됩니다. 위임받은 작업의 완료는 요청자에게 보고하되 중복 통지와 불필요한 확인 답장은 피하세요.

## 부하에게 위임

```
delegate_task(name="dave", instruction="API 테스트를 실행하고 검증한 결과를 보고한다", summary="API 테스트")
task_tracker()
```

부하 소유의 태스크 하나와 상사가 볼 수 있는 별칭을 만듭니다. 양쪽 보기에 같은 최신 상태가 즉시 반영되므로 별도 원장이나 Heartbeat 동기화가 필요 없습니다. `task_tracker(status="all")`은 종료된 작업도 포함하며 `status="completed"`는 done/cancelled를 표시합니다. 지시가 불분명하면 위임자에게 확인하고 완료 시 결과를 보고하세요.

## 작업 맥락과 결과

`state/current_state.md`에는 관찰, 맥락, 계획과 차단 요소를 간결하게 기록합니다. 태스크 목록을 복제하거나 영구 절차를 저장하지 마세요. 활성 맥락이 없으면 `status: idle`을 사용합니다. 일반 세션 경계에서 보존되며 프롬프트 표시와 디스크 정리 한도는 별도입니다(`anatomy/working-memory.md`).

TaskExec 결과 요약은 `state/task_results/{task_id}/{attempt_token}.md`에 저장됩니다. 호스트가 수락한 시도의 결과를 후속 태스크에 제공하며 임의의 오래된 파일을 사용하지 않습니다. 결과 파일을 조작하거나 파일 존재만으로 완료를 판단하지 마세요. 활동 로그와 에피소드는 증거로 유지되며 모든 상태 전이를 수동으로 중복 기록할 의무는 없습니다.

## 장시간 명령 도구는 별도 경로

이미지 생성이나 run_command 등 지원되는 장시간 외부 도구는 `animaworks-tool submit TOOL ...`로 실행합니다. 이는 `submit_tasks`와 별개로, 명령 기술자는 계속 `state/background_tasks/pending/`에 저장됩니다. BackgroundTaskManager는 `state/background_tasks/{task_id}.json`에 `running`, `completed`, `failed` 상태를 기록합니다. `list_background_tasks` / `check_background_task`로 확인하세요. 이 파일 기반 명령 경로와 알림은 유지됩니다. 자세한 내용은 `operations/background-tasks.md`를 참조하세요.
