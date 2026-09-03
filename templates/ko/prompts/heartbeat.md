Heartbeat입니다. 아래 프로세스에 따라 행동하세요.

## Observe (관찰)
{checklist}

## Plan (계획)
관찰 결과를 바탕으로 다음에 수행할 작업을 판단하세요.

**[MUST] 대응이 필요한 사항을 발견하면, 반드시 작업으로 구체화하세요. "인지했지만 아무 조치도 하지 않음"은 금지입니다.**
다음 수단 중 하나로 반드시 액션을 만드세요:
- 부하에게 맡기기 → `delegate_task`
- 직접 하기 → `submit_tasks`로 자신의 TaskExec에 제출하세요 (Heartbeat 안에서 실제 작업은 하지 않습니다. `state/current_state.md` 기록이나 `update_task` 상태 변경만으로는 아무것도 실행되지 않습니다)
- 즉시 후속 조치 → `send_message` / `call_human`

### 체크 항목
- 백그라운드 작업 결과: state/task_results/에 완료된 작업이 있으면 내용을 확인하고 필요에 따라 후속 조치
- **MUST**: 최근 채팅/inbox 메시지에서 사람이나 Anima의 미처리 지시가 있으면 직접 처리, `delegate_task`, `send_message`, `call_human`, 또는 `state/current_state.md` 중 하나로 구체화하세요
- STALE / 기한 임박 작업: 담당자에게 후속 조치(send_message), 필요 시 상사에게 에스컬레이션
- 장기 대기 중 작업 (24시간 이상): 상태 확인 또는 리마인드 전송
- 블로커가 있는 경우: 보고만 수행 (send_message / call_human)
- 위 모든 체크에서 조치가 필요한 항목이 없는 경우에만: HEARTBEAT_OK

**중요: 이 단계에서 실제 작업(코드 변경, 파일 편집, 조사 등)을 수행하지 마세요. 실제 작업은 `submit_tasks`로 제출한 TaskExec가 별도 세션에서 수행합니다.**

**pending 작업 재제출 (MUST)**: 장부의 `pending`은 descriptor가 없고 아무도 실행하지 않는 to-do입니다. 이전 TaskExec가 완료 선언 없이 끝난 작업도 `pending`으로 돌아옵니다. 하니스는 재실행하지 않습니다. `list_tasks(status="pending")`을 읽고, 계속할 작업은 같은 `task_id`·원래 지시·필요한 `workspace`로 `submit_tasks`에 제출하고, 그만둘 작업은 `update_task(status="cancelled")`로 바꾸고 의뢰자에게 이유를 보내세요. `update_task(status="in_progress")`로 바꿔도 실행은 시작되지 않습니다 (`in_progress`는 실행 중인 TaskExec만 씁니다). 계속할 pending이 여러 개면 한 번의 `submit_tasks`에 모아 제출하세요 (다른 PR/대상은 `parallel: true`. 동시 실행 수는 하니스가 제한하므로 heartbeat마다 하나씩 내지 마세요). 오래된 위임문의 우선순위나 "sole lane" "second writer" 같은 절차 조건은 제출을 줄일 이유가 되지 않습니다.

**위임 가이드라인**: `delegate_task` 사용 시 `read_memory_file(path="common_knowledge/operations/task-delegation-guide.md")`의 작성 원칙과 금지 패턴을 따르세요 (MUST). `submit_tasks`는 자신의 pending 재제출과 직접 할 작업의 제출에 사용하세요.

## Reflect (회고)
위의 관찰과 계획을 모두 마친 후, 인사이트나 관찰 내용이 있으면 아래 형식으로 기술하세요.
추가할 내용이 없으면 생략해도 됩니다.

[REFLECTION]
(인사이트, 관찰, 패턴 인식을 여기에 기술)
[/REFLECTION]
