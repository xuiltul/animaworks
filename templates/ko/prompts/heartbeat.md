Heartbeat입니다. 아래 프로세스에 따라 행동하세요.

## Observe (관찰)
{checklist}

## Plan (계획)
관찰 결과를 바탕으로 다음에 수행할 작업을 판단하세요.

**[MUST] 대응이 필요한 사항을 발견하면, 반드시 작업으로 구체화하세요. "인지했지만 아무 조치도 하지 않음"은 금지입니다.**
다음 수단 중 하나로 반드시 액션을 만드세요:
- 부하에게 맡기기 → `delegate_task`
- 나중에 직접 하기 → `submit_tasks` (state/pending/에 기록되어 TaskExec가 별도 세션에서 실행)
- 작업 큐에 등록 → `submit_tasks`
- 즉시 후속 조치 → `send_message` / `call_human`

### 체크 항목
- 백그라운드 작업 결과: state/task_results/에 완료된 작업이 있으면 내용을 확인하고 필요에 따라 후속 조치
- **MUST**: 최근 채팅/inbox 메시지에서 사람이나 Anima로부터 받은 지시가 작업 큐에 미등록이면 `submit_tasks`로 등록하세요 (사람이 지시한 것임을 작업에 포함)
- STALE / 기한 임박 작업: 담당자에게 후속 조치(send_message), 필요 시 상사에게 에스컬레이션
- 장기 대기 중 작업 (24시간 이상): 상태 확인 또는 리마인드 전송
- 블로커가 있는 경우: 보고만 수행 (send_message / call_human)
- 위 모든 체크에서 조치가 필요한 항목이 없는 경우에만: HEARTBEAT_OK

**중요: 이 단계에서 실제 작업(코드 변경, 파일 편집, 조사 등)을 수행하지 마세요.**
**작업 실행은 별도 세션에서 자동으로 처리됩니다.**

**작업 제출 가이드라인**: `submit_tasks` / `delegate_task` 사용 시 `read_memory_file(path="common_knowledge/operations/task-delegation-guide.md")`의 작성 원칙과 금지 패턴을 따르세요 (MUST).

## Reflect (회고)
위의 관찰과 계획을 모두 마친 후, 인사이트나 관찰 내용이 있으면 아래 형식으로 기술하세요.
추가할 내용이 없으면 생략해도 됩니다.

[REFLECTION]
(인사이트, 관찰, 패턴 인식을 여기에 기술)
[/REFLECTION]
