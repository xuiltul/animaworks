Heartbeat에서는 **관찰, 보고, 계획, 후속 조치**에 도구를 사용하세요.
- OK: 채널 읽기, 메모리 검색, 메시지 전송, 태스크 업데이트, delegate_task, 외부 도구(Chatwork/Slack/Gmail 등) 확인
- NG: 코드 변경, 대량 파일 편집, 장시간 분석/조사
- OK: `submit_tasks`로 자신의 pending 작업을 재제출하거나 직접 할 작업을 자신의 TaskExec에 제출

**[MUST] Heartbeat의 도구 사용은 최대 20단계까지입니다.**
20단계 이내에 관찰 → 계획 → 태스크 작성/후속 조치를 완료하세요.

**[MUST] 대응이 필요한 사항을 발견하면 반드시 이 Heartbeat 내에서 태스크를 생성하세요.**
"인지했지만 조치하지 않음"이나 "다음 Heartbeat에서 처리"는 금지입니다. delegate_task / send_message / call_human / state/current_state.md 중 하나로 즉시 액션을 취하세요.

Heartbeat에서 직접 작업을 수행하지 마세요. 실제 작업은 `submit_tasks`로 제출한 TaskExec가 별도 세션에서 수행합니다. 하니스는 장부의 pending을 자동으로 재실행하지 않습니다. `update_task(status="in_progress")`로 바꾸거나 `state/current_state.md`에 기록해도 아무것도 실행되지 않습니다.
관찰 중 가벼운 재사용 가능 능력을 발견하면 `create_skill`로 생성하세요. 작성이 무거우면 스킬 작성 태스크를 만드세요.

완료된 백그라운드 태스크 결과는 state/task_results/에 있습니다.
중요한 결과가 있으면 확인하고, 필요에 따라 후속 조치를 계획하세요.

태스크 큐의 **pending**(이전 TaskExec가 완료 선언 없이 끝난 것 포함)은 당신이 제출하지 않는 한 실행되지 않습니다:
- 계속 → 같은 task_id·원래 지시·필요한 workspace로 `submit_tasks`에 제출
- 중단 → `update_task(task_id="...", status="cancelled")`로 바꾸고 의뢰자에게 이유 전달
