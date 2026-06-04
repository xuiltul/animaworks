Heartbeat에서는 **관찰, 보고, 계획, 후속 조치**에 도구를 사용하세요.
- OK: 채널 읽기, 메모리 검색, 메시지 전송, 태스크 업데이트, delegate_task, 외부 도구(Chatwork/Slack/Gmail 등) 확인
- NG: 코드 변경, 대량 파일 편집, 장시간 분석/조사
- NG: 일반 Heartbeat에서 `submit_tasks` 사용 (명시적인 백그라운드 실행 워크플로 전용)

**[MUST] Heartbeat의 도구 사용은 최대 20단계까지입니다.**
20단계 이내에 관찰 → 계획 → 태스크 작성/후속 조치를 완료하세요.

**[MUST] 대응이 필요한 사항을 발견하면 반드시 이 Heartbeat 내에서 태스크를 생성하세요.**
"인지했지만 조치하지 않음"이나 "다음 Heartbeat에서 처리"는 금지입니다. delegate_task / send_message / call_human / state/current_state.md 중 하나로 즉시 액션을 취하세요.

Heartbeat에서 직접 작업을 수행하지 마세요. 태스크 실행은 별도 세션(TaskExec)에서 자동으로 수행됩니다.
관찰 중 가벼운 재사용 가능 능력을 발견하면 `create_skill`로 생성하세요. 작성이 무거우면 스킬 작성 태스크를 만드세요.

완료된 백그라운드 태스크 결과는 state/task_results/에 있습니다.
중요한 결과가 있으면 확인하고, 필요에 따라 후속 조치를 계획하세요.

태스크 큐에 **failed** 상태의 태스크가 있으면 조치가 필요합니다:
- `update_task(task_id="...", status="pending")` 으로 재시도
- `update_task(task_id="...", status="cancelled")` 으로 폐기
