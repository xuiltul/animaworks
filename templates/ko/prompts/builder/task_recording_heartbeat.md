### 태스크 기록 (Heartbeat 중)

- heartbeat에서 감지한 작업은 list_tasks로 중복을 확인한 후 착수한다.
- 외부 의존으로 진행 불가한 태스크는 이유를 붙여 cancelled로 처리한다.
- 정기 확인 항목은 heartbeat.md 또는 cron.md에 추가하여 내재화한다.
