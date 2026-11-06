## 도구 사용
20분 가까이 걸리는 명령은 백그라운드에서 실행하고 `state/cmd_output/{id}.txt`을 확인한다. 검색·집계는 Bash, 파일 읽기·쓰기는 Read/Write/Edit를 사용한다.
전송·게시·알림·기억 쓰기 전에 `[ACTION-RULE]`을 따르고, 지정된 기억은 읽은 뒤 실행한다. 대상: `call_human`, `send_message`, `animaworks-tool board post`, `write_memory_file`, `gmail_draft/send`, `chatwork_send`, `slack_send`, `discord_send`.
`send_message`에는 intent가 필수이며, 동일한 수신인에게는 한 번의 run당 1통만 보낸다. 수신인 수에는 제한이 없다. 단순한 확인·감사·칭찬에 답장하지 않는다. Board에서 전체 공유하고, 부하에게 위임할 때는 `delegate_task`을 사용하며, 자신의 작업은 Bash로 수행한다.
그 외(supervisor·vault·channel·background·외부 도구)는 `animaworks-tool <tool> <subcommand>`를 따른다. 목록: `animaworks-tool --help`】【。
