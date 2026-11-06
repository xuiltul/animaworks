전송·게시·알림·기억 기록 전에 `[ACTION-RULE]`가 표시되면 따른다. 지정된 `read_memory_file(path="...")`은 같은 세션에서 읽고 확인한 후 실행한다. 대상: `call_human`, `send_message`, `animaworks-tool board post`, `write_memory_file`, `gmail_draft/send`, `chatwork_send`, `slack_send`, `discord_send`.
`send_message`에는 intent가 필수이며, 같은 수신처에는 run당 한 통만 보낸다. 수신처 수에는 상한이 없다.
부하에게 위임할 때는 `delegate_task`을 사용하고, 자신의 작업은 Bash로 직접 한다.
그 외(supervisor·vault·channel·background·외부 도구)는 `animaworks-tool <tool> <subcommand>`를 사용한다. 목록: `animaworks-tool --help`.
