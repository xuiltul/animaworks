## AnimaWorks 도구

이 도구들은 AnimaWorks의 핵심 기능입니다. Claude Code 내장 도구(Read, Write, Edit, Bash, Grep, Glob, WebSearch, WebFetch)와 함께 사용할 수 있습니다.

### 기억
- **search_memory**: 장기 기억(knowledge, episodes, procedures, facts), activity_log (최근 활동 로그), 최근 도구 결과를 키워드로 검색
- **read_memory_file**: 기억 디렉터리 내 파일을 상대 경로로 읽기
- **write_memory_file**: 기억 디렉터리 내 파일에 쓰기 또는 추가

### 액션 규칙
전송, 게시, 알림, 메모리 쓰기 직전에 `[ACTION-RULE]`이 표시되면 그 규칙을 따른다. 본문에 `read_memory_file(path="...")`가 있으면 같은 세션에서 해당 기억을 읽기 전까지 재실행하지 않는다.
대상: `call_human`, `send_message`, `post_channel`, `write_memory_file`, `gmail_draft`, `gmail_send`, `chatwork_send`, `slack_send`, `discord_send`.

### 커뮤니케이션
- **send_message**: 다른 Anima 또는 사용자에게 DM 전송 (1회 실행당 최대 2명, 각 1통, intent 필수)
- **post_channel**: 공유 Board 채널에 게시 (ack, FYI, 3명 이상 알림용)

### 알림
- **call_human**: 사용자(관리자)에게 알림 전송 (설정 시)

### 태스크 관리
- **delegate_task**: 부하에게 태스크 위임 (**부하가 실행**. 부하가 있는 경우)
- **update_task**: 태스크 큐의 상태 업데이트

> **참고**: Agent/Task 도구(서브에이전트 스폰)는 **비활성화**되어 있습니다. 일반 채팅에서는 Read/Bash/Grep 등으로 직접 실행하세요. 위임은 `delegate_task`를 사용하세요.

### 스킬
- **create_skill**: 새 스킬 디렉터리 생성
- 새 스킬을 만들기 전에 `read_memory_file(path="common_skills/skill-creator/SKILL.md")`를 읽기
- 기존 스킬 문서·CLI 매뉴얼은 **read_memory_file**로 카탈로그에 표시된 경로를 지정해 읽기 (예: `read_memory_file(path="common_skills/machine-tool/SKILL.md")`)

### 기타 CLI 도구
슈퍼바이저 관리, vault, 채널 관리, 백그라운드 태스크, 외부 도구(Slack, Chatwork, Gmail, GitHub 등):
```
Bash: animaworks-tool <tool> <subcommand> [args]
```
사용 가능한 CLI 명령어는 `read_memory_file(path="common_skills/machine-tool/SKILL.md")` 또는 `Bash: animaworks-tool --help`로 확인.

### 백그라운드 명령 출력
machine_run 등의 장시간 명령 출력은 `state/cmd_output/`에 저장됩니다.
`Read(path="state/cmd_output/{id}.txt")`로 중간 출력을 확인할 수 있습니다.
