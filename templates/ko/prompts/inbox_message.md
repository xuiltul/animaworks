Inbox에 메시지가 도착했습니다. 아래 내용을 확인하고 적절히 응답하세요.

{messages}

## 대응 가이드라인
- 질문에는 직접 답변하세요
- 요청에는 확인과 예상 일정을 답변하세요
- **[MUST] 수행이 필요한 작업을 파악했다면, 반드시 작업으로 구체화하세요. 답변만 하고 잊어버려서는 안 됩니다.**
  - 부하에게 맡기기 → `delegate_task`
  - 나중에 직접 하기 → `submit_tasks` (state/pending/에 기록되어 TaskExec가 별도 세션에서 실행)
- 답변은 간결하게 (장문 불필요)

### 외부 플랫폼 메시지에 대한 응답
메시지에 `[reply_instruction: ...]` 메타데이터가 있는 경우:
- **반드시 해당 지시에 따라 응답**하세요 (`Bash`로 실행)
- `{reply_content}`를 실제 응답 내용으로 대체하세요
- `send_message`를 사용하지 마세요 (DM으로 전송되며, 스레드 응답이 아닙니다)

**작업 제출 가이드라인**: `submit_tasks` / `delegate_task` 사용 시 `read_memory_file(path="common_knowledge/operations/task-delegation-guide.md")`의 작성 원칙과 금지 패턴을 따르세요 (MUST).
