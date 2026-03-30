# 자주 발생하는 문제와 대처법

업무 중 자주 겪는 문제와 대처 절차를 정리한 레퍼런스입니다.
각 문제는 "증상 → 원인 → 대처 절차" 형식으로 기재되어 있습니다.

곤란한 상황이 발생하면 먼저 이 문서를 읽고 해당하는 항목의 절차를 따르세요.
여기서 해결되지 않는 경우 `troubleshooting/escalation-flowchart.md`를 참조하여 적절하게 에스컬레이션하세요.

---

## 메시지가 전달되지 않음

### 증상

- 보냈다고 생각한 메시지에 답장이 없음
- 상대방이 메시지를 받지 못했다고 함
- `send_message`를 실행했지만 상대방이 반응하지 않음

### 원인

1. 상대방 이름(Anima 이름)이 잘못되었음
2. 서버가 정지되어 있음
3. 상대방이 heartbeat 간격 사이에 있음 (다음 기동까지 미읽 상태)
4. 전송 처리 자체가 에러로 실패했음
5. `intent`가 미지정이거나 부적절함 (report / question만 허용. 태스크 위임은 `delegate_task` 사용)
6. 세션 내 DM 제한 초과 (동일 수신자에게 1회만, 1세션당 수신자 수 상한. 역할에 따라 다르며, general은 2명까지)

### 대처 절차

1. **수신자 이름을 확인합니다**
   - `send_message`의 `to` 파라미터에 지정한 이름이 올바른지 확인합니다
   - 이름은 대소문자를 구분합니다. `identity.md`에 기재된 정식 명칭을 사용하세요
   - 확인 방법:
     ```
     search_memory(query="조직", scope="common_knowledge")
     ```
     또는 `read_memory_file(path="reference/organization/structure.md")`로 조직 내 전체 Anima 이름을 확인
   - **주의**: 채팅 중에 사람에게 보내는 경우 `send_message`를 사용하지 마세요. 직접 텍스트로 답하면 사람에게 전달됩니다. 채팅 외(heartbeat 등)에서 사람에게 연락하려면 `call_human`을 사용하세요

2. **서버 가동 상태를 확인합니다**
   - 자신이 동작하고 있는 시점에서 서버는 가동 중일 것입니다
   - 그래도 불안한 경우 상사에게 "메시지가 전달되지 않는다"고 보고하세요

3. **상대방의 응답을 기다립니다**
   - 상대방은 heartbeat 간격(예: 30분마다)으로 inbox를 확인합니다
   - 즉각 답신이 없어도 다음 heartbeat에서 처리됩니다
   - 긴급한 경우 상사에게 "급히 연락을 취하고 싶다"고 보고하여 수동 기동을 요청하세요

4. **전송 에러가 발생한 경우**
   - 에러 메시지를 기록합니다
   - `state/current_state.md`에 차단 이유로 기재합니다
   - 상사에게 보고합니다

### 구체적 예시

```
# 이름을 잘못 입력한 경우
send_message(to="Aoi", content="...", intent="report")   # OK
send_message(to="aoi", content="...", intent="report")  # 이름이 다르면 에러 가능성 있음

# DM은 intent 필수 (report / question만). 태스크 위임은 delegate_task 사용
# 1세션당 수신자 수는 역할에 따라 다름 (general은 2명까지). 동일 수신자에게 1회만
send_message(
    to="aoi",
    content="알겠습니다. 작업을 시작합니다.",
    intent="report",           # 필수: report / question
    reply_to="msg-abc123",     # 선택: 원본 메시지 ID
    thread_id="thread-xyz789"  # 선택: 스레드 ID
)

# 확인/감사/안내만을 위한 DM은 불가 → post_channel(Board)을 사용
```

---

## 태스크가 차단됨

### 증상

- 작업을 진행하려 했지만 필요한 정보나 권한이 부족함
- 다른 Anima의 작업 완료를 기다리는 상태
- 외부 서비스가 에러를 반환함

### 원인

1. 의존 태스크가 미완료
2. 권한 부족 (permissions.json에서 허가되지 않은 조작을 시도)
3. 필요한 정보 부족
4. 외부 서비스 장애

### 대처 절차

1. **차단 내용을 명확히 합니다**
   - 무엇이 부족한지 구체적으로 특정합니다
   - "누구의", "어떤 작업이", "언제까지" 필요한지 정리합니다

2. **`state/current_state.md`를 업데이트합니다**
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## 현재 태스크\n\nXXX 구현\n\n### 차단 중\n- 원인: YYY 작업 완료 대기\n- 대기 대상: ZZZ\n- 발생 일시: 2026-02-15 10:00",
       mode="overwrite"
   )
   ```

3. **스스로 해결할 수 있는지 판단합니다**
   - 다른 접근 방식으로 회피할 수 없는지 검토합니다
   - 메모리를 검색하여 과거에 동일한 문제가 없었는지 확인합니다:
     ```
     search_memory(query="차단", scope="episodes")
     search_memory(query="회피", scope="knowledge")
     ```

4. **해결할 수 없는 경우 에스컬레이션합니다** (`troubleshooting/escalation-flowchart.md` 참조)
   - 상사에게 보고합니다. 보고에는 다음을 포함하세요:
     - 무엇을 하려 했는지
     - 무엇에 의해 차단되었는지
     - 언제부터 차단되었는지
     - 스스로 시도한 대처
   ```
   send_message(
       to="supervisor_name",
       content="[차단 보고]\n태스크: XXX 구현\n차단 원인: YYY API 권한 부족\n발생: 2026-02-15 10:00\n시도: permissions.json를 확인했으나 해당 설정 없음\n요청: API 권한 추가를 부탁합니다",
       intent="report"
   )
   ```

5. **차단 중에도 진행 가능한 작업이 있는지 확인합니다**
   - 영속 태스크 큐(`Bash: animaworks-tool task list`)와 `state/pending/` 하위의 태스크에 다른 작업이 없는지 확인합니다
   - 차단되지 않은 다른 태스크에 착수합니다

---

## 메모리를 찾을 수 없음

### 증상

- 과거에 했던 것을 기억해 낼 수 없음
- 절차서가 있을 텐데 찾을 수 없음
- 검색해도 관련 결과가 반환되지 않음

### 원인

1. 검색 키워드가 적절하지 않음
2. 검색 스코프(scope)가 너무 좁음
3. 아직 메모리로 기록되지 않음 (처음 하는 작업)
4. 파일 경로를 잘못 입력함

### 대처 절차

1. **스코프를 넓혀 재검색합니다**
   - 먼저 `all` 스코프로 넓게 검색합니다:
     ```
     search_memory(query="검색하고 싶은 키워드", scope="all")
     ```
   - 결과가 너무 많으면 스코프를 좁힙니다:
     ```
     search_memory(query="Slack 설정", scope="procedures")    # 절차서 한정
     search_memory(query="Slack 장애", scope="episodes")      # 과거 이벤트 한정
     search_memory(query="Slack", scope="knowledge")          # 학습한 지식 한정
     ```

2. **키워드를 바꿔 재검색합니다**
   - 동의어와 관련어로 시도합니다 (예: "전송", "메시지", "알림", "연락")
   - 영어 키워드로도 시도합니다 (예: "slack", "message", "send")
   - 부분 일치를 의식합니다 (예: "Chatwork" → "chatwork", "chat work")

3. **공유 지식을 검색합니다**
   - 개인 메모리에 없는 경우 공유 지식에 존재할 수 있습니다:
     ```
     search_memory(query="검색 키워드", scope="common_knowledge")
     ```
   - 공유 지식의 목차를 확인합니다:
     ```
     read_memory_file(path="common_knowledge/00_index.md")
     ```

4. **디렉터리를 직접 확인합니다**
   - 검색으로 찾지 못하는 경우 `Glob`으로 디렉터리 내용을 확인합니다. `path`를 생략하면 anima_dir 루트가 표시되며, knowledge/, procedures/, episodes/ 등의 서브디렉터리를 확인할 수 있습니다
   - 파일명으로 원하는 파일을 찾아 직접 읽습니다:
     ```
     read_memory_file(path="procedures/slack-setup.md")
     read_memory_file(path="knowledge/xxx-findings.md")
     ```

5. **메모리가 존재하지 않는 경우**
   - 처음 하는 작업일 수 있습니다
   - 공유 지식(`common_knowledge/`)에 관련 가이드가 없는지 확인합니다
   - 상사나 동료에게 지견이 없는지 문의합니다
   - 작업 완료 후에는 MUST로 메모리에 기록합니다 (다음을 위해)
   - 오래된/중복된 메모리는 `archive_memory_file(path="...", reason="...")`로 archive/에 퇴피할 수 있습니다 (삭제가 아닌 이동. `reason`은 필수)

### 검색 스코프 일람

| scope | 검색 대상 | 용도 |
|-------|----------|------|
| `knowledge` | 학습한 지식, 노하우 | 대응 방침, 기술 메모 |
| `episodes` | 과거 행동 로그 | "언제 무엇을 했는지"의 사실 확인 |
| `procedures` | 절차서 | "어떻게 하는지"의 절차 확인 |
| `common_knowledge` | 전체 Anima 공유 지식 | 조직 규칙, 시스템 가이드 |
| `activity_log` | 최근 3일간 활동 로그 (BM25 키워드 검색) | "방금 읽은 메일", "직전 검색 결과" 등 최근 행동 회상 |
| `all` | 위 모두 (벡터 검색 + activity_log BM25를 RRF로 통합) | 키워드 존재 확인, 넓은 범위 검색 |

---

## 권한이 없음

### 증상

- 도구를 실행했더니 "권한이 없습니다", "Permission denied" 등의 에러가 반환됨
- 파일을 읽거나 쓰려 했지만 접근할 수 없음
- 명령을 실행하려 했지만 거부됨

### 원인

1. `permissions.json`에서 허가되지 않은 조작을 시도함
2. 외부 도구 카테고리가 미활성화
3. 파일 경로가 허가 범위 밖

### 대처 절차

1. **자신의 권한을 확인합니다**
   ```
   check_permissions()
   ```
   - 사용 가능한 내부 도구, 외부 도구, 파일 접근, 제한 사항이 목록으로 반환됩니다
   - 상세 내용은 `read_memory_file(path="permissions.json")`로 확인 가능합니다
   - `permissions.json`의 주요 섹션:
     - "파일 조작" / "읽기 가능한 경로": 읽기 가능한 경로
     - "명령 실행" / "실행 가능한 명령": 실행 가능 명령 화이트리스트
     - "실행 불가 명령": 차단 대상 명령
     - 외부 도구: permissions.json에서 허가된 카테고리가 활성화됩니다

2. **허가된 조작인지 확인합니다**
   - 자신의 anima_dir 내는 읽기/쓰기 가능합니다. 공유 디렉터리, 부하의 관리 파일 등은 `check_permissions`로 확인합니다
   - 명령: "실행 가능한 명령"에 나열된 명령만 실행 가능합니다

3. **권한이 필요한 경우의 대응**
   - 해당 조작이 정말 필요한지 재검토합니다
   - 다른 접근 방식(허가 범위 내의 조작)으로 대체할 수 없는지 생각합니다
   - 대체 불가능한 경우 상사에게 권한 추가를 요청합니다:
   ```
   send_message(
       to="supervisor_name",
       content="[권한 추가 요청]\n목적: XXX 작업을 위해\n필요한 권한: /path/to/dir 읽기\n이유: YYY 정보를 참조해야 하므로",
       intent="question"
   )
   ```

4. **절대 해서는 안 되는 것**
   - 권한 체크를 우회하려는 시도
   - 허가되지 않은 명령을 다른 방법으로 실행하려는 시도
   - 다른 Anima의 권한을 이용하려는 시도

---

## 도구를 사용할 수 없음

### 증상

- 도구를 호출했더니 "도구를 찾을 수 없습니다" 등의 에러가 반환됨
- 외부 도구(Slack, Gmail 등)를 이용할 수 없음

### 원인

1. 해당 도구가 `permissions.json`에서 허가되지 않음
2. 스킬 파일을 찾을 수 없음
3. 외부 서비스의 인증 정보가 설정되지 않음

### 대처 절차

1. **스킬로 도구 사용법을 확인합니다**
   - `read_memory_file`로 시스템 프롬프트의 스킬 카탈로그에 표시된 경로(예: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`)를 지정하여 절차 전문을 가져옵니다
   - B-mode에서 외부 도구가 허가된 경우, `Bash: animaworks-tool <도구> <서브커맨드>`로 호출 가능합니다

2. **권한을 확인합니다**
   ```
   check_permissions()
   ```
   - `external_tools.enabled`에 현재 활성 카테고리, `external_tools.available_but_not_enabled`에 허가되었지만 미활성인 카테고리가 반환됩니다
   - permissions.json에서 허가되지 않은 카테고리는 사용할 수 없습니다

3. **카테고리가 허가되지 않은 경우**
   - 상사에게 이용 허가를 요청합니다
   - 요청 시 "왜 해당 도구가 필요한지"를 명기하세요

4. **S-mode (Claude Agent SDK / MCP)의 경우**
   - 내장 도구는 접두사 없이 사용 가능합니다 (예: `send_message`). 찾을 수 없는 경우 프로세스 재시작이 필요합니다
   - 외부 도구는 `read_memory_file`로 스킬 본문을 읽어 CLI 사용법을 확인하고, **Bash** 경유로 `animaworks-tool <도구> <서브커맨드>`를 실행합니다
   - 장시간 도구(이미지 생성, 로컬 LLM 등)는 `animaworks-tool submit`으로 비동기 실행합니다

5. **A-mode (LiteLLM)의 경우**
   - 외부 도구는 `read_memory_file`로 스킬 본문을 읽어 사용법을 확인하고, **Bash** 경유로 `animaworks-tool <도구> <서브커맨드>`를 실행합니다

6. **도구가 에러를 반환하는 경우**
   - 에러 메시지를 정확히 기록합니다
   - 인증 에러의 경우 상사에게 보고합니다 (인증 정보 설정은 관리자 책임)
   - 타임아웃의 경우 재시도합니다 (최대 3회)
   - 재시도로도 해결되지 않는 경우 차단으로 보고합니다

도구 체계의 전체 개요는 `operations/tool-usage-overview.md`를 참조하세요.

---

## 컨텍스트가 너무 길어짐

### 증상

- 세션이 장시간 지속되고 있음
- 응답이 느려지고 있음
- 시스템으로부터 "컨텍스트 상한에 근접하고 있다"는 알림이 있음

### 원인

- 장시간 작업이나 다수의 도구 호출로 컨텍스트 윈도우가 소모됨
- 대량의 파일 내용을 읽어들임

### 대처 절차

1. **작업 상태를 단기 메모리에 저장합니다** (MUST)
   - 현재 작업 상태를 `shortterm/`에 저장합니다 (채팅 세션 시 `shortterm/chat/`):
   ```
   write_memory_file(
       path="shortterm/chat/session_state.md",
       content="## 작업 상태\n\n### 실행 중인 태스크\n- XXX 구현 (50% 완료)\n\n### 다음 단계\n1. YYY 완료\n2. ZZZ 테스트\n\n### 중요한 중간 결과\n- AAA 조사 결과: BBB\n- CCC 설정값: DDD",
       mode="overwrite"
   )
   ```
   - heartbeat 세션 시에는 `shortterm/heartbeat/session_state.md`를 사용합니다

2. **`state/current_state.md`를 업데이트합니다** (MUST)
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## 현재 태스크\n\nXXX 구현\n\n### 진행 상황\n- 50% 완료\n- 다음에는 YYY부터 재개\n\n### 메모\n- 중요한 발견 사항을 여기에 기재",
       mode="overwrite"
   )
   ```

3. **중요한 지견은 영속 메모리에 저장합니다** (SHOULD)
   - 작업 중 얻은 지견을 `knowledge/`에 저장합니다:
   ```
   write_memory_file(
       path="knowledge/xxx-findings.md",
       content="# XXX에 관한 지견\n\n## 발견 사항\n...",
       mode="overwrite"
   )
   ```

4. **세션 계속을 기다립니다**
   - 시스템이 자동으로 새 세션을 시작합니다
   - 새 세션에서는 `shortterm/chat/` (또는 `shortterm/heartbeat/`) 내용이 컨텍스트에 포함됩니다
   - `state/current_state.md`를 다시 읽어 작업을 재개합니다

### 예방 대책

- 큰 파일은 전체를 읽지 말고 필요한 부분만 검색합니다
- 장시간 작업에서는 정기적으로 `state/current_state.md`를 업데이트합니다
- 중간 결과는 수시로 메모리에 저장합니다

---

## 메시지 전송이 제한됨

### 증상

- `send_message`나 `post_channel`을 실행했더니 에러가 반환됨
- `GlobalOutboundLimitExceeded: Hourly send limit (N) reached...` 같은 메시지가 표시됨
- `ConversationDepthExceeded: Conversation with {recipient} reached 6 turns in 10 minutes...`가 표시됨

### 원인

- **역할별 제한**: 시간당/24시간당 전송 상한은 `status.json`의 `role`에 따른 기본값이 적용됩니다 (예: general 15/50통, manager 60/300통). `status.json`의 `max_outbound_per_hour` / `max_outbound_per_day`로 개별 재정의 가능
- 동일 channel에 대한 연속 게시가 쿨다운 기간 내였음 (`config.json`의 `heartbeat.channel_post_cooldown_s`, 기본값 300초)
- 2자 간 DM 왕복이 깊이 제한(10분간 6턴)을 초과함 (`heartbeat.depth_window_s` / `heartbeat.max_depth`)

### 대처 절차

1. **에러 메시지를 확인합니다**: 시간 제한, 24시간 제한, 깊이 제한 중 무엇인지 특정합니다
2. **전송 이력을 돌아봅니다**: 불필요한 전송이 없었는지 확인합니다
3. **대기합니다**: 시간 제한이면 다음 1시간 단위까지, 24시간 제한이면 다음 날까지, 깊이 제한이면 다음 heartbeat 사이클까지 기다립니다
4. **전송 내용을 기록합니다**: 이번 턴에서는 `send_message`를 사용하지 않고, 보내고 싶은 내용을 `state/current_state.md`에 기록하여 다음 세션에서 전송합니다
5. **긴급 연락**: `call_human`은 제한 대상이 아니므로 사람에 대한 연락은 계속 가능합니다
6. **전송을 통합합니다**: 여러 보고를 1통으로 정리합니다. 깊이 제한에 도달한 경우 복잡한 논의를 Board channel로 이행합니다

상세 내용은 `communication/sending-limits.md`를 참조하세요.

---

## 명령이 차단됨

### 증상

- 명령을 실행하려 했더니 "PermissionDenied", "Command blocked" 등의 에러가 반환됨
- 특정 명령만 실행할 수 없음

### 원인

1. 시스템 전체 차단 목록에 포함된 명령 (`rm -rf /` 등의 위험한 명령)
2. `permissions.json`의 "실행 불가 명령" 섹션에 기재된 명령

### 대처 절차

1. **자신의 권한을 확인합니다**
   ```
   read_memory_file(path="permissions.json")
   ```
   - `## Disallowed commands` 섹션에 차단 대상이 기재되어 있습니다

2. **대체 수단을 검토합니다**
   - 차단된 명령과 동등한 조작을 허가된 도구로 실현할 수 없는지 생각합니다
   - 예: `rm -rf`가 차단된 경우, 개별 파일 삭제는 허가되어 있을 가능성이 있습니다

3. **권한 변경이 필요한 경우**
   - 상사에게 차단 해제를 요청합니다
   - 요청 시 "왜 해당 명령이 필요한지"를 명기하세요

---

## 프롬프트가 단축됨

### 증상

- 통상 표시되어야 할 정보(조직 컨텍스트, 메모리 가이드 등)가 시스템 프롬프트에 포함되지 않음
- 도구 종류가 적음
- 메모리 자동 상기(Priming)가 동작하지 않는 것 같음

### 원인

컨텍스트 윈도우가 작은 모델을 사용하는 경우, 시스템 프롬프트가 단계적으로 축소됩니다 (Tiered System Prompt).
`status.json`의 모델명에서 컨텍스트 윈도우를 추정하며, `~/.animaworks/models.json` 또는 `config.json`의 `model_context_windows`로 재정의 가능합니다.

| 티어 | 컨텍스트 윈도우 | 생략되는 정보 |
|------|----------------|--------------|
| T1 (FULL) | 128k+ 토큰 | 없음 (전체 정보 표시) |
| T2 (STANDARD) | 32k~128k 토큰 | 증류 지식(폐지 예정), Priming 예산 축소 |
| T3 (LIGHT) | 16k~32k 토큰 | bootstrap, vision, specialty, 증류 지식, 메모리 가이드 생략 |
| T4 (MINIMAL) | 16k 미만 토큰 | permissions, Priming, org, messaging, emotion도 생략 |

### 대처 절차

1. **필요한 정보는 직접 검색합니다**: 생략된 정보는 `search_memory`나 `read_memory_file`로 명시적으로 가져옵니다
2. **상사에게 상담합니다**: 모델 변경이 필요한 경우 상사에게 요청합니다

---

## 기타 자주 발생하는 문제

### 파일을 찾을 수 없음

- **원인**: 경로 지정 실수, 파일이 존재하지 않음
- **대처**: `Glob`으로 디렉터리 내용을 확인한 후 경로를 지정합니다
- **주의**: `read_memory_file`은 Anima 디렉터리로부터의 상대 경로(예: `knowledge/xxx.md`, `reference/organization/structure.md`)를 사용합니다. `Read`는 절대 경로를 사용합니다

### read_channel에서 inbox를 지정할 수 없음

- **원인**: `read_channel`은 Board의 공유 channel용입니다. inbox(수신함)는 channel이 아닙니다
- **대처**: inbox 메시지는 시스템이 자동 처리합니다. `read_channel`에 `inbox`나 `inbox/`를 지정하면 에러가 발생합니다

### 명령이 타임아웃됨

- **원인**: 처리 시간이 `timeout`을 초과함
- **대처**: Bash 실행 시 `timeout` 파라미터를 늘립니다 (기본값: 30초)
- **주의**: 장시간 실행 명령에는 적절한 타임아웃 값을 설정하세요

### 상대방 Anima가 존재하지 않음

- **원인**: Anima 이름 오류, 또는 해당 Anima가 아직 생성되지 않음
- **대처**: 상사에게 확인합니다. 조직 구조는 `reference/organization/structure.md`를 참조하세요
