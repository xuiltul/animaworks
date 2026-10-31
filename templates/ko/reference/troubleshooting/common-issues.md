# 자주 발생하는 문제와 해결 방법

업무 중 자주 마주치는 문제와 그 해결 절차를 정리한 참조 문서.
각 문제는 "증상 → 원인 → 해결 절차" 형식으로 기술되어 있다.

막히면 먼저 이 문서를 읽고 해당 항목의 절차를 따를 것.
여기서 해결되지 않으면 `troubleshooting/escalation-flowchart.md`를 참조하여 적절히 에스컬레이션할 것.

---

## 메시지가 도착하지 않음

### 증상

- 보냈어야 할 메시지에 답변이 없음
- 상대방이 메시지를 받지 못했다고 함
- `send_message`을 실행했지만 상대방이 반응하지 않음

### 원인

1. 수신자 지정 오류(Anima 공식 이름·사용자 별칭·`slack:` / `chatwork:` 접두사 등) 또는 해석 순서에 맞지 않는 지정
2. 서버가 종료되어 있음
3. 상대방의 Anima가 종료·비활성화되었거나 Inbox의 Provider 오류 대기 중
4. 수신자 해석, 권한, 외부 채널 배송 등 실제 전송 오류가 발생함
5. `intent`가 미지정 또는 잘못됨. DM에서는 `report` / `question`만 사용. 작업 위임은 `delegate_task`를 사용(`send_message`에 `intent="delegation"`을 붙이면 비권장 메시지가 반환됨)
6. 동일 run에서 같은 수신자에게 이미 DM을 전송함(같은 수신자에게 두 번째 메시지만 거부. 수신자 수 상한은 없음)

### 해결 절차

1. **전송 대상의 이름·주소 형식을 확인**
   - `send_message`의 `to` 파라미터가 의도한 상대방으로 해석되는지 확인
   - 구현(`core/messaging/outbound.py` `resolve_recipient`)의 해석 순서는 대략 다음과 같음:
     1. 알려진 Anima 이름과 **정확히 일치**(대소문자 구분) → 내부
     2. `config.json` `external_messaging.user_aliases`의 **별칭**(대소문자 무시) → 외부(preferred_channel)
     3. `slack:USERID` / `chatwork:ROOMID` → 외부 직접
     4. 베어 Slack 사용자 ID(`U` + 영숫자 8자 이상) → Slack 직접
     5. 알려진 Anima 이름의 **대소문자 무시 일치** → 내부
     6. 그 외 → 해석 실패
   - 내부 Anima에 확실히 전달하려면 `~/.animaworks/animas/<名前>/` 또는 `reference/organization/structure.md`의 **공식 이름**을 사용
   - 확인 방법:
     ```
     search_memory(query="조직", scope="common_knowledge")
     ```
     또는 `read_memory_file(path="reference/organization/structure.md")`로 조직의 모든 Anima 이름 확인
   - **주의**: 채팅 중 사람에게 보낼 때는 `send_message`를 사용할 수 없음. 직접 텍스트로 답하면 사람에게 전달됨. 채팅 외(하트비트 등)에서 사람에게 연락할 때는 `call_human`를 사용

2. **서버의 가동 상태를 확인**
   - 자신이 동작 중인 시점에 서버는 가동 중이어야 함
   - 그래도 불안하면 상급자에게 "메시지가 도착하지 않는다"고 보고

3. **상대방의 응답을 기다림**
   - 상대방은 하트비트 간격(예: 30분마다)으로 수신함을 확인함
   - 즉시 답변이 없어도 다음 하트비트에서 처리됨
   - 긴급하면 상급자에게 "긴급 연락을 취하고 싶다"고 보고하고 수동 시작을 요청

4. **전송 오류가 발생한 경우**
   - 오류 메시지를 기록
   - `state/current_state.md`에 상황을 기재
   - 상급자에게 보고

---

### 구체적인 예

```
# 名前を間違えていた場合
send_message(to="Aoi", content="...", intent="report")   # OK
send_message(to="aoi", content="...", intent="report")  # 名前が異なればエラーになる可能性あり

# DM は intent 必須（report / question のみ）。委譲は delegate_task
# 同一 run 内で同一宛先へ送れる DM は1通まで。宛先数の上限はない
send_message(
    to="aoi",
    content="了解しました。作業を開始します。",
    intent="report",           # 必須: report / question
    reply_to="msg-abc123",     # 任意: 元メッセージのID
    thread_id="thread-xyz789"  # 任意: スレッドID
)

# 確認・お礼・称賛だけのメッセージには返信しない。全体共有が必要なら post_channel（Board）を使用
```

---

## 작업을 진행할 수 없음

"조건이 갖춰질 때까지 대기" "블록 중"이라는 상태는 존재하지 않음(`blocked`은 폐지됨). 선택할 수 있는 것은 진행하기·종료하기·상담하기의 세 가지뿐임.

### 증상

- 작업을 진행하려 했지만 필요한 정보나 권한이 부족함
- 다른 Anima의 작업 완료를 기다리는 상태
- 외부 서비스가 오류를 반환함

### 원인

1. 의존 작업이 미완료
2. 권한 부족(`permissions.json`에서 허용되지 않은 작업. `permissions.md`만 있는 환경에서도 첫 `load_permissions`에서 JSON이 생성되고 MD는 `.bak`에 백업됨)
3. 필요한 정보가 부족함
4. 외부 서비스의 장애

### 해결 절차

1. **같은 작업을 반복하지 않음**
   - 무엇이 부족한지 구체적으로 파악
   - "누구의" "무엇 작업이" "언제까지" 필요한지 정리

2. **스스로 해결할 수 있는지 판단**
   - 다른 접근 방식으로 회피할 수 없는지 검토
   - 기억을 검색해 과거에 비슷한 문제가 있었는지 확인:
     ```
     search_memory(query="오류 내용이나 키워드", scope="episodes")
     search_memory(query="회피", scope="knowledge")
     ```

3. **해결할 수 없으면 의뢰자에게 보고**(`troubleshooting/escalation-flowchart.md` 참조). 보고에는 다음을 포함할 것:
   - 무엇을 하려 했는지
   - 무엇이 부족한지·무엇을 기다리고 있는지
   - 스스로 시도한 조치와 권장 사항
   ```
   send_message(
       to="上司の名前",
       content="【進捗報告】\nタスク: XXXの実装\n事実: YYYのAPI権限が不足\n試行: permissions.jsonを確認したが該当設定なし\n推奨: API権限の追加をお願いします",
       intent="report"
   )
   ```

4. **작업의 처리 방식을 결정**
   - 계속할 가능성이 있으면 아무것도 하지 않아도 됨(작업은 `pending` 상태로 유지. 선언 없이 세션이 끝나도 자동으로 `pending`로 복귀)
   - 더 이상 필요 없으면 `update_task(status="cancelled", summary="理由")`으로 처리

5. **기다리는 동안 진행할 수 있는 작업이 있는지 확인**
   - 영구 작업 큐: 도구를 사용할 수 있으면 `list_tasks` 또는 `Bash: animaworks-tool task list`로 확인
   - `list_tasks(detail=true)`로 미착수·의존 관계·대응 필요 사유를 확인. 기존 작업의 복제나 수동 재투입은 하지 않음
   - 다른 작업에 착수

---

## 기억을 찾을 수 없음

### 증상

- 과거에 했던 일을 기억하지 못함
- 절차서가 있어야 하는데 찾을 수 없음
- 검색해도 해당 결과가 반환되지 않음

### 원인

1. 검색 키워드가 적절하지 않음
2. 검색 범위(scope)가 너무 좁음
3. 아직 기억으로 기록되지 않음(처음 하는 작업)
4. 파일 경로를 잘못 지정함

### 해결 절차

1. **범위를 넓혀 재검색**
   - 먼저 `all` 범위로 넓게 검색:
     ```
     search_memory(query="검색할 키워드", scope="all")
     ```
   - 결과가 너무 많으면 범위를 좁힘:
     ```
     search_memory(query="Slack설정", scope="procedures")    # 절차서로 한정
     search_memory(query="Slack장애", scope="episodes")      # 과거 사건으로 한정
     search_memory(query="Slack", scope="knowledge")         # 학습한 지식으로 한정
     ```

2. **키워드를 바꿔 재검색**
   - 동의어나 관련어로 시도(예: "전송" "메시지" "알림" "연락")
   - 영어 키워드로도 시도(예: "slack", "message", "send")
   - 부분 일치를 고려(예: "Chatwork" → "chatwork" "챗워크")

3. **공유 지식을 검색**
   - 개인 기억에 없으면 공유 지식에 존재할 가능성이 있음:
     ```
     search_memory(query="검색 키워드", scope="common_knowledge")
     ```
   - 공유 지식의 목차를 확인:
     ```
     read_memory_file(path="common_knowledge/00_index.md")
     ```

4. **디렉터리를 직접 확인**
   - Mode S(Claude Agent SDK) 등에서는 내장 `Glob`로 Anima 디렉터리 하위를 나열할 수 있음. Mode A 등에서는 `read_memory_file`로 알려진 경로를 열거나 `search_memory`로 넓게 탐색
   - 파일 이름을 알면 직접 읽기:
     ```
     read_memory_file(path="procedures/slack-setup.md")
     read_memory_file(path="knowledge/xxx-findings.md")
     ```

5. **기억이 존재하지 않는 경우**
   - 처음 하는 작업일 가능성이 있음
   - 공유 지식(`common_knowledge/`)에 관련 가이드가 있는지 확인
   - 상급자나 동료에게 노하우가 있는지 문의
   - 작업 완료 후에는 MUST로 기억으로 기록(다음 번을 위해)
   - 오래된·중복된 기억은 `archive_memory_file(path="...", reason="...")`로 archive/에 백업할 수 있음(삭제가 아니라 이동. `reason`은 필수)

---

### 검색 범위 목록

| scope | 검색 대상 | 용도 |
|-------|---------|------|
| `knowledge` | 학습한 지식·노하우 | 대응 방침, 기술 메모 |
| `episodes` | 과거 행동 로그 | "언제 무엇을 했는지" 사실 확인 |
| `procedures` | 절차서 | "어떻게 하는지" 절차 확인 |
| `common_knowledge` | 모든 Anima 공유 지식 | 조직 규칙, 시스템 가이드 |
| `skills` | 스킬·공통 스킬(벡터 검색) | 스킬 발견·검색 |
| `activity_log` | 최근 행동 로그(도구 실행 결과·메시지 등) | "방금 읽은 메일" "아까의 검색 결과" 등 최근 사실 확인 |
| `all` | 위 모든 항목(벡터 검색 + activity_log BM25를 RRF로 통합) | 키워드 존재 확인, 광범위한 검색 |

---

## 권한이 없음

### 증상

- 도구를 실행했더니 "권한이 없습니다" "Permission denied" 등의 오류가 반환됨
- 파일을 읽거나 쓰려 했지만 접근할 수 없음
- 명령을 실행하려 했지만 거부됨

### 원인

1. `permissions.json`에서 허용되지 않은 작업(`permissions.md`만 있는 경우 `load_permissions`가 읽기 시 JSON 상당으로 정규화. 유효하지 않은 JSON은 경고 후 개방 기본값으로 폴백하기도 함)
2. 외부 도구의 카테고리가 레지스트리에 미활성화(`check_permissions`의 `available_but_not_enabled`)
3. 파일 경로가 허용 범위 밖(보호 파일에 쓰기, `file_roots` 밖 등). 또한 **전역** 거부는 `permissions.global.json`과 프레임워크 측 패턴이 있음

### 해결 절차

1. **자신의 권한을 확인**
   ```
   check_permissions()
   ```
   - JSON으로 반환됨. `internal_tools`·`external_tools.enabled` / `available_but_not_enabled`·`file_access`(read/write）・`restrictions`（명령 deny 등)를 확인
   - 원본 설정은 `read_memory_file(path="permissions.json")`(존재하지 않으면 `permissions.md`)로 확인 가능
   - 시스템 프롬프트에 주입되는 권한 설명은 런타임이 JSON에서 정리한 텍스트가 됨

2. **허용된 작업인지 확인**
   - 자신의 `anima_dir` 내는 원칙적으로 읽기·쓰기 가능(`IDENTITY.md` / `SOUL.md` 등의 보호 파일은 제외). 상급자·동료의 `activity_log`이나 하위의 `state/` 읽기는 역할에 따라 `check_permissions`의 `file_access`에 반영됨
   - 셸 명령: `permissions.json`의 `commands`(allow/deny）에 따름. 전역 위험 패턴은 프레임워크 측에서도 차단됨)

3. **권한이 필요한 경우의 대응**
   - 그 작업이 정말 필요한지 재검토
   - 다른 접근 방식(허용된 범위 내의 작업)으로 대체할 수 있는지 고려
   - 대체가 불가능하면 상급자에게 권한 추가를 요청:
   ```
   send_message(
       to="上司の名前",
       content="【権限追加依頼】\n目的: XXXの作業のため\n必要な権限: /path/to/dir の読み取り\n理由: YYYの情報を参照する必要があるため",
       intent="question"
   )
   ```

4. **절대 하면 안 되는 것**
   - 권한 확인을 회피하려는 시도
   - 허용되지 않은 명령을 다른 방법으로 실행하려는 시도
   - 다른 Anima의 권한을 이용하려는 시도

---

## 도구를 사용할 수 없음

### 증상

- 도구를 호출했지만 "도구를 찾을 수 없습니다" 등의 오류가 반환됨
- 외부 도구(Slack, Gmail 등)를 사용할 수 없음

### 원인

1. 해당 도구가 `permissions.json`(또는 읽기 시 정규화된 MD 유래 설정)에서 허용되지 않았거나 게이트가 있는 액션이 명시적으로 허용되지 않음
2. 스킬 파일을 찾을 수 없음
3. 외부 서비스의 인증 정보가 설정되지 않음

### 대처 절차

1. **스킬에서 도구 사용법을 확인한다**
   - `read_memory_file`에서 시스템 프롬프트의 스킬 카탈로그에 표시된 경로(예: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`)를 지정하고 절차의 전문을 획득한다
   - B-mode에서 외부 도구가 허용된 경우 `Bash: animaworks-tool <ツール> <サブコマンド>`로 호출이 가능하다

2. **권한을 확인한다**
   ```
   check_permissions()
   ```
   - `external_tools.enabled`: 이 Anima의 도구 레지스트리에 등록된 외부 도구 카테고리(세션에 실제로 전달되는 것)
   - `external_tools.available_but_not_enabled`: 프레임워크에 구현은 있지만 이 Anima에서는 레지스트리에 없는 카테고리. `permissions.json`의 허용·게이트형 액션·실행 모드와 함께 확인한다

3. **이용할 수 없는 경우**
   - `permissions.json`에서 해당 도구/액션이 허용되었는지, 인증 정보(`shared/credentials.json` 등)가 있는지 확인한다
   - 그래도 안 되면 상급자에게 요청한다(「왜 그 도구가 필요한지」를 명시)

4. **MCP 통합 모드(S/C/D/G: Claude Agent SDK / Codex CLI / Cursor Agent / Gemini CLI)의 경우**
   - 내장 도구는 프리픽스 없이 이용 가능(예: `send_message`). 찾을 수 없으면 프로세스 재시작이 필요하다
   - 외부 도구는 `read_memory_file`로 스킬 본문을 읽고 CLI 사용법을 확인한 후 **Bash** 경유로 `animaworks-tool <ツール> <サブコマンド>`를 실행한다(에이전트의 Bash 도구 사용)
   - 장시간 도구(이미지 생성, 로컬 LLM 등)는 `animaworks-tool submit`로 비동기 실행

5. **D-mode(Cursor Agent) 특유의 흔한 문제**
   - **CLI를 찾을 수 없음**: 호스트에 `cursor-agent` CLI가 설치되어 있는지 확인한다
   - **인증 오류**: 터미널에서 `agent login`를 실행하여 로그인한다
   - **폴백**: 해결되지 않으면 `execution_mode`을 `A`로 하거나, 모델을 LiteLLM 경유(Mode A)로 전환하여 운영한다

6. **G-mode(Gemini CLI) 특유의 흔한 문제**
   - **CLI를 찾을 수 없음**: 호스트에 `gemini` CLI가 설치되어 있는지 확인한다
   - **인증 오류**: `gemini auth login`를 실행하거나 환경 변수 `GEMINI_API_KEY`를 설정한다
   - **폴백**: 해결되지 않으면 `execution_mode`을 `A`로 하거나, 모델을 LiteLLM 경유(Mode A)로 전환한다. `gemini/` 프리픽스는 Google 프로바이더용으로 `google/`에 리매핑되는 경우가 있다

7. **A-mode(LiteLLM)의 경우**
   - 외부 도구는 `read_memory_file`로 스킬 본문을 읽고 사용법을 확인한 후 **Bash** 경유로 `animaworks-tool <ツール> <サブコマンド>`를 실행한다

8. **도구가 오류를 반환하는 경우**
   - 오류 메시지를 정확히 기록한다
   - 인증 오류면 상급자에게 보고한다(인증 정보 설정은 관리자의 책임)
   - 일시적인 타임아웃·레이트 제한이면 잠시 기다렸다가 재시도한다(횟수·간격은 도구 구현·서버 설정에 따름)
   - 개선되지 않으면 요청자에게 사실·시도한 내용을 보고한다

도구 체계의 전체 모습은 `operations/tool-usage-overview.md`을 참조한다.

---

## 컨텍스트가 너무 길어졌다

### 증상

- 세션이 장시간 지속되고 있다
- 응답이 느려졌다
- 시스템에서 「컨텍스트 상한에 가까워지고 있다」는 알림이 있다

### 원인

- 장시간 작업이나 다수의 도구 호출로 컨텍스트 윈도우가 소비되었다
- 대량의 파일 내용을 읽었다

### 대처 절차

1. **작업 상태를 단기 기억에 저장한다**(MUST)
   - 현재 작업 상태를 `shortterm/`에 기록한다(채팅 세션 시에는 `shortterm/chat/`):
   ```
   write_memory_file(
       path="shortterm/chat/session_state.md",
       content="## 作業状態\n\n### 実行中のタスク\n- XXXの実装（50%完了）\n\n### 次のステップ\n1. YYYを完了する\n2. ZZZをテストする\n\n### 重要な中間結果\n- AAAの調査結果: BBB\n- CCCの設定値: DDD",
       mode="overwrite"
   )
   ```
   - 하트비트 세션 시에는 `shortterm/heartbeat/session_state.md`를 사용한다

2. **`state/current_state.md`을 업데이트한다**(MUST)
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## 現在のタスク\n\nXXXの実装\n\n### 進捗\n- 50%完了\n- 次回はYYYから再開\n\n### メモ\n- 重要な発見事項をここに記載",
       mode="overwrite"
   )
   ```

3. **중요한 지식은 영구 기억에 저장한다**(SHOULD)
   - 작업 중에 얻은 지식은 `knowledge/`에 저장한다:
   ```
   write_memory_file(
       path="knowledge/xxx-findings.md",
       content="# XXXに関する知見\n\n## 発見事項\n...",
       mode="overwrite"
   )
   ```

4. **세션 지속을 기다린다**
   - 시스템이 자동으로 새 세션을 시작한다
   - 새 세션에서는 `shortterm/chat/`(또는 `shortterm/heartbeat/`)의 내용이 컨텍스트에 포함된다
   - `state/current_state.md`을 다시 읽고 작업을 재개한다

### 예방책

- 큰 파일은 전체를 읽지 말고 필요한 부분만 검색한다
- 긴 작업은 정기적으로 `state/current_state.md`을 업데이트한다
- 중간 결과는 수시로 기억에 기록한다

---

## 메시지 전송에서 오류가 반환되었다

### 증상과 원인

- `RecipientResolutionError`, 외부 채널의 `DeliveryFailed`, 권한·회사 경계 오류 등 실제 배송 오류가 반환된다
- `send_message`에서 동일 run 내에 같은 수신처로 두 번째 메시지를 보내려 하면 거부된다(중복 방지). 수신처 수의 상한은 없다
- `post_channel`에서 같은 run 내에 같은 채널로 두 번째 게시를 시도하면 거부된다. run 간 게시 cooldown은 없다
- 시간·일 단위 전송 예산이나 대화 깊이에 의한 전송 거부는 없다. 내부 Anima 간 깊이는 진단 로그에 기록되는 경우가 있다

### 대처 절차

1. **오류 내용을 확인한다**: `to`의 해결처, intent(`report` / `question`), 채널 ACL, 회사 경계, 외부 API의 응답을 확인한다
2. **중복 전송을 확인한다**: 동일 run에서 이미 보낸 수신처에는 추가 DM을 보낼 수 없다. 수신처 수를 이유로 기다릴 필요는 없다
3. **배송 실패를 조사한다**: Slack / Chatwork 등 외부 채널의 경우 반환된 배송 오류와 연결 설정을 확인한다
4. **Inbox의 Provider 오류의 경우**: `rate_guard`이 지정하는 복구 시간 후에 읽지 않은 메시지가 재처리된다

자세한 내용은 `communication/sending-limits.md`를 참조한다.

---

## 명령이 차단되었다

### 증상

- 명령을 실행하려 했더니 「PermissionDenied」「Command blocked」등의 오류가 반환되었다
- 특정 명령만 실행할 수 없다

### 원인

1. 프레임워크／`permissions.global.json`의 글로벌 거부 패턴에 해당하는 명령(예: `rm -rf /` 등)
2. `permissions.json`의 `commands.deny`에 열거된 명령

### 대처 절차

1. **자신의 권한을 확인한다**
   ```
   check_permissions()
   ```
   - `restrictions`에 deny된 명령이 열거된다. 함께 `read_memory_file(path="permissions.json")`로 설정을 직접 확인한다(레거시 환경에서는 `permissions.md`)

2. **대체 수단을 검토한다**
   - 차단된 명령과 동등한 작업을 허용된 도구로 실현할 수 없는지 생각한다
   - 예: `rm -rf`가 차단된 경우 개별 파일 삭제는 허용될 가능성이 있다

3. **권한 변경이 필요한 경우**
   - 상급자에게 차단 해제를 요청한다
   - 요청 시 「왜 그 명령이 필요한지」를 명시할 것

---

## 프롬프트가 축소되었다

### 증상

- 평소보다 시스템 프롬프트가 얇다, Priming(자동 회상)이 비어 있음에 가깝다
- 긴 대화나 큰 사용자 메시지 이후 응답 전에 프롬프트가 재구축된 듯한 동작이 있다

### 원인

크게 2개 층이 있다.

**1. Priming(자동 회상)의 티어** — `core/prompt/builder.py`의 `resolve_prompt_tier(context_window)`이 추정 컨텍스트 윈도우에서 티어를 결정한다. 윈도우의 해결 순서는 `core/prompt/context.py` `resolve_context_window`: **`~/.animaworks/models.json`(SSoT)** → 비권장의 `config.json` `model_context_windows` → `MODEL_CONTEXT_WINDOWS` 등의 코드 내 폴백 → 기본 128k.

| 티어 | 조건(`context_window`) | Priming의 처리(`core/agent/priming.py`) |
|--------|--------------------------|---------------------------------------------|
| full | **≥ 128_000** | compact의 일반 획득을 `priming.max_tokens`의 범위 내에서 정리하여 게재 |
| standard | **≥ 32_000 그리고 < 128_000** | 같은 compact 경로지만 획득 예산을 최대 1000 토큰으로 제한 |
| light | **≥ 16_000 그리고 < 32_000** | compact의 기본 컨텍스트를 획득하고 관련 지식·에피소드 검색을 억제 |
| minimal | **< 16_000** | compact의 기본 컨텍스트를 유지하고 관련 지식·에피소드 검색을 억제 |

하트비트／cron용 쿼리 문장은 최근의 `[REFLECTION]`을 activity_log에서 모은 텍스트가 된다(긴 템플릿 전문이 아님).

**2. 시스템 프롬프트 본체의 수축** — `core/agent/priming.py` `_fit_prompt_to_context_window`: 시스템＋사용자의 추정 토큰 + 도구 스키마 overhead가 **컨텍스트 윈도우의 약 80%**를 초과하면 `build_system_prompt`을 **시스템 예산 75% → 50% → 25%**로 단계적으로 줄여 재구축한다. **25% 이하의 단계**에서는 **Priming 블록과 인간용 알림 블록을 비운** 뒤 적용한다. 그래도 들어가지 않으면 시스템 프롬프트를 **바이트 단위로 하드 트렁케이트**한다.

### 대처 절차

1. **부족한 컨텍스트는 명시적으로 획득한다**: `search_memory` / `read_memory_file`로 조직·절차·공유 지식을 읽는다(특히 `minimal` / `light`에서는 Priming이 약함)
2. **작업 상태를 디스크에 남긴다**: `state/current_state.md`나 `shortterm/`에 요약을 써 두고 세션이 끊겨도 재개할 수 있게 한다
3. **상급자·관리자와 상담한다**: 실제 운영에서 비좁으면 `models.json`의 `context_window`이나 모델 변경을 검토한다

---

## 기타 흔한 문제

### 파일을 찾을 수 없다

- **원인**: 경로 지정 실수, 파일이 존재하지 않음
- **대처**: Mode S에서는 `Glob`, 그 외에는 `search_memory`이나 알려진 경로의 `read_memory_file`로 찾는다
- **주의**: `read_memory_file`은 Anima 디렉터리 상대(예: `knowledge/xxx.md`)에 더해 `common_knowledge/`·`reference/`·`common_skills/` 프리픽스로 공유 디렉터리를 읽을 수 있다. `Read`(에이전트 내장)은 별도 규칙으로 경로가 결정된다

### read_channel에서 inbox를 지정할 수 없다

- **원인**: `read_channel`은 Board의 공유 채널용. inbox(수신함)는 채널이 아니다
- **대처**: inbox의 메시지는 시스템이 자동 처리한다. `read_channel`에 `inbox`나 `inbox/`을 지정하면 오류가 된다

### 명령이 타임아웃된다

- **원인**: 처리 시간이 `timeout`을 초과했다
- **대처**: Bash 실행 시 `timeout` 파라미터를 늘린다(기본: 30초)
- **주의**: 장시간 실행하는 명령에는 적절한 타임아웃 값을 설정할 것

### 상대의 Anima가 존재하지 않는다

- **원인**: Anima 이름의 오기, 또는 그 Anima가 아직 생성되지 않음
- **대처**: 상급자에게 확인한다. 조직 구조는 `reference/organization/structure.md`을 참조한다

### venv 재구축 후 SEGV가 다발한다

- **원인**: ChromaDB나 PyTorch의 메이저 버전이 올라가면 기존 vectordb(HNSW 세그먼트)와 호환성이 없어져 읽기 시 SEGV(Segmentation Fault)가 발생한다. ChromaDB 1.5.x의 Rust 바인딩은 손상된 인덱스에 대해 Python 예외가 아닌 SEGV를 일으키는 알려진 문제가 있다(chromadb #6852, #6949, #6979)
- **대처**: venv를 재구축했으면 vectordb도 반드시 전체 재구축한다
  1. 서버를 종료한다
  2. `~/.animaworks/vectordb/`과 각 `~/.animaworks/animas/*/vectordb/`을 삭제한다(또는 백업 후 삭제)
  3. `~/.animaworks/index_meta.json`를 삭제한다
  4. 서버를 시작한다(시작 시 자동으로 재인덱스됨)
- **예방**: 패키지 업데이트 시 chromadb·torch·sentence-transformers의 버전 변경을 확인하고 변경이 있으면 vectordb를 재구축한다
