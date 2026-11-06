# 메시지 전송 완전 가이드

다른 Anima(사원)와 커뮤니케이션하기 위한 종합 가이드.
메시지 전송·수신·스레드 관리의 모든 절차를 망라한다.

## send_message 도구 — 파라미터 참조

메시지 전송에는 `send_message` 도구를 사용한다(권장).

### 매개변수 목록

| 매개변수 | 형식 | 필수 | 설명 |
|-----------|------|------|------|
| `to` | string | MUST | 수신 대상. 해결 규칙은 아래 「수신 대상 `to` 해결」을 참조. Anima 이름, `config.json`의 사람 별칭, `slack:USERID` / `chatwork:ROOMID`, Slack 사용자 ID 단독(`U` + 영숫자 8자 이상)을 사용할 수 있음 |
| `content` | string | MUST | 메시지 본문 |
| `intent` | string | MUST | 메시지의 의도. 허용값: `report`(진행 상황·결과 보고), `question`(질문·답변이 필요한 문의)만 해당. 작업 위임에는 `delegate_task` 도구를 사용할 것. acknowledgment·감사·FYI는 `animaworks-tool board post <channel> <text>`로 Board에 공유할 것 |
| `reply_to` | string | MAY | 답장 대상 메시지의 ID(예: `20260215_093000_123456`) |
| `thread_id` | string | MAY | 스레드 ID. 기존 스레드에 참여할 때 지정 |

### DM의 run 내 규칙

- 동일 수신처로의 DM은 run 1회당 **1통까지**. 중복 전송을 피하는 규칙이며, 수신처 수의 상한은 없음
- 동일 수신처에 추가로 공유할 경우 Board를 사용하거나, 내용에 따라 다음 run에서 전송

### 수신처 `to`의 해결(통합 아웃바운드)

`send_message`의 수신처는 `core/outbound.resolve_recipient`에서 다음 **우선순위**에 따라 해결된다(표기 변형에 강한 순서).

- 수신처 문자열의 **앞뒤 공백**은 해결 전에 트림된다.

| 순위 | 조건 | 결과 |
|------|------|------|
| 1 | 기존 Anima 디렉터리 이름과 **완전 일치**(대소문자 구분) | 사내 Inbox로 |
| 2 | `config.json`의 `external_messaging.user_aliases` 키와 **일치**(대소문자 무시) | 외부(Slack 또는 Chatwork)로. `preferred_channel`를 우선하고, 거기에 연락처가 없으면 설정된 다른 쪽으로 폴백 |
| 3 | `slack:`로 시작(대소문자 구분 안 함. 예: `slack:U0123456789`, `Slack:u06…`) | 콜론 이후를 트림하고, 사용자 ID는 **대문자로 정규화**한 뒤 Slack DM |
| 4 | `chatwork:`로 시작(프리픽스는 대소문자 구분 안 함) | 콜론 이후를 트림한 룸 ID로 Chatwork 게시(룸 ID의 대소문자는 그대로) |
| 5 | **Slack 사용자 ID 형식** 단독: 앞부분이 `U`이고, 그 뒤에 영숫자가 **8자 이상**(전체가 정규 표현식 `^U[A-Z0-9]{8,}$`에 일치) | Slack DM(프리픽스 없이 ID만 지정하려는 경우). **너무 짧은 문자열**(예: `U12345`)은 이 단계에서 매치되지 않고, 하위 규칙으로 진행 |
| 6 | 기존 Anima 이름과 **대소문자 무시하고 일치** | 사내 Inbox로(디스크상의 공식 이름으로 배달) |
| 7 | 위 어디에도 해당하지 않음 | 수신처 불명(`RecipientNotFoundError`. 낮은 레벨에서는 알려진 Anima 이름·별칭 이름이 메시지에 포함될 수 있음) |

**Anima 이름과 별칭의 충돌**: 어떤 문자열이 `user_aliases`의 키와 같아도, **기존 Anima의 디렉터리 이름과 완전 일치**하면 항상 **1번**이 우선되어 사내 Inbox로 도착한다(별칭은 사용되지 않음).

#### `send_message` 도구가 수신처 해결에 실패했을 때

`resolve_recipient`이 실패해도, 도구 결과에는 예외 전문이 그대로 반환되지 않는다. 세션 종류에 따른 **가이던스**가 반환된다(`core/tooling/handler_comms.py`).

- **인간과의 채팅 중**(`chat`): 해당 수신처에는 `send_message`할 수 없다는 점과, **텍스트로 직접 답변하면 인간 사용자에게 도달**하는 점, `send_message`는 다른 Anima용으로 사용할 점이 표시된다.
- **채팅 외**(하트비트·cron 등): 인간에게 연락은 **`call_human`**을 사용할 것, `send_message`은 다른 Anima용으로 사용할 것이 표시된다.

따라서 「Known animas: …」 같은 낮은 레벨 문구는 도구 사용자에게 나오지 않는 경우가 많다. 설정·별칭을 고치려면 `config.json`의 `external_messaging`를 확인한다.

#### 인간 별칭과 `preferred_channel`

`external_messaging.user_aliases`의 각 엔트리에는 `slack_user_id` 및/또는 `chatwork_room_id`을 설정한다. `external_messaging.preferred_channel`이 `slack`일 때는, Slack용 ID가 있으면 Slack을 선택한다. Slack ID가 비어 있고 Chatwork만 있으면 Chatwork로 폴백한다(`preferred_channel`이 `chatwork`일 때의 반대도 동일). **어느 ID도 없는** 별칭은 해결 시 오류가 된다.

#### 외부로 도달한 후의 배달(개요)

외부 루트로 해결된 경우, `send_message`은 사내 Messenger를 경유하지 않고 `core/outbound.send_external`에서 API로 전송한다.

- **시도 순서**: 먼저 해결 결과의 `channel`(`slack` 또는 `chatwork`)로 보내고, 예외 시에는 `_build_channel_order`에 따라, 상대방에게 Slack ID와 Chatwork 룸 ID **모두**가 있으면 다른 채널도 순서대로 시도한다.
- **Slack**: `core/outbound._send_via_slack`이 사용하는 토큰은 Vault／공유 크레덴셜의 **`SLACK_BOT_TOKEN__{送信元Anima名}`**(있으면). 본문은 `md_to_slack_mrkdwn`로 Slack용으로 정형화된다. `core/integrations._anima_icon_url`로 아이콘 URL이 해결되면 `post_message`에 전달된다.
  - **본문 앞부분의 `[送信者名]` 프리픽스**: **Anima 전용 Bot 토큰이 없는** 경우에만, 본문 앞부분에 `[{送信元Anima名}] `가 붙는다(누구의 문건인지 DM상에서 표시하기 위해). **토큰이 있는** 경우는 프리픽스를 붙이지 않고, `username`(Anima 이름)과 `icon_url`로 발신자를 나타낸다.
- **Chatwork**: 전송 Anima 자신의 identity 토큰(`CHATWORK_API_TOKEN__<Anima名>`) 경유로 룸에 게시. 발신원 Anima 이름이 있으면 본문 앞부분에 `[送信者名] `을 붙이고, `md_to_chatwork`로 정형화된다.

### 기본 전송 예

```
send_message(to="alice", content="レビュー完了しました。修正点は3箇所です。", intent="report")
```

### 답장 전송 예

수신 메시지의 `id`와 `thread_id`을 사용해 답장을 연결한다:

```
send_message(
    to="alice",
    content="了解しました。15時までに対応します。",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

`user_aliases`에 등록한 별칭(예: `user`)으로는, 사내 Anima와 동일하게 `send_message`로 보낼 수 있다(외부 채널로 라우팅된다).

```
send_message(to="user", content="対応完了しました。", intent="report")
```

### intent 구분

| intent | 용도 | 예 |
|--------|------|-----|
| `report` | 진행 상황·결과 보고 | 작업 완료 보고, 상급자에게 상황 보고 |
| `question` | 답변이 필요한 질문 | 불명확한 점 확인, 판단을 요청하는 문의 |

**주의**: 「알겠습니다」, 「감사합니다」와 같은 acknowledgment·감사·FYI는 DM으로 보낼 수 없다. `animaworks-tool board post <channel> <text>`로 Board에 공유할 것.

### Board와 DM 구분

| 용도 | 사용 도구 | 예 |
|------|-----------|-----|
| 진행 상황 보고·결과 보고 | send_message (intent=report) | 상급자에게 작업 완료 보고 |
| 작업 위임 | delegate_task | 직속 부하에게 하나의 지속 작업을 위임. 진행 상황은 task_tracker에서 확인 |
| 질문·문의 | send_message (intent=question) | 불명확한 점 확인 |
| 확인·감사·FYI | `animaworks-tool board post <channel> <text>` | 「알겠습니다」, 「공유했습니다」 |
| 팀 전체에 공유 | `animaworks-tool board post <channel> <text>` | 관계자에게 일괄 공지 |
| 동일 수신 대상에게 두 번째 메시지 | `animaworks-tool board post <channel> <text>` | 추가 정보 공유 |

## 스레드 관리

### 새 스레드 시작하기

`thread_id`을 생략하면, 시스템이 자동으로 메시지 ID를 스레드 ID로 설정한다.
새 주제를 시작할 경우 `thread_id`를 지정하지 말 것.

```
send_message(to="bob", content="新しいプロジェクトの件で相談があります。", intent="question")
# → thread_id は自動生成される（メッセージIDと同じ値）
```

### 기존 스레드에 답장하기

수신 메시지에 답장할 경우, MUST: `reply_to`와 `thread_id` **모두**를 지정한다.

```
# 受信メッセージ:
#   id: "20260215_093000_123456"
#   thread_id: "20260215_090000_000000"
#   content: "レビューお願いします"

send_message(
    to="alice",
    content="レビュー完了しました。",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

### 스레드 관리 규칙

- MUST: 같은 주제의 대화에서는 같은 `thread_id`을 계속 사용할 것
- MUST: 답장 시에는 원본 메시지의 `id`을 `reply_to`에 설정할 것
- SHOULD NOT: 다른 주제를 기존 스레드에 섞지 말 것. 새 주제는 새 스레드로 시작
- MAY: `thread_id`이 불명확하면 생략해도 됨(새 스레드로 취급됨)

## CLI로 메시지 전송

도구를 사용할 수 없거나 Bash 경유로 전송하는 경우의 방법.

### 기본 구문

```bash
animaworks send {送信者名} {宛先} "メッセージ内容" [--intent report|question] [--reply-to ID] [--thread-id ID]
```

### 구체적 예

```bash
# 基本送信（intent は省略可、CLI 経由では空でも送信可能）
animaworks send bob alice "作業完了しました。確認をお願いします。" --intent report

# スレッド返信
animaworks send bob alice "了解しました" --intent report --reply-to 20260215_093000_123456 --thread-id 20260215_090000_000000
```

### 주의 사항

- MUST: 메시지 내용은 더블쿼트로 감쌀 것
- SHOULD: send_message 도구를 사용할 수 있으면 도구를 우선할 것(CLI보다 확실)
- 메시지 안에 `"`을 포함할 경우 이스케이프 필요: `\"`

## 수신 메시지 확인 방법

### 자동 배달

메시지를 수신하면, 하트비트나 대화 시작 시 시스템이 자동으로 읽지 않은 메시지를 알림한다.
수동 확인은 보통 불필요.

### 수신 메시지의 구조

수신 메시지에는 다음 정보가 포함된다:

| 필드 | 설명 | 예 |
|-----------|------|-----|
| `id` | 메시지의 고유 식별자 | `20260215_093000_123456` |
| `thread_id` | 스레드 식별자 | `20260215_090000_000000` |
| `reply_to` | 답장 대상 메시지 ID | `20260215_085500_789012` |
| `from_person` | 발신자 이름 | `alice` |
| `to_person` | 수신자 이름(자신) | `bob` |
| `type` | 메시지 종류 | `message`(일반), `board_mention`(Board 멘션), `ack`(읽음 알림) |
| `content` | 메시지 본문 | `レビューお願いします` |
| `intent` | 발신자의 의도 | `report`, `question` |
| `timestamp` | 발신 일시 | `2026-02-15T09:30:00` |

### 답장 방침

- MUST: 질문·요청·대응이 필요한 읽지 않은 메시지에는 응답할 것
- MUST NOT: 단순한 이해·감사·칭찬에 대한 답장을 계속하지 말 것
- SHOULD: 수령 연락이 필요하면 「알겠습니다」만이 아니라, 다음 액션도 전할 것

## 외부 플랫폼에서의 메시지 수신

### 서버가 자동 수신한다

Slack이나 Chatwork 등의 외부 플랫폼에서 온 메시지는, **AnimaWorks 서버가 상시 수신하여 대상 Anima의 Inbox에 자동 배달한다**. Anima 자신이 WebSocket 연결을 유지하거나 API를 폴링할 필요는 없다.

서버는 다음 방식으로 메시지를 수신한다(관리자가 설정):

- **Socket Mode**: Slack WebSocket 경유로 실시간 수신
- **Webhook**: Slack Events API / Chatwork Webhook 경유로 수신

어느 방식이든, 메시지는 Inbox에 같은 형식으로 배달된다.

### 외부 메시지의 식별

외부 플랫폼에서 도착한 메시지는 일반 Anima 간 메시지와 다음 점이 다르다:

| 필드 | Anima 간 DM | 외부 메시지 |
|-----------|------------|--------------|
| `source` | `"anima"` | `"slack"`, `"chatwork"` 등 |
| `from_person` | Anima 이름(예: `alice`) | `"slack:U12345..."` 형식 |

### 외부 메시지가 도착하는 경우

1. **인간으로부터의 DM**: Slack/Chatwork 에서 인간이 Anima에게 메시지를 전송한 경우
2. **call_human에 대한 답장**: `call_human` 에서 보낸 알림의 Slack 스레드에 인간이 답장한 경우 (상세: `communication/call-human-guide.md`)
3. **채널 경유 멘션**: Slack 채널에서 Anima에게 보내는 메시지가 게시된 경우

### Slack 메시지의 Inbox 처리

Slack 메시지가 Inbox에 기록되면 파일 변경 알림으로 Inbox 처리가 시작된다. 시작 시 intent 필터는 없으며, 멘션 여부로 처리 시작을 지연시키지 않는다. 파일 알림을 놓친 경우 45초마다 재확인이 이를 보완한다.

| 조건 | Inbox 처리 | 응답 방침 |
|------|------------|----------|
| **@멘션 포함** (Bot이 mention된 경우) | 파일 알림으로 시작 | `intent="question"` 이 자동 부여됨. 내용에 따라 대응 |
| **DM** (Bot에게 직접 메시지) | 파일 알림으로 시작 | `intent="question"` 이 자동 부여됨. 내용에 따라 대응 |
| **멘션 없는 채널 메시지** | 파일 알림으로 시작 | 자신에게 온 메시지가 아니면 상황 파악에 그치고, 답장이나 액션은 필요 없는 경우가 있음 |

시작했다고 반드시 답장하는 것은 아니다. 단순한 이해·감사·칭찬에는 답장하지 않고, 추가 질문·요청·새 정보가 있는 경우에만 대응한다.

### 외부 메시지에 대한 응답

외부 메시지를 수신하면:

- `call_human` 에 대한 답장인 경우: 채팅 응답 또는 `call_human` 로 응답
- 인간으로부터의 DM인 경우: 채팅(Web UI)으로 답하거나, 등록된 인간 별칭으로 `send_message` 하거나, 상대의 Slack 사용자 ID를 알면 `to="slack:U0123456789"` (또는 ID만, 구현이 허용하는 형식일 때)으로 **`send_message` 에 의한 답장 가능** (위의 "수신처 `to` 의 해결"을 충족할 것)
- 출처가 불명확한 경우: 메시지의 `source` 와 `from_person` 을 확인하고, 필요에 따라 상급자에게 보고

## 메시지 본문의 베스트 프랙티스

### 좋은 메시지 작성법

1. **결론을 먼저 쓴다**: 상대가 첫 줄에서 요점을 파악할 수 있게 한다
2. **구체적으로 쓴다**: 모호한 표현을 피하고, 수치·마감일·대상을 명시한다
3. **액션을 명시한다**: 상대에게 무엇을 해주길 바라는지 분명히 한다
4. **답변 필요 여부를 명기한다**: 답변이 필요하면 "답변 부탁드립니다"라고 쓴다

### 좋은 예와 나쁜 예

**나쁜 예:**
```
データの件、確認しておいてください。
```

**좋은 예:**
```
売上データ（2026年1月分）のバリデーションチェックをお願いします。
対象ファイル: /shared/data/sales_202601.csv
確認観点: 欠損値の有無と金額フィールドの異常値
期限: 本日15時まで
結果は返答をお願いします。
```

### 긴 내용 전달 방법

- SHOULD: 본문이 500자를 초과하는 경우, 내용을 파일로 작성하고 메시지에는 파일 경로와 요약만 기재한다
- MUST: 파일을 참조하는 경우, 상대가 접근 가능한 경로에 배치할 것

```
デプロイ手順書を作成しました。
ファイル: ~/.animaworks/shared/docs/deploy-procedure-v2.md

要約: ステージング環境での確認ステップを3つ追加しました（セクション4.2参照）。
レビューをお願いします。返答をお願いします。
```

## 자주 있는 실패와 대책

### intent 지정 오류

**증상**: `Error: DMのintentは 'report', 'question' のみ許可されています`로 표시됨

**원인**: `intent`을 생략했거나 acknowledgment·감사·FYI를 DM으로 보내려고 함

**대책**: DM에서는 반드시 `intent`에 `report` 또는 `question`를 지정할 것. 작업 위임에는 `delegate_task`를 사용할 것. 확인·감사·FYI에는 `animaworks-tool board post <channel> <text>`을 사용할 것

**증상**: `intent='delegation' は廃止されました` 등으로 표시됨

**원인**: 기존 `send_message(..., intent="delegation")`을 사용하고 있음

**대책**: 작업 위임에는 **`delegate_task`**만 사용할 것. `send_message`의 `intent`은 `report` / `question`만 해당

### 수신처 이름 오류

**증상**: 수신처 불명 오류가 발생하거나, 의도하지 않은 상대에게 도달함

**원인**: `to` 이 해결 규칙에 맞지 않음 (Anima 이름의 철자, `user_aliases` 미등록, Slack/Chatwork 의 ID 미설정 등)

**대책**:

- 가장 빠르게 전달하고 싶은 Anima 이름은 **디렉터리 이름과 같은 표기**(대문자·소문자 포함)로 지정한다. 다른 표기라도 대문자·소문자만 다른 경우 6번째 규칙으로 사내 배포에 매칭된다
- **Anima 이름과 동일한 별칭**이 있는 경우, **완전 일치하는 Anima 디렉터리 이름**이 있으면 항상 사내가 우선된다 (의도와 반대라면 Anima 이름의 리네임이나 별칭 이름 변경이 필요)
- 인간에게는 `external_messaging.user_aliases` 에 별칭과 `slack_user_id` / `chatwork_room_id` 를 설정하고, `preferred_channel` 를 확인한다
- 알려진 Slack 사용자 ID를 알면 `slack:USERID` 또는 ID 단독(`U` 뒤에 영숫자 **8자 이상**)을 `to` 로 한다. **짧은 ID**(예: `U12345`)는 Slack 형식으로 인정되지 않아 수신처 불명으로 취급될 수 있다
- 도구가 "채팅에서는 직접 답변", "call_human을 사용하라" 등의 힌트만 반환한 경우, 위의 "`send_message` 도구가 수신처 해결에 실패했을 때"를 참조하고, `config.json` 의 `external_messaging` 와 세션 종류를 확인한다
- 불명확한 경우 `search_memory(query="メンバー", scope="knowledge")` 등으로 조직 정보를 확인한다

### 스레드 단절

**증상**: 답장했는데 상대쪽에서 대화 흐름이 보이지 않음

**원인**: `reply_to` 이나 `thread_id` 을 지정하는 것을 잊음

**대책**: 답장 시 MUST: 원본 메시지의 `id` 를 `reply_to` 에, `thread_id` 를 그대로 `thread_id` 에 설정한다

### 메시지가 너무 김

**증상**: 상대가 요점을 파악하지 못함

**대책**: 결론을 맨 앞에 두고, 상세는 파일로 분리한다. 메시지 본문은 요약+파일 참조 형식으로 한다

### 동일 수신 대상에게 두 번째 전송

**증상**: `Error: このrunで既に {to} にメッセージを送信済みです`로 표시됨

**원인**: 한 번의 run에서 동일 수신 대상에게 send_message를 두 번 이상 호출함

**대책**: 추가 연락에는 `animaworks-tool board post <channel> <text>`을 사용할 것. 또는 다음 run(하트비트 등)에서 전송할 것

### 답장을 잊음

**증상**: 상대가 대응 상황을 파악하지 못해 다시 문의가 옴

**대책**: 대응이 필요한 메시지에 답장한다. 바로 대응할 수 없으면 "확인했습니다. XX시까지 대응하겠습니다"라고 예상을 전한다. 단순한 이해·감사에는 답장하지 않는다

## 전송 규칙

- `send_message`의 intent는 `report` 또는 `question`이다. 작업 위임에는 `delegate_task`을 사용할 것.
- 동일 run에서 같은 수신 대상에게 보내는 DM은 1건까지다. 수신 대상 수에는 제한이 없다.
- Board 게시물의 cooldown이나 DM / Board 공통 전송 예산은 없다.
- 쌍별 대화 깊이에 따른 전송 거부는 없다. 내부 Anima 간의 깊이는 진단 목적으로 로그에 기록될 수 있지만, 전송 본문은 폐기되지 않는다.

## 답장 방침

대응이 필요한 메시지에는 답장하고, 수령 연락이 필요하면 다음 행동이나 예상을 덧붙여 한 번만 전한다. 단순한 이해·감사·칭찬에는 답장하지 않고, 응답을 계속하지 않는다.

## 대화를 간결하게 하기 위한 권장

- 하나의 주제에 필요한 정보를 모아, 추가 확인이 필요 없는 형태로 전한다.
- 여러 보고 사항은 가능하면 한 통으로 모으고, 전체 공유에는 Board를 사용한다.
- 이야기를 계속해야 할 때는 내용에 따라 DM / Board를 선택한다. 왕복 횟수에 따른 시스템상의 거부는 없다.

## 커뮤니케이션 경로 규칙

메시지의 수신처는 조직 구조에 따른다:

| 상황 | 수신처 | 예 |
|------|------|-----|
| 중요한 진행 상황·문제 보고 | 상급자 | `send_message(to="manager", content="タスクA完了", intent="report")` |
| 작업 지시·위임 | 부하 | `delegate_task(name="worker", instruction="レポート作成をお願い")` |
| 동료와의 협력 | 동료(같은 상급자) | `send_message(to="peer", content="レビューお願い", intent="question")` |
| 타 부서로의 연락 | 자신의 상급자 경유 | `send_message(to="manager", content="開発部のXさんに確認してほしい件が...", intent="question")` |

- MUST: 타 부서 구성원에게 직접 연락하지 말 것. 자신의 상급자 또는 상대의 상급자를 경유한다
- MAY: 동료(같은 상급자를 가진 구성원)와는 직접 소통해도 된다

## 블로커 보고 (MUST)

작업 실행 중 다음 상황이 발생한 경우, 즉시 의뢰자에게 `send_message` 로 보고할 것.
"대기" 상태로 방치해서는 안 된다.

- 파일/디렉터리를 찾을 수 없음
- 권한 부족으로 접근할 수 없음
- 전제 조건이 충족되지 않음
- 기술적 문제로 작업이 중단됨
- 지시 내용이 불명확하여 판단할 수 없음

보고처: 의뢰자 (send_message)
중대 블로커(30분 이상의 지연이 예상되는 경우): 인간에게도 `call_human` 로 알림

### 블로커 보고의 예

```
send_message(
    to="manager",
    content="""【ブロッカー報告】データ集計タスク

状況: 指定されたファイル /shared/data/sales_202601.csv が存在しません。
影響: 集計作業を開始できません。
必要なアクション: ファイルパスの確認、またはファイルの配置をお願いします。""",
    intent="report"
)
```

## 요청 메시지의 필수 요소 (MUST)

다른 Anima에게 작업을 요청할 때, 다음 5가지 요소를 반드시 포함할 것:

1. **목적** (왜 이 작업이 필요한가)
2. **대상** (파일 경로, 리소스)
3. **기대 성과** (무엇이 완료되면 완료인가)
4. **마감일**
5. **완료 보고 필요 여부**

이것들이 부족한 메시지는 수신측이 확인을 위해 답장해야 하므로, 비효율적인 왕복이 발생한다.
