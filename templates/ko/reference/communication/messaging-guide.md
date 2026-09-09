# 메시지 전송 완전 가이드

다른 Anima (동료)와 소통하기 위한 포괄적 가이드입니다.
메시지의 전송, 수신, 스레드 관리의 전 절차를 다룹니다.

## send_message 도구 — 파라미터 레퍼런스

메시지 전송에는 `send_message` 도구를 사용합니다 (권장).

### 파라미터 목록

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `to` | string | MUST | 수신자. Anima 이름 (예: `alice`) 또는 사람 별명 (예: `user`, `taka`). 사람 별명은 외부 channel (Slack/Chatwork) 경유로 전달됨 |
| `content` | string | MUST | 메시지 본문 |
| `intent` | string | MUST | 메시지 의도. 허용값: `report` (진행/결과 보고), `question` (질문/응답이 필요한 문의)만 가능. 태스크 위임에는 `delegate_task` 사용. 확인/감사/FYI는 Board (post_channel) 사용 |
| `reply_to` | string | MAY | 답장 대상 메시지 ID (예: `20260215_093000_123456`) |
| `thread_id` | string | MAY | 스레드 ID. 기존 스레드에 참여할 때 지정 |

### DM 제한 (1회 run당)

- 최대 **2명**까지 전송 가능
- 동일 수신자에게 **2번째 전송 불가** (추가 연락은 Board 사용)
- 3명 이상에게 전달 시 Board (post_channel) 사용

### 기본 전송 예시

```
send_message(to="alice", content="리뷰 완료했습니다. 수정 사항은 3건입니다.", intent="report")
```

### 답장 예시

수신 메시지의 `id`와 `thread_id`를 사용하여 답장을 연결합니다:

```
send_message(
    to="alice",
    content="확인했습니다. 15시까지 대응하겠습니다.",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

### intent 사용 구분

| intent | 용도 | 예시 |
|--------|------|------|
| `report` | 진행/결과 보고 | 태스크 완료 보고, 상사에게 상황 보고 |
| `question` | 응답이 필요한 질문 | 불명확한 점 확인, 판단을 구하는 문의 |

**참고**: "알겠습니다", "감사합니다" 등의 확인/감사/FYI는 DM으로 보낼 수 없습니다. Board (post_channel)를 사용하세요.

### Board와 DM 사용 구분

| 용도 | 사용 도구 | 예시 |
|------|----------|------|
| 진행/결과 보고 | send_message (intent=report) | 상사에게 태스크 완료 보고 |
| 태스크 위임 | delegate_task | 직속 부하에게 하나의 영속 태스크를 위임; task_tracker로 진행 확인 |
| 질문/문의 | send_message (intent=question) | 불명확한 점 확인 |
| 확인/감사/FYI | post_channel (Board) | "확인했습니다", "공유합니다" |
| 3명 이상에게 전달 | post_channel (Board) | 팀 전체 공지 |
| 동일 수신자 2번째 전송 | post_channel (Board) | 추가 정보 공유 |

## 스레드 관리

### 새 스레드 시작

`thread_id`를 생략하면 시스템이 자동으로 메시지 ID를 스레드 ID로 설정합니다.
새로운 주제를 시작할 때는 `thread_id`를 지정하지 마세요.

```
send_message(to="bob", content="새 프로젝트 건으로 상담이 있습니다.", intent="question")
# → thread_id는 자동 생성됨 (메시지 ID와 동일)
```

### 기존 스레드에 답장

수신 메시지에 답장할 때, MUST: `reply_to`와 `thread_id` 모두 지정하세요.

```
# 수신 메시지:
#   id: "20260215_093000_123456"
#   thread_id: "20260215_090000_000000"
#   content: "리뷰 부탁합니다"

send_message(
    to="alice",
    content="리뷰 완료했습니다.",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

### 스레드 관리 규칙

- MUST: 같은 주제의 대화에서는 동일 `thread_id`를 계속 사용
- MUST: 답장 시 원본 메시지의 `id`를 `reply_to`에 설정
- SHOULD NOT: 다른 주제를 기존 스레드에 섞지 말 것. 새 주제는 새 스레드로 시작
- MAY: `thread_id`를 모르는 경우 생략 가능 (새 스레드로 취급됨)

## CLI를 통한 메시지 전송

도구를 사용할 수 없거나 Bash를 통해 전송하는 경우의 방법입니다.

### 기본 구문

```bash
animaworks send {발신자명} {수신자} "메시지 내용" [--intent report|question] [--reply-to ID] [--thread-id ID]
```

### 예시

```bash
# 기본 전송 (intent는 생략 가능, CLI에서는 빈 값으로도 전송 가능)
animaworks send bob alice "작업 완료했습니다. 확인 부탁합니다." --intent report

# 스레드 답장
animaworks send bob alice "확인했습니다" --intent report --reply-to 20260215_093000_123456 --thread-id 20260215_090000_000000
```

### 주의 사항

- MUST: 메시지 내용을 쌍따옴표로 감쌀 것
- SHOULD: send_message 도구가 사용 가능한 경우 도구를 우선 사용 (CLI보다 확실)
- 메시지 내 `"`를 포함할 경우 이스케이프 필요: `\"`

## 수신 메시지 확인 방법

### 자동 배달

메시지를 수신하면 heartbeat이나 대화 시작 시 시스템이 자동으로 미읽 메시지를 알립니다.
수동 확인은 보통 불필요합니다.

### 수신 메시지 구조

수신 메시지에는 다음 정보가 포함됩니다:

| 필드 | 설명 | 예시 |
|------|------|------|
| `id` | 메시지 고유 식별자 | `20260215_093000_123456` |
| `thread_id` | 스레드 식별자 | `20260215_090000_000000` |
| `reply_to` | 답장 대상 메시지 ID | `20260215_085500_789012` |
| `from_person` | 발신자명 | `alice` |
| `to_person` | 수신자명 (자신) | `bob` |
| `type` | 메시지 유형 | `message` (일반), `board_mention` (Board 멘션), `ack` (읽음 확인) |
| `content` | 메시지 본문 | `리뷰 부탁합니다` |
| `intent` | 발신자의 의도 | `report`, `question` |
| `timestamp` | 전송 일시 | `2026-02-15T09:30:00` |

### 답장 의무

- MUST: 미읽 메시지를 받으면 발신자에게 답장
- MUST: 질문과 의뢰에는 반드시 응답
- SHOULD: "알겠습니다"만이 아닌 다음 액션도 전달

## 외부 플랫폼에서의 메시지 수신

### 서버가 자동 수신

Slack이나 Chatwork 등 외부 플랫폼의 메시지는 **AnimaWorks 서버가 상시 수신하여 대상 Anima의 Inbox에 자동 배달합니다**. Anima가 직접 WebSocket 연결을 유지하거나 API를 폴링할 필요는 없습니다.

서버는 다음 방식으로 메시지를 수신합니다 (관리자가 설정):

- **Socket Mode**: Slack WebSocket 경유 실시간 수신
- **Webhook**: Slack Events API / Chatwork Webhook 경유 수신

어떤 방식이든 메시지는 Inbox에 동일한 형식으로 배달됩니다.

### 외부 메시지 식별

외부 플랫폼에서 온 메시지는 Anima 간 메시지와 다음 점이 다릅니다:

| 필드 | Anima 간 DM | 외부 메시지 |
|------|------------|-----------|
| `source` | `"anima"` | `"slack"`, `"chatwork"` 등 |
| `from_person` | Anima 이름 (예: `alice`) | `"slack:U12345..."` 형식 |

### 외부 메시지가 도착하는 경우

1. **사람의 DM**: Slack/Chatwork에서 사람이 Anima에게 메시지를 보낸 경우
2. **call_human 답장**: `call_human`으로 보낸 알림의 Slack 스레드에 사람이 답장한 경우 (상세: `communication/call-human-guide.md`)
3. **Channel 멘션**: Slack channel에서 Anima를 대상으로 메시지가 게시된 경우

### Slack 메시지의 즉시 처리와 지연 처리

Slack 메시지는 내용에 따라 즉시 처리되거나 정기 heartbeat까지 대기합니다:

| 조건 | 처리 시점 | 이유 |
|------|----------|------|
| **@멘션** (Bot이 mention된 경우) | **즉시 처리** | `intent="question"`이 자동 부여, actionable로서 즉시 inbox 처리 실행 |
| **DM** (Bot에 직접 메시지) | **즉시 처리** | DM은 Bot을 대상으로 하므로 `intent="question"`이 자동 부여 |
| **멘션 없는 channel 메시지** | **다음 heartbeat에서 처리** | `intent`가 비어 있어 즉시 트리거되지 않고, 정기 순찰 시 미읽으로 처리 |

이를 통해 channel 내 일상 대화에서는 Anima가 매번 기동하지 않고, @멘션이나 DM으로 명시적으로 호출한 경우에만 빠르게 반응합니다.

### 외부 메시지 응답

외부 메시지를 받은 경우:

- `call_human` 답장인 경우: 채팅 응답 또는 `call_human`으로 응답
- 사람의 DM인 경우: 채팅 응답으로 대응 (`send_message`는 Anima 간 메시지용이므로 외부 플랫폼 사용자에게 직접 전달되지 않을 수 있음)
- 알 수 없는 발신자인 경우: 메시지의 `source`와 `from_person`을 확인하고, 필요하면 상사에게 보고

## 메시지 본문 모범 사례

### 좋은 메시지 작성법

1. **결론을 먼저 작성**: 상대방이 첫 줄에서 요점을 파악할 수 있게
2. **구체적으로 작성**: 모호한 표현을 피하고 수치, 기한, 대상을 명시
3. **액션을 명시**: 상대방에게 무엇을 해주길 원하는지 명확하게
4. **답장 필요 여부 명시**: 답장이 필요하면 "답장 부탁합니다"라고 작성

### 좋은 예와 나쁜 예

**나쁜 예:**
```
데이터 건, 확인 부탁합니다.
```

**좋은 예:**
```
매출 데이터 (2026년 1월분) 검증 체크를 부탁합니다.
대상 파일: /shared/data/sales_202601.csv
확인 관점: 결측값 유무와 금액 필드 이상값
기한: 오늘 15시까지
결과를 답장 부탁합니다.
```

### 긴 내용 전달 방법

- SHOULD: 본문이 500자를 넘으면 내용을 파일에 작성하고, 메시지에는 파일 경로와 요약만 기재
- MUST: 파일을 참조할 때 상대방이 접근 가능한 경로에 배치

```
배포 절차서를 작성했습니다.
파일: ~/.animaworks/shared/docs/deploy-procedure-v2.md

요약: 스테이징 환경 확인 스텝을 3개 추가했습니다 (섹션 4.2 참조).
리뷰 부탁합니다. 답장 부탁합니다.
```

## 자주 발생하는 문제와 대책

### intent 지정 오류

**증상**: `Error: DM intent must be 'report' or 'question' only`

**원인**: `intent`를 생략했거나, 확인/감사/FYI를 DM으로 보내려 했음

**대책**: DM에서는 반드시 `intent`에 `report` 또는 `question`을 지정. 태스크 위임은 `delegate_task` 사용. 확인/감사/FYI는 Board (post_channel) 사용

### 수신자 이름 오류

**증상**: 전송해도 상대방에게 도착하지 않음

**원인**: `to` 파라미터의 Anima 이름이 부정확하거나, 사람 별명이 config.json에 미등록

**대책**: Anima 이름은 대소문자를 구분함. 사람에게 전송 시 `config.json`의 `external_messaging.user_aliases`에 별명을 등록. 불확실하면 `search_memory(query="멤버", scope="knowledge")`로 조직 멤버 확인

### 스레드 단절

**증상**: 답장했는데 상대방 측에서 대화 흐름이 보이지 않음

**원인**: `reply_to`나 `thread_id`를 지정하지 않았음

**대책**: 답장 시 MUST: 원본 메시지의 `id`를 `reply_to`에, `thread_id`를 그대로 `thread_id`에 설정

### 메시지가 너무 길다

**증상**: 상대방이 요점을 파악하지 못함

**대책**: 결론을 맨 앞에 놓고, 상세 내용은 파일에 분리. 메시지 본문은 요약 + 파일 참조 형식으로

### 동일 수신자 2번째 전송

**증상**: `Error: Already sent a message to {to} in this run`

**원인**: 1회 run에서 같은 수신자에게 2회 이상 send_message를 호출

**대책**: 추가 연락은 Board (post_channel)를 사용. 또는 다음 run (heartbeat 등)에서 전송

### 3명 이상에게 전송

**증상**: `Error: Maximum 2 recipients per run for DMs`

**대책**: 3명 이상에게 전달 시 Board (post_channel) 사용

### 답장 누락

**증상**: 상대방이 대응 상황을 파악하지 못하고 재문의가 옴

**대책**: 수신 메시지에는 MUST: 반드시 답장. 즉시 대응할 수 없어도 "확인했습니다. XX시까지 대응하겠습니다"라고 답장

## 전송 제한

메시지 전송에는 시스템 전체 레이트 제한이 적용됩니다.
과도한 전송은 루프와 장애의 원인이 되므로 아래 제한을 이해하고 행동하세요.

### 글로벌 전송 제한 (activity_log 기반)

| 제한 | 기본값 | 대상 |
|------|--------|------|
| 시간당 상한 | 30통/시 | DM (message_sent) 카운트 |
| 일당 상한 | 100통/일 | DM (message_sent) 카운트 |

제한에 도달하면 전송이 오류가 됩니다. `ack`, `error`, `system_alert` 타입 메시지는 제한 대상 외.
값은 `config.json`의 `heartbeat.max_messages_per_hour` / `heartbeat.max_messages_per_day`로 변경 가능.

### 1회 run당 제한

- **DM**: 최대 2명까지, 동일 수신자에게 1통만
- **Board**: 동일 channel 게시 1회까지 (쿨다운 포함)

### 캐스케이드 감지 (2자 간 왕복 제한)

같은 상대와 짧은 시간 내 왕복이 너무 많으면 전송이 차단됩니다.
`config.json`의 `heartbeat.depth_window_s` (시간 윈도우)와 `heartbeat.max_depth` (최대 깊이)로 제어.

### 제한에 도달한 경우 대처

1. 제한은 activity_log의 슬라이딩 윈도우로 계산됨
2. 시간 제한 도달: 전송 내용을 current_state.md에 기록하고 다음 세션에서 전송
3. 일 제한 도달: 정말 필요한 메시지만으로 줄이고 다음 날까지 대기
4. 긴급 연락이 필요한 경우 `call_human` 사용 (레이트 제한 대상 외)

### 전송량 절약 모범 사례

- 여러 보고 사항을 1통의 메시지로 통합
- 확인/감사/FYI는 Board에 게시 (DM 할당량 절약)
- 정기적인 정보 공유는 Board channel 게시로 통합

## 1라운드 규칙

DM (`send_message`)의 주고받기는 **1주제 1왕복**을 원칙으로 합니다.

### 규칙

- MUST: 1개 주제에 대해 전송과 답장의 1왕복으로 완결
- MUST: 3왕복 이상의 주고받기가 필요해지면 Board channel로 이동
- SHOULD: 첫 메시지에 필요한 정보를 모두 포함하여 추가 질문이 필요 없는 형태로

### 1라운드 규칙이 필요한 이유

- DM 왕복이 늘면 레이트 제한에 빨리 도달
- 2자 간 메시지 루프는 **캐스케이드 감지**로 억제됨 (설정 가능한 시간 윈도우 내에서 최대 깊이를 넘으면 전송 차단)
- Board 게시는 다른 멤버도 참조할 수 있어 정보 중복 방지

### 예외

- 긴급 차단 요소 보고는 횟수 제한 대상 외

## 커뮤니케이션 경로 규칙

메시지 수신자는 조직 구조에 따릅니다:

| 상황 | 수신자 | 예시 |
|------|--------|------|
| 중요 진행/문제 보고 | 상사 | `send_message(to="manager", content="태스크A 완료", intent="report")` |
| 태스크 지시/위임 | 부하 | `delegate_task(name="worker", instruction="보고서 작성 부탁합니다", deadline="1d")` |
| 동료와의 연계 | 동료 (같은 상사) | `send_message(to="peer", content="리뷰 부탁합니다", intent="question")` |
| 다른 부서에 연락 | 자신의 상사를 경유 | `send_message(to="manager", content="개발팀 X님에게 확인 부탁드릴 건이...", intent="question")` |

- MUST: 다른 부서 멤버에게 직접 연락하지 말 것. 자신의 상사 또는 상대의 상사를 경유
- MAY: 동료 (같은 상사를 둔 멤버)와는 직접 소통 가능

## 차단 요소 보고 (MUST)

태스크 실행 중 다음 상황이 발생하면, 즉시 의뢰자에게 `send_message`로 보고하세요.
"대기" 상태로 방치하면 안 됩니다.

- 파일/디렉토리를 찾을 수 없음
- 권한 부족으로 접근 불가
- 전제 조건이 충족되지 않음
- 기술적 문제로 작업이 중단됨
- 지시 내용이 불명확하여 판단할 수 없음

보고 대상: 의뢰자 (send_message)
중대 차단 요소 (30분 이상 지연 예상): 사람에게도 `call_human`으로 통지

### 차단 요소 보고 예시

```
send_message(
    to="manager",
    content="""[차단 요소 보고] 데이터 집계 태스크

상황: 지정된 파일 /shared/data/sales_202601.csv가 존재하지 않습니다.
영향: 집계 작업을 시작할 수 없습니다.
필요한 조치: 파일 경로 확인 또는 파일 배치를 부탁합니다.""",
    intent="report"
)
```

## 의뢰 메시지 필수 요소 (MUST)

다른 Anima에게 태스크를 의뢰할 때 다음 5가지 요소를 반드시 포함하세요:

1. **목적** (왜 이 작업이 필요한지)
2. **대상** (파일 경로, 리소스)
3. **기대 성과** (무엇이 완료된 상태인지)
4. **기한**
5. **완료 보고 필요 여부**

이들이 부족한 메시지는 수신자가 확인을 위해 답장해야 하여 비효율적인 왕복이 발생합니다.
