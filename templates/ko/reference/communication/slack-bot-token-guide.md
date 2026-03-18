# Slack 봇 토큰 설정 가이드

Slack 연동에서 사용하는 봇 토큰의 구조와 Per-Anima(개별) 토큰의 설정 규칙입니다.

## 2종류의 봇 토큰

AnimaWorks의 Slack 연동에는 **공유 봇**과 **Per-Anima 봇** 2종류가 있습니다.

| 종류 | 키 이름 | 용도 |
|------|---------|------|
| 공유 봇 | `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | 전체 폴백용. Per-Anima 토큰이 미설정인 Anima가 사용 |
| Per-Anima 봇 | `SLACK_BOT_TOKEN__<name>` / `SLACK_APP_TOKEN__<name>` | 특정 Anima 전용 Slack App. 해당 Anima만 사용 |

**Per-Anima 봇이 설정되어 있으면 그쪽이 우선됩니다.** 공유 봇은 어디까지나 폴백입니다.

## Per-Anima 토큰 명명 규칙

`__`(언더스코어 2개) + Anima 이름(소문자)을 접미사로 추가합니다.

```
SLACK_BOT_TOKEN__sumire    ← sumire 전용 Bot User OAuth Token
SLACK_APP_TOKEN__sumire    ← sumire 전용 App-Level Token
```

## 저장 위치

`shared/credentials.json`에 저장합니다.

```json
{
  "SLACK_BOT_TOKEN": "xoxb-...(공유 봇)",
  "SLACK_APP_TOKEN": "xapp-...(공유 App)",
  "SLACK_BOT_TOKEN__sumire": "xoxb-...(sumire 전용 봇)",
  "SLACK_APP_TOKEN__sumire": "xapp-...(sumire 전용 App)"
}
```

## 반드시 지켜야 할 규칙

### 공유 토큰을 덮어쓰면 안 됩니다

**MUST**: Per-Anima 토큰을 설정할 때는 **새로운 키를 추가**하세요. 기존의 `SLACK_BOT_TOKEN`이나 `SLACK_APP_TOKEN`을 자신의 토큰으로 교체하면 안 됩니다.

공유 토큰은 다른 Anima와 시스템 전체가 사용합니다. 덮어쓰면:

- 다른 Anima의 Slack 통신이 깨짐
- Socket Mode 연결과 Bot Token의 App이 불일치
- `not_in_channel` 등의 예기치 않은 오류 발생

### 파일 편집으로 토큰을 추가하는 방법

```bash
# credentials.json을 읽고, 새로운 키를 추가하여 다시 저장
python3 -c "
import json
from pathlib import Path
p = Path.home() / '.animaworks/shared/credentials.json'
d = json.loads(p.read_text())
d['SLACK_BOT_TOKEN__<자신의_이름>'] = 'xoxb-...'
d['SLACK_APP_TOKEN__<자신의_이름>'] = 'xapp-...'
p.write_text(json.dumps(d, indent=2))
"
```

**주의**: `str_replace` 등으로 기존 행을 교체하지 말고, JSON에 키를 추가하는 방식을 사용하세요.

## 서버 검출과 재시작

Per-Anima 토큰은 **서버 기동 시**에 검출됩니다. `shared/credentials.json`에 토큰을 추가한 후 **서버 재시작이 필요합니다**.

서버는 기동 시에 다음을 수행합니다:

1. `SLACK_BOT_TOKEN__*`과 `SLACK_APP_TOKEN__*` 쌍을 검출
2. 각 쌍에 대해 Per-Anima Socket Mode 핸들러를 등록
3. `auth.test`로 Bot User ID를 취득하여 채널 라우팅에 사용

재시작 후 서버 로그에 다음과 같이 표시되면 성공입니다:

```
Per-anima Slack bot registered: <name> (bot_uid=U...)
```

## 트러블슈팅

### not_in_channel 오류

**증상**: Slack 채널에 응답하려 할 때 `not_in_channel` 오류 발생

**원인**: Per-Anima 토큰이 미설정이어서 공유 봇이 사용되고 있음. 공유 봇이 해당 채널의 멤버가 아님.

**조치**:
1. `shared/credentials.json`에 `SLACK_BOT_TOKEN__<name>`과 `SLACK_APP_TOKEN__<name>`을 추가
2. 서버를 재시작
3. 대상 채널에 Per-Anima 봇이 초대되어 있는지 확인

### 공유 봇으로 폴백되고 있는 경우

**증상**: 자신 전용 봇이 있어야 하는데 공유 봇 이름으로 게시됨

**원인**: `credentials.json`에 `SLACK_BOT_TOKEN__<name>` / `SLACK_APP_TOKEN__<name>`이 존재하지 않음

**확인 방법**:
```bash
cat ~/.animaworks/shared/credentials.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
for k in sorted(d):
    if 'SLACK' in k:
        print(f'{k}: {d[k][:20]}...')
"
```

### 토큰 취득 방법

관리자(인간)에게 Slack App의 Bot User OAuth Token (`xoxb-`)과 App-Level Token (`xapp-`)을 제공받으세요.

- Bot Token: Slack App 관리 화면 → OAuth & Permissions → Bot User OAuth Token
- App-Level Token: Slack App 관리 화면 → Basic Information → App-Level Tokens (scope: `connections:write`)
