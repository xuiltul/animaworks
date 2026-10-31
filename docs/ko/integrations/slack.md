<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/integrations/slack.md -->
<!-- i18n: source-sha256=11e51c325cd259cbe03ddee2b1da8a43d4008ed90aff26e14165864ae4646efc generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 581e20f1

# Slack 연동

Slack 연동은 Slack의 이벤트를 Anima의 inbox로 가져오고, 설정에 따라 메시지 전송이나 검색 등을 수행한다. NAT 내부의 서버에서는 Socket Mode를 사용할 수 있다. Webhook Mode를 사용하는 경우, 외부에서 접근 가능한 HTTPS endpoint가 필요하다.

## Slack App 준비

Slack API의 App 설정에서 Bot을 생성하고, 필요한 Bot Token Scopes와 수신 이벤트를 추가한다. 수신 이벤트에는 사용 범위에 따라 채널, DM, 그룹 DM의 message와 `app_mention`을 지정한다. Socket Mode에서는 Socket Mode를 활성화하고, `connections:write`을 가진 App-Level Token을 발급한다. Webhook Mode에서는 Event Subscriptions의 Request URL에 `/api/webhooks/slack/events`를 등록하고, Signing Secret을 준비한다.

필요한 권한은 실제로 사용하는 기능에 한정한다. 읽기, 전송, 사용자 정보 조회 등의 scope는 Slack App에 등록하는 이벤트와 사용하는 도구의 범위에 맞춰 설정한다.

## 자격 증명

Bot Token은 `SLACK_BOT_TOKEN`, Socket Mode의 App-Level Token은 `SLACK_APP_TOKEN`, Webhook의 서명 검증에는 `SLACK_SIGNING_SECRET`를 사용한다. 값은 vault, 공유 credentials, 또는 환경 변수 등의 자격 증명 관리 방법에 등록하고, 공개 설정 파일이나 문서에 실제 값을 기재하지 않는다. Socket Mode 이외에는 App-Level Token이 필요 없다.

## AnimaWorks 설정

`config.json`의 `external_messaging.slack`에서 연동을 활성화하고, `mode`을 `socket` 또는 `webhook`로 설정한다. 공유 Bot의 `anima_mapping`은 Slack channel ID에서 Anima 이름으로의 대응, `default_anima`은 명시적 mapping이 없는 경우의 기본 수신처이다. 값을 비운 채널은 기본 Anima로 보내지 않고 무시할 수 있다. 설정 항목의 목록은 [설정 참조의 external_messaging](../reference/config.md)을 참조한다.

```json
{
  "external_messaging": {
    "slack": {
      "enabled": true,
      "mode": "socket",
      "anima_mapping": {
        "C0123456789": "sample-anima"
      },
      "default_anima": ""
    }
  }
}
```

channel ID는 Slack의 채널 상세에서 확인한다. Bot은 수신 대상 채널에 초대한다. 개별 Bot을 사용하는 경우, 해당 자격 증명이 개별 Anima의 이름 공간에 등록되어 있는지 확인한다.

## 연결 및 확인

Slack의 연결 설정을 저장한 후, 서버를 재시작하거나 구성이 지원하면 Slack의 hot reload를 실행한다. 서버 로그에 연결 성공이 표시되는지 확인하고, 테스트용 채널에서 메시지를 보내 올바른 Anima의 inbox에 도달하는지 확인한다. Webhook Mode는 공개 endpoint, 서명 검증, 이벤트 구독 모두를 확인한다.

도달하지 않는 경우 `enabled`과 `mode`, 자격 증명, channel ID와 Anima의 대응, Bot의 초대 상태, Slack App에 등록한 이벤트와 scope를 순서대로 확인한다. Webhook의 경우 Signing Secret과 HTTPS 경로를 확인한다. 수신을 확인한 후, 외부 응답이나 Board 동기화를 필요 최소한의 범위에서 활성화한다.
