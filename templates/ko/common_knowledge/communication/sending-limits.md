# 메시지 전송·수신 운영 가이드

메시지 배달, Inbox 시작, 대화 루프를 피하기 위한 운영을 정리한다.
기존의 전송 예산·깊이 거부·Inbox cooldown / cascade / intent 필터는 폐지되었다.

## 전송 시 남아 있는 규칙

- `send_message`의 intent는 `report` 또는 `question`입니다. 작업 위임에는 `delegate_task`을 사용합니다.
- 같은 실행에서 동일한 수신자에게 보내는 DM은 한 건까지입니다. 중복 전송을 방지하기 위한 가드이며, 수신자 수에는 제한이 없습니다.
- 시간당·일일 전송 예산이나 쌍별 대화 깊이에 따른 전송 거부는 없습니다.
- 내부 Anima 간 대화 깊이는 진단 목적으로 로그에 기록될 수 있습니다. 이는 관측만을 위한 것으로, 메시지 본문을 버리거나 전송을 거부하지 않습니다.

## 이해·감사·칭찬에 대한 응답

수신 내용이 이해, 감사, 칭찬뿐이라면 답장하지 않는다. 응답을 계속하지 않고, 추가 질문·요청·새 정보가 있는 경우에만 필요한 대응을 한다.
작업 요청에 대한 수령 연락은 필요한 경우에 한 번만, 전망이나 다음 행동과 함께 답한다.

## Inbox 시작과 처리

Inbox의 새 JSON 파일을 파일 변경 알림으로 감지하고, 읽지 않은 메시지가 있으면 Inbox 처리를 시작한다.
동시에 시작하는 Inbox 처리는 1개만. 실행 중에 도착한 메시지는 다음 1회에 모아서 처리한다.
파일 알림을 놓친 경우에 대비해 45초마다 읽지 않은 메시지를 다시 확인한다. Provider 쪽 실패에서는 `rate_guard`의 복구 시간을 기다리고, 읽지 않은 메시지는 남겨 둔다.

메시지가 대량인 경우, `state/overflow_inbox/`로의 대피는 용량 보호로 남아 있다. 이는 발신자나 intent에 의한 억제가 아니라, 대피된 메시지는 나중에 확인·처리할 수 있다.

## 기록과 확인

`Messenger.send`은 전송 메시지를 activity log에 기록한다. 대화가 짧은 시간에 이어진 경우 진단 로그를 추가할 수 있지만, 전송 결과는 변하지 않는다.
최근의 `message_sent` / `channel_post`는 `core/memory/priming/outbound.py`이 프라이밍에 이용한다.

전송할 수 없는 경우, 수신자 해결·권한·외부 채널의 실제 배달 오류를 확인한다. 레이트 상한이나 대화 깊이 초과를 기다릴 필요는 없다.

## 구현 위치

| 역할 | 모듈 |
|------|------------|
| 수신처 해결 · Slack / Chatwork 외부 전송 | `core/messaging/outbound.py` |
| 내부 DM 배송 · activity log 기록 · 깊이 진단 로그 | `core/messaging/messenger.py` |
| DM 중복 방지 · Board의 run 내 중복 방지 | `core/tooling/handler_comms.py` |
| Inbox의 파일 wake · 단일 실행 · provider backoff | `core/runtime/inbox_rate_limiter.py` |
| Inbox 용량 보호의 overflow | `core/anima/inbox_overflow.py` |
| 최근 전송의 프롬프트 주입 | `core/memory/priming/outbound.py` |
