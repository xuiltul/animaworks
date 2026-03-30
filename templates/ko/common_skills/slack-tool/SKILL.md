---
name: slack-tool
description: >-
  Slack 연동 도구. 메시지 송수신·검색·미회신 확인·채널 목록·이모지 리액션을 수행한다.
  Use when: Slack에 게시, 채널 목록, 스레드 답장, 미회신 확인, 리액션 추가가 필요할 때.
tags: [communication, slack, external]
---

# Slack 도구

Slack의 메시지 송수신, 검색, 리액션을 수행하는 외부 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool slack <서브커맨드> [인수]`로 실행합니다.

## 액션 목록

### send — 메시지 송신
```bash
animaworks-tool slack send CHANNEL MESSAGE [--thread TS]
```

### messages — 메시지 조회
```bash
animaworks-tool slack messages CHANNEL [-n 20]
```

### search — 메시지 검색
```bash
animaworks-tool slack search KEYWORD [-c CHANNEL] [-n 50]
```

### unreplied — 미회신 메시지 확인
```bash
animaworks-tool slack unreplied [--json]
```

### channels — 채널 목록
```bash
animaworks-tool slack channels
```

### react — 이모지 리액션
- `emoji`: Slack 이모지 이름 (콜론 없이. 예: `thumbsup`, `eyes`, `white_check_mark`)
- `message_ts`: 리액션 대상 메시지의 타임스탬프 (`messages` 액션 결과에서 확인 가능)
- **참고**: `react` 액션은 CLI 미지원. MCP를 통해 사용하세요.

## CLI 사용법

```bash
animaworks-tool slack send CHANNEL MESSAGE [--thread TS]
animaworks-tool slack messages CHANNEL [-n 20]
animaworks-tool slack search KEYWORD [-c CHANNEL] [-n 50]
animaworks-tool slack unreplied [--json]
animaworks-tool slack channels
```

## 주의사항

- Slack Bot Token이 credentials에 사전 설정되어 있어야 합니다
- 채널은 # 접두사가 붙은 이름 또는 채널 ID로 지정합니다
- 리액션에는 `reactions:write` 스코프가 필요합니다
