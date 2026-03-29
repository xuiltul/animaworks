---
name: machine-tool
description: >-
  외부 에이전트 CLI(공작 기계)에 태스크를 위임합니다. 대규모 코드 변경·조사·분석을 외부에 맡긴다.
  Use when: machine 명령으로 구현 위임, 대규모 리팩터·조사 배치, 무거운 에이전트 실행이 필요할 때.
tags: [machine, agent, external, delegation]
---

# Machine 도구 (공작 기계)

외부 에이전트 CLI에 태스크를 위임하는 도구입니다.
코드 변경, 조사, 분석 등 직접 수행하면 부담이 큰 작업을 외부 에이전트에 맡길 수 있습니다.

## 엔진과 기본값

사용 가능한 엔진과 기본값은 **환경에 따라 다릅니다**. 반드시 아래에서 확인하세요:

```bash
animaworks-tool machine run --help
```

`-e`를 생략하면 시스템이 선택한 기본 엔진이 사용됩니다.

## 설계 철학

당신은 **장인(craftsperson)**이고, machine은 **공작 기계(CNC, 레이저 커터 등)**입니다.
공작 기계는 정밀한 가공이 가능하지만, 무엇을 만들지 결정하지 않습니다. 기억도 통신도 없습니다.
**정확한 설계도(instruction)를 전달하는 것이 당신의 역할입니다.**

## 호출 방법

**Bash**: `animaworks-tool machine run [옵션] "지시" -d /path/to/workdir`로 실행합니다.

```bash
# 기본형 (기본 엔진 자동 선택)
animaworks-tool machine run "상세 작업 지시" -d /path/to/workdir

# 엔진을 명시적으로 지정
animaworks-tool machine run -e cursor-agent "지시" -d /path/to/workdir
animaworks-tool machine run -e claude "지시" -d /path/to/workdir

# 모델 오버라이드
animaworks-tool machine run -e claude -m claude-sonnet-4-6 "지시" -d /path/to/workdir

# 백그라운드 실행 (결과는 다음 heartbeat에서 확인)
animaworks-tool machine run --background "지시" -d /path/to/workdir

# 타임아웃 지정 (초)
animaworks-tool machine run -t 300 "지시" -d /path/to/workdir
```

## 파라미터

| 파라미터 | 필수 | 설명 |
|---------|------|------|
| engine | YES | 엔진명 (`--help`에서 사용 가능한 엔진 확인) |
| instruction | YES | 작업 지시. 목표, 대상, 제약, 기대 출력을 명시 |
| working_directory | YES | 작업 디렉토리. 절대 경로 또는 워크스페이스 에일리어스 (예: `aischreiber` / `aischreiber#3af4be6e`) |
| background | no | true로 비동기 실행 (기본값: false) |
| model | no | 모델 오버라이드 (생략 시 엔진 기본값) |
| timeout | no | 타임아웃 초 (동기: 600, 비동기: 1800) |

## instruction 작성법 (중요)

모호한 지시는 품질 저하로 이어집니다. 다음을 반드시 포함하세요:

1. **달성해야 할 목표** — 무엇을 완성할 것인가
2. **대상 파일/모듈** — 어디를 수정할 것인가
3. **제약 조건** — 코딩 규약, 기존 API와의 호환성 등
4. **기대하는 출력 형식** — 코드, 리포트, diff 등

## 사용 판단 기준

| 상황 | 적합 여부 |
|------|----------|
| 다중 파일 코드 변경 | YES |
| 버그 조사 및 원인 분석 | YES |
| 테스트 코드 생성 | YES |
| 리팩터링 | YES |
| 짧은 질문에 대한 답변 | NO (직접 답변) |
| 기억/통신이 필요한 작업 | NO (직접 수행) |

## 주의사항

- 공작 기계는 AnimaWorks 인프라에 접근할 수 없습니다 (기억, 메시지, 조직 정보 없음)
- 레이트 제한이 있습니다 (세션당 5회, heartbeat당 2회)
- background=true 결과는 `state/background_notifications/`에 기록되며 다음 heartbeat에서 확인됩니다
