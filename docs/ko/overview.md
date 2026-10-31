<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/overview.md -->
<!-- i18n: source-sha256=f595d1acb48049d421a5181d012ee14ad061f4d04e7ba9d4f5697e84f26d48ea generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: b304b7dc

# 기능 개요

AnimaWorks는 여러 Digital Anima가 기억, 도구, 조직 내 메시지를 사용하여 지속적으로 작업하기 위한 실행 기반이다. 각 기능의 구현 세부 사항은 아래 담당 장과 [아키텍처 전체 그림](architecture/index.md)을 참조한다.

## 자율 에이전트

Anima는 정의 파일과 상태를 가지며, chat, 수신 메시지, heartbeat, cron, task에서 실행된다. 프로세스 구성은 [process](architecture/process.md), 개별 파일은 [Anima의 파일](architecture/anima-files.md)을 참조한다.

## 기억

대화 기록, 현재 작업 상태, 장기 기억, 공유 지식을 목적별로 다룬다. prompt 반영과 Anima 데이터 구성은 [Anima의 파일](architecture/anima-files.md)과 [prompt 구축](architecture/prompt.md)을 참조한다.

## 멀티 모델 실행

모델 이름과 per-anima 설정에서 6가지 실행 모드를 선택하고, SDK, CLI 또는 API로 호출한다. 엔진과 fallback 개요는 [모델 실행](architecture/execution.md)에 정리되어 있다.

## 조직

Anima는 회사, role, supervisor를 가지며, 조직 내에서 작업을 위임하고 공유 workspace나 channel을 사용한다. 권한과 조직 경계는 [보안](security.md), 메시지 구조는 [메시징](architecture/messaging.md)을 참조한다.

## 메시징

Anima 간의 DM, 공유 Board, 외부 서비스를 통한 연락을 처리한다. Inbox는 파일 변경 알림으로 시작되며, 메시지의 집계 및 중복 방지는 [메시징](architecture/messaging.md)에 기술한다.

## 작업 관리

작업을 공유 TaskStore에 영속화하고, 위임, attempt, lease와 Web Task Board를 일원적으로 다룬다. CLI와 background task의 역할은 [작업 관리](architecture/tasks.md)를 참조한다.

## 스킬과 도구

MCP는 trigger에 따라 사용 가능한 도구를 추리고, Anima별 스킬 catalog는 요청에 맞춰 선택한다. prompt와 tool guide 구성은 [prompt 구축](architecture/prompt.md), `animaworks-tool`의 인자는 [도구 CLI 참조](reference/tool-cli.md)를 참조한다.

## Web UI

Web UI에는 Home, Chat, Animas, Activity, Logs, Settings, Board, Task Board, Users 페이지가 있다. API 목록은 [API 참조](reference/api.md)를 참조한다.

## 캐릭터 에셋

Anima는 캐릭터 이미지나 표정 등의 에셋을 가지며, Web UI나 대화 표현에서 사용한다. 파일 배치와 Anima 정의는 [Anima의 파일](architecture/anima-files.md)을 참조한다.

## 보안

개별·전체 권한 설정, tool 실행 시 검사, 외부 연동 인증을 조합한다. 권한 경계와 설정 방법은 [보안](security.md)을 참조한다.

## 프로세스 관리

server와 supervisor가 Anima root, task runner 시작, 통신, 재시작을 관리한다. 자세한 내용은 [프로세스 구성](architecture/process.md)을 참조한다.

## CLI

`animaworks`은 초기화, 설정, Anima, 작업, 기억 등의 관리를 제공한다. 명령 이름과 인자 목록은 [CLI 참조](reference/cli.md)를 참조한다.

## 설정 관리

전체 설정, 모델 선택, per-anima의 `status.json`, 권한 설정을 나누어 관리한다. 설정 값과 기본값 목록은 [설정 참조](reference/config.md)를 참조한다.

## 운영

시작 준비, 로그 확인, 데이터 유지보수, 이상 시 확인 절차를 제공한다. 운영 절차는 [운영 가이드](operations/index.md)를 참조한다.

## MCP

MCP의 허용 목록 `MCP_TOOL_NAMES`과 트리거·역할에 따른 `resolve_tool_surface`은 `core/tooling/surface.py`에 집약되어 있다. 실제로 공개되는 목록은 트리거와 Anima의 실행 조건에 따라 다르다.
