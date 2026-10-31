<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/README.md -->
<!-- i18n: source-sha256=582644ac13997cf7c0bb82404f6cb1c785b02b7ed687492367888946a3304901 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 193a5e72

# 문서 색인

`docs/ja/`는 공개 문서의 일본어 정본이다. 영어판과 한국어판은 R22의 번역 파이프라인으로 생성하며, `docs/en/`·`docs/ko/`는 직접 편집하지 않는다. `docs/ja/reference/`도 R21이 코드에서 생성하므로 직접 편집하지 않는다.

## 문서 구성

| 문서 | 내용 |
|---|---|
| [설계 철학](vision.md) | AnimaWorks가 지향하는 설계 사상과 가치관. |
| [기능 개요](overview.md) | 기능의 전체적인 모습과 상세 장으로 가는 입구. |
| [뇌과학과의 대응](brain-mapping.md) | 기억·주의·자율 기제와 뇌과학의 대응 및 설계 근거. |
| [전체 아키텍처](architecture/index.md) | 시스템 구성과 각 아키텍처 장에 대한 안내. |
| [Anima 파일](architecture/anima-files.md) | 각 Anima의 정의 파일, 설정, 상태 파일. |
| [프로세스 구성](architecture/process.md) | server, root, task runner, IPC, 재시작. |
| [실행 모드](architecture/execution.md) | 실행 엔진, 모델 해결, fallback, 컨텍스트 관리. |
| [수명 주기](architecture/lifecycle.md) | chat, inbox, heartbeat, cron, task의 시작 경로와 잠금. |
| [프롬프트 구성](architecture/prompt.md) | 시스템 프롬프트와 런타임 컨텍스트의 구성. |
| [작업 관리](architecture/tasks.md) | Task Board, 위임, background task. |
| [메시징](architecture/messaging.md) | DM, Board, 사람에게 보내는 알림, Inbox 깨우기와 메시지 처리, 외부 연동. |
| [기억 시스템](memory/index.md) | 기억 설계, 디렉터리, frontmatter. |
| [자동 회상](memory/priming.md) | 런타임에 기억을 컨텍스트로 가져오는 방식. |
| [의도적 회상과 검색](memory/retrieval.md) | `search_memory`, 검색 처리, RAG, 복구. |
| [통합과 망각](memory/consolidation.md) | 일간·주간 처리, 기억 검토, 절차 기억. |
| [활동 로그](memory/activity-log.md) | JSONL 활동 기록과 스트리밍 중 복구 기록. |
| [보안](security.md) | 권한 경계, 보호, 보안 운영. |
| [Enclave](enclave.md) | 격리 런타임 구축, 데이터 참조, 점검 및 감사. |
| [설정](operations/configuration.md) | 전체 설정과 각 Anima의 설정 방법. |
| [회사 관리](operations/company.md) | 조직 및 회사 정보 관리. |
| [GPU 운영](operations/gpu.md) | GPU를 사용하는 구성 요소의 운영. |
| [개발 팀](operations/dev-team.md) | 개발 환경과 팀 운영. |
| [Slim Runtime 마이그레이션](operations/slim-runtime-migration.md) | Slim Runtime 마이그레이션 절차. |
| [Slack 연동](integrations/slack.md) | Slack 연결 설정과 운영. |
| [Zoom 연동](integrations/zoom.md) | Zoom RTMS 연결 설정과 운영. |
| [CLI 참조](reference/cli.md) | `animaworks` 명령어의 자동 생성 참조. |
| [도구 CLI 참조](reference/tool-cli.md) | `animaworks-tool`의 자동 생성 참조. |
| [API 참조](reference/api.md) | HTTP API의 자동 생성 참조. |
| [설정 참조](reference/config.md) | 설정 항목과 기본값의 자동 생성 참조. |
| [모듈 참조](reference/modules.md) | `core/`의 구성에서 생성한 모듈 안내. |

## 읽는 순서

- **처음 접하는 사람**: [기능 개요](overview.md) → [아키텍처 전체](architecture/index.md) → [기억 시스템](memory/index.md). 동작의 흐름은 [라이프사이클](architecture/lifecycle.md)을 읽는다.
- **설계에 관심이 있는 사람**: [설계 이념](vision.md) → [뇌과학과의 대응](brain-mapping.md) → [아키텍처 전체](architecture/index.md) → 기억의 각 장.
- **운영자**: [설정](operations/configuration.md) → [보안](security.md) → [CLI 참조](reference/cli.md)·[설정 참조](reference/config.md). 개별 연계는 [Slack](integrations/slack.md)·[Zoom](integrations/zoom.md)을 참조한다.

## 작성자용

- 문체는 "이다"체로 한다. 각 파일의 머리에 `> 確認したコミット: <短縮 sha>`을 두고, 내용을 대조한 코드의 시점을 나타낸다.
- 코드에 대한 참조는 `core/memory/priming/engine.py`의 `_prime_compact`처럼 경로와 심볼 이름으로 적는다. 줄 번호는 쓰지 않는다.
- 식별자·설정 키·경로·명령은 백틱으로 감싸 번역 시 보호할 수 있게 한다.
- 설정 키의 목록표는 본문에 만들지 않고, [설정 참조](reference/config.md)로 링크한다. 본문에서는 동작 설명에 필요한 키만 다룬다.
- 폐지된 기능이나 과거의 구현 경위는 기재하지 않는다. 변경 이력은 CHANGELOG의 역할이다.
- 제목은 번역의 차이 단위가 된다. 각 절은 대체로 2,000자 이내로 유지한다.
- 하나의 사실은 원칙적으로 한 곳에 적고, 개요나 대응표에서는 상세 장으로 링크한다.
