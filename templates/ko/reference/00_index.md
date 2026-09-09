# Reference — 기술 레퍼런스 목차

AnimaWorks의 상세 기술 사양 및 관리자용 설정 가이드 목록입니다.
RAG 검색 대상이 아닙니다. 필요할 때 `read_memory_file(path="reference/...")`로 직접 참조하세요.

## 참조 방법

```
read_memory_file(path="reference/00_index.md")          # 이 목차
read_memory_file(path="reference/anatomy/anima-anatomy.md")  # 예시
```

## 카테고리

### anatomy/ — 구성 파일 및 아키텍처

| 파일 | 내용 |
|------|------|
| `anatomy/anima-anatomy.md` | Anima 구성 파일 완전 가이드 (전체 파일의 역할, 변경 규칙, 캡슐화) |
| `anatomy/environment-layout.md` | 런타임 디렉토리 구성과 권한 |
| `anatomy/memory-system.md` | 기억 시스템 가이드 |
| `anatomy/priming-channels.md` | Priming 채널 기술 레퍼런스 |
| `anatomy/working-memory.md` | Working Memory(state/) 기술 레퍼런스 |

### communication/ — 메시징 및 외부 연동 설정

| 파일 | 내용 |
|------|------|
| `communication/instruction-patterns.md` | 지시 패턴 |
| `communication/messaging-guide.md` | 메시징 완전 가이드 |
| `communication/reporting-guide.md` | 보고 및 에스컬레이션 가이드 |
| `communication/slack-bot-token-guide.md` | Slack 봇 토큰 설정 방법 (Per-Anima vs 공유) |

### internals/ — 프레임워크 내부 사양

| 파일 | 내용 |
|------|------|
| `internals/common-knowledge-access-paths.md` | common_knowledge의 5가지 참조 경로와 RAG 인덱스 메커니즘 |

### operations/ — 관리 및 운영 설정

| 파일 | 내용 |
|------|------|
| `operations/browser-automation-guide.md` | agent-browser를 이용한 헤드리스 브라우저 자동화 |
| `operations/heartbeat-cron-guide.md` | 정기 실행 설정 및 운영 |
| `operations/memory-writing-guide.md` | 기억 기록 위치와 정기 실행 선택 |
| `operations/mode-s-auth-guide.md` | Mode S 인증 모드 설정 (API/Bedrock/Vertex/Max) |
| `operations/model-guide.md` | 모델 선택, 실행 모드, 컨텍스트 윈도우 기술 상세 |
| `operations/project-setup.md` | 프로젝트 초기 설정 (`animaworks init`, 디렉토리 구조) |
| `operations/task-management.md` | 태스크 관리 |
| `operations/tool-usage-overview.md` | 도구 사용 가이드 |
| `operations/voice-chat-guide.md` | 음성 채팅 아키텍처, STT/TTS, 설치 |

### organization/ — 조직 구조 내부 사양

| 파일 | 내용 |
|------|------|
| `organization/roles.md` | 역할과 책임 범위 |
| `organization/structure.md` | 조직 구조의 데이터 소스, supervisor/speciality 해석 방법 |

### troubleshooting/ — 인증 및 자격 증명 설정

| 파일 | 내용 |
|------|------|
| `troubleshooting/common-issues.md` | 자주 발생하는 문제와 해결법 |
| `troubleshooting/escalation-flowchart.md` | 막혔을 때 판단 플로차트 |
| `troubleshooting/gmail-credential-setup.md` | Gmail Tool OAuth 인증 설정 절차 |

### usecases/ — 활용 사례 가이드

| 파일 | 내용 |
|------|------|
| `usecases/usecase-communication.md` | 활용 사례: 커뮤니케이션 자동화 |
| `usecases/usecase-customer-support.md` | 활용 사례: 고객 지원 |
| `usecases/usecase-development.md` | 활용 사례: 소프트웨어 개발 지원 |
| `usecases/usecase-knowledge.md` | 활용 사례: 지식 관리 및 문서 정비 |
| `usecases/usecase-monitoring.md` | 활용 사례: 인프라 및 서비스 모니터링 |
| `usecases/usecase-overview.md` | AnimaWorks 활용 사례 가이드 |
| `usecases/usecase-research.md` | 활용 사례: 조사 및 리서치 분석 |
| `usecases/usecase-secretary.md` | 활용 사례: 비서 및 사무 지원 |

## 관련 항목

- 일상적인 실용 가이드 → `common_knowledge/00_index.md`
- 공통 스킬 → `common_skills/`
