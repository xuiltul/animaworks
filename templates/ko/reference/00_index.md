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
| `anima-anatomy.md` | Anima 구성 파일 완전 가이드 (전체 파일의 역할, 변경 규칙, 캡슐화) |

### communication/ — 외부 연동 설정

| 파일 | 내용 |
|------|------|
| `slack-bot-token-guide.md` | Slack 봇 토큰 설정 방법 (Per-Anima vs 공유) |

### internals/ — 프레임워크 내부 사양

| 파일 | 내용 |
|------|------|
| `common-knowledge-access-paths.md` | common_knowledge의 5가지 참조 경로와 RAG 인덱스 메커니즘 |

### operations/ — 관리 및 운영 설정

| 파일 | 내용 |
|------|------|
| `project-setup.md` | 프로젝트 초기 설정 (`animaworks init`, 디렉토리 구조) |
| `model-guide.md` | 모델 선택, 실행 모드, 컨텍스트 윈도우 기술 상세 |
| `mode-s-auth-guide.md` | Mode S 인증 모드 설정 (API/Bedrock/Vertex/Max) |
| `voice-chat-guide.md` | 음성 채팅 아키텍처, STT/TTS, 설치 |
| `browser-automation-guide.md` | agent-browser를 이용한 헤드리스 브라우저 자동화 |

### organization/ — 조직 구조 내부 사양

| 파일 | 내용 |
|------|------|
| `structure.md` | 조직 구조의 데이터 소스, supervisor/speciality 해석 방법 |

### troubleshooting/ — 인증 및 자격 증명 설정

| 파일 | 내용 |
|------|------|
| `gmail-credential-setup.md` | Gmail Tool OAuth 인증 설정 절차 |

## 관련 항목

- 일상적인 실용 가이드 → `common_knowledge/00_index.md`
- 공통 스킬 → `common_skills/`
