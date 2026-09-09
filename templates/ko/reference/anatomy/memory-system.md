# 기억 시스템 가이드

Anima 기억의 구조, 종류, 활용법에 대한 레퍼런스입니다.
기억의 검색, 기록, 정리 방법을 확인할 때 참조하세요.

## 기억의 전체 구조

당신의 기억은 인간 뇌의 기억 모델에 대응하는 여러 종류로 구성됩니다:

| 기억 유형 | 디렉토리 | 인간 비유 | 내용 |
|-----------|----------|----------|------|
| **단기 기억** | `shortterm/` | 작업 기억 | 최근 대화의 맥락 |
| **일화 기억** | `episodes/` | 경험 기억 | 언제 무엇을 했는지 |
| **의미 기억** | `knowledge/` | 지식 | 배운 것, 노하우 |
| **절차 기억** | `procedures/` | 체득한 절차 | 단계별 수행 절차 |
| **스킬** | `skills/` | 특기/전문 기술 | 실행 가능한 절차서 |

추가로 모든 Anima가 공유하는 기억도 있습니다:

| 공유 기억 | 경로 | 내용 |
|-----------|------|------|
| **공유 지식** | `common_knowledge/` | 프레임워크 레퍼런스 (이 파일 자체 포함) |
| **공통 스킬** | `common_skills/` | 모든 Anima가 사용할 수 있는 스킬 |
| **조직 공유 지식** | `shared/common_knowledge/` | 조직이 운영 중 축적한 지식 |
| **사용자 프로필** | `shared/users/` | Anima 간 공유 사용자 정보 |

---

## 단기 기억 (shortterm/)

**최근 대화와 세션의 맥락**을 보유합니다. 인간의 작업 기억에 해당합니다.

- Chat용 (`shortterm/chat/`)과 Heartbeat용 (`shortterm/heartbeat/`)으로 분리
- 컨텍스트 윈도우 사용률이 임계값을 초과하면 오래된 부분이 자동으로 외부화됨
- 세션 간 맥락 연속성을 위해 사용됨

단기 기억은 직접 조작할 필요가 없습니다. 프레임워크가 자동으로 관리합니다.

---

## 일화 기억 (episodes/)

**"언제 무엇을 했는지"의 일별 로그**입니다. 인간의 경험 기억에 해당합니다.

- 날짜별 파일 (예: `2026-03-09.md`)에 자동 기록됨
- "지난주에 뭘 했지?", "이 문제에 전에 대응한 적 있나?"를 떠올릴 때 사용
- Consolidation을 통해 패턴과 교훈이 `knowledge/`로 정제됨

### 기억 기록

```
write_memory_file(path="episodes/2026-03-09.md", content="...")
```

### 기억 검색

```
search_memory(query="Slack API 접속 테스트", scope="episodes")
```

---

## 의미 기억 (knowledge/)

**배운 지식, 노하우, 패턴**입니다. 인간이 "알고 있는 것"에 해당합니다.

- 일화에서 추출된 교훈과 패턴
- 기술 메모, 대응 방침, 판단 기준
- 일별 Consolidation으로 자동 축적되며, 직접 능동적으로 기록할 수도 있음
- **재고정화**: front matter에서 `failure_count >= 2`이고 `confidence < 0.6`인 knowledge는 절차와 마찬가지로 LLM 개정 대상이 될 수 있음

예:
- "Slack API 레이트 제한은 Tier 1에서 1req/sec"
- "이 고객은 월요일에 연락이 많은 편"
- "배포 전 확인 항목 체크리스트"

### 기억 기록

```
write_memory_file(path="knowledge/slack-api-notes.md", content="...")
```

### 기억 검색

```
search_memory(query="Slack API 레이트 제한", scope="knowledge")
```

---

## 절차 기억 (procedures/)

**"어떻게 하는지"의 단계별 절차서**입니다. 인간의 "몸이 기억한 절차"에 해당합니다.

- 문제 해결 절차, 정형 작업 흐름
- `issue_resolved` 이벤트에서 자동 생성될 수 있음 (confidence 0.4)
- **스킬만큼 전면 보호되지는 않음**: 메타데이터에 따라 망각 파이프라인 대상이 될 수 있음
- **재고정화**: front matter에서 `failure_count >= 2`이고 `confidence < 0.6`이면 LLM이 절차를 개정할 수 있음. 개정 후 카운터를 리셋하고 버전을 올리며 기존 버전은 `archive/`로 이동함

예:
- "SSL 인증서 갱신 절차"
- "신규 Anima 온보딩 절차"
- "운영 장애 시 에스컬레이션 절차"

### 기억 기록

```
write_memory_file(path="procedures/ssl-renewal.md", content="...")
```

### 기억 검색

```
search_memory(query="SSL 인증서 갱신", scope="procedures")
```

---

## 스킬 (skills/)

**실행 가능한 절차서 및 도구 사용 가이드**입니다. "특기"에 해당합니다.

- 개인 스킬 (`skills/`)과 공통 스킬 (`common_skills/`)이 있음
- 필요한 스킬은 active skill context, Skill Router, Skill Hub 또는 `read_memory_file(path="...")`로 읽음
- 스킬 본문을 항상 전부 읽을 필요는 없음. 먼저 이름, 설명, 포인터를 보고 필요할 때만 전문을 읽음
- 실적이 있는 `procedures/`는 probation skill 또는 quarantine skill로 승격될 수 있음
- **벡터 저장소에서는 항상 망각 대상에서 제외됨** (`skills` / `shared_users` 유형 보호)

### 스킬 확인

```
read_memory_file(path="skills/newstaff/SKILL.md")  # 스킬 전문 가져오기
```

### 스킬 생성

```
create_skill(skill_name="deploy-procedure", description="운영 배포 절차", body="...")
```

---

## 기억의 자동 프로세스

기본 `compact` 프로필은 발신자 정보, 미완료 작업, 명시적 상주 포인터, 최근 발신 이력과 미전달 인간 알림을 제공합니다. 대화·작업 요청과 질문에는 관련 지식을 검색하지만 일반 heartbeat·cron·보고에는 검색하지 않습니다. 넓은 활동·일화·그래프 확장은 선택적 `full` 또는 명시적 검색으로 사용합니다. `priming.max_tokens` 기본값은 2,000이며 알림과 필수 상주 규칙은 별도로 보존합니다. Anima별 `status.json: priming_profile`로 모델 라우팅과 독립적으로 설정할 수 있습니다.

과거 지시·고객 정보·진행 중 작업이 필요할 때 검색하세요. 모든 답변에서 검색하거나 사용마다 성공을 보고할 필요는 없습니다. 스킬·절차 본문은 필요할 때 읽습니다. `[IMPORTANT]` 표시만으로 항상 상주하는 것은 아닙니다.

부작용 있는 행동에는 `[ACTION-RULE]`, 권한, 승인, 중복 실행 방지가 계속 적용됩니다. 중단되면 지정된 규칙을 읽으세요. 신뢰할 수 없는 검색 결과는 신뢰된 문맥과 분리됩니다.

일일 통합은 미처리 활동 청크로 일화를 생성하고 성공한 입력을 기록합니다. 원본 활동과 기억을 보존합니다. 지식 수정은 별도 단계이며 기본 비활성화입니다(`consolidation.knowledge_mutation_enabled`). 주간·월간 변경, 증류, 저활성화, 자기 수정, 자동 스킬 학습과 fact 생성도 기본 비활성화입니다. 인덱싱·복구·기존 fact 읽기는 유지합니다. 선택적 유지보수도 고객별 세부 정보·출처·안전 규칙을 보존해야 합니다. 건너뛰기나 변경 없음은 정상이며 재시도 이유가 아닙니다.

Curator 승격·퇴역은 기본적으로 제안만 생성합니다. 안전 차단은 즉시 격리할 수 있고 운영자의 명시적 변경도 가능합니다. 결과 횟수는 진단 자료이지 작업 품질의 증명이 아닙니다.

---

## 기억 도구의 활용

| 하고 싶은 것 | 도구 | 예시 |
|-------------|------|------|
| 키워드로 기억 검색 | `search_memory` | `search_memory(query="API 설정", scope="all")` |
| 특정 파일 읽기 | `read_memory_file` | `read_memory_file(path="knowledge/api-notes.md")` |
| 기억 기록 | `write_memory_file` | `write_memory_file(path="knowledge/new-insight.md", content="...")` |
| 불필요한 기억 정리 | `archive_memory_file` | `archive_memory_file(path="knowledge/outdated.md")` |

### scope (검색 범위) 선택

| scope | 검색 대상 | 사용 시점 |
|-------|----------|----------|
| `knowledge` | 지식, 노하우 | "이것에 대해 알고 있는 게 있나?" |
| `episodes` | 과거 행동 로그 | "전에 이걸 한 적이 있나?" |
| `procedures` | 절차서 | "이 작업의 절차는?" |
| `common_knowledge` | 공유 레퍼런스 | "프레임워크 사양은?" |
| `activity_log` | 최근 활동 로그 (도구 실행 결과, 메시지 등) | "방금 읽은 이메일 내용", "이전 검색 결과" |
| `all` | 위의 모든 항목 (벡터 검색 + activity_log BM25를 RRF로 통합) | 폭넓은 검색 |

---

## RAG (벡터 검색)의 구조

기억 검색에는 RAG (Retrieval-Augmented Generation)가 사용됩니다:

1. **인덱싱**: 모든 기억 파일이 embedding 벡터로 변환되어 ChromaDB에 저장됨
2. **검색**: 쿼리 텍스트를 벡터화하여 유사도가 높은 기억 청크를 조회
3. **그래프 확산**: NetworkX 그래프 기반 확산 활성화로 관련 주변 기억도 추출
4. **증분 갱신**: 변경된 파일만 재인덱싱하므로 대량의 기억이 있어도 빠른 처리 가능
5. **복구**: ChromaDB 또는 벡터 검색 불일치가 발생하면 RAG repair가 `vectordb`를 격리하고 기억 파일에서 인덱스를 재구축할 수 있음

RAG는 `search_memory`를 호출하면 자동으로 사용됩니다. 구조를 의식할 필요는 없지만, **검색 정확도를 높이는 팁**:
- 구체적인 키워드를 포함한 쿼리 사용
- 기억을 기록할 때 제목과 내용을 명확하게 작성
- 관련 정보는 같은 파일에 모아서 정리
