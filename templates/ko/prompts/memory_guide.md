## 당신의 메모리

모든 메모리는 `{anima_dir}/`에 있습니다.

| 디렉토리 | 유형 | 내용 | 기록 |
|-----------|------|------|------|
| `episodes/` | 에피소드 기억 | 과거 행동 로그 (날짜별) | 자동 |
| `knowledge/` | 지식 | 학습한 사실, 정책, 노하우 | 발견 시 기록 |
| `procedures/` | 절차서 | 작업 수행 방법 | 절차 확립 시 작성 |
| `skills/` | 스킬 | 실행 가능한 능력 | 스킬 습득 시 작성 |
| `state/` | 현재 상태 | 현재 무엇을 하고 있는지 | 수시 업데이트 (`pending/`은 `submit_tasks` 경유) |

지식: {knowledge_count}건 | 절차서: {procedure_count}건
스킬과 절차서 목록은 skill 도구로 확인할 수 있습니다.

공유 사용자: {shared_users_list}

### 경로 규칙
- `read_memory_file` / `write_memory_file` → **상대 경로** (예: `knowledge/foo.md`, `common_knowledge/ops/guide.md`)
- `Read` / `Write` / `read_file` / `write_file` → **절대 경로**
