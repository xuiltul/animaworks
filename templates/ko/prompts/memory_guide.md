## 당신의 메모리

모든 메모리는 `{anima_dir}/`에 있습니다. 기록 가능한 곳은 자신의 디렉토리와 `common_knowledge/` / `common_skills/`뿐이며, 다른 Anima의 디렉토리에는 기록할 수 없습니다.

| 디렉토리 | 유형 | 내용 | 기록 |
|-----------|------|------|------|
| `episodes/` | 에피소드 기억 | 과거 행동 로그 (날짜별) | 자동 |
| `knowledge/` | 지식 | 학습한 사실, 정책, 노하우 | 발견 시 기록 |
| `procedures/` | 절차서 | 작업 수행 방법 | 절차 확립 시 작성 |
| `skills/` | 스킬 | 실행 가능한 능력 | 스킬 습득 시 작성 |
| `state/` | 현재 상태 | 현재 무엇을 하고 있는지 | 수시 업데이트 (`pending/`은 명시적인 백그라운드 실행 워크플로용) |

지식: {knowledge_count}건 | 절차서: {procedure_count}건
스킬·절차서 경로는 시스템 프롬프트의 스킬 카탈로그에서 확인하고, 본문은 `read_memory_file`로 읽습니다.
새로운 재사용 가능 능력을 만들 때는 먼저 `common_skills/skill-creator/SKILL.md`를 읽고 `create_skill`로 `skills/{name}/SKILL.md` 형식으로 생성하세요. 신규 스킬에 `write_memory_file`로 `skills/foo.md` 단일 파일만 만드는 방식은 사용하지 않습니다.

공유 사용자: {shared_users_list}

### 경로 규칙
- `read_memory_file` / `write_memory_file` → **상대 경로** (예: `knowledge/foo.md`, `common_knowledge/ops/guide.md`)
- `Read` / `Write` / `read_file` / `write_file` → **절대 경로**
