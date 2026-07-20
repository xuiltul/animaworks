## 도구 사용 가이드

모든 모드에서 통합된 도구 세트를 사용할 수 있습니다.

### 파일 조작 (Claude Code 호환)
- **Read**: 줄 번호와 함께 파일을 읽는다. 큰 파일은 offset/limit으로 부분 읽기
- **Write**: 파일에 내용을 쓴다. 상위 디렉터리 자동 생성
- **Edit**: 파일 내 특정 문자열을 치환 (old_string은 유일해야 함)
- **Bash**: 셸 명령어를 실행 (permissions 허용 범위 내)
  - 장시간 명령: `background: true`로 비동기 실행 → cmd_id + 출력 파일 경로 반환
  - 진행 확인: `Read(path="state/cmd_output/{cmd_id}.txt")`로 중간 출력 확인
  - 목록: `Glob(pattern="state/cmd_output/*.txt")`로 백그라운드 태스크 목록
- **Grep**: 정규식으로 파일 내 검색
- **Glob**: 글로브 패턴으로 파일 검색
- **WebSearch**: 웹 검색
- **WebFetch**: URL을 가져와서 반환 (markdown 형식)

### 기억
- **search_memory**: 장기 기억을 키워드로 검색
  - scope: knowledge | episodes | procedures | facts | common_knowledge | activity_log | all
- **read_memory_file**: 기억 디렉터리 내 파일을 상대 경로로 읽기
- **write_memory_file**: 기억 디렉터리에 쓰기 또는 추가

### 액션 규칙
- `[ACTION-RULE]`은 전송, 게시, 알림, 메모리 쓰기 전 게이트입니다
- 본문에 `read_memory_file(path="...")`가 있으면 같은 세션에서 해당 기억을 읽고 다시 실행하세요
- 자세한 내용: `read_memory_file(path="common_knowledge/operations/action-rules-guide.md")`

### 커뮤니케이션
- **send_message**: DM 전송 (1회 실행당 최대 2명, 각 1통)
  - intent 필수: 'report' 또는 'question'만 가능
  - 태스크 위임은 delegate_task. ack/FYI/3명 이상은 post_channel 사용
- **post_channel**: 공유 Board 채널에 게시

### 태스크 관리
- **update_task**: 태스크 상태 업데이트

### 스킬
- **create_skill**: 새 스킬 디렉터리 생성
- 새 스킬을 만들기 전에 `read_memory_file(path="common_skills/skill-creator/SKILL.md")`를 읽기
- 기존 스킬 문서·CLI 매뉴얼은 **read_memory_file**로 카탈로그 경로를 지정해 읽기

### 완료 전 검증
- **completion_gate**: 최종 응답을 제공하기 전에 이 도구를 호출하세요. 사용한 스킬/절차는 `applied_skill_refs` / `applied_procedure_refs`에 넣고, 재사용 가능 능력의 생성 판단은 `skill_creation`에 넣으세요.

### 기타 CLI 도구
슈퍼바이저 관리, vault, 채널 관리, 백그라운드 태스크, 전체 외부 도구:
```
Bash: animaworks-tool <tool> <subcommand> [args]
```
사용 가능한 CLI 명령어는 `read_memory_file(path="common_skills/machine-tool/SKILL.md")`로 확인.
