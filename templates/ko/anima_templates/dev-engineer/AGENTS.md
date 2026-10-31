# 개발 엔지니어 가이드라인

## 구현 레인
- 다른 작업과 충돌하지 않도록 작업은 worktree로 분리해서 진행한다. worktree는 `$ANIMAWORKS_SCRATCH_DIR`에 만들고, 다른 Anima에 넘길 것만 `shared/worktrees/`에 만든다(`common_knowledge/operations/scratch-and-worktrees.md`).
- 기존 코드 변경은 꼭 필요한 최소한으로 하고, 범위 밖의 수정은 별도 작업으로 기록한다.
- 여러 번 반복하는 작업은 절차화하고, 기계(머신)를 활용하는 워크플로 규약을 따른다.

## PR 작성 및 설명
- 목적, 변경 내용, 검증 방법을 정리하여 PR을 작성한다.
- 변경 파일을 나열하고, 리뷰어가 따라가기 쉬운 설명을 작성한다.

## 본인 PR의 CI 모니터링 및 자율 수정
- 본인 PR의 CI가 빨간불이면 방치하지 않고 원인을 조사하여 수정한다.
- 충돌이 발생하면 리베이스 또는 병합으로 해결한다.

## 완료 보고 형식
- 변경 파일 목록과 실행한 검증 명령 및 그 결과를 반드시 포함한다.

## 참조
- 저장소 고유의 규약은 저장소 루트의 CLAUDE.md 및 설정 파일을 참조한다.
- 작업 배치: read_memory_file(path="common_knowledge/operations/workspace-guide.md")
