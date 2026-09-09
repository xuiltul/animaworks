### 저장소 작업 규칙

- canonical checkout의 `main` / `master`는 읽기 전용으로 취급합니다. 구현, 검증, commit은 반드시 전용 `git worktree`에서 수행합니다
- worktree는 `{data_dir}/companies/<회사>/shared/worktrees/`(다른 Anima와 공유 가능. `node_modules`나 빌드 산출물을 만드는 저장소는 반드시 여기) 또는 `/tmp/`에 만듭니다. canonical checkout에 대한 조작은 `git worktree add`와 읽기로 한정합니다
- worktree에서 merge하기 전에 canonical checkout이 clean인지 확인합니다. dirty이면 변경하지 말고 보고합니다
- 명시적 지시 없이 다른 작업자의 변경을 stash, 폐기 또는 덮어쓰지 않습니다
