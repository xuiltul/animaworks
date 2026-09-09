### リポジトリ作業ルール

- canonical checkout の `main` / `master` は参照専用。実装・検証・commitは必ず専用の `git worktree` で行う
- worktree は `{data_dir}/companies/<会社>/shared/worktrees/`（他の anima と共有できる場所。`node_modules` やビルド成果物を作るリポジトリは必ずここ）か `/tmp/` に作る。canonical checkout への操作は `git worktree add` と参照に限る
- worktreeからのmergeは、canonical checkoutがcleanであることを確認してから行う。dirtyなら変更せず報告する
- 他者の変更を独断でstash・破棄・上書きしない
