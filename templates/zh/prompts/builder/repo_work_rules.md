### 仓库作业规则

- canonical checkout 的 `main` / `master` 仅供参照。实现、验证、commit 必须始终在专用的 `git worktree` 中进行
- 一次性作业产物（克隆、构建产物、验证日志、临时脚本）放在环境变量 `ANIMAWORKS_SCRATCH_DIR` 下，不要在 `state/` 正下方创建作业用目录。scratch 会在任务结束时或放置 24 小时后自动删除
- 只在 `{data_dir}/companies/<会社>/shared/worktrees/` 中创建与其他 anima 共享的 worktree，作业后用 `git worktree remove` 清理。不共享的 worktree 放在 scratch 中，24 小时未被触碰的 worktree 会被自动删除。对 canonical checkout 的操作仅限于 `git worktree add` 和参照
- 包缓存使用已配置的共用缓存（`UV_CACHE_DIR` 等），不要为每个 PR 创建缓存目录
- 从 worktree 进行 merge 时，先确认 canonical checkout 是 clean 的再执行。若为 dirty，则不做任何更改并报告
- 不要擅自 stash、丢弃或覆盖他人的更改
