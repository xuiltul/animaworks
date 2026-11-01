# 工作物的放置位置（scratch・共享 worktree・缓存）

一次性工作物与需要保留之物的放置位置。

## 放置位置

| 用途 | 放置位置 |
|---|---|
| 仅自己使用的 worktree・clone・构建・验证日志・临时脚本 | `$ANIMAWORKS_SCRATCH_DIR` |
| 交给其他 Anima 进行评审・验证的 worktree | `{data_dir}/companies/<company>/shared/worktrees/` |
| 日后回看的手顺・脚本 | 自己的 `procedures/` |
| 日后回看的证据（报告・截图・汇总） | `shared/evidence/` 或 PR 评论 |
| `state/` | 仅常设的状态文件（`current_state.md`・`background_tasks/` 等） |

`$ANIMAWORKS_SCRATCH_DIR` 会在每次执行时自动准备。任务执行中指向 `animas/<self>/scratch/task-<task_id>/`，其他情况（chat・heartbeat・inbox・cron）指向 `animas/<self>/scratch/<type>/`。

```bash
WT="$ANIMAWORKS_SCRATCH_DIR/myrepo"
git -C /path/to/canonical worktree add --detach "$WT" origin/main
TMP="$(mktemp -d -p "$ANIMAWORKS_SCRATCH_DIR")"
```

## 自动删除

- 任务用的 scratch 会在任务完成・取消时消失。其中创建的 worktree 注册也会从父仓库中移除。
- scratch 若 24 小时未被触碰就会消失。
- `shared/worktrees/` 若 24 小时未被触碰，即使有未提交变更也会消失（使用中的会保留）。
- 清理会自动进行，因此无需在作业最后执行删除命令。需要保留的内容请移到上表“日后回看”行的位置。

## 包缓存

`UV_CACHE_DIR`・`PIP_CACHE_DIR`・`npm_config_cache`・`npm_config_store_dir`（pnpm）・`GRADLE_USER_HOME`・`NUGET_PACKAGES`・`COMPOSER_CACHE_DIR` 已自动配置为指向共用的 `{data_dir}/cache/<type>`。直接使用即可。
