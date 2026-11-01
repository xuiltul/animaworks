# 开发工程师指南

## 实现通道
- 为避免与其他作业冲突，作业应在 worktree 中隔离进行。worktree 创建在 `$ANIMAWORKS_SCRATCH_DIR` 中，仅将需要交接给其他 Anima 的 worktree 创建在 `shared/worktrees/` 中（`common_knowledge/operations/scratch-and-worktrees.md`）。
- 对现有代码的修改应保持在必要最小限度，范围外的修正应作为单独任务记录。
- 对多次重复的作业应将其流程化，并遵循利用机器的工作流规范。

## PR 的创建与说明
- 汇总目的、变更内容、验证方法后创建 PR。
- 列出变更文件，力求说明便于审阅者跟进。

## 自己 PR 的 CI 监控与自主修正
- 自己 PR 的 CI 变红时不要放任不管，应调查原因并修正。
- 发生冲突时通过 rebase 或 merge 解决。

## 完成报告的格式
- 必须包含变更文件列表，以及执行的验证命令及其结果。

## 参考
- 仓库特有的规范请参照仓库根目录的 CLAUDE.md 或配置文件。
- 作业的放置位置: read_memory_file(path="common_knowledge/operations/workspace-guide.md")
