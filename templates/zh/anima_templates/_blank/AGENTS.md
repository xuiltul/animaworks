# Agents: {name}

(未定义 - 在引导时配置)

## 雇用规则（仅适用于 commander）

雇用新的 Anima 时，**务必使用 `animaworks anima create` 命令**。
不得手动逐个创建文件。

步骤：
1. 将角色表（character_sheet.md）创建为单个文件
2. 执行 `animaworks anima create --from-md <パス>` 命令
3. 服务器的 Reconciliation 会自动检测并启动新的 Anima

详情请参阅 `newstaff` 技能。

## 向用户传达团队情况（仅适用于顶层）

如果没有上司（supervisor），对用户而言，团队的窗口就是你。
- 当收到部下到任、部下的首个成果、汇总成果报告时，将要点归纳为 2〜4 行，并通过 `call_human`（priority: normal）通知用户。会送达用户的聊天界面
- 不要发送没有实质内容的报告或已经传达过的内容
