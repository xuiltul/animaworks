---
name: subordinate-management
description: >-
  部下Anima的监督者操作。执行暂停、恢复、模型变更、重启、任务委派、状态读取、审计。
  Use when: 需要禁用/重新启用部下、变更主模型或BG模型、重启进程、delegate_task、确认组织仪表板时。
---


# 技能：部下管理（监督者工具）

拥有部下的Anima会自动启用的监督者工具群。对全部下属（子、孙、曾孙……）执行暂停、恢复、模型变更、后台模型变更、重启、状态确认，以及向直属部下委派任务和跟踪进度。

## 可用工具

### 可对全部下属（子、孙、曾孙……所有）操作

| 工具 | 用途 |
|--------|------|
| `disable_subordinate` | 向 root 请求 status.json `enabled: false` 的配置（进程停止 + 防止自动恢复） |
| `enable_subordinate` | 恢复暂停中的下属 |
| `set_subordinate_model` | 向 root 请求更新 status.json 的主模型并 reload 启动中的进程 |
| `set_subordinate_background_model` | 向 root 请求更新 heartbeat/cron 用模型。从下次后台任务起应用（用空字符串清除） |
| `restart_subordinate` | 重启下属进程（status.json `restart_requested` 标志。Reconciliation 会在约30秒内重启） |
| `delegate_task` | 向直属部下委派任务（队列追加 + DM发送 + 自己侧跟踪条目创建） |
| `org_dashboard` | 以树形显示全部下属的进程状态、最终活动、当前任务、任务数 |
| `ping_subordinate` | 确认下属存活（`name` 省略时全员批量，指定时单个） |
| `read_subordinate_state` | 读取下属的 `current_state.md` |
| `audit_subordinate` | 对下属的近期活动进行全面审计（活动摘要、任务状况、错误频率、工具使用统计、通信模式） |

### 委派任务跟踪

| 工具 | 用途 |
|--------|------|
| `task_tracker` | 从部下侧队列跟踪通过 `delegate_task` 委派的任务进度（`status`: all / active / completed。默认: active） |

委派方的跟踪卡片参照委派目标的正式任务记录。确认状态时使用 `task_tracker`。

## 重要：disable_subordinate 与 send_message 的区别

- **disable_subordinate**: 将 status.json 变更为 `enabled: false`。Reconciliation 不会自动恢复。**应使用此工具**
- 仅通过 send_message 告知“休息”**进程不会停止**。即使发送消息，Reconciliation 也会重启

## 使用方法

### 暂停・恢复

暂停多人时逐个调用 `disable_subordinate`:

```
disable_subordinate(name="aoi", reason="業務縮小のため一時休止")
disable_subordinate(name="taro", reason="業務縮小のため一時休止")
enable_subordinate(name="aoi")
```

### 模型变更与重启

这些工具向 root 请求更新 root 所有的 `status.json`。不要从 Anima 进程直接编辑。主模型会 reload 到启动中的进程，停止中的 Anima 会在下次启动时读取新配置:

```
set_subordinate_model(name="aoi", model="claude-sonnet-5-5", reason="負荷分散のため")
```

后台模型（用于 heartbeat/cron）从下次任务运行器启动起应用，执行中的任务会照常完成:

```
set_subordinate_background_model(name="aoi", model="claude-sonnet-5-5", reason="heartbeat負荷軽減")
```

清除后台模型并返回主模型时:

```
set_subordinate_background_model(name="aoi", model="", reason="メインモデルに統一")
```

仅在确实需要重启整个进程时，另行使用 `restart_subordinate`。

### 状态确认・审计

```
org_dashboard()                         # 配下全体のダッシュボード
ping_subordinate()                      # 全配下の生存確認
ping_subordinate(name="aoi")            # 単一の生存確認
read_subordinate_state(name="aoi")      # 現在タスク・保留タスクの内容
audit_subordinate(name="aoi")           # 直近1日の包括監査レポート
audit_subordinate(name="aoi", days=7)   # 直近7日間の監査（days は 1〜30）
audit_subordinate(since="09:00")        # 全配下の今日9時以降の監査
audit_subordinate(name="aoi", since="13:00")  # aoi の今日13時以降
```

也可从 CLI 执行（通过 S/C/D/G-mode 的 Bash 使用时很方便）:

```bash
animaworks anima audit aoi              # 直近1日の監査
animaworks anima audit aoi --days 7     # 直近7日間の監査
animaworks anima audit --all --since 09:00  # 全Anima、今日9時以降
```

### 任务委派

```
delegate_task(name="aoi", instruction="週次レポートをまとめて", summary="週次レポート作成")
# 必須: `name`, `instruction`。任意: `summary`, `workspace`, `acceptance_criteria`, `model`
# workspace を指定すると委譲先がそのワークスペースで作業する（workspace-manager スキル参照）
task_tracker(status="active")      # 委譲タスクの進捗確認（status: all / active / completed）
```

向部下分配工作区（指定主要工作目录）请参照 `workspace-manager` 技能。

## 权限

- **全部下属（子、孙、曾孙……递归）**: 可使用状态确认・管理工具
- **仅直属部下**: `delegate_task`（任务委派）
- 不可操作自身
