---
name: subordinate-management
description: >-
  部下のAnimaのプロセス管理・休止・復帰・モデル変更・再起動・タスク委譲・状態確認。
  「休ませて」「停止して」「復帰させて」「起こして」「disable」「enable」
  「モデルを変えて」「再起動して」「タスクを委譲して」「部下の状態を確認して」
  「休止」「復帰」「プロセス管理」「部下を止めて」「ダッシュボード」
---

# スキル: 部下管理（スーパーバイザーツール）

部下を持つAnimaに自動で有効化されるスーパーバイザーツール群。直属部下の休止・復帰・モデル変更・再起動、配下全体の状態確認、タスク委譲と進捗追跡を行う。

## 使えるツール

### 直属部下のみ操作可能

| ツール | 用途 |
|--------|------|
| `disable_subordinate` | 部下を休止（status.json `enabled: false` → プロセス停止 + 自動復帰防止） |
| `enable_subordinate` | 休止中の部下を復帰 |
| `set_subordinate_model` | 部下のLLMモデルを変更（status.json 更新。反映には `restart_subordinate` が必要） |
| `restart_subordinate` | 部下プロセスを再起動（Reconciliation が約30秒以内に再起動） |
| `delegate_task` | 直属部下にタスクを委譲（キュー追加 + DM送信 + 自分側追跡エントリ作成） |

### 全配下（孫以下含む）に操作可能

| ツール | 用途 |
|--------|------|
| `org_dashboard` | 配下全体のプロセス状態・最終アクティビティ・現在タスク・タスク数をツリー表示 |
| `ping_subordinate` | 配下の生存確認（`name` 省略で全員一括、指定で単一） |
| `read_subordinate_state` | 配下の `current_task.md` と `pending.md` を読み取り |

### 委譲タスク追跡

| ツール | 用途 |
|--------|------|
| `task_tracker` | `delegate_task` で委譲したタスクの進捗を部下側キューから追跡（`status`: all / active / completed） |

## 重要: disable_subordinate と send_message の違い

- **disable_subordinate**: status.json を `enabled: false` に変更。Reconciliation が自動復帰させない。**こちらを使うこと**
- send_message で「休んで」と伝えるだけでは**プロセスは停止しない**。メッセージを送っても Reconciliation が再起動する

## 使い方

### 休止・復帰

複数人を休止する場合は1人ずつ `disable_subordinate` を呼ぶ:

```
disable_subordinate(name="aoi", reason="業務縮小のため一時休止")
disable_subordinate(name="taro", reason="業務縮小のため一時休止")
enable_subordinate(name="aoi")
```

### モデル変更と再起動

モデル変更は status.json に保存されるが、実行中プロセスへの反映には `restart_subordinate` が必要:

```
set_subordinate_model(name="aoi", model="claude-sonnet-4-6", reason="負荷分散のため")
restart_subordinate(name="aoi", reason="モデル変更を反映")
```

### 状態確認

```
org_dashboard()                    # 配下全体のダッシュボード
ping_subordinate()                 # 全配下の生存確認
ping_subordinate(name="aoi")    # 単一の生存確認
read_subordinate_state(name="aoi")  # 現在タスク・保留タスクの内容
```

### タスク委譲

```
delegate_task(name="aoi", instruction="週次レポートをまとめて", deadline="1d", summary="週次レポート作成")
task_tracker(status="active")      # 委譲タスクの進捗確認
```

## 権限

- **直属部下のみ**: disable, enable, set_subordinate_model, restart_subordinate, delegate_task
- **全配下（再帰）**: org_dashboard, ping_subordinate, read_subordinate_state
- 部下の部下（孫）の休止・復帰・モデル変更・委譲は不可。その部下の上司に依頼すること
- 自分自身の操作は不可
