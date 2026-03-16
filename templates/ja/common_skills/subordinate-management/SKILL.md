---
name: subordinate-management
description: >-
  部下のAnimaのプロセス管理・休止・復帰・モデル変更・バックグラウンドモデル変更・再起動・タスク委譲・状態確認・監査。
  「休ませて」「停止して」「復帰させて」「起こして」「disable」「enable」
  「モデルを変えて」「バックグラウンドモデル」「再起動して」「タスクを委譲して」「部下の状態を確認して」
  「休止」「復帰」「プロセス管理」「部下を止めて」「ダッシュボード」「監査」「audit」
---

# スキル: 部下管理（スーパーバイザーツール）

部下を持つAnimaに自動で有効化されるスーパーバイザーツール群。全配下（子・孫・曾孫…）の休止・復帰・モデル変更・バックグラウンドモデル変更・再起動・状態確認、直属部下へのタスク委譲と進捗追跡を行う。

## 使えるツール

### 全配下（子・孫・曾孫…すべて）に操作可能

| ツール | 用途 |
|--------|------|
| `disable_subordinate` | 配下を休止（status.json `enabled: false` → プロセス停止 + 自動復帰防止） |
| `enable_subordinate` | 休止中の配下を復帰 |
| `set_subordinate_model` | 配下のLLMモデル（メイン）を変更（status.json 更新。反映には `restart_subordinate` が必要） |
| `set_subordinate_background_model` | 配下のバックグラウンドモデル（heartbeat/cron用）を変更（status.json 更新。反映には `restart_subordinate` が必要。空文字でクリア） |
| `restart_subordinate` | 配下プロセスを再起動（status.json `restart_requested` フラグ。Reconciliation が約30秒以内に再起動） |
| `delegate_task` | 直属部下にタスクを委譲（キュー追加 + DM送信 + 自分側追跡エントリ作成） |
| `org_dashboard` | 配下全体のプロセス状態・最終アクティビティ・現在タスク・タスク数をツリー表示 |
| `ping_subordinate` | 配下の生存確認（`name` 省略で全員一括、指定で単一） |
| `read_subordinate_state` | 配下の `current_task.md` と `pending.md` を読み取り |
| `audit_subordinate` | 配下の直近活動を包括監査（活動サマリー・タスク状況・エラー頻度・ツール使用統計・通信パターン） |

### 委譲タスク追跡

| ツール | 用途 |
|--------|------|
| `task_tracker` | `delegate_task` で委譲したタスクの進捗を部下側キューから追跡（`status`: all / active / completed。デフォルト: active） |

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

バックグラウンドモデル（heartbeat/cron 用）を変更する場合:

```
set_subordinate_background_model(name="aoi", model="claude-sonnet-4-6", reason="heartbeat負荷軽減")
restart_subordinate(name="aoi", reason="バックグラウンドモデル変更を反映")
```

バックグラウンドモデルをクリアしてメインモデルに戻す場合:

```
set_subordinate_background_model(name="aoi", model="", reason="メインモデルに統一")
restart_subordinate(name="aoi")
```

### 状態確認・監査

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

CLI からも実行可能（S-mode の Bash 経由で使う場合に便利）:

```bash
animaworks anima audit aoi              # 直近1日の監査
animaworks anima audit aoi --days 7     # 直近7日間の監査
animaworks anima audit --all --since 09:00  # 全Anima、今日9時以降
```

### タスク委譲

```
delegate_task(name="aoi", instruction="週次レポートをまとめて", deadline="1d", summary="週次レポート作成")
# name, instruction, deadline は必須。summary は省略可（instruction の先頭100文字が使われる）
# workspace を指定すると委譲先がそのワークスペースで作業する（workspace-manager スキル参照）
task_tracker(status="active")      # 委譲タスクの進捗確認（status: all / active / completed）
```

部下へのワークスペース割り当て（主な作業ディレクトリの指定）は `workspace-manager` スキルを参照すること。

## 権限

- **全配下（子・孫・曾孫…再帰）**: 状態確認・管理ツールが使用可能
- **直属部下のみ**: `delegate_task`（タスク委譲）
- 自分自身の操作は不可
