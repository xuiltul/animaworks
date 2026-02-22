---
name: subordinate-management
description: >-
  部下のAnimaを「休止」「復帰」させるプロセス管理スキル。
  「休ませて」「停止して」「復帰させて」「起こして」「disable」「enable」
  「休止」「復帰」「プロセス管理」「部下を止めて」
---

# スキル: 部下の休止・復帰管理

## 使えるツール

| ツール | 用途 |
|--------|------|
| `disable_subordinate` | 部下を休止（プロセス停止 + 自動復帰防止） |
| `enable_subordinate` | 休止中の部下を復帰 |

## 重要: disable_subordinate と send_message の違い

- **disable_subordinate**: status.json を `enabled: false` に変更。Reconciliation が自動復帰させない。**こちらを使うこと**
- send_message で「休んで」と伝えるだけでは**プロセスは停止しない**。メッセージを送っても Reconciliation が再起動する

## 使い方

### 複数の部下を休止させる場合

1人ずつ `disable_subordinate` を呼ぶ:

```
disable_subordinate(name="hinata", reason="業務縮小のため一時休止")
disable_subordinate(name="natsume", reason="業務縮小のため一時休止")
```

### 特定の部下だけ残して他を休止する場合

残すべき部下以外を1人ずつ `disable_subordinate` で休止する。

### 休止中の部下を復帰させる場合

```
enable_subordinate(name="hinata")
```

## 権限

- **自分の直属部下のみ**操作可能
- 部下の部下（孫）は直接操作不可。その部下の上司に依頼すること
- 自分自身の休止は不可
