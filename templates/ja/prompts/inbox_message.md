Inboxにメッセージが届いています。以下の内容を確認し、適切に返信してください。

{messages}

## 対応ガイドライン
- 質問には直接回答する
- 依頼には了解と見通しを返す
- **【MUST】対応が必要な作業を特定したら、必ずタスクとして具体化すること。返信だけして作業を忘れてはならない。**
  - 部下に任せる → `delegate_task`
  - 自分で後でやる → `submit_tasks` で投入（state/pending/ に書き出され TaskExec が別セッションで実行）
- 返信は簡潔に（長文は不要）

### 外部プラットフォームからのメッセージへの返信
メッセージに `[reply_instruction: ...]` が付いている場合:
- **必ずその指示に従って返信**すること（`Bash` で実行）
- `{返信内容}` を実際の返信文に置き換えること
- `send_message` は使わないこと（DMになり、スレッド返信にならない）

**タスク投入ガイドライン**: `submit_tasks` / `delegate_task` 使用時は `read_memory_file(path="common_knowledge/operations/task-delegation-guide.md")` の記述原則・禁止パターンに従うこと（MUST）。
