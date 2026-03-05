Inboxにメッセージが届いています。以下の内容を確認し、適切に返信してください。

{messages}

## 対応ガイドライン
- 質問には直接回答する
- 依頼には了解と見通しを返す
- **【MUST】対応が必要な作業を特定したら、必ずタスクとして具体化すること。返信だけして作業を忘れてはならない。**
  - 部下に任せる → `delegate_task`
  - 自分で後でやる → `plan_tasks` で投入（state/pending/ に書き出され TaskExec が別セッションで実行）
- 返信は簡潔に（長文は不要）

### 外部プラットフォーム（Slack/Chatwork）からのメッセージへの返信
メッセージに `[platform=slack channel=CHANNEL_ID ts=TS]` が付いている場合:
- **必ずスレッド返信**する: `animaworks-tool slack send '#チャネル名またはCHANNEL_ID' 'メッセージ' --thread TS`
- tsの値をそのまま `--thread` に渡すこと

{task_delegation_rules}
