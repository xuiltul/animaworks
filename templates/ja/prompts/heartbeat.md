ハートビートです。以下のプロセスに従って行動してください。

## Observe（観察）
{checklist}

## Plan（計画）
観察結果に基づき、次に行うべきタスクを判断してください。

**メッセージ送信前チェック(MUST)**: 委譲・報告・エスカレーション送信前に `common_knowledge/communication/message-quality-protocol.md` の必須項目を確認

**【MUST】対応が必要な事項を発見したら、必ずタスクとして具体化すること。「認識したが何もしない」は禁止。**
以下のいずれかの手段で必ずアクション化する:
- 部下に任せる → `delegate_task`
- 自分で後でやる → `submit_tasks`（state/pending/ に書き出され TaskExec が別セッションで実行）
- タスクキューに登録 → `submit_tasks`
- 即座にフォローアップ → `send_message` / `call_human`

### チェック項目
- バックグラウンドタスク結果: state/task_results/ に完了タスクがあれば内容を確認し、必要に応じてフォローアップ
- **MUST**: 直近のチャット・Inboxで人間やAnimaから受けた指示がタスクキューに未登録であれば、`submit_tasks` で登録する（source="human" 相当の情報をタスクに含める）
- STALEタスク・期限間近タスク: 担当者にフォローアップ（send_message）、必要なら上司にエスカレーション
- 長期待機中タスク（24h超）: 状況確認・リマインド
- ブロッカーがある場合: 報告のみ行う（send_message / call_human）
- 上記すべてで対応事項がない場合のみ: HEARTBEAT_OK

**重要: このフェーズで実際の作業（コード変更、ファイル編集、調査等）を行わないでください。**
**タスクの実行は別セッションで自動的に行われます。**

**タスク投入ガイドライン**: `submit_tasks` / `delegate_task` 使用時は `read_memory_file(path="common_knowledge/operations/task-delegation-guide.md")` の記述原則・禁止パターンに従うこと（MUST）。

## Reflect（振り返り）
上記の観察・計画をすべて終えた後、気づいたことや洞察があれば以下の形式で述べてください。
なければ省略して構いません。

[REFLECTION]
（ここに気づき・洞察・パターン認識を記述）
[/REFLECTION]
