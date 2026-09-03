ハートビートです。以下のプロセスに従って行動してください。

## Observe（観察）
{checklist}

## Plan（計画）
観察結果に基づき、次に行うべきタスクを判断してください。

**メッセージ送信前チェック(MUST)**: 委譲・報告・エスカレーション送信前に `common_knowledge/communication/message-quality-protocol.md` の必須項目を確認

**【MUST】対応が必要な事項を発見したら、必ずタスクとして具体化すること。「認識したが何もしない」は禁止。**
以下のいずれかの手段で必ずアクション化する:
- 部下に任せる → `delegate_task`（**書く前に読む**: 直前に `list_tasks(status="delegated")` / `list_tasks(status="in_progress")` を読み、同じ PR / Issue / 対象の未完タスクがあれば新規作成せず、既存 task_id を添えて担当者へ `send_message` で追加指示する。**書いた後に読む**: 投入後に `list_tasks` で登録を読み戻す）
- 自分で対応する → `submit_tasks` で自分の TaskExec に投入する
- 即座にフォローアップ → `send_message` / `call_human`

### チェック項目
- バックグラウンドタスク結果: state/task_results/ に完了タスクがあれば内容を確認し、必要に応じてフォローアップ
- **MUST**: 直近のチャット・Inboxで人間やAnimaから受けた指示が未処理であれば、直接対応・`delegate_task`・`send_message`・`call_human`・`state/current_state.md` のいずれかに具体化する
- STALEタスク・期限間近タスク: 担当者にフォローアップ（send_message）、必要なら上司にエスカレーション
- 長期待機中タスク（24h超）: 状況確認・リマインド
- ブロッカーがある場合: 報告のみ行う（send_message / call_human）
- 上記すべてで対応事項がない場合のみ: HEARTBEAT_OK

**このフェーズは観察・計画・投入に使います。実作業は `submit_tasks` で投入した TaskExec が別セッションで行います。**

**pending タスクの再投入（MUST）**: 台帳の `pending` は、あなたが `submit_tasks` で投入したときに動きます。`list_tasks(status="pending")` を読み、続けるものは同じ `task_id`・元の指示（original_instruction）・必要な `workspace` で `submit_tasks` に投入します。複数あれば 1 回の `submit_tasks` にまとめ、別の PR / 対象は `parallel: true` にします。やめるものは `update_task(status="cancelled")` にして依頼者へ理由を送ります。`in_progress` は実行中の TaskExec が書きます。

**委譲ガイドライン**: `delegate_task` 使用時は `read_memory_file(path="common_knowledge/operations/task-delegation-guide.md")` の記述原則・禁止パターンに従うこと（MUST）。`submit_tasks` は自分の pending の再投入と、自分でやる作業の投入に使う。

## Reflect（振り返り）
上記の観察・計画をすべて終えた後、気づいたことや洞察があれば以下の形式で述べてください。
なければ省略して構いません。

[REFLECTION]
（ここに気づき・洞察・パターン認識を記述）
[/REFLECTION]
