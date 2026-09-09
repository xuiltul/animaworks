# タスク管理の方法

## 正本は一つ

確認には `list_tasks(detail=true)` または `animaworks-tool task list` を使う。ホスト管理の TaskStore に指示・依存関係・実行試行・結果・委譲エイリアスを永続化する。データベースの直接編集、実行権の捏造、ファイルを書いてのキュー修復は禁止。旧 `state/task_queue.jsonl` と `state/pending/` は移行・エクスポート用の証跡であり、稼働中の投入先ではない。運用者による移行のため保存する。

通常チャットで処理できる依頼は直接対応してよい。バックグラウンド実行・並列化・継続追跡が必要な場合にだけタスクを登録する。人間由来の依頼を最優先とし、同等の優先度なら上司の依頼を同僚より優先する。引き継ぎでは原指示・完了条件・制約・必要な文脈を保持する。

## 実行経路を選ぶ

- Inbox はメッセージと軽量な返信を処理する。
- Heartbeat は意味のある変化を確認して対応を判断する。長時間のコーディングや大量のツール実行は行わず、自分の TaskExec には `submit_tasks`、直属の部下には `delegate_task` を使う。
- TaskExec は永続化されたタスクをツール付きで実行する。実行権の取得、並列数、依存関係、取消、試行の復旧はホストが管理し、定期 Heartbeat に依存しない。
- Agent/Task のサブエージェント起動ツールは無効。上記の投入・委譲ツールを使う。

## 投入と確認

`submit_tasks` の実行者は部下ではなく**自分自身の TaskExec**。

```
submit_tasks(batch_id="report-build", tasks=[
  {"task_id": "collect", "title": "根拠収集", "description": "依頼された根拠を出典付きで収集する。", "parallel": true},
  {"task_id": "report", "title": "報告作成", "description": "収集結果から依頼された報告書を作る。", "depends_on": ["collect"]}
])
list_tasks(detail=true)
```

新規タスクには `task_id`、`title`、`description` が必要。任意項目は `context`、`acceptance_criteria`、`constraints`、`file_paths`、`workspace`、`parallel`、`depends_on`、`reply_to`、`model`。`workspace` は登録済みワークスペースのエイリアス。モデル選択は通常ランタイム設定に従う。長い原指示を短い要約に置換しない。

投入時にバッチを検証し、タスクと実行入力を一括で確定する。同じ投入の再配信は冪等であり、再試行ではない。依存先が正常完了するまで後続は実行されない。ファイルの不在や `pending` 表示だけから実行可能と判断しない。

## 結果宣言と明示的な再開

```
update_task(task_id="TASK_ID", status="done", summary="検証済みの結果", result="根拠と成果物の場所")
update_task(task_id="TASK_ID", status="pending", summary="指定した入力を待っている")
update_task(task_id="TASK_ID", status="cancelled", summary="不要になった理由")
```

`in_progress` は実行権取得時にホストが設定する閲覧用状態。`update_task` で設定しない。完了宣言なしで試行が終了したタスクは、要対応理由を伴う pending になる場合がある。pending は自動再試行の約束ではない。再開のために別 ID で複製しない。

中断理由を解消した後、同じ未終了タスクを明示的に再開する。

```
submit_tasks(batch_id="resume-report", tasks=[{"task_id": "TASK_ID", "resume": true}])
```

保存済み入力を再利用し履歴を保持する。稼働中の試行は再開できず、完了・取消済みタスクもこの方法では再開できない。依存先が取消・要対応なら詳細を確認し、依頼者に確認するか不要になった作業を取り消す。成功を捏造しない。

進められないときは事実、試したこと、不足する権限・情報、次の一手を依頼者に伝え、同じ失敗を繰り返さない。関連知識の検索は必要な場合に行い、儀式化しない。待機中は他の許可済み作業に取り組んでよい。委譲された仕事の完了は依頼者に報告し、重複通知や不要な了解返信を避ける。

## 部下への委譲

```
delegate_task(name="dave", instruction="API テストを実施し検証した結果を報告する", summary="API テスト")
task_tracker()
```

部下が所有する一つのタスクと、上司から見えるエイリアスを作る。双方に同じ最新状態が即座に反映され、別台帳や Heartbeat での同期は不要。`task_tracker(status="all")` は終了済みを含み、`status="completed"` は done/cancelled を表示する。指示が不明なら委譲元に確認し、完了時に結果を報告する。

## 作業文脈と結果

`state/current_state.md` には観察・文脈・計画・ブロッカーを簡潔に残す。タスク一覧の複製や恒久的な手順を置かない。作業文脈がなければ `status: idle`。通常のセッション境界では保持され、表示制限とディスク整理の制限は別（`anatomy/working-memory.md`）。

TaskExec の結果要約は `state/task_results/{task_id}/{attempt_token}.md` に保存される。ホストが受理した試行の結果を後続に渡し、古いファイルを任意に採用しない。結果ファイルの捏造や、存在だけを根拠にした完了判定は禁止。活動ログ・エピソードは証跡として残り、各状態遷移を手動で二重記録する義務はない。

## 長時間コマンドツールは別経路

画像生成や run_command など対応する長時間外部ツールには `animaworks-tool submit TOOL ...` を使う。これは `submit_tasks` とは別で、コマンド記述子は引き続き `state/background_tasks/pending/` に保存される。BackgroundTaskManager は `state/background_tasks/{task_id}.json` に `running` / `completed` / `failed` を記録する。`list_background_tasks` / `check_background_task` で確認する。このファイル経路と通知を維持する。詳細は `operations/background-tasks.md`。
