# 正本タスクのアーキテクチャ

## 永続的な正本は一つ

LLM タスクの正本はホスト管理の TaskStore。タスク ID、完全な実行入力、依存関係、結果、実行試行、委譲エイリアス、永続的な起床通知を必要な単位で一括確定する。別々のファイル実行キューと上司台帳を突き合わせる構成ではない。

確認は `list_tasks` / `task_tracker`、変更は `submit_tasks` / `delegate_task` / `update_task` を使う。DB やタスクファイルを直接編集しない。`backlog_task` は追跡のみの作業を登録し、実行権は取得しない。

## 実行契約

1. 新規 `submit_tasks` はタスクと完全な入力を原子的に公開する。原指示・制約・workspace・完了条件を保持する。
2. ホストが依存関係を確認し、一意の試行トークンで実行権を取得する。`in_progress` を設定するのはホストだけ。
3. エージェントは `update_task` で `done` / `pending` / `cancelled` を宣言する。古い試行は新しい試行の完了や受理済み結果を上書きできない。
4. 依存先の完了と永続的な起床はホストが扱い、定期 Heartbeat を必要としない。取消・異常終了・中断は証跡と要対応理由を残す。
5. 中断タスクは盲目的に再試行しない。既済操作を確認して原因を解消したら、`submit_tasks(..., tasks=[{"task_id": "ID", "resume": true}])` で同じ未終了タスクを明示的に再開する。入力と履歴は保持される。resume なしの再配信は冪等。

上司の委譲ビューは部下の正本タスクへのエイリアス。別の可変台帳や Heartbeat 同期を介さず、双方に最新状態が反映される。依存先の終了が成功とは限らず、取消済みを done と見なして後続を動かさない。

## 作業文脈と証跡

`state/current_state.md` は簡潔な作業文脈で、タスクの正本ではない。観察・計画・ブロッカーを残し、通常のセッション境界では保持する。恒久知識と手順は専用の記憶領域に保存する。

TaskExec の結果要約は `state/task_results/{task_id}/{attempt_token}.md` に置き、TaskStore が受理済み結果を選ぶ。ファイル名や古い要約だけで完了を証明できない。活動ログと原指示を証跡として保持する。

## 旧保存先とコマンドタスク

旧 `state/task_queue.jsonl` と `state/pending/` は移行・エクスポート用の証跡のみであり、保存する。移行は旧書き込み処理の停止とバックアップ後に運用者が明示的に行う。任意の読み取りで稼働中の旧データをインポートしない。

長時間コマンドツールは別。`animaworks-tool submit` は引き続き `state/background_tasks/pending/` を使い、BackgroundTaskManager がコマンド状態・通知を保存する。LLM タスクの変更を理由にこのファイル経路を撤去しない。

ツール例は `reference/operations/task-management.md`、コマンド実行は `operations/background-tasks.md` を参照。
