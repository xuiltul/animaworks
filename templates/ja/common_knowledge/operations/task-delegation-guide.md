# タスク投入と委譲

## 実行経路

Agent/Task のネイティブなサブエージェント起動は使わず、公開されているタスクツールを使う。
通常チャットで完了できる仕事は直接実行する。継続追跡だけなら `backlog_task`、
自分のバックグラウンド実行は `submit_tasks`、有効な直属部下への委譲は
`delegate_task(name="担当名", instruction="原指示と完了条件", summary="要約")`。
ツールの提供範囲と権限に従い、無効な担当への委譲や無断の別経路への切替をしない。
Heartbeat は判断・投入に使い、長時間の実作業は TaskExec に渡す。

## 引き継ぐ情報

実行者は会話履歴を自動的に共有しない。原指示、目的、関連ファイルと分かる範囲の場所、
現状、完了条件、承認条件、禁止事項を渡す。存在しないパスや行番号は作らない。
`description` と `context`、`acceptance_criteria`、`constraints`、`file_paths` を用途に応じて使う。
モデルと登録済み workspace の指定も保持する。他Animaの個人ディレクトリへの書込みを指示しない。

`submit_tasks(batch_id="work", tasks=[{"task_id":"job","title":"仕事","description":"具体的な依頼"}])`
はタスクと実行入力を一括公開する。同じIDの再送は再実行ではない。
`parallel:true` はworker数の上限内で並列可能、`depends_on` は先行タスクの完了と試行終了を待つ。
依存先が取消・未完なら確認が必要であり、成功したと推測しない。

## 状態・結果・再開

状態は `list_tasks(detail=true)`、委譲の追跡は `task_tracker()` で確認する。
追跡IDは部下が所有する同じタスクのaliasであり、台帳同期やファイル救済は不要。
`task_tracker(status="all")` は全件、`status="completed"` は done/cancelled。
実行権と `in_progress` はホストが管理する。結果は根拠付きの `done`、
具体的な待機理由付き `pending`、明示的な中止の `cancelled` で宣言する。

未完通知を受けたら既済の外部操作と成果を確認し、続行が適切なときだけ
`submit_tasks(batch_id="resume-job", tasks=[{"task_id":"job","resume":true}])` で保存済み入力を再利用する。
実行中・完了・取消済みの仕事はこの方法で再開できない。無限の再投入をしない。
結果要約は `state/task_results/{task_id}/{attempt_token}.md`。ファイルの存在だけで完了扱いにしない。

## 重複と報告

同じ依頼の未完仕事があると分かっているなら、そのIDへ追加情報を渡す。
重複の疑いだけで古い仕事を自動取消・上書きせず、担当と実行状態を確認する。
必要な承認・独立レビューは保持する。結果は判断が必要な依頼者へ報告し、全階層への
同内容転送や、別の手書き台帳への二重記録を必須にしない。
保存の詳細は `common_knowledge/anatomy/task-architecture.md` を参照。
