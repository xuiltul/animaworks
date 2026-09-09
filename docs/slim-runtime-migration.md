# スリムruntimeへの移行

この変更はモデルの更新ではなく、AnimaWorks側の起動・想起・タスク保存の整理です。
S/C/D/G/X/A/B、自動ルーティング、背景モデル、明示override、許可済みfallbackを維持します。
本番データの自動移行や、組織全員のheartbeat停止は行いません。

## 起動前の重要事項

既存のタスクJSONLやLLM pendingファイルがある担当者は、明示的な移行が必要です。
新コードの通常起動・読取で勝手に取り込まず、移行が必要というエラーで停止します。
旧コードと新コードを同じランタイムで同時に実行しないでください。
`scripts/migrate_task_queue_teardown.py` の変更操作は廃止し、dry-runのみ残しています。

タスク正本は既存の `shared/taskboard.sqlite3` に統合しました。
LLMタスクの `state/pending/*.json` と `task_queue.jsonl` は実行条件ではありません。
旧ファイルは移行証拠として保持しますが、変更しても新runtimeには反映されません。
`state/background_tasks/pending/` の長時間CLIツール投入は別の仕組みで、変更していません。

## 停止・移行・照合

以下のパスと担当名は例です。対象runtimeと未使用のbackupパスを明示してください。

```sh
ANIMAWORKS_DATA_DIR=/path/to/runtime uv run animaworks task-store status --anima alice
ANIMAWORKS_DATA_DIR=/path/to/runtime uv run animaworks task-store quiesce --anima alice
```

quiesceは新しいclaimを止めます。既に動いている試行は止めません。
active_attemptsが0になるまで待ち、既存の通常手順で対象サーバー・ワーカーを停止します。
旧runtimeを使用中の場合も、全プロセスの停止を確認してから移行してください。

```sh
ANIMAWORKS_DATA_DIR=/path/to/runtime uv run animaworks task-store migrate \
  --anima alice --backup /new/path/before.sqlite3
```

移行はWALを含むSQLite backup、プロセスロック、短いtransactionを使用します。
不正行、実行中または生死不明のlease、競合した入力があれば停止します。
ただし完了・取消済み台帳の古いdescriptorは再投入せず、そのまま証跡として残します。
アーカイブの失敗は失敗履歴のまま非実行状態で保持し、未処理通知を再生成しません。
`completed` は `done` に正規化します。既知の完了heartbeat記録は非タスクとして、
pending直下・processing/failed/suppressed直下以外のJSONは補助資料として計数します。
これらも削除しません。現役の競合入力や不正なタスク行は引き続き移行を停止します。
移行件数・元指示全文・モデル・workspace・依存関係・追跡aliasを照合してください。
結果不明の旧試行は未完の確認対象として保持し、自動的には再実行しません。

タスクデータの移行だけでは、既存の手順書は置き換わりません。通常起動時の
shared template同期は既存ファイルを保持します。使用localeの更新テンプレートと
runtimeの `prompts/`・`reference/`・`common_knowledge/` を差分照合し、旧JSONL投入・
descriptor救済・無条件再投入の案内を更新してください。各担当の独自 `injection.md`・
`heartbeat.md`・役割にも同じ確認が必要です。顧客固有の指示、承認、業務制約は保持します。
`init --force` による一括上書きをこの照合の代わりにしないでください。

```sh
ANIMAWORKS_DATA_DIR=/path/to/runtime uv run animaworks task-store resume --anima alice
```

照合が済んだ担当だけ取得を再開し、新コードで起動します。
全員を一度に切り替える必要はありませんが、同じDBを旧新プロセスが共有しないようにします。

## タスクの公開契約

- `backlog_task`: 記録のみ。実行開始ではありません。
- `submit_tasks` / `delegate_task`: 指示と実行入力を一つのtransactionで公開します。
- 同一IDの再送は再実行になりません。終了済み仕事も再公開しません。
- workerがclaimし、試行IDを付けます。古い試行からの更新は拒否します。
- `update_task` の宣言は `done` / 理由付き `pending` / `cancelled`。
  `in_progress` はworker所有の状態です。
- 未完で終了しても自動反復せず、永続通知から次の判断を求めます。
  再開は `submit_tasks(batch_id="resume-ID", tasks=[{"task_id":"ID","resume":true}])`。
  指示・制約・modelは保存済み入力を使います。
- 実行中の試行を重ねて再開できません。完了・cancel済みは新しい依頼として新IDを使用します。
- 委譲の追跡IDは同じ仕事へのaliasです。別台帳との同期は不要です。
- 完了通知の失敗は完了を取り消しません。通知outboxから再送します。

DB claimは外部操作のexactly-onceを保証しません。通信失敗・プロセス停止で結果が
不明な送信や変更は、外部状態を確認してから再開してください。

## 想起・記憶の設定

新規既定値は `priming.profile="compact"`、framework指示の目標6,000推定tokens、
想起は別枠最大2,000です。案件本文・権限条件はこの目標のために切り詰めません。
比較時はglobalを `full` に保ち、少数担当の `status.json` の `priming_profile` を
`compact` にして切り替えられます。明示的な検索・読み書き・既存facts読取は維持します。

次の自動加工は既定で無効です。既存設定に明示されたtrueは維持されます。

- `consolidation.knowledge_mutation_enabled`
- `consolidation.weekly_distillation_enabled`
- `consolidation.synaptic_downscaling_enabled`
- `consolidation.skill_autolearn_enabled`
- `consolidation.curator_auto_apply_enabled`
- `consolidation.weekly_enabled` / `monthly_enabled`
- `consolidation.knowledge_self_correction_enabled`
- `rag.facts_extraction_enabled`

daily episode生成は差分処理にし、索引・容量管理・原記録・確定指示は維持します。
必要な機能は個別に再有効化して比較できます。facts抽出停止期間は記録し、
再有効化のみで過去の抽出が埋め戻されるとは考えないでください。
Neo4jは任意backendのままです。既に遅延ロードされる経路を削除したとは主張しません。

`heartbeat_enabled` の既定trueは変えていません。停止する前に、受信・cron・期限業務・
未完通知・明示resume・cancelが対象担当で動くことを確認してください。
正常cronは正常と確定できる狭い `skip_pattern` で抑止します。stderrや非zero終了は
確認対象です。`trigger_heartbeat:false` はコマンド側が異常通知まで担当する場合に限ります。

## 切戻し

完了済み仕事を再実行しないため、古いDB backupだけを戻して起動しないでください。
再びclaim停止・全試行終了・サーバー停止を確認し、現在状態を新規ディレクトリへexportします。

```sh
ANIMAWORKS_DATA_DIR=/path/to/runtime uv run animaworks task-store export \
  --anima alice --destination /new/path/current-task-snapshot
```

`manifest.json` の `complete:true` を確認します。タスク以外の設定・原記録・成果物は
別に保持し、既存pendingへ上書き混合しないでください。
exportは現在ready/pendingの入力だけを旧形式の実行対象にし、非実行入力も別に保持します。

## 検証と限界

再現用スクリプトと匿名fixtureは [tests/fixtures/slim_runtime/README.md](../tests/fixtures/slim_runtime/README.md) を参照。
Dockerの実server/worker試験と、実モデルの回答比較は別です。
合成fixtureの成功は実務受入率、人間の修正時間、全組織の費用削減を保証しません。
