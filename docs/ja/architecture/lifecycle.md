> 確認したコミット: b304b7dc

# 起動トリガーとライフサイクル

Anima は server からの要求だけでなく、メッセージやスケジュールを契機として実行される。root process が常駐して scheduler と受信処理を管理し、実際の会話・自動実行は task runner の専用 lane に切り出す。

## トリガーと lane

| トリガー | 実行される処理 |
|---|---|
| chat | ユーザーとの会話を thread ごとに処理する。 |
| inbox | 他 Anima や連携先から届いたメッセージを専用 inbox lane で処理する。 |
| heartbeat | `HEARTBEAT.md` の指示に基づく定期的な確認を heartbeat task runner で行う。 |
| cron | `cron.md` に定義した定期処理を cron task runner で実行する。 |
| task | TaskStore の実行可能タスクを background worker が引き受ける。 |
| greet | 起動後や会話開始時の挨拶を専用 chat task runner で行う。 |

チャットは thread ごとの conversation lock で同じ会話の競合を防ぐ。inbox と scheduled background work は別の制御 lock を持つ。TaskExec は専用 worker slot と AgentCore を使い、heartbeat/Cron の実行 lock を共有しない。ファイル `state/current_state.md` の更新には process-safe な state file lock が使われる。

## Heartbeat と cron

heartbeat の基本間隔は全体設定で 30 分が既定であり、`status.json` の `heartbeat_interval_minutes` で Anima ごとに上書きできる。有効な範囲は 1〜1440 分である。全体の `activity_level` は既定 100%、許容範囲 10〜400% で、値が高いほど実効間隔が短くなる。`activity_schedule` を使うと時刻帯ごとに活動度を切り替えられる。`HEARTBEAT.md` からは活動時間帯を設定する。

`cron.md` は見出し単位の定義で、`schedule` に標準の5フィールド cron 式を記す。`type`、説明、必要に応じて command/tool と引数を設定し、`core/runtime/schedule_parser.py` が解析する。セットアップや hot-reload 時には、解析・登録できなかったスケジュールだけを `cron_health_*.md` として通知する。登録済み cron の実行頻度や失敗を定期監視する機能はない。

## 初回起動と1サイクル

初回起動では `core/anima/bootstrap_state.py` が `IDENTITY.md` / `SOUL.md` や `AGENTS.md` などの定義状態を確認し、bootstrap の開始・失敗・完了を記録する。supervisor はプロセスを起動し、IPC endpoint と依存サービスの準備を確認してから通常の処理を開始する。

1. trigger を session type と lane に対応づけ、必要な lock または worker slot を取得する。
2. `status.json` と共通設定を解決し、記憶、権限、現在状態、メッセージなどから prompt を準備する。
3. 実行モードに合う engine adapter がモデル呼び出しと tool 実行を仲介し、イベントを root に返す。
4. 応答、会話記録、タスク状態、活動記録を確定し、後続処理に必要な通知を保存する。
5. lock / worker slot を解放し、root の scheduler と inbox dispatcher が次の依頼を受けられる状態に戻る。

## 設計判断

- **cron は各 Anima の内部時計である。** 組織全体のスケジューラーではなく、人が自分の日課を持つのと同じく、各 Anima が自分の習慣として持つ。
