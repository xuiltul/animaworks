from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""i18n strings for the ``animaworks migrate`` command."""

STRINGS: dict[str, dict[str, str]] = {
    "task_store.teardown_retired": {
        "ja": "この旧スクリプトの更新機能は廃止されました。--dry-run の診断のみ利用できます。サーバーとワーカーを停止し、対象データディレクトリで animaworks task-store migrate --anima <名前> --backup <新規DBパス> を実行してください。何も変更していません。",
        "en": "This legacy script's write mode is retired; only --dry-run diagnostics remain. Stop the server and workers, then use animaworks task-store migrate --anima <name> --backup <new-db-path> for the intended data directory. Nothing was changed.",
        "ko": "이 기존 스크립트의 쓰기 기능은 폐기되었습니다. --dry-run 진단만 사용할 수 있습니다. 서버와 워커를 중지한 후 대상 데이터 디렉터리에서 animaworks task-store migrate --anima <이름> --backup <새-DB-경로>를 실행하세요. 아무것도 변경하지 않았습니다.",
    },
    "task_store.help": {"ja": "担当者単位のタスク正本の保守・移行", "en": "Maintain and migrate a scoped task store"},
    "task_store.status_help": {"ja": "停止ゲートと実行数を表示", "en": "Show claim gate and active attempt counts"},
    "task_store.quiesce_help": {"ja": "新しい実行の取得を永続停止", "en": "Persistently pause new task claims"},
    "task_store.resume_help": {"ja": "新しい実行の取得を再開", "en": "Resume new task claims"},
    "task_store.migrate_help": {
        "ja": "停止中の旧台帳を取り込む（バックアップ必須）",
        "en": "Import offline legacy state with a required backup",
    },
    "task_store.backup_help": {
        "ja": "WAL込みのDBバックアップを新規作成",
        "en": "Create a new SQLite backup including committed WAL data",
    },
    "task_store.export_help": {
        "ja": "停止中の現在状態を新規ディレクトリへ書き出す",
        "en": "Export current offline state into a fresh directory",
    },
    "task_store.anima_help": {"ja": "対象の担当者名", "en": "Anima whose task state is in scope"},
    "task_store.destination_help": {"ja": "未使用の出力先パス", "en": "New destination path (must not exist)"},
    "task_store.offline_required": {
        "ja": "対象のサーバーとワーカーを停止してから実行してください。自動停止はしません。",
        "en": "Stop the runtime server and worker first; this command does not stop them.",
    },
    "task_store.migration_required": {
        "ja": "{name} の旧タスク状態は未移行です。サーバーとワーカーを停止し、animaworks task-store migrate --anima {name} --backup <新規DBパス> を実行してください。任意の読み取りで自動移行はしません。",
        "en": "Legacy task state for {name} requires offline migration. Stop the server and workers, then run animaworks task-store migrate --anima {name} --backup <new-db-path>. Runtime reads do not auto-migrate populated legacy state.",
        "ko": "{name}의 기존 작업 상태는 오프라인 마이그레이션이 필요합니다. 서버와 워커를 중지한 후 animaworks task-store migrate --anima {name} --backup <새-DB-경로>를 실행하세요. 읽기 작업으로 자동 마이그레이션하지 않습니다.",
    },
    "task_store.invalid_pid": {
        "ja": "server.pidが不正です。停止状態を確認してください。",
        "en": "Invalid server.pid; verify that the runtime is offline.",
    },
    "task_store.invalid_anima": {
        "ja": "対象担当者が存在しないか、パスが不正です: {name}",
        "en": "Missing anima or invalid path: {name}",
    },
    "task_store.active_attempts": {
        "ja": "未終了の試行が{count}件あります。状態を確定してから再実行してください。新規取得は停止したままです。",
        "en": "{count} attempts remain active; resolve them before retrying. New claims remain paused.",
    },
    "task_store.invalid_rows": {
        "ja": "旧データに不正な行が{count}件あるため、取り込みを取り消しました。原本を確認してください。",
        "en": "Import rolled back because {count} legacy rows are invalid; inspect the originals.",
    },
    "task_store.error": {
        "ja": "タスク正本の保守に失敗しました: {error}",
        "en": "Task store maintenance failed: {error}",
    },
    "migrate.help": {
        "ja": "ランタイムデータのマイグレーションを実行",
        "en": "Run runtime data migrations",
    },
    "migrate.no_runtime": {
        "ja": "ランタイムディレクトリが初期化されていません: {data_dir}\n'animaworks init' を実行してください。",
        "en": "Runtime directory not initialized: {data_dir}\nRun 'animaworks init' first.",
    },
    "migrate.dry_run_header": {
        "ja": "=== ドライラン — 変更は行いません ===",
        "en": "=== Dry run — no changes will be made ===",
    },
    "migrate.step_result": {
        "ja": "[{name}] changed: {changed}, skipped: {skipped}",
        "en": "[{name}] changed: {changed}, skipped: {skipped}",
    },
    "migrate.complete": {
        "ja": "マイグレーション完了: {changed}件変更, {skipped}件スキップ",
        "en": "Migration complete: {changed} changed, {skipped} skipped",
    },
    "migrate.error_summary": {
        "ja": "エラー: {count}件のステップで失敗",
        "en": "Errors: {count} step(s) failed",
    },
    "migrate.server_warning": {
        "ja": "⚠ サーバーが実行中です。SQLite WALモードで安全ですが、プロンプト変更は次回ロード時に反映されます。",
        "en": "⚠ Server is running. SQLite WAL mode is safe, but prompt changes take effect on next load.",
    },
    "migrate.list_header": {
        "ja": "マイグレーションステップ一覧:",
        "en": "Migration steps:",
    },
}
