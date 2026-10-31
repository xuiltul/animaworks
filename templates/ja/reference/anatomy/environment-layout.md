## ランタイムデータディレクトリ

すべての実行時データは `{data_dir}/` に格納されています。

```
{data_dir}/
├── company/          # 会社のビジョン・方針（読み取り専用）
├── animas/          # 全社員のデータ
│   ├── {anima_name}/    # ← あなた自身
│   └── ...               # 他の社員
├── prompts/          # プロンプトテンプレート（キャラクター設計ガイド等）
├── vault.json        # 共有クレデンシャル保管庫
├── shared/           # 社員間の共有領域
│   ├── channels/     # Board共有チャネル（general.jsonl, ops.jsonl 等）
│   ├── credentials.json  # レガシー互換用フォールバック
│   ├── inbox/        # メッセージ受信箱
│   └── users/        # 共有ユーザー記憶（ユーザーごとのサブディレクトリ）
├── common_skills/    # 全社員共通スキル（読み取り専用）
└── tmp/              # 作業用ディレクトリ
    └── attachments/  # メッセージ添付ファイル
```

## 活動範囲のルール

1. **自分のディレクトリ** (`{data_dir}/animas/{anima_name}/`): 自由に読み書き可能
2. **共有領域** (`{data_dir}/shared/`): 読み書き可能。メッセージ送受信およびユーザー記憶の共有に使用
3. **共通スキル** (`{data_dir}/common_skills/`): トップレベルメンバー（supervisor未設定）のみ書き込み可能。その他のメンバーは読み取り専用。全員が使えるスキル
4. **会社情報** (`{data_dir}/company/`): トップレベルメンバーのみ書き込み可能
5. **プロンプト** (`{data_dir}/prompts/`): 読み取り専用。キャラクター設計ガイド等のテンプレート
6. **他の社員のディレクトリ**: permissions.json に明示された範囲のみアクセス可能
7. **配下のディレクトリ**（supervisorのみ。子・孫・曾孫…全配下に同じ権限）:
   - **Anima が編集できるスケジュール**: supervisor は `cron.md` / `HEARTBEAT.md` を読み書き可能
   - **root 所有設定**: `status.json`, `IDENTITY.md`, `SOUL.md`, `AGENTS.md`, `permissions.json` および root の `config.json` は Anima プロセスから直接書き込めない。対応する上司向けツールまたは root 管理 API/CLI を使う
   - **状態参照**: `activity_log/` と `state/current_state.md` は**読み取りのみ**。部下のタスクは権限のあるタスクツールで確認する。正本の保存先はホスト管理で直接編集は禁止。
   - **その他の配下メモリ**（`IDENTITY.md` / `SOUL.md` を含む）: 専用の root 管理操作がない限り読み取りのみ
8. **同僚のactivity_log**: 同じsupervisorを持つ同僚の `activity_log/` は読み取り可能（検証用）。書き込みは不可
