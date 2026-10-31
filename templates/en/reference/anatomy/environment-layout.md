## Runtime Data Directory

All runtime data is stored in `{data_dir}/`.

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

## Activity Scope Rules

1. **Your own directory** (`{data_dir}/animas/{anima_name}/`): Read and write freely
2. **Shared area** (`{data_dir}/shared/`): Read and write. Used for sending and receiving messages and sharing user memory
3. **Common skills** (`{data_dir}/common_skills/`): Only top-level members (no supervisor set) can write. Other members have read-only access. Skills available to everyone
4. **Company information** (`{data_dir}/company/`): Only top-level members can write
5. **Prompts** (`{data_dir}/prompts/`): Read-only. Templates such as character design guides
6. **Other employees' directories**: Only accessible within the scope specified in permissions.json
7. **Subordinate directories** (supervisor only. Same permission for children, grandchildren, great-grandchildren, and all descendants):
   - **Schedules Anima can edit**: The supervisor can read and write `cron.md` / `HEARTBEAT.md`
   - **Root-owned configuration**: `status.json`, `IDENTITY.md`, `SOUL.md`, `AGENTS.md`, `permissions.json`, and root's `config.json` cannot be written directly from the Anima process. Use the corresponding supervisor tools or root management API/CLI
   - **Status reference**: `activity_log/` and `state/current_state.md` are **read-only**. Check subordinate tasks with the authorized task tools. The authoritative copy is stored in host management; direct editing is prohibited.
   - **Other subordinate memory** (including `IDENTITY.md` / `SOUL.md`): Read-only unless a dedicated root management operation exists
8. **Colleagues' activity_log**: `activity_log/` of colleagues with the same supervisor is readable (for validation). Writing is not allowed
