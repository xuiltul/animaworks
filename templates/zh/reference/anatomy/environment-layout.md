## 运行时数据目录

所有运行时数据都存储在 `{data_dir}/` 中。

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

## 活动范围规则

1. **自己的目录**（`{data_dir}/animas/{anima_name}/`）：可自由读写
2. **共享区域**（`{data_dir}/shared/`）：可读写。用于消息收发和用户记忆共享
3. **通用技能**（`{data_dir}/common_skills/`）：仅顶层成员（未设置 supervisor）可写入。其他成员只读。全员可用的技能
4. **公司信息**（`{data_dir}/company/`）：仅顶层成员可写入
5. **提示词**（`{data_dir}/prompts/`）：只读。角色设计指南等模板
6. **其他员工的目录**：仅可访问 permissions.json 中明确指定的范围
7. **下属目录**（仅 supervisor。子、孙、曾孙……所有下属具有相同权限）：
   - **Anima 可编辑的日程**：supervisor 可读写 `cron.md` / `HEARTBEAT.md`
   - **root 所有的配置**：`status.json`、`IDENTITY.md`、`SOUL.md`、`AGENTS.md`、`permissions.json` 以及 root 的 `config.json` 无法由 Anima 进程直接写入。请使用对应的上司用工具或 root 管理的 API/CLI
   - **状态参照**：`activity_log/` 和 `state/current_state.md` **仅可读取**。部下的任务请用有权限的任务工具确认。正本的保存位置由主机管理，禁止直接编辑。
   - **其他下属记忆**（包括 `IDENTITY.md` / `SOUL.md`）：除非有专用的 root 管理操作，否则仅可读取
8. **同事的 activity_log**：与同一 supervisor 的同事的 `activity_log/` 可读取（用于验证）。不可写入
