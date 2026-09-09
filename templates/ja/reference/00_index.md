# Reference — 技術リファレンス目次

AnimaWorks の詳細な技術仕様・管理者向け設定ガイドの一覧。
RAG 検索対象外。必要なときに `read_memory_file(path="reference/...")` で直接参照すること。

## Reference files

### anatomy/ — 構成・アーキテクチャ

| File | Title |
|------|-------|
| `anatomy/anima-anatomy.md` | Anima 構成ファイル完全ガイド |
| `anatomy/environment-layout.md` | ランタイムディレクトリ構成と権限 |
| `anatomy/memory-system.md` | 記憶システムガイド |
| `anatomy/priming-channels.md` | Priming チャネル技術リファレンス |
| `anatomy/working-memory.md` | ワーキングメモリ（state/）技術リファレンス |

### communication/ — メッセージング・連携

| File | Title |
|------|-------|
| `communication/instruction-patterns.md` | 指示の出し方パターン集 |
| `communication/messaging-guide.md` | メッセージ送信の完全ガイド |
| `communication/reporting-guide.md` | 報告・エスカレーションの方法 |
| `communication/slack-bot-token-guide.md` | Slack ボットトークン設定ガイド |

### internals/ — フレームワーク内部仕様

| File | Title |
|------|-------|
| `internals/common-knowledge-access-paths.md` | common_knowledge の参照経路 |

### operations/ — 管理・運用設定

| File | Title |
|------|-------|
| `operations/browser-automation-guide.md` | ブラウザ操作ガイド |
| `operations/heartbeat-cron-guide.md` | 定期実行の設定と運用 |
| `operations/memory-writing-guide.md` | 記憶の記録先と定期実行の選択 |
| `operations/mode-s-auth-guide.md` | Mode S（Agent SDK）認証モード設定ガイド |
| `operations/model-guide.md` | モデル選択・設定ガイド |
| `operations/project-setup.md` | プロジェクト設定方法 |
| `operations/task-management.md` | タスク管理の方法 |
| `operations/tool-usage-overview.md` | ツール使用ガイド |
| `operations/voice-chat-guide.md` | 音声チャット（Voice Chat）ガイド |

### organization/ — 組織構造

| File | Title |
|------|-------|
| `organization/roles.md` | 役割と責任範囲 |
| `organization/structure.md` | 組織構造の仕組み |

### troubleshooting/ — トラブルシューティング

| File | Title |
|------|-------|
| `troubleshooting/common-issues.md` | よくある問題と対処法 |
| `troubleshooting/escalation-flowchart.md` | 困ったときのフローチャート |
| `troubleshooting/gmail-credential-setup.md` | Gmail Tool 認証設定ガイド |

### usecases/ — ユースケースガイド

| File | Title |
|------|-------|
| `usecases/usecase-communication.md` | ユースケース: コミュニケーション自動化 |
| `usecases/usecase-customer-support.md` | ユースケース: カスタマーサポート |
| `usecases/usecase-development.md` | ユースケース: ソフトウェア開発支援 |
| `usecases/usecase-knowledge.md` | ユースケース: ナレッジ管理・ドキュメント整備 |
| `usecases/usecase-monitoring.md` | ユースケース: インフラ・サービス監視 |
| `usecases/usecase-overview.md` | AnimaWorks ユースケースガイド |
| `usecases/usecase-research.md` | ユースケース: 調査・リサーチ・分析 |
| `usecases/usecase-secretary.md` | ユースケース: 秘書・事務サポート |

## Related

- 日常の実用ガイド → `common_knowledge/00_index.md`
- 共通スキル → `common_skills/`
