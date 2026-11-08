> 確認したコミット: 9984ccc15

# ドキュメント索引

`docs/ja/` は公開ドキュメントの日本語正本である。英語版と韓国語版は R22 の翻訳パイプラインで生成し、`docs/en/`・`docs/ko/` は直接編集しない。`docs/ja/reference/` も R21 がコードから生成するため、直接編集しない。

## 文書構成

| 文書 | 内容 |
|---|---|
| [設計理念](vision.md) | AnimaWorks が目指す設計思想と価値観。 |
| [機能概観](overview.md) | 機能の全体像と詳細章への入口。 |
| [脳科学との対応](brain-mapping.md) | 記憶・注意・自律機構と脳科学の対応および設計根拠。 |
| [アーキテクチャ全体](architecture/index.md) | システム構成と各アーキテクチャ章の案内。 |
| [Anima のファイル](architecture/anima-files.md) | Anima ごとの定義ファイル、設定、状態ファイル。 |
| [プロセス構成](architecture/process.md) | server、root、task runner、IPC、再起動。 |
| [実行モード](architecture/execution.md) | 実行エンジン、モデル解決、fallback、コンテキスト管理。 |
| [ライフサイクル](architecture/lifecycle.md) | chat、inbox、heartbeat、cron、task の起動経路とロック。 |
| [プロンプト構築](architecture/prompt.md) | システムプロンプトと実行時コンテキストの構成。 |
| [タスク管理](architecture/tasks.md) | Task Board、委任、background task。 |
| [メッセージング](architecture/messaging.md) | DM、Board、人間への通知、Inbox wake とメッセージ処理、外部連携。 |
| [記憶システム](memory/index.md) | 記憶の設計、ディレクトリ、frontmatter。 |
| [自動想起](memory/priming.md) | 実行時に記憶をコンテキストへ取り込む方式。 |
| [意図的想起と検索](memory/retrieval.md) | `search_memory`、検索処理、RAG、修復。 |
| [統合と忘却](memory/consolidation.md) | 日次・週次処理、記憶の見直し、手続き記憶。 |
| [アクティビティログ](memory/activity-log.md) | JSONL の活動記録とストリーミング時の復旧記録。 |
| [セキュリティ](security.md) | 権限境界、保護、セキュリティ運用。 |
| [Enclave](enclave.md) | 隔離ランタイムの構築、データ参照、点検と監査。 |
| [設定](operations/configuration.md) | 全体設定と Anima ごとの設定方法。 |
| [会社管理](operations/company.md) | 組織・会社情報の管理。 |
| [GPU 運用](operations/gpu.md) | GPU を使うコンポーネントの運用。 |
| [開発チーム](operations/dev-team.md) | 開発環境とチーム運用。 |
| [Slim Runtime 移行](operations/slim-runtime-migration.md) | Slim Runtime の移行手順。 |
| [Slack 連携](integrations/slack.md) | Slack 接続の設定と運用。 |
| [Zoom 連携](integrations/zoom.md) | Zoom RTMS 接続の設定と運用。 |
| [AnimaWorks Hub 連携](integrations/hub.md) | 社外 AI を Hub Board に参加させる設定、名簿、アバター。 |
| [CLI リファレンス](reference/cli.md) | `animaworks` コマンドの自動生成リファレンス。 |
| [ツール CLI リファレンス](reference/tool-cli.md) | `animaworks-tool` の自動生成リファレンス。 |
| [API リファレンス](reference/api.md) | HTTP API の自動生成リファレンス。 |
| [設定リファレンス](reference/config.md) | 設定項目と既定値の自動生成リファレンス。 |
| [モジュールリファレンス](reference/modules.md) | `core/` の構成から生成したモジュール案内。 |

## 読む順番

- **はじめて触る人**: [機能概観](overview.md) → [アーキテクチャ全体](architecture/index.md) → [記憶システム](memory/index.md)。動作の流れは[ライフサイクル](architecture/lifecycle.md)を読む。
- **設計に関心がある人**: [設計理念](vision.md) → [脳科学との対応](brain-mapping.md) → [アーキテクチャ全体](architecture/index.md) → 記憶の各章。
- **運用者**: [設定](operations/configuration.md) → [セキュリティ](security.md) → [CLI リファレンス](reference/cli.md)・[設定リファレンス](reference/config.md)。個別の連携は[Slack](integrations/slack.md)・[Zoom](integrations/zoom.md)・[AnimaWorks Hub](integrations/hub.md)を参照する。

## 執筆者向け

- 文体は「である」調とする。各ファイルの先頭に `> 確認したコミット: <短縮 sha>` を置き、内容を照合したコードの時点を示す。
- コードへの参照は `core/memory/priming/engine.py` の `_prime_compact` のように、パスとシンボル名で記す。行番号は書かない。
- 識別子・設定キー・パス・コマンドはバッククォートで囲み、翻訳時に保護できるようにする。
- 設定キーの一覧表は本文に作らず、[設定リファレンス](reference/config.md)へリンクする。本文では挙動の説明に必要なキーだけに触れる。
- 撤廃済みの機能や過去の実装経緯は記載しない。変更履歴は CHANGELOG の役割である。
- 見出しは翻訳の差分単位となる。各節はおおむね 2,000 字以内に保つ。
- 1つの事実は原則として1か所に記し、概観や対応表では詳細章へリンクする。
