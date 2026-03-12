# Common Knowledge — 目次・クイックガイド

AnimaWorks の全 Anima が共有するリファレンスドキュメントの目次。
困ったとき・手順が不明なときは、このファイルで該当ドキュメントを特定し、
`read_memory_file(path="common_knowledge/...")` で詳細を参照すること。

> 💡 詳細な技術リファレンス（構成ファイル仕様・モデル設定・認証設定等）は `reference/` に移動しました。
> 目次: `reference/00_index.md`

---

## 困ったときのクイックガイド

### コミュニケーション

| 困りごと | 参照先 |
|---------|--------|
| メッセージの送り方がわからない | `communication/messaging-guide.md` |
| Board（共有チャネル）の使い方がわからない | `communication/board-guide.md` |
| 指示の出し方・報告の仕方がわからない | `communication/instruction-patterns.md` / `communication/reporting-guide.md` |
| メッセージ送信が制限された | `communication/sending-limits.md` |
| 人間への通知方法がわからない | `communication/call-human-guide.md` |
| Slack ボットトークンの設定がわからない | `reference/communication/slack-bot-token-guide.md` ※技術リファレンス |

### 組織・階層

| 困りごと | 参照先 |
|---------|--------|
| 組織構造・誰に連絡すべきかわからない | `reference/organization/structure.md` ※技術リファレンス |
| 役割と責任範囲を確認したい | `organization/roles.md` |
| 階層間の通信ルールがわからない | `organization/hierarchy-rules.md` |

### タスク・運用

| 困りごと | 参照先 |
|---------|--------|
| タスク管理の方法がわからない | `operations/task-management.md` |
| タスクボード（人間向けダッシュボード）を使いたい | `operations/task-board-guide.md` |
| ハートビートやcronの設定がわからない | `operations/heartbeat-cron-guide.md` |
| 長時間ツールの実行方法がわからない | `operations/background-tasks.md` |
| プロジェクト設定を変更したい | `reference/operations/project-setup.md` ※技術リファレンス |

### ツール・モデル・技術

| 困りごと | 参照先 |
|---------|--------|
| ツールの使い方・呼び出し方がわからない | `operations/tool-usage-overview.md` |
| モデルの選び方・変更方法がわからない | `reference/operations/model-guide.md` ※技術リファレンス |
| Mode S の認証方式を変えたい | `reference/operations/mode-s-auth-guide.md` ※技術リファレンス |
| 音声チャットの設定・使い方がわからない | `reference/operations/voice-chat-guide.md` ※技術リファレンス |

### 自分自身の理解

| 困りごと | 参照先 |
|---------|--------|
| Animaとは何か知りたい | `anatomy/what-is-anima.md` |
| 自分の構成ファイルの役割を知りたい | `reference/anatomy/anima-anatomy.md` ※技術リファレンス |
| 記憶の仕組み・種類を知りたい | `anatomy/memory-system.md` |

### トラブルシューティング

| 困りごと | 参照先 |
|---------|--------|
| ツールやコマンドが使えない / エラーが出る | `troubleshooting/common-issues.md` |
| タスクがブロックされた / 判断に迷う | `troubleshooting/escalation-flowchart.md` |
| Gmail ツールの認証設定がうまくいかない | `reference/troubleshooting/gmail-credential-setup.md` ※技術リファレンス |

### セキュリティ

| 困りごと | 参照先 |
|---------|--------|
| 外部データの信頼性が気になる | `security/prompt-injection-awareness.md` |

### 活用例

| 困りごと | 参照先 |
|---------|--------|
| AnimaWorksで何ができるか知りたい | `usecases/usecase-overview.md` |

**上記に該当しない場合** → `search_memory(query="キーワード", scope="common_knowledge")` で検索する

---

## ドキュメント一覧

### anatomy/ — Animaの構成要素

| ファイル | 概要 |
|---------|------|
| `what-is-anima.md` | Animaとは何か（概念・設計思想・ライフサイクル・実行パス） |
| `anima-anatomy.md` | → `reference/anatomy/anima-anatomy.md` に移動。構成ファイル完全ガイド |
| `memory-system.md` | 記憶システムガイド（記憶の種類・Priming・Consolidation・Forgetting・ツール使い分け） |

### organization/ — 組織・構造

| ファイル | 概要 |
|---------|------|
| `structure.md` | → `reference/organization/structure.md` に移動。組織構造の仕組み |
| `roles.md` | 役割と責任範囲（トップレベル / 中間管理 / 実行 Anima の責務） |
| `hierarchy-rules.md` | 階層間のルール（通信経路、スーパーバイザーツール、緊急時の例外） |

### communication/ — コミュニケーション

| ファイル | 概要 |
|---------|------|
| `messaging-guide.md` | メッセージ送受信の完全ガイド（send_message のパラメータ、スレッド管理、1ラウンドルール） |
| `board-guide.md` | Board（共有チャネル）ガイド（post_channel / read_channel の使い分け、投稿ルール） |
| `instruction-patterns.md` | 指示の出し方パターン集（明確な指示の書き方、委任パターン、進捗確認） |
| `reporting-guide.md` | 報告・エスカレーションの方法（報告タイミング、フォーマット、緊急 vs 定期） |
| `sending-limits.md` | 送信制限の詳細（3層レート制限、30/h・100/day 上限、カスケード検出、対処法） |
| `call-human-guide.md` | 人間への通知ガイド（call_human の使い方、返信の受け取り、通知チャネル設定） |
| `slack-bot-token-guide.md` | → `reference/communication/slack-bot-token-guide.md` に移動。Slack ボットトークン設定ガイド |

### operations/ — 運用・タスク管理

| ファイル | 概要 |
|---------|------|
| `project-setup.md` | → `reference/operations/project-setup.md` に移動。プロジェクト設定方法 |
| `task-management.md` | タスク管理（current_task.md / pending.md の使い方、状態遷移、優先順位） |
| `task-board-guide.md` | タスクボード（人間向けダッシュボード）の仕組みと運用方法 |
| `heartbeat-cron-guide.md` | 定期実行の設定と運用（ハートビートの仕組み、cron タスク定義、自己更新） |
| `tool-usage-overview.md` | ツール使用の概要（S/A/B モード別のツール体系、内部/外部ツール、呼び出し方法） |
| `background-tasks.md` | バックグラウンドタスク実行ガイド（submit の使い方、判断基準、結果の受け取り方） |
| `model-guide.md` | → `reference/operations/model-guide.md` に移動。モデル選択・設定ガイド |
| `mode-s-auth-guide.md` | → `reference/operations/mode-s-auth-guide.md` に移動。Mode S 認証モード設定ガイド |
| `voice-chat-guide.md` | → `reference/operations/voice-chat-guide.md` に移動。音声チャットガイド |

### security/ — セキュリティ

| ファイル | 概要 |
|---------|------|
| `prompt-injection-awareness.md` | プロンプトインジェクション防御ガイド（信頼レベル、境界タグ、untrusted データの処理ルール） |

### troubleshooting/ — トラブルシューティング

| ファイル | 概要 |
|---------|------|
| `common-issues.md` | よくある問題と対処法（メッセージ不達、送信制限、権限、ツール、コンテキスト） |
| `escalation-flowchart.md` | 困ったときの判断フローチャート（問題分類、緊急度判定、エスカレーション先） |
| `gmail-credential-setup.md` | → `reference/troubleshooting/gmail-credential-setup.md` に移動。Gmail Tool認証設定ガイド |

### usecases/ — ユースケースガイド

| ファイル | 概要 |
|---------|------|
| `usecase-overview.md` | ユースケースガイド概要（AnimaWorksでできること・始め方・全テーマ一覧） |
| `usecase-communication.md` | コミュニケーション自動化（チャット・メール監視、エスカレーション、定期連絡） |
| `usecase-development.md` | ソフトウェア開発支援（コードレビュー、CI/CD監視、Issue実装、バグ調査） |
| `usecase-monitoring.md` | インフラ・サービス監視（死活監視、リソース監視、SSL証明書、ログ分析） |
| `usecase-secretary.md` | 秘書・事務サポート（スケジュール管理、連絡調整、日報作成、リマインダー） |
| `usecase-research.md` | 調査・リサーチ・分析（Web検索、競合分析、市場調査、レポート作成） |
| `usecase-knowledge.md` | ナレッジ管理・ドキュメント整備（手順書作成、FAQ構築、教訓の蓄積） |
| `usecase-customer-support.md` | カスタマーサポート（一次対応、FAQ自動回答、エスカレーション管理） |

---

## キーワード索引

| キーワード | 参照先 |
|-----------|--------|
| メッセージ, send_message, 送信, 返信, スレッド, inbox | `communication/messaging-guide.md` |
| Board, チャネル, post_channel, read_channel | `communication/board-guide.md` |
| DM履歴, read_dm_history, 過去の会話 | `communication/board-guide.md` |
| 指示, 委任, タスク依頼, デリゲーション | `communication/instruction-patterns.md` |
| 報告, 日報, サマリー, 完了報告, エスカレーション | `communication/reporting-guide.md` |
| レート制限, 送信制限, 30通, 100通, 1ラウンドルール | `communication/sending-limits.md` |
| call_human, 人間通知, 人間に連絡, 通知チャネル | `communication/call-human-guide.md` |
| Slack, ボットトークン, SLACK_BOT_TOKEN, not_in_channel | `reference/communication/slack-bot-token-guide.md` |
| 組織, supervisor, 上司, 部下, 同僚 | `reference/organization/structure.md` |
| 役割, 責任, speciality, 専門 | `organization/roles.md` |
| 階層, 通信経路, org_dashboard, ping_subordinate | `organization/hierarchy-rules.md` |
| delegate_task, タスク委譲, task_tracker | `organization/hierarchy-rules.md`, `operations/task-management.md` |
| タスク, current_task, pending, 進捗, 優先順位 | `operations/task-management.md` |
| backlog_task, タスクキュー, submit_tasks, TaskExec | `operations/task-management.md` |
| タスクボード, ダッシュボード, 人間向け | `operations/task-board-guide.md` |
| 設定, config, status.json, SSoT, reload | `reference/operations/project-setup.md` |
| ハートビート, heartbeat, 定期チェック | `operations/heartbeat-cron-guide.md` |
| cron, スケジュール, 定時タスク | `operations/heartbeat-cron-guide.md` |
| ツール, animaworks-tool, MCP, mcp__aw__, skill | `operations/tool-usage-overview.md` |
| 実行モード, S-mode, A-mode, B-mode, C-mode | `operations/tool-usage-overview.md` |
| バックグラウンド, submit, 長時間ツール | `operations/background-tasks.md` |
| モデル, models.json, credential, set-model, コンテキストウィンドウ | `reference/operations/model-guide.md` |
| background_model, バックグラウンドモデル, コスト最適化 | `reference/operations/model-guide.md` |
| Mode S, 認証, API直接, Bedrock, Vertex AI, Max plan | `reference/operations/mode-s-auth-guide.md` |
| 音声, voice, STT, TTS, VOICEVOX, ElevenLabs | `reference/operations/voice-chat-guide.md` |
| WebSocket, /ws/voice, barge-in, VAD, PTT | `reference/operations/voice-chat-guide.md` |
| Anima, 自分, 構成, 設計, ライフサイクル | `anatomy/what-is-anima.md` |
| identity, injection, 人格, 行動指針, 不変, 可変 | `reference/anatomy/anima-anatomy.md` |
| permissions.md, bootstrap, heartbeat.md, cron.md | `reference/anatomy/anima-anatomy.md` |
| 記憶, memory, episodes, knowledge, procedures | `anatomy/memory-system.md` |
| Priming, RAG, Consolidation, Forgetting, 忘却 | `anatomy/memory-system.md` |
| search_memory, write_memory_file, 記憶検索 | `anatomy/memory-system.md` |
| プロンプトインジェクション, trust, untrusted, 境界タグ | `security/prompt-injection-awareness.md` |
| エラー, 問題, 動かない, 権限, ブロックコマンド | `troubleshooting/common-issues.md` |
| フローチャート, 判断, 迷い, 緊急, セキュリティ | `troubleshooting/escalation-flowchart.md` |
| Gmail, token.json, OAuth, pickle | `reference/troubleshooting/gmail-credential-setup.md` |
| ティア, tiered, T1, T2, T3, T4 | `troubleshooting/common-issues.md` |
| ユースケース, 活用例, 何ができる | `usecases/usecase-overview.md` |

---

## 使い方

```
# キーワードで検索
search_memory(query="メッセージ 送信", scope="common_knowledge")

# パスを直接指定
read_memory_file(path="common_knowledge/communication/messaging-guide.md")

# 技術リファレンスを参照
read_memory_file(path="reference/anatomy/anima-anatomy.md")

# このファイル自体を参照
read_memory_file(path="common_knowledge/00_index.md")
```
