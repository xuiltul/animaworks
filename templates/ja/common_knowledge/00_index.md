# Common Knowledge — 目次・クイックガイド

AnimaWorks の全 Anima が共有するリファレンスドキュメントの目次。
困ったとき・手順が不明なときは、このファイルで該当ドキュメントを特定し、
`read_memory_file(path="common_knowledge/...")` で詳細を参照すること。

> 💡 詳細な技術リファレンス（構成ファイル仕様・モデル設定・認証設定等）は `reference/` に移動しました。
> 目次: `reference/00_index.md`

---

## ⭐ まずここを読む

AnimaWorks を初めて使う場合、または全体像を整理したい場合は、以下の1ファイルを最初に読むこと。
Heartbeat / Cron / machine / チーム設計 / 記憶 / コスト最適化の要点が1枚にまとまっている。

| ファイル | 内容 |
|---------|------|
| **`anatomy/essentials.md`** | **AnimaWorks エッセンシャルガイド** — 全体像・5つの実行パス・Heartbeat vs Cron・machine の使い方・チーム設計・タスクの流し方・記憶システム・コスト最適化を1枚で俯瞰 |

読了後、各トピックの詳細は下記の目次から辿る。

---

## 困ったときのクイックガイド

### コミュニケーション

| 困りごと | 参照先 |
|---------|--------|
| メッセージの送り方がわからない | `communication/messaging-guide.md` |
| Board（共有チャネル）の使い方がわからない | `communication/board-guide.md` |
| 指示の出し方・報告の仕方がわからない | `communication/instruction-patterns.md` / `communication/reporting-guide.md` |
| 委譲・完了報告・エスカレーションの必須項目を確認したい | `communication/message-quality-protocol.md` |
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
| 完了前検証（completion_gate）の仕組みを知りたい | `operations/completion-gate-guide.md` |
| タスクボード（人間向けダッシュボード）を使いたい | `operations/task-board-guide.md` |
| ハートビートやcronの設定がわからない | `operations/heartbeat-cron-guide.md` |
| 送信・投稿・記憶書き込み前に確認ルールを入れたい | `operations/action-rules-guide.md` |
| 長時間ツールの実行方法がわからない | `operations/background-tasks.md` |
| ワークスペースの登録・使い方がわからない | `operations/workspace-guide.md` |
| 新しいスキルの作り方・メタデータを確認したい | `common_skills/skill-creator/SKILL.md` |
| プロジェクト設定を変更したい | `reference/operations/project-setup.md` ※技術リファレンス |

### ツール・モデル・技術

| 困りごと | 参照先 |
|---------|--------|
| ツールの使い方・呼び出し方がわからない | `operations/tool-usage-overview.md` |
| machine ツールの使い方がわからない | `operations/machine/tool-usage.md` |
| 自分のロールでの machine ワークフローが知りたい | `operations/machine/workflow-{pdm,engineer,reviewer,tester}.md` |
| 目的別にチーム（ロール・ハンドオフ）を設計したい | `team-design/guide.md` |
| 法務チーム（契約レビュー・監査・検証）を運用したい | `team-design/legal/team.md` |
| 財務チーム（分析・検証・データ収集）を運用したい | `team-design/finance/team.md` |
| トレーディングチーム（bot運用・P&L検証・リスク管理）を運用したい | `team-design/trading/team.md` |
| 営業・マーケティングチーム（コンテンツ制作・リード開発・パイプライン管理）を運用したい | `team-design/sales-marketing/team.md` |
| 秘書（情報トリアージ・代行送信・書類作成・スケジュール管理）を運用したい | `team-design/secretary/team.md` |
| COO（事業統括: 委任判断・部門監視・KPI集計・経営報告・部門横断調整）を運用したい | `team-design/coo/team.md` |
| CS（カスタマーサクセス）チーム（オンボーディング・ヘルス分析・リテンション・VoC集約）を運用したい | `team-design/customer-success/team.md` |
| 経営企画チーム（戦略立案・事業分析・独立検証・KPI追跡）を運用したい | `team-design/corporate-planning/team.md` |
| インフラ/SRE チーム（定期監視・異常検知・エスカレーション・集約報告）を運用したい | `team-design/infrastructure/team.md` |
| 推奨組織図・部門配置・導入順序を知りたい | `team-design/org-chart-template.md` |
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
| ⭐ `essentials.md` | **エッセンシャルガイド** — AnimaWorks の全体像を1枚で俯瞰（実行パス・Heartbeat vs Cron・machine・チーム設計・記憶・コスト最適化） |
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
| `message-quality-protocol.md` | メッセージ品質プロトコル（委譲4項目・完了報告3項目・エスカレーション4項目の必須チェック） |
| `sending-limits.md` | 送信制限の詳細（3層レート制限、30/h・100/day 上限、カスケード検出、対処法） |
| `call-human-guide.md` | 人間への通知ガイド（call_human の使い方、返信の受け取り、通知チャネル設定） |
| `slack-bot-token-guide.md` | → `reference/communication/slack-bot-token-guide.md` に移動。Slack ボットトークン設定ガイド |

### operations/ — 運用・タスク管理

| ファイル | 概要 |
|---------|------|
| `project-setup.md` | → `reference/operations/project-setup.md` に移動。プロジェクト設定方法 |
| `task-management.md` | タスク管理（current_state.md の使い方とタスクキュー、状態遷移、優先順位） |
| `task-board-guide.md` | タスクボード（人間向けダッシュボード）の仕組みと運用方法 |
| `heartbeat-cron-guide.md` | 定期実行の設定と運用（ハートビートの仕組み、cron タスク定義、自己更新） |
| `action-rules-guide.md` | アクションルール（`[ACTION-RULE]`、`trigger_tools`、送信前確認、必須 `read_memory_file`） |
| `tool-usage-overview.md` | ツール使用の概要（S/C/D/G/A/B モード別のツール体系、内部/外部ツール、呼び出し方法） |
| `completion-gate-guide.md` | 完了前検証ガイド（completion_gate の概念・モード別動作・トリガー適用ルール） |
| `background-tasks.md` | バックグラウンドタスク実行ガイド（submit の使い方、判断基準、結果の受け取り方） |
| `workspace-guide.md` | ワークスペースガイド（概念・登録・ツールでの使用・トラブルシューティング） |
| `model-guide.md` | → `reference/operations/model-guide.md` に移動。モデル選択・設定ガイド |
| `mode-s-auth-guide.md` | → `reference/operations/mode-s-auth-guide.md` に移動。Mode S 認証モード設定ガイド |
| `voice-chat-guide.md` | → `reference/operations/voice-chat-guide.md` に移動。音声チャットガイド |

### operations/machine/ — machine ツールワークフロー

| ファイル | 概要 |
|---------|------|
| `tool-usage.md` | machine ツール利用ガイド（共通原則・メタパターン・ステータス管理・レート制限） |
| `workflow-pdm.md` | machine ワークフロー — PdM（調査→計画書作成） |
| `workflow-engineer.md` | machine ワークフロー — Engineer（実装詳細化→実装→検証） |
| `workflow-reviewer.md` | machine ワークフロー — Reviewer（レビュー→メタレビュー） |
| `workflow-tester.md` | machine ワークフロー — Tester（テスト設計→実行→結果検証） |

### team-design/ — チーム設計（目的別）

| ファイル | 概要 |
|---------|------|
| `guide.md` | Anima チーム設計の基本原則（役割分離・ハンドオフ・スケール・兼務） |
| `development/team.md` | 開発フルチーム — 4ロール（PdM・Engineer・Reviewer・Tester）・ハンドオフ・スケーリング |
| `legal/team.md` | 法務フルチーム — 3ロール（Director・Verifier・Researcher）・carry-forward tracker・スケーリング |
| `finance/team.md` | 財務フルチーム — 4ロール（Director・Auditor・Analyst・Collector）・Variance Tracker・Data Lineage・スケーリング |
| `trading/team.md` | トレーディングフルチーム — 4ロール（Director・Analyst・Engineer・Auditor）・Performance Tracker・Ops Issue Tracker・PDCA対応・スケーリング |
| `sales-marketing/team.md` | 営業・マーケティングフルチーム — 4ロール（Director・Creator・SDR・Researcher）・Campaign Pipeline Tracker・Deal Pipeline Tracker・2実行モード・スケーリング |
| `secretary/team.md` | 秘書チーム（人間直属）— 1ロール（Secretary）・情報トリアージ・代行送信・書類作成（machine）・スケーリング |
| `coo/team.md` | COO（事業統括）チーム（人間直属）— 1ロール（COO）・委任判断・部門監視・KPI集計・経営報告（machine）・スケーリング |
| `customer-success/team.md` | CS（カスタマーサクセス）フルチーム — 2ロール（CS Lead・Support）・Customer Health Score Tracker・VoC レポート・4フェーズ machine 活用・スケーリング |
| `corporate-planning/team.md` | 経営企画フルチーム — 3ロール（Strategist・Analyst・Coordinator）・Strategic Initiative Tracker・独立検証（メタ検証）・スケーリング |
| `infrastructure/team.md` | インフラ/SRE 監視チーム — 2ロール（Infra Director・Monitor）・監視チームパターン（machine 不使用）・報告テンプレート3種・3段階エスカレーション・スケーリング |
| `org-chart-template.md` | 組織図テンプレート — 推奨階層（スタッフ/ライン分離）・部門間ハンドオフマップ・段階的導入ガイド |

ロール別テンプレート: `team-design/development/{pdm,engineer,reviewer,tester}/` — `injection.template.md`, `machine.md`, `checklist.md`

ロール別テンプレート: `team-design/legal/{director,verifier,researcher}/` — `injection.template.md`, `machine.md`（該当ロールのみ）, `checklist.md`

ロール別テンプレート: `team-design/finance/{director,auditor,analyst,collector}/` — `injection.template.md`, `machine.md`（該当ロールのみ）, `checklist.md`

ロール別テンプレート: `team-design/trading/{director,analyst,engineer,auditor}/` — `injection.template.md`, `machine.md`, `checklist.md`

ロール別テンプレート: `team-design/sales-marketing/{director,creator,sdr,researcher}/` — `injection.template.md`, `machine.md`（researcher 除く）, `checklist.md`

ロール別テンプレート: `team-design/secretary/secretary/` — `injection.template.md`, `machine.md`, `checklist.md`

ロール別テンプレート: `team-design/coo/coo/` — `injection.template.md`, `machine.md`, `checklist.md`

ロール別テンプレート: `team-design/customer-success/{cs-lead,support}/` — `injection.template.md`, `machine.md`（CS Lead のみ）, `checklist.md`

ロール別テンプレート: `team-design/corporate-planning/{strategist,analyst,coordinator}/` — `injection.template.md`, `machine.md`, `checklist.md`

ロール別テンプレート: `team-design/infrastructure/{director,monitor}/` — `injection.template.md`, `checklist.md`（machine.md なし）

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
| 基礎, 入門, 全体像, エッセンシャル, 始め方, 概要 | `anatomy/essentials.md` |
| メッセージ, send_message, 送信, 返信, スレッド, inbox | `communication/messaging-guide.md` |
| Board, チャネル, post_channel, read_channel | `communication/board-guide.md` |
| DM履歴, read_dm_history, 過去の会話 | `communication/board-guide.md` |
| 指示, 委任, タスク依頼, デリゲーション | `communication/instruction-patterns.md` |
| 報告, 日報, サマリー, 完了報告, エスカレーション | `communication/reporting-guide.md` |
| 品質プロトコル, 必須項目, 検証根拠, 完了条件, 委譲チェック | `communication/message-quality-protocol.md` |
| レート制限, 送信制限, 30通, 100通, 1ラウンドルール | `communication/sending-limits.md` |
| call_human, 人間通知, 人間に連絡, 通知チャネル | `communication/call-human-guide.md` |
| Slack, ボットトークン, SLACK_BOT_TOKEN, not_in_channel | `reference/communication/slack-bot-token-guide.md` |
| 組織, supervisor, 上司, 部下, 同僚 | `reference/organization/structure.md` |
| 役割, 責任, speciality, 専門 | `organization/roles.md` |
| 階層, 通信経路, org_dashboard, ping_subordinate | `organization/hierarchy-rules.md` |
| delegate_task, タスク委譲, task_tracker | `organization/hierarchy-rules.md`, `operations/task-management.md` |
| sync_delegated, 委譲同期, 自動同期 | `operations/task-management.md`, `operations/task-delegation-guide.md` |
| タスク, current_state, pending, 進捗, 優先順位 | `operations/task-management.md` |
| タスクキュー, submit_tasks, update_task, TaskExec, animaworks-tool task list | `operations/task-management.md` |
| タスクボード, ダッシュボード, 人間向け | `operations/task-board-guide.md` |
| completion_gate, 完了検証, 事前チェック, Stop hook | `operations/completion-gate-guide.md` |
| 設定, config, status.json, SSoT, reload | `reference/operations/project-setup.md` |
| ハートビート, heartbeat, 定期チェック | `operations/heartbeat-cron-guide.md` |
| cron, スケジュール, 定時タスク | `operations/heartbeat-cron-guide.md` |
| ツール, animaworks-tool, MCP, skill | `operations/tool-usage-overview.md` |
| 実行モード, S-mode, C-mode, D-mode, G-mode, A-mode, B-mode | `operations/tool-usage-overview.md` |
| バックグラウンド, submit, 長時間ツール | `operations/background-tasks.md` |
| machine, machine run, 外部エージェント, 計画書 | `operations/machine/tool-usage.md` |
| 調査, investigation, PdM, plan.md | `operations/machine/workflow-pdm.md` |
| impl-plan, 具体化, 実装計画 | `operations/machine/workflow-engineer.md` |
| レビュー, review, メタレビュー | `operations/machine/workflow-reviewer.md` |
| テスト, test, E2E, テスター | `operations/machine/workflow-tester.md` |
| チーム設計, 役割分離, 開発チーム, PdM, ハンドオフ | `team-design/guide.md`, `team-design/development/team.md` |
| 法務, 契約, リスク, 監査, 検証, carry-forward, 楽観バイアス | `team-design/legal/team.md` |
| 財務, 分析, 試算表, ポートフォリオ, 検算, variance tracker, data lineage | `team-design/finance/team.md` |
| トレーディング, bot, P&L, ドローダウン, バックテスト, PDCA, performance tracker, ops issue tracker, リスク監査 | `team-design/trading/team.md` |
| 営業, マーケティング, コンテンツ, リード, ナーチャリング, BANT, パイプライン, campaign tracker, deal tracker, SDR, Brand Voice | `team-design/sales-marketing/team.md` |
| 秘書, secretary, トリアージ, 代行送信, 書類作成, スケジュール, 人間直属, call_human, 情報分配 | `team-design/secretary/team.md` |
| COO, 事業統括, 委任判断, 部門監視, KPI, 経営報告, スパンオブコントロール, 部門横断調整, 組織分析 | `team-design/coo/team.md` |
| CS, カスタマーサクセス, オンボーディング, ヘルススコア, チャーン, リテンション, VoC, NPS, CSAT, cs-handoff, Health Tracker | `team-design/customer-success/team.md` |
| 経営企画, 戦略, OKR, KPI, 事業分析, イニシアチブ, Strategic Initiative Tracker, SWOT, PEST, 独立検証, メタ検証 | `team-design/corporate-planning/team.md` |
| インフラ, SRE, 監視, モニタリング, NOC, 異常検知, エスカレーション, 集約報告, Infra Director, Monitor, 報告テンプレート, cron, heartbeat | `team-design/infrastructure/team.md` |
| 組織図, org-chart, 部門配置, 導入順序, ハンドオフマップ, スタッフ, ライン, 段階的導入 | `team-design/org-chart-template.md` |
| ワークスペース, workspace, 作業ディレクトリ, working_directory | `operations/workspace-guide.md` |
| モデル, models.json, credential, set-model, コンテキストウィンドウ | `reference/operations/model-guide.md` |
| background_model, バックグラウンドモデル, コスト最適化 | `reference/operations/model-guide.md` |
| Mode S, 認証, API直接, Bedrock, Vertex AI, Max plan | `reference/operations/mode-s-auth-guide.md` |
| 音声, voice, STT, TTS, VOICEVOX, ElevenLabs | `reference/operations/voice-chat-guide.md` |
| WebSocket, /ws/voice, barge-in, VAD, PTT | `reference/operations/voice-chat-guide.md` |
| Anima, 自分, 構成, 設計, ライフサイクル | `anatomy/what-is-anima.md` |
| identity, injection, 人格, 行動指針, 不変, 可変 | `reference/anatomy/anima-anatomy.md` |
| permissions.json, bootstrap, heartbeat.md, cron.md | `reference/anatomy/anima-anatomy.md` |
| 記憶, memory, episodes, knowledge, procedures | `anatomy/memory-system.md` |
| Priming, RAG, Consolidation, Forgetting, 忘却 | `anatomy/memory-system.md` |
| consolidation, 2-phase, multipass, エラートレース | `anatomy/memory-system.md` |
| search_memory, write_memory_file, 記憶検索 | `anatomy/memory-system.md` |
| skills, スキル検索, common_skills, search_memory scope="skills" | `anatomy/memory-system.md`, `operations/tool-usage-overview.md` |
| activity_log, BM25, RRF, 直近のログ検索 | `anatomy/memory-system.md`, `troubleshooting/common-issues.md` |
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
