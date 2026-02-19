# AnimaWorks 機能一覧

**[English version](features.md)**

> 最終更新: 2026-02-18
> 関連: [spec.md](spec.md), [memory.ja.md](memory.ja.md), [vision.ja.md](vision.ja.md)

AnimaWorksフレームワークの実装済み機能を18カテゴリに分類した索引。各エントリの「設計」リンクは設計・実装文書、「Review」リンクはコードレビューレポートを参照する。

---

## 1. コアアーキテクチャ

agent.pyリファクタリング、階層設計、プロセス隔離、リネーム等のフレームワーク基盤変更。

- **agent.py リファクタリング計画書** (2026-02-14) — エージェントコアの責務分離とクラス設計の見直し
  [設計](implemented/20260214_agent-refactoring_implementation.md)
- **設計書と実装の差異ノート** (2026-02-14) — 原案設計書と実装コードの乖離を特定・修正
  [設計](implemented/20260214_design-implementation-gap_issue.md)
- **動的システムプロンプト注入アーキテクチャ** (2026-02-14) — 18セクション構成のシステムプロンプト動的構築
  [設計](implemented/20260214_dynamic-prompt-injection_implementation.md)
- **階層構造・非同期通信・マルチモデル設計書** (2026-02-14) — supervisor階層、A1/A2/Bモード、マルチプロバイダ設計
  [設計](implemented/20260214_hierarchy-and-delegation_design.md)
- **設計書準拠 修正計画ノート** (2026-02-14) — 設計書と実装のギャップを埋める修正計画
  [設計](implemented/20260214_plan-gap-fix_implementation.md)
- **API Critical Refactoring — スケーリング対応** (2026-02-15) — FastAPIルート分割とスケーリング基盤整備
  [設計](implemented/20260215_api-critical-refactoring-for-scaling_implemented-20260215.md) | [Review](implemented/20260215_review_api-critical-refactoring-for-scaling_approved-20260215.md)
- **システムリファレンスドキュメント作成** (2026-02-15) — Animaが自律参照できる共有知識ベース
  [設計](implemented/20260215_system-reference-documents_implemented-20260215.md)
- **Person ライフサイクル CLI コマンド統合** (2026-02-16) — CLIコマンド体系のAnima対応整理
  [設計](implemented/20260216_person-lifecycle-cli-commands-20260217.md) | [Review](implemented/20260217_review_person-lifecycle-cli-commands_approved-20260217.md)
- **Anima → Anima 全面リネーム** (2026-02-16) — Person→Animaへの全コードベースリネーム
  [設計](implemented/20260216_rename-person-to-anima_implemented-20260216.md) | [Review](implemented/20260216_review_rename-person-to-anima_approved-20260216.md)
- **ドキュメント全面改訂** (2026-02-17) — CLAUDE.md・README・specを現行コードベースに同期
  [設計](implemented/20260217_documentation-overhaul-implemented-20260217.md) | [Review](implemented/20260217_review_documentation-overhaul_approved-20260217.md)

---

## 2. 実行エンジン

A1/A2/B各モード改善、Agent SDKクラッシュリカバリー、SSE改善等。

- **A2エージェンティックループの高度化** (2026-02-15) — LiteLLM + tool_useループの信頼性・機能向上
  [設計](implemented/20260215_a2-agentic-loop-enhancement_implemented-20260215.md) | [Review](implemented/20260215_review_a2-agentic-loop-enhancement_approved-20260215.md)
- **エラー時の会話データ消失問題** (2026-02-15) — 実行エラー発生時のコンテキスト保全
  [設計](implemented/20260215_error-conversation-data-loss_implemented-20260215.md) | [Review](implemented/20260215_review_error-conversation-data-loss_approved-20260215.md)
- **ストリーム切断時のチェックポイントリトライ機構** (2026-02-16) — SSEストリーム切断時の自動復旧
  [設計](implemented/20260216_checkpoint-retry-on-stream-disconnect-implemented-20260216.md) | [Review](implemented/20260216_review_checkpoint-retry-on-stream-disconnect_approved-20260216.md)
- **Mode A1 で create_anima ツールが利用不可** (2026-02-16) — Agent SDKモードでのAnima雇用連鎖障害修正
  [設計](implemented/20260216_mode-a1-create-anima-unavailable.md) | [Review](implemented/20260216_review_mode-a1-create-anima-unavailable_approved.md)
- **SSEストリームがアニマ再起動後に空応答を返す** (2026-02-16) — プロセス再起動後のSSEストリーム復旧
  [設計](implemented/20260216_sse-stream-empty-after-anima-restart_implemented-20260216.md) | [Review](implemented/20260216_review_sse-stream-empty-after-anima-restart_approved-20260216.md)
- **uvicornタイムアウト設定不備とAgent SDK hookレースコンディション** (2026-02-16) — サーバー初期化の安定性向上
  [設計](implemented/20260216_uvicorn-timeout-and-agent-sdk-hook-errors_implemented-20260216.md) | [Review](implemented/20260216_review_uvicorn-timeout-and-agent-sdk-hook-errors_approved-20260216.md)
- **Agent SDKクラッシュ時のプロセス自動復旧** (2026-02-17) — クラッシュ検知・自動再起動メカニズム
  [設計](implemented/20260217_agent-sdk-crash-recovery_implemented-20260217.md) | [Review](implemented/20260217_review_agent-sdk-crash-recovery_approved-20260217.md)
- **Mode B: テキストベース擬似ツールコールループ** (2026-02-17) — tool_use非対応モデル向けのテキストベースツール実行
  [設計](implemented/20260217_mode-b-text-based-tool-loop_implemented-20260217.md) | [Review](implemented/20260217_review_mode-b-text-based-tool-loop_approved-20260217.md)
- **SSE チャットストリーミング 3重複コードの統合** (2026-02-17) — SSEストリーミングコードのDRY化
  [設計](implemented/20260217_sse-chat-code-deduplication_implemented-20260217.md) | [Review](implemented/20260217_review_sse-chat-code-deduplication_approved-20260217.md)

---

## 3. 記憶システム

Priming、RAG、記憶統合/忘却、アクティビティログ、ストリーミングジャーナル等。

- **プライミングレイヤー実装計画書** (2026-02-14) — RAG設計、固定化アーキテクチャを含む全体計画
  [設計](implemented/20260214_priming-layer_design.md)
- **プライミングレイヤー Phase 1 実装完了** (2026-02-14) — 4チャネル並列プライミングの初期実装
  [設計](implemented/20260214_priming-layer-phase1_implementation.md)
- **プライミングレイヤー Phase 2: 日次固定化** (2026-02-14) — NREM睡眠アナログの日次記憶固定化
  [設計](implemented/20260214_priming-layer-phase2-consolidation_implementation.md)
- **記憶パフォーマンス評価 Phase 2: データセット生成** (2026-02-14) — 記憶検索精度評価用データセットの自動生成
  [設計](implemented/20260214_memory-eval-phase2_dataset-generation.md)
- **common_knowledge RAGインフラ整備** (2026-02-15) — 共有知識ベースのベクトル検索対応
  [設計](implemented/20260215_common-knowledge-rag-infrastructure-implemented-20260215.md) | [Review](implemented/20260215_common-knowledge-rag-infrastructure_review-approved-20260215.md)
- **記憶の活性化強度と能動的忘却メカニズム** (2026-02-15) — ヘブ則LTP + 3段階忘却の実装
  [設計](implemented/20260215_memory-access-frequency-and-forgetting-implemented-20260215.md) | [Review](implemented/20260215_review_memory-access-frequency-and-forgetting_approved-20260215.md)
- **記憶検索の神経科学的拡張 — 3つの改善提案** (2026-02-15) — 拡散活性化・時間減衰・アクセス頻度の改善
  [設計](implemented/20260215_memory-retrieval-neuroscience-enhancements.md)
- **プライミング導入後のhiring_context修正** (2026-02-15) — プライミングと雇用コンテキストの競合解消
  [設計](implemented/20260215_priming-hiring-context-fix.md) | [Review](implemented/20260215_review_priming-hiring-context-fix_approved.md)
- **RAG: Embedding Model の多重初期化によるパフォーマンス劣化** (2026-02-15) — シングルトン化による起動時間短縮
  [設計](implemented/20260215_rag-embedding-model-multi-init_implemented-20260215.md)
- **Dense Vector単独検索への移行 + 知識グラフ完全実装** (2026-02-15) — BM25廃止、密ベクトル検索一本化 + NetworkX PageRank
  [設計](implemented/20260215_simplify-rag-to-dense-vector-only_implemented-20260215.md) | [Review](implemented/20260215_review_simplify-rag-to-dense-vector-only_approved-20260215.md)
- **_chunk_by_markdown_headings のチャンクID衝突バグ** (2026-02-16) — preambleセクションのID重複修正
  [設計](implemented/20260216_chunk-id-preamble-collision-20260216.md) | [Review](implemented/20260216_review_chunk-id-preamble-collision_approved-20260216.md)
- **統合プロンプトの改善とLLMレスポンスのログ追加** (2026-02-16) — 記憶統合品質向上とデバッグ支援
  [設計](implemented/20260216_consolidation-prompt-and-logging-improvement-20260216.md)
- **RAG インデクサーのチャンクID重複バグ & shared_common_knowledge 初期化欠落** (2026-02-16) — RAGインデックスのパスバグ修正
  [設計](implemented/20260216_rag-indexer-path-bug-and-shared-knowledge-init-20260216.md) | [Review](implemented/20260216_review_rag-indexer-path-bug-and-shared-knowledge-init_approved-20260216.md)
- **Anima間メッセージがエピソード記憶に記録されない** (2026-02-17) — DM/チャネルメッセージの記憶記録漏れ修正
  [設計](implemented/20260217_anima-message-episode-recording_implemented-20260217.md) | [Review](implemented/20260217_review_anima-message-episode-recording_approved-20260217.md)
- **日次統合エンジンがサフィックス付きエピソードファイルを無視する** (2026-02-17) — ファイル名パターンマッチの修正
  [設計](implemented/20260217_consolidation-episode-filename-pattern_implemented-20260217.md)
- **記憶統合が停止中のAnimaに対して実行されない** (2026-02-17) — 全初期化済みAnimaへの統合実行保証
  [設計](implemented/20260217_consolidation-run-for-all-animas_implemented-20260217.md) | [Review](implemented/20260217_review_consolidation-run-for-all-animas_approved-20260217.md)
- **埋め込みモデル比較テスト結果** (2026-02-17) — multilingual-e5-small vs 他モデルの精度比較
  [設計](implemented/20260217_embedding-model-comparison.md)
- **write_memory_file ツールに episodes/ パス検証を追加** (2026-02-17) — 不正パスへの記憶書き込み防止
  [設計](implemented/20260217_write-memory-file-episode-path-validation_implemented-20260217.md)
- **Unified Activity Log: 仕様準拠修正（6項目）** (2026-02-18) — アクティビティログの仕様準拠性向上
  [設計](implemented/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md) | [Review](implemented/20260218_review_activity-log-spec-compliance_approved-20260218.md)
- **ストリーミングジャーナル — クラッシュ耐性のある応答出力永続化** (2026-02-18) — WALによるストリーミング出力の耐障害性
  [設計](implemented/20260218_streaming-journal-implemented-20260218.md) | [Review](implemented/20260218_review_streaming-journal_approved-20260218.md)
- **統一アクティビティログ — 全インタラクションの単一時系列記録化** (2026-02-18) — transcript/dm_log/heartbeat_historyの統合
  [設計](implemented/20260218_unified-activity-log-implemented-20260218.md) | [Review](implemented/20260218_review_unified-activity-log_approved-20260218.md)

---

## 4. 通信・メッセージング

Board/共有チャネル、外部メッセージング、アウトバウンドルーティング等。

- **A1モードのAnima間メッセージング統合** (2026-02-15) — Agent SDKモードでの内部メッセージング実装
  [設計](implemented/20260215_a1-messaging-integration_implemented-20260215.md) | [Review](implemented/20260215_review_a1-messaging-integration_approved-20260215.md)
- **Greet重複呼び出しとInbox孤立ファイルの残存問題** (2026-02-16) — メッセージングの重複・孤立問題修正
  [設計](implemented/20260216_greet-duplicate-and-orphaned-inbox-files_implemented-20260216.md) | [Review](implemented/20260216_review_greet-duplicate-and-orphaned-inbox-files_approved-20260216.md)
- **外部メッセージングWebhook統合（Slack・Chatwork）** (2026-02-17) — 外部サービスからのWebhookメッセージ受信
  [設計](implemented/20260217_external-messaging-integration_implemented-20260217.md) | [Review](implemented/20260217_review_external-messaging-integration_approved-20260217.md)
- **send_message 統一アウトバウンドルーティング** (2026-02-17) — 宛先に応じたSlack/Chatwork/内部の自動振り分け
  [設計](implemented/20260217_send-message-unified-outbound-routing.md)
- **Slack Socket Modeによるリアルタイムメッセージ受信** (2026-02-17) — WebSocket経由のSlakリアルタイム連携
  [設計](implemented/20260217_slack-socket-mode-integration_implemented-20260217.md) | [Review](implemented/20260217_review_slack-socket-mode-integration_approved-20260217.md)
- **Board WebUI実装 — ダッシュボード + ワークスペース** (2026-02-18) — 共有チャネルのWebUIフロントエンド
  [設計](implemented/20260218_channel-webui-and-onboarding_implemented-20260218.md) | [Review](implemented/20260218_review_channel-webui-and-onboarding_approved-20260218.md)
- **Slack型共有チャネル + 統一メッセージログアーキテクチャ** (2026-02-18) — #general/#ops等の共有チャネル基盤
  [設計](implemented/20260218_shared-channel-messaging_implemented-20260218.md) | [Review](implemented/20260218_review_shared-channel-messaging_approved-20260218.md)

---

## 5. スケジュール・ライフサイクル

ハートビート、cron、ブートストラップ、Reconciliation等。

- **bootstrap.md が全 Anima に無条件で配置される** (2026-02-14) — 初回起動指示の条件付き配置化
  [設計](implemented/20260214_bootstrap-unconditional-placement_issue.md)
- **Cron Command型タスク設計書** (2026-02-14) — cronタスクのcommand型実行方式
  [設計](implemented/20260214_cron-command-type_issue.md)
- **ハートビートカスケード問題 — 修正検討ノート** (2026-02-14) — 連鎖的ハートビート暴走の防止
  [設計](implemented/20260214_heartbeat-cascade-fix_notes.md)
- **ハートビートカスケード問題 — インシデント報告** (2026-02-14) — 2026-02-13発生のカスケード障害分析
  [設計](implemented/20260214_heartbeat-cascade_incident.md)
- **ブートストラップのバックグラウンド実行化 + タイムアウト改善** (2026-02-15) — 初回起動の非同期化とタイムアウト制御
  [設計](implemented/20260215_bootstrap-background-execution-and-timeout_implemented-20260215.md) | [Review](implemented/20260215_review_bootstrap-background-execution_approved-20260215.md)
- **ハートビート間隔を30分固定にする** (2026-02-15) — 間隔パース機能の削除と固定化
  [設計](implemented/20260215_fix-heartbeat-interval-to-30min_implemented-20260215.md)
- **Animaごとの自律スケジューラ導入** (2026-02-15) — heartbeat/cronの子プロセス内実行
  [設計](implemented/20260215_person-autonomous-scheduler_implemented-20260215.md) | [Review](implemented/20260215_review_person-autonomous-scheduler_approved-20260215.md)
- **Animaの自律的業務管理: ハートビート/cronの自己更新** (2026-02-15) — Anima自身によるスケジュール更新
  [設計](implemented/20260215_self-modify-heartbeat-cron-implemented-20260215.md) | [Review](implemented/20260215_review_self-modify-heartbeat-cron_approved-20260215.md)
- **Supervisor定期リコンシリエーション** (2026-02-15) — 未起動Anima自動検出・起動メカニズム
  [設計](implemented/20260215_supervisor-person-reconciliation_implemented-20260215.md) | [Review](implemented/20260215_review_supervisor-person-reconciliation_approved-20260215.md)
- **Heartbeat競合クラッシュ + 孤児ディレクトリ検出 + 組織ツリー可視化** (2026-02-16) — ハートビート排他制御と孤児プロセス検出
  [設計](implemented/20260216_heartbeat-collision-orphan-detection-org-tree-implemented-20260216.md) | [Review](implemented/20260216_review_heartbeat-collision-orphan-detection-org-tree_approved-20260216.md)
- **スケジューラーリグレッション・Activity Tab表示不具合** (2026-02-16) — スケジューラー回帰バグの修正
  [設計](implemented/20260216_scheduler-regression-and-activity-tab-20260216.md) | [Review](implemented/20260216_review_scheduler-regression-and-activity-tab_approved-20260216.md)
- **cron.mdテンプレート改善 + Mode Bスキル注入修正** (2026-02-17) — cronテンプレートの品質向上
  [設計](implemented/20260217_cron-template-and-mode-b-skill-fix_implemented-20260217.md) | [Review](implemented/20260217_review_cron-template-and-mode-b-skill-fix_approved-20260217.md)
- **ハートビート・対話間のコンテキスト断絶とメッセージング改善** (2026-02-17) — ハートビートと会話のコンテキスト共有
  [設計](implemented/20260217_heartbeat-dialogue-context-gap-and-messaging_implemented-20260217.md) | [Review](implemented/20260217_review_heartbeat-dialogue-context-gap-and-messaging_approved-20260217.md)
- **heartbeat処理中のLLM応答をSSEで中継** (2026-02-17) — ハートビート中の応答リアルタイム配信
  [設計](implemented/20260217_heartbeat-sse-relay-implemented-20260217.md)
- **Reconciliation がブートストラップ中の Anima を kill し無限ループ** (2026-02-17) — ブートストラップ保護
  [設計](implemented/20260217_protect-bootstrapping-from-reconciliation-implemented-20260217.md) | [Review](implemented/20260217_review_protect-bootstrapping-from-reconciliation_approved-20260217.md)
- **Reconciliation が status.json 未作成の Anima を30秒周期で KILL** (2026-02-17) — status.json未生成時の保護
  [設計](implemented/20260217_reconciliation-kills-animas-missing-status-json_implemented-20260217.md) | [Review](implemented/20260217_review_reconciliation-kills-animas-missing-status-json_revision-20260217.md)

---

## 6. スキル・ツール

トリガーベース注入、自動検出、バックグラウンドタスク実行等。

- **ツールコールのタイムアウト改善とバックグラウンド実行** (2026-02-15) — 長時間ツールの非同期実行化
  [設計](implemented/20260215_tool-call-timeout-background-execution_implemented-20260216.md) | [Review](implemented/20260216_review_tool-call-timeout-background-execution_approved-20260216.md)
- **長時間ツール実行がチャットをブロックする問題** (2026-02-16) — ツール実行のノンブロッキング化
  [設計](implemented/20260216_long-running-tool-chat-blocking_implemented-20260216.md) | [Review](implemented/20260216_review_long-running-tool-chat-blocking_approved-20260216.md)
- **ツール自動発見・Anima作成・ホットリロード・dispatch規約統一** (2026-02-16) — ツールプラグインの自動検出と統一ディスパッチ
  [設計](implemented/20260216_tool-auto-discovery-creation-hotreload-implemented-20260216.md) | [Review](implemented/20260216_review_tool-auto-discovery-creation-hotreload_approved-20260216.md)
- **バックグラウンドタスク投入システム（Mode A1 長時間ツール対策）** (2026-02-17) — 非同期タスクキューの導入
  [設計](implemented/20260217_background-task-submission_implemented-20260217.md) | [Review](implemented/20260217_review_background-task-submission_approved-20260217.md)
- **スキルマッチング3段階化 + スキルクリエイター** (2026-02-17) — description/trigger/keywordの3段階スキルマッチング
  [設計](implemented/20260217_skill-matching-enhancement-and-skill-creator_implemented-20260218.md) | [Review](implemented/20260217_review_skill-matching-enhancement-and-skill-creator_approved-20260218.md)
- **外部ツール権限をデフォルト全許可（ブラックリスト方式）に変更** (2026-02-17) — ホワイトリストからブラックリスト方式への転換
  [設計](implemented/20260217_tool-permissions-default-all-implemented-20260217.md)
- **スキル形式のClaude Code準拠化 + description ベース自動注入** (2026-02-17) — トリガーベースのスキル注入アーキテクチャ
  [設計](implemented/20260217_trigger-based-skill-injection_implemented-20260217.md) | [Review](implemented/20260217_review_trigger-based-skill-injection_approved-20260217.md)

---

## 7. 設定・認証

クレデンシャル一元化、ロールテンプレート、埋め込みモデル選択等。

- **統一Credential管理: config.json優先カスケードと汎用スキーマ** (2026-02-15) — 認証情報の3層マージ管理
  [設計](implemented/20260215_unified-credential-management_implemented-20260215.md) | [Review](implemented/20260215_review_unified-credential-management_approved-20260215.md)
- **組織構造（supervisor）がconfig.jsonに同期されない** (2026-02-16) — 作成パイプライン修正 + 定期同期機構
  [設計](implemented/20260216_org-structure-config-sync_implemented-20260216.md) | [Review](implemented/20260216_review_org-structure-config-sync_approved-20260216.md)
- **クレデンシャルを shared/credentials.json に一元化** (2026-02-17) — 散在する認証情報の統合
  [設計](implemented/20260217_centralize-credentials-to-shared-file_implemented-20260217.md) | [Review](implemented/20260217_review_centralize-credentials-to-shared-file_approved-20260217.md)
- **埋め込みモデルをconfig.jsonから選択可能にする** (2026-02-17) — RAGモデルの動的切り替え
  [Review](implemented/20260217_review_embedding-model-config_approved.md)
- **load_config() mtime-based cache reload** (2026-02-17) — 設定ファイルの自動リロード
  [Review](implemented/20260217_review_config-mtime-reload_approved-20260217.md)
- **ロールテンプレート導入 — Animaの「役職＋能力値」システム** (2026-02-17) — 6ロールのテンプレート化
  [設計](implemented/20260217_role-templates-and-ability-scores-implemented-20260217.md) | [Review](implemented/20260217_review_role-templates-and-ability-scores_approved-20260217.md)

---

## 8. Web UI: ダッシュボード

SPA移行、アクティビティタイムライン、スケジューラータブ等。

- **ダッシュボードGUI化 — SPA構成への移行** (2026-02-15) — 静的HTMLからSingle Page Applicationへ
  [設計](implemented/20260215_dashboard-gui-spa-migration_implemented-20260215.md) | [Review](implemented/20260215_review_dashboard-gui-spa-migration_approved-20260215.md)
- **Animaステートの視覚的表現改善** (2026-02-15) — Sleeping/Bootstrapping/Activeのステート表示
  [設計](implemented/20260215_bootstrap-ui-during-creation_implemented-20260215.md) | [Review](implemented/20260215_review_bootstrap-ui-during-creation_approved-20260215.md)
- **ダッシュボードUI表示不具合の一括修正** (2026-02-15) — 複数のUI表示問題の包括修正
  [設計](implemented/20260215_fix-dashboard-ui-display-issues_implemented-20260215.md) | [Review](implemented/20260215_review_fix-dashboard-ui-display-issues_approved-20260215.md)
- **Web UI: ブートストラップ中のメッセージがフロントエンドに表示されない** (2026-02-15) — ブートストラップ進捗のリアルタイム表示
  [設計](implemented/20260215_webui-bootstrap-message-invisible_implemented-20260215.md)
- **Web UI: システムステータス表示が壊れている** (2026-02-15) — scheduler_runningフィールド欠落修正
  [設計](implemented/20260215_webui-status-display-broken_implemented-20260215.md)
- **Activity Timeline: メッセージ表示欠損・Cron信頼性・書式統一** (2026-02-16) — タイムライン表示の品質向上
  [設計](implemented/20260216_activity-timeline-message-cron-reliability_implemented-20260216.md) | [Review](implemented/20260216_review_activity-timeline-message-cron-reliability_approved-20260216.md)
- **Activity Timeline: Anima間メッセージ表示・ページネーション・フィルタUI** (2026-02-16) — タイムラインの機能拡張
  [設計](implemented/20260216_activity-timeline-message-visibility-and-pagination-implemented-20260216.md) | [Review](implemented/20260216_review_activity-timeline-message-visibility-and-pagination_revision-20260216.md)
- **crypto.randomUUID 非セキュアコンテキストでフロントエンド全機能クラッシュ** (2026-02-17) — HTTPコンテキストでのUUID生成修正
  [設計](implemented/20260217_fix-crypto-randomuuid-crash_implemented-20260217.md)
- **Activity Timeline メッセージ詳細ポップアップ** (2026-02-17) — タイムラインエントリのクリック詳細表示
  [設計](implemented/20260217_timeline-message-detail-popup_implemented-20260217.md) | [Review](implemented/20260217_review_timeline-message-detail-popup_approved-20260217.md)

---

## 9. Web UI: ワークスペース

3Dオフィス、キャラクター表示、レスポンシブ、iPad対応等。

- **フロントエンド (Workspace) 設計** (2026-02-14) — Three.js + WebSocket ワークスペース初期設計
  [設計](implemented/20260214_frontend-viewer_implementation.md)
- **3Dオフィス キャラクターシミュレーション — 詳細実装仕様書** (2026-02-14) — Three.jsベース3Dオフィスの仕様
  [設計](implemented/20260214_office-simulation_issue.md)
- **Workspace会話画面: エモーションタグ表示 & ステータス通知重複** (2026-02-15) — 感情表現とステータス通知のUI修正
  [設計](implemented/20260215_fix-workspace-chat-emotion-tag-and-status-notifications_implemented-20260215.md) | [Review](implemented/20260215_review_fix-workspace-chat-emotion-tag-and-status-notifications_approved-20260215.md)
- **Workspace: Anima誕生時の確定演出（Reveal Animation）** (2026-02-15) — 新Anima作成時のアニメーション演出
  [設計](implemented/20260215_person-birth-reveal-animation_implemented-20260215.md) | [Review](implemented/20260215_review_person-birth-reveal-animation_approved-20260215.md)
- **Live2D Canvas手続き描画の削除 — 静的イラスト配置への簡素化** (2026-02-15) — Live2Dから静的画像表示への移行
  [設計](implemented/20260215_remove-live2d-canvas-rendering_implemented-20260215.md) | [Review](implemented/20260215_review_remove-live2d-canvas-rendering_approved-20260215.md)
- **Workspace チャット吹き出しの途中切れ問題** (2026-02-15) — チャットバブルのテキスト表示修正
  [設計](implemented/20260215_workspace-chat-bubble-cutoff_implemented-20260215.md) | [Review](implemented/20260215_review_workspace-chat-bubble-cutoff_approved-20260215.md)
- **ワークスペース3Dキャラクタークリック時の挨拶機能** (2026-02-15) — 3Dキャラクターとのインタラクション
  [設計](implemented/20260215_workspace_character_greeting_implemented-20260215.md) | [Review](implemented/20260215_review_workspace_character_greeting_approved-20260215.md)
- **GLBキャッシュの scene.clone(true) によるスケルトンバインディング破壊** (2026-02-16) — SkinnedMeshクローン時の3Dモデル修正
  [設計](implemented/20260216_glb-skinnedmesh-clone-skeleton-binding-broken-20260216.md) | [Review](implemented/20260216_review_glb-skinnedmesh-clone-skeleton-binding-broken_approved-20260216.md)
- **レスポンシブデザイン対応 — ダッシュボード & ワークスペース モバイルUX** (2026-02-16) — モバイル・タブレット対応
  [設計](implemented/20260216_responsive-design-mobile-ux_implemented-20260216.md) | [Review](implemented/20260216_review_responsive-design-mobile-ux_approved-20260216.md)
- **Workspace 3Dオフィス: キャラクタースケール異常 + 組織階層ツリー不正** (2026-02-16) — 3Dスケーリングとツリーレイアウトの修正
  [設計](implemented/20260216_workspace-character-scale-and-hierarchy-bug_implemented-20260216.md) | [Review](implemented/20260216_review_workspace-character-scale-and-hierarchy-bug_approved-20260216.md)
- **Workspace iPad表示修正** (2026-02-16) — ビューポート・タイムライン配置・レスポンシブ
  [設計](implemented/20260216_workspace-ipad-viewport-fix_implemented-20260216.md) | [Review](implemented/20260216_review_workspace-ipad-viewport-fix_approved-20260216.md)
- **Workspace 3Dオフィス: ツリーレイアウトが機能せず全員横並び** (2026-02-16) — 組織階層に基づくレイアウト修正
  [設計](implemented/20260216_workspace-tree-layout-broken_implemented-20260216.md)
- **Workspace 3D キャラクタースケーリング修正** (2026-02-17) — キャラクター表示サイズの修正
  [設計](implemented/20260217_workspace-character-scaling-fix_implemented-20260217.md)

---

## 10. Web UI: チャット

無限スクロール、マルチモーダル画像入力、SSE再接続等。

- **チャット画面のメッセージ二重表示問題** (2026-02-15) — WebSocketとSSEの重複メッセージ修正
  [設計](implemented/20260215_chat-duplicate-message_implemented-20260215.md) | [Review](implemented/20260215_review_chat-duplicate-message_approved-20260215.md)
- **グリーティングメッセージの二重表示とプロンプト改善** (2026-02-15) — 挨拶メッセージの重複排除
  [設計](implemented/20260215_greeting-duplicate-and-prompt-improvement-20260216.md) | [Review](implemented/20260216_review_greeting-duplicate-and-prompt-improvement_approved.md)
- **ストリーミング中のローディング表示改善** (2026-02-15) — ツール呼び出し中のUI改善
  [設計](implemented/20260215_tool-call-loading-indicator-visibility-implemented-20260215.md) | [Review](implemented/20260215_review_tool-call-loading-indicator-visibility_approved-20260215.md)
- **会話履歴の無限スクロール（ページネーション）** (2026-02-17) — 過去のチャット履歴の動的読み込み
  [設計](implemented/20260217_chat-history-infinite-scroll_implemented-20260217.md) | [Review](implemented/20260217_review_chat-history-infinite-scroll_approved-20260217.md)
- **WebUIチャット マルチモーダル画像入力対応** (2026-02-17) — チャットへの画像添付・送信機能
  [設計](implemented/20260217_multimodal-image-input-for-chat_implemented-20260217.md) | [Review](implemented/20260217_review_multimodal-image-input-for-chat_approved-20260217.md)
- **SSEストリーム再接続・進行状態リカバリ** (2026-02-17) — SSE切断時の自動再接続と状態復元
  [設計](implemented/20260217_sse-reconnection-and-progress-recovery_implemented-20260217.md) | [Review](implemented/20260217_review_sse-reconnection-and-progress-recovery_approved-20260217.md)

---

## 11. アセット生成

画像生成パイプライン、表情差分、3Dモデルキャッシュ、NovelAI V4等。

- **キャラクター画像の絵柄一貫性** (2026-02-14) — 複数Animaの絵柄統一手法
  [設計](implemented/20260214_avatar-style-consistency_issue.md) | [Review](implemented/20260214_review_avatar-style-consistency_revision.md)
- **バストアップ表情差分システム** (2026-02-14) — 感情に応じた表情バリエーション生成
  [設計](implemented/20260214_bustup-expression-system_implemented-20260215.md) | [Review](implemented/20260214_review_bustup-expression-system_approved-20260215.md)
- **キャラクター画像・3Dモデル生成パイプライン** (2026-02-14) — NovelAI/Flux/Meshyを統合した生成パイプライン
  [設計](implemented/20260214_image-gen-pipeline_issue.md)
- **部下作成時に上司の画像をVibe Transfer参照画像として自動適用** (2026-02-15) — 絵柄継承の自動化
  [設計](implemented/20260215_supervisor-image-as-vibe-reference_implemented-20260215.md) | [Review](implemented/20260215_review_supervisor-image-as-vibe-reference_approved-20260215.md)
- **3Dモデルのダウンロード量削減・キャッシュ・圧縮の3層最適化** (2026-02-16) — GLBモデルのパフォーマンス改善
  [設計](implemented/20260216_3d-model-cache-and-optimization_implemented-20260216.md) | [Review](implemented/20260216_review_3d-model-cache-and-optimization_approved-20260216.md)
- **アセット欠落時の自動生成フォールバックパイプライン** (2026-02-16) — 画像未生成時の自動フォールバック
  [設計](implemented/20260216_asset-generation-fallback-pipeline_implemented-20260216.md) | [Review](implemented/20260216_review_asset-generation-fallback-pipeline_approved-20260216.md)
- **Asset Reconciler: LLM による画像プロンプト自動合成** (2026-02-16) — キャラクター情報からの画像プロンプト生成
  [設計](implemented/20260216_asset-reconciler-llm-prompt-synthesis_implemented-20260216.md) | [Review](implemented/20260216_review_asset-reconciler-llm-prompt-synthesis_approved-20260216.md)
- **キャラクターアセット リメイク機能（Vibe Transfer + Web UI プレビュー）** (2026-02-16) — 既存画像のスタイル変換
  [設計](implemented/20260216_character-asset-remake-with-style-transfer-implemented-20260216.md) | [Review](implemented/20260216_review_character-asset-remake-with-style-transfer_revision-20260216.md)
- **NovelAI V4/V4.5 Vibe Transfer: encode-vibe 未使用による 500 エラー** (2026-02-16) — Vibe Transfer APIの修正
  [設計](implemented/20260216_novelai-v4-vibe-transfer-encode-fix_implemented-20260216.md)

---

## 12. プロセス管理・IPC

ゾンビ検出、keepalive、バッファオーバーフロー修正等。

- **プロセス隔離アーキテクチャ** (2026-02-14) — Unix Domain Socket + 子プロセスによるAnima隔離
  [設計](implemented/20260214_process-isolation_issue.md) | [Review](implemented/20260214_review_process-isolation-design_revision.md)
- **IPC層でdatetimeオブジェクトのJSONシリアライズに失敗する** (2026-02-15) — IPC通信のシリアライズ修正
  [設計](implemented/20260215_fix-ipc-datetime-serialization_implemented-20260215.md) | [Review](implemented/20260215_review_fix-ipc-datetime-serialization_approved-20260215.md)
- **個別Animaプロセス管理APIの実装** (2026-02-16) — REST APIによるプロセス制御
  [設計](implemented/20260216_individual-anima-restart-api_implemented-20260216.md) | [Review](implemented/20260216_review_individual-anima-restart-api_approved-20260216.md)
- **ping() が FAILED 状態でカウンタを増加させず、ゾンビ化** (2026-02-16) — ゾンビプロセス検出の修正
  [設計](implemented/20260216_ipc-ping-counter-zombie-state_implemented-20260216.md)
- **IPC readline() バッファ上限 64KB で大きなメッセージ送信時に即死** (2026-02-16) — バッファオーバーフロー対策
  [設計](implemented/20260216_ipc-readline-buffer-overflow_implemented-20260216.md)
- **IPC/SSEストリームにkeep-aliveとチャンク間タイムアウトを導入** (2026-02-16) — 接続維持とタイムアウト制御
  [設計](implemented/20260216_ipc-stream-keepalive-and-chunk-timeout-20260216.md) | [Review](implemented/20260216_review_ipc-stream-keepalive-and-chunk-timeout_approved-20260216.md)
- **is_alive() が IPC 接続死を検出できない** (2026-02-16) — 死活監視の精度向上
  [設計](implemented/20260216_is-alive-ipc-death-detection_implemented-20260216.md)
- **PIDファイル消失時のサーバー停止・起動耐性** (2026-02-17) — PIDファイルの堅牢性向上
  [Review](implemented/20260217_review_pid-file-resilience_approved-20260217.md)
- **WebSocket接続安定性の包括的改善** (2026-02-16) — WebSocket通信の信頼性向上
  [Review](implemented/20260216_review_websocket-stability-improvements_approved-20260216.md)

---

## 13. 人間通知

call_human統合、組織構成プロンプト注入等。

- **トップレベルAnimaの人間通知機能 — A1メッセージング統合** (2026-02-15) — supervisorなしAnimaからの人間通知
  [設計](implemented/20260215_a1-messaging-integration_human-notification_implemented-20260215.md) | [Review](implemented/20260215_review_a1-messaging-integration_human-notification_approved-20260215.md)
- **Notify Human のチャットウィンドウ統合確認** (2026-02-15) — 人間通知のチャットUI統合
  [設計](implemented/20260215_notify-human-chat-window-integration-implemented-20260216.md) | [Review](implemented/20260216_review_notify-human-chat-window-integration_approved-20260216.md)
- **組織構造情報のシステムプロンプト注入** (2026-02-15) — supervisor/subordinates/peersの自動注入
  [設計](implemented/20260215_org-structure-prompt-injection_implemented-20260215.md) | [Review](implemented/20260215_review_org-structure-prompt-injection_approved-20260215.md)
- **call_human統一: 人間への連絡を単一関数に集約** (2026-02-17) — Slack/Chatwork/LINE/Telegram/ntfyの統合
  [設計](implemented/20260217_unify-call-human-notification_implemented-20260217.md) | [Review](implemented/20260217_review_unify-call-human-notification_approved-20260217.md)

---

## 14. セットアップ・オンボーディング

セットアップウィザード、i18n、自動起動等。

- **Setup完了後にAnimaが起動しない（RAG初期化タイムアウト）** (2026-02-15) — 初期化フローの修正
  [設計](implemented/20260215_person-startup-timeout-after-setup_implemented-20260215.md)
- **セットアップ画面の言語選択を17言語対応ドロップダウンに拡張** (2026-02-15) — 多言語対応の強化
  [設計](implemented/20260215_setup-language-selector-expansion_implemented-20260215.md)
- **セットアップウィザード: ユーザー情報ステップ追加 & Anima自動起動** (2026-02-15) — 初期設定フローの拡充
  [設計](implemented/20260215_setup-user-info-and-person-autostart-20260215.md) | [Review](implemented/20260215_review_setup-user-info-and-person-autostart_approved-20260215.md)
- **セットアップウィザード: i18n反映不備 & ブラウザキャッシュ問題** (2026-02-15) — 国際化と表示の修正
  [設計](implemented/20260215_setup-wizard-i18n-and-cache-fix-20260215.md) | [Review](implemented/20260215_review_setup-wizard-i18n-and-cache-fix_approved-20260215.md)
- **セットアップウィザード: キャラクター作成ステップの簡素化** (2026-02-15) — リーダー作成ステップへの変更
  [設計](implemented/20260215_setup-wizard-simplify-character-step_implemented-20260215.md)
- **GUIセットアップウィザード — 初回起動時のWeb UIベース初期設定** (2026-02-15) — Web UIによる初期設定フロー
  [設計](implemented/20260215_setup-wizard_implemented-20260215.md) | [Review](implemented/20260215_review_setup-wizard_approved-20260215.md)

---

## 15. ログ・可観測性

ロギング強化、フロントエンドログ配信等。

- **ログ基盤の包括的強化** (2026-02-17) — フロントエンドサーバー送信 + structlog + バックエンドトレーサビリティ
  [設計](implemented/20260217_comprehensive-logging-enhancement_implemented-20260217.md) | [Review](implemented/20260217_review_comprehensive-logging-enhancement_approved-20260217.md)
- **フロントエンドログがサーバーに到達しない問題の修正** (2026-02-17) — ログ配信パイプラインの修正
  [設計](implemented/20260217_fix-frontend-log-delivery_implemented-20260217.md) | [Review](implemented/20260217_review_fix-frontend-log-delivery_approved-20260217.md)

---

## 16. テスト・品質

テスト修正、500エラー根本原因等。

- **500サーバーエラーの根本原因と包括的エラーハンドリング改善** (2026-02-15) — app.state.animasのKeyError修正
  [設計](implemented/20260215_500-server-error-root-cause-implemented-20260216.md) | [Review](implemented/20260216_review_500-server-error-root-cause_approved-20260216.md)
- **Phase 3 ProcessSupervisor リファクタリングに伴うテスト修正** (2026-02-15) — Supervisor移行後のテスト適合
  [設計](implemented/20260215_fix-failing-unit-tests-after-supervisor-refactor_implemented-20260215.md) | [Review](implemented/20260215_review_fix-failing-unit-tests-after-supervisor-refactor_approved-20260215.md)
- **残存テスト失敗23件の修正** (2026-02-15) — テストスイートの安定化
  [設計](implemented/20260215_fix-remaining-23-test-failures_implemented-20260215.md) | [Review](implemented/20260215_review_fix-remaining-23-test-failures_approved-20260215.md)
- **Fix app.state.animas KeyError (500 Server Error)** (2026-02-15) — サーバーステート初期化の修正
  [Review](implemented/20260215_review_fix-state-persons-keyerror_approved-20260215.md)
- **テストスイート 135件の失敗・エラー修正** (2026-02-17) — 大規模テスト修正
  [設計](implemented/20260217_test-suite-135-failures-cleanup.md)

---

## 17. セキュリティ

記憶書き込みセキュリティ、ライセンス等。

- **ライセンス戦略設計書** (2026-02-14) — Apache-2.0 ライセンス戦略
  [設計](implemented/20260214_licensing-strategy_design.md)
- **メモリ書き込みセキュリティ: 全実行モードの保護ファイル・パストラバーサル対策** (2026-02-15) — 記憶書き込みのセキュリティ強化
  [設計](implemented/20260215_memory-write-security-20260216.md) | [Review](implemented/20260216_review_memory-write-security_approved-20260216.md)

---

## 18. Anima生成

ハイブリッド作成、動的プロンプト注入等。

- **Anima作成ハイブリッド化: create_animaツール + キャラクターシート仕様統一** (2026-02-16) — ツール呼出し + MDシートの統一作成方式
  [設計](implemented/20260216_person-creation-hybrid-and-create-tool_implemented-20260216.md) | [Review](implemented/20260216_review_person-creation-hybrid-and-create-tool_approved-20260216.md)
