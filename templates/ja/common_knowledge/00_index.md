# Common Knowledge 目次・キーワード索引

AnimaWorks の全 Anima が共有するリファレンスドキュメントの目次。

困ったとき・手順が不明なときは、まずこのファイルを読んで該当ドキュメントを特定し、
`read_memory_file(path="common_knowledge/...")` で詳細を参照すること。

---

## 困ったときはまずここを読む

以下のフローで該当ドキュメントを見つける:

1. **メッセージの送り方がわからない**
   → `communication/messaging-guide.md` を読む

1.5. **Board（共有チャネル）の使い方がわからない**
   → `communication/board-guide.md` を読む

2. **指示の出し方・報告の仕方がわからない**
   → `communication/instruction-patterns.md` または `communication/reporting-guide.md` を読む

3. **組織構造・誰に連絡すべきかわからない**
   → `organization/structure.md` を読む

4. **ツールの使い方・呼び出し方がわからない**
   → `operations/tool-usage-overview.md` を読む

4.5. **ツールやコマンドが使えない / エラーが出る**
   → `troubleshooting/common-issues.md` を読む

5. **タスクがブロックされた / 判断に迷う**
   → `troubleshooting/escalation-flowchart.md` を読む

6. **ハートビートやcronの設定方法がわからない**
   → `operations/heartbeat-cron-guide.md` を読む

6.5. **長時間ツールの実行方法がわからない**
   → `operations/background-tasks.md` を読む

7. **タスク管理の方法がわからない**
   → `operations/task-management.md` を読む

7.5. **音声チャットの設定・使い方がわからない**
   → `operations/voice-chat-guide.md` を読む

7.6. **Mode S の認証方式（API直接 / Max plan / Bedrock）を変えたい**
   → `operations/mode-s-auth-guide.md` を読む

7.7. **モデルの選び方・変更方法がわからない / 実行モードの仕組みを知りたい**
   → `operations/model-guide.md` を読む

8. **メッセージ送信が制限された / 送信制限について知りたい**
   → `communication/sending-limits.md` を読む

9. **外部データの信頼性が気になる / プロンプトインジェクション対策**
   → `security/prompt-injection-awareness.md` を読む

10. **上記に該当しない**
   → `search_memory(query="キーワード", scope="common_knowledge")` で検索する

---

## ドキュメント一覧

### organization/ — 組織・構造

| ファイル | 概要 |
|---------|------|
| `organization/structure.md` | 組織構造の仕組み（supervisor による階層定義、上司・部下・同僚の決定方法） |
| `organization/roles.md` | 役割と責任範囲（トップレベル / 中間管理 / 実行 Anima の責務、speciality の意味） |
| `organization/hierarchy-rules.md` | 階層間のルール（通信経路、直属 vs 他部署、緊急時の例外） |

### communication/ — コミュニケーション

| ファイル | 概要 |
|---------|------|
| `communication/messaging-guide.md` | メッセージ送受信の完全ガイド（send_message のパラメータ、スレッド管理、送信制限、1ラウンドルール） |
| `communication/board-guide.md` | Board（共有チャネル）ガイド（post_channel / read_channel / read_dm_history の使い分け、投稿ルール） |
| `communication/instruction-patterns.md` | 指示の出し方パターン集（明確な指示の書き方、委任パターン、進捗確認） |
| `communication/reporting-guide.md` | 報告・エスカレーションの方法（報告タイミング、フォーマット、緊急 vs 定期） |
| `communication/sending-limits.md` | 送信制限の詳細（3層レート制限、30/h・100/day 上限、カスケード検出、対処法） |

### operations/ — 運用・タスク管理

| ファイル | 概要 |
|---------|------|
| `operations/project-setup.md` | プロジェクト設定方法（config.json 構造、Anima 追加、モデル設定、権限設定） |
| `operations/task-management.md` | タスク管理の方法（current_task.md / pending.md の使い方、状態遷移、優先順位） |
| `operations/heartbeat-cron-guide.md` | 定期実行の設定と運用（ハートビートの仕組み、cron タスク定義、自己更新） |
| `operations/tool-usage-overview.md` | ツール使用の概要（S/A/B モード別のツール体系、内部/外部/スーパーバイザーツール、呼び出し方法） |
| `operations/background-tasks.md` | バックグラウンドタスク実行ガイド（submit の使い方、判断基準、結果の受け取り方） |
| `operations/voice-chat-guide.md` | 音声チャットガイド（STT/TTS設定、WebSocketプロトコル、Per-Anima声設定、トラブルシューティング） |
| `operations/mode-s-auth-guide.md` | Mode S 認証モード設定ガイド（API直接/Bedrock/Vertex AI/Max plan の切り替え、Anima単位設定） |
| `operations/model-guide.md` | モデル選択・設定ガイド（実行モード、対応モデル一覧、models.json、credential設定、コンテキストウィンドウ） |

### security/ — セキュリティ

| ファイル | 概要 |
|---------|------|
| `security/prompt-injection-awareness.md` | プロンプトインジェクション防御ガイド（信頼レベル、境界タグ、untrusted データの処理ルール） |

### troubleshooting/ — トラブルシューティング

| ファイル | 概要 |
|---------|------|
| `troubleshooting/common-issues.md` | よくある問題と対処法（メッセージ不達、送信制限、ブロック、記憶検索、権限、ツール、コンテキスト） |
| `troubleshooting/escalation-flowchart.md` | 困ったときの判断フローチャート（問題分類、緊急度判定、エスカレーション先、テンプレート） |
| `troubleshooting/gmail-credential-setup.md` | Gmail Tool認証設定ガイド（token.json配置手順、pickle変換、client_id不一致問題） |

---

## キーワード索引

該当するキーワードから、参照すべきドキュメントを見つける。

| キーワード | 参照先 |
|-----------|--------|
| メッセージ, 送信, 返信, スレッド, inbox | `communication/messaging-guide.md` |
| send_message, reply_to, thread_id | `communication/messaging-guide.md` |
| Board, チャネル, 共有, general, ops | `communication/board-guide.md` |
| post_channel, read_channel, read_dm_history | `communication/board-guide.md` |
| DM履歴, やり取り, 過去の会話 | `communication/board-guide.md` |
| 指示, 委任, タスク依頼, デリゲーション | `communication/instruction-patterns.md` |
| 報告, 日報, サマリー, 完了報告 | `communication/reporting-guide.md` |
| エスカレーション, 相談, 仲介 | `communication/reporting-guide.md`, `troubleshooting/escalation-flowchart.md` |
| 組織, supervisor, 上司, 部下, 同僚 | `organization/structure.md` |
| 役割, 責任, speciality, 専門 | `organization/roles.md` |
| 階層, ルール, 権限, 通信経路 | `organization/hierarchy-rules.md` |
| 設定, config, ホットリロード, 設定反映 | `operations/project-setup.md` |
| モデル, models, 実行モード, モデル変更, モデル一覧, models.json, プロバイダ | `operations/model-guide.md` |
| モデル選択, コンテキストウィンドウ, credential, set-model, models list | `operations/model-guide.md` |
| Anima追加, テンプレート, identity | `operations/project-setup.md` |
| タスク, 進捗, ブロック, 優先順位 | `operations/task-management.md` |
| current_task, pending, 状態管理 | `operations/task-management.md` |
| ハートビート, heartbeat, 定期チェック | `operations/heartbeat-cron-guide.md` |
| cron, スケジュール, 定時タスク | `operations/heartbeat-cron-guide.md` |
| 問題, エラー, 困った, 動かない | `troubleshooting/common-issues.md` |
| Gmail, gmail_unread, token.json, OAuth, credential | `troubleshooting/gmail-credential-setup.md` |
| 権限, permission, アクセス拒否 | `troubleshooting/common-issues.md` |
| ツール, discover_tools, 見つからない | `troubleshooting/common-issues.md` |
| 記憶, search_memory, 検索, 見つからない | `troubleshooting/common-issues.md` |
| フローチャート, 判断, 迷い, どうすべき | `troubleshooting/escalation-flowchart.md` |
| 緊急, 至急, セキュリティ | `troubleshooting/escalation-flowchart.md` |
| コンテキスト, 上限, セッション継続 | `troubleshooting/common-issues.md` |
| バックグラウンド, submit, 長時間, ブロック, background | `operations/background-tasks.md` |
| animaworks-tool, 外部ツール, Bash, CLI | `operations/tool-usage-overview.md`, `operations/background-tasks.md` |
| MCP, mcp__aw__, Sモード, ツール呼び出し | `operations/tool-usage-overview.md` |
| skill, スキルツール, 手順書取得, procedures | `operations/tool-usage-overview.md` |
| 実行モード, S-mode, A-mode, B-mode | `operations/tool-usage-overview.md` |
| レート制限, 送信制限, 30通, 100通, outbound limit | `communication/sending-limits.md` |
| 1ラウンドルール, 往復, ループ, カスケード | `communication/messaging-guide.md`, `communication/sending-limits.md` |
| プロンプトインジェクション, trust, untrusted, 境界タグ | `security/prompt-injection-awareness.md` |
| delegate_task, タスク委譲, task_tracker | `operations/task-management.md`, `organization/hierarchy-rules.md` |
| add_task, タスクキュー, TaskQueue | `operations/task-management.md` |
| pending, TaskExec, 3パス, 実行パス | `operations/task-management.md`, `operations/heartbeat-cron-guide.md` |
| org_dashboard, ping_subordinate, スーパーバイザーツール | `organization/hierarchy-rules.md` |
| status.json, SSoT, reload, リロード | `operations/project-setup.md` |
| anima info, anima list, anima set-model, models list, models info | `operations/model-guide.md` |
| ブロックコマンド, 実行できないコマンド, blocked | `operations/project-setup.md`, `troubleshooting/common-issues.md` |
| ティア, tiered, プロンプト短縮, T1, T2, T3, T4 | `troubleshooting/common-issues.md` |
| 音声, voice, マイク, STT, TTS, 音声チャット | `operations/voice-chat-guide.md` |
| VOICEVOX, ElevenLabs, Style-BERT-VITS2, SBV2 | `operations/voice-chat-guide.md` |
| voice_id, 声, 話者, speaker, 音声設定 | `operations/voice-chat-guide.md` |
| WebSocket, /ws/voice, barge-in, VAD, PTT | `operations/voice-chat-guide.md` |
| Mode S, 認証, API直接, Max plan, Bedrock, Vertex AI, credential, _build_env | `operations/mode-s-auth-guide.md` |
| Agent SDK, 環境変数, ANTHROPIC_API_KEY, CLAUDE_CODE_USE_BEDROCK | `operations/mode-s-auth-guide.md` |

---

## 使い方

### 検索で見つける場合

```
search_memory(query="メッセージ 送信", scope="common_knowledge")
```

検索結果に該当ファイルのパスが表示されるので、`read_memory_file` で読む。

### パスを直接指定する場合

```
read_memory_file(path="common_knowledge/troubleshooting/common-issues.md")
```

### このファイル自体を参照する場合

```
read_memory_file(path="common_knowledge/00_index.md")
```
