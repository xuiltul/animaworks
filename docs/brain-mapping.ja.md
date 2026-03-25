# AnimaWorks Brain Mapping — 人間の脳とのアーキテクチャ対応

**[English version](brain-mapping.md)**

> 作成日: 2026-02-19 | 更新日: 2026-03-25
> 関連: [vision.ja.md](vision.ja.md), [memory.ja.md](memory.ja.md)

---

## 背景

AnimaWorksの設計者は精神科医であり、30年以上のプログラミング経験を持つエンジニアでもある。AnimaWorksの記憶システム・自律機構・実行アーキテクチャは、神経科学の臨床的知見に基づいて**意図的に**人間の脳の構造をマッピングしている。これは単なるメタファーではなく、脳の情報処理アーキテクチャを設計パターンとして再利用する試みである。

精神科の臨床では、記憶障害・注意障害・実行機能障害など、脳の各サブシステムの機能不全を日常的に観察する。どのサブシステムが欠けると何が起きるかを知っているからこそ、AIエージェントに必要なサブシステムを同定し、それぞれの役割を明確に分離する設計が可能になった。

---

## 全体マッピング

### 大脳新皮質 — LLMモデル

| LLMの機能 | 脳領域 | 説明 |
|---|---|---|
| 推論・意思決定 | 前頭前野（PFC） | 実行機能。プライミングで注入された記憶を受け取り、判断する |
| 言語理解 | ウェルニッケ野（側頭葉） | 入力メッセージの意味理解 |
| 言語生成 | ブローカ野（前頭葉） | 応答テキストの産出 |
| 事前学習済み知識 | 側頭皮質の結晶化パターン | LLMの重みに焼き込まれた世界知識。ファイルベース記憶とは別系統の「生まれ持った知性」 |
| Transformer Attention | 頭頂連合野 + PFCの選択的注意 | コンテキスト内の関連情報への注意配分 |

LLMはその能力全体としては**大脳新皮質（neocortex）全体**に相当する。ただしAnimaWorksの設計では、フレームワークが皮質下構造（記憶の固定化・忘却・覚醒維持）を代行するため、LLMに残された役割は実質的に**前頭前野（PFC）の意識的処理**に純化されている。

memory.ja.mdが述べる通り：

> エージェント（LLM）は「考える人」であり、「自分の脳の管理者」ではない。

### 事前学習知識とファイルベース記憶の二重性

LLMの事前学習済み重みに焼き込まれた知識と、AnimaWorksのファイルベース記憶は**別系統**である。人間の脳でも、大脳皮質に結晶化されたパターン（暗黙知・結晶性知能）と海馬経由のエピソード記憶は独立した系統として機能する。

| 知識の種類 | 人間の脳 | AnimaWorks |
|---|---|---|
| 生まれ持った知性 | 結晶性知能（大脳皮質のパターン） | LLMの事前学習済み重み |
| 経験で獲得した知識 | 流動性知能 + エピソード記憶 | ファイルベース記憶（episodes/, knowledge/, procedures/） |

この区別はvision.ja.mdの「不完全な個」の設計思想と整合する。事前学習知識だけでは足りないからこそ、経験に基づく記憶システムが必要になる。

---

### 記憶システム — 海馬・大脳皮質・基底核

| 人間の記憶 | 脳領域 | AnimaWorks実装 | 特性 |
|---|---|---|---|
| **ワーキングメモリ** | 前頭前皮質 | LLMコンテキストウィンドウ | 容量制限あり。「今考えていること」の一時保持 |
| **エピソード記憶** | 海馬 → 新皮質 | `episodes/` | 「いつ何があったか」の時系列記録 |
| **意味記憶** | 側頭葉皮質 | `knowledge/` | 文脈から切り離された教訓・知識 |
| **手続き記憶** | 基底核・小脳 | `procedures/`, `skills/` | 「どうやるか」。繰り返しで強化される |
| **対人記憶** | 紡錘状回・側頭極 | `shared/users/` | 「この人は誰か」の自動想起 |

### ワーキングメモリの内部構造 — Baddeleyモデル

Baddeley (2000) のワーキングメモリモデルに基づく：

| Baddeleyの構成要素 | 機能 | AnimaWorks実装 |
|---|---|---|
| **中央実行系** | 注意制御・長期記憶からの取得統括 | エージェントオーケストレーター |
| **エピソードバッファ** | 複数ソースの統一表象への統合 | コンテキスト組立層（プライミング結果 + 会話履歴） |
| **音韻ループ** | 言語情報の一時保持 | テキストバッファ（直近の会話ターン） |

Cowan (2005) の知見に従い、ワーキングメモリを「活性化された長期記憶のスポットライト」として捉える。コンテキストウィンドウは独立したストアではなく、長期記憶のうち現在注意が向いている部分である。

---

### 記憶の想起 — 二重経路

| 想起経路 | 脳のプロセス | AnimaWorks実装 |
|---|---|---|
| **自動想起** | 海馬CA3の自己連合ネットワークによるパターン補完。無意識的・高速（250-500ms）・抑制不可 | プライミングレイヤー（6チャネル並列検索） |
| **意図的想起** | 前頭前皮質（PFC）による戦略的検索。意識的・遅い | `search_memory` / `read_memory_file` ツール |

### 拡散活性化 — Collins & Loftus (1975)

| 検索信号 | 脳の対応 | AnimaWorks実装 |
|---|---|---|
| 意味的近傍の発見 | 概念ノード間の拡散活性化 | 密ベクトル類似度検索（ChromaDB） |
| 最近の記憶の優先 | 近時効果 | 時間減衰関数（半減期30日） |
| 頻用記憶の強化 | ヘブの法則・長期増強（LTP） | アクセス頻度ブースト |
| 多ホップ連想 | 連想ネットワークの伝播 | ナレッジグラフ + Personalized PageRank（暗黙リンクのベクトル類似度しきい値 0.75。手続き蒸留の RAG 重複検出 0.85 は別経路） |

### プライミングチャネルと動的バジェット — 選択的注意

`PrimingEngine`（`core/memory/priming/engine.py`）は `asyncio.gather` で複数ソースを**並列**に取り込む。概念上は従来どおり **A〜F の6チャネル**に対応するが、実装では次が加わる：

- **C0（[IMPORTANT] 知識）**（`channel_c.py`）: RAG の `get_important_chunks` で `[IMPORTANT]` チャンクを常時取得し、見出し要約と `read_memory_file(path=...)` へのポインタだけを先頭に載せる（専用枠 500 トークン相当）。その後ろにベクトル検索による関連知識が続く。
- **直近アウトバウンド**（`priming/outbound.py`）: 直近2時間の `channel_post` / `message_sent` を最大3件。
- **保留中人間通知**（`priming/outbound.py` の `collect_pending_human_notifications`）: 過去24時間の `human_notify` を最大約500トークン相当で整形。ログ読取は `channel` が `chat` / `heartbeat` のとき、または `channel.startswith("message:")` のとき（Anima 間 DM 系プライミング）。**`build_system_prompt`（`core/prompt/builder.py`）の Group 3** では `pending_human_notifications` を載せるのは **`is_chat or is_heartbeat` のみ**（`is_heartbeat` は **`trigger == "heartbeat"` の厳密一致**）。`inbox:` / `cron:` / `task:` / `consolidation:` 等では `is_chat` が false のため **プロンプトに出ない**。`message:peer` の通常 DM は `is_chat` が true のため **出る**。なお `consolidation:` プライミングは `channel="heartbeat"` のため **収集は走りうる**が、上記 `is_heartbeat` 判定とずれて **注入はされない**ことがある。`cron` では収集側も空。

動的バジェットが有効なとき、全体上限 `token_budget` に対し `budget_ratio = token_budget / 2000` で各チャネルの文字枠が一様スケールする。関連知識は `related_knowledge`（trusted 側）と `related_knowledge_untrusted` を結合したうえで truncate し、信頼度別の分離を維持する。`config.json` の `priming.budget_*` で既定値を上書き可能。

| チャネル | 機能 | 脳の対応 | 基準トークン枠（`constants.py`、スケール前） |
|---|---|---|---|
| A: 送信者プロファイル | 「誰が話しかけているか？」 | 紡錘状顔領域・側頭極（人物認識） | 500 |
| B: 直近の活動 | 「最近何があったか？」 | 海馬リプレイ（直近エピソードの再活性化） | 1300 |
| C: 関連知識 | 「これについて何を知っているか？」（IMPORTANT 前置＋RAG） | 意味記憶の検索（側頭葉皮質） | IMPORTANT 500 + 関連検索 1000 |
| D: スキルマッチング | 「これを処理できるか？」 | 手続き記憶の活性化（基底核） | 200 |
| E: 未完了タスク | 「何をすべきだったか？」 | 展望的記憶・意図モニタリング（吻側前頭前野） | 500 |
| F: エピソード | 「過去に似た経験は？」 | エピソード記憶の意味検索（海馬-皮質） | 800 |

チャネルDはマッチしたスキル・共通スキル・手続きの**名前のみ**を返す（`channel_d.py` の `channel_d_skill_match`、`core.memory.manager.match_skills_by_description` の 3 段階マッチ、最大 `_MAX_SKILL_MATCHES` = 5）。全文は `skill` ツールでオンデマンド取得。チャネルBは `activity_log/` と `shared/channels/` を読む。ノイズ除去は `channel_b.py` の集合で分岐する。**heartbeat**（`channel == "heartbeat"`）では `tool_use` / `tool_result` / `heartbeat_start` / `heartbeat_reflection` / `inbox_processing_start` / `inbox_processing_end` を除外する。**バックグラウンド扱い**は docstring 上 `channel.startswith("cron:")` も含むが、**`PrimingMixin`（`core/_agent_priming.py`）は cron 実行時も `prime_memories(..., channel="cron")` と固定文字列を渡す**ため、`startswith("cron:")` に**一致しない**。その結果 **cron プライミングでもチャット用の `_CHAT_NOISE_TYPES` が適用**され、`memory_write` / `cron_executed` / `heartbeat_end` 等も除外対象になる（`channel` 引数を `cron:{task}` のように渡せば `channel_b` 側は heartbeat 相当フィルタに切り替わる設計）。

builder は ActivityLogger を直接読まず、プライミング経路（チャネルB・直近アウトバウンド・保留通知）がプロンプト用のアクティビティ読取の中心となる（海馬モデル）。

チャネルE（`channel_e.py`、未完了タスク＋周辺状態）は**展望的記憶**に対応する。吻側前頭前皮質（ブロードマン10野）は、適切な文脈がトリガーとなるまで将来の意図を低活性状態で維持する。AnimaWorksのタスクキューがエージェントの意識に未完了タスクを浮上させるメカニズムと同型である。実装上は `TaskQueueManager.format_for_priming` に加え、**実行中の並列タスク**（`PrimingEngine._get_active_parallel_tasks`、`submit_tasks` DAG）、**`state/overflow_inbox/`** の要約、**`state/task_results/`** にある完了済みバックグラウンドタスクの抜粋が同チャネルに連結される（`asyncio.to_thread` でキュー I/O を非ブロッキング化）。

#### 動的バジェット配分 — 注意資源の管理

`priming.dynamic_budget = true` の場合、プライミングのトークンバジェットはメッセージタイプに応じて動的に調整される。これはシステムレベルでの**選択的注意**の実装である：

| メッセージタイプ | バジェット（既定・`PrimingConfig`） | 脳のアナロジー |
|---|---|---|
| 挨拶 | 500 | 低注意負荷（定型的な社会的交流） |
| 質問 | 2000 | 中〜高の注意負荷（検索指向） |
| 要求 | 3000 | 高注意負荷（タスク指向、最大リソース配分） |
| ハートビート | max(200, context_window * 5%) | トニック覚醒（最低限の覚醒維持） |

ハートビートバジェットの算出式 `max(budget_heartbeat, int(context_window * heartbeat_context_pct))` により、コンテキストウィンドウが大きいモデルは自律巡回時により多くのプライミングデータを受け取る。これは、網様体賦活系のトニック発火率が皮質容量全体に比例してスケールすることに対応する。

`classify_message_type`（`priming/budget.py`）は **`channel == "heartbeat"` のときだけ** `"heartbeat"` 型とみなす。**cron プライミング**では `channel="cron"` のため上記に該当せず、挨拶 / 質問 / 要求のヒューリスティックに従い **greeting / question / request のいずれかのバジェット**が使われる（ハートビート用の % スケールは掛からない）。

この動的バジェット配分は、Kahneman (1973) の注意資源理論を反映している。高負荷タスクにはより多くの認知リソースを配分し、定型的刺激には少なく配分することで、限られたコンテキストウィンドウ内のS/N比を最適化する。

#### 段階的プロンプトとトリガーベースフィルタリング

コンテキストウィンドウのサイズに応じて、`build_system_prompt()` は4段階（T1〜T4）で注入セクションを調整する。128k以上で全セクション、32k〜128kで縮小、16k〜32kでbootstrap/vision/specialty/DK/memory_guideを省略、16k未満でさらにpermissions/org/messaging/emotionを省略する。これは**注意資源の限界に応じた選択的取捨**の実装である。

加えて、トリガー（`chat` / `inbox` / `heartbeat` / `cron` / `task` 等）に応じたセクション選択が行われる。heartbeatとcronはspecialty・emotion・a_reflectionを省略し、taskはidentity 3行とタスク記述のみの最小コンテキストで実行する。実行パスごとに「何を意識に上らせるか」を制御することで、認知負荷を最適化する。プライミング本文とは別に、未処理の `human_notify` 要約は `pending_human_notifications` として Group 3 に注入される条件は **上記 `is_chat or is_heartbeat` と同じ**（詳細は本節バレット参照）。

#### 統一アクティビティログ

チャネルBの主データソースとして、`ActivityLogger`（`core/memory/activity.py`）が `{anima_dir}/activity_log/{date}.jsonl` に時系列記録する。クラス本体は **ファサード**で、タイムライン・会話ビュー・プライミング整形・ログローテーションは `_activity_timeline` / `_activity_conversation` / `_activity_priming` / `_activity_rotation` 等のミックスインに分割されている。代表例：`message_received` / `message_sent`（`dm_*` エイリアスと相互解決）、`response_sent`、`channel_read` / `channel_post`、`human_notify`、`tool_use` / `tool_result`、`heartbeat_start` / `heartbeat_end` / `heartbeat_reflection`、`inbox_processing_start` / `inbox_processing_end`、`cron_executed`、`memory_write`、`error`、`issue_resolved`、`task_created` / `task_updated` など。UI のライブ更新用に一部ツールイベントが WebSocket 配信対象になる（`_LIVE_EVENT_TYPES` / `_VISIBLE_TOOL_NAMES` に一致する `tool_use`）。ストリーミング出力のクラッシュ耐性は `core/memory/streaming_journal.py` 経由の `shortterm/streaming_journal_{session_type}.jsonl`（chat / heartbeat 別）の Write-Ahead Log で実現する。日次の統合ログ掃除・保持期限付き削除はスーパーバイザ経由の **`core/memory/housekeeping.py`**（プロンプトログ、ショートターム、`task_results` 等）が担う。

---

### 記憶の固定化 — 睡眠と統合

| AnimaWorks | 脳のプロセス | 説明 |
|---|---|---|
| **即時符号化**（セッション境界） | 海馬の高速1ショット符号化 | 会話終了時に差分要約をepisodes/に記録 |
| **日次固定化**（深夜cron） | NREM睡眠の徐波-紡錘波-リップルカスケード | 本質的な要約・抽出は Anima のツールループが実行。`ConsolidationEngine`（`core/memory/consolidation.py`）は **前処理**（エピソード収集、`issue_resolved` 収集）と **後処理**（RAG インデックス更新・再構築、月次忘却の呼び出し、レガシー知識マイグレーション等）のヘルパに特化したモジュールである |
| **issue_resolved → procedure** | 解決の手続き化 | activity_log の `issue_resolved` イベントをスキャンし、ProceduralDistiller で手順書を生成（`create_procedures_from_resolved`） |
| **週次統合** | 新皮質の長期統合 | knowledge/の重複排除・マージ、パターン蒸留 |
| **NLI+LLMバリデーション** | 海馬のパターン分離 | ハルシネーション排除。エピソードと抽出知識の整合性検証 |
| **予測誤差ベース再固定化**（`reconsolidation.py`） | Nader et al. (2000) の再固定化理論 | 失敗カウントが閾値超の手続きをLLMで改訂。version管理とアーカイブ |

---

### 忘却 — シナプスホメオスタシス

Tononi & Cirelli (2003) のシナプスホメオスタシス仮説に基づく：

| AnimaWorks | 脳のプロセス | 説明 |
|---|---|---|
| **日次ダウンスケーリング** | NREM睡眠のシナプスダウンスケーリング | 低活性チャンクのマーク |
| **神経新生的再編** | 海馬歯状回の神経新生による記憶回路再編 | 低活性+類似チャンクのLLM統合 |
| **完全忘却**（月次） | 閾値以下のシナプス消失 | 知識などは低活性 **90日** 超・`access_count` が閾値以下でアーカイブ→削除（`FORGETTING_LOW_ACTIVATION_DAYS` / `FORGETTING_MAX_ACCESS_COUNT`）。**手続き**のダウンスケール判定は別閾値（例: **180日** 非使用かつ利用回数が少ない、等）を `PROCEDURE_INACTIVITY_DAYS` 等で用いる。手続きアーカイブは `PROCEDURE_ARCHIVE_KEEP_VERSIONS`（**5**版）のみ保持 |
| **忘却耐性**（procedures, skills, knowledge） | 基底核の手続き記憶は忘却に強い | procedures: `_is_protected_procedure` により `importance == important` / `protected: true` / `version >= 3`。低ユーティリティ・高失敗は `_should_downscale_procedure`（`PROCEDURE_*` 定数）。knowledge: `_is_protected_knowledge` により `importance == important` または `success_count >= 2`。`IMPORTANT_SAFETY_NET_DAYS`（**365日**）は `_is_protected` 先頭で `importance == important` の**第一段**保護の解除に使われるが、**knowledge / procedures は型別チェックが引き続き `[IMPORTANT]` を保護**する。skills / `shared_users` は `PROTECTED_MEMORY_TYPES` として完全スキップ |

### 手続き的蒸留とメタ可塑性

3段階の忘却サイクルに加え、AnimaWorksは神経可塑性のより精緻な側面に対応する追加の記憶サブシステムを実装している：

| AnimaWorks | 脳のプロセス | 説明 |
|---|---|---|
| **手続き的蒸留**（`distillation.py`） | 基底核-小脳回路でのスキル固定化 | エピソード記憶をLLMで知識・手続きに分類。活動ログから反復的な行動パターンを検出し、再利用可能な手続きファイルに蒸留する。反復的な運動シーケンスが基底核ループの統合を通じて自動化される過程に類似 |
| **週次パターン検出** | メタ可塑性（Abraham & Bear, 1996） | 活動ログのクラスタリングが7日間ウィンドウで反復的な行動パターンを同定する。「学び方を学ぶ」メタ可塑性を表現し、記憶内容だけでなく記憶形成プロセス自体を適応させる |
| **RAG重複検出**（類似度 >= 0.85） | 海馬のパターン分離 | `distillation.py` の `RAG_DUPLICATE_THRESHOLD = 0.85`。新しい手続きを保存する前にベクトル類似度チェックで冗長な符号化を防止する。歯状回が類似記憶を直交化して区別するメカニズムに対応 |
| **解決追跡**（`resolution_tracker.py`） | 組織的長期記憶（交差記憶システム） | `shared/resolutions.jsonl` に Anima横断の共有解決ログを記録。「誰が何を解決したか」の組織的知識を実現。Wegner (1987) の交差記憶理論に対応 |
| **永続タスクキュー**（`task_queue.py`） | 展望的記憶・ワーキングメモリの拡張 | デッドライン追跡と停滞タスク検出を備えたappend-only JSONLタスクキュー。コンテキストウィンドウを超えてワーキングメモリを拡張する、中央実行系の外部メモ帳のような機能 |

手続き的蒸留のパイプラインは2つの時間スケールで動作する：

- **日次**: LLMがエピソードセクションを知識・手続き・スキップに分類し、YAMLフロントマター付き（信頼度スコア、成功/失敗カウント）の構造化された手続きファイルを出力
- **週次**: 活動ログエントリのベクトルベースクラスタリングが反復的行動パターンを検出し、汎化された手続きに蒸留

この二重時間スケールアーキテクチャはスキル習得の神経科学を反映している。初期の明示的学習（日次分類）が反復を通じて暗黙的な手続き知識（週次パターン蒸留）へ移行する。これはDoyon & Benali (2005) が記述した、海馬依存処理から基底核依存処理への同じ移行である。

---

### 覚醒・自律機構

| AnimaWorks | 脳領域 | 説明 |
|---|---|---|
| **Heartbeat**（定期巡回） | **網様体賦活系（ARAS）** | 覚醒状態の維持。意識の内容は指定せず、意識の前提条件を提供する。律動的に発火し、なければ休眠（昏睡）に陥る |
| **Cron**（定時タスク） | 視床下部の概日リズム（SCN） | 時刻に基づく定期的な行動トリガー。睡眠-覚醒サイクル、日次/週次/月次の生体リズム |
| **ProcessSupervisor** | 自律神経系 | プロセスの生死を管理。意識外で動作し、各Animaの起動・監視・再起動を担う |
| **Unix Domain Socket IPC** | 神経線維束（白質路） | Anima間プロセスの物理的な通信路 |
| **Messenger** | シナプス伝達 | メッセージの送受信。カプセル化された個の間をテキストで接続する |

#### Heartbeat = 網様体賦活系（ARAS）の詳細

上行性網様体賦活系（Ascending Reticular Activating System）は脳幹の網様体から視床を経由して大脳皮質全体に投射し、覚醒状態を維持する。その特性とAnimaWorksのheartbeatは以下の点で対応する：

| ARAS の特性 | Heartbeat の特性 |
|---|---|
| 覚醒状態を維持する（意識の内容は指定しない） | Animaを周期的に起動する（何を考えるかは heartbeat.md に委ねる） |
| 自動的・律動的に発火する | 設定された間隔で自動実行される |
| 機能停止すると昏睡に陥る | heartbeatがなければ外部刺激（メッセージ）がないと休眠する |
| 意識の前提条件であり、意識そのものではない | 自律行動の前提条件であり、判断そのものではない |
| 感覚入力で覚醒レベルが変動する | メッセージ受信で即座に起動する（heartbeat周期外でも） |

---

### 組織構造 — 社会脳

| AnimaWorks | 脳/心理学の対応 | 説明 |
|---|---|---|
| **supervisor-subordinate階層** | 社会的階層の神経基盤（PFC-扁桃体回路） | 指示と報告の流れ |
| **カプセル化（内部不可視）** | 心の理論（Theory of Mind） | 他者の内部状態は推測するしかない |
| **メッセージング通信** | 言語コミュニケーション | テキストのみで繋がる。共有メモリや直接参照はしない |
| **identity.md（人格）** | パーソナリティ（前頭前野-辺縁系の安定パターン） | 不変のベースライン。判断の基盤 |
| **injection.md（役割）** | 社会的役割・職業的アイデンティティ | 可変。組織内での行動指針 |

### 実行モード — 自律性のレベル

記憶サブシステムはモード非依存だが、皮質（LLM）の実行エンジン選びは自律性の度合いを変える。現行コードでは **6 モード**（`resolve_execution_mode()`）を区別する：

| モード | エグゼキュータ | 脳のアナロジー | 説明 |
|---|---|---|---|
| **S**（SDK） | Claude Agent SDK | 実行制御を伴う完全な皮質機能 | Claude ネイティブツールとセッション継続 |
| **C**（Codex） | Codex CLI | S に近い皮質機能 | Codex 経由で OpenAI Codex 系モデルを実行 |
| **D**（Cursor Agent） | Cursor Agent CLI | 外部エージェントループ | MCP 統合の別経路 |
| **G**（Gemini CLI） | Gemini CLI | 外部エージェントループ | stream-json とツールループ |
| **A**（Autonomous） | LiteLLM + tool_use ループ | 外部媒介による皮質機能 | マルチプロバイダのツール使用をフレームワークが管理 |
| **B**（Basic） | 1ショット（assisted） | 実行支援が大きい皮質機能 | 記憶 I/O 代行。セッションチェイニングは非対応寄り |

`models.json` 等のワイルドカードで自動判定され、`status.json` の `execution_mode` で上書き可能。S/C/D/G はツール＋継続セッション寄り、B はワーキングメモリ外部化が強い、という対応で本稿の記憶マッピングと整合する。

---

## 設計原則の神経科学的根拠

### なぜコンテキストウィンドウの制限は「機能」なのか

人間のワーキングメモリ容量は 4±1 チャンク（Cowan, 2001）に制限される。これは欠陥ではなく、**選択的注意を強制することで判断の質を担保する進化的適応**である。全ての記憶が同時に意識に上ると、関連情報の選別ができず、判断が劣化する。

AnimaWorksはこの原則を「設計上の特徴」として採用している。必要な情報だけをプライミングで想起し、すっきりしたコンテキストで判断させる。

### なぜ忘却が必要なのか

睡眠中のシナプスダウンスケーリングは、覚醒時に強化されたシナプスを全体的に弱め、信号対雑音比（S/N比）を維持する。忘却なしには記憶の蓄積がノイズとなり、検索精度が劣化する。

AnimaWorksの能動的忘却は、この生物学的メカニズムを再現し、ベクトル検索の精度を長期的に維持する。

### なぜ「不完全な個」の協働が全能の個体より堅牢なのか

人間の組織が機能するのは、各メンバーが限られた視野と記憶で判断し、不完全な情報を自分の言葉で伝え合うからである（vision.ja.md）。これは認知負荷理論（Sweller, 1988）と分散認知（Hutchins, 1995）の知見と整合する。

---

## まとめ

AnimaWorksは、精神科医の臨床的知見とエンジニアリング経験の交差点から生まれた設計である。脳の情報処理アーキテクチャは、生物学的基質（ニューロン）に依存しない**汎用的な設計パターン**として再利用可能であり、AnimaWorksはそれを実証するシステムである。

LLMを大脳新皮質、記憶システムを海馬-皮質系、heartbeatを網様体賦活系、忘却をシナプスホメオスタシスとしてマッピングし、これらが統合的に動作することで、「自律的に考え、学び、忘れ、協働する存在」を実現する。
