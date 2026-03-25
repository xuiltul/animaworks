# AnimaWorks 記憶システム設計仕様書

**[English version](memory.md)**

> 作成日: 2026-02-14
> 更新日: 2026-03-25
> 関連: [vision.ja.md](vision.ja.md), [spec.md](spec.md), [specs/20260214_priming-layer_design.md](specs/20260214_priming-layer_design.md)


---

## 設計思想

AnimaWorksの記憶システムは**人間の脳の記憶メカニズム**に基づいて設計する。

人間の脳には「ワーキングメモリ」「エピソード記憶」「意味記憶」「手続き記憶」という異なる記憶システムがあり、それぞれが異なる脳領域で処理される。記憶の想起には「自動想起（プライミング）」と「意図的想起」の2経路があり、記憶の定着には「即時符号化」「睡眠時固定化」「長期統合」という3段階の自動プロセスがある。

AnimaWorksはこれらのメカニズムを忠実に再現する。エージェント（LLM）は「考える人」であり、「自分の脳の管理者」ではない。記憶インフラの管理はフレームワークが担い、符号化・固定化にはバックグラウンドで別途LLMをワンショット呼出しする（エージェント本人のLLMセッションとは独立）。

---

## 人間の記憶モデルとの対応

| 人間の記憶 | 脳領域 | AnimaWorks実装 | 特性 |
|---|---|---|---|
| **ワーキングメモリ** | 前頭前皮質 | LLMコンテキストウィンドウ | 容量制限あり。「今考えていること」の一時保持。活性化された長期記憶のスポットライト |
| **エピソード記憶** | 海馬 → 新皮質 | `episodes/` | 「いつ何があったか」。日次ログとして時系列に格納。会話終了時にフレームワークが自動記録 |
| **意味記憶** | 側頭葉皮質 | `knowledge/` | 「何を知っているか」。文脈から切り離された教訓・方針・知識。日次固定化でエピソードから抽出 |
| **手続き記憶** | 基底核・小脳 | `procedures/`, `skills/` | 「どうやるか」。作業手順、スキル、ワークフロー |
| **対人記憶** | 紡錘状回・側頭極 | `shared/users/` | 「この人は誰か」。Anima横断で共有するユーザープロファイル |

### ワーキングメモリ = コンテキストウィンドウ

Baddeley (2000) のワーキングメモリモデルに基づく。

- **中央実行系** = エージェントオーケストレーター。注意制御と長期記憶からの取得を統括
- **エピソードバッファ** = コンテキスト組立層。プライミング結果と会話履歴を統一的な表象に統合
- **音韻ループ** = テキストバッファ。直近の会話ターンを保持

Cowan (2005) の知見に従い、ワーキングメモリを「活性化された長期記憶」として捉える。コンテキストウィンドウは別個のストアではなく、長期記憶のうち現在注意が向いている部分である。

### 長期記憶 = ファイルベース書庫

記憶はプロンプトに切り詰めて注入するのではなく、ファイルシステム上の書庫に格納する（書庫型記憶）。記憶量に上限はない。コンテキストに入るのは「今必要なもの」だけ。

```
~/.animaworks/animas/{name}/
├── activity_log/    統一アクティビティログ（全インタラクションのJSONL時系列記録）
├── episodes/        エピソード記憶（日次ログ、行動記録）
├── knowledge/       意味記憶（学習済み知識、教訓、方針）
├── procedures/      手続き記憶（作業手順書）
├── skills/          スキル記憶（個人スキル）
├── shortterm/       短期記憶（セッション状態、ストリーミングジャーナル。chat/heartbeat分離）
└── state/           ワーキングメモリの永続部分（現在タスク、短期記憶）
```

---

## アーキテクチャ全体像

```
┌──────────────────────────────────────────────────────┐
│          ワーキングメモリ（前頭前皮質）                  │
│          = LLMコンテキストウィンドウ                     │
│                                                        │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐   │
│  │中央実行系    │  │エピソード  │  │音韻ループ    │   │
│  │=オーケスト   │  │バッファ    │  │=テキスト     │   │
│  │ レーター    │  │=コンテキスト│  │ バッファ     │   │
│  │             │  │ 組立層     │  │              │   │
│  └──────┬──────┘  └─────┬──────┘  └──────────────┘   │
│         │               │                              │
│    意図的検索      自動想起結果                          │
│    (search_memory)  (プライミング)                       │
└─────────┬──────────────┬───────────────────────────────┘
          │              │
    ┌─────┴──────┐  ┌───┴──────────────────┐
    │  前頭前皮質  │  │  プライミングレイヤー  │
    │  =意図的検索 │  │  =自動想起            │
    │  エージェント │  │  フレームワーク自動実行 │
    │  がツール呼出│  │                       │
    └─────┬──────┘  └───┬──────────────────┘
          │              │
          │    ┌─────────┴────────────────┐
          │    │  拡散活性化               │
          │    │  ベクトル類似度 + 時間減衰│
          │    │  → 関連記憶の自動活性化   │
          │    └─────────┬────────────────┘
          │              │
┌─────────┴──────────────┴───────────────────────────────┐
│                長期記憶（海馬 + 大脳皮質）                │
│                                                          │
│  ┌───────────────────────────────────────────────┐      │
│  │  統一アクティビティログ activity_log/           │      │
│  │  = 全インタラクションのJSONL時系列記録          │      │
│  │  Primingの"直近アクティビティ"ソース             │      │
│  └───────────────────────────────────────────────┘      │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐    │
│  │エピソード記憶│  │意味記憶    │  │手続き記憶      │    │
│  │episodes/   │  │knowledge/ │  │procedures/     │    │
│  │            │  │           │  │skills/         │    │
│  │日次ログ    │  │学習済知識  │  │手順書・スキル  │    │
│  │行動記録    │  │教訓・方針  │  │ワークフロー    │    │
│  └────────────┘  └────────────┘  └────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │  共有記憶 shared/                               │     │
│  │  users/           対人記憶（ユーザープロファイル）│     │
│  │  resolutions.jsonl 解決レジストリ（組織横断）    │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │  ストリーミングジャーナル shortterm/             │     │
│  │  = WAL（Write-Ahead Log）。クラッシュ耐性      │     │
│  │  ストリーミング出力中のテキストを逐次永続化      │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  ── 記憶固定化（Anima主導 + フレームワーク後処理） ──      │
│                                                          │
│  [即時] セッション境界検出 → 差分要約 → episodes/        │
│         + ステート自動更新 + 解決伝播                     │
│  [日次] 深夜cron → Anima.run_consolidation("daily")      │
│         (ツールで知識抽出・手続き作成・矛盾解決)           │
│         → Synaptic Downscaling + RAG再構築               │
│  [週次] 週次cron → Anima.run_consolidation("weekly")     │
│         → 神経新生的再編 + RAG再構築                      │
│  [月次] 月次cron → 完全忘却 + アーカイブクリーンアップ     │
│                                                          │
│  ── 忘却（シナプスホメオスタシス） ──                     │
│                                                          │
│  [日次] Synaptic Downscaling: knowledge(90日)            │
│         + procedures(180日 or 効用低下) → 低活性マーク    │
│  [週次] 神経新生的再編: 低活性+類似チャンクのLLM統合      │
│  [月次] 完全忘却: 低活性90日超+access_count≤2 → アーカイブ削除 │
│         archive/forgotten/ へ移動 + archive/versions/ クリーンアップ   │
│                                                          │
│  ※ エージェントは意図的記銘（write_memory_file）のみ     │
└──────────────────────────────────────────────────────────┘
```

---

## 記憶の想起: 2つの経路

人間の記憶想起は単一のプロセスではなく、**自動想起**と**意図的想起**の2段階で構成される。AnimaWorksはこの両方を実装する。

### 自動想起 — プライミングレイヤー

**脳科学的基盤**: 知覚刺激が入力されると、海馬CA3領域の自己連合ネットワークが自動的にパターン補完を実行する。無意識的、高速（250-500ms）、抑制不可能。

**AnimaWorks実装**: メッセージを受信した時点で、フレームワークがエージェント起動前に関連記憶を自動検索し、コンテキストに注入する。エージェントにとって、記憶は「既に思い出している」状態で会話が始まる。

```
メッセージ受信 → コンテキスト抽出 → プライミング検索 → コンテキスト組立 → エージェント実行
                (送信者、キーワード)   (多ソース並列)     (トークン予算内)    (記憶が既にある)
```

プライミングは `core/memory/priming/` パッケージ（`engine.py` の `PrimingEngine`）で実装する。`prime_memories()` は `asyncio.gather` で **9 本のコルーチンを並列**実行する（送信者 A・直近活動 B・**C0: [IMPORTANT] 知識**・関連知識 C・スキル/手順 D・保留タスク E・**Recent Outbound**・エピソード F・**保留中人間通知**）。クラスコメントの「6-channel」は主系 A〜F を指すが、C0 とアウトバウンド系は別収集として同時に走る。取得後、`token_budget` に応じた比率で各ブロックを `truncate_*` し、注入文字列を組み立てる。

**動的バジェットの実際のスイッチ**は `PrimingConfig.dynamic_budget` ではなく、`prime_memories(..., enable_dynamic_budget=...)` 引数である（`config.json` の `priming.dynamic_budget` は現状スキーマ定義のみでエンジンからは参照されない）。チャット等の通常経路では `core/_agent_priming.py` が **`enable_dynamic_budget=True`** を渡す。`False` のとき全体上限は `_DEFAULT_MAX_PRIMING_TOKENS`（2000 相当）固定。

**主なデータソースと既定バジェット**（`priming/constants.py` の配分 × 動的バジェット比。全体上限は `dynamic_budget` 無効時 2000 トークン相当）:

| ソース | 対象 | 既定配分（トークン） | 方式 | 脳の対応 |
|---|---|---|---|---|
| **A: 送信者プロファイル** | shared/users/ | 500 | 完全一致ルックアップ | 顔を見た瞬間の自動想起 |
| **B: 直近アクティビティ** | activity_log/ + shared/channels/ | 1300（適用時 `max(400, 1300×比)`） | ActivityLogger + 共有チャネル | 短期〜近時記憶 |
| **C0: [IMPORTANT] 知識** | knowledge / shared_common_knowledge | 500（要約ポインタ） | `get_important_chunks()` | 情動顕著な記憶の常時想起 |
| **C: 関連知識** | knowledge + 共有 common_knowledge | 1000（trusted 優先、残りを untrusted） | 1〜3クエリのベクトル検索（`build_queries`）+ 信頼度分割 | 意味連想 |
| **D: スキル/手順マッチ** | skills/, procedures/, common_skills/ | 200 | description ベース3段階 + ベクトル | **名前のみ**（チャネル D 自体に HB/cron スキップなし。件数は後段でバジェット依存） |
| **E: 未完了タスク** | task_queue + 並列タスク | 500 | TaskQueueManager 等 | 「やるべきこと」 |
| **F: エピソード** | episodes/ | 800 | ベクトル検索（RAG）+ 任意でグラフ拡散 | 過去行動の意味的検索 |
| **Recent Outbound** | activity_log/ | 直近2時間・最大3件 | `channel_post`, `message_sent` | アウトバウンド行動の自己認識 |
| **保留中人間通知** | activity_log/ `human_notify` | 最大約500トークン | 直近24時間、`chat` / `heartbeat` / `message:*` のみ | 未処理の call_human 文脈 |

チャネルBは `ActivityLogger.recent(days=2, limit=100)` と `shared/channels/*.jsonl`（メンバーシップを `is_channel_member` で確認し、チャネルあたり最新5件＋24h以内の human 投稿・@メンションを追加）を統合する。共有チャネル由来エントリは時刻降順で **最大 15 件**に間引く（`_MAX_CHANNEL_ENTRIES`）。アクティビティログが空のときは `episodes/` + 旧チャネル読み取りにフォールバックする。

**ノイズフィルタリング（チャネルB）**はトリガーで異なる。

- **heartbeat**、または channel が `cron:` で始まる場合: `tool_use`, `tool_result`, `heartbeat_start`, `heartbeat_reflection`, `inbox_processing_start`, `inbox_processing_end` を除外（実行詳細・HB自己参照・Inboxライフサイクルを抑える）。`heartbeat_end` は含めない。
- **chat 等フォアグラウンド**: 上記に加え `memory_write`, `cron_executed`, `heartbeat_end`, `heartbeat_start`, `heartbeat_reflection`, `inbox_*` を除外し、メッセージ系・エラー・タスク系を優先する。

**優先度スコアリング**（`channel_b.prioritize_entries`）: 上位50件をスコア順に選び、時系列で並べ替えてから `format_for_priming` に渡す。

| 要因 | スコア | 説明 |
|---|---|---|
| 自身の送信・応答 | +15.0 | `message_sent`, `response_sent` |
| `message_received` | +15.0 | `meta.from_type != "anima"` のとき、または `origin_chain` に `"human"` を含むとき |
| 送信者との関連 | +10.0 | `from_person` / `to_person` が現在の送信者と一致 |
| キーワード | +3.0/個 | 本文・要約とキーワードの一致 |
| 直近性 | 可変 | 先頭エントリ基準の `elapsed_seconds / 600` |

**チャネルC（関連知識）の詳細**: 個人コレクションと `shared_common_knowledge` を `include_shared=True` でマージし、`config.json` の `rag.min_retrieval_score`（既定 0.3）で素のベクトル類似度を下回るヒットを落とす。各チャンクの `origin` を `resolve_trust` し、**medium** と **untrusted** に分割して文字列化する。注入時は trusted 側を `_BUDGET_RELATED_KNOWLEDGE` で頭から切り、残り予算で untrusted を切る。C0 で得た「[IMPORTANT] Knowledge（要約ポインタ）」ブロックは、この related テキストの**先頭に連結**される（同一バジェットプール内で優先表示）。

**バジェット管理**: `enable_dynamic_budget=True` のとき、`budget.adjust_token_budget()` が `classify_message_type()`（短文挨拶 / 質問 / 長文依頼 / heartbeat。Inbox の `intent` で上書きあり）と `context_window` から全体トークン上限を決め、各チャネル配分に `budget_ratio = token_budget / 2000` を掛ける。heartbeat 時は `max(budget_heartbeat, int(context_window × heartbeat_context_pct))`（いずれも `PrimingConfig` で上書き可）。チャネル B の整形は先に `format_for_priming(..., budget_tokens=1300)` で行い、最終段で `budget_activity = max(400, int(_BUDGET_RECENT_ACTIVITY × budget_ratio))` 文字相当に `truncate_tail` する。

チャネルDは `skills/`・共通 `common_skills/`・`procedures/` を対象に `match_skills_by_description()` で 3 段階マッチングする。`channel_d` では最大 **5 件**まで名前を返し、エンジン側でさらに `matched_skills[: max(1, budget_skills // 50)]` に切り詰める。**本文は返さずスキル名のみ**（詳細は `skill` ツール）。**heartbeat / cron 専用のスキップは実装されていない**（空メッセージでも `keywords` や状態由来の文脈があればマッチし得る）。

メッセージタイプ別の動的バジェット（`PrimingConfig` 既定。`config.json` で変更可）:

| メッセージタイプ | トークンバジェット | 用途 |
|---|---|---|
| greeting | 500 | 挨拶（短文、低負荷） |
| question | 2000 | 質問・中程度の記憶検索 |
| request | 3000 | 依頼・指示（広範な記憶検索） |
| heartbeat | 200（+ 大コンテキスト時は比例拡大） | 定期巡回。大きい `context_window` では `heartbeat_context_pct` 分まで拡張 |

### 意図的想起 — search_memory ツール

**脳科学的基盤**: 前頭前皮質（PFC）が自動想起の出力を監視し、不足する場合に戦略的検索を実行する。意識的で遅い。

**AnimaWorks実装**: プライミングで注入された記憶では不足する場合にのみ、エージェントが `search_memory` / `read_memory_file` ツールを呼び出して追加検索する。

意図的想起が必要な典型例:
- 具体的な日時・数値を正確に答える必要がある時
- 過去の特定のやり取りの詳細を確認したい時
- 手順書に従って作業する時
- コンテキストに該当する記憶がない未知のトピックの時

---

## 拡散活性化による記憶検索

**脳科学的基盤**: Collins & Loftus (1975) の拡散活性化理論。意味記憶は概念ノードが連想リンクで接続されたネットワークとして組織化される。あるノードが活性化されると、隣接ノードへ自動的に伝播する。「医者」の活性化が「看護師」「病院」を事前活性化する。

**AnimaWorks実装**: 密ベクトル検索、時間減衰、重要度ブースト、およびグラフベースの拡散活性化を組み合わせる。`config.rag.enable_spreading_activation` の既定は **True**（`core/config/schemas.py` の `RAGConfig`）。`MemoryRetriever.search(..., enable_spreading_activation=None)` は設定を読み、設定読み込み失敗時のみ拡散をオフにフォールバックする。適用対象は `spreading_memory_types`（既定 `knowledge`, `episodes`）。

初期設計ではBM25（キーワード）とベクトル検索のハイブリッドをRRFで統合する方針だったが、調査の結果、多言語対応の密ベクトル検索単体の方がキーワード検索よりも高精度であることが判明したため、**ベクトル類似度検索に一本化**した。

| 検索信号 | 方式 | 脳の対応 |
|---|---|---|
| **意味ベクトル** | 密ベクトル類似度検索（`intfloat/multilingual-e5-small`, 384次元, ChromaDB） | 概念的近傍の発見。拡散活性化の近似 |
| **時間減衰** | 指数減衰（半減期30日）+ `WEIGHT_RECENCY` 0.2 | 近時効果（`updated_at` 基準） |
| **アクセス頻度** | `min(WEIGHT_FREQUENCY × log1p(access_count), WEIGHT_FREQUENCY × FREQUENCY_LOG_CAP)` | 共有チャンクは per-anima キー `ac_{anima_name}` を優先。上限でスコア暴走を抑制 |
| **重要度** | メタデータ `importance == "important"`（[IMPORTANT]）に `+0.20` | 扁桃体モデルに相当する平坦ブースト |
| **グラフ拡散** | ナレッジグラフ + Personalized PageRank | 多ホップ連想。`max_graph_hops`（既定2）まで。明示 `[[link]]` + 暗黙（類似度≥閾値） |

**その他の検索挙動**（`MemoryRetriever`）:

- **superseded 除外**: `memory_type=="knowledge"` かつ `include_superseded=False`（既定）のとき、メタデータ `valid_until` が空でないチャンクをベクトル検索から除外する。
- **共有コレクション**: `include_shared=True` 時、`shared_common_knowledge`（およびスキル検索時は `shared_common_skills`）をマージする。

スコア調整後のベクトルヒットに対し、任意でグラフ拡散結果を追加する（拡散のみの結果は `min_score` フィルタの対象外）。

```
vector調整後スコア = vector_similarity
                     + WEIGHT_RECENCY × (0.5 ^ (age_days / 30))
                     + min(WEIGHT_FREQUENCY × log1p(access_count), WEIGHT_FREQUENCY × FREQUENCY_LOG_CAP)
                     + (importance=="important" ? WEIGHT_IMPORTANCE : 0)

WEIGHT_RECENCY = 0.2
WEIGHT_FREQUENCY = 0.1
FREQUENCY_LOG_CAP = 3.0   # 実効は最大 WEIGHT_FREQUENCY × 3.0
WEIGHT_IMPORTANCE = 0.20

# グラフ拡散で追加される近傍ノード:
spread_contribution = pagerank_score × 0.5
```

### グラフ拡散の実装フロー

> 実装: `core/memory/rag/graph.py` — `KnowledgeGraph`, `core/memory/rag/retriever.py` — `_apply_spreading_activation()`

ベクトル検索の結果を起点に、ナレッジグラフ上で Personalized PageRank を実行し、直接検索では見つからなかった関連記憶を活性化する。

```
ベクトル検索 → 初期結果
    │
    ▼
初期結果の doc_id をグラフノードにマッピング
    │
    ▼
Personalized PageRank（alpha=0.85、`graph.py` の `PAGERANK_*` 定数）
    │  起点ノードに personalization 重み
    │  エッジの "similarity" 属性を重みとして使用
    │
    ▼
上位5件の活性化隣接ノードを選出（`max_hops` は `rag.max_graph_hops`、既定2）
    │  初期結果のノードは除外
    │  スコア > 0.001 のノードのみ
    │
    ▼
活性化ノードのコンテンツを取得（ファイル読み込み or ベクトルストア）
    │
    ▼
score × 0.5 で最終結果に追加（activation: "spreading" タグ付き）
```

### ナレッジグラフの構造

| 要素 | 説明 |
|---|---|
| **ノード** | `knowledge/` と `episodes/` の各 `.md` ファイル。属性: `path`, `memory_type`, `stem` |
| **明示リンク** | Markdown 内の `[[filename]]` / `[[filename\|display]]` 記法。`similarity=1.0` |
| **暗黙リンク** | 各ノードの埋め込みに対し上位5件の類似ドキュメントをクエリし、類似度 ≥ 0.75 のペアをエッジとして追加。`similarity=score` |

グラフは `{anima_dir}/vectordb/knowledge_graph.json` にキャッシュされ、記憶ファイルの変更時に増分更新される。

### グラフ拡散・RAG の主な設定（`config.json` → `RAGConfig`）

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `enable_spreading_activation` | `true` | グラフ拡散の有効/無効 |
| `max_graph_hops` | `2` | グラフビルド・拡散の最大ホップ |
| `implicit_link_threshold` | `0.75` | 暗黙リンク生成の類似度閾値 |
| `spreading_memory_types` | `["knowledge", "episodes"]` | グラフに含める記憶タイプおよび拡散適用対象 |
| `min_retrieval_score` | `0.3` | プライミング等で渡す素のベクトル類似度下限（`None` で無効） |
| `skill_match_min_score` | `0.75` | スキルマッチのベクトル段階での閾値 |
| `enabled` | `true` | RAG 全体の有効フラグ |
| `embedding_model` | `intfloat/multilingual-e5-small` | 埋め込みモデル ID |
| `use_gpu` | `false` | 埋め込み推論の GPU 利用 |
| `enable_file_watcher` | `true` | 記憶ファイル監視（増分インデックス） |
| `graph_cache_enabled` | `true` | ナレッジグラフ JSON キャッシュ |

---

## YAMLフロントマター

`knowledge/` および `procedures/` のファイルにはYAMLフロントマターが付与され、記憶のメタデータを構造化して管理する。フロントマターは日次固定化パイプラインで自動付与されるほか、レガシーマイグレーションで既存ファイルにも遡及適用される。

### knowledge/ フロントマター

```yaml
---
created_at: "2026-02-18T03:00:00+09:00"
updated_at: "2026-02-18T03:00:00+09:00"
source_episodes: 3
confidence: 0.9
auto_consolidated: true
version: 1
---
```

| フィールド | 型 | 説明 |
|---|---|---|
| `created_at` | ISO8601 | 作成日時 |
| `updated_at` | ISO8601 | 最終更新日時 |
| `source_episodes` | int | 抽出元エピソード数 |
| `confidence` | float | 信頼度（NLI+LLMバリデーション結果）。0.0-1.0 |
| `auto_consolidated` | bool | 自動固定化で生成されたか |
| `version` | int | バージョン番号（再固定化のたびにインクリメント） |
| `superseded_by` | str | この知識を置き換えた新しいファイル（矛盾解決時） |
| `supersedes` | str | この知識が置き換えた古いファイル（矛盾解決時） |

### procedures/ フロントマター

```yaml
---
description: 手順の説明
confidence: 0.5
success_count: 0
failure_count: 0
version: 1
created_at: "2026-02-18T03:00:00+09:00"
updated_at: "2026-02-18T03:00:00+09:00"
auto_distilled: true
protected: false
---
```

| フィールド | 型 | 説明 |
|---|---|---|
| `description` | str | 手順の説明（スキルマッチングに使用） |
| `confidence` | float | 信頼度。success_count / max(1, success_count + failure_count) で算出 |
| `success_count` | int | 成功回数 |
| `failure_count` | int | 失敗回数 |
| `version` | int | バージョン番号（再固定化のたびにインクリメント） |
| `created_at` | ISO8601 | 作成日時 |
| `updated_at` | ISO8601 | 最終更新日時 |
| `auto_distilled` | bool | 自動蒸留で生成されたか |
| `protected` | bool | 手動保護指定（忘却からの保護） |

---

## 記憶の固定化: 3段階自動プロセス

人間の脳は、記憶の固定化を無意識の自動プロセスとして実行する。AnimaWorksでは**Anima主導の統合**と**フレームワーク後処理**の組み合わせで実現する。

- **Anima主導**: Animaが `run_consolidation()` でツール（search_memory, read_memory_file, write_memory_file, archive_memory_file）を駆使し、エピソード要約・知識抽出・矛盾解決・手続き作成を自律的に実行する
- **フレームワーク後処理**: シナプスダウンスケーリング（メタデータベース）、RAGインデックス再構築、月次忘却をフレームワークが自動実行する

```
覚醒時（会話中）                        睡眠時（非会話時）
────────────────                      ────────────────

 会話 → セッション境界検出              深夜cron
     │  (10分アイドル or heartbeat)       │
     ▼                                  ▼
 [即時符号化]                          [日次固定化]
 差分要約 → episodes/                  Anima.run_consolidation("daily")
 + ステート自動更新                     (ツール呼出しで知識抽出・手続き作成・矛盾解決)
 + 解決伝播                            → 後処理: Synaptic Downscaling
 海馬の1ショット記録                    → 後処理: RAGインデックス再構築
                                          │
                                     週次cron
                                          │
                                          ▼
                                     [週次統合]
                                     Anima.run_consolidation("weekly")
                                     → 後処理: 神経新生的再編
                                     → 後処理: RAGインデックス再構築
                                          │
                                     月次cron
                                          │
                                          ▼
                                     [月次忘却]
                                     ForgettingEngine.complete_forgetting()
                                     archive/versions/ クリーンアップ（旧手順バージョン）
```

### 日次固定化フロー

> 実装: `core/_anima_lifecycle.py` — `Anima.run_consolidation()`、`core/memory/consolidation.py` — `ConsolidationEngine`（前処理・後処理）
> スケジュール: `core/lifecycle/system_crons.py` が `core/lifecycle/system_consolidation.py` の日次ハンドラを登録（時刻は `ConsolidationConfig.daily_time`、既定 02:00 JST）

**1. 前処理**（ConsolidationEngine）: 以下の4種のデータを収集し、`consolidation_instruction` プロンプトに注入する:

| 収集データ | メソッド | 内容 |
|---|---|---|
| 直近エピソード | `_collect_recent_episodes(hours=24)` | 直近24時間の `episodes/` エントリ |
| 解決済みイベント | `_collect_resolved_events(hours=24)` | activity_log 内の `issue_resolved` イベント |
| アクティビティ要約 | `_collect_activity_entries(hours=24)` | 通信イベント + `tool_result`（約4000トークン上限） |
| 振り返り | `_extract_reflections_from_episodes()` | エピソード内の `[REFLECTION]...[/REFLECTION]` ブロック |

**2. Anima実行**: `consolidation_instruction` プロンプトに従い、Animaがツールを使って自律的に以下を実行する（`max_turns=30`）:

1. 今日のエピソードと解決済みイベントを確認
2. `search_memory` で関連する既存の knowledge/ と procedures/ を検索
3. `write_memory_file` で knowledge を更新・新規作成
4. 解決済みイベントから得られた教訓・手順を `procedures/` に記録
5. 重複・陳腐化した記憶を `archive_memory_file` でアーカイブ

**3. 後処理**: `ForgettingEngine.synaptic_downscaling()`（メタデータベースの低活性マーク）、`ConsolidationEngine._rebuild_rag_index()`

### 週次統合フロー

> スケジュール: `system_consolidation.py` の週次ハンドラ（`ConsolidationConfig.weekly_time`、既定 `sun:03:00` JST）

**1. Anima実行**: `run_consolidation("weekly")` で `weekly_consolidation_instruction` に従い、以下の4タスクを実行:

| タスク | 内容 |
|---|---|
| **knowledge 統合** | `knowledge/` を一覧し、`search_memory` で重複を検出。統合してアーカイブ |
| **procedure クリーンアップ** | 陳腐化・未使用の手順を更新またはアーカイブ |
| **episode 圧縮** | 30日超のエピソードをエッセンスに圧縮（`[IMPORTANT]` タグ付きは除外） |
| **矛盾解決** | 矛盾する knowledge を検出し、アーカイブまたは統合 |

**2. 後処理**: `ForgettingEngine.neurogenesis_reorganize()`（非同期）、RAGインデックス再構築

### 月次忘却パイプライン

> スケジュール: `system_consolidation.py` の月次忘却（`ConsolidationConfig.monthly_time`、既定 1日 04:00 JST）

- `ForgettingEngine.complete_forgetting()`（knowledge + episodes + procedures）
- `cleanup_procedure_archives()` — `archive/versions/` 内の旧手順スナップショットを整理（手順 stem あたり直近 5 件のみ保持、`PROCEDURE_ARCHIVE_KEEP_VERSIONS`）

### 固定化で使用されるモデル

日次・週次の consolidation は **バックグラウンドトリガー**（`consolidation:daily`, `consolidation:weekly`）として実行される。使用されるモデルの解決順序:

1. Per-anima `status.json` の `background_model`
2. `config.json` `heartbeat.default_model`
3. メインモデル（`model`）にフォールバック

`background_model` に軽量モデル（例: `claude-sonnet-4-6`）を設定することで、メインモデル（例: `claude-opus-4-6`）を chat に温存しつつ、consolidation のコストを最適化できる。後処理の `neurogenesis_reorganize`（週次 LLM マージ）に渡すモデルも同じ解決ロジックに従う。

### 固定化段階一覧

| 段階 | 脳のプロセス | AnimaWorks実装 | 担当 | 頻度 |
|---|---|---|---|---|
| **即時符号化** | 海馬の高速1ショット符号化 | セッション境界検出（10分アイドル or heartbeat）→ 差分要約 → episodes/ 自動記録 + ステート自動更新 + 解決伝播 | フレームワーク（bg LLM呼出） | セッション境界時 |
| **日次固定化** | NREM睡眠の徐波-紡錘波-リップル カスケード | 深夜cron → Anima.run_consolidation("daily")（ツールで知識抽出・手続き作成・矛盾解決）→ 後処理: Synaptic Downscaling + RAG再構築 | Anima + フレームワーク後処理 | 毎日深夜 |
| **週次統合** | 新皮質の長期統合・シナプスダウンスケーリング | 週次cron → Anima.run_consolidation("weekly") → 後処理: 神経新生的再編 + RAG再構築 | Anima + フレームワーク後処理 | 毎週 |
| **月次忘却** | 閾値以下のシナプス消失 | 月次cron → ForgettingEngine.complete_forgetting() + アーカイブクリーンアップ | フレームワーク（bg cron） | 毎月 |
| **意図的記銘** | 前頭前皮質の精緻化符号化 | write_memory_file で直接書き込み | エージェント | 随時 |

エージェントに残る唯一の書き込み経路は**意図的記銘**（write_memory_file）。これは人間が意識的にメモを取る行為に相当する。日次・週次の固定化・統合はAnimaがツールで自律実行し、シナプスダウンスケーリング・RAG再構築・月次忘却はフレームワークが自動で行う。

### 即時符号化の詳細: セッション境界ベースの差分要約

旧設計ではメッセージ応答のたびに全ターンを再要約していたが、これにより同一会話の要約がN-2回重複記録される問題があった。現設計では `last_finalized_turn_index` で記録済み位置を追跡し、**未記録ターンのみを差分要約**する。

**セッション境界**: メッセージ応答時ではなく、以下の2つの条件でのみ `finalize_session()` が実行される:
- **10分アイドル**: 最終ターンから10分経過時（`finalize_if_session_ended()` で検出）
- **heartbeat到達**: 定期巡回時に `finalize_if_session_ended()` を呼び出し

**統合ポイント**: `finalize_session()` は差分要約に加え、以下を同時実行する:
1. **エピソード記録**: 未記録ターンのLLM要約を `episodes/` に追記
2. **ステート自動更新**: LLM要約から「解決済みアイテム」「新規タスク」を自動パースし `state/current_state.md` に追記
3. **解決伝播**: 解決アイテムを ActivityLogger（`issue_resolved` イベント）と `shared/resolutions.jsonl` に記録
4. **ターン圧縮**: 記録済みターンを `compressed_summary` に統合し conversation.json の肥大化を防止

### 解決伝播メカニズム

解決情報は3層で伝播し、自Animaと他Animaの両方に反映される:

| 層 | 対象 | 実装 | 伝播先 |
|---|---|---|---|
| **層1: ActivityLogger** | 自Anima | `issue_resolved` イベントを activity_log に記録 | PrimingチャネルB（自Animaの直近アクティビティ） |
| **層2: 解決レジストリ** | 全Anima | `shared/resolutions.jsonl` に組織横断記録 | builder.py の「解決済み案件」セクション（全Animaのシステムプロンプト） |
| **層3: Consolidation注入** | 自Anima | `_collect_resolved_events()` で解決イベント収集 | 日次固定化プロンプトに注入（knowledge/ の「未解決」記載を「解決済み」に更新）|

---

## 記憶のバリデーション: NLI+LLMカスケード

> 実装: `core/memory/validation.py` — `KnowledgeValidator` クラス

LLMが抽出した knowledge 候補をそのまま書き込むとハルシネーション（元のエピソードに存在しない情報の捏造）が混入する可能性がある。**NLI（Natural Language Inference）モデルとLLMのカスケード検証**でこれを排除する。Anima主導の日次固定化では、Animaがツールで直接 knowledge/ に書き込むため、本パイプラインは別経路（例: バッチ処理、レガシーパイプライン）で使用される。

### NLIモデル

- モデル: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
- 多言語対応（日本語含む）のゼロショットNLI
- GPU利用可能時はGPU、不可時はCPUにフォールバック
- NLIモデルが利用不可の場合はLLMのみでバリデーション（グレースフルデグラデーション）

### カスケードフロー

```
knowledge候補（前提: 元エピソード、仮説: 抽出された知識）
    │
    ▼
[NLI判定]
    ├── entailment ≥ 0.6  → confidence=0.9 で承認（LLMスキップ）
    ├── contradiction ≥ 0.7 → 棄却（LLMスキップ）
    └── neutral / 閾値未満  → LLMレビューへ
                                  │
                                  ▼
                             [LLM判定]
                                  ├── 承認 → confidence=0.7 で書き込み
                                  └── 棄却 → 破棄
```

NLIで高確信の判定が出た場合はLLM呼出しをスキップすることで、コストとレイテンシを最適化する。NLIが neutral を返した曖昧なケースのみLLMの判断を仰ぐ。

---

## 知識矛盾検出・解決

> 実装: `core/memory/contradiction.py` — `ContradictionDetector` クラス

`knowledge/` に蓄積された知識ファイル間で矛盾が発生することがある（例: 「Aさんの担当はX」と「Aさんの担当はY」）。Anima主導の統合では、Animaが `consolidation_instruction` の指示に従いツールで矛盾を検出・解決する。`ContradictionDetector` はNLI+LLMカスケードによる自動検出・解決のユーティリティとして利用可能。

### 矛盾検出フロー

```
新規/更新された knowledge ファイル
    │
    ▼
[RAG検索] 類似 knowledge を取得
    │
    ▼
[NLI判定] ペアごとに entailment/contradiction/neutral を判定
    ├── entailment ≥ 0.7  → 矛盾なし（LLMスキップ、コスト最適化）
    ├── contradiction ≥ 0.7 → 矛盾検出 → LLM解決へ
    └── neutral / 閾値未満  → LLM詳細分析へ
                                  │
                                  ▼
                             [LLM分析]
                                  ├── 矛盾あり → 解決戦略決定
                                  └── 矛盾なし → スキップ
```

### 3つの解決戦略

| 戦略 | 条件 | 処理 |
|---|---|---|
| **supersede**（置換） | 新しい情報が古い情報を明確に更新 | 古いファイルに `superseded_by` を付与してアーカイブ、新ファイルに `supersedes` を記録 |
| **merge**（統合） | 両方の情報を統合可能 | LLMが統合テキストを生成し新ファイルを作成、元ファイル両方をアーカイブ |
| **coexist**（共存） | 文脈依存で両方が正しい | 両ファイルに矛盾の存在と条件をアノテーション |

### 実行タイミング

| タイミング | 対象 | 説明 |
|---|---|---|
| **日次** | 当日新規作成・更新されたファイル | 日次固定化パイプラインの最終ステップ |
| **週次** | 全 `knowledge/` ファイル | 日次で未検出の矛盾を網羅的にスキャン |

---

## 手続き記憶ライフサイクル

> 実装: `core/memory/distillation.py` — `ProceduralDistiller`, `core/memory/reconsolidation.py` — `ReconsolidationEngine`

手続き記憶（`procedures/`）は「どうやるか」を保持する記憶で、脳の基底核・小脳に対応する。意味記憶（knowledge/）が「何を知っているか」を静的に保持するのに対し、手続き記憶は繰り返しの実行と結果フィードバックにより動的に強化・修正される。

### 手続きの作成

**Anima主導**（`run_consolidation()` 内）:

`consolidation_instruction` プロンプトの指示に従い、Animaが `write_memory_file` で procedures/ に手順を直接作成・更新する。解決済みイベントから得られた教訓と手順もここで記録する。

**ReconsolidationEngine**（別経路）:

`create_procedures_from_resolved()` が `issue_resolved` イベントをスキャンし、`ProceduralDistiller` で手順書を生成する。メインの日次固定化フローでは呼ばれず、バッチ処理等で利用可能。

### 3段階マッチング（スキル注入）

プライミング（チャネルD）および `builder.py` のスキル注入で、メッセージに対して `procedures/` のマッチングを行う:

| 段階 | 方式 | 説明 |
|---|---|---|
| **1. ブラケットキーワード** | `[keyword]` 完全一致 | メッセージ中の `[keyword]` がフロントマターの `description` に含まれる場合にマッチ |
| **2. 語彙マッチ** | 内容語オーバーラップスコアリング | メッセージとdescriptionの内容語（名詞・動詞等）の重複度で順位付け |
| **3. RAGベクトル検索** | 密ベクトル類似度 | sentence-transformersによる意味的類似度検索 |

段階1が最優先で、段階3はフォールバック。これにより、明示的なキーワード指定から曖昧な意味的検索まで幅広く手順を想起できる。

### 成功/失敗追跡

手続き記憶の信頼度は実行結果のフィードバックで動的に更新される:

| 追跡方式 | 説明 |
|---|---|
| **report_procedure_outcome ツール** | エージェントがツール呼出しで明示的に成功/失敗を報告 |
| **フレームワーク自動追跡** | セッション中に注入された手順に対し、セッション境界で成否を自動判定 |

信頼度の算出:

```
confidence = success_count / max(1, success_count + failure_count)
```

初期値（自動蒸留時）: `confidence: 0.4`, `success_count: 0`, `failure_count: 0`

### 予測誤差ベースの再固定化

> 実装: `core/memory/reconsolidation.py` — `ReconsolidationEngine`

**脳科学的基盤**: Nader et al. (2000) の再固定化理論。想起された記憶は不安定化し、新しい情報と統合された後に再固定化される。予測誤差（期待と実際のギャップ）が再固定化のトリガーとなる。

**AnimaWorks実装**: Anima主導の統合では、Animaが `consolidation_instruction` の「既存知識との照合」「矛盾する知識があればアーカイブ」の指示に従いツールで実行する。`ReconsolidationEngine` はNLI+LLMによる自動再固定化のユーティリティとして別経路で利用可能。以下は ReconsolidationEngine の処理フロー:

```
新エピソード
    │
    ▼
[RAG検索] 関連する既存 knowledge/procedures を取得
    │
    ▼
[NLI判定] エピソードと既存記憶の矛盾検出
    ├── 矛盾なし → スキップ
    └── 矛盾あり → LLM分析
                      │
                      ▼
                 [LLM更新判断]
                      ├── 更新必要 → 旧バージョンを archive/versions/ に保存
                      │               → 記憶を更新、version++
                      └── 更新不要 → スキップ
```

**procedures/ 再固定化時の特別処理**:
- 旧バージョンを `archive/versions/` に保存（`ReconsolidationEngine._archive_version`）
- `version` をインクリメント
- `success_count: 0`, `failure_count: 0`, `confidence: 0.5` にリセット（再検証が必要なため）
- `updated_at` を更新

---

## 能動的忘却: シナプスホメオスタシス

人間の脳は「覚えること」だけでなく「忘れること」も能動的に行う。AnimaWorksはシナプスホメオスタシス仮説（Tononi & Cirelli, 2003）に基づき、3段階の能動的忘却を実装する。

```
覚醒時（会話中）                          睡眠時（非会話時）
────────────────                        ────────────────

 会話・検索 → access_count++              深夜cron
     │                                    │
     ▼                                    ▼
 [アクセス記録]                          [日次ダウンスケーリング]
 頻繁に使われる記憶は強化                  knowledge: 90日+未アクセス+低頻度
 (ヘブ則・LTP)                            procedures: 180日+未使用+低頻度
                                          or 効用<0.3+failure≥3 → 即座マーク
                                          (シナプスホメオスタシス)
                                           │
                                      週次cron
                                           │
                                           ▼
                                      [神経新生的再編]
                                      低活性+類似記憶のLLM統合
                                           │
                                      月次cron
                                           │
                                           ▼
                                      [完全忘却]
                                      低活性90日超+access_count≤2 → アーカイブ削除
                                      knowledge + episodes + procedures
                                      archive/versions/ クリーンアップ
```

| 段階 | 脳のプロセス | AnimaWorks実装 | 頻度 |
|---|---|---|---|
| **日次ダウンスケーリング** | NREM睡眠のシナプスダウンスケーリング | knowledge: 90日+未アクセス → 低活性マーク。procedures: 180日+未使用 or 効用<0.3+failure≥3 → 低活性マーク | 日次cron |
| **神経新生的再編** | 海馬歯状回の神経新生による記憶回路再編 | 低活性チャンク同士の類似ペアをLLM統合 | 週次cron |
| **完全忘却** | 閾値以下のシナプス消失 | 低活性90日超+access_count≤2のベクトルインデックス削除、ソースをアーカイブ（knowledge + episodes + procedures） | 月次cron |

### knowledge/ の忘却閾値

| 条件 | 値 | 説明 |
|---|---|---|
| 未アクセス期間 | 90日 | 最終アクセスから90日経過 |
| アクセス回数 | < 3回 | 使用頻度が低い |

### procedures/ の忘却閾値

procedures/ は knowledge/ より緩い閾値を持つ（手続き記憶は脳でも忘却耐性が高い）:

| 条件 | 値 | 説明 |
|---|---|---|
| 未使用期間 | 180日 | 最終使用から180日経過（knowledge の2倍） |
| 使用回数 | < 3回 | 使用頻度が低い |
| 即座マーク条件 | 効用 < 0.3 AND failure_count >= 3 | 繰り返し失敗した低効用手順は即座に低活性マーク |

### 忘却からの保護

| 対象 | 保護条件 | 理由 |
|---|---|---|
| `skills/` | 常に保護 | description-basedマッチングの起点。削除すると想起経路が断たれる |
| `shared/users/`（memory_type: shared_users） | 常に保護 | 対人記憶の保護 |
| `[IMPORTANT]` / `importance: important` | 条件付き（実装注意） | `_is_protected()` 先頭で `IMPORTANT_SAFETY_NET_DAYS`（365日）未アクセスなら保護解除の分岐があるが、`memory_type` が `knowledge` / `procedures` のチャンクは続く `_is_protected_knowledge` / `_is_protected_procedure` が **`importance == "important"` で再び True** を返すため、**現実装では [IMPORTANT] 付き knowledge / procedures は安全網満了後も忘却処理から除外され続ける**。エピソード等、後段チェックが無いタイプでは安全網が効く。 |
| `knowledge/` (success_count >= 2) | 条件付き保護 | 複数回有用と確認された知識 |
| `procedures/` (version >= 3) | 条件付き保護 | 再固定化を3回以上経た成熟手順 |
| `procedures/` (protected: true) | 条件付き保護 | フロントマターで手動保護指定 |
| `procedures/` ([IMPORTANT]) | 条件付き保護 | タグによる忘却耐性 |

### 月次アーカイブクリーンアップ

月次忘却パイプラインでは、`archive/versions/` に蓄積された旧バージョンを整理する。手順ファイルごとに直近5バージョンのみを保持し、それより古いバージョンは削除する。

---

## 統一アクティビティログ

> 実装: `core/memory/activity.py` — `ActivityLogger` クラス（Mixin構成: `PrimingMixin`, `TimelineMixin`, `ConversationMixin`, `RotationMixin`）

全インタラクションを単一のJSONL時系列に記録する統一ログ基盤。従来 transcript、dm_log、heartbeat_history 等に分散していた記録を一本化し、Primingレイヤーの「直近アクティビティ」チャネル（Channel B）の単一データソースとなる。実装は `_activity_models.py`（データモデル）、`_activity_priming.py`（プライミング整形）、`_activity_timeline.py`（API用タイムライン）、`_activity_conversation.py`（会話ビュー）、`_activity_rotation.py`（ローテーション）に分割されている。

### 保存場所

```
{anima_dir}/activity_log/{date}.jsonl
```

日付ごとに1ファイル。append-onlyで書き込み、各行が1エントリのJSON。

### JSONL形式

```json
{"ts":"2026-02-17T14:30:00","type":"message_received","content":"...","from":"user","channel":"chat"}
{"ts":"2026-02-17T14:30:05","type":"response_sent","content":"...","to":"user","channel":"chat"}
{"ts":"2026-02-17T15:00:00","type":"tool_use","tool":"web_search","summary":"検索実行"}
```

空フィールドは省略される。`from`/`to` は送信者/受信者名（内部では `from_person`/`to_person`）、`channel` はチャネル名、`tool` はツール名、`via` は通知チャネル（human_notifyイベント用）、`meta` は任意メタデータ（`from_type` 等）。`origin` と `origin_chain` はデータの出自追跡用（例: `"human"`, `"external_platform"`）。

### イベントタイプ一覧

| イベントタイプ | ASCIIラベル | 説明 |
|---|---|---|
| `message_received` | `MSG<` | メッセージ受信（人間・Anima両方。`meta.from_type` で区別） |
| `response_sent` | `RESP>` | Animaの応答送信（人間との会話応答） |
| `message_sent` | `MSG>` | DM送信（他Animaへのダイレクトメッセージ。旧 `dm_sent` からリネーム） |
| `channel_post` | `CH.W` | 共有チャネルへの投稿 |
| `channel_read` | `CH.R` | 共有チャネルの閲覧 |
| `human_notify` | `NTFY` | 人間への通知（call_human経由） |
| `tool_use` | `TOOL` | 外部ツールの使用 |
| `heartbeat_start` | `HB` | ハートビート開始 |
| `heartbeat_end` | `HB` | ハートビート終了 |
| `cron_executed` | `CRON` | cronタスクの実行 |
| `memory_write` | `MEM` | 記憶ファイルへの書き込み |
| `error` | `ERR` | エラー発生 |
| `issue_resolved` | `RSLV` | 課題の解決（ステート自動更新から自動記録） |
| `task_created` | `TSK+` | タスク作成 |
| `task_updated` | `TSK~` | タスク更新 |
| `tool_result` | `TRES` | ツール実行結果（consolidation用。メタのみ注入、生コンテンツは省略） |
| `inbox_processing_start` / `inbox_processing_end` | — | Inbox処理の開始/終了（ライブイベント配信対象） |

後方互換エイリアス: `dm_sent` → `message_sent`、`dm_received` → `message_received`（読み取り時に自動変換）

**ライブイベント**: `tool_use`、`inbox_processing_start`、`inbox_processing_end` は ProcessSupervisor 経由で WebSocket に即時配信され、UI のリアルタイム表示に利用される。

### Priming連携

`ActivityLogger.format_for_priming()` メソッドが、取得したエントリをトークンバジェット（デフォルト1300トークン、heartbeat時は最低400トークン保証）内で整形する。

**ASCIIラベル化**: 各イベントタイプを2-4文字のASCIIラベル（`MSG<`, `DM>`, `HB` 等）で表示。旧絵文字アイコン（`📨`, `💓` 等）は2-3トークン消費していたが、ASCIIラベルは1トークンで安定認識される。

**トピックグルーピング**: 関連エントリをグループ化してコンパクトに表示する。

| グループタイプ | 条件 | 表示形式 |
|---|---|---|
| DM | 同一peer、30分以内の連続DM | `[HH:MM-HH:MM] DM {peer}: {topic}` + 子行 |
| HB | 連続する heartbeat_start/end | `[HH:MM-HH:MM] HB: {summary}` |
| CRON | 同一task_nameのcron_executed | `[HH:MM] CRON {task}: exit={code}` |
| single | 上記以外 | `[HH:MM] {LABEL} {content}` |

**ポインタ参照**: 200文字超でtruncateされた場合、末尾にソースファイルポインタ `(-> activity_log/{date}.jsonl)` を付与。グループにはグループ末尾に `-> activity_log/{date}.jsonl#L{range}` を付与。LLMが詳細を必要とする場合に `read_memory_file` で元データを参照可能。

### アクティビティログローテーション

`config.json` の `activity_log` セクションでローテーションを設定する。`RotationMixin.rotate()` が実行され、古い日付のファイルを削除してディスク使用量を抑制する。

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `rotation_enabled` | true | ローテーションを有効にするか |
| `rotation_mode` | `"size"` | `"size"`（合計サイズ上限）、`"time"`（経過日数）、`"both"` |
| `max_size_mb` | 1024 | 1Animaあたりの最大合計サイズ（MB） |
| `max_age_days` | 7 | `time`/`both` モード時の最大保持日数 |
| `rotation_time` | `"05:00"` | 実行時刻（JST） |

ProcessSupervisor のスケジューラが `rotation_time` に従い、全 Anima に対して `ActivityLogger.rotate_all()` を実行する。

---

## ストリーミングジャーナル

> 実装: `core/memory/streaming_journal.py` — `StreamingJournal` クラス

LLMのストリーミング応答出力中に、テキストチャンクを逐次ディスクに書き込むWrite-Ahead Log（WAL）。プロセスのハードクラッシュ（SIGKILL, OOM等）が発生しても、最大約1秒分のテキスト損失に抑える。

### 保存場所

```
{anima_dir}/shortterm/streaming_journal_{session_type}.jsonl
```

セッションタイプ（`chat` / `heartbeat`）ごとにファイルが分離される。Chat と Heartbeat は独立したロックで並行動作するため、同一ファイルへの同時書き込みを避ける。thread_id 指定時は `shortterm/{session_type}/{thread_id}/streaming_journal.jsonl`。レガシー: `streaming_journal.jsonl`（chat 時のみ、マイグレーションで自動リネーム）

### WALライフサイクル

```
正常フロー:
  open() → write_text() / write_tool_*() → finalize() → ジャーナルファイル削除

異常フロー（クラッシュ）:
  open() → write_text() / write_tool_*() → <crash> → ジャーナルファイル残存
                                                        ↓
  次回起動時: recover() → JournalRecovery として復元 → ジャーナルファイル削除
```

- **open()**: 既存の孤立ジャーナルがあれば先に recover してエピソードに永続化。その後ジャーナルファイルを新規作成し、`start` イベント（トリガー、送信者、セッションID）を書き込む
- **write_text()**: テキストフラグメントをバッファに追加。バッファ条件を満たすとフラッシュ
- **write_tool_start() / write_tool_end()**: ツール実行の開始・終了を記録
- **finalize()**: `done` イベントを書き込み、ファイルを閉じて削除（正常完了）
- **recover()**: 孤立ジャーナルを読み込み、`JournalRecovery` データクラスとして返却

### バッファ設定

| パラメータ | 値 | 説明 |
|---|---|---|
| `_FLUSH_INTERVAL_SEC` | 1.0秒 | 最低フラッシュ間隔 |
| `_FLUSH_SIZE_CHARS` | 500文字 | バッファがこのサイズに達したらフラッシュ |

いずれかの条件を満たすとバッファ内容が `text` イベントとしてJSONL行に書き出され、`fsync()` される。

### リカバリー

`StreamingJournal.has_orphan(anima_dir, session_type)` で孤立ジャーナルの存在を確認し、`recover(anima_dir, session_type, thread_id)` で以下の情報を復元する:

- 復元テキスト（全 `text` イベントの結合）
- ツールコールの記録（開始/完了ステータス付き）
- セッション情報（トリガー、送信者、開始時刻）
- 完了フラグ（`done` イベントの有無）

壊れたJSONL行（クラッシュ時の部分書き込み）はスキップされる。復元後、`_persist_recovery()` で `episodes/recovered_{timestamp}.md` に永続化し、ジャーナルファイルは削除される。

---

## 設計原則

1. **二重ストアは必須** — エピソード記憶（raw記録）と意味記憶（蒸留された知識）の両方を保持する
2. **想起は二重経路** — 自動想起（プライミング）と意図的想起（ツール呼び出し）の2つを実装する
3. **記憶インフラはフレームワークの責務** — プライミング・RAG・忘却・ログ等の基盤はフレームワークが担う。日次/週次の固定化はスケジューラが `run_consolidation` を起動し、**バックグラウンド用モデル**で Anima がツールループにより実行する（インフラの「いつ動かすか」と後処理はフレームワーク、具体の読み書き判断はそのセッション上の LLM）
4. **固定化は毎日実行する** — 脳のNREM睡眠は毎晩行われる。日次固定化 + 週次統合の2段階が最小要件
5. **文脈は一級の検索次元** — 記憶の格納時にリッチなメタデータを付与し、検索時に現在の文脈との一致度で優先する
6. **ワーキングメモリの容量制限は設計上の特徴** — コンテキストウィンドウの制限はバグではなく機能。最も関連性の高い情報を選択的に保持する
7. **能動的忘却はシステムの健全性を維持する** — 記憶は増え続ける一方ではなく、低活性の記憶を能動的に刈り込むことで検索精度（S/N比）を維持する
8. **手続き記憶は使用で強化される** — 手順の信頼度は成功/失敗フィードバックで動的に更新される。繰り返し成功した手順ほど忘却耐性が高まる
9. **矛盾は検出し解決する** — 知識間の矛盾を放置せず、NLI+LLMカスケードで自動検出・解決する

---

## core/memory/ モジュールリファレンス

記憶サブシステムは `core/memory/` 配下の専門モジュール群で実装されている。

### プライミング（`priming/` パッケージ）

| モジュール | 役割 |
|---|---|
| `priming/engine.py` | `PrimingEngine`, `PrimingResult` — 並列取得オーケストレーション |
| `priming/budget.py` | `classify_message_type` / `adjust_token_budget` / `load_config_budgets`（`PrimingConfig` の数値。`dynamic_budget` フィールド自体はエンジン未参照） |
| `priming/constants.py` | チャネル別既定バジェット・キーワード用定数 |
| `priming/format.py` | プロンプト用セクション整形 `format_priming_section` |
| `priming/utils.py` | `RetrieverCache`, `build_queries`, `search_and_merge`, キーワード抽出, truncate |
| `priming/outbound.py` | Recent Outbound、`human_notify`（保留中人間通知）収集 |
| `priming/channel_a.py` 〜 `channel_f.py` | チャネル A〜F の実装（送信者、活動、知識、スキル、タスク、エピソード） |

公開 API は `from core.memory.priming import PrimingEngine, PrimingResult, format_priming_section`（`core/memory/__init__.py` から再エクスポート）。チャット経路での `prime_memories` 呼び出しは `core/_agent_priming.py`。

### 会話記憶（分割モジュール）

| モジュール | 役割 |
|---|---|
| `conversation.py` | `ConversationMemory` ファサード（圧縮・確定・状態更新を委譲） |
| `conversation_models.py` | `ConversationTurn`, `ConversationState` 等のデータモデル |
| `conversation_compression.py` | ローリング要約・圧縮 |
| `conversation_finalize.py` | セッション確定・エピソード追記・解決伝播など |
| `conversation_prompt.py` | LLM 向けプロンプト断片の組み立て |
| `conversation_state_update.py` | `current_state.md` 等の自動更新ヘルパ |

### その他コア

| モジュール | クラス / 役割 | 説明 |
|---|---|---|
| `manager.py` | `MemoryManager` | ファイルベース記憶、スキルマッチ、RAG 検索のファサード |
| `shortterm.py` | `ShortTermMemory` | `shortterm/{session_type}/` のセッション状態外部化 |
| `activity.py` | `ActivityLogger` | タイムライン本体（Mixin は `_activity_*.py` に分割） |
| `_activity_models.py` 等 | データ / Priming 整形 / API タイムライン / 会話ビュー / ローテーション | `activity.py` から利用 |
| `consolidation.py` | `ConsolidationEngine` | 固定化前処理・後処理（RAG 再構築等） |
| `forgetting.py` | `ForgettingEngine` | `synaptic_downscaling` / `neurogenesis_reorganize` / `complete_forgetting` / `cleanup_procedure_archives` |
| `streaming_journal.py` | `StreamingJournal` | WAL ストリーミング出力 |
| `task_queue.py` | `TaskQueueManager` | 永続タスクキュー JSONL |
| `distillation.py` | `ProceduralDistiller` | 手続き蒸留（補助経路） |
| `reconsolidation.py` | `ReconsolidationEngine` | 再固定化・issue_resolved→procedure |
| `resolution_tracker.py` | `ResolutionTracker` | `shared/resolutions.jsonl` |
| `cron_logger.py` | `CronLogger` | `state/cron_logs/` |
| `skill_metadata.py` | 関数群 | スキルマッチ用正規化・キーワード |
| `validation.py` | `KnowledgeValidator` | NLI+LLM 検証 |
| `contradiction.py` | `ContradictionDetector` | 矛盾検出・解決ユーティリティ |
| `dedup.py` | — | メッセージ重複排除・heartbeat レート制限 |
| `housekeeping.py` | `run_housekeeping()` | ログ・shortterm 等の日次クリーンアップ |
| `frontmatter.py` | `FrontmatterService` | YAML フロントマター |
| `rag_search.py` | `RAGMemorySearch` | 検索・インデクサーラッパー |
| `audit.py` | `AuditAggregator` 等 | スーパーバイザー向け activity + タスク集計 |
| `token_usage.py` | — | `token_usage/{date}.jsonl` への利用トークン・コスト記録。単価はモジュール内 `DEFAULT_PRICING` を基準に `~/.animaworks/pricing.json` で上書き可能 |
| `config_reader.py` | — | 記憶関連設定の読み取りヘルパ |
| `_io.py`, `_llm_utils.py` | — | 内部 I/O・LLM 呼出し補助 |

### RAG（`rag/`）

| モジュール | 役割 |
|---|---|
| `rag/indexer.py` | `MemoryIndexer` — チャンキング・埋め込み・増分インデックス |
| `rag/retriever.py` | `MemoryRetriever` — ベクトル検索・減衰・重要度・拡散・`record_access` |
| `rag/graph.py` | `KnowledgeGraph` — グラフ構築・PageRank・結果拡張 |
| `rag/store.py` | `ChromaVectorStore` 等 — Chroma 抽象化 |
| `rag/http_store.py` | — | HTTP バックエンド向けストア実装（利用時） |
| `rag/singleton.py` | — | ベクトルストア・埋め込みのプロセス内シングルトン |
| `rag/watcher.py` | `FileWatcher` | ファイル変更監視と再インデックス |

---

## 関連ドキュメント

- [vision.ja.md](vision.ja.md) — Digital Animaの基本理念
- [spec.md](spec.md) — 要件定義書（書庫型記憶の基本設計）
- [features.ja.md](features.ja.md) — 機能一覧（記憶システム関連の実装履歴を含む）
- [specs/20260214_priming-layer_design.md](specs/20260214_priming-layer_design.md) — プライミングレイヤー実装計画書(RAG設計、固定化アーキテクチャ含む)
- [specs/20260218_unified-activity-log-implemented-20260218.md](specs/20260218_unified-activity-log-implemented-20260218.md) — 統一アクティビティログ設計書
- [specs/20260218_streaming-journal-implemented-20260218.md](specs/20260218_streaming-journal-implemented-20260218.md) — ストリーミングジャーナル設計書
- [specs/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md](specs/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md) — アクティビティログ仕様準拠修正
- [specs/20260218_priming-format-redesign_implemented-20260218.md](specs/20260218_priming-format-redesign_implemented-20260218.md) — Primingフォーマット再設計（ASCIIラベル化・トピックグルーピング・ポインタ参照）
- [specs/20260218_episode-dedup-state-autoupdate-resolution-propagation.md](specs/20260218_episode-dedup-state-autoupdate-resolution-propagation.md) — エピソード重複修正・ステート自動更新・解決伝播メカニズム
- [specs/20260218_memory-system-enhancement-checklist-20260218.md](specs/20260218_memory-system-enhancement-checklist-20260218.md) — 記憶システム強化チェックリスト
- [specs/20260218_consolidation-validation-pipeline-20260218.md](specs/20260218_consolidation-validation-pipeline-20260218.md) — 日次固定化バリデーションパイプライン
- [specs/20260218_knowledge-contradiction-detection-resolution-20260218.md](specs/20260218_knowledge-contradiction-detection-resolution-20260218.md) — 知識矛盾検出・解決
- [specs/20260218_procedural-memory-foundation-20260218.md](specs/20260218_procedural-memory-foundation-20260218.md) — 手続き記憶基盤（YAMLフロントマター・3段階マッチング）
- [specs/20260218_procedural-memory-auto-distillation-20260218.md](specs/20260218_procedural-memory-auto-distillation-20260218.md) — 手続き記憶自動蒸留
- [specs/20260218_procedural-memory-reconsolidation-20260218.md](specs/20260218_procedural-memory-reconsolidation-20260218.md) — 予測誤差ベース再固定化
- [specs/20260218_procedural-memory-utility-forgetting-20260218.md](specs/20260218_procedural-memory-utility-forgetting-20260218.md) — 手続き記憶の効用ベース忘却

