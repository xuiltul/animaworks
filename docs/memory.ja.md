# AnimaWorks 記憶システム設計仕様書

**[English version](memory.md)**

> 作成日: 2026-02-14
> 更新日: 2026-02-18
> 関連: [vision.ja.md](vision.ja.md), [spec.md](spec.md), [implemented/20260214_priming-layer_design.md](implemented/20260214_priming-layer_design.md)
> 調査資料: [AI Agent記憶アーキテクチャ調査](research/20260212_AI_Agent記憶アーキテクチャ調査.md) セクション10

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
├── shortterm/       短期記憶（セッション状態、ストリーミングジャーナル）
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
│  ── 記憶固定化（フレームワーク自動実行） ──                │
│                                                          │
│  [即時] セッション境界検出 → 差分要約 → episodes/        │
│         + ステート自動更新 + 解決伝播                     │
│  [日次] 深夜cron → episodes/ → 手続き分類・蒸留          │
│         → NLI+LLMバリデーション → knowledge/ 書き込み    │
│         → 再固定化 → 矛盾検出・解決                      │
│  [週次] 週次cron → knowledge/ 統合 + episodes/ 圧縮      │
│         + 週次パターン蒸留 + 全knowledge矛盾スキャン      │
│  [月次] 月次cron → 完全忘却（knowledge + episodes        │
│         + procedures） + アーカイブクリーンアップ          │
│                                                          │
│  ── 忘却（シナプスホメオスタシス） ──                     │
│                                                          │
│  [日次] Synaptic Downscaling: knowledge(90日)            │
│         + procedures(180日 or 効用低下) → 低活性マーク    │
│  [週次] 神経新生的再編: 低活性+類似チャンクのLLM統合      │
│  [月次] 完全忘却: 低活性60日超 → アーカイブ削除           │
│         + archive/procedure_versions/ クリーンアップ       │
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
                (送信者、キーワード)   (4チャネル並列)    (トークン予算内)    (記憶が既にある)
```

4つの検索チャネル（`core/memory/priming.py`）:

| チャネル | 対象 | バジェット | 方式 | 脳の対応 |
|---|---|---|---|---|
| **A: 送信者プロファイル** | shared/users/ | 500トークン | 完全一致ルックアップ | 顔を見た瞬間の自動想起 |
| **B: 直近アクティビティ** | activity_log/ | 1300トークン | ActivityLoggerから時系列取得 | 短期〜近時記憶。「最近何があったか」 |
| **C: 関連知識** | knowledge/ | 700トークン | 密ベクトル類似度検索（RAG） | 拡散活性化による連想 |
| **D: スキル/手順マッチ** | skills/, procedures/, common_skills/ | 200トークン | descriptionベースマッチング | 「できること」「やり方」のリスト想起 |

チャネルBは旧来の `episodes/` 日付フィルタと共有チャネル読み取りを統合し、`ActivityLogger` による統一アクティビティログからの取得に変更された。アクティビティログが空の場合は旧形式（episodes/ + channels/）にフォールバックする。

チャネルDは `skills/` に加え `procedures/` も検索対象に含む。3段階マッチング（ブラケットキーワード、語彙マッチ、RAGベクトル検索）で手続き記憶の想起精度を高める（詳細は「手続き記憶ライフサイクル」セクション参照）。

メッセージタイプ別の動的バジェット配分:

| メッセージタイプ | トークンバジェット | 用途 |
|---|---|---|
| greeting | 500 | 挨拶（短文、低負荷） |
| question | 1500 | 質問（中程度の記憶検索） |
| request | 3000 | 依頼・指示（広範な記憶検索） |
| heartbeat | 200 | 定期巡回（最小限の記憶参照） |

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

**AnimaWorks実装**: 密ベクトル検索と時間減衰の組み合わせにより拡散活性化を近似する。

初期設計ではBM25（キーワード）とベクトル検索のハイブリッドをRRFで統合する方針だったが、調査の結果、多言語対応の密ベクトル検索単体の方がキーワード検索よりも高精度であることが判明したため、**ベクトル類似度検索に一本化**した。

| 検索信号 | 方式 | 脳の対応 |
|---|---|---|
| **意味ベクトル** | 密ベクトル類似度検索（`intfloat/multilingual-e5-small`, 384次元, ChromaDB） | 概念的近傍の発見。拡散活性化の近似 |
| **時間減衰** | 指数減衰関数（半減期30日, 重み0.2） | 最近の記憶ほど活性化されやすい（近時効果） |
| **アクセス頻度** | 対数ブースト（`log(1 + access_count)`, 重み0.1） | ヘブの法則・LTP。繰り返し使用された記憶はより想起されやすい |
| **グラフ拡散** | ナレッジグラフ + Personalized PageRank（オプション） | 多ホップの連想伝播。明示リンク `[[]]` + 暗黙リンク（類似度≥0.75） |

最終スコアの算出:

```
final_score = vector_similarity_score + (decay_factor × WEIGHT_RECENCY) + (WEIGHT_FREQUENCY × log(1 + access_count))

decay_factor = 0.5 ^ (age_days / 30.0)   # 30日で半減
WEIGHT_RECENCY = 0.2                       # 時間減衰の最大寄与は0.2
WEIGHT_FREQUENCY = 0.1                     # アクセス頻度の寄与（ヘブ則のLTP相当）

# グラフ拡散が有効な場合:
final_score += pagerank_score × 0.5
```

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

人間の脳は、記憶の固定化を無意識の自動プロセスとして実行する。AnimaWorksも同様に、**すべての固定化をフレームワーク側で自動実行**する。フレームワークはバックグラウンドで別途LLMをワンショット呼出しし、会話要約（即時符号化）やパターン抽出（日次固定化）を実行する。エージェント本人はこのプロセスに関与しない。

```
覚醒時（会話中）                        睡眠時（非会話時）
────────────────                      ────────────────

 会話 → セッション境界検出              深夜cron
     │  (10分アイドル or heartbeat)       │
     ▼                                  ▼
 [即時符号化]                          [日次固定化]
 差分要約 → episodes/                  episodes/ → 手続き分類
 + ステート自動更新                     → 手続き蒸留 → procedures/
 + 解決伝播                            → LLM知識抽出
 海馬の1ショット記録                    → NLI+LLMバリデーション
                                       → knowledge/ 書き込み
                                       → RAGインデックス更新
                                       → Synaptic Downscaling
                                       → 再固定化 → 矛盾検出・解決
                                       NREM睡眠の固定化
                                          │
                                     週次cron
                                          │
                                          ▼
                                     [週次統合]
                                     knowledge/ 重複排除・マージ
                                     episodes/ 圧縮
                                     週次パターン蒸留
                                     全knowledge矛盾スキャン
                                     新皮質の長期統合
                                          │
                                     月次cron
                                          │
                                          ▼
                                     [月次忘却]
                                     完全忘却（knowledge + episodes
                                       + procedures）
                                     archive/procedure_versions/
                                       クリーンアップ
```

### 日次固定化パイプライン

> 実装: `core/memory/consolidation.py` — `daily_consolidate()`

```
daily_consolidate()
├── レガシーマイグレーション（フロントマター未付与ファイルの自動変換）
├── エピソード収集
├── 手続きコンテンツ分類（エピソードをknowledge/procedures部分に分類）
├── 手続き自動蒸留（手続き的エピソードからprocedures/に自動抽出）
├── LLM知識抽出（意味的エピソードからknowledge候補を抽出）
├── コードフェンスサニタイズ
├── NLI+LLMバリデーション（ハルシネーション排除）
├── knowledge/ 書き込み（YAMLフロントマター付き）
├── RAGインデックス更新
├── Synaptic Downscaling（低活性チャンクマーク、procedures含む）
├── 予測誤差ベースの再固定化（既存記憶と新エピソードの矛盾検出・更新）
└── 矛盾検出・解決（新規/更新knowledge間のsupersede/merge/coexist判定）
```

### 週次統合パイプライン

- `knowledge/` 重複排除・マージ（`archive/merged/` にアーカイブ）
- `episodes/` 圧縮
- 週次パターン蒸留（`activity_log/` から反復パターンを検出し `procedures/` に手順化）
- 全 `knowledge/` 矛盾スキャン（日次で未検出の矛盾を網羅的に検出）

### 月次忘却パイプライン

- 完全忘却（`knowledge/` + `episodes/` + `procedures/`）
- `archive/procedure_versions/` クリーンアップ（手順ごとに直近5バージョンのみ保持）

### 固定化段階一覧

| 段階 | 脳のプロセス | AnimaWorks実装 | 担当 | 頻度 |
|---|---|---|---|---|
| **即時符号化** | 海馬の高速1ショット符号化 | セッション境界検出（10分アイドル or heartbeat）→ 差分要約 → episodes/ 自動記録 + ステート自動更新 + 解決伝播 | フレームワーク（bg LLM呼出） | セッション境界時 |
| **日次固定化** | NREM睡眠の徐波-紡錘波-リップル カスケード | 深夜cron → 手続き分類・蒸留 → LLM知識抽出 → NLI+LLMバリデーション → knowledge/ 書き込み → 再固定化 → 矛盾検出・解決 | フレームワーク（bg LLM呼出） | 毎日深夜 |
| **週次統合** | 新皮質の長期統合・シナプスダウンスケーリング | 週次cron → knowledge/ 統合 + episodes/ 圧縮 + 週次パターン蒸留 + 全knowledge矛盾スキャン | フレームワーク（bg LLM呼出） | 毎週 |
| **月次忘却** | 閾値以下のシナプス消失 | 月次cron → 完全忘却（knowledge + episodes + procedures） + アーカイブクリーンアップ | フレームワーク（bg cron） | 毎月 |
| **意図的記銘** | 前頭前皮質の精緻化符号化 | write_memory_file で直接書き込み | エージェント | 随時 |

エージェントに残る唯一の書き込み経路は**意図的記銘**（write_memory_file）。これは人間が意識的にメモを取る行為に相当する。それ以外の符号化・固定化・統合はすべてフレームワークが自動で行う。

### 即時符号化の詳細: セッション境界ベースの差分要約

旧設計ではメッセージ応答のたびに全ターンを再要約していたが、これにより同一会話の要約がN-2回重複記録される問題があった。現設計では `last_finalized_turn_index` で記録済み位置を追跡し、**未記録ターンのみを差分要約**する。

**セッション境界**: メッセージ応答時ではなく、以下の2つの条件でのみ `finalize_session()` が実行される:
- **10分アイドル**: 最終ターンから10分経過時（`finalize_if_session_ended()` で検出）
- **heartbeat到達**: 定期巡回時に `finalize_if_session_ended()` を呼び出し

**統合ポイント**: `finalize_session()` は差分要約に加え、以下を同時実行する:
1. **エピソード記録**: 未記録ターンのLLM要約を `episodes/` に追記
2. **ステート自動更新**: LLM要約から「解決済みアイテム」「新規タスク」を自動パースし `state/current_task.md` に追記
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

日次固定化でLLMが抽出した knowledge 候補は、そのまま書き込むとハルシネーション（元のエピソードに存在しない情報の捏造）が混入する可能性がある。これを排除するため、**NLI（Natural Language Inference）モデルとLLMのカスケード検証**を行う。

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

`knowledge/` に蓄積された知識ファイル間で矛盾が発生することがある（例: 「Aさんの担当はX」と「Aさんの担当はY」）。AnimaWorksはNLI+LLMカスケードで矛盾を検出し、3つの解決戦略で自動解決する。

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

### 自動蒸留: エピソードから手順を抽出

**日次蒸留**（`daily_consolidate()` 内）:

1. **手続きコンテンツ分類**: エピソードを12パターンの正規表現（「手順」「設定方法」「コマンド」「ワークフロー」等）で手続き的/意味的に分類
2. **LLM手順抽出**: 手続き的と分類されたエピソードからLLMが構造化された手順書を生成
3. **RAG重複チェック**: 類似度0.85以上の既存手順が存在する場合は新規作成をスキップ（手順の増殖防止）
4. **procedures/ 書き込み**: YAMLフロントマター付きで保存。`auto_distilled: true`, `confidence: 0.4`

**週次パターン蒸留**（週次統合パイプライン内）:

1. `activity_log/` から1週間分のアクティビティを分析
2. 反復的に出現するパターン（同じ手順を複数回実行等）を検出
3. LLMがパターンを手順書に蒸留し `procedures/` に保存

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

**AnimaWorks実装**: 新しいエピソードが既存の `knowledge/` や `procedures/` と矛盾する場合、NLIで矛盾を検出し、LLMが分析・更新を行う。

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
- 旧バージョンを `archive/procedure_versions/` に保存
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
                                      低活性60日+未アクセス → アーカイブ削除
                                      knowledge + episodes + procedures
                                      archive/procedure_versions/ クリーンアップ
```

| 段階 | 脳のプロセス | AnimaWorks実装 | 頻度 |
|---|---|---|---|
| **日次ダウンスケーリング** | NREM睡眠のシナプスダウンスケーリング | knowledge: 90日+未アクセス → 低活性マーク。procedures: 180日+未使用 or 効用<0.3+failure≥3 → 低活性マーク | 日次cron |
| **神経新生的再編** | 海馬歯状回の神経新生による記憶回路再編 | 低活性チャンク同士の類似ペアをLLM統合 | 週次cron |
| **完全忘却** | 閾値以下のシナプス消失 | 低活性60日+未アクセスのベクトルインデックス削除、ソースをアーカイブ（knowledge + episodes + procedures） | 月次cron |

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
| `shared/users/` | 常に保護 | 対人記憶の保護 |
| `[IMPORTANT]` タグ付き | 常に保護 | 精緻化符号化による忘却耐性 |
| `procedures/` (version >= 3) | 条件付き保護 | 再固定化を3回以上経た成熟手順 |
| `procedures/` (protected: true) | 条件付き保護 | フロントマターで手動保護指定 |
| `procedures/` ([IMPORTANT]) | 条件付き保護 | タグによる忘却耐性 |

### 月次アーカイブクリーンアップ

月次忘却パイプラインでは、`archive/procedure_versions/` に蓄積された旧バージョンを整理する。手順ファイルごとに直近5バージョンのみを保持し、それより古いバージョンは削除する。

---

## 統一アクティビティログ

> 実装: `core/memory/activity.py` — `ActivityLogger` クラス

全インタラクションを単一のJSONL時系列に記録する統一ログ基盤。従来 transcript、dm_log、heartbeat_history 等に分散していた記録を一本化し、Primingレイヤーの「直近アクティビティ」チャネル（Channel B）の単一データソースとなる。

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

空フィールドは省略される。`from`/`to` は送信者/受信者名、`channel` はチャネル名、`tool` はツール名、`via` は通知チャネル（human_notifyイベント用）。

### イベントタイプ一覧

| イベントタイプ | ASCIIラベル | 説明 |
|---|---|---|
| `message_received` | `MSG<` | ユーザーからのメッセージ受信 |
| `response_sent` | `MSG>` | Animaからの応答送信 |
| `dm_sent` | `DM>` | 他Animaへのダイレクトメッセージ送信 |
| `dm_received` | `DM<` | 他Animaからのダイレクトメッセージ受信 |
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

---

## ストリーミングジャーナル

> 実装: `core/memory/streaming_journal.py` — `StreamingJournal` クラス

LLMのストリーミング応答出力中に、テキストチャンクを逐次ディスクに書き込むWrite-Ahead Log（WAL）。プロセスのハードクラッシュ（SIGKILL, OOM等）が発生しても、最大約1秒分のテキスト損失に抑える。

### 保存場所

```
{anima_dir}/shortterm/streaming_journal.jsonl
```

### WALライフサイクル

```
正常フロー:
  open() → write_text() / write_tool_*() → finalize() → ジャーナルファイル削除

異常フロー（クラッシュ）:
  open() → write_text() / write_tool_*() → <crash> → ジャーナルファイル残存
                                                        ↓
  次回起動時: recover() → JournalRecovery として復元 → ジャーナルファイル削除
```

- **open()**: ジャーナルファイルを新規作成し、`start` イベント（トリガー、送信者、セッションID）を書き込む
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

次回起動時に `StreamingJournal.has_orphan()` で孤立ジャーナルの存在を確認し、`recover()` で以下の情報を復元する:

- 復元テキスト（全 `text` イベントの結合）
- ツールコールの記録（開始/完了ステータス付き）
- セッション情報（トリガー、送信者、開始時刻）
- 完了フラグ（`done` イベントの有無）

壊れたJSONL行（クラッシュ時の部分書き込み）はスキップされる。

---

## 設計原則

1. **二重ストアは必須** — エピソード記憶（raw記録）と意味記憶（蒸留された知識）の両方を保持する
2. **想起は二重経路** — 自動想起（プライミング）と意図的想起（ツール呼び出し）の2つを実装する
3. **記憶インフラはフレームワークの責務** — 符号化・固定化・統合はフレームワークがバックグラウンドLLMをワンショット呼出しして自動実行する。エージェント（Animaの主LLM）は記憶インフラを管理しない
4. **固定化は毎日実行する** — 脳のNREM睡眠は毎晩行われる。日次固定化 + 週次統合の2段階が最小要件
5. **文脈は一級の検索次元** — 記憶の格納時にリッチなメタデータを付与し、検索時に現在の文脈との一致度で優先する
6. **ワーキングメモリの容量制限は設計上の特徴** — コンテキストウィンドウの制限はバグではなく機能。最も関連性の高い情報を選択的に保持する
7. **能動的忘却はシステムの健全性を維持する** — 記憶は増え続ける一方ではなく、低活性の記憶を能動的に刈り込むことで検索精度（S/N比）を維持する
8. **手続き記憶は使用で強化される** — 手順の信頼度は成功/失敗フィードバックで動的に更新される。繰り返し成功した手順ほど忘却耐性が高まる
9. **矛盾は検出し解決する** — 知識間の矛盾を放置せず、NLI+LLMカスケードで自動検出・解決する

---

## 関連ドキュメント

- [vision.ja.md](vision.ja.md) — Digital Animaの基本理念
- [spec.md](spec.md) — 要件定義書（書庫型記憶の基本設計）
- [features.ja.md](features.ja.md) — 機能一覧（記憶システム関連の実装履歴を含む）
- [implemented/20260214_priming-layer_design.md](implemented/20260214_priming-layer_design.md) — プライミングレイヤー実装計画書(RAG設計、固定化アーキテクチャ含む)
- [implemented/20260218_unified-activity-log-implemented-20260218.md](implemented/20260218_unified-activity-log-implemented-20260218.md) — 統一アクティビティログ設計書
- [implemented/20260218_streaming-journal-implemented-20260218.md](implemented/20260218_streaming-journal-implemented-20260218.md) — ストリーミングジャーナル設計書
- [implemented/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md](implemented/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md) — アクティビティログ仕様準拠修正
- [implemented/20260218_priming-format-redesign_implemented-20260218.md](implemented/20260218_priming-format-redesign_implemented-20260218.md) — Primingフォーマット再設計（ASCIIラベル化・トピックグルーピング・ポインタ参照）
- [implemented/20260218_episode-dedup-state-autoupdate-resolution-propagation.md](implemented/20260218_episode-dedup-state-autoupdate-resolution-propagation.md) — エピソード重複修正・ステート自動更新・解決伝播メカニズム
- [implemented/20260218_memory-system-enhancement-checklist-20260218.md](implemented/20260218_memory-system-enhancement-checklist-20260218.md) — 記憶システム強化チェックリスト
- [implemented/20260218_consolidation-validation-pipeline-20260218.md](implemented/20260218_consolidation-validation-pipeline-20260218.md) — 日次固定化バリデーションパイプライン
- [implemented/20260218_knowledge-contradiction-detection-resolution-20260218.md](implemented/20260218_knowledge-contradiction-detection-resolution-20260218.md) — 知識矛盾検出・解決
- [implemented/20260218_procedural-memory-foundation-20260218.md](implemented/20260218_procedural-memory-foundation-20260218.md) — 手続き記憶基盤（YAMLフロントマター・3段階マッチング）
- [implemented/20260218_procedural-memory-auto-distillation-20260218.md](implemented/20260218_procedural-memory-auto-distillation-20260218.md) — 手続き記憶自動蒸留
- [implemented/20260218_procedural-memory-reconsolidation-20260218.md](implemented/20260218_procedural-memory-reconsolidation-20260218.md) — 予測誤差ベース再固定化
- [implemented/20260218_procedural-memory-utility-forgetting-20260218.md](implemented/20260218_procedural-memory-utility-forgetting-20260218.md) — 手続き記憶の効用ベース忘却
- [AI Agent記憶アーキテクチャ調査](research/20260212_AI_Agent記憶アーキテクチャ調査.md) — 脳科学調査・先行研究サーベイ
