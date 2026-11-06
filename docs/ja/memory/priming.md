> 確認したコミット: 193a5e72

# 自動想起（Priming）

自動想起は、入力を受けた実行経路から関連する記憶の手がかりを取得し、LLM のコンテキストに加える処理である。構成は `core/memory/priming/engine.py` の `PrimingEngine`、既定プロファイルは `core/config/schemas.py` の `PrimingConfig` で定義される。

## プロファイルとチャネル

`PrimingConfig.profile` の既定値は `compact` である。Anima ごとの `status.json` に `priming_profile` を `compact` または `full` と指定すると、その Anima の設定を上書きできる。チャネルの概略は次の通りである。

| チャネル | 取得するもの | `compact` | `full` |
|---|---|---:|---:|
| A | 送信者プロフィール | ○ | ○ |
| B | アクティビティログと共有チャネルの最近の動き | 背景トリガーで条件付き | ○ |
| C0 | 常駐候補・重要知識のポインター | ○ | ○ |
| C | 関連する知識の検索 | 条件付き | ○ |
| E | 未完了タスク | ○ | ○ |
| F | 過去のエピソード | — | ○ |
| G | 独立チャネルではない。エピソード検索に付随するグラフ由来候補 | — | ○（F の検索内） |
| 補助 | 最近の送信内容、人間への未送通知 | ○ | ○ |

`compact` ではチャネル A・E・C0 と補助情報を並列に集める。C の検索を行うのは `channel` が `chat` または `task`、あるいは `intent` が `question`、`request`、`delegation` の場合である。B は `heartbeat`・`inbox`・`cron` の背景トリガーで、`compact_background_recall_enabled` が有効かつ最近の活動の件数・トークン上限が正の場合に取得する。F も背景トリガーでクエリがあり、エピソードの件数・トークン上限が正の場合に取得する。G は独立した Priming チャネルではなく、F の検索内で利用され得る。

## C0 と `always_prime`

知識ファイルに `always_prime: true` を設定すると、RAG 索引上で自動想起の常駐候補として扱われる。`compact` は C0 を `resident_only=True` で呼び出し、この明示的 opt-in の候補から通常の知識ポインターを最大 3 件選ぶ。現行の `compact` 経路は入力クエリとの一致を確認せず、常駐候補を更新日時順に選ぶ。ACTION-RULE として扱われる候補は本文で注入される場合がある。`[IMPORTANT]` であることだけを理由に、無関係な知識を常駐注入するわけではない。`full` は常駐候補に加え、クエリに関連する重要知識も検索する。

## 検索方針

検索方針は `core/memory/retrieval/unified_search.py` の `TRIGGER_POLICIES` で定義される。`scope="all"` の検索では trigger に応じて scope、候補数、rerank の有無が変わる。C と F は個別の scope を指定するため、その scope を使いつつ trigger ごとの候補数と rerank 方針が適用される。

| trigger | 主な検索 scope | rerank |
|---|---|---:|
| `chat` | facts、episodes、knowledge、procedures、activity log | 有効 |
| `inbox` | facts、episodes、activity log | 有効 |
| `heartbeat` | episodes、activity log | 無効 |
| `task` | facts、procedures、knowledge | 有効 |
| `cron` | facts、episodes、knowledge、activity log | 有効 |
| `tool` | ツール検索用の scope | 有効 |

個々の検索段階と `search_memory` の scope は[意図的想起と検索](retrieval.md)を参照する。

## 上限とタイムアウト

`priming.max_tokens` の既定値は 2,000 であり、プロファイルに渡す想起結果の上限として使われる。メッセージ種別に応じてこの値を動的に配分する仕組みではない。未送の人間通知は別の契約で扱われ、通常の想起予算を満たすために除外されない。

`priming.channel_timeout_seconds` の既定値は 60 秒である。チャネルごとに上限を設け、時間切れのチャネルのみを空として扱い、ほかの取得結果は継続する。設定の全項目は[設定リファレンス](../reference/config.md)を参照する。
