# 調査: Priming Channel C が episodes を検索対象にしていない問題

## 調査日
2026-03-04

## 背景
AnimaWorksフレームワークの記憶想起エンジン（PrimingEngine）の Channel C（related_knowledge）が、episodes（エピソード記憶）を検索対象に含めていない。これにより、過去に行った作業（スクリプト作成等）の記憶が自動想起されず、Animaが記憶を思い出せない問題が発生した。

---

## 1. Channel C で episodes が除外されている具体的コード箇所

### 1.1 `core/memory/priming.py`

**箇所**: 557-567行目

```python
# Vector search (personal + shared common_knowledge)
results = retriever.search(
    query=query,
    anima_name=anima_name,
    memory_type="knowledge",   # ← ここで knowledge のみ指定、episodes は含まれない
    top_k=5,
    include_shared=True,
)
```

**原因**: `memory_type="knowledge"` がハードコードされており、episodes コレクションへの検索が行われない。

**補足**: `_channel_c_related_knowledge()` メソッド（541-623行目）全体が Channel C の実装。`knowledge_dir` の存在チェック（556行目）はあるが、episodes は別コレクションのため検索対象外。

---

## 2. search_memory で episodes が除外されている具体的コード箇所

### 2.1 `core/memory/rag_search.py`

**箇所1**: 217-230行目 `_resolve_search_types()` メソッド

```python
@staticmethod
def _resolve_search_types(scope: str) -> list[str]:
    """Map scope to memory_type list for vector search."""
    if scope == "knowledge":
        return ["knowledge"]
    if scope == "procedures":
        return ["procedures"]
    if scope == "common_knowledge":
        return ["knowledge"]
    if scope == "conversation_summary":
        return ["conversation_summary"]
    if scope == "all":
        return ["knowledge", "procedures", "conversation_summary"]  # ← episodes が含まれていない
    return ["knowledge"]
```

**原因**: `scope="all"` 時に返す memory_type リストに `"episodes"` が含まれていない。

**箇所2**: 202-212行目 ベクトル検索の条件分岐

```python
# Hybrid: append vector search results when RAG is available
if self._indexer is not None and scope in (
    "knowledge", "common_knowledge", "procedures", "conversation_summary", "all",
):
    try:
        vector_hits = self._vector_search_memory(query, scope, knowledge_dir)
        ...
```

**補足**: `scope="episodes"` の場合はこの条件に入らず、ベクトル検索が実行されない。episodes はキーワード検索（176-184行目）では `scope in ("episodes", "all")` のとき `dirs` に含まれるが、ベクトル検索は `_resolve_search_types` 経由で memory_type を決めるため、`scope="all"` でも episodes のベクトル検索は行われない。

---

## 3. RAGインデックスに episodes が含まれているかの確認結果

### 3.1 インデックス構造

**結論: episodes は RAG インデックスに含まれている。**

| ソース | ファイル | 行 | 内容 |
|-------|---------|-----|------|
| indexer | `core/memory/rag/indexer.py` | 335-336, 444-445 | `memory_type == "episodes"` 時に `_chunk_by_time_headings()` でチャンキング |
| indexer | `core/memory/rag/indexer.py` | 194 | コレクション名 `{prefix}_{memory_type}` → `{anima}_episodes` |
| manager | `core/memory/manager.py` | 282 | `append_episode()` で `index_file(path, "episodes", origin=origin)` を呼び出し |
| consolidation | `core/memory/consolidation.py` | 462-463 | 週次インデックスで `indexer.index_file(episode_file, memory_type="episodes")` |
| watcher | `core/memory/rag/watcher.py` | 129, 309 | episodes ディレクトリを監視、`memory_type_map` に episodes を定義 |

### 3.2 retriever の episodes 対応

- `core/memory/rag/retriever.py` 187行目: `collection_name = f"{anima_name}_{memory_type}"` により、`memory_type="episodes"` を渡せば `{anima}_episodes` コレクションを検索可能
- 161行目: spreading activation は `memory_type in ("knowledge", "episodes")` に対応済み

---

## 4. episodes を追加する場合の具体的な変更箇所と変更内容の提案

### 4.1 Channel C（priming.py）の変更

**方針A: 複数 memory_type を検索してマージ（推奨）**

```python
# 変更前（557-567行付近）
results = retriever.search(
    query=query,
    anima_name=anima_name,
    memory_type="knowledge",
    top_k=5,
    include_shared=True,
)

# 変更後
all_results: list = []
for mt in ("knowledge", "episodes"):
    r = retriever.search(
        query=query,
        anima_name=anima_name,
        memory_type=mt,
        top_k=3,  # 各タイプから3件、合計最大6件
        include_shared=(mt == "knowledge"),
    )
    all_results.extend(r)

# スコアでソートして top_k に絞る
all_results.sort(key=lambda r: r.score, reverse=True)
results = all_results[:5]
```

**方針B: memory_type パラメータを追加して設定で制御**

```python
# config.json の priming セクションに channel_c_memory_types: ["knowledge", "episodes"] を追加
# デフォルトは ["knowledge"] で後方互換を維持
```

### 4.2 search_memory（rag_search.py）の変更

**_resolve_search_types() の修正（217-230行）**

```python
# 変更前
if scope == "all":
    return ["knowledge", "procedures", "conversation_summary"]

# 変更後
if scope == "all":
    return ["knowledge", "episodes", "procedures", "conversation_summary"]
```

**scope="episodes" 時のベクトル検索有効化（202行付近）**

現状、`scope="episodes"` のときはベクトル検索の条件に入っていない。`_resolve_search_types("episodes")` が未定義のため `["knowledge"]` が返る。以下を追加:

```python
if scope == "episodes":
    return ["episodes"]
```

---

## 5. episodes を追加した場合の影響

### 5.1 Primingバジェットへの影響

- Channel C のバジェットは `_BUDGET_RELATED_KNOWLEDGE = 700` トークン（50行目）
- episodes を追加すると、knowledge と episodes の両方から結果が返るため、同一バジェット内で結果が増える
- **対策**: 各 memory_type の `top_k` を減らす（例: knowledge 3件 + episodes 2件）か、バジェットを微増（例: 800トークン）

### 5.2 検索結果の品質（ノイズ）

- episodes は日次ログ形式（`## HH:MM` 区切り）で、作業内容・会話要約が含まれる
- スクリプト作成等の「過去の作業」は episodes に記録されるため、追加することで想起が改善される
- ノイズ懸念: 古いエピソードが多数ヒットする可能性 → **temporal decay**（retriever.py の `_apply_score_adjustments`）で新しいドキュメントが優先されるため、過度なノイズは抑制される

### 5.3 パフォーマンス

- 追加クエリ: knowledge に加えて episodes コレクションへの検索が1回増える
- ChromaDB のベクトル検索は軽量なため、影響は小さいと想定
- spreading activation は episodes にも対応済み（retriever.py 161行）

---

## 6. まとめ

| 項目 | 結果 |
|------|------|
| Channel C の除外箇所 | `priming.py` 559行: `memory_type="knowledge"` ハードコード |
| search_memory の除外箇所 | `rag_search.py` 229行: `_resolve_search_types("all")` に episodes なし |
| RAG インデックス | episodes は `{anima}_episodes` コレクションにインデックス済み |
| 推奨変更 | (1) priming.py: knowledge + episodes の複数検索、(2) rag_search.py: `_resolve_search_types` に episodes 追加 |
