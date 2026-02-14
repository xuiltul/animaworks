# RAG ハイブリッド検索実装ガイド

> 実装日: 2026-02-14
> 設計ドキュメント: [priming-layer-design.md](design/implemented/priming-layer-design.md) Phase 2
> ステータス: ✅ Phase 2 実装完了

---

## 目次

1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [インストール](#インストール)
4. [使い方](#使い方)
5. [API リファレンス](#apiリファレンス)
6. [パフォーマンス](#パフォーマンス)
7. [トラブルシューティング](#トラブルシューティング)

---

## 概要

AnimaWorks の RAG (Retrieval-Augmented Generation) システムは、人間の記憶想起メカニズムにヒントを得たハイブリッド検索を提供します。

### 主要機能

- **ベクトル検索**: 意味的類似度による検索 (multilingual-e5-small)
- **BM25 検索**: キーワード完全一致検索 (ripgrep)
- **RRF 統合**: Reciprocal Rank Fusion による検索結果の統合
- **時間減衰**: 最新の記憶を優先するスコアリング
- **自動インデキシング**: 記憶書き込み時の自動的なベクトル化

### なぜハイブリッド検索か

単一の検索手法では以下の問題があります:

| 手法 | 強み | 弱み |
|------|------|------|
| ベクトル検索のみ | 意味的類似性を捉える | 固有名詞が不安定 |
| BM25のみ | 固有名詞・正確なマッチ | 意味的関連を捉えない |

ハイブリッド検索は両方の強みを組み合わせ、**検索失敗率を最大49%削減**します (Anthropic 2025, Apple ML Research 2024)。

---

## アーキテクチャ

### ディレクトリ構成

```
core/memory/rag/
├── __init__.py          # パッケージエントリポイント
├── store.py             # VectorStore抽象化 + ChromaDB実装
├── indexer.py           # MemoryIndexer (ファイル→ベクトル化)
└── retriever.py         # HybridRetriever (検索統合)
```

### データフロー

```
1. 記憶書き込み (write_knowledge, append_episode)
      ↓
2. MemoryIndexer.index_file()
      ├─ チャンク分割 (Markdown見出し単位、時刻単位)
      ├─ エンベディング生成 (multilingual-e5-small)
      └─ ChromaDB upsert
      ↓
3. ハイブリッド検索 (HybridRetriever.search())
      ├─ ベクトル検索 → ChromaDB.query()
      ├─ BM25検索 → ripgrep
      ├─ RRF統合 → スコア計算
      └─ 時間減衰 → 最終スコア
      ↓
4. プライミングレイヤー注入
```

### ChromaDB コレクション構成

```
~/.animaworks/vectordb/chroma.sqlite3

コレクション:
├── {person_name}_knowledge      # 意味記憶
├── {person_name}_episodes       # エピソード記憶
├── {person_name}_procedures     # 手続き記憶
├── {person_name}_skills         # スキル記憶
└── shared_users                 # 共有ユーザー記憶
```

### ドキュメントスキーマ

```python
{
  "id": "sakura/knowledge/chatwork-policy.md#0",
  "content": "チャンクされたテキスト",
  "embedding": [0.12, -0.34, ...],  # 384次元
  "metadata": {
    "person": "sakura",
    "memory_type": "knowledge",
    "source_file": "knowledge/chatwork-policy.md",
    "created_at": "2026-02-10T09:00:00+09:00",
    "updated_at": "2026-02-14T15:30:00+09:00",
    "importance": "normal",  # or "important"
    "tags": ["chatwork", "対応方針"],
    "chunk_index": 0,
    "total_chunks": 3
  }
}
```

---

## インストール

### 1. 依存ライブラリのインストール

```bash
# RAG機能を含む全依存をインストール
pip install -e ".[rag]"

# または個別にインストール
pip install chromadb sentence-transformers rank-bm25
```

### 2. エンベディングモデルのダウンロード

初回実行時に自動ダウンロードされます (約100MB):

```bash
# 手動で事前ダウンロードする場合
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
```

モデルは `~/.animaworks/models/` にキャッシュされます。

### 3. 初回インデキシング

```bash
# 全Personの記憶をインデキシング
animaworks index

# 特定のPersonのみ
animaworks index --person sakura

# フル再構築 (既存インデックスを削除)
animaworks index --full
```

---

## 使い方

### 自動インデキシング (推奨)

記憶書き込み時に自動的にインデキシングされます:

```python
from core.memory import MemoryManager

memory = MemoryManager(person_dir)

# 知識を書き込み → 自動的にベクトル化
memory.write_knowledge("chatwork-policy", "Chatworkの対応方針...")

# エピソードを追記 → 自動的にベクトル化
memory.append_episode("## 09:30 — 山田さんから依頼受信...")
```

**注意**: Agent SDK モード (A1) ではエージェントが直接ファイルを書き込む可能性があるため、定期的な手動インデキシングを推奨します:

```bash
# cron で毎晩実行
0 2 * * * animaworks index --person sakura
```

### 手動インデキシング

```bash
# 差分インデキシング (変更されたファイルのみ)
animaworks index --person sakura

# フル再構築
animaworks index --person sakura --full

# ドライラン (実際にはインデキシングしない)
animaworks index --dry-run
```

### ハイブリッド検索の利用

```python
from core.memory.rag import HybridRetriever, ChromaVectorStore, MemoryIndexer

# 初期化
vector_store = ChromaVectorStore()
indexer = MemoryIndexer(vector_store, "sakura", person_dir)
retriever = HybridRetriever(vector_store, indexer, knowledge_dir)

# 検索
results = retriever.search(
    query="山田さんからの緊急依頼への対応方針",
    person_name="sakura",
    memory_type="knowledge",
    top_k=3,
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content[:100]}...")
    print(f"Source: {result.metadata['source_file']}")
```

### プライミングレイヤーとの統合

プライミングレイヤーは自動的にハイブリッド検索を使用します:

```python
from core.memory.priming import PrimingEngine

engine = PrimingEngine(person_dir)

# メッセージに関連する記憶を自動想起
result = await engine.prime_memories(
    message="山田さんからChatworkで依頼が来ました",
    sender_name="yamada",
)

print(result.related_knowledge)  # ハイブリッド検索結果
```

---

## API リファレンス

### ChromaVectorStore

```python
class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: Path | None = None)
    def create_collection(self, name: str, dimension: int) -> None
    def upsert(self, collection: str, documents: list[Document]) -> None
    def query(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]
    def delete_documents(self, collection: str, ids: list[str]) -> None
```

### MemoryIndexer

```python
class MemoryIndexer:
    def __init__(
        self,
        vector_store: VectorStore,
        person_name: str,
        person_dir: Path,
        embedding_model: str = "intfloat/multilingual-e5-small",
    )

    def index_file(
        self,
        file_path: Path,
        memory_type: str,
        force: bool = False,
    ) -> int  # 返り値: インデキシングされたチャンク数

    def index_directory(
        self,
        directory: Path,
        memory_type: str,
        force: bool = False,
    ) -> int
```

#### チャンク分割戦略

| 記憶タイプ | 分割方法 | 理由 |
|-----------|---------|------|
| knowledge | Markdown見出し単位 (`## ...`) | 1つの知識は1見出しに完結 |
| episodes | 時刻見出し単位 (`## HH:MM`) | 1つのエピソードは時刻ブロック |
| procedures | ファイル全体 | 手順は分割すると意味を失う |
| skills | ファイル全体 | スキルは原子的単位 |

### HybridRetriever

```python
class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        indexer: MemoryIndexer,
        knowledge_dir: Path,
    )

    def search(
        self,
        query: str,
        person_name: str,
        memory_type: str = "knowledge",
        top_k: int = 3,
    ) -> list[RetrievalResult]
```

#### スコアリング

```python
# RRF (Reciprocal Rank Fusion)
RRF_score(d) = Σ 1 / (k + rank_i(d))
# k = 60 (標準値)

# 重み付け統合
final_score = 0.5 * vector_score + 0.3 * bm25_score + 0.2 * recency_score

# 時間減衰 (半減期: 30日)
recency_score = 0.5 ^ (age_days / 30.0)
```

---

## パフォーマンス

### ベンチマーク (参考値)

| 操作 | レイテンシ | 備考 |
|------|-----------|------|
| エンベディング生成 | ~10ms/doc | CPU: multilingual-e5-small |
| ベクトル検索 | ~50ms | 1000チャンク、top_k=10 |
| BM25検索 | ~20ms | ripgrep, 100ファイル |
| ハイブリッド検索 (全体) | ~100ms | ベクトル + BM25 + RRF |
| 差分インデキシング | ~50ms/file | 変更検出 + 再インデキシング |

### ストレージサイズ

- エンベディング: 約1.5KB/チャンク (384次元 × 4 bytes)
- ChromaDB オーバーヘッド: 約30%
- 例: 1000チャンク → 約2MB

### 最適化のヒント

1. **インデキシング頻度の調整**
   - リアルタイム: 書き込み時に自動インデキシング (推奨)
   - バッチ: 深夜cronで一括インデキシング (負荷分散)

2. **検索パラメータのチューニング**
   ```python
   # 精度重視
   results = retriever.search(query, top_k=10)

   # 速度重視
   results = retriever.search(query, top_k=3)
   ```

3. **エンベディングモデルの選択**
   - `multilingual-e5-small` (384次元): 高速、CPU対応 ← **Phase 2推奨**
   - `multilingual-e5-large` (1024次元): 高精度、GPU推奨
   - `BGE-M3` (1024次元): 最高精度、GPU必須

---

## トラブルシューティング

### インデックスが作成されない

**症状**: `animaworks index` を実行してもエラーなく終了するが、検索結果が返ってこない

**原因と対処**:

1. **依存ライブラリ未インストール**
   ```bash
   pip install 'animaworks[rag]'
   ```

2. **エンベディングモデルのダウンロード失敗**
   ```bash
   # ログを確認
   animaworks index --person sakura 2>&1 | grep -i error

   # 手動ダウンロードを試行
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
   ```

3. **ファイルパスの問題**
   ```bash
   # 記憶ディレクトリの存在確認
   ls -la ~/.animaworks/persons/sakura/knowledge/
   ```

### 検索結果がおかしい

**症状**: 関連性の低い記憶が上位に来る

**原因と対処**:

1. **インデックスが古い**
   ```bash
   # フル再構築
   animaworks index --person sakura --full
   ```

2. **BM25が機能していない (ripgrep未インストール)**
   ```bash
   # ripgrepをインストール
   # Ubuntu/Debian
   sudo apt install ripgrep

   # macOS
   brew install ripgrep
   ```

3. **クエリが短すぎる**
   ```python
   # NG: 検索キーワードが1語のみ
   results = retriever.search("Chatwork", ...)

   # OK: 文脈を含むクエリ
   results = retriever.search("Chatwork での山田さんからの依頼対応", ...)
   ```

### メモリ不足エラー

**症状**: `MemoryError` または `RuntimeError: CUDA out of memory`

**原因と対処**:

1. **エンベディングモデルがGPUメモリを消費**
   ```python
   # CPU強制モード
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = ""

   # またはモデル切り替え
   indexer = MemoryIndexer(
       vector_store,
       person_name,
       person_dir,
       embedding_model="intfloat/multilingual-e5-small",  # 軽量モデル
   )
   ```

2. **ChromaDBのバッチサイズ調整**
   ```python
   # 一度に大量のドキュメントをupsertしない
   # 1000チャンクずつに分割
   for i in range(0, len(documents), 1000):
       batch = documents[i:i+1000]
       vector_store.upsert(collection, batch)
   ```

### プライミングで検索が使われない

**症状**: プライミングレイヤーがBM25フォールバックを使用している

**ログ例**:
```
Channel C: RAG not installed, falling back to BM25-only
```

**原因と対処**:

1. **RAGライブラリのインポートエラー**
   ```bash
   # インストール確認
   python -c "from core.memory.rag import HybridRetriever; print('OK')"
   ```

2. **ChromaDB初期化エラー**
   ```bash
   # ログで詳細確認
   tail -f ~/.animaworks/logs/animaworks.log | grep -i chroma
   ```

3. **インデックス未作成**
   ```bash
   # 初回インデキシング
   animaworks index --person sakura
   ```

---

## 今後の拡張 (Phase 3)

- 知識グラフ構築 (エンティティ・リレーション抽出)
- Personalized PageRank による多段階拡散活性化
- エンベディングモデルのアップグレード (multilingual-e5-large, BGE-M3)
- 動的予算調整 (クエリタイプに応じたトークン予算の変動)
- ファイル監視による非同期インデキシング (watchdog)

---

## 参考文献

- Anthropic (2025). "Contextual Retrieval"
- Apple ML Research (2024). "Dense-sparse hybrid search reduces failure rate by 49%"
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Cormack et al. (2009). "Reciprocal Rank Fusion outperforms individual systems"
