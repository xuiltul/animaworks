# 調査: mikotoが「思い出すのに時間がかかった」原因

**日付**: 2026-03-04  
**対象**: AnimaWorks Anima「mikoto」が「sakuraのシステムプロンプトを解析してHTMLにするスクリプト」の質問に回答するまで時間がかかった事象

---

## 1. 遅延の原因（31分の遅延）

### タイムライン（activity_log 2026-03-04.jsonl より）

| 時刻 | イベント | 詳細 |
|------|----------|------|
| 18:04:21 | message_received | taka: 「mikotoに実装してもらった、sakuraのシステムプロンプトを解析するスクリプトってどれだっけ？」 |
| 18:04:21 | **error** | `ModuleNotFoundError: No module named 'core.tooling.prompt_db'` |
| 18:09:00 | heartbeat_start | 定期巡回開始 |
| 18:09:00 | **error** | `FileNotFoundError: Template not found: prompts/task_delegation_rules.md` |
| 18:14:39 | **error** | 応答が中断されました（前回セッションの未完了ストリームを回復, chat）— recovered_chars: 0 |
| 18:35:34 | message_received | **同じメッセージが再受信**（リトライ） |
| 18:35:52〜 | tool_use | search_memory, Grep 等で探索開始 |
| 18:36:xx | response_sent | 初回応答（dump_prompt.py を誤答、その後 debug_system_prompt.py を発見） |

### 結論: 遅延の主因は**処理失敗**であり、mikotoの「思い出す速度」ではない

1. **18:04 に即座に処理失敗**: メッセージ受信直後に `ModuleNotFoundError: No module named 'core.tooling.prompt_db'` が発生し、応答パイプラインが起動できなかった。
2. **18:14 のリカバリ試行も失敗**: 未完了ストリームの回復が試みられたが `recovered_chars: 0` で実質何も復旧していない。
3. **18:35 の再送で初めて正常処理**: 同一メッセージが再受信され、その時点で正常に処理が開始された。

**31分の遅延は、mikotoが「思い出すのに時間がかかった」のではなく、最初のリクエストがサーバー側エラーで即失敗し、再送まで応答が返らなかったことが原因。**

---

## 2. RAG/Priming がヒットしなかった理由

### 2.1 記憶の保存状態

- **episodes/2026-03-02.md** の「18:39 — プロンプトデバッグツール作成と修正」セクションに、`scripts/debug_system_prompt.py` の作成記録が**明確に存在**している。
- **knowledge/** にはスクリプト作成に関する専用ファイルが**ない**（consolidation で episode → knowledge への統合が行われていない）。
- **index_meta.json** によると `episodes/2026-03-02.md` は RAG にインデックス済み（18 chunks）。

### 2.2 Priming Channel C の検索範囲

`core/memory/priming.py` の `_channel_c_related_knowledge()` は:

```python
results = retriever.search(
    query=query,
    anima_name=anima_name,
    memory_type="knowledge",  # ← knowledge のみ！
    top_k=5,
    include_shared=True,
)
```

**Channel C（related_knowledge）は `memory_type="knowledge"` のみを検索しており、episodes は検索対象に含まれていない。**

### 2.3 search_memory の scope="all" のベクトル検索

`core/memory/rag_search.py` の `_resolve_search_types()`:

```python
if scope == "all":
    return ["knowledge", "procedures", "conversation_summary"]  # episodes なし！
```

**scope="all" でもベクトル検索の memory_type に `episodes` が含まれていない。**

キーワード検索では `episodes_dir` が対象に含まれるが、条件は `q in line.lower()`（クエリ全体が1行に含まれる必要がある）。「sakura システムプロンプト 解析 スクリプト」というクエリが1行にそのまま含まれている行は episodes 内に存在しないため、キーワード検索でもヒットしなかった。

### 2.4 まとめ: RAG/Priming 未ヒットの原因

| 要因 | 説明 |
|------|------|
| Priming Channel C | episodes を検索していない（knowledge のみ） |
| search_memory ベクトル検索 | scope="all" でも episodes が memory_type に含まれていない |
| キーワード検索 | `q in line` の完全包含条件のため、分散した記述ではヒットしない |
| knowledge への統合 | 該当エピソードが knowledge に consolidation されていない |

---

## 3. 改善提案

### 3.1 即時対応（エラー解消）

- **ModuleNotFoundError (prompt_db)**: 18:04 時点で発生したエラーの原因を特定し、同様のインポート失敗が起きないよう修正する。
- **task_delegation_rules.md 不在**: ハートビート用テンプレートの存在確認と、不足時のフォールバックを検討する。

### 3.2 Priming/RAG の改善

1. **Priming Channel C に episodes を追加**
   - `_channel_c_related_knowledge` で `memory_type` に `episodes` を追加するか、episodes を別チャネルとして検索し、結果をマージする。

2. **search_memory の scope="all" に episodes を追加**
   - `_resolve_search_types("all")` の戻り値に `"episodes"` を追加する。

3. **キーワード検索の緩和**
   - `q in line.lower()` の完全包含ではなく、クエリを単語分割して「いずれかの単語が含まれる」などの OR 条件を検討する（検索ノイズとのトレードオフに注意）。

### 3.3 記憶の構造改善

1. **重要な実装成果の knowledge 化**
   - スクリプト作成など、後から参照されやすい成果物は、episode に留めず `knowledge/` に明示的に記録する運用を検討する。
   - 例: `knowledge/debug-system-prompt-script.md` のようなファイルを、作成時に手動または半自動で追加する。

2. **Consolidation の強化**
   - episode の「ツール/スクリプト作成」のようなパターンを検出し、knowledge への自動統合を促すルールを検討する。

### 3.4 運用上のヒント

- 「以前作ってもらった〇〇」のような質問では、`search_memory` を `scope="episodes"` で明示的に呼ぶよう、プロンプトで誘導する。
- 重要な成果物（スクリプト、手順書など）は、作成完了時に `knowledge/` への要約記録を habit として組み込む。

---

## 付録: 参照ファイル

- `~/.animaworks/animas/mikoto/activity_log/2026-03-04.jsonl` (行 3143〜3164 付近)
- `~/.animaworks/animas/mikoto/episodes/2026-03-02.md` (18:39 セクション)
- `core/memory/priming.py` (`_channel_c_related_knowledge`)
- `core/memory/rag_search.py` (`_resolve_search_types`, `search_memory_text`)
