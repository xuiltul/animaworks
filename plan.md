# 記憶蒸留リファクタリング: ワンショット → Anima自律ツールコールループ

## 概要

現在のConsolidationEngineが行っている「ワンショットLLM呼び出し + 正規表現パース」を廃止し、
Animaの既存実行パイプライン（`run_cycle()` のツールコールループ）を使って
Anima自身に記憶整理を行わせる。

**設計思想**: 「記憶の整理」はAnima自身のタスクであり、外部エンジンがLLMを代理呼び出しするのではなく、
Animaに「あなたの記憶を整理して」と指示して、ツールを使って自律的に作業させるべき。

## 実行モード互換性

全3モードが `run_cycle()` のツールコールループを持っており、すべて対応可能:

| Mode | Executor | ツールコールループ | max_turns制御 |
|------|----------|-------------------|--------------|
| A1 | AgentSDKExecutor | Agent SDK subprocess | `ClaudeAgentOptions(max_turns=)` |
| A1 Fallback | AnthropicFallbackExecutor | Anthropic SDK直接 | `for iteration in range(max_iterations)` |
| A2 | LiteLLMExecutor | LiteLLM + tool_use | `for iteration in range(max_iterations)` |
| B | AssistedExecutor | テキストベース疑似ツールコール | `for iteration in range(max_iterations)` |

**Mode B の疑似ツールコール**: LLMがテキストでツール呼び出しを出力 → `extract_tool_call()` で
パース → `ToolHandler.handle()` で実行 → 結果をメッセージに注入 → ループ継続。
`max_turns` で回数制限される点は他モードと同一。

全モードで `ToolHandler` が共通のため、`archive_memory_file` を含む全メモリツールが
どのモードでも利用可能。Animaの自身のモデル・実行モードをそのまま使う。

## 変更ファイル一覧

### 新規作成
1. `templates/prompts/memory/consolidation_instruction.md` — 日次統合指示プロンプト
2. `templates/prompts/memory/weekly_consolidation_instruction.md` — 週次統合指示プロンプト

### 変更
3. `core/tooling/schemas.py` — `archive_memory_file` ツールスキーマ追加
4. `core/tooling/handler.py` — `_handle_archive_memory_file()` 実装追加
5. `core/tooling/prompt_db.py` — `archive_memory_file` ガイド追加
6. `core/config/models.py` — `ConsolidationConfig.max_turns` 追加（デフォルト30）
7. `core/agent.py` — `run_cycle()` に `max_turns_override` パラメータ追加
8. `core/execution/litellm_loop.py` — `max_turns_override` の受け渡し
9. `core/execution/agent_sdk.py` — 同上
10. `core/execution/anthropic_fallback.py` — 同上
11. `core/execution/assisted.py` — 同上（Mode B テキストベースループも対応）
12. `core/execution/base.py` — `execute()` シグネチャに `max_turns_override` 追加
13. `core/lifecycle.py` — `_handle_daily_consolidation` / `_handle_weekly_integration` を書き換え
14. `core/memory/consolidation.py` — LLM呼び出し系メソッドをすべて削除、前処理・後処理のみ残す
15. `core/anima.py` — `run_consolidation()` メソッド追加

### 変更不要（そのまま残す）
- `core/memory/forgetting.py` — メタデータベース操作（LLM不要）
- `core/memory/reconsolidation.py` — Anima自身がツールで記憶を修正する中で自然にカバーされるが、
  当面は forgetting パイプラインの一部として残す
- `core/memory/distillation.py` — 同上（Animaのループ内で手続き知識の抽出もカバー）

## 詳細設計

### Step 1: `archive_memory_file` ツール追加

Animaが不要な記憶をアーカイブするためのツール。既存の `write_memory_file` では
ファイルの削除/移動ができないため必要。

**schemas.py に追加:**
```python
{
    "name": "archive_memory_file",
    "description": (
        "Archive a memory file (knowledge, procedures) that is no longer needed. "
        "The file is moved to archive/ directory, not permanently deleted. "
        "Use this to clean up stale, outdated, or redundant memory files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path within anima dir (e.g. 'knowledge/old-info.md')",
            },
            "reason": {
                "type": "string",
                "description": "Reason for archiving (e.g. 'superseded by new-info.md')",
            },
        },
        "required": ["path", "reason"],
    },
}
```

**handler.py の実装:**
- `archive/superseded/` にファイルを移動（既存パターンと同一）
- `identity.md`, `injection.md` 等の保護ファイルはブロック
- アクティビティログに記録

### Step 2: `max_turns_override` の追加

統合タスクでは通常の `max_turns`（20）ではなく30ターンが必要。

**変更チェーン:**
1. `AgentCore.run_cycle(prompt, ..., max_turns_override=None)` — 新パラメータ追加
2. `AgentCore._run_cycle_inner()` — override値を各executor呼び出しに伝搬
3. `BaseExecutor.execute()` — `max_turns_override: int | None = None` パラメータ追加
4. `LiteLLMExecutor`: `max_iterations = max_turns_override or self._model_config.max_turns`
5. `AgentSDKExecutor`: `ClaudeAgentOptions(max_turns=override or config.max_turns)`
6. `AnthropicFallbackExecutor`: 同様（A2と同じパターン）
7. `AssistedExecutor`: 同様（`max_iterations = max_turns_override or self._model_config.max_turns`）
   — Mode Bもテキストベースツールコールループを持つため、全く同じ方法で制御可能

**ConsolidationConfig:**
```python
class ConsolidationConfig(BaseModel):
    # 既存フィールド...
    max_turns: int = 30  # 統合タスクのツールコールループ上限
```

### Step 3: 統合指示プロンプトテンプレート

`templates/prompts/memory/consolidation_instruction.md`:

```markdown
# 記憶統合タスク（日次）

あなたの記憶を整理する時間です。以下の手順で作業してください。

## 今日のエピソード

{episodes_summary}

## 作業手順

### 1. エピソード確認
上記の本日のエピソードを確認してください。

### 2. 既存知識との照合
`search_memory` を使って、今日のエピソードに関連する既存の knowledge/ と procedures/ を検索してください。

### 3. 知識の更新・作成
- **既存ファイルの更新**: 今日の経験で更新すべき知識があれば `read_memory_file` で読み、
  `write_memory_file` で追記・修正してください
- **新規知識の作成**: 新しいパターンや教訓があれば `write_memory_file` で
  knowledge/ に新規ファイルを作成してください
- **手続き知識**: 繰り返し使える手順やワークフローは procedures/ に記録してください

### 4. 不要な記憶の整理
- 重複している knowledge/ ファイルがあれば統合し、古い方を `archive_memory_file` でアーカイブ
- 内容が古くなった procedures/ があれば更新するかアーカイブ
- 矛盾する知識があれば、より正確な方を残して古い方をアーカイブ

### 5. 品質チェック
- 作成・更新した知識ファイルが今日のエピソードの事実と矛盾していないか確認
- ファイル名はトピックを表すわかりやすい名前にすること（英数字とハイフン推奨）

## 注意事項
- 挨拶のみの会話や実質的な情報を含まないやり取りは知識化不要
- 具体的な設定値・ID・手順は必ず記録
- [REFLECTION] タグ付きエントリは本人が意識的に記録した洞察。優先的に知識化を検討
- knowledge/ のファイルにはYAMLフロントマターを付けないこと（システムが自動付与）
- 完了後、実施した内容のサマリーを出力してください
```

`templates/prompts/memory/weekly_consolidation_instruction.md`:
```markdown
# 記憶統合タスク（週次）

1週間分の記憶を整理する時間です。

## 作業手順

### 1. 知識ファイルの重複チェック
`list_directory` で knowledge/ 内のファイルを確認し、
`search_memory` と `read_memory_file` で類似・重複ファイルを探してください。
重複があれば統合し、古い方を `archive_memory_file` でアーカイブしてください。

### 2. 手続き知識の整理
`list_directory` で procedures/ 内のファイルを確認し、
古くなった手順や使われていない手順をアーカイブしてください。

### 3. 古いエピソードの圧縮
30日以上前のエピソードで [IMPORTANT] タグがないものは、
要点のみに圧縮して `write_memory_file` で上書きしてください。

### 4. 知識の矛盾解消
矛盾する知識ファイルがないか確認し、正確な方を残してください。

完了後、実施した内容のサマリーを出力してください。
```

### Step 4: `Anima.run_consolidation()` メソッド追加

```python
async def run_consolidation(
    self,
    consolidation_type: str = "daily",
    max_turns: int = 30,
) -> CycleResult:
    """Run memory consolidation as an Anima-driven task.

    Instead of external LLM calls, the Anima itself uses its tools
    to organize, consolidate, and clean up its memories.
    """
    async with self._lock:
        self._status = "consolidating"
        self._current_task = f"Memory consolidation ({consolidation_type})"

        # Pre-collect episodes for injection into prompt
        episodes_summary = self._collect_episodes_summary()

        # Load prompt template
        prompt = load_prompt(
            f"memory/{consolidation_type}_consolidation_instruction",
            episodes_summary=episodes_summary,
            anima_name=self.name,
        )

        self._activity.log(
            "consolidation_start",
            summary=f"{consolidation_type}記憶統合開始",
        )

        try:
            result = await self.agent.run_cycle(
                prompt,
                trigger=f"consolidation:{consolidation_type}",
                message_intent="request",
                max_turns_override=max_turns,
            )

            self._activity.log(
                "consolidation_end",
                summary=f"{consolidation_type}記憶統合完了",
                content=result.summary[:500],
                meta={"duration_ms": result.duration_ms},
            )

            return result
        except Exception as exc:
            self._activity.log(
                "error",
                summary=f"記憶統合エラー: {type(exc).__name__}",
            )
            raise
        finally:
            self._status = "idle"
            self._current_task = ""
```

`_collect_episodes_summary()` は現在の `ConsolidationEngine._collect_recent_episodes()` の
ロジックを簡略化したもの。プロンプトに注入するためのエピソード要約テキストを生成。

### Step 5: `lifecycle.py` の書き換え

```python
async def _handle_daily_consolidation(self) -> None:
    """Run daily consolidation for all animas."""
    from core.config import load_config
    config = load_config()
    consolidation_cfg = getattr(config, "consolidation", None)

    if consolidation_cfg and not consolidation_cfg.daily_enabled:
        return

    max_turns = getattr(consolidation_cfg, "max_turns", 30)

    for anima_name, anima in self.animas.items():
        try:
            # Skip if no recent episodes
            episode_count = anima.count_recent_episodes(hours=24)
            if episode_count < getattr(consolidation_cfg, "min_episodes_threshold", 1):
                continue

            result = await anima.run_consolidation(
                consolidation_type="daily",
                max_turns=max_turns,
            )

            # Post-processing: synaptic downscaling (metadata-based, no LLM)
            from core.memory.forgetting import ForgettingEngine
            forgetter = ForgettingEngine(anima.memory.anima_dir, anima_name)
            forgetter.synaptic_downscaling()

            # Post-processing: RAG index rebuild
            from core.memory.consolidation import ConsolidationEngine
            engine = ConsolidationEngine(anima.memory.anima_dir, anima_name)
            engine._rebuild_rag_index()

            # WebSocket broadcast
            if self._ws_broadcast:
                await self._ws_broadcast({...})

        except Exception:
            logger.exception("Daily consolidation failed for %s", anima_name)
```

### Step 6: `consolidation.py` の簡素化

**削除するメソッド（LLMワンショット呼び出し系）:**
- `_summarize_episodes()` — Animaのループが代替
- `_validate_consolidation()` — Animaが検証しながら書く
- `_merge_to_knowledge()` — Animaが `write_memory_file` で直接書く
- `_run_validity_review()`, `_select_review_candidates()`, `_process_review_verdicts()` — Animaが自律判断
- `_run_contradiction_check()` — Animaがループ内で矛盾チェック
- `_run_reconsolidation()`, `_run_knowledge_reconsolidation()` — Animaが直接修正
- `_run_resolved_to_procedure()` — Animaがepisodeから手続き抽出
- `_merge_knowledge_files()` — Animaが `read_memory_file` + `write_memory_file` + `archive_memory_file`
- `_compress_old_episodes()` — Animaが `read_memory_file` + `write_memory_file`
- `_detect_duplicates()` — Animaが `search_memory` で発見
- `_sanitize_llm_output()` — 不要に
- `_filter_entries_by_text()` — 不要に
- `_fetch_related_knowledge()` — Animaが `search_memory` で代替
- `daily_consolidate()` — 大部分削除（エントリポイントとしてのみ残すか、完全にAnima側に移行）
- `weekly_integrate()` — 同上

**残すメソッド（メタデータ/ファイル操作系）:**
- `__init__()` — パス初期化
- `_collect_recent_episodes()` — エピソード収集（プロンプト注入用）
- `_collect_resolved_events()` — 解決イベント収集
- `_migrate_legacy_knowledge()` — レガシーマイグレーション
- `_list_knowledge_files()` — ファイル一覧
- `_update_rag_index()` — RAGインデックス更新
- `_rebuild_rag_index()` — RAGインデックス再構築
- `monthly_forget()` — 月次忘却（メタデータベース）

## 実装順序

1. **Step 1**: `archive_memory_file` ツール追加（schemas.py, handler.py, prompt_db.py）
2. **Step 2**: `max_turns_override` パラメータ追加（base.py → 各executor → agent.py）
3. **Step 3**: プロンプトテンプレート作成（2ファイル）
4. **Step 4**: `ConsolidationConfig.max_turns` 追加 + `Anima.run_consolidation()` 追加
5. **Step 5**: `lifecycle.py` の書き換え
6. **Step 6**: `consolidation.py` の簡素化（LLM系メソッド削除）
7. **Step 7**: テスト追加・既存テスト修正

## テスト戦略

- `archive_memory_file` ツールの単体テスト（保護ファイルブロック、正常移動）
- `max_turns_override` がexecutorに正しく伝搬するテスト
- `run_consolidation()` の統合テスト（モック executor でプロンプトと trigger を検証）
- 既存の consolidation テストから LLM ワンショット系を削除
