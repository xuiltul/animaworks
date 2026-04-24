# Copyright 2026 AnimaWorks
# Licensed under the Apache License, Version 2.0
"""Japanese prompts for entity / fact extraction."""

from __future__ import annotations

# ── Entity extraction ──────────────────────────────────────

ENTITY_SYSTEM = (
    "あなたは情報抽出エージェントです。"
    "与えられたテキストからエンティティ"
    "（人物・場所・組織・概念・イベント・物・時間）を"
    "JSON形式で抽出してください。"
)

ENTITY_USER = """## テキスト
{content}

## 既知のエンティティ（参考）
{previous_entities}

## 指示
上記テキストからエンティティを抽出し、以下のJSON形式で返してください。エンティティが見つからない場合は空リストを返してください。

```json
{{
  "entities": [
    {{"name": "正規化された名前", "entity_type": "Person|Place|Organization|Concept|Event|Object|Time", "summary": "1-2文の説明"}}
  ]
}}
```"""

# ── Fact extraction ────────────────────────────────────────

FACT_SYSTEM = "あなたは関係抽出エージェントです。与えられたエンティティのペア間の関係をJSON形式で抽出してください。"

FACT_USER = """## テキスト
{content}

## 抽出済みエンティティ
{entities_json}

## エッジ型（edge_type）
以下の型から最も適切なものを選んでください。該当しない場合は "RELATES_TO" を使用してください。
{edge_types_list}

## 指示
上記エンティティ間の関係（事実）を抽出し、以下のJSON形式で返してください。関係が見つからない場合は空リストを返してください。

```json
{{
  "facts": [
    {{"source_entity": "エンティティA", "target_entity": "エンティティB", "fact": "AとBの関係を自然言語で記述", "edge_type": "WORKS_AT", "valid_at": "YYYY-MM-DDTHH:MM:SS or null"}}
  ]
}}
```"""

# ── Entity deduplication ──────────────────────────────────

DEDUPE_SYSTEM = (
    "あなたはエンティティ重複判定エージェントです。新しいエンティティが既存の候補と同一かどうかを判定してください。"
)

DEDUPE_USER = """## 新規エンティティ
名前: {new_entity_name}
タイプ: {new_entity_type}
概要: {new_entity_summary}

## 既存エンティティ候補
{candidates_json}

## 指示
新規エンティティが既存候補のいずれかと同一の実体を指す場合、そのUUIDと統合サマリーをJSON形式で返してください。
同一でない場合、または判断に自信がない場合は duplicate_of_uuid を null にしてください。

```json
{{"duplicate_of_uuid": "既存のUUID or null", "merged_summary": "統合した1-2文の説明"}}
```"""

# ── Fact invalidation ─────────────────────────────────────

INVALIDATE_SYSTEM = (
    "あなたは事実の矛盾判定エージェントです。新しい事実が既存の事実と矛盾するかどうかを判定してください。"
)

INVALIDATE_USER = """## 新しい事実
{new_fact}

## 既存の事実（まだ有効）
{existing_facts_json}

## 指示
新しい事実が既存の事実のいずれかと矛盾する場合、矛盾する事実のUUIDをリストで返してください。
矛盾とは、両方が同時に真であり得ない場合を指します。補完的な情報は矛盾ではありません。
判断に自信がない場合は空リストを返してください。

```json
{{"contradicted_uuids": ["矛盾するfactのUUID", ...]}}
```"""

# ── Community summarization ───────────────────────────────

COMMUNITY_SYSTEM = "あなたはグループ分析エージェントです。関連するエンティティのグループに名前と要約を付けてください。"

COMMUNITY_USER = """## グループメンバー
{members}

## 指示
上記メンバーの共通テーマに基づいて、このグループに名前と要約を付けてください。

```json
{{"name": "グループ名（10文字以内）", "summary": "このグループの1-2文の説明"}}
```"""
