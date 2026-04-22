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

## 指示
上記エンティティ間の関係（事実）を抽出し、以下のJSON形式で返してください。関係が見つからない場合は空リストを返してください。

```json
{{
  "facts": [
    {{"source_entity": "エンティティA", "target_entity": "エンティティB", "fact": "AとBの関係を自然言語で記述", "valid_at": "YYYY-MM-DDTHH:MM:SS or null"}}
  ]
}}
```"""
