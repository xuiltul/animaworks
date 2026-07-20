from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tool-schema format converters."""

from typing import Any

from core.i18n import t as _t


def to_anthropic_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical schemas to Anthropic API format (``input_schema``)."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def to_litellm_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical schemas to LiteLLM/OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def to_text_format(
    schemas: list[dict[str, Any]],
    *,
    locale: str | None = None,
) -> str:
    """Convert canonical tool schemas to text specification for Mode B.

    Generates a markdown-formatted tool guide that instructs the LLM to
    output tool calls as JSON code blocks.  Used by ``AssistedExecutor``
    to inject tool specifications into the system prompt.

    Includes imperative instructions, few-shot examples, and
    anti-hallucination rules to maximise tool-call compliance from
    weaker models.
    """
    header = _t("schema.text_format.header")
    instruction = _t("schema.text_format.instruction")
    example = _t("schema.text_format.example")
    rules = [
        _t("schema.text_format.rule_wait"),
        _t("schema.text_format.rule_plain_text"),
        _t("schema.text_format.rule_one_call"),
        _t("schema.text_format.rule_no_fabricate"),
        _t("schema.text_format.rule_no_empty_promise"),
    ]
    fewshot_header = _t("schema.text_format.fewshot_header")
    fewshot_items = [
        (
            _t("schema.text_format.fewshot1_prompt"),
            '```json\n{"tool": "Bash", "arguments": {"command": "docker ps"}}\n```',
        ),
        (
            _t("schema.text_format.fewshot2_prompt"),
            '```json\n{"tool": "Bash", "arguments": {"command": "free -h"}}\n```',
        ),
    ]
    args_label = _t("schema.text_format.args_label")
    required_label = _t("schema.text_format.required_label")
    tools_header = _t("schema.text_format.tools_header")

    lines = [
        header,
        "",
        instruction,
        "",
        "```json",
        example,
        "```",
        "",
    ]
    for rule in rules:
        lines.append(f"- {rule}")
    lines.append("")

    # Few-shot examples
    lines.append(fewshot_header)
    lines.append("")
    for prompt_ex, call_ex in fewshot_items:
        lines.append(prompt_ex)
        lines.append("")
        lines.append(call_ex)
        lines.append("")

    # Tool list
    lines.append(tools_header)
    lines.append("")
    for schema in schemas:
        name = schema["name"]
        desc = schema.get("description", "")
        params = schema.get("parameters", {}).get("properties", {})
        required = set(schema.get("parameters", {}).get("required", []))
        args_parts = []
        for k, v in params.items():
            type_str = v.get("type", "?")
            req_str = f" {required_label}" if k in required else ""
            args_parts.append(f"{k}: {type_str}{req_str}")
        args_desc = ", ".join(args_parts)
        lines.append(f"- **{name}**: {desc}")
        if args_desc:
            lines.append(f"  - {args_label}: {args_desc}")
    return "\n".join(lines)
