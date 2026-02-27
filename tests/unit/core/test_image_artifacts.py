"""Unit tests for core/image_artifacts.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from core.image_artifacts import extract_image_artifacts_from_tool_records


def test_extract_image_artifacts_from_nested_payload():
    records = [
        {
            "tool_name": "web_search",
            "result_summary": {
                "results": [
                    {"title": "a", "image_url": "https://images.unsplash.com/photo-1.png"},
                    {"title": "b", "thumbnail": "https://cdn.search.brave.com/img.webp"},
                ]
            },
        }
    ]

    artifacts = extract_image_artifacts_from_tool_records(records)

    assert len(artifacts) == 2
    assert artifacts[0]["type"] == "image"
    assert artifacts[0]["source"] == "searched"
    assert artifacts[0]["trust"] == "untrusted"


def test_extract_image_artifacts_marks_image_gen_as_generated():
    records = [
        {
            "tool_name": "image_gen",
            "result_summary": "Saved to assets/avatar_fullbody.png",
        }
    ]

    artifacts = extract_image_artifacts_from_tool_records(records)

    assert len(artifacts) == 1
    assert artifacts[0]["source"] == "generated"
    assert artifacts[0]["trust"] == "trusted"
    assert artifacts[0]["path"] == "assets/avatar_fullbody.png"

