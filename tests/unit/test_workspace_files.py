from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for canonical workspace filenames and legacy identity splitting."""

from core.anima.workspace_files import (
    AGENTS_FILE,
    BOOTSTRAP_FILE,
    HEARTBEAT_FILE,
    IDENTITY_FILE,
    LEGACY_IDENTITY,
    LEGACY_RENAMES,
    SOUL_FILE,
    is_identity_section,
    split_legacy_identity,
)


def test_canonical_names_and_legacy_mapping() -> None:
    assert (IDENTITY_FILE, SOUL_FILE, AGENTS_FILE, HEARTBEAT_FILE, BOOTSTRAP_FILE) == (
        "IDENTITY.md",
        "SOUL.md",
        "AGENTS.md",
        "HEARTBEAT.md",
        "BOOTSTRAP.md",
    )
    assert LEGACY_IDENTITY == "identity.md"
    assert LEGACY_RENAMES == {
        "injection.md": "AGENTS.md",
        "heartbeat.md": "HEARTBEAT.md",
        "bootstrap.md": "BOOTSTRAP.md",
    }


def test_identity_section_keywords_are_case_insensitive_and_multilingual() -> None:
    assert is_identity_section("## 基本プロフィール")
    assert is_identity_section("## Appearance")
    assert is_identity_section("## AI MODEL")
    assert is_identity_section("## 기본 프로필")
    assert not is_identity_section("## 性格")
    assert not is_identity_section("## Personality")


def test_split_japanese_identity_moves_profile_and_personality() -> None:
    legacy = """\
# Identity: アリス

はじめまして。よろしくお願いします。

## 基本プロフィール

- 名前: アリス
- 役割: エンジニア

## 性格・キャラクター

好奇心旺盛で、丁寧。

## 外見

銀色の髪。

## AIモデル情報

- モデル: example/model
"""
    identity, soul = split_legacy_identity(legacy)

    assert identity.startswith("# Identity: アリス\n\n## 基本プロフィール")
    assert "## 基本プロフィール" in identity
    assert "## 外見" in identity
    assert "## AIモデル情報" in identity
    assert "## 性格・キャラクター" not in identity
    assert "はじめまして" not in identity

    assert soul.startswith("# Soul: アリス")
    assert "はじめまして。よろしくお願いします。" in soul
    assert "## 性格・キャラクター" in soul
    assert "好奇心旺盛で、丁寧。" in soul
    assert "## 基本プロフィール" not in soul
    assert "## 外見" not in soul

    assert identity.index("## 基本プロフィール") < identity.index("## 外見")
    assert identity.index("## 外見") < identity.index("## AIモデル情報")
    assert soul.index("はじめまして") < soul.index("## 性格・キャラクター")


def test_split_english_and_korean_profile_sections() -> None:
    english = """\
# Identity: Alice

Preamble.

## Profile

Engineer.

## Personality

Curious.

## Appearance

Silver hair.
"""
    identity, soul = split_legacy_identity(english)
    assert "## Profile" in identity
    assert "## Appearance" in identity
    assert "## Personality" not in identity
    assert "Preamble." in soul
    assert "## Personality" in soul

    korean = """\
# Identity: 앨리스

소개.

## 기본 프로필

엔지니어.

## 성격

호기심이 많다.

## 외모

은색 머리.
"""
    identity, soul = split_legacy_identity(korean)
    assert "## 기본 프로필" in identity
    assert "## 외모" in identity
    assert "## 성격" not in identity
    assert "소개." in soul
    assert "## 성격" in soul


def test_split_without_h2_keeps_title_in_identity_and_body_in_soul() -> None:
    identity, soul = split_legacy_identity("# Identity: Alice\n\nAll prose.\n")

    assert identity == "# Identity: Alice\n"
    assert soul == "# Soul: Alice\n\nAll prose.\n"


def test_split_without_title_puts_everything_in_soul() -> None:
    legacy = "Preamble.\n\n## Personality\n\nCurious.\n"
    identity, soul = split_legacy_identity(legacy)

    assert identity == ""
    assert soul == legacy


def test_split_ignores_h2_inside_code_fences() -> None:
    legacy = """\
# Identity: Alice

## Personality

```markdown
## Profile
```
"""
    identity, soul = split_legacy_identity(legacy)

    assert "## Profile" in soul
    assert "## Profile" not in identity


def test_split_adds_blank_line_between_title_and_first_profile_section() -> None:
    identity, soul = split_legacy_identity("# Identity: Alice\n## Profile\nEngineer.\n## Personality\nCurious.\n")

    assert identity.startswith("# Identity: Alice\n\n## Profile")
    assert "## Profile" in identity
    assert soul.startswith("# Soul: Alice\n\n## Personality")


def test_split_legacy_identity_keeps_front_matter_in_identity() -> None:
    from core.anima.workspace_files import split_legacy_identity

    text = "---\nrole: manager\n---\n\n# Identity: rin\n\n## 基本プロフィール\n\n名前\n\n## 性格\n\n明るい\n"
    identity, soul = split_legacy_identity(text)
    assert identity.startswith("---\nrole: manager\n---\n\n# Identity: rin\n")
    assert "名前" in identity and "明るい" in soul
    assert "role: manager" not in soul
