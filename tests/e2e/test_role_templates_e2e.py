# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Role Templates + Ability Scores feature.

Tests the full pipeline from character sheet creation with role specification
through to config resolution and system prompt injection.  Exercises
create_from_md, MemoryManager, resolve_anima_config, and build_system_prompt
with isolated filesystem fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.anima_factory import create_from_md
from core.config import (
    AnimaModelConfig,
    invalidate_cache,
    load_config,
    resolve_anima_config,
    save_config,
)
from core.memory import MemoryManager
from core.prompt.builder import build_system_prompt

# ── Character sheets ────────────────────────────────────────

ENGINEER_SHEET = """\
# Character: testeng

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | testeng |
| 上司 | (なし) |
| 実行モード | autonomous |

## 人格

テスト用エンジニア人格です。

## 役割・行動方針

テスト用の行動方針です。
"""

GENERAL_SHEET = """\
# Character: testgen

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | testgen |
| 上司 | (なし) |
| 実行モード | autonomous |

## 人格

テスト用汎用人格です。

## 役割・行動方針

汎用的な行動方針です。
"""

RESEARCHER_SHEET = """\
# Character: testres

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | testres |
| 上司 | (なし) |
| 実行モード | autonomous |

## 人格

テスト用リサーチャー人格です。

## 役割・行動方針

リサーチ業務を担当します。
"""


# ── Helpers ─────────────────────────────────────────────────


def _write_sheet(directory: Path, content: str, filename: str = "sheet.md") -> Path:
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


# ── Test 1: create_from_md with role="engineer" ────────────


class TestCreateFromMdEngineerRole:
    """Verify that role='engineer' applies engineer template files and defaults."""

    def test_specialty_prompt_created_with_engineer_content(
        self, data_dir: Path, tmp_path: Path
    ):
        """specialty_prompt.md should be created with engineer-specific content."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, ENGINEER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, role="engineer")

        specialty = anima_dir / "specialty_prompt.md"
        assert specialty.exists(), "specialty_prompt.md was not created"
        content = specialty.read_text(encoding="utf-8")
        # Engineer specialty prompt contains coding-specific guidance
        assert "エンジニア" in content or "コーディング" in content, (
            f"Engineer specialty content not found.  Got: {content[:200]}"
        )

    def test_permissions_created_with_engineer_content(
        self, data_dir: Path, tmp_path: Path
    ):
        """permissions.md should be overwritten with engineer template content."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, ENGINEER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, role="engineer")

        permissions = anima_dir / "permissions.md"
        assert permissions.exists(), "permissions.md was not created"
        content = permissions.read_text(encoding="utf-8")
        # Engineer permissions template includes tool list
        assert "Permissions" in content or "ツール" in content
        # The {name} placeholder should be replaced with the anima name
        assert "testeng" in content

    def test_status_json_contains_engineer_defaults(
        self, data_dir: Path, tmp_path: Path
    ):
        """status.json should contain model config values from engineer defaults.json."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, ENGINEER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, role="engineer")

        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        # Engineer defaults.json specifies these values
        assert status["model"] == "claude-opus-4-6"
        assert status["max_turns"] == 200
        assert status["max_chains"] == 10
        assert status["context_threshold"] == 0.80
        assert status["conversation_history_threshold"] == 0.40
        assert status["role"] == "engineer"

    def test_status_json_sheet_model_overrides_role_default(
        self, data_dir: Path, tmp_path: Path
    ):
        """If the character sheet specifies a model, it should override the role default."""
        sheet_with_model = """\
# Character: testeng2

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | testeng2 |
| モデル | claude-sonnet-4-6 |
| 上司 | (なし) |
| 実行モード | autonomous |

## 人格

テスト用エンジニアです。

## 役割・行動方針

エンジニアの行動方針です。
"""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, sheet_with_model)

        anima_dir = create_from_md(animas_dir, sheet_path, role="engineer")

        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        # Character sheet model should override role default
        assert status["model"] == "claude-sonnet-4-6"
        # But other role defaults should still be applied
        assert status["max_turns"] == 200
        assert status["max_chains"] == 10


# ── Test 2: create_from_md with role="general" (default) ───


class TestCreateFromMdGeneralRole:
    """Verify that omitting role defaults to 'general' template."""

    def test_default_role_is_general(self, data_dir: Path, tmp_path: Path):
        """When no role is specified, role should default to 'general'."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, GENERAL_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        assert status["role"] == "general"

    def test_general_specialty_prompt_created(self, data_dir: Path, tmp_path: Path):
        """specialty_prompt.md should contain general guidance content."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, GENERAL_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        specialty = anima_dir / "specialty_prompt.md"
        assert specialty.exists(), "specialty_prompt.md was not created"
        content = specialty.read_text(encoding="utf-8")
        assert "汎用" in content, (
            f"General specialty content not found.  Got: {content[:200]}"
        )

    def test_general_status_json_has_general_defaults(
        self, data_dir: Path, tmp_path: Path
    ):
        """status.json should contain values from general/defaults.json."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, GENERAL_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        # general defaults.json values
        assert status["model"] == "claude-sonnet-4-6"
        assert status["max_turns"] == 20
        assert status["max_chains"] == 2
        assert status["context_threshold"] == 0.50
        assert status["conversation_history_threshold"] == 0.30


# ── Test 3: create_from_md with role="researcher" ──────────


class TestCreateFromMdResearcherRole:
    """Verify researcher role defaults are applied correctly."""

    def test_researcher_defaults_in_status_json(
        self, data_dir: Path, tmp_path: Path
    ):
        """status.json should contain values from researcher/defaults.json."""
        from pathlib import Path as _P
        import importlib.resources

        # Read expected values from the template itself
        template_defaults_path = (
            _P(__file__).resolve().parent.parent.parent
            / "templates" / "_shared" / "roles" / "researcher" / "defaults.json"
        )
        expected = json.loads(template_defaults_path.read_text(encoding="utf-8"))

        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, RESEARCHER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, role="researcher")

        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        assert status["model"] == expected["model"]
        assert status["max_turns"] == expected["max_turns"]
        assert status["max_chains"] == expected["max_chains"]
        assert status["context_threshold"] == expected["context_threshold"]
        assert status["conversation_history_threshold"] == expected["conversation_history_threshold"]
        assert status["role"] == "researcher"


# ── Test 4: 2-layer config resolution (status.json SSoT) ────


class TestTwoLayerConfigResolution:
    """Verify the 2-layer config priority: status.json >> anima_defaults."""

    def test_status_json_overrides_global_defaults(
        self, data_dir: Path, tmp_path: Path
    ):
        """Layer 1 (status.json/role defaults) should override layer 2 (global defaults)."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, ENGINEER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, role="engineer")

        invalidate_cache()

        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()

        # Engineer role defaults in status.json should override global defaults
        assert model_config.model == "claude-opus-4-6"
        assert model_config.max_turns == 200
        assert model_config.max_chains == 10
        assert model_config.context_threshold == 0.80
        assert model_config.conversation_history_threshold == 0.40

    def test_status_json_direct_edit_takes_effect(
        self, data_dir: Path, tmp_path: Path
    ):
        """Editing status.json directly should change the resolved model."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, ENGINEER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, role="engineer")

        # Directly update status.json (simulates `anima set-model`)
        import json
        status_path = anima_dir / "status.json"
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        status_data["model"] = "claude-sonnet-4-6"
        status_data["max_turns"] = 50
        status_path.write_text(
            json.dumps(status_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        invalidate_cache()

        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()

        # Updated fields from status.json
        assert model_config.model == "claude-sonnet-4-6"
        assert model_config.max_turns == 50
        # Non-updated fields still come from original status.json (engineer role)
        assert model_config.max_chains == 10
        assert model_config.context_threshold == 0.80

    def test_resolve_anima_config_directly(
        self, data_dir: Path, tmp_path: Path
    ):
        """Test resolve_anima_config with 2-layer merge (status.json > defaults)."""
        import json
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, ENGINEER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, role="engineer")

        # Edit status.json to override one field
        status_path = anima_dir / "status.json"
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        status_data["max_turns"] = 99
        status_path.write_text(json.dumps(status_data, indent=2) + "\n", encoding="utf-8")
        invalidate_cache()

        config_path = data_dir / "config.json"
        config = load_config(config_path)
        resolved, _cred = resolve_anima_config(
            config, "testeng", anima_dir=anima_dir
        )

        # Layer 1: status.json (engineer role defaults + our override)
        assert resolved.max_turns == 99
        assert resolved.model == "claude-opus-4-6"
        assert resolved.max_chains == 10
        assert resolved.context_threshold == 0.80
        # Layer 2: global defaults (for fields not set in status.json)
        assert resolved.max_tokens == 1024  # from test config anima_defaults


# ── Test 5: specialty_prompt injection in system prompt ─────


class TestSpecialtyPromptInjection:
    """Verify specialty_prompt.md content appears in the system prompt."""

    def test_specialty_prompt_appears_in_system_prompt(
        self, data_dir: Path, tmp_path: Path
    ):
        """build_system_prompt should include specialty_prompt content."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, ENGINEER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, role="engineer")

        memory = MemoryManager(anima_dir)
        invalidate_cache()

        prompt = build_system_prompt(memory)

        # The engineer specialty prompt should be present
        assert "エンジニア" in prompt or "コーディング" in prompt, (
            "Engineer specialty content not found in system prompt"
        )

    def test_specialty_prompt_between_injection_and_permissions(
        self, data_dir: Path, tmp_path: Path
    ):
        """specialty_prompt should appear after injection.md and before permissions.md."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, ENGINEER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, role="engineer")

        memory = MemoryManager(anima_dir)
        invalidate_cache()

        result = build_system_prompt(memory)
        prompt = result.system_prompt

        # Read files to find their unique content markers
        injection_content = memory.read_injection()
        specialty_content = memory.read_specialty_prompt()
        permissions_content = memory.read_permissions()

        assert injection_content, "injection.md should have content"
        assert specialty_content, "specialty_prompt.md should have content"
        assert permissions_content, "permissions.md should have content"

        # Find the positions in the assembled prompt
        # Use a distinctive string from each section
        injection_marker = "テスト用の行動方針です"
        specialty_marker = "エンジニア専門ガイドライン"
        permissions_marker = "Permissions"

        injection_pos = prompt.find(injection_marker)
        specialty_pos = prompt.find(specialty_marker)
        permissions_pos = prompt.find(permissions_marker)

        assert injection_pos >= 0, (
            f"Injection marker '{injection_marker}' not found in prompt"
        )
        assert specialty_pos >= 0, (
            f"Specialty marker '{specialty_marker}' not found in prompt"
        )
        assert permissions_pos >= 0, (
            f"Permissions marker '{permissions_marker}' not found in prompt"
        )

        # Verify ordering: injection < specialty < permissions
        assert injection_pos < specialty_pos < permissions_pos, (
            f"Incorrect ordering: injection@{injection_pos}, "
            f"specialty@{specialty_pos}, permissions@{permissions_pos}"
        )

    def test_general_role_specialty_prompt_in_system_prompt(
        self, data_dir: Path, tmp_path: Path
    ):
        """General role specialty prompt should also appear in system prompt."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_sheet(tmp_path, GENERAL_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        memory = MemoryManager(anima_dir)
        invalidate_cache()

        prompt = build_system_prompt(memory)

        # General specialty has "汎用" content
        assert "汎用" in prompt, (
            "General specialty content not found in system prompt"
        )

    def test_no_specialty_prompt_when_file_missing(
        self, data_dir: Path, tmp_path: Path, make_anima
    ):
        """If specialty_prompt.md does not exist, system prompt should still build."""
        # Use make_anima fixture which does NOT create specialty_prompt.md
        anima_dir = make_anima("no-specialty")

        memory = MemoryManager(anima_dir)
        invalidate_cache()

        # Should not raise
        prompt = build_system_prompt(memory)
        assert prompt  # non-empty prompt
