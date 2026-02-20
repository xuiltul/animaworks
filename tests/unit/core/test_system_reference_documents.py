# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the system-reference-documents feature.

Tests cover:
1. core/init.py — _INFRASTRUCTURE_DIRS includes "common_knowledge",
   _copy_infrastructure / merge_templates / _ensure_runtime_only_dirs
2. core/prompt/builder.py — build_system_prompt includes/excludes
   the common_knowledge reference hint section
3. scripts/generate_reference.py — auto-generation of reference content
   from code into AUTO-GENERATED markers
4. templates/common_knowledge/ — structural validation of template files
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ════════════════════════════════════════════════════════════════════
# 1. Init tests (core/init.py)
# ════════════════════════════════════════════════════════════════════


class TestInfrastructureDirsContainsCommonKnowledge:
    """Verify _INFRASTRUCTURE_DIRS includes the common_knowledge entry."""

    def test_common_knowledge_in_infrastructure_dirs(self):
        from core.init import _INFRASTRUCTURE_DIRS

        assert "common_knowledge" in _INFRASTRUCTURE_DIRS

    def test_infrastructure_dirs_is_set(self):
        from core.init import _INFRASTRUCTURE_DIRS

        assert isinstance(_INFRASTRUCTURE_DIRS, set)

    def test_other_expected_dirs_present(self):
        from core.init import _INFRASTRUCTURE_DIRS

        for name in ("prompts", "company", "common_skills", "common_knowledge"):
            assert name in _INFRASTRUCTURE_DIRS, f"{name} missing"


class TestCopyInfrastructure:
    """Verify _copy_infrastructure copies common_knowledge/ to data_dir."""

    def test_copies_common_knowledge_directory(self, tmp_path: Path):
        from core.init import _copy_infrastructure

        # Set up a fake templates directory with common_knowledge
        templates = tmp_path / "templates"
        ck_src = templates / "common_knowledge"
        ck_src.mkdir(parents=True)
        (ck_src / "00_index.md").write_text("# Index", encoding="utf-8")
        sub = ck_src / "organization"
        sub.mkdir()
        (sub / "structure.md").write_text("# Structure", encoding="utf-8")

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("core.init.TEMPLATES_DIR", templates):
            _copy_infrastructure(data_dir)

        assert (data_dir / "common_knowledge" / "00_index.md").exists()
        assert (data_dir / "common_knowledge" / "organization" / "structure.md").exists()

    def test_does_not_copy_anima_templates(self, tmp_path: Path):
        from core.init import _copy_infrastructure

        templates = tmp_path / "templates"
        (templates / "anima_templates").mkdir(parents=True)
        (templates / "anima_templates" / "identity.md").write_text(
            "# Identity", encoding="utf-8"
        )
        # Also add common_knowledge so there's something valid
        (templates / "common_knowledge").mkdir()
        (templates / "common_knowledge" / "test.md").write_text("test", encoding="utf-8")

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("core.init.TEMPLATES_DIR", templates):
            _copy_infrastructure(data_dir)

        assert not (data_dir / "anima_templates").exists()

    def test_copytree_dirs_exist_ok(self, tmp_path: Path):
        """Copying when destination already exists should succeed (dirs_exist_ok)."""
        from core.init import _copy_infrastructure

        templates = tmp_path / "templates"
        ck_src = templates / "common_knowledge"
        ck_src.mkdir(parents=True)
        (ck_src / "00_index.md").write_text("# Index v1", encoding="utf-8")

        data_dir = tmp_path / "data"
        ck_dst = data_dir / "common_knowledge"
        ck_dst.mkdir(parents=True)
        (ck_dst / "00_index.md").write_text("# Old Index", encoding="utf-8")

        with patch("core.init.TEMPLATES_DIR", templates):
            _copy_infrastructure(data_dir)

        content = (ck_dst / "00_index.md").read_text(encoding="utf-8")
        assert content == "# Index v1"


class TestMergeTemplates:
    """Verify merge_templates copies missing common_knowledge files."""

    def test_copies_missing_common_knowledge_files(self, tmp_path: Path):
        from core.init import merge_templates

        templates = tmp_path / "templates"
        ck_src = templates / "common_knowledge"
        ck_src.mkdir(parents=True)
        (ck_src / "00_index.md").write_text("# Index", encoding="utf-8")
        org = ck_src / "organization"
        org.mkdir()
        (org / "roles.md").write_text("# Roles", encoding="utf-8")

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Create common_knowledge dir but leave it empty
        (data_dir / "common_knowledge").mkdir(parents=True)

        with patch("core.init.TEMPLATES_DIR", templates):
            added = merge_templates(data_dir)

        assert "common_knowledge/00_index.md" in added
        assert "common_knowledge/organization/roles.md" in added
        assert (data_dir / "common_knowledge" / "00_index.md").exists()

    def test_does_not_overwrite_existing_files(self, tmp_path: Path):
        from core.init import merge_templates

        templates = tmp_path / "templates"
        ck_src = templates / "common_knowledge"
        ck_src.mkdir(parents=True)
        (ck_src / "00_index.md").write_text("# Template version", encoding="utf-8")

        data_dir = tmp_path / "data"
        (data_dir / "common_knowledge").mkdir(parents=True)
        (data_dir / "common_knowledge" / "00_index.md").write_text(
            "# User-customized version", encoding="utf-8"
        )

        with patch("core.init.TEMPLATES_DIR", templates):
            added = merge_templates(data_dir)

        # File already existed, should not be overwritten or listed
        assert "common_knowledge/00_index.md" not in added
        content = (data_dir / "common_knowledge" / "00_index.md").read_text(
            encoding="utf-8"
        )
        assert content == "# User-customized version"


class TestEnsureRuntimeOnlyDirs:
    """Verify _ensure_runtime_only_dirs creates common_knowledge dir."""

    def test_creates_common_knowledge_dir(self, tmp_path: Path):
        from core.init import _ensure_runtime_only_dirs

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        _ensure_runtime_only_dirs(data_dir)

        assert (data_dir / "common_knowledge").is_dir()

    def test_creates_all_runtime_dirs(self, tmp_path: Path):
        from core.init import _ensure_runtime_only_dirs

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        _ensure_runtime_only_dirs(data_dir)

        expected = [
            "animas",
            "shared/inbox",
            "shared/users",
            "tmp/attachments",
            "common_skills",
            "common_knowledge",
        ]
        for rel in expected:
            assert (data_dir / rel).is_dir(), f"{rel} not created"

    def test_idempotent(self, tmp_path: Path):
        from core.init import _ensure_runtime_only_dirs

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        _ensure_runtime_only_dirs(data_dir)
        _ensure_runtime_only_dirs(data_dir)

        assert (data_dir / "common_knowledge").is_dir()


# ════════════════════════════════════════════════════════════════════
# 2. Prompt builder tests (core/prompt/builder.py)
# ════════════════════════════════════════════════════════════════════


def _make_mock_memory(anima_dir: Path) -> MagicMock:
    """Build a minimal MemoryManager mock for build_system_prompt."""
    mem = MagicMock()
    mem.anima_dir = anima_dir
    mem.read_bootstrap.return_value = ""
    mem.read_company_vision.return_value = ""
    mem.read_identity.return_value = "# Test Anima"
    mem.read_injection.return_value = ""
    mem.read_specialty_prompt.return_value = ""
    mem.read_permissions.return_value = ""
    mem.read_current_state.return_value = ""
    mem.read_pending.return_value = ""
    mem.list_knowledge_files.return_value = []
    mem.list_episode_files.return_value = []
    mem.list_procedure_files.return_value = []
    mem.list_skill_summaries.return_value = []
    mem.list_common_skill_summaries.return_value = []
    mem.list_skill_metas.return_value = []
    mem.list_common_skill_metas.return_value = []
    mem.list_shared_users.return_value = []
    mem.common_skills_dir = anima_dir.parent.parent / "common_skills"
    return mem


class TestBuildSystemPromptCommonKnowledge:
    """Test that the common_knowledge hint section appears/disappears correctly."""

    def test_includes_section_when_common_knowledge_has_md_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from core.prompt.builder import build_system_prompt

        data_dir = tmp_path / "data"
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True)
        (ck_dir / "00_index.md").write_text("# Index", encoding="utf-8")

        animas_dir = data_dir / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        memory = _make_mock_memory(anima_dir)

        monkeypatch.setattr("core.prompt.builder.get_data_dir", lambda: data_dir)
        monkeypatch.setattr(
            "core.prompt.builder.load_prompt",
            lambda name, **kw: (
                "## 共有リファレンス\n\n"
                "困ったとき・手順が不明なときは `common_knowledge/` を "
                "`search_memory` で検索するか、`read_memory_file` で直接読んでください。\n"
                "目次: `common_knowledge/00_index.md`"
                if name == "builder/common_knowledge_hint"
                else f"[{name}]"
            ),
        )

        prompt = build_system_prompt(memory)

        assert "共有リファレンス" in prompt
        assert "search_memory" in prompt
        assert "read_memory_file" in prompt
        assert "common_knowledge/00_index.md" in prompt

    def test_excludes_section_when_common_knowledge_is_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from core.prompt.builder import build_system_prompt

        data_dir = tmp_path / "data"
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True)
        # Directory exists but no .md files

        animas_dir = data_dir / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)

        memory = _make_mock_memory(anima_dir)

        monkeypatch.setattr("core.prompt.builder.get_data_dir", lambda: data_dir)
        monkeypatch.setattr(
            "core.prompt.builder.load_prompt",
            lambda name, **kw: f"[{name}]",
        )

        prompt = build_system_prompt(memory)

        assert "共有リファレンス" not in prompt

    def test_excludes_section_when_common_knowledge_dir_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from core.prompt.builder import build_system_prompt

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        # No common_knowledge directory at all

        animas_dir = data_dir / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)

        memory = _make_mock_memory(anima_dir)

        monkeypatch.setattr("core.prompt.builder.get_data_dir", lambda: data_dir)
        monkeypatch.setattr(
            "core.prompt.builder.load_prompt",
            lambda name, **kw: f"[{name}]",
        )

        prompt = build_system_prompt(memory)

        assert "共有リファレンス" not in prompt

    def test_section_includes_subdirectory_md_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """rglob('*.md') should detect .md files in subdirectories too."""
        from core.prompt.builder import build_system_prompt

        data_dir = tmp_path / "data"
        ck_dir = data_dir / "common_knowledge"
        sub = ck_dir / "operations"
        sub.mkdir(parents=True)
        (sub / "guide.md").write_text("# Guide", encoding="utf-8")

        animas_dir = data_dir / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)

        memory = _make_mock_memory(anima_dir)

        monkeypatch.setattr("core.prompt.builder.get_data_dir", lambda: data_dir)
        monkeypatch.setattr(
            "core.prompt.builder.load_prompt",
            lambda name, **kw: (
                "## 共有リファレンス\n\ncommon_knowledge/ を検索してください。"
                if name == "builder/common_knowledge_hint"
                else f"[{name}]"
            ),
        )

        prompt = build_system_prompt(memory)

        assert "共有リファレンス" in prompt

    def test_non_md_files_do_not_trigger_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Only .md files should count; .txt or other files should not."""
        from core.prompt.builder import build_system_prompt

        data_dir = tmp_path / "data"
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True)
        (ck_dir / "notes.txt").write_text("not markdown", encoding="utf-8")

        animas_dir = data_dir / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)

        memory = _make_mock_memory(anima_dir)

        monkeypatch.setattr("core.prompt.builder.get_data_dir", lambda: data_dir)
        monkeypatch.setattr(
            "core.prompt.builder.load_prompt",
            lambda name, **kw: f"[{name}]",
        )

        prompt = build_system_prompt(memory)

        assert "共有リファレンス" not in prompt


# ════════════════════════════════════════════════════════════════════
# 3. Generate reference script tests (scripts/generate_reference.py)
# ════════════════════════════════════════════════════════════════════


class TestGenerateToolParameters:
    """Test _generate_tool_parameters() produces valid markdown."""

    def test_output_contains_tool_names(self):
        from scripts.generate_reference import _generate_tool_parameters

        result = _generate_tool_parameters()

        # Should reference the canonical tool names from MEMORY_TOOLS
        from core.tooling.schemas import MEMORY_TOOLS

        for tool in MEMORY_TOOLS:
            assert tool["name"] in result, f"Missing tool name: {tool['name']}"

    def test_output_has_markdown_table_headers(self):
        from scripts.generate_reference import _generate_tool_parameters

        result = _generate_tool_parameters()

        assert "パラメータ" in result
        assert "型" in result
        assert "必須" in result
        assert "説明" in result

    def test_output_has_heading(self):
        from scripts.generate_reference import _generate_tool_parameters

        result = _generate_tool_parameters()

        assert "### ツールパラメータリファレンス" in result

    def test_output_is_nonempty_string(self):
        from scripts.generate_reference import _generate_tool_parameters

        result = _generate_tool_parameters()

        assert isinstance(result, str)
        assert len(result) > 100  # Should be substantial


class TestGenerateConfigFields:
    """Test _generate_config_fields() produces valid markdown."""

    def test_output_contains_anima_model_config_fields(self):
        from scripts.generate_reference import _generate_config_fields
        from core.config.models import AnimaModelConfig

        result = _generate_config_fields()

        for fname in AnimaModelConfig.model_fields:
            assert f"`{fname}`" in result, f"Missing field: {fname}"

    def test_output_contains_animaworks_config_sections(self):
        from scripts.generate_reference import _generate_config_fields

        result = _generate_config_fields()

        assert "AnimaWorksConfig" in result
        assert "animas" in result
        assert "credentials" in result

    def test_output_has_heading(self):
        from scripts.generate_reference import _generate_config_fields

        result = _generate_config_fields()

        assert "### 設定項目リファレンス" in result

    def test_output_is_nonempty_string(self):
        from scripts.generate_reference import _generate_config_fields

        result = _generate_config_fields()

        assert isinstance(result, str)
        assert len(result) > 100


class TestGenerateCronFields:
    """Test _generate_cron_fields() produces valid markdown."""

    def test_output_contains_cron_task_fields(self):
        from scripts.generate_reference import _generate_cron_fields
        from core.schemas import CronTask

        result = _generate_cron_fields()

        for fname in CronTask.model_fields:
            assert f"`{fname}`" in result, f"Missing field: {fname}"

    def test_output_has_heading(self):
        from scripts.generate_reference import _generate_cron_fields

        result = _generate_cron_fields()

        assert "### CronTaskフィールドリファレンス" in result

    def test_output_has_table_structure(self):
        from scripts.generate_reference import _generate_cron_fields

        result = _generate_cron_fields()

        assert "フィールド" in result
        assert "型" in result
        assert "デフォルト" in result
        assert "|---" in result

    def test_output_is_nonempty_string(self):
        from scripts.generate_reference import _generate_cron_fields

        result = _generate_cron_fields()

        assert isinstance(result, str)
        assert len(result) > 50


class TestInjectAutoGenerated:
    """Test inject_auto_generated() marker replacement logic."""

    def test_replaces_known_marker(self, tmp_path: Path):
        from scripts.generate_reference import inject_auto_generated

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# Title\n\n"
            "<!-- AUTO-GENERATED:START cron_fields -->\n"
            "old content here\n"
            "<!-- AUTO-GENERATED:END -->\n\n"
            "## Footer\n",
            encoding="utf-8",
        )

        modified = inject_auto_generated(md_file)

        assert modified is True
        content = md_file.read_text(encoding="utf-8")
        assert "CronTaskフィールドリファレンス" in content
        assert "old content here" not in content
        assert "# Title" in content
        assert "## Footer" in content

    def test_leaves_file_without_markers_unchanged(self, tmp_path: Path):
        from scripts.generate_reference import inject_auto_generated

        md_file = tmp_path / "no_markers.md"
        original = "# No markers here\n\nJust plain content.\n"
        md_file.write_text(original, encoding="utf-8")

        modified = inject_auto_generated(md_file)

        assert modified is False
        assert md_file.read_text(encoding="utf-8") == original

    def test_handles_unknown_section_name_gracefully(self, tmp_path: Path):
        from scripts.generate_reference import inject_auto_generated

        md_file = tmp_path / "unknown.md"
        original = (
            "# Doc\n\n"
            "<!-- AUTO-GENERATED:START nonexistent_section -->\n"
            "placeholder\n"
            "<!-- AUTO-GENERATED:END -->\n"
        )
        md_file.write_text(original, encoding="utf-8")

        modified = inject_auto_generated(md_file)

        # Unknown section: marker is left as-is, file not modified
        assert modified is False
        assert md_file.read_text(encoding="utf-8") == original

    def test_dry_run_does_not_modify_file(self, tmp_path: Path):
        from scripts.generate_reference import inject_auto_generated

        md_file = tmp_path / "dryrun.md"
        original = (
            "<!-- AUTO-GENERATED:START cron_fields -->\n"
            "old\n"
            "<!-- AUTO-GENERATED:END -->\n"
        )
        md_file.write_text(original, encoding="utf-8")

        modified = inject_auto_generated(md_file, dry_run=True)

        assert modified is True
        # File should remain unchanged on disk
        assert md_file.read_text(encoding="utf-8") == original

    def test_replaces_multiple_markers_in_same_file(self, tmp_path: Path):
        from scripts.generate_reference import inject_auto_generated

        md_file = tmp_path / "multi.md"
        md_file.write_text(
            "<!-- AUTO-GENERATED:START cron_fields -->\n"
            "placeholder1\n"
            "<!-- AUTO-GENERATED:END -->\n"
            "\n"
            "Some text in between.\n"
            "\n"
            "<!-- AUTO-GENERATED:START config_fields -->\n"
            "placeholder2\n"
            "<!-- AUTO-GENERATED:END -->\n",
            encoding="utf-8",
        )

        modified = inject_auto_generated(md_file)

        assert modified is True
        content = md_file.read_text(encoding="utf-8")
        assert "CronTaskフィールドリファレンス" in content
        assert "設定項目リファレンス" in content
        assert "placeholder1" not in content
        assert "placeholder2" not in content

    def test_preserves_start_end_markers(self, tmp_path: Path):
        from scripts.generate_reference import inject_auto_generated

        md_file = tmp_path / "markers.md"
        md_file.write_text(
            "<!-- AUTO-GENERATED:START cron_fields -->\n"
            "old\n"
            "<!-- AUTO-GENERATED:END -->\n",
            encoding="utf-8",
        )

        inject_auto_generated(md_file)

        content = md_file.read_text(encoding="utf-8")
        assert "<!-- AUTO-GENERATED:START cron_fields -->" in content
        assert "<!-- AUTO-GENERATED:END -->" in content


class TestProcessDirectory:
    """Test process_directory() finds and processes files with markers."""

    def test_finds_and_processes_files_with_markers(self, tmp_path: Path):
        from scripts.generate_reference import process_directory

        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir()
        sub = ck_dir / "operations"
        sub.mkdir()

        (sub / "with_marker.md").write_text(
            "# Guide\n\n"
            "<!-- AUTO-GENERATED:START cron_fields -->\n"
            "old\n"
            "<!-- AUTO-GENERATED:END -->\n",
            encoding="utf-8",
        )
        (ck_dir / "no_marker.md").write_text(
            "# Plain file\nNo markers.\n",
            encoding="utf-8",
        )

        modified = process_directory(ck_dir)

        assert len(modified) == 1
        assert modified[0] == sub / "with_marker.md"

    def test_returns_empty_when_no_markers(self, tmp_path: Path):
        from scripts.generate_reference import process_directory

        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir()
        (ck_dir / "plain.md").write_text("# Plain\n", encoding="utf-8")

        modified = process_directory(ck_dir)

        assert modified == []

    def test_dry_run_does_not_modify_files(self, tmp_path: Path):
        from scripts.generate_reference import process_directory

        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir()
        md_file = ck_dir / "test.md"
        original = (
            "<!-- AUTO-GENERATED:START cron_fields -->\n"
            "old\n"
            "<!-- AUTO-GENERATED:END -->\n"
        )
        md_file.write_text(original, encoding="utf-8")

        modified = process_directory(ck_dir, dry_run=True)

        assert len(modified) == 1
        # Content should be unchanged on disk
        assert md_file.read_text(encoding="utf-8") == original

    def test_handles_empty_directory(self, tmp_path: Path):
        from scripts.generate_reference import process_directory

        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir()

        modified = process_directory(ck_dir)

        assert modified == []


class TestFormatType:
    """Test _format_type helper for type annotation formatting."""

    def test_none_annotation(self):
        from scripts.generate_reference import _format_type

        assert _format_type(None) == "Any"

    def test_str_annotation(self):
        from scripts.generate_reference import _format_type

        result = _format_type(str)
        assert "str" in result

    def test_int_annotation(self):
        from scripts.generate_reference import _format_type

        result = _format_type(int)
        assert "int" in result


class TestMarkerRegex:
    """Test the _MARKER_RE regex pattern matching."""

    def test_matches_valid_marker_block(self):
        from scripts.generate_reference import _MARKER_RE

        text = (
            "<!-- AUTO-GENERATED:START tool_parameters -->\n"
            "some content\n"
            "<!-- AUTO-GENERATED:END -->"
        )
        match = _MARKER_RE.search(text)

        assert match is not None
        assert match.group(2) == "tool_parameters"

    def test_matches_multiline_content(self):
        from scripts.generate_reference import _MARKER_RE

        text = (
            "<!-- AUTO-GENERATED:START config_fields -->\n"
            "line 1\n"
            "line 2\n"
            "line 3\n"
            "<!-- AUTO-GENERATED:END -->"
        )
        match = _MARKER_RE.search(text)

        assert match is not None
        assert match.group(2) == "config_fields"

    def test_no_match_without_markers(self):
        from scripts.generate_reference import _MARKER_RE

        text = "# Normal markdown\n\nNo markers here."
        match = _MARKER_RE.search(text)

        assert match is None

    def test_captures_section_name(self):
        from scripts.generate_reference import _MARKER_RE

        text = (
            "<!-- AUTO-GENERATED:START cron_fields -->\n"
            "x\n"
            "<!-- AUTO-GENERATED:END -->"
        )
        match = _MARKER_RE.search(text)

        assert match is not None
        assert match.group(2) == "cron_fields"


class TestGeneratorsRegistry:
    """Test the _GENERATORS mapping has expected entries."""

    def test_has_tool_parameters(self):
        from scripts.generate_reference import _GENERATORS

        assert "tool_parameters" in _GENERATORS
        assert callable(_GENERATORS["tool_parameters"])

    def test_has_config_fields(self):
        from scripts.generate_reference import _GENERATORS

        assert "config_fields" in _GENERATORS
        assert callable(_GENERATORS["config_fields"])

    def test_has_cron_fields(self):
        from scripts.generate_reference import _GENERATORS

        assert "cron_fields" in _GENERATORS
        assert callable(_GENERATORS["cron_fields"])

    def test_exactly_three_generators(self):
        from scripts.generate_reference import _GENERATORS

        assert len(_GENERATORS) == 3


# ════════════════════════════════════════════════════════════════════
# 4. Template validation tests
# ════════════════════════════════════════════════════════════════════


# Path to the real templates in the project
_TEMPLATES_CK_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "templates"
    / "common_knowledge"
)

# All 12 expected markdown files
_EXPECTED_FILES = [
    "00_index.md",
    "organization/structure.md",
    "organization/roles.md",
    "organization/hierarchy-rules.md",
    "communication/messaging-guide.md",
    "communication/instruction-patterns.md",
    "communication/reporting-guide.md",
    "operations/project-setup.md",
    "operations/task-management.md",
    "operations/heartbeat-cron-guide.md",
    "troubleshooting/common-issues.md",
    "troubleshooting/escalation-flowchart.md",
]


class TestTemplateFilesExist:
    """Verify all expected template markdown files are present."""

    @pytest.mark.parametrize("rel_path", _EXPECTED_FILES)
    def test_expected_file_exists(self, rel_path: str):
        full_path = _TEMPLATES_CK_DIR / rel_path
        assert full_path.exists(), f"Missing template: {rel_path}"

    def test_total_file_count(self):
        """Exactly 14 .md files should exist in common_knowledge."""
        md_files = list(_TEMPLATES_CK_DIR.rglob("*.md"))
        assert len(md_files) == 14, (
            f"Expected 14 .md files, found {len(md_files)}: "
            f"{[str(f.relative_to(_TEMPLATES_CK_DIR)) for f in md_files]}"
        )

    def test_directory_structure(self):
        """Expected subdirectories should exist."""
        for subdir in ("organization", "communication", "operations", "troubleshooting"):
            assert (_TEMPLATES_CK_DIR / subdir).is_dir(), f"Missing subdir: {subdir}"


class TestTemplateHeadings:
    """Each template file should have ## level headings for RAG chunking."""

    @pytest.mark.parametrize("rel_path", _EXPECTED_FILES)
    def test_file_has_h2_headings(self, rel_path: str):
        full_path = _TEMPLATES_CK_DIR / rel_path
        content = full_path.read_text(encoding="utf-8")
        # Check for ## heading (with possible ### or ####)
        has_heading = any(
            line.startswith("## ") or line.startswith("### ") or line.startswith("#### ")
            for line in content.splitlines()
        )
        assert has_heading, f"{rel_path} has no ## or deeper headings"


class TestTemplateAutoGeneratedMarkers:
    """Files with AUTO-GENERATED markers should have matching START/END pairs."""

    _FILES_WITH_MARKERS = [
        "communication/messaging-guide.md",
        "operations/project-setup.md",
        "operations/heartbeat-cron-guide.md",
    ]

    @pytest.mark.parametrize("rel_path", _FILES_WITH_MARKERS)
    def test_start_end_markers_match(self, rel_path: str):
        import re

        full_path = _TEMPLATES_CK_DIR / rel_path
        content = full_path.read_text(encoding="utf-8")

        starts = re.findall(r"<!-- AUTO-GENERATED:START (\w+) -->", content)
        ends = re.findall(r"<!-- AUTO-GENERATED:END -->", content)

        assert len(starts) > 0, f"{rel_path} should have AUTO-GENERATED markers"
        assert len(starts) == len(ends), (
            f"{rel_path} has {len(starts)} START markers but {len(ends)} END markers"
        )

    @pytest.mark.parametrize("rel_path", _FILES_WITH_MARKERS)
    def test_marker_section_names_are_valid(self, rel_path: str):
        import re

        from scripts.generate_reference import _GENERATORS

        full_path = _TEMPLATES_CK_DIR / rel_path
        content = full_path.read_text(encoding="utf-8")

        starts = re.findall(r"<!-- AUTO-GENERATED:START (\w+) -->", content)
        for section_name in starts:
            assert section_name in _GENERATORS, (
                f"{rel_path} references unknown section '{section_name}'"
            )

    def test_no_markers_in_other_files(self):
        """Files not in the markers list should not have AUTO-GENERATED markers."""
        files_with_markers = set(self._FILES_WITH_MARKERS)
        for md_file in _TEMPLATES_CK_DIR.rglob("*.md"):
            rel = str(md_file.relative_to(_TEMPLATES_CK_DIR))
            if rel in files_with_markers:
                continue
            content = md_file.read_text(encoding="utf-8")
            assert "AUTO-GENERATED:START" not in content, (
                f"Unexpected AUTO-GENERATED marker in {rel}"
            )


class TestIndexFileReferences:
    """00_index.md should reference all expected subdirectory files."""

    _EXPECTED_REFERENCED_FILES = [
        "organization/structure.md",
        "organization/roles.md",
        "organization/hierarchy-rules.md",
        "communication/messaging-guide.md",
        "communication/instruction-patterns.md",
        "communication/reporting-guide.md",
        "operations/project-setup.md",
        "operations/task-management.md",
        "operations/heartbeat-cron-guide.md",
        "troubleshooting/common-issues.md",
        "troubleshooting/escalation-flowchart.md",
    ]

    @pytest.mark.parametrize("rel_path", _EXPECTED_REFERENCED_FILES)
    def test_index_references_file(self, rel_path: str):
        index_path = _TEMPLATES_CK_DIR / "00_index.md"
        content = index_path.read_text(encoding="utf-8")
        assert rel_path in content, (
            f"00_index.md does not reference {rel_path}"
        )

    def test_index_has_keyword_section(self):
        index_path = _TEMPLATES_CK_DIR / "00_index.md"
        content = index_path.read_text(encoding="utf-8")
        assert "キーワード索引" in content

    def test_index_has_document_list_section(self):
        index_path = _TEMPLATES_CK_DIR / "00_index.md"
        content = index_path.read_text(encoding="utf-8")
        assert "ドキュメント一覧" in content
