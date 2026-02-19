# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Role Templates + Ability Scores feature.

Tests _apply_role_defaults(), _create_status_json() with role parameter,
_load_status_json(), resolve_anima_config() 3-layer merge,
read_specialty_prompt(), and _PROTECTED_FILES in agent_sdk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.anima_factory import (
    ROLES_DIR,
    VALID_ROLES,
    _apply_role_defaults,
    _create_status_json,
)
from core.config.models import (
    AnimaDefaults,
    AnimaModelConfig,
    AnimaWorksConfig,
    CredentialConfig,
    _load_status_json,
    invalidate_cache,
    resolve_anima_config,
)
from core.execution.agent_sdk import _PROTECTED_FILES


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    """Invalidate the config singleton before and after each test."""
    invalidate_cache()
    yield  # type: ignore[misc]
    invalidate_cache()


def _write_status_json(anima_dir: Path, data: dict[str, Any]) -> None:
    """Write a status.json file into an anima directory."""
    (anima_dir / "status.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_status_json(anima_dir: Path) -> dict[str, Any]:
    """Read and parse status.json from an anima directory."""
    return json.loads(
        (anima_dir / "status.json").read_text(encoding="utf-8")
    )


# ── 1. _apply_role_defaults ──────────────────────────────────────


class TestApplyRoleDefaults:
    """Tests for _apply_role_defaults() in core/anima_factory.py."""

    def _make_role_dir(
        self,
        tmp_path: Path,
        role: str,
        *,
        permissions_content: str | None = None,
        specialty_content: str | None = None,
        defaults_content: dict[str, Any] | None = None,
    ) -> Path:
        """Create a fake role template directory with optional files."""
        role_dir = tmp_path / "roles" / role
        role_dir.mkdir(parents=True, exist_ok=True)
        if permissions_content is not None:
            (role_dir / "permissions.md").write_text(
                permissions_content, encoding="utf-8"
            )
        if specialty_content is not None:
            (role_dir / "specialty_prompt.md").write_text(
                specialty_content, encoding="utf-8"
            )
        if defaults_content is not None:
            (role_dir / "defaults.json").write_text(
                json.dumps(defaults_content, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return role_dir

    def test_copies_permissions_md(self, tmp_path: Path) -> None:
        """permissions.md is copied from role template to anima dir."""
        roles_root = tmp_path / "roles"
        self._make_role_dir(
            tmp_path, "engineer",
            permissions_content="# Engineer permissions\nAll tools allowed.\n",
        )
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _apply_role_defaults(anima_dir, "engineer")

        perm_path = anima_dir / "permissions.md"
        assert perm_path.exists()
        content = perm_path.read_text(encoding="utf-8")
        assert "Engineer permissions" in content

    def test_copies_specialty_prompt_md(self, tmp_path: Path) -> None:
        """specialty_prompt.md is copied from role template to anima dir."""
        roles_root = tmp_path / "roles"
        self._make_role_dir(
            tmp_path, "researcher",
            specialty_content="# Research guidelines\nFocus on accuracy.\n",
        )
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _apply_role_defaults(anima_dir, "researcher")

        spec_path = anima_dir / "specialty_prompt.md"
        assert spec_path.exists()
        content = spec_path.read_text(encoding="utf-8")
        assert "Research guidelines" in content
        assert "Focus on accuracy." in content

    def test_name_placeholder_replaced_in_permissions(self, tmp_path: Path) -> None:
        """{name} placeholder in permissions.md is replaced with anima dir name."""
        roles_root = tmp_path / "roles"
        self._make_role_dir(
            tmp_path, "engineer",
            permissions_content="# Permissions: {name}\nAllowed tools for {name}.\n",
        )
        anima_dir = tmp_path / "mybot"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _apply_role_defaults(anima_dir, "engineer")

        content = (anima_dir / "permissions.md").read_text(encoding="utf-8")
        assert "{name}" not in content
        assert "mybot" in content
        assert "Permissions: mybot" in content
        assert "Allowed tools for mybot." in content

    def test_fallback_to_general_for_unknown_role(self, tmp_path: Path) -> None:
        """Unknown role falls back to 'general' template."""
        roles_root = tmp_path / "roles"
        self._make_role_dir(
            tmp_path, "general",
            permissions_content="# General permissions\n",
            specialty_content="# General specialty\n",
        )
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _apply_role_defaults(anima_dir, "unknown_role")

        # Should have general permissions, not crash
        perm_content = (anima_dir / "permissions.md").read_text(encoding="utf-8")
        assert "General permissions" in perm_content
        spec_content = (anima_dir / "specialty_prompt.md").read_text(encoding="utf-8")
        assert "General specialty" in spec_content

    def test_graceful_when_role_template_dir_missing(self, tmp_path: Path) -> None:
        """Returns without error when role template directory doesn't exist."""
        roles_root = tmp_path / "roles"
        # Don't create the "engineer" subdirectory at all
        roles_root.mkdir(parents=True, exist_ok=True)
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            # Should not raise
            _apply_role_defaults(anima_dir, "engineer")

        # No files should have been created
        assert not (anima_dir / "permissions.md").exists()
        assert not (anima_dir / "specialty_prompt.md").exists()

    def test_no_permissions_in_template(self, tmp_path: Path) -> None:
        """Works when role template has specialty_prompt but no permissions.md."""
        roles_root = tmp_path / "roles"
        self._make_role_dir(
            tmp_path, "writer",
            specialty_content="# Writing guidelines\n",
            # No permissions_content
        )
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _apply_role_defaults(anima_dir, "writer")

        assert not (anima_dir / "permissions.md").exists()
        assert (anima_dir / "specialty_prompt.md").exists()

    def test_no_specialty_in_template(self, tmp_path: Path) -> None:
        """Works when role template has permissions.md but no specialty_prompt.md."""
        roles_root = tmp_path / "roles"
        self._make_role_dir(
            tmp_path, "ops",
            permissions_content="# Ops permissions\n",
            # No specialty_content
        )
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _apply_role_defaults(anima_dir, "ops")

        assert (anima_dir / "permissions.md").exists()
        assert not (anima_dir / "specialty_prompt.md").exists()

    def test_valid_roles_constant(self) -> None:
        """VALID_ROLES contains all expected role names."""
        expected = {"engineer", "researcher", "manager", "writer", "ops", "general"}
        assert VALID_ROLES == expected


# ── 2. _create_status_json with role parameter ───────────────────


class TestCreateStatusJsonWithRole:
    """Tests for _create_status_json() with role defaults merging."""

    def _make_roles_dir(
        self, tmp_path: Path, role: str, defaults: dict[str, Any]
    ) -> Path:
        """Create a role defaults.json and return the roles root directory."""
        roles_root = tmp_path / "roles"
        role_dir = roles_root / role
        role_dir.mkdir(parents=True, exist_ok=True)
        (role_dir / "defaults.json").write_text(
            json.dumps(defaults, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return roles_root

    def test_role_defaults_merged_into_status(self, tmp_path: Path) -> None:
        """defaults.json values from role template are merged into status.json."""
        roles_root = self._make_roles_dir(tmp_path, "engineer", {
            "model": "claude-opus-4-20250514",
            "context_threshold": 0.80,
            "max_turns": 200,
            "max_chains": 10,
            "conversation_history_threshold": 0.40,
        })
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _create_status_json(anima_dir, {}, role="engineer")

        status = _read_status_json(anima_dir)
        assert status["model"] == "claude-opus-4-20250514"
        assert status["context_threshold"] == 0.80
        assert status["max_turns"] == 200
        assert status["max_chains"] == 10
        assert status["conversation_history_threshold"] == 0.40
        assert status["role"] == "engineer"

    def test_character_sheet_model_overrides_role_defaults(
        self, tmp_path: Path,
    ) -> None:
        """Character sheet model value takes priority over role defaults.json."""
        roles_root = self._make_roles_dir(tmp_path, "engineer", {
            "model": "claude-opus-4-20250514",
            "max_turns": 200,
        })
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        info = {"モデル": "openai/gpt-4o"}
        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _create_status_json(anima_dir, info, role="engineer")

        status = _read_status_json(anima_dir)
        # Character sheet overrides role default
        assert status["model"] == "openai/gpt-4o"
        # Role default still applies for non-overridden fields
        assert status["max_turns"] == 200

    def test_character_sheet_credential_overrides_role_defaults(
        self, tmp_path: Path,
    ) -> None:
        """Character sheet credential value takes priority over role defaults."""
        roles_root = self._make_roles_dir(tmp_path, "engineer", {
            "model": "claude-opus-4-20250514",
        })
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        info = {"credential": "my_custom_key"}
        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _create_status_json(anima_dir, info, role="engineer")

        status = _read_status_json(anima_dir)
        assert status["credential"] == "my_custom_key"

    def test_supervisor_override_works(self, tmp_path: Path) -> None:
        """supervisor_override parameter takes priority over sheet value."""
        roles_root = self._make_roles_dir(tmp_path, "general", {})
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        info = {"上司": "tanaka"}
        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _create_status_json(
                anima_dir, info, supervisor_override="yamada", role="general"
            )

        status = _read_status_json(anima_dir)
        assert status["supervisor"] == "yamada"

    def test_engineer_defaults_json_values_appear(self, tmp_path: Path) -> None:
        """With role='engineer', engineer defaults.json values appear in status.json."""
        # Use the actual engineer defaults from the repository
        roles_root = self._make_roles_dir(tmp_path, "engineer", {
            "model": "claude-opus-4-20250514",
            "context_threshold": 0.80,
            "max_turns": 200,
            "max_chains": 10,
            "conversation_history_threshold": 0.40,
        })
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _create_status_json(anima_dir, {}, role="engineer")

        status = _read_status_json(anima_dir)
        assert status["role"] == "engineer"
        assert status["model"] == "claude-opus-4-20250514"
        assert status["context_threshold"] == 0.80
        assert status["max_turns"] == 200
        assert status["max_chains"] == 10
        assert status["conversation_history_threshold"] == 0.40
        assert status["enabled"] is True

    def test_missing_role_defaults_json(self, tmp_path: Path) -> None:
        """Status.json is created even when role defaults.json is missing."""
        roles_root = tmp_path / "roles"
        (roles_root / "engineer").mkdir(parents=True)
        # No defaults.json created
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _create_status_json(anima_dir, {}, role="engineer")

        status = _read_status_json(anima_dir)
        assert status["role"] == "engineer"
        assert status["enabled"] is True
        # No model config fields from role defaults
        assert "model" not in status

    def test_invalid_defaults_json_is_handled(self, tmp_path: Path) -> None:
        """Invalid JSON in defaults.json is handled gracefully."""
        roles_root = tmp_path / "roles"
        role_dir = roles_root / "engineer"
        role_dir.mkdir(parents=True)
        (role_dir / "defaults.json").write_text("not valid json", encoding="utf-8")

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            # Should not raise
            _create_status_json(anima_dir, {}, role="engineer")

        status = _read_status_json(anima_dir)
        assert status["role"] == "engineer"
        assert status["enabled"] is True

    def test_empty_model_and_credential_not_overridden(
        self, tmp_path: Path,
    ) -> None:
        """Empty string model/credential from sheet do not override role defaults."""
        roles_root = self._make_roles_dir(tmp_path, "engineer", {
            "model": "claude-opus-4-20250514",
        })
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        # Empty strings should not override
        info = {"モデル": "", "credential": ""}
        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _create_status_json(anima_dir, info, role="engineer")

        status = _read_status_json(anima_dir)
        # Role default should remain since sheet values are empty
        assert status["model"] == "claude-opus-4-20250514"
        assert "credential" not in status


# ── 3. _load_status_json ─────────────────────────────────────────


class TestLoadStatusJson:
    """Tests for _load_status_json() in core/config/models.py."""

    def test_loads_valid_status_json(self, tmp_path: Path) -> None:
        """Extracts model config fields from a valid status.json."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "model": "claude-opus-4-20250514",
            "context_threshold": 0.80,
            "max_turns": 200,
            "max_chains": 10,
            "conversation_history_threshold": 0.40,
            "credential": "anthropic",
            "execution_mode": "autonomous",
            "supervisor": "kotoha",
        })

        result = _load_status_json(anima_dir)
        assert result["model"] == "claude-opus-4-20250514"
        assert result["context_threshold"] == 0.80
        assert result["max_turns"] == 200
        assert result["max_chains"] == 10
        assert result["conversation_history_threshold"] == 0.40
        assert result["credential"] == "anthropic"
        assert result["execution_mode"] == "autonomous"
        assert result["supervisor"] == "kotoha"

    def test_missing_status_json_returns_empty(self, tmp_path: Path) -> None:
        """Returns empty dict when status.json doesn't exist."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        # No status.json created

        result = _load_status_json(anima_dir)
        assert result == {}

    def test_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        """Returns empty dict when status.json contains invalid JSON."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text(
            "this is not json", encoding="utf-8"
        )

        result = _load_status_json(anima_dir)
        assert result == {}

    def test_none_values_are_skipped(self, tmp_path: Path) -> None:
        """Fields with None values are not included in the result."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "model": "claude-sonnet-4-20250514",
            "supervisor": None,
            "credential": None,
        })

        result = _load_status_json(anima_dir)
        assert "model" in result
        assert "supervisor" not in result
        assert "credential" not in result

    def test_empty_string_values_are_skipped(self, tmp_path: Path) -> None:
        """Fields with empty string values are not included in the result."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "model": "claude-sonnet-4-20250514",
            "supervisor": "",
            "credential": "",
        })

        result = _load_status_json(anima_dir)
        assert "model" in result
        assert "supervisor" not in result
        assert "credential" not in result

    def test_only_recognized_fields_extracted(self, tmp_path: Path) -> None:
        """Only fields in the field_mapping are extracted; others are ignored."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "model": "claude-sonnet-4-20250514",
            "enabled": True,
            "role": "engineer",
            "unknown_field": "should_be_ignored",
        })

        result = _load_status_json(anima_dir)
        assert "model" in result
        # These should NOT appear in the result
        assert "enabled" not in result
        assert "role" not in result
        assert "unknown_field" not in result

    def test_partial_fields(self, tmp_path: Path) -> None:
        """Only present recognized fields are extracted."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "model": "openai/gpt-4o",
            "max_turns": 50,
            "enabled": True,
        })

        result = _load_status_json(anima_dir)
        assert result == {"model": "openai/gpt-4o", "max_turns": 50}


# ── 4. resolve_anima_config 3-layer merge ─────────────────────────


class TestResolveAnimaConfig3Layer:
    """Tests for 3-layer merge in resolve_anima_config()."""

    def _make_config(
        self,
        *,
        anima_name: str = "testbot",
        anima_overrides: dict[str, Any] | None = None,
        defaults_overrides: dict[str, Any] | None = None,
    ) -> AnimaWorksConfig:
        """Build an AnimaWorksConfig with optional per-anima and default overrides."""
        config = AnimaWorksConfig()
        if defaults_overrides:
            config.anima_defaults = AnimaDefaults(**{
                **config.anima_defaults.model_dump(),
                **defaults_overrides,
            })
        if anima_overrides:
            config.animas[anima_name] = AnimaModelConfig(**anima_overrides)
        return config

    def test_config_override_beats_status_json(self, tmp_path: Path) -> None:
        """config.json per-anima override has highest priority."""
        anima_dir = tmp_path / "testbot"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "model": "from-status-json",
            "max_turns": 100,
        })

        config = self._make_config(
            anima_overrides={"model": "from-config-override"},
        )

        resolved, _ = resolve_anima_config(config, "testbot", anima_dir=anima_dir)
        assert resolved.model == "from-config-override"

    def test_status_json_beats_defaults(self, tmp_path: Path) -> None:
        """status.json values are used when config.json has no per-anima override."""
        anima_dir = tmp_path / "testbot"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "model": "from-status-json",
            "max_turns": 100,
        })

        # No per-anima overrides in config
        config = self._make_config(
            defaults_overrides={"model": "from-defaults", "max_turns": 20},
        )

        resolved, _ = resolve_anima_config(config, "testbot", anima_dir=anima_dir)
        assert resolved.model == "from-status-json"
        assert resolved.max_turns == 100

    def test_defaults_used_when_no_override_or_status(
        self, tmp_path: Path,
    ) -> None:
        """anima_defaults are used when neither config override nor status.json exist."""
        anima_dir = tmp_path / "testbot"
        anima_dir.mkdir()
        # Empty status.json
        _write_status_json(anima_dir, {"enabled": True})

        config = self._make_config(
            defaults_overrides={"model": "from-defaults", "max_turns": 42},
        )

        resolved, _ = resolve_anima_config(config, "testbot", anima_dir=anima_dir)
        assert resolved.model == "from-defaults"
        assert resolved.max_turns == 42

    def test_three_layer_priority_all_set(self, tmp_path: Path) -> None:
        """Full 3-layer merge: config override > status.json > defaults."""
        anima_dir = tmp_path / "testbot"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "model": "status-model",
            "max_turns": 150,
            "max_chains": 8,
        })

        config = self._make_config(
            anima_overrides={
                "model": "override-model",
                # max_turns NOT overridden - should come from status
            },
            defaults_overrides={
                "model": "default-model",
                "max_turns": 20,
                "max_chains": 2,
                "context_threshold": 0.50,
            },
        )

        resolved, _ = resolve_anima_config(config, "testbot", anima_dir=anima_dir)
        # Layer 1 wins for model
        assert resolved.model == "override-model"
        # Layer 2 wins for max_turns (no config override)
        assert resolved.max_turns == 150
        # Layer 2 wins for max_chains (no config override)
        assert resolved.max_chains == 8
        # Layer 3 wins for context_threshold (not in status.json or override)
        assert resolved.context_threshold == 0.50

    def test_backward_compatible_without_anima_dir(self) -> None:
        """2-layer merge when anima_dir is None (backward compatibility)."""
        config = self._make_config(
            anima_overrides={"model": "override-model"},
            defaults_overrides={"model": "default-model", "max_turns": 20},
        )

        resolved, _ = resolve_anima_config(config, "testbot", anima_dir=None)
        # Layer 1 wins
        assert resolved.model == "override-model"
        # Layer 3 defaults
        assert resolved.max_turns == 20

    def test_backward_compatible_defaults_only(self) -> None:
        """Pure defaults used when no anima override and anima_dir=None."""
        config = self._make_config(
            defaults_overrides={"model": "default-model"},
        )

        resolved, _ = resolve_anima_config(config, "testbot", anima_dir=None)
        assert resolved.model == "default-model"

    def test_status_json_supervisor_used_when_no_config_override(
        self, tmp_path: Path,
    ) -> None:
        """supervisor from status.json is used when config has no override."""
        anima_dir = tmp_path / "testbot"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "supervisor": "kotoha",
        })

        config = self._make_config()

        resolved, _ = resolve_anima_config(config, "testbot", anima_dir=anima_dir)
        assert resolved.supervisor == "kotoha"

    def test_config_override_supervisor_beats_status_json(
        self, tmp_path: Path,
    ) -> None:
        """config.json per-anima supervisor override beats status.json."""
        anima_dir = tmp_path / "testbot"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "supervisor": "from-status",
        })

        config = self._make_config(
            anima_overrides={"supervisor": "from-config"},
        )

        resolved, _ = resolve_anima_config(config, "testbot", anima_dir=anima_dir)
        assert resolved.supervisor == "from-config"

    def test_credential_resolution(self, tmp_path: Path) -> None:
        """Credential is correctly resolved through 3-layer merge."""
        anima_dir = tmp_path / "testbot"
        anima_dir.mkdir()
        _write_status_json(anima_dir, {
            "credential": "openai",
        })

        config = self._make_config()
        # Make sure "openai" credential exists in config
        config.credentials["openai"] = CredentialConfig(api_key="test-key")

        resolved, credential = resolve_anima_config(
            config, "testbot", anima_dir=anima_dir
        )
        assert resolved.credential == "openai"
        assert credential.api_key == "test-key"


# ── 5. read_specialty_prompt ──────────────────────────────────────


class TestReadSpecialtyPrompt:
    """Tests for read_specialty_prompt() in core/memory/manager.py."""

    def _make_memory_manager(self, anima_dir: Path):
        """Create a MemoryManager with mock paths to avoid real FS dependencies."""
        from core.memory.manager import MemoryManager

        with patch("core.memory.manager.get_company_dir", return_value=anima_dir / "company"), \
             patch("core.memory.manager.get_common_skills_dir", return_value=anima_dir / "common_skills"), \
             patch("core.memory.manager.get_common_knowledge_dir", return_value=anima_dir / "common_knowledge"), \
             patch("core.memory.manager.get_shared_dir", return_value=anima_dir / "shared"):
            return MemoryManager(anima_dir)

    def test_reads_existing_specialty_prompt(self, tmp_path: Path) -> None:
        """Returns content when specialty_prompt.md exists."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "specialty_prompt.md").write_text(
            "# Engineer specialty\nFocus on code quality.\n",
            encoding="utf-8",
        )

        mm = self._make_memory_manager(anima_dir)
        result = mm.read_specialty_prompt()
        assert "Engineer specialty" in result
        assert "Focus on code quality." in result

    def test_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        """Returns empty string when specialty_prompt.md doesn't exist."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        # No specialty_prompt.md created

        mm = self._make_memory_manager(anima_dir)
        result = mm.read_specialty_prompt()
        assert result == ""


# ── 6. specialty_prompt.md in _PROTECTED_FILES ────────────────────


class TestProtectedFiles:
    """Tests for _PROTECTED_FILES in core/execution/agent_sdk.py."""

    def test_specialty_prompt_is_protected(self) -> None:
        """specialty_prompt.md is included in _PROTECTED_FILES frozenset."""
        assert "specialty_prompt.md" in _PROTECTED_FILES

    def test_other_protected_files_present(self) -> None:
        """All expected protected files are in the frozenset."""
        assert "permissions.md" in _PROTECTED_FILES
        assert "identity.md" in _PROTECTED_FILES
        assert "bootstrap.md" in _PROTECTED_FILES

    def test_protected_files_is_frozenset(self) -> None:
        """_PROTECTED_FILES is an immutable frozenset."""
        assert isinstance(_PROTECTED_FILES, frozenset)


# ── Integration: _apply_role_defaults + _create_status_json ──────


class TestRoleTemplateIntegration:
    """Integration-style tests combining role template application with status creation."""

    def test_full_role_application_flow(self, tmp_path: Path) -> None:
        """Full flow: apply role defaults then create status.json."""
        roles_root = tmp_path / "roles"
        engineer_dir = roles_root / "engineer"
        engineer_dir.mkdir(parents=True)
        (engineer_dir / "permissions.md").write_text(
            "# Permissions: {name}\nAll tools.\n", encoding="utf-8"
        )
        (engineer_dir / "specialty_prompt.md").write_text(
            "# Engineer guidelines\n", encoding="utf-8"
        )
        (engineer_dir / "defaults.json").write_text(
            json.dumps({
                "model": "claude-opus-4-20250514",
                "max_turns": 200,
                "context_threshold": 0.80,
                "max_chains": 10,
                "conversation_history_threshold": 0.40,
            }),
            encoding="utf-8",
        )

        anima_dir = tmp_path / "hinata"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _apply_role_defaults(anima_dir, "engineer")
            _create_status_json(anima_dir, {}, role="engineer")

        # Verify permissions
        perm = (anima_dir / "permissions.md").read_text(encoding="utf-8")
        assert "Permissions: hinata" in perm

        # Verify specialty prompt
        spec = (anima_dir / "specialty_prompt.md").read_text(encoding="utf-8")
        assert "Engineer guidelines" in spec

        # Verify status.json
        status = _read_status_json(anima_dir)
        assert status["role"] == "engineer"
        assert status["model"] == "claude-opus-4-20250514"
        assert status["max_turns"] == 200

    def test_status_json_feeds_into_resolve_anima_config(
        self, tmp_path: Path,
    ) -> None:
        """status.json created by _create_status_json works with resolve_anima_config."""
        roles_root = tmp_path / "roles"
        engineer_dir = roles_root / "engineer"
        engineer_dir.mkdir(parents=True)
        (engineer_dir / "defaults.json").write_text(
            json.dumps({
                "model": "claude-opus-4-20250514",
                "max_turns": 200,
                "context_threshold": 0.80,
            }),
            encoding="utf-8",
        )

        anima_dir = tmp_path / "testbot"
        anima_dir.mkdir()

        with patch("core.anima_factory.ROLES_DIR", roles_root):
            _create_status_json(anima_dir, {}, role="engineer")

        # Now resolve with no config override
        config = AnimaWorksConfig()

        resolved, _ = resolve_anima_config(
            config, "testbot", anima_dir=anima_dir
        )
        # status.json values should be picked up
        assert resolved.model == "claude-opus-4-20250514"
        assert resolved.max_turns == 200
        assert resolved.context_threshold == 0.80
