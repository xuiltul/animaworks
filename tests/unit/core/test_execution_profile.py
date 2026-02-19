"""Tests for EXECUTION_PROFILE constants and profile-loading utilities."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import pytest

from core.background import BackgroundTaskManager, _DEFAULT_ELIGIBLE_TOOLS
from core.tools._base import get_eligible_tools_from_profiles, load_execution_profiles


# ── Module registry ──────────────────────────────────────────

# All core tool modules that should define EXECUTION_PROFILE
_TOOL_MODULES = {
    "image_gen": "core.tools.image_gen",
    "local_llm": "core.tools.local_llm",
    "transcribe": "core.tools.transcribe",
    "web_search": "core.tools.web_search",
    "x_search": "core.tools.x_search",
    "slack": "core.tools.slack",
    "chatwork": "core.tools.chatwork",
    "gmail": "core.tools.gmail",
    "github": "core.tools.github",
    "aws_collector": "core.tools.aws_collector",
}

# Expected background-eligible subcommands per tool
_EXPECTED_BG_ELIGIBLE = {
    "image_gen": {"pipeline", "3d", "rigging", "animations", "fullbody", "bustup", "chibi"},
    "local_llm": {"generate", "chat"},
    "transcribe": {"transcribe"},
}

# Modules that require optional dependencies and may fail to import
_OPTIONAL_DEPS: dict[str, str] = {
    "gmail": "google.oauth2",
}


def _try_import(module_path: str, tool_name: str):
    """Import a tool module, skipping if an optional dependency is missing."""
    try:
        return importlib.import_module(module_path)
    except ImportError:
        dep = _OPTIONAL_DEPS.get(tool_name)
        reason = f"{tool_name} requires optional dependency ({dep})"
        pytest.skip(reason)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path):
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    return d


# ── EXECUTION_PROFILE existence & structure ──────────────────


class TestExecutionProfileExists:
    @pytest.mark.parametrize("tool_name,module_path", list(_TOOL_MODULES.items()))
    def test_profile_exists(self, tool_name: str, module_path: str):
        """Each tool module should define EXECUTION_PROFILE."""
        mod = _try_import(module_path, tool_name)
        assert hasattr(mod, "EXECUTION_PROFILE"), f"{tool_name} missing EXECUTION_PROFILE"

    @pytest.mark.parametrize("tool_name,module_path", list(_TOOL_MODULES.items()))
    def test_profile_structure(self, tool_name: str, module_path: str):
        """EXECUTION_PROFILE should be dict[str, dict] with required keys."""
        mod = _try_import(module_path, tool_name)
        profile = mod.EXECUTION_PROFILE
        assert isinstance(profile, dict)
        for subcmd, info in profile.items():
            assert isinstance(subcmd, str), f"{tool_name}.{subcmd}: key must be str"
            assert isinstance(info, dict), f"{tool_name}.{subcmd}: value must be dict"
            assert "expected_seconds" in info, (
                f"{tool_name}.{subcmd}: missing expected_seconds"
            )
            assert "background_eligible" in info, (
                f"{tool_name}.{subcmd}: missing background_eligible"
            )
            assert isinstance(info["expected_seconds"], (int, float))
            assert isinstance(info["background_eligible"], bool)


# ── Background eligibility ───────────────────────────────────


class TestBackgroundEligibility:
    @pytest.mark.parametrize(
        "tool_name,expected_subcmds", list(_EXPECTED_BG_ELIGIBLE.items()),
    )
    def test_eligible_subcommands(self, tool_name: str, expected_subcmds: set[str]):
        """Tools with long-running subcommands should have them marked eligible."""
        mod = _try_import(_TOOL_MODULES[tool_name], tool_name)
        profile = mod.EXECUTION_PROFILE
        actual_eligible = {
            name for name, info in profile.items() if info.get("background_eligible")
        }
        assert actual_eligible == expected_subcmds

    def test_short_tools_not_eligible(self):
        """Tools like web_search, github, aws_collector should have no eligible subcommands."""
        for tool_name in ("web_search", "github", "aws_collector"):
            mod = _try_import(_TOOL_MODULES[tool_name], tool_name)
            profile = mod.EXECUTION_PROFILE
            eligible = [
                name for name, info in profile.items() if info.get("background_eligible")
            ]
            assert not eligible, (
                f"{tool_name} should have no eligible subcommands, got {eligible}"
            )

    def test_optional_short_tools_not_eligible(self):
        """Tools with optional deps (gmail) should have no eligible subcommands."""
        for tool_name in ("gmail",):
            mod = _try_import(_TOOL_MODULES[tool_name], tool_name)
            profile = mod.EXECUTION_PROFILE
            eligible = [
                name for name, info in profile.items() if info.get("background_eligible")
            ]
            assert not eligible, (
                f"{tool_name} should have no eligible subcommands, got {eligible}"
            )


# ── load_execution_profiles ──────────────────────────────────


class TestLoadExecutionProfiles:
    def test_loads_core_profiles(self):
        """load_execution_profiles loads EXECUTION_PROFILE from core tool modules."""
        from core.tools import TOOL_MODULES

        profiles = load_execution_profiles(TOOL_MODULES)
        # Should have entries for all tools with EXECUTION_PROFILE
        assert "image_gen" in profiles
        assert "local_llm" in profiles
        assert "web_search" in profiles
        # Check structure
        assert "3d" in profiles["image_gen"]
        assert profiles["image_gen"]["3d"]["background_eligible"] is True

    def test_handles_missing_profile(self):
        """Modules without EXECUTION_PROFILE are silently skipped."""
        # ``os`` is a stdlib module that has no EXECUTION_PROFILE attribute
        profiles = load_execution_profiles({"nonexistent_fake": "os"})
        assert "nonexistent_fake" not in profiles

    def test_handles_import_error(self):
        """Non-importable modules are silently skipped."""
        profiles = load_execution_profiles({"bad_module": "this.does.not.exist"})
        assert "bad_module" not in profiles


# ── get_eligible_tools_from_profiles ─────────────────────────


class TestGetEligibleToolsFromProfiles:
    def test_extracts_eligible(self):
        """get_eligible_tools_from_profiles returns only background_eligible entries."""
        profiles = {
            "image_gen": {
                "3d": {"expected_seconds": 600, "background_eligible": True},
                "bustup": {"expected_seconds": 120, "background_eligible": True},
            },
            "web_search": {
                "search": {"expected_seconds": 10, "background_eligible": False},
            },
        }
        eligible = get_eligible_tools_from_profiles(profiles)
        assert "image_gen:3d" in eligible
        assert eligible["image_gen:3d"] == 600
        assert "image_gen:bustup" in eligible
        assert eligible["image_gen:bustup"] == 120
        assert "web_search:search" not in eligible

    def test_empty_profiles(self):
        """Empty profiles dict returns empty eligible dict."""
        assert get_eligible_tools_from_profiles({}) == {}


# ── BackgroundTaskManager.from_profiles ──────────────────────


class TestFromProfiles:
    def test_from_profiles_merges_defaults(self, anima_dir):
        """from_profiles creates manager with both default and profile-based eligible tools."""
        profiles = {
            "image_gen": {
                "3d": {"expected_seconds": 600, "background_eligible": True},
            },
        }
        mgr = BackgroundTaskManager.from_profiles(
            anima_dir, anima_name="test", profiles=profiles,
        )
        # Default tools should still be eligible
        assert mgr.is_eligible("generate_3d_model")
        # Profile-based tools should also be eligible
        assert mgr.is_eligible("image_gen:3d")

    def test_from_profiles_without_profiles(self, anima_dir):
        """from_profiles without profiles param uses only defaults."""
        mgr = BackgroundTaskManager.from_profiles(anima_dir, anima_name="test")
        assert mgr.is_eligible("generate_3d_model")
        assert not mgr.is_eligible("image_gen:3d")

    def test_from_profiles_profile_overrides_default_timeout(self, anima_dir):
        """Profile-derived entries are merged; profile values take precedence."""
        profiles = {
            "custom": {
                "slow": {"expected_seconds": 900, "background_eligible": True},
            },
        }
        mgr = BackgroundTaskManager.from_profiles(
            anima_dir, anima_name="test", profiles=profiles,
        )
        # custom:slow should be eligible from profiles
        assert mgr.is_eligible("custom:slow")
        # Non-eligible profile entries should NOT appear
        assert not mgr.is_eligible("custom:fast")
