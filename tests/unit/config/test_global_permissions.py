"""Tests for core.config.global_permissions — GlobalPermissionsCache."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config.global_permissions import (
    GlobalPermissionsCache,
    _build_injection_re,
    _compile_patterns,
)
from core.config.schemas import GlobalDenyPattern

# ── Helpers ──────────────────────────────────────────────────

MINIMAL_CONFIG = {
    "version": 1,
    "injection_patterns": [{"pattern": "[;\\n]", "reason": "injection"}],
    "commands": {
        "deny": [
            {"pattern": "(?i)\\brm\\s+(-\\S+\\s+)*/(?!\\w)", "reason": "rm root blocked"},
            {"pattern": "\\bmkfs\\b", "reason": "mkfs blocked"},
        ]
    },
}


def _write_config(path: Path, config: dict | None = None) -> Path:
    fp = path / "permissions.global.json"
    fp.write_text(json.dumps(config or MINIMAL_CONFIG), encoding="utf-8")
    return fp


# ── _compile_patterns ────────────────────────────────────────


class TestCompilePatterns:
    def test_compiles_valid_patterns(self):
        items = [
            GlobalDenyPattern(pattern="\\brm\\b", reason="rm"),
            GlobalDenyPattern(pattern="\\bls\\b", reason="ls"),
        ]
        result = _compile_patterns(items)
        assert len(result) == 2
        assert result[0][0].search("rm -rf /")
        assert result[1][1] == "ls"

    def test_skips_invalid_regex(self):
        items = [
            GlobalDenyPattern(pattern="[invalid", reason="bad"),
            GlobalDenyPattern(pattern="\\bgood\\b", reason="good"),
        ]
        result = _compile_patterns(items)
        assert len(result) == 1
        assert result[0][1] == "good"

    def test_empty_list(self):
        assert _compile_patterns([]) == []


# ── _build_injection_re ──────────────────────────────────────


class TestBuildInjectionRe:
    def test_combines_patterns(self):
        items = [
            GlobalDenyPattern(pattern="[;`]", reason="a"),
            GlobalDenyPattern(pattern="\\$\\(", reason="b"),
        ]
        result = _build_injection_re(items)
        assert result is not None
        assert result.search(";")
        assert result.search("$(cmd)")
        assert not result.search("hello")

    def test_returns_none_for_empty(self):
        assert _build_injection_re([]) is None

    def test_skips_invalid_patterns(self):
        items = [
            GlobalDenyPattern(pattern="[bad", reason="bad"),
        ]
        assert _build_injection_re(items) is None


# ── GlobalPermissionsCache ───────────────────────────────────


class TestGlobalPermissionsCache:
    @pytest.fixture(autouse=True)
    def _reset(self):
        GlobalPermissionsCache.reset()
        yield
        GlobalPermissionsCache.reset()

    def test_singleton(self):
        a = GlobalPermissionsCache.get()
        b = GlobalPermissionsCache.get()
        assert a is b

    def test_reset(self):
        a = GlobalPermissionsCache.get()
        GlobalPermissionsCache.reset()
        b = GlobalPermissionsCache.get()
        assert a is not b

    def test_load_success(self, tmp_path: Path):
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)

        assert cache.loaded
        assert cache.injection_re is not None
        assert cache.injection_re.search(";")
        assert len(cache.blocked_patterns) == 2
        assert cache.blocked_patterns[0][0].search("rm -rf /")
        assert not cache.blocked_patterns[0][0].search("rm -rf /home")

    def test_load_missing_file_raises(self, tmp_path: Path):
        cache = GlobalPermissionsCache.get()
        with pytest.raises(FileNotFoundError, match="permissions.global.json not found"):
            cache.load(tmp_path / "nonexistent.json", interactive=False)

    def test_load_invalid_json_raises(self, tmp_path: Path):
        fp = tmp_path / "permissions.global.json"
        fp.write_text("{invalid json", encoding="utf-8")
        cache = GlobalPermissionsCache.get()
        with pytest.raises((json.JSONDecodeError, ValueError)):
            cache.load(fp, interactive=False)

    def test_load_creates_hash_file(self, tmp_path: Path):
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)

        hash_path = tmp_path / "run" / "permissions_global.sha256"
        assert hash_path.exists()
        content = fp.read_text(encoding="utf-8")
        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        assert hash_path.read_text().strip() == expected_hash

    def test_not_loaded_defaults(self):
        cache = GlobalPermissionsCache.get()
        assert not cache.loaded
        assert cache.injection_re is None
        assert cache.blocked_patterns == []

    def test_empty_config(self, tmp_path: Path):
        fp = tmp_path / "permissions.global.json"
        fp.write_text(json.dumps({"version": 1}), encoding="utf-8")
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)
        assert cache.loaded
        assert cache.injection_re is None
        assert cache.blocked_patterns == []


# ── Integrity Check ──────────────────────────────────────────


class TestIntegrityCheck:
    @pytest.fixture(autouse=True)
    def _reset(self):
        GlobalPermissionsCache.reset()
        yield
        GlobalPermissionsCache.reset()

    def test_intact_returns_true(self, tmp_path: Path):
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)
        assert cache.check_integrity() is True

    def test_tampered_restores_and_returns_false(self, tmp_path: Path):
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)

        original = fp.read_text(encoding="utf-8")

        # Tamper with file
        tampered = json.loads(original)
        tampered["commands"]["deny"] = []
        fp.write_text(json.dumps(tampered), encoding="utf-8")

        assert cache.check_integrity() is False

        # File should be restored
        assert fp.read_text(encoding="utf-8") == original

    def test_missing_file_restores(self, tmp_path: Path):
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)
        original = fp.read_text(encoding="utf-8")

        fp.unlink()
        assert cache.check_integrity() is False
        assert fp.read_text(encoding="utf-8") == original

    def test_not_loaded_returns_true(self):
        cache = GlobalPermissionsCache.get()
        assert cache.check_integrity() is True


# ── Startup Hash Mismatch ────────────────────────────────────


class TestStartupHashCheck:
    @pytest.fixture(autouse=True)
    def _reset(self):
        GlobalPermissionsCache.reset()
        yield
        GlobalPermissionsCache.reset()

    def test_first_load_no_prompt(self, tmp_path: Path):
        """First load with no existing hash file should succeed without prompt."""
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=True)
        assert cache.loaded

    def test_same_hash_no_prompt(self, tmp_path: Path):
        """Second load with same content should succeed without prompt."""
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)

        GlobalPermissionsCache.reset()
        cache2 = GlobalPermissionsCache.get()
        cache2.load(fp, interactive=True)
        assert cache2.loaded

    def test_hash_mismatch_non_tty_exits(self, tmp_path: Path):
        """Non-TTY session with hash mismatch should raise SystemExit."""
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)

        # Modify file to create mismatch
        modified = MINIMAL_CONFIG.copy()
        modified["version"] = 2
        fp.write_text(json.dumps(modified), encoding="utf-8")

        GlobalPermissionsCache.reset()
        cache2 = GlobalPermissionsCache.get()

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit, match="non-interactive"):
                cache2.load(fp, interactive=True)

    def test_hash_mismatch_tty_accepted(self, tmp_path: Path):
        """TTY session where user types 'yes' should proceed."""
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)

        modified = MINIMAL_CONFIG.copy()
        modified["version"] = 2
        fp.write_text(json.dumps(modified), encoding="utf-8")

        GlobalPermissionsCache.reset()
        cache2 = GlobalPermissionsCache.get()

        with patch("sys.stdin") as mock_stdin, patch("builtins.input", return_value="yes"):
            mock_stdin.isatty.return_value = True
            cache2.load(fp, interactive=True)
            assert cache2.loaded

    def test_hash_mismatch_tty_rejected(self, tmp_path: Path):
        """TTY session where user types 'no' should raise SystemExit."""
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)

        modified = MINIMAL_CONFIG.copy()
        modified["version"] = 2
        fp.write_text(json.dumps(modified), encoding="utf-8")

        GlobalPermissionsCache.reset()
        cache2 = GlobalPermissionsCache.get()

        with patch("sys.stdin") as mock_stdin, patch("builtins.input", return_value="no"):
            mock_stdin.isatty.return_value = True
            with pytest.raises(SystemExit, match="rejected"):
                cache2.load(fp, interactive=True)

    def test_interactive_false_skips_prompt(self, tmp_path: Path):
        """interactive=False should skip hash check entirely."""
        fp = _write_config(tmp_path)
        cache = GlobalPermissionsCache.get()
        cache.load(fp, interactive=False)

        modified = MINIMAL_CONFIG.copy()
        modified["version"] = 2
        fp.write_text(json.dumps(modified), encoding="utf-8")

        GlobalPermissionsCache.reset()
        cache2 = GlobalPermissionsCache.get()
        cache2.load(fp, interactive=False)
        assert cache2.loaded


# ── Default Template Validation ──────────────────────────────


class TestDefaultTemplate:
    def test_default_template_loads(self):
        """The shipped default permissions.global.json template is valid."""
        from core.paths import TEMPLATES_DIR

        src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
        assert src.exists(), "Default template missing"

        data = json.loads(src.read_text(encoding="utf-8"))
        from core.config.schemas import GlobalPermissionsConfig

        config = GlobalPermissionsConfig.model_validate(data)
        assert config.version == 1
        assert len(config.injection_patterns) >= 1
        assert len(config.commands.deny) >= 20

    def test_all_patterns_compile(self):
        """All regex patterns in the default template must compile."""
        import re

        from core.paths import TEMPLATES_DIR

        src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
        data = json.loads(src.read_text(encoding="utf-8"))

        for entry in data.get("injection_patterns", []):
            re.compile(entry["pattern"])

        for entry in data.get("commands", {}).get("deny", []):
            re.compile(entry["pattern"])

    def test_animaworks_restart_blocked(self):
        """Default template must block animaworks restart/stop/start."""
        from core.paths import TEMPLATES_DIR

        src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
        data = json.loads(src.read_text(encoding="utf-8"))
        from core.config.schemas import GlobalPermissionsConfig

        config = GlobalPermissionsConfig.model_validate(data)
        compiled = _compile_patterns(config.commands.deny)

        for cmd in ["animaworks restart", "animaworks stop", "animaworks start"]:
            assert any(p.search(cmd) for p, _ in compiled), f"{cmd!r} should be blocked"

    def test_systemctl_animaworks_blocked(self):
        """Default template must block systemctl restart animaworks."""
        from core.paths import TEMPLATES_DIR

        src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
        data = json.loads(src.read_text(encoding="utf-8"))
        from core.config.schemas import GlobalPermissionsConfig

        config = GlobalPermissionsConfig.model_validate(data)
        compiled = _compile_patterns(config.commands.deny)

        cmd = "systemctl restart animaworks"
        assert any(p.search(cmd) for p, _ in compiled), f"{cmd!r} should be blocked"
