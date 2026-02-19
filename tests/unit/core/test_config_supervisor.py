# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for supervisor-related functions in core/config/models.py.

Tests _resolve_supervisor_name(), read_anima_supervisor(), and
register_anima_in_config().
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core.config.models import (
    AnimaWorksConfig,
    AnimaModelConfig,
    _resolve_supervisor_name,
    invalidate_cache,
    load_config,
    read_anima_supervisor,
    register_anima_in_config,
    save_config,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    """Invalidate the config singleton before and after each test."""
    invalidate_cache()
    yield  # type: ignore[misc]
    invalidate_cache()


def _make_anima(
    tmp_path: Path,
    name: str,
    identity_content: str | None = None,
    status_content: dict | None = None,
) -> Path:
    """Helper to create an anima directory with optional identity.md and status.json."""
    anima_dir = tmp_path / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    if identity_content is not None:
        (anima_dir / "identity.md").write_text(identity_content, encoding="utf-8")
    if status_content is not None:
        (anima_dir / "status.json").write_text(
            json.dumps(status_content, ensure_ascii=False), encoding="utf-8",
        )
    return anima_dir


# ── _resolve_supervisor_name ─────────────────────────────────────


class TestResolveSupervisorName:
    """Tests for the raw supervisor name resolution function."""

    def test_fullwidth_parens(self) -> None:
        """Japanese name with full-width parenthesised English name."""
        assert _resolve_supervisor_name("琴葉（kotoha）") == "kotoha"

    def test_halfwidth_parens(self) -> None:
        """Japanese name with half-width parenthesised English name."""
        assert _resolve_supervisor_name("琴葉(kotoha)") == "kotoha"

    def test_plain_ascii(self) -> None:
        """Plain ASCII name is returned as-is (lowered)."""
        assert _resolve_supervisor_name("sakura") == "sakura"

    def test_uppercase_lowered(self) -> None:
        """Uppercase ASCII name is lowered."""
        assert _resolve_supervisor_name("Sakura") == "sakura"

    def test_nashi_halfwidth_parens(self) -> None:
        """Half-width (なし) returns None."""
        assert _resolve_supervisor_name("(なし)") is None

    def test_nashi_bare(self) -> None:
        """Bare なし returns None."""
        assert _resolve_supervisor_name("なし") is None

    def test_nashi_fullwidth_parens(self) -> None:
        """Full-width （なし） returns None."""
        assert _resolve_supervisor_name("（なし）") is None

    def test_dash(self) -> None:
        """Dash value returns None."""
        assert _resolve_supervisor_name("-") is None

    def test_empty_string(self) -> None:
        """Empty string returns None."""
        assert _resolve_supervisor_name("") is None

    def test_whitespace_only(self) -> None:
        """Whitespace-only string returns None after stripping."""
        assert _resolve_supervisor_name("   ") is None

    def test_japanese_only_no_english(self, caplog: pytest.LogCaptureFixture) -> None:
        """Japanese-only name without English in parens returns None with warning."""
        with caplog.at_level(logging.WARNING, logger="animaworks.config"):
            result = _resolve_supervisor_name("純日本語名")
        assert result is None
        assert "no English name" in caplog.text

    def test_mixed_case_in_parens(self) -> None:
        """Mixed-case English name in parens is lowered."""
        assert _resolve_supervisor_name("凛堂 凛（Rin）") == "rin"

    def test_underscore_in_name(self) -> None:
        """Name with underscores is accepted."""
        assert _resolve_supervisor_name("chatwork_checker") == "chatwork_checker"

    def test_triple_dash(self) -> None:
        """Triple-dash is treated as none value."""
        assert _resolve_supervisor_name("---") is None

    def test_leading_trailing_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped before resolution."""
        assert _resolve_supervisor_name("  sakura  ") == "sakura"


# ── read_anima_supervisor ───────────────────────────────────────


class TestReadAnimaSupervisor:
    """Tests for reading supervisor from status.json / identity.md."""

    def test_status_json_with_supervisor(self, tmp_path: Path) -> None:
        """Reads supervisor from status.json when present."""
        anima_dir = _make_anima(
            tmp_path, "hinata",
            status_content={"supervisor": "kotoha"},
        )
        assert read_anima_supervisor(anima_dir) == "kotoha"

    def test_status_json_without_supervisor_falls_to_identity(
        self, tmp_path: Path,
    ) -> None:
        """Falls through to identity.md when status.json lacks supervisor."""
        anima_dir = _make_anima(
            tmp_path, "hinata",
            identity_content="| 上司 | 琴葉（kotoha） |\n",
            status_content={"enabled": True},
        )
        assert read_anima_supervisor(anima_dir) == "kotoha"

    def test_identity_md_table_fullwidth_parens(self, tmp_path: Path) -> None:
        """Parses supervisor from identity.md table with full-width parens."""
        anima_dir = _make_anima(
            tmp_path, "hinata",
            identity_content=(
                "| 項目 | 設定 |\n"
                "|------|------|\n"
                "| 上司 | 琴葉（kotoha） |\n"
            ),
        )
        assert read_anima_supervisor(anima_dir) == "kotoha"

    def test_no_status_json_no_identity_md(self, tmp_path: Path) -> None:
        """Returns None when neither file exists."""
        anima_dir = _make_anima(tmp_path, "hinata")
        assert read_anima_supervisor(anima_dir) is None

    def test_status_json_empty_supervisor_falls_to_identity(
        self, tmp_path: Path,
    ) -> None:
        """Falls through to identity.md when status.json supervisor is empty."""
        anima_dir = _make_anima(
            tmp_path, "hinata",
            identity_content="| 上司 | sakura |\n",
            status_content={"supervisor": ""},
        )
        assert read_anima_supervisor(anima_dir) == "sakura"

    def test_status_json_takes_priority(self, tmp_path: Path) -> None:
        """status.json supervisor wins over identity.md when both present."""
        anima_dir = _make_anima(
            tmp_path, "hinata",
            identity_content="| 上司 | alice |\n",
            status_content={"supervisor": "bob"},
        )
        assert read_anima_supervisor(anima_dir) == "bob"

    def test_status_json_nashi_falls_to_identity(self, tmp_path: Path) -> None:
        """status.json with なし resolves to None, falls to identity.md."""
        anima_dir = _make_anima(
            tmp_path, "hinata",
            identity_content="| 上司 | sakura |\n",
            status_content={"supervisor": "なし"},
        )
        # "なし" resolves to None in _resolve_supervisor_name, so it falls through
        assert read_anima_supervisor(anima_dir) == "sakura"

    def test_identity_no_supervisor_row(self, tmp_path: Path) -> None:
        """Returns None when identity.md has no supervisor row."""
        anima_dir = _make_anima(
            tmp_path, "hinata",
            identity_content="# Identity\n| 項目 | 設定 |\n| 誕生日 | 1月1日 |\n",
        )
        assert read_anima_supervisor(anima_dir) is None

    def test_invalid_status_json(self, tmp_path: Path) -> None:
        """Returns None when status.json is invalid JSON."""
        anima_dir = tmp_path / "hinata"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text("not valid json", encoding="utf-8")
        assert read_anima_supervisor(anima_dir) is None

    def test_identity_embedded_in_real_file(self, tmp_path: Path) -> None:
        """Extracts supervisor from a realistic multi-row identity.md."""
        anima_dir = _make_anima(
            tmp_path, "hinata",
            identity_content=(
                "# Identity: hinata\n\n"
                "## 基本プロフィール\n\n"
                "| 項目 | 設定 |\n"
                "|------|------|\n"
                "| 誕生日 | 3月21日 |\n"
                "| 上司 | 凛堂 凛（rin） |\n"
                "| 身長 | 155cm |\n\n"
                "## 性格\nGenki.\n"
            ),
        )
        assert read_anima_supervisor(anima_dir) == "rin"

    def test_nonexistent_anima_dir(self, tmp_path: Path) -> None:
        """Returns None when anima directory does not exist."""
        assert read_anima_supervisor(tmp_path / "nonexistent") is None


# ── register_anima_in_config ────────────────────────────────────


class TestRegisterAnimaInConfig:
    """Tests for registering an anima in config.json with supervisor synced."""

    def test_registers_new_anima_with_supervisor(self, tmp_path: Path) -> None:
        """New anima with status.json supervisor is registered correctly."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()

        # Create config.json
        config = AnimaWorksConfig(setup_complete=True)
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        # Create anima directory with supervisor
        _make_anima(
            animas_dir, "hinata",
            identity_content="| 上司 | sakura |\n",
        )

        register_anima_in_config(data_dir, "hinata")

        invalidate_cache()
        cfg = load_config(data_dir / "config.json")
        assert "hinata" in cfg.animas
        assert cfg.animas["hinata"].supervisor == "sakura"

    def test_registers_new_anima_without_supervisor(self, tmp_path: Path) -> None:
        """New anima without supervisor source is registered with supervisor=None."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()

        # Create config.json
        config = AnimaWorksConfig(setup_complete=True)
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        # Create anima directory without supervisor info
        _make_anima(
            animas_dir, "hinata",
            identity_content="# Identity\nNo supervisor info.\n",
        )

        register_anima_in_config(data_dir, "hinata")

        invalidate_cache()
        cfg = load_config(data_dir / "config.json")
        assert "hinata" in cfg.animas
        assert cfg.animas["hinata"].supervisor is None

    def test_existing_anima_is_noop(self, tmp_path: Path) -> None:
        """Anima already in config is not modified."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()

        # Create config.json with existing anima
        config = AnimaWorksConfig(setup_complete=True)
        config.animas["hinata"] = AnimaModelConfig(
            model="openai/gpt-4o",
            supervisor="original",
        )
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        # Create anima directory with different supervisor
        _make_anima(
            animas_dir, "hinata",
            identity_content="| 上司 | different |\n",
        )

        register_anima_in_config(data_dir, "hinata")

        invalidate_cache()
        cfg = load_config(data_dir / "config.json")
        assert cfg.animas["hinata"].supervisor == "original"
        assert cfg.animas["hinata"].model == "openai/gpt-4o"

    def test_config_json_does_not_exist(self, tmp_path: Path) -> None:
        """No error when config.json does not exist."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # No config.json created — should return without error
        register_anima_in_config(data_dir, "hinata")
        # No assertion needed — just verifying no exception is raised

    def test_anima_dir_does_not_exist(self, tmp_path: Path) -> None:
        """Registers anima with supervisor=None when anima dir is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create config.json but no animas directory
        config = AnimaWorksConfig(setup_complete=True)
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        register_anima_in_config(data_dir, "hinata")

        invalidate_cache()
        cfg = load_config(data_dir / "config.json")
        assert "hinata" in cfg.animas
        assert cfg.animas["hinata"].supervisor is None

    def test_registers_with_status_json_supervisor(self, tmp_path: Path) -> None:
        """Anima registered with supervisor from status.json when identity.md has none."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()

        config = AnimaWorksConfig(setup_complete=True)
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        _make_anima(
            animas_dir, "hinata",
            identity_content="# Identity\nNo supervisor.\n",
            status_content={"supervisor": "kotoha"},
        )

        register_anima_in_config(data_dir, "hinata")

        invalidate_cache()
        cfg = load_config(data_dir / "config.json")
        assert cfg.animas["hinata"].supervisor == "kotoha"
