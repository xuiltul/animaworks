"""Unit tests for core/config/cli.py — config CLI subcommands."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.cli import (
    _coerce_value,
    _flatten_dict,
    _mask_secret,
    _set_nested,
    cmd_config_get,
    cmd_config_list,
    cmd_config_set,
    cmd_config_dispatch,
)
from core.config.models import AnimaWorksConfig, invalidate_cache


# ── _flatten_dict ─────────────────────────────────────────


class TestFlattenDict:
    def test_flat_dict(self):
        result = _flatten_dict({"a": 1, "b": 2})
        assert ("a", 1) in result
        assert ("b", 2) in result

    def test_nested_dict(self):
        result = _flatten_dict({"x": {"y": {"z": 3}}})
        assert ("x.y.z", 3) in result

    def test_empty_dict(self):
        assert _flatten_dict({}) == []

    def test_mixed_depth(self):
        result = _flatten_dict({"a": 1, "b": {"c": 2}})
        keys = [k for k, _ in result]
        assert "a" in keys
        assert "b.c" in keys

    def test_prefix(self):
        result = _flatten_dict({"a": 1}, prefix="root")
        assert ("root.a", 1) in result


# ── _mask_secret ──────────────────────────────────────────


class TestMaskSecret:
    def test_masks_api_key(self):
        result = _mask_secret("credentials.anthropic.api_key", "sk-abcdef1234567890")
        assert result == "sk-abcde..."

    def test_no_mask_for_non_key(self):
        result = _mask_secret("system.mode", "server")
        assert result == "server"

    def test_empty_api_key(self):
        result = _mask_secret("api_key", "")
        assert result == ""

    def test_none_value(self):
        result = _mask_secret("api_key", None)
        assert result == "None"

    def test_short_api_key(self):
        result = _mask_secret("api_key", "short")
        assert result == "short..."


# ── _coerce_value ─────────────────────────────────────────


class TestCoerceValue:
    def test_null(self):
        assert _coerce_value("null") is None
        assert _coerce_value("none") is None
        assert _coerce_value("None") is None
        assert _coerce_value("NULL") is None

    def test_bool_true(self):
        assert _coerce_value("true") is True
        assert _coerce_value("True") is True
        assert _coerce_value("TRUE") is True

    def test_bool_false(self):
        assert _coerce_value("false") is False
        assert _coerce_value("False") is False

    def test_int(self):
        assert _coerce_value("42") == 42
        assert _coerce_value("0") == 0
        assert _coerce_value("-1") == -1

    def test_float(self):
        assert _coerce_value("3.14") == 3.14
        assert _coerce_value("0.5") == 0.5

    def test_string(self):
        assert _coerce_value("hello") == "hello"
        assert _coerce_value("claude-sonnet-4") == "claude-sonnet-4"


# ── _set_nested ───────────────────────────────────────────


class TestSetNested:
    def test_simple(self):
        d: dict = {}
        _set_nested(d, ["a"], 1)
        assert d == {"a": 1}

    def test_nested(self):
        d: dict = {}
        _set_nested(d, ["a", "b", "c"], 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_existing_dict(self):
        d = {"a": {"b": 1}}
        _set_nested(d, ["a", "c"], 2)
        assert d == {"a": {"b": 1, "c": 2}}

    def test_overwrite_non_dict(self):
        d = {"a": "string"}
        _set_nested(d, ["a", "b"], 1)
        assert d == {"a": {"b": 1}}


# ── cmd_config_dispatch ───────────────────────────────────


class TestCmdConfigDispatch:
    def test_no_subcommand_prints_help(self, data_dir):
        mock_parser = MagicMock()
        args = argparse.Namespace(
            interactive=False,
            config_command=None,
            config_parser=mock_parser,
        )
        cmd_config_dispatch(args)
        mock_parser.print_help.assert_called_once()

    def test_interactive_flag(self, data_dir):
        args = argparse.Namespace(interactive=True)
        with patch("core.config.cli._interactive_setup") as mock_wizard:
            cmd_config_dispatch(args)
            mock_wizard.assert_called_once()


# ── cmd_config_get ────────────────────────────────────────


class TestCmdConfigGet:
    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    def test_get_existing_key(self, data_dir, capsys):
        args = argparse.Namespace(key="version", show_secrets=False)
        cmd_config_get(args)
        captured = capsys.readouterr()
        assert "1" in captured.out

    def test_get_nested_key(self, data_dir, capsys):
        args = argparse.Namespace(key="system.mode", show_secrets=False)
        cmd_config_get(args)
        captured = capsys.readouterr()
        assert "server" in captured.out

    def test_get_missing_key(self, data_dir):
        args = argparse.Namespace(key="nonexistent.key", show_secrets=False)
        with pytest.raises(SystemExit):
            cmd_config_get(args)


# ── cmd_config_set ────────────────────────────────────────


class TestCmdConfigSet:
    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    def test_set_value(self, data_dir, capsys):
        args = argparse.Namespace(key="system.log_level", value="DEBUG")
        cmd_config_set(args)
        captured = capsys.readouterr()
        assert "DEBUG" in captured.out

        # Verify persisted
        invalidate_cache()
        from core.config.models import load_config
        config = load_config(data_dir / "config.json")
        assert config.system.log_level == "DEBUG"

    def test_set_new_anima(self, data_dir, capsys):
        args = argparse.Namespace(key="animas.newanima.model", value="gpt-4o")
        cmd_config_set(args)

        invalidate_cache()
        from core.config.models import load_config
        config = load_config(data_dir / "config.json")
        assert "newanima" in config.animas
        assert config.animas["newanima"].model == "gpt-4o"

    def test_set_new_credential(self, data_dir, capsys):
        args = argparse.Namespace(key="credentials.openrouter.api_key", value="sk-test")
        cmd_config_set(args)

        invalidate_cache()
        from core.config.models import load_config
        config = load_config(data_dir / "config.json")
        assert "openrouter" in config.credentials
        assert config.credentials["openrouter"].api_key == "sk-test"


# ── cmd_config_list ───────────────────────────────────────


class TestCmdConfigList:
    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    def test_list_all(self, data_dir, capsys):
        args = argparse.Namespace(section=None, show_secrets=False)
        cmd_config_list(args)
        captured = capsys.readouterr()
        assert "version" in captured.out
        assert "system.mode" in captured.out

    def test_list_section(self, data_dir, capsys):
        args = argparse.Namespace(section="system", show_secrets=False)
        cmd_config_list(args)
        captured = capsys.readouterr()
        assert "system.mode" in captured.out
        # version should NOT appear
        lines = captured.out.strip().split("\n")
        assert all("system" in line.split(" = ")[0] for line in lines if line.strip())

    def test_list_masks_secrets(self, data_dir, capsys):
        # Set an API key first
        config_path = data_dir / "config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data["credentials"]["anthropic"]["api_key"] = "sk-abcdef1234567890"
        config_path.write_text(json.dumps(data), encoding="utf-8")
        invalidate_cache()

        args = argparse.Namespace(section="credentials", show_secrets=False)
        cmd_config_list(args)
        captured = capsys.readouterr()
        assert "sk-abcde..." in captured.out
        assert "sk-abcdef1234567890" not in captured.out
