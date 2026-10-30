from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import gen_reference


def test_build_parser_does_not_load_config():
    from cli.parser import build_parser

    with patch("core.config.load_config") as load_config:
        parser = build_parser()

    assert parser.description == "AnimaWorks - Digital Anima Framework"
    load_config.assert_not_called()


def test_cli_extraction_contains_real_command_paths_only():
    _, payload, _ = gen_reference.extract_cli()
    command_paths = {command["path"] for command in payload["commands"]}

    assert {"task board", "anima set-model", "migrate"} <= command_paths
    assert "config export-sections" not in command_paths


def test_api_extraction_includes_setup_websocket_and_auth_classes():
    body, _, _ = gen_reference.extract_api()

    assert "/api/setup/environment" in body
    assert "| WS | `/ws` |" in body
    assert gen_reference._auth_classification("POST", "/api/auth/login") == "不要（除外一覧）"
    assert gen_reference._auth_classification("POST", "/api/internal/embed") == "内部"
    assert "/api/internal/vector/query" in body
    assert "| POST | `/api/internal/vector/query` | 内部 |" in body


def test_config_omits_retired_priming_profile_and_has_localized_sections():
    _, payload, _ = gen_reference.extract_config()
    keys = {row["key"] for row in payload["config"]}
    direct_sections = [row for row in payload["config"] if row["model"] == "AnimaWorksConfig"]

    assert "priming.profile" not in keys
    assert "priming.max_tokens" in keys
    assert all(row["description"] != "—" for row in direct_sections)


def test_config_requires_dictionary_entries_for_top_level_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(gen_reference, "DICTIONARY_DIR", tmp_path)

    with pytest.raises(gen_reference.ReferenceGenerationError, match="Missing required AnimaWorksConfig descriptions"):
        gen_reference.extract_config()


def test_stale_dictionary_key_fails_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "cli.yaml").write_text('not-a-real-command: "存在しないコマンド"\n', encoding="utf-8")
    monkeypatch.setattr(gen_reference, "DICTIONARY_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["gen_reference.py", "cli", "--check"])

    with pytest.raises(SystemExit) as exc_info:
        gen_reference.main()

    assert exc_info.value.code == 1


def test_module_file_discovery_uses_current_worktree_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tracked = tmp_path / "core" / "memory" / "__init__.py"
    untracked = tmp_path / "core" / "memory" / "new_module.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("", encoding="utf-8")
    untracked.write_text("", encoding="utf-8")
    monkeypatch.setattr(gen_reference, "PROJECT_DIR", tmp_path)

    git_files = "core/memory/__init__.py\\ncore/memory/deleted.py\\n"
    with patch(
        "scripts.gen_reference.subprocess.run",
        return_value=SimpleNamespace(stdout=git_files),
    ):
        files = gen_reference._tracked_python_files(["core"])

    assert files == [tracked, untracked]


# Builds every reference twice: about 15 s locally, and over 30 s in the
# PyPI publish job on GitHub runners.
@pytest.mark.timeout(120)
def test_generation_is_byte_deterministic():
    for kind in gen_reference.KINDS:
        first, _ = gen_reference._make_reference(kind)
        second, _ = gen_reference._make_reference(kind)
        assert first.encode("utf-8") == second.encode("utf-8"), kind


def test_unset_data_dir_never_creates_default_runtime_under_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANIMAWORKS_DATA_DIR", raising=False)

    isolated_dir = gen_reference.ensure_isolated_data_dir()

    assert isolated_dir.exists()
    assert Path.home() / ".animaworks" == tmp_path / ".animaworks"
    assert not (tmp_path / ".animaworks").exists()
