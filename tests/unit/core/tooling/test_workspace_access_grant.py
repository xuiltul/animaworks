from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from unittest.mock import MagicMock

from core.config.models import AnimaModelConfig, AnimaWorksConfig, load_config, save_config
from core.execution.codex_sdk import CodexSDKExecutor
from core.schemas import ModelConfig
from core.tooling.handler import ToolHandler
from core.tooling.schemas import build_tool_list, build_unified_tool_list


def _write_config(animas: dict[str, AnimaModelConfig]) -> None:
    save_config(AnimaWorksConfig(animas=animas))


def _make_anima(data_dir: Path, name: str, *, file_roots: list[str] | None = None) -> Path:
    anima_dir = data_dir / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "permissions.json").write_text(
        json.dumps({"version": 1, "file_roots": file_roots or []}, indent=2),
        encoding="utf-8",
    )
    (anima_dir / "status.json").write_text("{}", encoding="utf-8")
    return anima_dir


def _handler(anima_dir: Path) -> ToolHandler:
    return ToolHandler(
        anima_dir=anima_dir,
        memory=MagicMock(),
        messenger=None,
        tool_registry=[],
    )


def _grant(handler: ToolHandler, alias: str, path: Path, **kwargs: object) -> dict:
    result = handler.handle(
        "grant_workspace_access",
        {
            "alias": alias,
            "path": str(path),
            **kwargs,
        },
    )
    return json.loads(result)


def test_top_level_human_can_grant_self_workspace(data_dir: Path, tmp_path: Path) -> None:
    top_dir = _make_anima(data_dir, "ritsu")
    _write_config({"ritsu": AnimaModelConfig(supervisor=None)})
    workspace = tmp_path / "finance-dashboard"
    workspace.mkdir()

    handler = _handler(top_dir)
    handler.set_session_origin("human")
    parsed = _grant(handler, "finance-dashboard", workspace)

    assert parsed["status"] == "ok"
    assert parsed["target_anima"] == "ritsu"
    assert parsed["permissions_changed"] is True
    assert load_config().workspaces["finance-dashboard"] == str(workspace.resolve())

    permissions = json.loads((top_dir / "permissions.json").read_text(encoding="utf-8"))
    assert str(workspace.resolve()) in permissions["file_roots"]

    status = json.loads((top_dir / "status.json").read_text(encoding="utf-8"))
    assert status["default_workspace"] == parsed["qualified_alias"]


def test_granted_workspace_is_in_next_codex_writable_roots(data_dir: Path, tmp_path: Path) -> None:
    top_dir = _make_anima(data_dir, "ritsu")
    _write_config({"ritsu": AnimaModelConfig(supervisor=None)})
    workspace = tmp_path / "finance-dashboard"
    workspace.mkdir()

    handler = _handler(top_dir)
    handler.set_session_origin("human")
    _grant(handler, "finance-dashboard", workspace)

    model_config = ModelConfig(model="codex/o4-mini", credential="openai", api_key="test")
    executor = CodexSDKExecutor(model_config=model_config, anima_dir=top_dir)
    executor._write_codex_config("prompt")
    config_toml = (top_dir / ".codex_home" / "config.toml").read_text(encoding="utf-8")

    assert "workspace-write" in config_toml
    assert str(workspace.resolve()) in config_toml


def test_non_top_level_cannot_self_grant(data_dir: Path, tmp_path: Path) -> None:
    _make_anima(data_dir, "owner")
    child_dir = _make_anima(data_dir, "ritsu")
    _write_config(
        {
            "owner": AnimaModelConfig(supervisor=None),
            "ritsu": AnimaModelConfig(supervisor="owner"),
        }
    )
    workspace = tmp_path / "finance-dashboard"
    workspace.mkdir()

    handler = _handler(child_dir)
    handler.set_session_origin("human")
    parsed = _grant(handler, "finance-dashboard", workspace)

    assert parsed["status"] == "error"
    assert parsed["error_type"] == "PermissionDenied"
    permissions = json.loads((child_dir / "permissions.json").read_text(encoding="utf-8"))
    assert str(workspace.resolve()) not in permissions["file_roots"]


def test_top_level_can_grant_descendant_workspace(data_dir: Path, tmp_path: Path) -> None:
    top_dir = _make_anima(data_dir, "owner")
    _make_anima(data_dir, "manager")
    child_dir = _make_anima(data_dir, "ritsu")
    _write_config(
        {
            "owner": AnimaModelConfig(supervisor=None),
            "manager": AnimaModelConfig(supervisor="owner"),
            "ritsu": AnimaModelConfig(supervisor="manager"),
        }
    )
    workspace = tmp_path / "finance-dashboard"
    workspace.mkdir()

    handler = _handler(top_dir)
    handler.set_session_origin("human")
    parsed = _grant(handler, "finance-dashboard", workspace, target_anima="ritsu")

    assert parsed["status"] == "ok"
    assert parsed["target_anima"] == "ritsu"
    permissions = json.loads((child_dir / "permissions.json").read_text(encoding="utf-8"))
    assert str(workspace.resolve()) in permissions["file_roots"]


def test_non_human_origin_is_denied(data_dir: Path, tmp_path: Path) -> None:
    top_dir = _make_anima(data_dir, "ritsu")
    _write_config({"ritsu": AnimaModelConfig(supervisor=None)})
    workspace = tmp_path / "finance-dashboard"
    workspace.mkdir()

    handler = _handler(top_dir)
    handler.set_session_origin("system")
    handler._trigger = "heartbeat"
    parsed = _grant(handler, "finance-dashboard", workspace)

    assert parsed["status"] == "error"
    assert parsed["error_type"] == "PermissionDenied"


def test_anima_home_path_is_denied(data_dir: Path) -> None:
    top_dir = _make_anima(data_dir, "owner")
    child_dir = _make_anima(data_dir, "ritsu")
    _write_config(
        {
            "owner": AnimaModelConfig(supervisor=None),
            "ritsu": AnimaModelConfig(supervisor="owner"),
        }
    )

    handler = _handler(top_dir)
    handler.set_session_origin("human")
    parsed = _grant(handler, "ritsu-home", child_dir, target_anima="ritsu")

    assert parsed["status"] == "error"
    assert parsed["error_type"] == "PermissionDenied"


def test_grant_workspace_access_is_exposed_in_tool_schemas() -> None:
    mode_ab_names = {tool["name"] for tool in build_unified_tool_list(trigger="message:human")}
    mode_b_names = {tool["name"] for tool in build_tool_list(trigger="message:human")}

    assert "grant_workspace_access" in mode_ab_names
    assert "grant_workspace_access" in mode_b_names


def test_grant_workspace_access_is_exposed_via_mcp() -> None:
    from core.mcp.server import _EXPOSED_TOOL_NAMES, MCP_TOOLS

    assert "grant_workspace_access" in _EXPOSED_TOOL_NAMES
    assert any(tool.name == "grant_workspace_access" for tool in MCP_TOOLS)
