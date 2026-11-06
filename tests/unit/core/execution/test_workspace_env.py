# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scratch directories and child-process environment overrides."""

from __future__ import annotations

import re
from pathlib import Path

from core.execution.session.session_context import RuntimeSessionContext, runtime_session_scope
from core.execution.workspace_env import build_workspace_env, resolve_scratch_dir, update_workspace_env
from core.tasks.board.tasks import attempt_scope


def test_resolve_scratch_dir_uses_sanitized_task_id(tmp_path: Path) -> None:
    anima_dir = tmp_path / "data" / "animas" / "alice"
    identity = {"anima": "alice", "task_id": "../../pr/42?", "token": "attempt-token"}

    with attempt_scope(identity):
        scratch_dir = resolve_scratch_dir(anima_dir)

    assert scratch_dir.parent == anima_dir.resolve() / "scratch"
    assert scratch_dir.name == "task-.._.._pr_42_"
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", scratch_dir.name.removeprefix("task-"))
    assert scratch_dir.is_dir()


def test_resolve_scratch_dir_limits_task_id_length(tmp_path: Path) -> None:
    anima_dir = tmp_path / "data" / "animas" / "alice"
    identity = {"anima": "alice", "task_id": "x" * 100, "token": "attempt-token"}

    with attempt_scope(identity):
        scratch_dir = resolve_scratch_dir(anima_dir)

    assert len(scratch_dir.name.removeprefix("task-")) == 64


def test_resolve_scratch_dir_uses_session_type_or_misc(tmp_path: Path) -> None:
    anima_dir = tmp_path / "data" / "animas" / "alice"
    context = RuntimeSessionContext.create(
        session_type="cron/job",
        thread_id="daily",
        trigger="cron",
    )

    with runtime_session_scope(context):
        assert resolve_scratch_dir(anima_dir).name == "cron_job"

    assert resolve_scratch_dir(anima_dir).name == "misc"


def test_build_workspace_env_sets_scratch_and_shared_package_caches(tmp_path: Path) -> None:
    anima_dir = tmp_path / "data" / "animas" / "alice"
    cache_root = tmp_path / "data" / "cache"

    env = build_workspace_env(anima_dir)

    assert env["ANIMAWORKS_SCRATCH_DIR"] == str(anima_dir.resolve() / "scratch" / "misc")
    expected_dirs = {
        "UV_CACHE_DIR": "uv",
        "PIP_CACHE_DIR": "pip",
        "npm_config_cache": "npm",
        "npm_config_store_dir": "pnpm-store",
        "GRADLE_USER_HOME": "gradle",
        "NUGET_PACKAGES": "nuget-packages",
        "COMPOSER_CACHE_DIR": "composer",
    }
    assert {name: env[name] for name in expected_dirs} == {
        name: str(cache_root / directory) for name, directory in expected_dirs.items()
    }


def test_build_workspace_env_omits_cache_variables_without_cache_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("core.config.file_access_policy.shared_tool_cache_write_root", lambda _anima_dir: None)

    env = build_workspace_env(tmp_path / "data" / "animas" / "alice")

    assert set(env) == {"ANIMAWORKS_SCRATCH_DIR", "ANIMAWORKS_BOARD_REPLY_CHANNELS"}


def test_update_workspace_env_drops_inherited_caches_without_cache_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("core.config.file_access_policy.shared_tool_cache_write_root", lambda _anima_dir: None)
    env = {"UV_CACHE_DIR": "/host/cache", "PIP_CACHE_DIR": "/host/pip"}

    update_workspace_env(env, tmp_path / "data" / "animas" / "alice")

    assert set(env) == {"ANIMAWORKS_SCRATCH_DIR", "ANIMAWORKS_BOARD_REPLY_CHANNELS"}


def test_command_env_overrides_host_package_cache(tmp_path: Path, monkeypatch) -> None:
    from core.tooling.handler_exec import _build_command_env

    anima_dir = tmp_path / "data" / "animas" / "alice"
    monkeypatch.setenv("UV_CACHE_DIR", "/host/cache")

    env = _build_command_env(anima_dir)

    assert env["UV_CACHE_DIR"] == str(tmp_path / "data" / "cache" / "uv")
    assert Path(env["ANIMAWORKS_SCRATCH_DIR"]).is_dir()
