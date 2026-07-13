from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E coverage for per-Anima filesystem deny roots in Codex Mode C."""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from core.execution.codex_sdk import CodexSDKExecutor
from core.schemas import ModelConfig


def _codex_supports_permission_profiles(codex: str) -> bool:
    result = subprocess.run(
        [codex, "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
    return bool(match and tuple(int(part) for part in match.groups()) >= (0, 138, 0))


@pytest.mark.e2e
def test_generated_profile_denies_direct_and_symlink_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """A generated Mode C profile is enforced by the real Codex Linux sandbox."""
    if sys.platform != "linux":
        pytest.skip("Codex permission-profile smoke test currently targets the Linux deployment")
    codex = shutil.which("codex")
    if codex is None or not _codex_supports_permission_profiles(codex):
        pytest.skip("Codex 0.138.0 or newer is required for permission profiles")

    runtime_root = Path.cwd() / f".e2e-read-deny-{tmp_path.name}"
    request.addfinalizer(lambda: shutil.rmtree(runtime_root, ignore_errors=True))
    anima_dir = runtime_root / "animas" / "kotoha"
    allowed_dir = tmp_path / "allowed"
    denied_dir = tmp_path / "denied"
    anima_dir.mkdir(parents=True)
    (anima_dir / "state").mkdir()
    allowed_dir.mkdir()
    denied_dir.mkdir()
    (allowed_dir / "visible.txt").write_text("ALLOWED_DATA\n", encoding="utf-8")
    (denied_dir / "secret.txt").write_text("DENIED_SECRET\n", encoding="utf-8")
    (allowed_dir / "secret-link").symlink_to(denied_dir / "secret.txt")
    (anima_dir / "permissions.json").write_text(
        json.dumps(
            {
                "version": 1,
                "file_roots": [str(allowed_dir)],
                "file_roots_denied": [str(denied_dir), str(anima_dir / ".codex_home")],
            }
        ),
        encoding="utf-8",
    )
    current_state = anima_dir / "state" / "current_state.md"
    current_state.write_text("status: idle\n", encoding="utf-8")

    model_config = ModelConfig(
        model="codex/o4-mini",
        max_tokens=4096,
        max_turns=30,
        credential="openai",
        api_key="test-key",
        context_threshold=0.5,
        max_chains=2,
    )
    executor = CodexSDKExecutor(model_config=model_config, anima_dir=anima_dir)
    monkeypatch.setattr(executor, "_propagate_auth", lambda: None)
    executor._write_codex_config("test instructions")
    codex_secret = anima_dir / ".codex_home" / "auth-canary.json"
    codex_secret.write_text("AUTH_SECRET\n", encoding="utf-8")

    script = """
set -eu
test "$(cat "$1/visible.txt")" = "ALLOWED_DATA"
printf 'WRITE_OK\n' > "$1/writable.txt"
! cat "$2/secret.txt" > "$1/direct-leak.txt" 2>/dev/null
test ! -s "$1/direct-leak.txt"
! cat "$1/secret-link" > "$1/symlink-leak.txt" 2>/dev/null
test ! -s "$1/symlink-leak.txt"
! "$3" -c 'from pathlib import Path; import sys; Path(sys.argv[1]).read_text()' "$2/secret.txt" 2>/dev/null
! printf '{"version": 1, "file_roots": ["/"]}\n' > "$4" 2>/dev/null
! rm "$4" 2>/dev/null
! cat "$5" > "$1/codex-home-leak.txt" 2>/dev/null
test ! -s "$1/codex-home-leak.txt"
! ln -sfn "$2/secret.txt" "$6" 2>/dev/null
"""
    env = os.environ.copy()
    env["CODEX_HOME"] = str(anima_dir / ".codex_home")
    result = subprocess.run(
        [
            codex,
            "sandbox",
            "-P",
            "animaworks",
            "--",
            "/bin/bash",
            "-c",
            script,
            "read-deny-smoke",
            str(allowed_dir),
            str(denied_dir),
            sys.executable,
            str(anima_dir / "permissions.json"),
            str(codex_secret),
            str(current_state),
        ],
        cwd=anima_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (allowed_dir / "writable.txt").read_text(encoding="utf-8") == "WRITE_OK\n"
    persisted_permissions = json.loads((anima_dir / "permissions.json").read_text(encoding="utf-8"))
    assert persisted_permissions["file_roots_denied"] == [str(denied_dir), str(anima_dir / ".codex_home")]
    assert not current_state.is_symlink()
    assert current_state.read_text(encoding="utf-8") == "status: idle\n"

    # The exact generated MCP command must itself be runnable under the same
    # profile.  stdio EOF makes the server exit after startup, without an API
    # request, while still exercising the real outer sandbox wrapper.
    parsed_config = tomllib.loads((anima_dir / ".codex_home" / "config.toml").read_text(encoding="utf-8"))
    mcp_config = parsed_config["mcp_servers"]["aw"]
    mcp_env = os.environ.copy()
    mcp_env.update(mcp_config["env"])
    assert mcp_env["CODEX_HOME"] == str(anima_dir / ".codex_home")
    mcp_result = subprocess.run(
        [mcp_config["command"], *mcp_config["args"]],
        cwd=anima_dir,
        env=mcp_env,
        input="",
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert mcp_result.returncode == 0, mcp_result.stderr
    assert "AnimaWorks MCP server starting" in mcp_result.stderr
