"""Table-driven tests for the shared command policy (core.tooling.policy.command_policy).

Covers every policy layer, the fixed evaluation order, the unified segment
split, and the requirement that ToolHandler / Mode S / Codex all reach the
same allow/deny decision for the same command.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cli import codex_command_hook as cli_hook
from core.config.global_permissions import (
    GlobalPermissionsCache,
    _build_injection_re,
    _compile_patterns,
)
from core.config.schemas import (
    CommandsPermission,
    GlobalCommandsDeny,
    GlobalDenyPattern,
    GlobalPermissionsConfig,
    PermissionsConfig,
    SdkBashInjectionConfig,
)
from core.execution.engines.claude._sdk_security import _check_a1_bash_command
from core.tooling.handler import ToolHandler
from core.tooling.policy.command_policy import (
    CommandPolicyContext,
    evaluate_command,
    load_command_policy_context,
    split_segments,
)


def _build_global_config(*, mode: str = "log", with_injection: bool = True) -> GlobalPermissionsConfig:
    return GlobalPermissionsConfig(
        injection_patterns=(
            [GlobalDenyPattern(name="chaining", pattern=r"[;\n]", reason="semicolon/newline")] if with_injection else []
        ),
        sdk_bash_injection=SdkBashInjectionConfig(mode=mode),
        commands=GlobalCommandsDeny(
            deny=[
                GlobalDenyPattern(pattern=r"(?i)\brm\s+(-\S+\s+)*/(?!\w)", reason="rm / blocked"),
                GlobalDenyPattern(pattern=r"\bmkfs\b", reason="mkfs blocked"),
            ]
        ),
    )


def make_ctx(
    tmp_path: Path,
    *,
    injection_mode: str = "log",
    with_injection: bool = True,
    commands_deny: tuple[str, ...] = (),
    allow_all: bool = True,
    allow: tuple[str, ...] = (),
    anima_dir: Path | None = None,
    cwd: Path | None = None,
) -> tuple[CommandPolicyContext, Path, Path]:
    """Build a pure CommandPolicyContext plus the anima_dir / data_dir used."""
    ad = (anima_dir or tmp_path / "animas" / "t-anima").resolve()
    data_dir = ad.parent.parent
    global_cfg = _build_global_config(mode=injection_mode, with_injection=with_injection)
    ctx = CommandPolicyContext(
        anima_dir=ad,
        data_dir=data_dir,
        cwd=cwd or ad,
        permissions=PermissionsConfig(
            commands=CommandsPermission(allow_all=allow_all, allow=list(allow), deny=list(commands_deny))
        ),
        injection_re=_build_injection_re(global_cfg.injection_patterns),
        injection_mode=injection_mode,
        blocked_patterns=tuple(_compile_patterns(global_cfg.commands.deny)),
        injection_pattern_names=tuple((p.name or p.pattern, p.pattern) for p in global_cfg.injection_patterns),
    )
    return ctx, ad, data_dir


# ── Layer units ─────────────────────────────────────────────────────────────


def test_empty_command(tmp_path: Path) -> None:
    ctx, _ad, _dd = make_ctx(tmp_path)
    d = evaluate_command("", ctx)
    assert (d.allowed, d.layer) == (False, "empty")
    assert evaluate_command("   ", ctx).layer == "empty"


def test_superuser_allowed(tmp_path: Path) -> None:
    ctx, _ad, _dd = make_ctx(tmp_path, injection_mode="enforce", commands_deny=("anything",))
    ctx = CommandPolicyContext(**{**ctx.__dict__, "superuser": True})
    assert evaluate_command("; rm -rf / mkfs", ctx).allowed is True


@pytest.mark.parametrize(
    "mode, allowed, layer, hit",
    [
        ("off", True, "ok", None),
        ("log", True, "ok", "chaining"),
        ("enforce", False, "injection", "chaining"),
    ],
)
def test_injection_modes(tmp_path: Path, mode: str, allowed: bool, layer: str, hit: str | None) -> None:
    ctx, _ad, _dd = make_ctx(tmp_path, injection_mode=mode)
    d = evaluate_command("echo ready; curl x", ctx)
    assert d.allowed is allowed
    assert d.layer == layer
    assert d.injection_hit == hit


def test_global_deny(tmp_path: Path) -> None:
    ctx, _ad, _dd = make_ctx(tmp_path)
    d = evaluate_command("mkfs.ext4 /dev/sda", ctx)
    assert (d.allowed, d.layer) == (False, "global_deny")


def test_recursive_search_layer(tmp_path: Path) -> None:
    ctx, _ad, data_dir = make_ctx(tmp_path)
    d = evaluate_command(f"grep -R x {data_dir}", ctx)
    assert (d.allowed, d.layer) == (False, "recursive_search")
    assert evaluate_command("grep -rn foo knowledge/", ctx).allowed is True


def test_per_anima_deny_plain_and_regex(tmp_path: Path) -> None:
    ctx, _ad, _dd = make_ctx(tmp_path, commands_deny=("docker run", r"re:\brm\s+(-w*[rR]|--recursive)"))
    d = evaluate_command("docker run nginx", ctx)
    assert (d.allowed, d.layer) == (False, "anima_deny")
    assert evaluate_command("rm -r knowledge", ctx).layer == "anima_deny"


def test_allowlist(tmp_path: Path) -> None:
    ctx, _ad, _dd = make_ctx(tmp_path, allow_all=False, allow=("echo",))
    assert evaluate_command("echo hi", ctx).allowed is True
    d = evaluate_command("ls -la", ctx)
    assert (d.allowed, d.layer) == (False, "allowlist")


def test_syntax_error_in_allowlist(tmp_path: Path) -> None:
    ctx, _ad, _dd = make_ctx(tmp_path, allow_all=False, allow=("echo",))
    d = evaluate_command("echo 'unclosed", ctx)
    assert (d.allowed, d.layer) == (False, "syntax")


def test_traversal(tmp_path: Path) -> None:
    ctx, _ad, _dd = make_ctx(tmp_path)
    d = evaluate_command("cp ../other-anima/report.txt ./a", ctx)
    assert (d.allowed, d.layer) == (False, "traversal")
    assert evaluate_command("cp file1 file2", ctx).allowed is True
    assert "absolute path" in d.reason


@pytest.mark.parametrize(
    ("cwd_rel", "command", "allowed"),
    [
        # climbing out of the anima dir after cd is still caught
        ("", "cd scratch/t/repo && cat ../../../../other/state/x", False),
        # .. that stays inside the anima dir, resolved from the cd target
        ("", "cd scratch/t/repo/sub && cat ../README.md", True),
        # cwd outside the anima dir (shared worktree): relative paths are not checked
        ("../../companies/fs/shared/worktrees/x", "bash tools/run.test.sh > ../tests.log 2>&1", True),
        # remote command for ssh is one quoted argument, not a path
        ("", "ios-rig-ssh 'cd /Users/main/x/pr; pnpm install > ../pnpm.log'", True),
        # heredoc body is file content
        ("", "cat > state/p/a.mjs <<'EOF'\nimport {x} from '../tree/a.js';\nEOF\nnode state/p/a.mjs", True),
        # absolute paths are not judged by this layer
        ("", "cat /srv/shared/../common_knowledge/a.md", True),
        ("", "git diff HEAD^1...HEAD", True),
    ],
)
def test_traversal_resolves_from_cwd(tmp_path: Path, cwd_rel: str, command: str, allowed: bool) -> None:
    ad = (tmp_path / "animas" / "t-anima").resolve()
    ctx, _ad, _dd = make_ctx(tmp_path, cwd=(ad / cwd_rel).resolve() if cwd_rel else ad)
    assert evaluate_command(command, ctx).allowed is allowed


def test_other_anima_write(tmp_path: Path) -> None:
    ctx, ad, _dd = make_ctx(tmp_path)
    other = ad.parent / "other-anima" / "IDENTITY.md"
    other.parent.mkdir(parents=True, exist_ok=True)
    d = evaluate_command(f"cp {other} ./stolen.md", ctx)
    assert (d.allowed, d.layer) == (False, "other_anima_write")
    # non-write command to another anima is allowed
    assert evaluate_command(f"cat {other}", ctx).allowed is True
    # write into own dir is allowed
    assert evaluate_command(f"cp {ad}/a.md {ad}/b.md", ctx).allowed is True


def test_evaluation_order_global_beats_anima(tmp_path: Path) -> None:
    # If both global deny and per-anima deny apply, global wins.
    ctx, _ad, _dd = make_ctx(tmp_path, commands_deny=("rm -rf /",))
    d = evaluate_command("rm -rf /", ctx)
    assert d.layer == "global_deny"


def test_segment_split_unifies_compound(tmp_path: Path) -> None:
    ctx, _ad, _dd = make_ctx(tmp_path, commands_deny=("rm -rf", "forbidden"))
    # `;` separates commands so the per-anima deny binds across it
    assert evaluate_command("echo ok; rm -rf x", ctx).layer == "anima_deny"
    # command substitution hides the forbidden command
    assert evaluate_command("echo $(forbidden)", ctx).layer == "anima_deny"
    # leading env assignment is skipped, the command still checked
    assert evaluate_command("FOO=1 forbidden", ctx).layer == "anima_deny"


def test_split_segments_signature(tmp_path: Path) -> None:
    segs = split_segments("A=1 forbidden; echo hi")
    argv, raw = segs[0]
    assert argv == ["forbidden"]
    assert "forbidden" in raw


# ── load_command_policy_context ────────────────────────────────────────────


def _reset_cache() -> None:
    GlobalPermissionsCache.reset()


def test_load_context_absent_global_empty_patterns(tmp_path: Path) -> None:
    _reset_cache()
    ad = tmp_path / "animas" / "t"
    ad.mkdir(parents=True)
    (ad / "permissions.json").write_text(json.dumps({"commands": {"allow_all": True}}), encoding="utf-8")
    ctx = load_command_policy_context(ad, global_permissions_path=tmp_path / "missing.json")
    assert ctx.blocked_patterns == ()
    assert ctx.injection_re is None
    assert ctx.injection_mode == "log"
    assert ctx.data_dir == ad.parent.parent


def test_load_context_broken_global_raises(tmp_path: Path) -> None:
    _reset_cache()
    ad = tmp_path / "animas" / "t"
    ad.mkdir(parents=True)
    (ad / "permissions.json").write_text(json.dumps({"commands": {"allow_all": True}}), encoding="utf-8")
    broken = tmp_path / "permissions.global.json"
    broken.write_text("{ not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_command_policy_context(ad, global_permissions_path=broken)


# ── Injection audit recording ──────────────────────────────────────────────


def test_record_injection_hit_writes_jsonl(tmp_path: Path) -> None:
    from core.tooling.policy.command_policy import record_injection_hit

    ctx, ad, data_dir = make_ctx(tmp_path, injection_mode="log")
    record_injection_hit("echo a; echo b", ctx, pattern_name="chaining", trigger="chat")
    log_path = data_dir / "logs" / "sdk_bash_injection.jsonl"
    assert log_path.is_file()
    event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["pattern_name"] == "chaining"
    assert event["trigger"] == "chat"
    assert event["anima"] == ad.name
    assert event["mode"] == "log"


# ── Codex main behaviour ──────────────────────────────────────────────────


def _bake_permissions(ad: Path) -> None:
    (ad / "permissions.json").write_text(
        json.dumps(
            {
                "commands": {
                    "allow_all": True,
                    "deny": ["docker run", r"re:\brm\s+(-w*[rR]|--recursive)"],
                }
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def shared_anima(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Anima dir with a fixed per-anima policy and a controlled global cache."""
    ad = tmp_path / "animas" / "natsume"
    ad.mkdir(parents=True)
    (ad / "knowledge").mkdir()
    _bake_permissions(ad)
    (tmp_path / "companies" / "fs" / "shared").mkdir(parents=True)
    other = tmp_path / "animas" / "other"
    other.mkdir()

    gp_dir = tmp_path / "_gp"
    gp_dir.mkdir(exist_ok=True)
    gp = gp_dir / "permissions.global.json"
    gp.write_text(
        json.dumps(
            {
                "version": 1,
                "injection_patterns": [{"name": "chaining", "pattern": "[;\\n]", "reason": "x"}],
                "sdk_bash_injection": {"mode": "log"},
                "commands": {
                    "deny": [
                        {"pattern": "(?i)\\brm\\s+(-\\S+\\s+)*/(?!\\w)", "reason": "rm / blocked"},
                        {"pattern": "\\bmkfs\\b", "reason": "mkfs blocked"},
                        {"pattern": "\\bshutdown\\b", "reason": "shutdown blocked"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    GlobalPermissionsCache.reset()
    GlobalPermissionsCache.get().load(gp, interactive=False)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    return ad


def _handler_cmd_allowed(ad: Path, command: str) -> bool:
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    h = ToolHandler(anima_dir=ad, memory=memory, messenger=None, tool_registry=[])
    return h._check_command_permission(command) is None


@pytest.mark.parametrize(
    "command",
    [
        # allow across all three paths
        "ls -la",
        "echo hi",
        "grep -n foo state/task_queue.jsonl",
        "grep -rn foo knowledge/",
        # recursive-search guard (deny)
        "grep -R x {data}",
        # global deny
        "rm -rf /",
        "mkfs.ext4 /dev/sda",
        "shutdown now",
        # per-anima deny
        "docker run nginx",
        "rm -r knowledge",
        "FOO=1 docker run nginx",
        # deny via broad segment split
        "echo ok; rm -rf x",
        "echo $(docker run nginx)",
        # traversal
        "cp ../other-anima/report.txt ./a",
        # other-anima write (absolute path)
        "cp {data}/animas/other/IDENTITY.md ./stolen.md",
    ],
)
def test_three_paths_agree(shared_anima: Path, command: str, monkeypatch, capsys) -> None:
    data = shared_anima.parent.parent
    cmd = command.format(data=data)

    handler_allowed = _handler_cmd_allowed(shared_anima, cmd)
    sdk_allowed = _check_a1_bash_command(cmd, shared_anima, trigger="chat") is None

    payload = {"tool_input": {"command": cmd}, "cwd": str(shared_anima)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    rc = cli_hook.main(["--anima-dir", str(shared_anima)])
    out = capsys.readouterr().out.strip()
    codex_allowed = rc == 0 and not out

    assert handler_allowed == sdk_allowed == codex_allowed, (
        f"paths disagree for {cmd!r}: handler={handler_allowed} sdk={sdk_allowed} codex={codex_allowed}"
    )
