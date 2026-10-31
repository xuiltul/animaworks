"""Cross-entry-point tests for the shared file-access policy."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.config.file_access_policy import FileAccessContext, evaluate_file_access, resolve_denied_roots
from core.config.schemas import PermissionsConfig
from core.execution.engines.claude._sdk_security import _check_a1_file_access
from core.tooling.handler import ToolHandler


@dataclass(frozen=True)
class _AccessCase:
    name: str
    target: str
    write: bool
    sdk_allowed: bool
    file_roots: tuple[str, ...] = ()
    readonly_roots: tuple[str, ...] = ()
    denied_roots: tuple[str, ...] = ()
    grant: str = ""
    task_cwd: bool = False
    superuser: bool = False
    handler_allowed: bool | None = None
    intentional_difference: bool = False


_CASES = (
    _AccessCase("empty_path_noop", "@empty", False, True),
    _AccessCase("superuser_bypass", "outside:secret.md", False, True, superuser=True),
    _AccessCase("own_read", "own:IDENTITY.md", False, True),
    _AccessCase("own_write", "own:knowledge/note.md", True, True),
    _AccessCase("global_permissions_under_data_dir", "data:nested/permissions.global.json", True, False, ("/",)),
    _AccessCase("explicit_deny", "private:secret.md", False, False, ("/",), denied_roots=("private",)),
    _AccessCase(
        "explicit_deny_precedes_shared_grant",
        "shared:private.md",
        False,
        False,
        denied_roots=("shared",),
    ),
    _AccessCase("internal_cache_read", "own:.codex_home/auth.json", False, False, ("/",), (), ("private",)),
    _AccessCase("internal_cache_write", "own:.codex_home/auth.json", True, False, ("/",), (), ("private",)),
    _AccessCase("other_anima_private", "other:episodes/secret.md", False, False, ("/",)),
    _AccessCase(
        "subordinate_activity_read", "other:activity_log/today.jsonl", False, True, grant="subordinate_activity"
    ),
    _AccessCase("descendant_activity_read", "other:activity_log/today.jsonl", False, True, grant="descendant_activity"),
    _AccessCase("peer_activity_read", "other:activity_log/today.jsonl", False, True, grant="peer_activity"),
    _AccessCase("subordinate_management_read", "other:status.json", False, True, grant="management"),
    _AccessCase("subordinate_management_write", "other:status.json", True, False, grant="management"),
    _AccessCase("descendant_state_file_read", "other:IDENTITY.md", False, True, grant="descendant_file"),
    _AccessCase("descendant_plans_read", "other:state/plans/plan.md", False, True, grant="descendant_dir"),
    _AccessCase("subordinate_root_listing", "other:", False, True, grant="subordinate_root"),
    _AccessCase("framework_shared_read", "shared:public.md", False, True),
    _AccessCase("external_skill_root_read", "external:SKILL.md", False, True),
    _AccessCase("configured_read_root", "allowed:readme.md", False, True, ("allowed",)),
    _AccessCase("configured_readonly_root", "readonly:reference.md", False, True, ("allowed",), ("readonly",)),
    _AccessCase(
        "sdk_read_outside_nonempty_file_roots",
        "outside:source.py",
        False,
        True,
        ("allowed",),
        handler_allowed=False,
        intentional_difference=True,
    ),
    _AccessCase("configured_write_root", "allowed:src/main.py", True, True, ("allowed",)),
    _AccessCase("write_readonly_root", "readonly:reference.md", True, False, ("allowed",), ("readonly",)),
    _AccessCase("write_outside_configured_roots", "outside:main.py", True, False, ("allowed",)),
    _AccessCase("own_protected_state_index", "own:state/bm25_longterm_index.json", True, False),
    _AccessCase("activity_log_directory_write", "own:activity_log/today.jsonl", True, False),
    _AccessCase("activity_log_substring_is_not_a_directory", "own:knowledge/activity_log_notes.md", True, True),
    _AccessCase(
        "sdk_task_cwd_write",
        "task:main.py",
        True,
        True,
        ("allowed",),
        task_cwd=True,
        handler_allowed=False,
        intentional_difference=True,
    ),
    _AccessCase(
        "sdk_tempdir_write",
        "system_tmp:scratch.txt",
        True,
        True,
        ("allowed",),
        handler_allowed=False,
        intentional_difference=True,
    ),
    _AccessCase(
        "sdk_common_knowledge_write",
        "common_knowledge:note.md",
        True,
        True,
        ("allowed",),
        handler_allowed=False,
        intentional_difference=True,
    ),
    _AccessCase(
        "sdk_common_skills_write",
        "common_skills:note.md",
        True,
        True,
        ("allowed",),
        handler_allowed=False,
        intentional_difference=True,
    ),
)


def _resolve_spec(spec: str, roots: dict[str, Path]) -> Path:
    root_name, _, suffix = spec.partition(":")
    return roots[root_name] / suffix if suffix else roots[root_name]


def _resolve_root_specs(specs: tuple[str, ...], roots: dict[str, Path]) -> list[str]:
    return [spec if Path(spec).is_absolute() else str(roots[spec]) for spec in specs]


def _make_handler(
    anima_dir: Path,
    *,
    grant: str,
    other_anima_dir: Path,
    external_roots: list[SimpleNamespace],
    task_cwd: Path | None,
    superuser: bool,
) -> ToolHandler:
    handler = ToolHandler.__new__(ToolHandler)
    handler._anima_dir = anima_dir
    handler._anima_name = anima_dir.name
    handler._superuser = superuser
    from core.tooling.tool_context import ToolContext

    handler._tool_context = ToolContext(
        anima_dir=anima_dir,
        anima_name=anima_dir.name,
        check_command_permission=lambda _command: None,
        task_cwd=task_cwd,
    )
    handler._subordinate_activity_dirs = []
    handler._descendant_activity_dirs = []
    handler._peer_activity_dirs = []
    handler._subordinate_management_files = []
    handler._subordinate_root_dirs = []
    handler._descendant_state_files = []
    handler._descendant_state_dirs = []

    activity_dir = other_anima_dir / "activity_log"
    if grant == "subordinate_activity":
        handler._subordinate_activity_dirs = [activity_dir]
    elif grant == "descendant_activity":
        handler._descendant_activity_dirs = [activity_dir]
    elif grant == "peer_activity":
        handler._peer_activity_dirs = [activity_dir]
    elif grant == "management":
        handler._subordinate_management_files = [other_anima_dir / "status.json"]
    elif grant == "descendant_file":
        handler._descendant_state_files = [other_anima_dir / "IDENTITY.md"]
    elif grant == "descendant_dir":
        handler._descendant_state_dirs = [other_anima_dir / "state" / "plans"]
    elif grant == "subordinate_root":
        handler._subordinate_root_dirs = [other_anima_dir]

    config = SimpleNamespace(skills=SimpleNamespace(external_roots=external_roots))
    handler._external_roots_config = config
    return handler


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_sdk_and_handler_use_the_shared_file_access_policy(
    case: _AccessCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "agent"
    other_anima_dir = data_dir / "animas" / "worker"
    roots = {
        "data": data_dir,
        "own": anima_dir,
        "other": other_anima_dir,
        "shared": data_dir / "shared",
        "common_knowledge": data_dir / "common_knowledge",
        "common_skills": data_dir / "common_skills",
        "external": tmp_path / "external-skills",
        "allowed": tmp_path / "allowed",
        "readonly": tmp_path / "readonly",
        "private": tmp_path / "private",
        "outside": tmp_path / "outside",
        "task": tmp_path / "task-workspace",
        "system_tmp": tmp_path / "system-tmp",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(roots["system_tmp"]))

    target = "" if case.target == "@empty" else _resolve_spec(case.target, roots)
    file_roots = _resolve_root_specs(case.file_roots, roots)
    readonly_roots = _resolve_root_specs(case.readonly_roots, roots)
    denied_roots = _resolve_root_specs(case.denied_roots, roots)
    permissions_data = {
        "version": 1,
        "file_roots": file_roots,
        "file_roots_readonly": readonly_roots,
        "file_roots_denied": denied_roots,
    }
    (anima_dir / "permissions.json").write_text(json.dumps(permissions_data), encoding="utf-8")

    external_roots = [SimpleNamespace(path=roots["external"], enabled=True)]
    task_cwd = roots["task"] if case.task_cwd else None
    handler = _make_handler(
        anima_dir,
        grant=case.grant,
        other_anima_dir=other_anima_dir,
        external_roots=external_roots,
        task_cwd=task_cwd,
        superuser=case.superuser,
    )
    config = PermissionsConfig.model_validate(permissions_data)
    resolved_denied_roots = resolve_denied_roots(config.file_roots_denied)

    with patch("core.config.models.load_config", return_value=handler._external_roots_config):
        sdk_error = _check_a1_file_access(
            str(target),
            anima_dir,
            write=case.write,
            subordinate_activity_dirs=(
                [other_anima_dir / "activity_log"]
                if case.grant in {"subordinate_activity", "descendant_activity", "subordinate_root"}
                else None
            ),
            subordinate_management_files=[other_anima_dir / "status.json"] if case.grant == "management" else None,
            descendant_read_files=[other_anima_dir / "IDENTITY.md"] if case.grant == "descendant_file" else None,
            descendant_read_dirs=[other_anima_dir / "state" / "plans"] if case.grant == "descendant_dir" else None,
            peer_activity_dirs=[other_anima_dir / "activity_log"] if case.grant == "peer_activity" else None,
            task_cwd=task_cwd,
            superuser=case.superuser,
        )
        handler_error = handler._check_file_permission(
            str(target),
            write=case.write,
            config=config,
            denied_roots=resolved_denied_roots,
        )

    sdk_allowed = sdk_error is None
    handler_allowed = handler_error is None
    expected_handler_allowed = case.sdk_allowed if case.handler_allowed is None else case.handler_allowed
    assert sdk_allowed is case.sdk_allowed
    assert handler_allowed is expected_handler_allowed
    assert (sdk_allowed != handler_allowed) is case.intentional_difference


@pytest.mark.parametrize("link_location", ["inside_cache", "outside_cache"])
def test_internal_cache_denial_checks_both_sides_of_symlinks(tmp_path: Path, link_location: str) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "agent"
    cache_dir = anima_dir / ".codex_home"
    external_dir = tmp_path / "external"
    cache_dir.mkdir(parents=True)
    external_dir.mkdir()

    cache_file = cache_dir / "auth.json"
    external_file = external_dir / "public.txt"
    cache_file.write_text("secret", encoding="utf-8")
    external_file.write_text("public", encoding="utf-8")
    if link_location == "inside_cache":
        target = cache_dir / "external-alias"
        target.symlink_to(external_file)
    else:
        target = external_dir / "cache-alias"
        target.symlink_to(cache_file)

    context = FileAccessContext(
        anima_dir=anima_dir,
        data_dir=data_dir,
        denied_roots=((tmp_path / "blocked").resolve(),),
        restrict_reads_to_roots=False,
    )

    decision = evaluate_file_access(target, context, write=False)

    assert not decision.allowed
    assert decision.reason == "internal_cache"


def test_protected_directory_marker_matches_path_components_only(tmp_path: Path) -> None:
    from core.config.file_access_policy import evaluate_protected_write

    anima_dir = tmp_path / "animas" / "alice"
    assert evaluate_protected_write(anima_dir, anima_dir / "activity_log" / "2026-09-30.jsonl") is not None
    assert evaluate_protected_write(anima_dir, anima_dir / "knowledge" / "activity_log_notes.md") is None


@pytest.mark.parametrize(
    "name",
    [
        "IDENTITY.md",
        "identity.MD",
        "SOUL.md",
        "soul.md",
        "AGENTS.md",
        "agents.Md",
        "BOOTSTRAP.md",
        "bootstrap.md",
        "injection.md",
    ],
)
def test_protected_write_names_are_case_insensitive_and_include_legacy(tmp_path: Path, name: str) -> None:
    from core.config.file_access_policy import evaluate_protected_write

    anima_dir = tmp_path / "animas" / "alice"
    assert evaluate_protected_write(anima_dir, anima_dir / name) is not None
