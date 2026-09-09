# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.company import (
    BoundaryCheck,
    check_company_boundary,
    get_company,
    get_company_display_name,
    is_cross_company,
    read_company_config,
)
from core.config.models import (
    AnimaModelConfig,
    AnimaWorksConfig,
    invalidate_cache,
    read_anima_company,
    read_anima_company_checked,
    save_config,
)
from core.messenger import Messenger
from core.org_sync import sync_org_structure
from core.tooling.handler import ToolHandler
from core.tooling.handler_base import meeting_context, meeting_mode
from core.tooling.handler_delegation import DelegationMixin


def _make_anima(
    data_dir: Path,
    name: str,
    *,
    company: str | None = None,
    supervisor: str | None = None,
) -> Path:
    anima_dir = data_dir / "animas" / name
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text(f"# Identity: {name}\n", encoding="utf-8")
    status: dict[str, object] = {"enabled": True}
    if company is not None:
        status["company"] = company
    if supervisor is not None:
        status["supervisor"] = supervisor
    (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return anima_dir


def _write_company(data_dir: Path, name: str, display_name: str) -> None:
    company_dir = data_dir / "companies" / name
    company_dir.mkdir(parents=True)
    (company_dir / "company.json").write_text(
        json.dumps({"name": name, "display_name": display_name}),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    invalidate_cache()
    yield  # type: ignore[misc]
    invalidate_cache()


class TestCompanyResolution:
    def test_reads_membership_from_status_only(self, tmp_path: Path) -> None:
        anima_dir = _make_anima(tmp_path, "alice", company="  acme  ")

        assert read_anima_company(anima_dir) == "acme"
        assert get_company("alice", data_dir=tmp_path) == "acme"

    def test_missing_blank_and_non_string_membership_are_unassigned(self, tmp_path: Path) -> None:
        missing = _make_anima(tmp_path, "missing")
        blank = _make_anima(tmp_path, "blank", company=" ")
        non_string = _make_anima(tmp_path, "non_string")
        (non_string / "status.json").write_text(json.dumps({"company": 123}), encoding="utf-8")
        non_object = _make_anima(tmp_path, "non_object")
        (non_object / "status.json").write_text("[]", encoding="utf-8")

        assert read_anima_company(missing) is None
        assert read_anima_company(blank) is None
        assert read_anima_company(non_string) is None
        assert read_anima_company(non_object) is None

    def test_checked_membership_distinguishes_permission_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        anima_dir = _make_anima(tmp_path, "alice", company="alpha")
        status_path = anima_dir / "status.json"
        original_read_text = Path.read_text

        def denied_read(path: Path, *args: object, **kwargs: object) -> str:
            if path == status_path:
                raise PermissionError(13, "Permission denied", str(path))
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied_read)

        assert read_anima_company_checked(anima_dir) == (False, None)
        assert read_anima_company(anima_dir) is None

    def test_reads_company_json_and_display_name(self, tmp_path: Path) -> None:
        _write_company(tmp_path, "acme", "Acme Holdings")

        assert read_company_config("acme", data_dir=tmp_path) == {
            "name": "acme",
            "display_name": "Acme Holdings",
        }
        assert get_company_display_name("acme", data_dir=tmp_path) == "Acme Holdings"
        assert get_company_display_name("missing", data_dir=tmp_path) == "missing"

    def test_unsafe_company_path_is_not_read(self, tmp_path: Path) -> None:
        (tmp_path / "company.json").write_text('{"display_name": "outside"}', encoding="utf-8")

        assert read_company_config("..", data_dir=tmp_path) is None

    def test_company_config_stat_permission_error_is_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Regression: Path.is_file raised the incident's unhandled EACCES."""
        _write_company(tmp_path, "fs", "FS Corporation")
        config_path = tmp_path / "companies" / "fs" / "company.json"
        original_is_file = Path.is_file

        def denied_is_file(path: Path) -> bool:
            if path == config_path:
                raise PermissionError(13, "Permission denied", str(path))
            return original_is_file(path)

        monkeypatch.setattr(Path, "is_file", denied_is_file)

        assert read_company_config("fs", data_dir=tmp_path) is None
        assert get_company_display_name("fs", data_dir=tmp_path) == "fs"

    @pytest.mark.parametrize(
        ("company_a", "company_b", "expected"),
        [
            ("alpha", "alpha", False),
            ("alpha", "beta", True),
            ("alpha", None, False),
            (None, "beta", False),
            (None, None, False),
        ],
    )
    def test_cross_company_semantics(
        self,
        tmp_path: Path,
        company_a: str | None,
        company_b: str | None,
        expected: bool,
    ) -> None:
        _make_anima(tmp_path, "alice", company=company_a)
        _make_anima(tmp_path, "bob", company=company_b)

        assert is_cross_company("alice", "bob", data_dir=tmp_path) is expected

    def test_membership_change_is_visible_without_cache_invalidation(self, tmp_path: Path) -> None:
        alice = _make_anima(tmp_path, "alice", company="alpha")
        _make_anima(tmp_path, "bob", company="beta")
        assert is_cross_company("alice", "bob", data_dir=tmp_path)

        (alice / "status.json").write_text(json.dumps({"company": "beta"}), encoding="utf-8")

        assert not is_cross_company("alice", "bob", data_dir=tmp_path)


class TestBoundaryCheck:
    def test_readable_same_company_is_allowed_locally(self, tmp_path: Path) -> None:
        """(a) Both memberships readable and equal -> allow."""
        _make_anima(tmp_path, "alice", company="alpha")
        _make_anima(tmp_path, "bob", company="alpha")

        result = check_company_boundary("alice", "bob", data_dir=tmp_path)

        assert result == BoundaryCheck(False, "alpha", "alpha", "local")

    def test_readable_cross_company_is_blocked_with_display_name(self, tmp_path: Path) -> None:
        """(b) Both readable but different -> block with display name."""
        _make_anima(tmp_path, "alice", company="alpha")
        _make_anima(tmp_path, "bob", company="beta")
        _write_company(tmp_path, "beta", "Beta Corporation")

        result = check_company_boundary("alice", "bob", data_dir=tmp_path)

        assert result == BoundaryCheck(True, "beta", "Beta Corporation", "local")

    @pytest.mark.parametrize(
        ("server_cross_company", "expected_blocked"),
        [(True, True), (False, False)],
        ids=["cross-company-blocked", "same-company-allowed"],
    )
    def test_unreadable_target_uses_server_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        server_cross_company: bool,
        expected_blocked: bool,
    ) -> None:
        """(c,d) An unreadable target trusts a valid host response."""
        _make_anima(tmp_path, "alice", company="alpha")
        bob_dir = _make_anima(tmp_path, "bob", company="beta")
        status_path = bob_dir / "status.json"
        original_read_text = Path.read_text

        def denied_read(path: Path, *args: object, **kwargs: object) -> str:
            if path == status_path:
                raise PermissionError(13, "Permission denied", str(path))
            return original_read_text(path, *args, **kwargs)

        response = MagicMock(status_code=200)
        response.json.return_value = {
            "from_company": "alpha" if server_cross_company else "beta",
            "to_company": "beta",
            "cross_company": server_cross_company,
            "to_display_name": "Beta Corporation",
        }
        monkeypatch.setattr(Path, "read_text", denied_read)
        monkeypatch.setattr("httpx.get", MagicMock(return_value=response))

        result = check_company_boundary("alice", "bob", data_dir=tmp_path)

        assert result.cross_company is expected_blocked
        assert result.to_company == "beta"
        assert result.display_name == "Beta Corporation"
        assert result.resolved_via == "server"

    @pytest.mark.parametrize(
        ("from_company", "to_company", "cross_company"),
        [("alpha", "beta", False), ("alpha", "alpha", True)],
        ids=["different-but-allowed", "same-but-blocked"],
    )
    def test_inconsistent_server_response_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        from_company: str,
        to_company: str,
        cross_company: bool,
    ) -> None:
        _make_anima(tmp_path, "alice", company="alpha")
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "from_company": from_company,
            "to_company": to_company,
            "cross_company": cross_company,
            "to_display_name": "Target Company",
        }
        monkeypatch.setattr("httpx.get", MagicMock(return_value=response))

        result = check_company_boundary("alice", "missing", data_dir=tmp_path)

        assert result == BoundaryCheck(True, None, "", "fail_closed")

    def test_unreachable_server_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """(e) An unreadable target plus API failure -> fail closed."""
        _make_anima(tmp_path, "alice", company="alpha")
        monkeypatch.setattr("httpx.get", MagicMock(side_effect=OSError("connection refused")))

        result = check_company_boundary("alice", "missing", data_dir=tmp_path)

        assert result == BoundaryCheck(True, None, "", "fail_closed")

    def test_display_name_error_falls_back_to_company_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """(f) Display-name resolution cannot break a boundary rejection."""
        _make_anima(tmp_path, "alice", company="alpha")
        _make_anima(tmp_path, "bob", company="beta")
        monkeypatch.setattr(
            "core.company.get_company_display_name",
            MagicMock(side_effect=PermissionError("denied")),
        )

        result = check_company_boundary("alice", "bob", data_dir=tmp_path)

        assert result == BoundaryCheck(True, "beta", "beta", "local")

    def test_unassigned_animas_keep_legacy_fail_open(self, tmp_path: Path) -> None:
        """(g) Confirmed unassigned memberships remain unrestricted."""
        _make_anima(tmp_path, "alice")
        _make_anima(tmp_path, "bob")

        result = check_company_boundary("alice", "bob", data_dir=tmp_path)

        assert result == BoundaryCheck(False, None, "", "local")


def test_anima_model_config_supports_company() -> None:
    config = AnimaModelConfig(company="alpha")

    assert config.company == "alpha"
    assert AnimaModelConfig().company is None


def test_org_sync_adds_and_updates_company(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    alice = _make_anima(tmp_path, "alice", company="alpha")
    config_path = tmp_path / "config.json"
    save_config(AnimaWorksConfig(setup_complete=True), config_path)

    sync_org_structure(animas_dir, config_path)
    invalidate_cache()
    from core.config.models import load_config

    assert load_config(config_path).animas["alice"].company == "alpha"

    (alice / "status.json").write_text(json.dumps({"company": "beta"}), encoding="utf-8")
    sync_org_structure(animas_dir, config_path)
    invalidate_cache()
    assert load_config(config_path).animas["alice"].company == "beta"


def _save_org_config(data_dir: Path) -> None:
    config = AnimaWorksConfig(setup_complete=True)
    config.animas = {
        "boss": AnimaModelConfig(supervisor=None),
        "worker": AnimaModelConfig(supervisor="boss"),
    }
    save_config(config, data_dir / "config.json")
    invalidate_cache()


def _make_tool_handler(anima_dir: Path, messenger: Messenger) -> ToolHandler:
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    return ToolHandler(anima_dir=anima_dir, memory=memory, messenger=messenger, tool_registry=[])


def test_send_message_rejects_cross_company_with_display_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    boss_dir = _make_anima(tmp_path, "boss", company="alpha")
    _make_anima(tmp_path, "worker", company="beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    _save_org_config(tmp_path)
    messenger = Messenger(tmp_path / "shared", "boss")
    handler = _make_tool_handler(boss_dir, messenger)

    result = handler._handle_send_message({"to": "worker", "content": "hello", "intent": "report"})

    assert "Beta Corporation" in result
    assert "owner" in result.lower() or "オーナー" in result
    assert not list((tmp_path / "shared" / "inbox" / "worker").glob("*.json"))


def test_send_message_allows_same_company(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    boss_dir = _make_anima(tmp_path, "boss", company="alpha")
    _make_anima(tmp_path, "worker", company="alpha")
    _save_org_config(tmp_path)
    messenger = Messenger(tmp_path / "shared", "boss")
    handler = _make_tool_handler(boss_dir, messenger)

    result = handler._handle_send_message({"to": "worker", "content": "hello", "intent": "report"})

    assert "Message sent to worker" in result
    assert len(list((tmp_path / "shared" / "inbox" / "worker").glob("*.json"))) == 1


def test_send_message_meeting_redirect_rejects_cross_company(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    boss_dir = _make_anima(tmp_path, "boss", company="alpha")
    _make_anima(tmp_path, "worker", company="beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    _save_org_config(tmp_path)
    messenger = Messenger(tmp_path / "shared", "boss")
    handler = _make_tool_handler(boss_dir, messenger)
    mode_token = meeting_mode.set(True)
    context_token = meeting_context.set({"participants": ["worker"]})
    try:
        result = handler._handle_send_message({"to": "worker", "content": "hello", "intent": "report"})
    finally:
        meeting_context.reset(context_token)
        meeting_mode.reset(mode_token)

    assert "Beta Corporation" in result
    assert not list((tmp_path / "shared" / "inbox" / "worker").glob("*.json"))


def test_send_message_unverifiable_boundary_fails_closed_with_stable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    boss_dir = _make_anima(tmp_path, "boss", company="alpha")
    worker_dir = _make_anima(tmp_path, "worker", company="beta")
    _save_org_config(tmp_path)
    messenger = Messenger(tmp_path / "shared", "boss")
    handler = _make_tool_handler(boss_dir, messenger)
    worker_status = worker_dir / "status.json"
    original_read_text = Path.read_text

    def denied_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == worker_status:
            raise PermissionError(13, "Permission denied", str(path))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied_read)
    monkeypatch.setattr("httpx.get", MagicMock(side_effect=OSError("connection refused")))

    result = handler._handle_send_message({"to": "worker", "content": "hello", "intent": "report"})

    assert "会社境界" in result or "company boundary" in result.lower()
    assert "Permission denied" not in result
    assert not list((tmp_path / "shared" / "inbox" / "worker").glob("*.json"))


class _DelegationHarness(DelegationMixin):
    def __init__(self, anima_dir: Path, messenger: Messenger) -> None:
        self._anima_dir = anima_dir
        self._anima_name = anima_dir.name
        self._activity = MagicMock()
        self._messenger = messenger
        self._session_origin = "test"
        self._session_origin_chain = []


def _delegate(harness: _DelegationHarness) -> str:
    return harness._handle_delegate_task(
        {
            "name": "worker",
            "instruction": "Prepare the report",
            "summary": "Report",
            "deadline": "2h",
        }
    )


def test_delegate_task_rejects_cross_company_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    boss_dir = _make_anima(tmp_path, "boss", company="alpha")
    worker_dir = _make_anima(tmp_path, "worker", company="beta", supervisor="boss")
    _write_company(tmp_path, "beta", "Beta Corporation")
    _save_org_config(tmp_path)
    harness = _DelegationHarness(boss_dir, Messenger(tmp_path / "shared", "boss"))

    result = _delegate(harness)

    assert "Beta Corporation" in result
    assert "owner" in result.lower() or "オーナー" in result
    from core.memory.task_queue import TaskQueueManager

    assert TaskQueueManager(worker_dir).list_tasks() == []
    assert TaskQueueManager(boss_dir).list_tasks() == []


def test_delegate_task_allows_same_company(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    boss_dir = _make_anima(tmp_path, "boss", company="alpha")
    worker_dir = _make_anima(tmp_path, "worker", company="alpha", supervisor="boss")
    _save_org_config(tmp_path)
    harness = _DelegationHarness(boss_dir, Messenger(tmp_path / "shared", "boss"))

    result = _delegate(harness)

    assert "worker" in result
    from core.memory.task_queue import TaskQueueManager

    children = TaskQueueManager(worker_dir).list_tasks()
    aliases = TaskQueueManager(boss_dir).list_tasks()
    assert len(children) == len(aliases) == 1
    assert children[0].original_instruction == "Prepare the report"
    assert aliases[0].meta["delegated_to"] == "worker"
    assert aliases[0].meta["delegated_task_id"] == children[0].task_id
