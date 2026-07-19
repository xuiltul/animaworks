# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.company import (
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

    result = handler._handle_send_message(
        {"to": "worker", "content": "hello", "intent": "report"}
    )

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

    result = handler._handle_send_message(
        {"to": "worker", "content": "hello", "intent": "report"}
    )

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
        result = handler._handle_send_message(
            {"to": "worker", "content": "hello", "intent": "report"}
        )
    finally:
        meeting_context.reset(context_token)
        meeting_mode.reset(mode_token)

    assert "Beta Corporation" in result
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
    assert not (worker_dir / "state" / "task_queue.jsonl").exists()
    assert not (boss_dir / "state" / "task_queue.jsonl").exists()


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
    assert (worker_dir / "state" / "task_queue.jsonl").exists()
    assert (boss_dir / "state" / "task_queue.jsonl").exists()
