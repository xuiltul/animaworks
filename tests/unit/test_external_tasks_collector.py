# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for external tasks store and collector skeleton."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.config.schemas import ExternalTasksConfig, ExternalTasksSourcesConfig
from core.external_tasks.collector import (
    SOURCE_REGISTRY,
    CredentialNotFoundError,
    collect_all,
)
from core.external_tasks.models import ExternalTask, Snapshot, SourceHealth
from core.external_tasks.store import ExternalTaskStore


def _task(
    *,
    id: str = "github-1",
    title: str = "Sample task",
    status: str = "open",
    source_type: str = "github",
    source_icon: str = "github",
    source_url: str | None = "https://example.com/task/1",
    created_at: str | None = None,
    last_updated_at: str | None = None,
    priority: int = 50,
) -> ExternalTask:
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    return ExternalTask(
        id=id,
        title=title,
        status=status,
        source_type=source_type,
        source_icon=source_icon,
        source_url=source_url,
        created_at=created_at or now.isoformat(),
        last_updated_at=last_updated_at or now.isoformat(),
        priority=priority,
    )


def _enabled_config(**source_flags: bool) -> ExternalTasksConfig:
    defaults = {"github": True, "slack": True, "chatwork": True, "gmail": True}
    defaults.update(source_flags)
    return ExternalTasksConfig(
        enabled=True,
        sources=ExternalTasksSourcesConfig(**defaults),
    )


# ── store ──────────────────────────────────────────


def test_store_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "external_tasks.json"
    store = ExternalTaskStore(path)
    snapshot = Snapshot(
        version=1,
        last_collected_at="2026-07-20T12:00:00+00:00",
        sources={
            "github": SourceHealth(
                status="ok",
                collected_at="2026-07-20T12:00:00+00:00",
                error=None,
            )
        },
        tasks=[_task()],
    )
    store.save(snapshot)
    loaded = store.load()
    assert loaded.version == 1
    assert loaded.last_collected_at == snapshot.last_collected_at
    assert loaded.sources["github"].status == "ok"
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].id == "github-1"
    assert loaded.tasks[0].title == "Sample task"


def test_store_missing_file_returns_empty_snapshot(tmp_path: Path) -> None:
    store = ExternalTaskStore(tmp_path / "missing.json")
    loaded = store.load()
    assert loaded == Snapshot()
    assert loaded.tasks == []
    assert loaded.sources == {}
    assert loaded.last_collected_at is None


def test_store_corrupt_json_returns_empty_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "external_tasks.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = ExternalTaskStore(path)
    loaded = store.load()
    assert loaded == Snapshot()


def test_store_schema_mismatch_returns_empty_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "external_tasks.json"
    path.write_text('{"version": 1, "tasks": [{"id": 123}]}', encoding="utf-8")
    store = ExternalTaskStore(path)
    loaded = store.load()
    assert loaded == Snapshot()


# ── collect_all ────────────────────────────────────


def test_collect_all_all_sources_success(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)

    def _github() -> list[ExternalTask]:
        return [_task(id="github-1", source_type="github", source_icon="github")]

    def _slack() -> list[ExternalTask]:
        return [
            _task(
                id="slack-1",
                source_type="slack",
                source_icon="slack",
                source_url="https://slack.com/archives/C01",
            )
        ]

    def _chatwork() -> list[ExternalTask]:
        return [
            _task(
                id="chatwork-1",
                source_type="chatwork",
                source_icon="chatwork",
                source_url="https://www.chatwork.com/",
            )
        ]

    def _gmail() -> list[ExternalTask]:
        return [
            _task(
                id="gmail-1",
                source_type="gmail",
                source_icon="gmail",
                source_url="https://mail.google.com/",
            )
        ]

    monkeypatch.setitem(SOURCE_REGISTRY, "github", _github)
    monkeypatch.setitem(SOURCE_REGISTRY, "slack", _slack)
    monkeypatch.setitem(SOURCE_REGISTRY, "chatwork", _chatwork)
    monkeypatch.setitem(SOURCE_REGISTRY, "gmail", _gmail)

    result = collect_all(_enabled_config(), Snapshot(), now)

    assert result.last_collected_at == now.isoformat()
    assert set(result.sources) == {"github", "slack", "chatwork", "gmail"}
    assert all(h.status == "ok" for h in result.sources.values())
    assert all(h.error is None for h in result.sources.values())
    assert {t.id for t in result.tasks} == {
        "github-1",
        "slack-1",
        "chatwork-1",
        "gmail-1",
    }


def test_collect_all_one_source_exception_keeps_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    previous = Snapshot(
        tasks=[
            _task(id="github-old", source_type="github", title="Old GH"),
            _task(id="slack-old", source_type="slack", title="Old Slack"),
        ],
        sources={
            "github": SourceHealth(status="ok", collected_at="2026-07-19T00:00:00+00:00"),
            "slack": SourceHealth(status="ok", collected_at="2026-07-19T00:00:00+00:00"),
        },
    )

    def _github_ok() -> list[ExternalTask]:
        return [_task(id="github-new", source_type="github", title="New GH")]

    def _slack_fail() -> list[ExternalTask]:
        raise RuntimeError("slack API down")

    def _empty() -> list[ExternalTask]:
        return []

    monkeypatch.setitem(SOURCE_REGISTRY, "github", _github_ok)
    monkeypatch.setitem(SOURCE_REGISTRY, "slack", _slack_fail)
    monkeypatch.setitem(SOURCE_REGISTRY, "chatwork", _empty)
    monkeypatch.setitem(SOURCE_REGISTRY, "gmail", _empty)

    result = collect_all(_enabled_config(), previous, now)

    assert result.sources["github"].status == "ok"
    assert result.sources["slack"].status == "unavailable"
    assert "slack API down" in (result.sources["slack"].error or "")
    ids = {t.id for t in result.tasks}
    assert "github-new" in ids
    assert "slack-old" in ids
    assert "github-old" not in ids


def test_collect_all_credential_not_found_marks_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)

    def _github_no_cred() -> list[ExternalTask]:
        raise CredentialNotFoundError("GITHUB_TOKEN missing")

    def _empty() -> list[ExternalTask]:
        return []

    monkeypatch.setitem(SOURCE_REGISTRY, "github", _github_no_cred)
    monkeypatch.setitem(SOURCE_REGISTRY, "slack", _empty)
    monkeypatch.setitem(SOURCE_REGISTRY, "chatwork", _empty)
    monkeypatch.setitem(SOURCE_REGISTRY, "gmail", _empty)

    result = collect_all(_enabled_config(), Snapshot(), now)

    assert result.sources["github"].status == "unavailable"
    assert "GITHUB_TOKEN missing" in (result.sources["github"].error or "")
    assert result.tasks == []


def test_collect_all_disabled_source_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    called: list[str] = []

    def _github() -> list[ExternalTask]:
        called.append("github")
        return [_task(id="github-1")]

    def _should_not_run() -> list[ExternalTask]:
        called.append("slack")
        raise AssertionError("disabled source must not be called")

    monkeypatch.setitem(SOURCE_REGISTRY, "github", _github)
    monkeypatch.setitem(SOURCE_REGISTRY, "slack", _should_not_run)
    monkeypatch.setitem(SOURCE_REGISTRY, "chatwork", _should_not_run)
    monkeypatch.setitem(SOURCE_REGISTRY, "gmail", _should_not_run)

    config = _enabled_config(github=True, slack=False, chatwork=False, gmail=False)
    result = collect_all(config, Snapshot(), now)

    assert called == ["github"]
    assert set(result.sources) == {"github"}
    assert "slack" not in result.sources
    assert len(result.tasks) == 1


def test_collect_all_priority_age_decay(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    old = (now - timedelta(days=4)).isoformat()
    recent = (now - timedelta(days=1)).isoformat()

    def _github() -> list[ExternalTask]:
        return [
            _task(id="github-old", last_updated_at=old, priority=50),
            _task(id="github-floor", last_updated_at=old, priority=25),
            _task(id="github-recent", last_updated_at=recent, priority=50),
        ]

    def _empty() -> list[ExternalTask]:
        return []

    monkeypatch.setitem(SOURCE_REGISTRY, "github", _github)
    monkeypatch.setitem(SOURCE_REGISTRY, "slack", _empty)
    monkeypatch.setitem(SOURCE_REGISTRY, "chatwork", _empty)
    monkeypatch.setitem(SOURCE_REGISTRY, "gmail", _empty)

    result = collect_all(
        _enabled_config(slack=False, chatwork=False, gmail=False),
        Snapshot(),
        now,
    )
    by_id = {t.id: t for t in result.tasks}
    assert by_id["github-old"].priority == 30  # 50 - 20
    assert by_id["github-floor"].priority == 10  # max(10, 25-20)
    assert by_id["github-recent"].priority == 50


def test_collect_all_title_control_chars_and_truncate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    dirty = "hello\x00world\x1f!\x7f" + ("あ" * 250)

    def _github() -> list[ExternalTask]:
        return [_task(id="github-1", title=dirty)]

    def _empty() -> list[ExternalTask]:
        return []

    monkeypatch.setitem(SOURCE_REGISTRY, "github", _github)
    monkeypatch.setitem(SOURCE_REGISTRY, "slack", _empty)
    monkeypatch.setitem(SOURCE_REGISTRY, "chatwork", _empty)
    monkeypatch.setitem(SOURCE_REGISTRY, "gmail", _empty)

    result = collect_all(
        _enabled_config(slack=False, chatwork=False, gmail=False),
        Snapshot(),
        now,
    )
    title = result.tasks[0].title
    assert "\x00" not in title
    assert "\x1f" not in title
    assert "\x7f" not in title
    assert title.startswith("helloworld!")
    assert len(title) == 200


def test_collect_all_javascript_url_becomes_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)

    def _github() -> list[ExternalTask]:
        return [
            _task(id="github-js", source_url="javascript:alert(1)"),
            _task(id="github-http", source_url="http://example.com/a"),
            _task(id="github-https", source_url="https://example.com/b"),
            _task(id="github-ftp", source_url="ftp://example.com/c"),
        ]

    def _empty() -> list[ExternalTask]:
        return []

    monkeypatch.setitem(SOURCE_REGISTRY, "github", _github)
    monkeypatch.setitem(SOURCE_REGISTRY, "slack", _empty)
    monkeypatch.setitem(SOURCE_REGISTRY, "chatwork", _empty)
    monkeypatch.setitem(SOURCE_REGISTRY, "gmail", _empty)

    result = collect_all(
        _enabled_config(slack=False, chatwork=False, gmail=False),
        Snapshot(),
        now,
    )
    by_id = {t.id: t for t in result.tasks}
    assert by_id["github-js"].source_url is None
    assert by_id["github-http"].source_url == "http://example.com/a"
    assert by_id["github-https"].source_url == "https://example.com/b"
    assert by_id["github-ftp"].source_url is None
