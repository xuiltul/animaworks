from __future__ import annotations

import json
import queue
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from core.activity.logger import ActivityLogger
from core.config import invalidate_cache, save_config
from core.config.schemas import AnimaWorksConfig, EventExportConfig
from core.infra.event_export import EventExporter, get_event_exporter, reset_event_exporters
from core.usage.token_usage import TokenUsageLogger


@pytest.fixture(autouse=True)
def _reset_exporters() -> None:
    reset_event_exporters()
    yield
    reset_event_exporters()


def test_disabled_exporter_has_no_spool_or_worker(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    exporter = EventExporter(anima_dir, EventExportConfig())

    exporter.emit({"kind": "activity", "event": {"type": "test"}})

    assert exporter.worker_alive is False
    assert not exporter.spool_dir.exists()
    assert get_event_exporter(anima_dir, EventExportConfig()) is None


def test_unconfigured_loggers_do_not_create_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    config = AnimaWorksConfig()
    monkeypatch.setattr("core.config.load_config", lambda: config)

    ActivityLogger(anima_dir).log("custom_event", content="local only")
    TokenUsageLogger(anima_dir).log(
        model="claude-sonnet-4-6",
        trigger="chat",
        mode="a",
    )

    assert not (anima_dir / "state" / "event_export_spool").exists()
    assert get_event_exporter(anima_dir, config.event_export) is None


def test_unconfigured_hooks_propagate_disable_to_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    config = AnimaWorksConfig()
    get_exporter = Mock(return_value=None)
    monkeypatch.setattr("core.config.load_config", lambda: config)
    monkeypatch.setattr("core.infra.event_export.get_event_exporter", get_exporter)

    ActivityLogger(anima_dir).log("custom_event", content="local only")
    TokenUsageLogger(anima_dir).log(
        model="claude-sonnet-4-6",
        trigger="chat",
        mode="a",
    )

    assert get_exporter.call_count == 2
    assert all(call.args[1].url is None for call in get_exporter.call_args_list)


def test_registry_reuses_one_worker_per_anima(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    config = EventExportConfig(url="http://127.0.0.1:1", backoff_base_seconds=0)

    first = get_event_exporter(anima_dir, config)
    second = get_event_exporter(anima_dir, config)

    assert first is not None
    assert first is second
    assert first.worker_alive


def test_cross_process_lock_allows_only_one_delivery_worker(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    config = EventExportConfig(url="http://127.0.0.1:1")
    first = EventExporter(anima_dir, config)
    second = EventExporter(anima_dir, config)
    try:
        assert first.worker_alive is True
        assert second.worker_alive is False
    finally:
        first.stop(timeout=2)
        second.stop(timeout=2)


def test_disable_and_reenable_preserves_worker_identity(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    enabled = EventExportConfig(url="http://127.0.0.1:1")
    first = get_event_exporter(anima_dir, enabled)

    assert get_event_exporter(anima_dir, EventExportConfig()) is None
    second = get_event_exporter(anima_dir, enabled)

    assert second is first
    assert second is not None
    assert second.worker_alive


def test_activity_event_type_filter_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    config = AnimaWorksConfig(
        event_export=EventExportConfig(
            url="http://127.0.0.1:1",
            event_types=["external_send"],
        )
    )
    exporter = Mock()
    monkeypatch.setattr("core.config.load_config", lambda: config)
    monkeypatch.setattr("core.infra.event_export.get_event_exporter", lambda *_args: exporter)
    activity = ActivityLogger(anima_dir)

    activity.log("external", content="not an exact match")
    activity.log("external_send", content="send payload", origin="human")

    exporter.emit.assert_called_once()
    payload = exporter.emit.call_args.args[0]
    assert payload["kind"] == "activity"
    assert payload["anima"] == "alice"
    assert payload["event"]["type"] == "external_send"
    assert payload["event"]["content"] == "send payload"
    assert payload["event"]["origin"] == "human"


def test_activity_safe_write_failure_is_not_exported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activity = ActivityLogger(tmp_path / "animas" / "alice")
    export = Mock()
    monkeypatch.setattr(activity, "_append", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(activity, "_export_event", export)

    entry = activity.log("custom_event", content="still returns", safe=True)

    assert entry.content == "still returns"
    export.assert_not_called()


def test_token_usage_respects_include_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    config = AnimaWorksConfig(
        event_export=EventExportConfig(
            url="http://127.0.0.1:1",
            include_token_usage=False,
        )
    )
    exporter = Mock()
    monkeypatch.setattr("core.config.load_config", lambda: config)
    monkeypatch.setattr("core.infra.event_export.get_event_exporter", lambda *_args: exporter)

    TokenUsageLogger(anima_dir).log(
        model="claude-sonnet-4-6",
        trigger="chat",
        mode="a",
        input_tokens=10,
        output_tokens=5,
    )

    exporter.emit.assert_not_called()


def test_token_usage_write_failure_is_not_exported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    token_logger = TokenUsageLogger(anima_dir)
    export = Mock()
    monkeypatch.setattr(token_logger, "_export_event", export)

    def fail_open(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", fail_open)

    token_logger.log(model="claude-sonnet-4-6", trigger="chat", mode="a")

    export.assert_not_called()


def test_spool_limit_discards_oldest_event(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    exporter = EventExporter(
        anima_dir,
        EventExportConfig(
            url="http://127.0.0.1:1",
            max_retries=0,
            backoff_base_seconds=0,
            spool_max_mb=1,
        ),
    )
    try:
        exporter.emit({"sequence": "oldest", "content": "a" * 700_000})
        exporter.emit({"sequence": "newest", "content": "b" * 700_000})

        deadline = time.monotonic() + 2
        files: list[Path] = []
        while time.monotonic() < deadline:
            files = sorted(exporter.spool_dir.glob("*.jsonl"))
            if len(files) == 1 and "discarded 1 oldest event" in caplog.text:
                break
            time.sleep(0.01)

        assert len(files) == 1
        assert json.loads(files[0].read_text(encoding="utf-8"))["sequence"] == "newest"
        assert "discarded 1 oldest event" in caplog.text
    finally:
        exporter.stop(timeout=2)


def test_delivery_rejects_redirects_and_preserves_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock(status_code=307)
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    exporter = EventExporter(
        tmp_path / "animas" / "alice",
        EventExportConfig(
            url="http://127.0.0.1:1",
            headers={"X-API-Key": "secret"},
            max_retries=0,
            backoff_base_seconds=0,
        ),
    )
    try:
        exporter.emit({"kind": "activity", "event": {"type": "test"}})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not post.called:
            time.sleep(0.01)

        assert post.called
        assert post.call_args.kwargs["headers"] == {"X-API-Key": "secret"}
        assert post.call_args.kwargs["allow_redirects"] is False
        assert list(exporter.spool_dir.glob("*.jsonl"))
    finally:
        exporter.stop(timeout=2)


def test_emit_write_failure_does_not_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exporter = EventExporter(
        tmp_path / "animas" / "alice",
        EventExportConfig(url="http://127.0.0.1:1"),
    )
    exporter.stop(timeout=2)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("core.infra.event_export.atomic_write_text", fail_write)

    exporter.emit({"kind": "activity", "event": {"type": "test"}})

    assert "Failed to append event export spool" in caplog.text
    assert not list(exporter.spool_dir.glob("*.tmp"))


class _ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class _Receiver:
    def __init__(self, port: int = 0) -> None:
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                receiver.events.put(payload)
                self.send_response(204)
                self.end_headers()

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self.server = _ReusableHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/events"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def receiver() -> Iterator[_Receiver]:
    value = _Receiver()
    yield value
    value.close()


def _configure(data_dir: Path, url: str | None) -> None:
    save_config(
        AnimaWorksConfig(
            event_export=EventExportConfig(
                url=url,
                max_retries=0,
                backoff_base_seconds=0.01,
            )
        ),
        data_dir / "config.json",
    )
    invalidate_cache()


def _log_both(anima_dir: Path) -> None:
    ActivityLogger(anima_dir).log(
        "external_send",
        content="audit me",
        origin="human",
        ctx="chat",
    )
    TokenUsageLogger(anima_dir).log(
        model="claude-sonnet-4-6",
        trigger="chat",
        mode="a",
        input_tokens=100,
        output_tokens=25,
    )


@pytest.mark.integration
@pytest.mark.e2e
def test_activity_and_token_usage_reach_loopback_receiver(
    data_dir: Path,
    receiver: _Receiver,
) -> None:
    _configure(data_dir, receiver.url)
    anima_dir = data_dir / "animas" / "alice"
    anima_dir.mkdir(parents=True)

    _log_both(anima_dir)

    events = [receiver.events.get(timeout=3), receiver.events.get(timeout=3)]
    by_kind = {event["kind"]: event for event in events}
    assert set(by_kind) == {"activity", "token_usage"}
    assert by_kind["activity"]["anima"] == "alice"
    assert by_kind["activity"]["event"]["type"] == "external_send"
    assert by_kind["activity"]["event"]["origin"] == "human"
    assert by_kind["token_usage"]["event"]["total_tokens"] == 125


@pytest.mark.integration
@pytest.mark.e2e
def test_receiver_down_does_not_block_and_spool_recovers(data_dir: Path) -> None:
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    url = f"http://127.0.0.1:{port}/events"
    _configure(data_dir, url)
    anima_dir = data_dir / "animas" / "alice"
    anima_dir.mkdir(parents=True)

    started = time.monotonic()
    _log_both(anima_dir)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert len(list((anima_dir / "state" / "event_export_spool").glob("*.jsonl"))) == 2

    receiver = _Receiver(port)
    try:
        events = [receiver.events.get(timeout=4), receiver.events.get(timeout=4)]
        assert {event["kind"] for event in events} == {"activity", "token_usage"}

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if not list((anima_dir / "state" / "event_export_spool").glob("*.jsonl")):
                break
            time.sleep(0.01)
        assert not list((anima_dir / "state" / "event_export_spool").glob("*.jsonl"))
    finally:
        receiver.close()


@pytest.mark.integration
@pytest.mark.e2e
def test_persisted_spool_is_delivered_after_exporter_restart(data_dir: Path) -> None:
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    _configure(data_dir, f"http://127.0.0.1:{port}/events")
    anima_dir = data_dir / "animas" / "alice"
    anima_dir.mkdir(parents=True)

    ActivityLogger(anima_dir).log("external_send", content="survive restart")
    spool_dir = anima_dir / "state" / "event_export_spool"
    assert len(list(spool_dir.glob("*.jsonl"))) == 1
    reset_event_exporters()

    receiver = _Receiver(port)
    try:
        # Logger construction is the per-Anima process-start path.  No new
        # event is needed to wake delivery of the persisted spool entry.
        ActivityLogger(anima_dir)

        event = receiver.events.get(timeout=4)
        assert event["event"]["content"] == "survive restart"
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and list(spool_dir.glob("*.jsonl")):
            time.sleep(0.01)
        assert not list(spool_dir.glob("*.jsonl"))
    finally:
        receiver.close()


@pytest.mark.integration
@pytest.mark.e2e
def test_worker_picks_up_endpoint_change_without_new_event(data_dir: Path) -> None:
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        unavailable_port = reserved.getsockname()[1]
    _configure(data_dir, f"http://127.0.0.1:{unavailable_port}/events")
    anima_dir = data_dir / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    ActivityLogger(anima_dir).log("external_send", content="reload endpoint")
    assert list((anima_dir / "state" / "event_export_spool").glob("*.jsonl"))

    receiver = _Receiver()
    try:
        _configure(data_dir, receiver.url)

        event = receiver.events.get(timeout=4)
        assert event["event"]["content"] == "reload endpoint"
    finally:
        receiver.close()
