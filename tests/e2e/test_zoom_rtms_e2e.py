# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the Zoom RTMS ingestion pipeline.

Stands up an in-process ``asyncio`` WebSocket mock of Zoom's RTMS signaling
and media servers (using the real :mod:`websockets` library) and drives the
real :class:`server.zoom_gateway.ZoomRTMSManager` through the full path:

    rtms_started webhook
      → signaling WS connect + HMAC-SHA256 handshake
      → media WS connect + data handshake + CLIENT_READY_ACK
      → transcript frames (msg_type=17)
      → utterance buffer + chunk flush
      → Messenger.receive_external → shared/inbox/{anima}/*.json

Nothing in the pipeline is mocked below the gateway: the Messenger writes
real inbox JSON files into the ``data_dir`` fixture's isolated filesystem,
and assertions read those files back.

Covered scenarios (issue 20260703_zoom-rtms-meeting-listener.md):
  1. Full flow — handshake → transcript → chunk → inbox (meeting_transcript)
  2. End flow — rtms_stopped flushes the tail and injects meeting_ended
     (and the zero-transcript meeting injects nothing)
  3. Disconnect/reconnect recovery + source_message_id dedup
  4. Two concurrent meetings with independent buffers / sequence numbers
  5. Handshake signature rejection (wrong client_secret → no ingestion)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import core.messenger as messenger_mod
from core.config import invalidate_cache
from server.zoom_gateway import ZoomRTMSManager

websockets = pytest.importorskip("websockets")

pytestmark = pytest.mark.e2e

# ── RTMS protocol constants (mirror server/zoom_gateway.py) ───────────────
MSG_SIGNALING_HANDSHAKE_REQ = 1
MSG_SIGNALING_HANDSHAKE_RESP = 2
MSG_DATA_HANDSHAKE_REQ = 3
MSG_DATA_HANDSHAKE_RESP = 4
MSG_CLIENT_READY_ACK = 7
MSG_KEEP_ALIVE_REQ = 12
MSG_KEEP_ALIVE_RESP = 13
MSG_MEDIA_DATA_TRANSCRIPT = 17
STATUS_OK = 0
STATUS_INVALID_SIGNATURE = 13

CLIENT_ID = "test-zoom-client-id"
CLIENT_SECRET = "test-zoom-client-secret"

_DROP = object()  # sentinel: close the media WebSocket to simulate a drop


def _signature(client_id: str, client_secret: str, meeting_uuid: str, stream_id: str) -> str:
    """Reproduce the gateway's HMAC-SHA256 handshake signature."""
    message = f"{client_id},{meeting_uuid},{stream_id}"
    return hmac.new(client_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


class MockRTMSServer:
    """In-process stand-in for Zoom's RTMS signaling + media WebSocket servers.

    Two ``asyncio`` WebSocket servers bind ephemeral localhost ports.  The
    signaling server verifies the handshake signature and hands back the
    media server URL; the media server verifies its own handshake and then
    streams whatever transcript frames the test injects.
    """

    def __init__(self, *, expected_secret: str = CLIENT_SECRET, reject_bad_signature: bool = True) -> None:
        self._expected_secret = expected_secret
        self._reject_bad_signature = reject_bad_signature

        self._signaling_server: Any = None
        self._media_server: Any = None
        self.signaling_url = ""
        self.media_url = ""

        # per-meeting outbound transcript queues (media plane)
        self._queues: dict[str, asyncio.Queue[Any]] = {}

        # observability for assertions
        self.signaling_handshakes: list[tuple[str, bool]] = []
        self.media_handshakes: list[tuple[str, bool]] = []
        self.signaling_connect_count: dict[str, int] = {}
        self.media_connect_count: dict[str, int] = {}
        self.client_ready_count: dict[str, int] = {}
        self.keepalive_ack_count: dict[str, int] = {}

    async def start(self) -> None:
        self._signaling_server = await websockets.serve(self._signaling_handler, "127.0.0.1", 0)
        self._media_server = await websockets.serve(self._media_handler, "127.0.0.1", 0)
        sig_port = self._signaling_server.sockets[0].getsockname()[1]
        media_port = self._media_server.sockets[0].getsockname()[1]
        self.signaling_url = f"ws://127.0.0.1:{sig_port}"
        self.media_url = f"ws://127.0.0.1:{media_port}"

    async def stop(self) -> None:
        for server in (self._signaling_server, self._media_server):
            if server is not None:
                server.close()
                await server.wait_closed()

    # ── injection API (called from the test coroutine) ──────────────────
    def inject_transcript(self, meeting_uuid: str, speaker: str, text: str) -> None:
        """Queue a transcript frame for delivery on the media plane."""
        self._queue(meeting_uuid).put_nowait(
            {"msg_type": MSG_MEDIA_DATA_TRANSCRIPT, "user_name": speaker, "content": text}
        )

    def drop_media(self, meeting_uuid: str) -> None:
        """Force the current media WebSocket for a meeting to close."""
        self._queue(meeting_uuid).put_nowait(_DROP)

    # ── internals ───────────────────────────────────────────────────────
    def _queue(self, meeting_uuid: str) -> asyncio.Queue[Any]:
        q = self._queues.get(meeting_uuid)
        if q is None:
            q = asyncio.Queue()
            self._queues[meeting_uuid] = q
        return q

    def _verify(self, msg: dict[str, Any]) -> tuple[str, bool]:
        meeting_uuid = str(msg.get("meeting_uuid") or "")
        stream_id = str(msg.get("rtms_stream_id") or "")
        expected = _signature(CLIENT_ID, self._expected_secret, meeting_uuid, stream_id)
        ok = hmac.compare_digest(str(msg.get("signature") or ""), expected)
        return meeting_uuid, ok

    async def _signaling_handler(self, ws: Any) -> None:
        meeting_uuid = ""
        try:
            async for raw in ws:
                msg = json.loads(raw)
                mt = msg.get("msg_type")
                if mt == MSG_SIGNALING_HANDSHAKE_REQ:
                    meeting_uuid, ok = self._verify(msg)
                    self.signaling_handshakes.append((meeting_uuid, ok))
                    if not ok and self._reject_bad_signature:
                        await ws.send(
                            json.dumps(
                                {
                                    "msg_type": MSG_SIGNALING_HANDSHAKE_RESP,
                                    "status_code": STATUS_INVALID_SIGNATURE,
                                    "meeting_uuid": meeting_uuid,
                                }
                            )
                        )
                        continue
                    self.signaling_connect_count[meeting_uuid] = (
                        self.signaling_connect_count.get(meeting_uuid, 0) + 1
                    )
                    await ws.send(
                        json.dumps(
                            {
                                "msg_type": MSG_SIGNALING_HANDSHAKE_RESP,
                                "status_code": STATUS_OK,
                                "meeting_uuid": meeting_uuid,
                                "media_server": {"server_urls": {"all": self.media_url}},
                            }
                        )
                    )
                    # Probe keep-alive handling on the control plane.
                    await ws.send(json.dumps({"msg_type": MSG_KEEP_ALIVE_REQ, "timestamp": 1234567890}))
                elif mt == MSG_KEEP_ALIVE_RESP:
                    self.keepalive_ack_count[meeting_uuid] = self.keepalive_ack_count.get(meeting_uuid, 0) + 1
                elif mt == MSG_CLIENT_READY_ACK:
                    self.client_ready_count[meeting_uuid] = self.client_ready_count.get(meeting_uuid, 0) + 1
        except Exception:
            return

    async def _media_handler(self, ws: Any) -> None:
        meeting_uuid = ""
        try:
            raw = await ws.recv()
            msg = json.loads(raw)
            meeting_uuid, ok = self._verify(msg)
            self.media_handshakes.append((meeting_uuid, ok))
            if not ok and self._reject_bad_signature:
                await ws.send(
                    json.dumps(
                        {
                            "msg_type": MSG_DATA_HANDSHAKE_RESP,
                            "status_code": STATUS_INVALID_SIGNATURE,
                            "meeting_uuid": meeting_uuid,
                        }
                    )
                )
                return
            await ws.send(
                json.dumps(
                    {
                        "msg_type": MSG_DATA_HANDSHAKE_RESP,
                        "status_code": STATUS_OK,
                        "meeting_uuid": meeting_uuid,
                    }
                )
            )
            self.media_connect_count[meeting_uuid] = self.media_connect_count.get(meeting_uuid, 0) + 1

            queue = self._queue(meeting_uuid)
            # Race the transcript queue against connection closure so the
            # handler unblocks (and the server can shut down) when the client
            # closes — a plain ``queue.get()`` would otherwise wait forever.
            closed = asyncio.ensure_future(ws.wait_closed())
            try:
                while True:
                    getter = asyncio.ensure_future(queue.get())
                    done, _ = await asyncio.wait({getter, closed}, return_when=asyncio.FIRST_COMPLETED)
                    if closed in done:
                        getter.cancel()
                        return
                    item = getter.result()
                    if item is _DROP:
                        await ws.close()
                        return
                    await ws.send(json.dumps(item))
            finally:
                closed.cancel()
        except Exception:
            return


# ── generic helpers ──────────────────────────────────────────────────────


async def _wait_for(predicate: Callable[[], Any], *, timeout: float = 10.0, interval: float = 0.02) -> Any:
    """Poll *predicate* until it returns truthy, then return that value."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def _read_inbox(shared_dir: Path, anima: str) -> list[dict[str, Any]]:
    """Return every inbox message for *anima*, sorted by filename."""
    inbox = shared_dir / "inbox" / anima
    if not inbox.exists():
        return []
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(inbox.glob("*.json"))]


def _transcripts(shared_dir: Path, anima: str) -> list[dict[str, Any]]:
    return [m for m in _read_inbox(shared_dir, anima) if m.get("intent") == "meeting_transcript"]


def _started_payload(
    server: MockRTMSServer,
    *,
    meeting_uuid: str,
    meeting_id: str,
    topic: str,
    stream_id: str = "stream-xyz",
) -> dict[str, Any]:
    return {
        "event": "meeting.rtms_started",
        "meeting_uuid": meeting_uuid,
        "meeting_id": meeting_id,
        "rtms_stream_id": stream_id,
        "topic": topic,
        "server_urls": server.signaling_url,
    }


def _configure_zoom(
    data_dir: Path,
    *,
    enabled: bool = True,
    default_anima: str = "",
    meeting_mapping: dict[str, str] | None = None,
    chunk_interval_seconds: int = 300,
    chunk_max_chars: int = 4000,
) -> None:
    """Write the Zoom RTMS block into config.json and refresh the cache."""
    config_path = data_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("external_messaging", {})
    config["external_messaging"]["zoom"] = {
        "enabled": enabled,
        "default_anima": default_anima,
        "meeting_mapping": meeting_mapping or {},
        "chunk_interval_seconds": chunk_interval_seconds,
        "chunk_max_chars": chunk_max_chars,
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    invalidate_cache()


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_external_dedup():
    """Isolate the module-level external-dedup cache between tests."""
    messenger_mod._external_seen.clear()
    yield
    messenger_mod._external_seen.clear()


@pytest.fixture
async def harness(data_dir: Path, make_anima, monkeypatch: pytest.MonkeyPatch):
    """Provide RTMS credentials plus mock-server / manager factories.

    All spawned mock servers and managers are torn down on exit.
    """
    monkeypatch.setenv("ZOOM_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("ZOOM_CLIENT_SECRET", CLIENT_SECRET)

    servers: list[MockRTMSServer] = []
    managers: list[ZoomRTMSManager] = []

    async def make_server(**kwargs: Any) -> MockRTMSServer:
        server = MockRTMSServer(**kwargs)
        await server.start()
        servers.append(server)
        return server

    async def make_manager() -> ZoomRTMSManager:
        manager = ZoomRTMSManager()
        await manager.start()
        managers.append(manager)
        return manager

    yield SimpleNamespace(
        data_dir=data_dir,
        shared_dir=data_dir / "shared",
        make_anima=make_anima,
        make_server=make_server,
        make_manager=make_manager,
        configure=lambda **kw: _configure_zoom(data_dir, **kw),
    )

    for manager in managers:
        await manager.stop()
    for server in servers:
        await server.stop()


# ── scenario 1: full flow ─────────────────────────────────────────────────


async def test_full_flow_handshake_transcript_to_inbox(harness):
    """rtms_started → signed handshake → transcript → meeting_transcript chunk."""
    harness.make_anima("kotoha")
    text1, text2 = "予算の件ですが来週締めます", "了解しました確認します"
    line1 = f"田中: {text1}"
    # Threshold placed so the first utterance stays buffered and the second
    # crosses it — a single deterministic chunk holding both speakers.
    harness.configure(
        meeting_mapping={"1000000001": "kotoha"},
        chunk_interval_seconds=3600,
        chunk_max_chars=len(line1) + 2,
    )
    server = await harness.make_server()
    manager = await harness.make_manager()

    meeting_uuid = "uuid-full-flow-001"
    await manager.handle_rtms_started(
        _started_payload(server, meeting_uuid=meeting_uuid, meeting_id="1000000001", topic="予算会議")
    )

    session = await _wait_for(lambda: manager._sessions.get(meeting_uuid))
    await _wait_for(lambda: session.ready)

    # Signature verified on both planes; keep-alive answered on signaling.
    assert (meeting_uuid, True) in server.signaling_handshakes
    assert (meeting_uuid, True) in server.media_handshakes
    await _wait_for(lambda: server.keepalive_ack_count.get(meeting_uuid, 0) >= 1)
    assert server.client_ready_count.get(meeting_uuid, 0) >= 1

    server.inject_transcript(meeting_uuid, "田中", text1)
    server.inject_transcript(meeting_uuid, "佐藤", text2)

    chunks = await _wait_for(lambda: _transcripts(harness.shared_dir, "kotoha") or None)
    assert len(chunks) == 1
    msg = chunks[0]
    assert msg["source"] == "zoom"
    assert msg["intent"] == "meeting_transcript"
    assert msg["to_person"] == "kotoha"
    assert msg["external_channel_id"] == "1000000001"
    assert msg["external_thread_ts"] == f"zoom-{meeting_uuid}"
    assert msg["source_message_id"] == f"{meeting_uuid}:1"

    content = msg["content"]
    assert content.startswith("[Zoom会議実況 チャンク#1 | 会議: 予算会議 (1000000001) |")
    assert f"田中: {text1}" in content
    assert f"佐藤: {text2}" in content
    # Speaker order preserved.
    assert content.index(f"田中: {text1}") < content.index(f"佐藤: {text2}")


# ── scenario 2: end flow ──────────────────────────────────────────────────


async def test_end_flow_flushes_tail_and_injects_meeting_ended(harness):
    """rtms_stopped flushes the remaining buffer, then injects meeting_ended."""
    harness.make_anima("kotoha")
    harness.configure(
        meeting_mapping={"1000000002": "kotoha"},
        chunk_interval_seconds=3600,  # never time-flush during the test
        chunk_max_chars=100_000,  # never char-flush; only rtms_stopped flushes
    )
    server = await harness.make_server()
    manager = await harness.make_manager()

    meeting_uuid = "uuid-end-flow-001"
    await manager.handle_rtms_started(
        _started_payload(server, meeting_uuid=meeting_uuid, meeting_id="1000000002", topic="定例会")
    )
    session = await _wait_for(lambda: manager._sessions.get(meeting_uuid))
    await _wait_for(lambda: session.ready)

    server.inject_transcript(meeting_uuid, "山田", "本日の議題を始めます")
    await _wait_for(lambda: len(session.buffer) == 1)

    await manager.handle_rtms_stopped({"meeting_uuid": meeting_uuid})

    messages = _read_inbox(harness.shared_dir, "kotoha")
    transcripts = [m for m in messages if m["intent"] == "meeting_transcript"]
    ended = [m for m in messages if m["intent"] == "meeting_ended"]

    assert len(transcripts) == 1  # tail flushed as one chunk
    assert "山田: 本日の議題を始めます" in transcripts[0]["content"]

    assert len(ended) == 1
    end_msg = ended[0]
    assert end_msg["source"] == "zoom"
    assert end_msg["source_message_id"] == f"{meeting_uuid}:end"
    assert end_msg["external_channel_id"] == "1000000002"
    assert end_msg["content"].startswith("[Zoom会議終了 | 会議: 定例会 (1000000002) |")
    assert "全1チャンク配信済み" in end_msg["content"]
    assert "会議が終了しました" in end_msg["content"]

    # Session finalised and removed.
    assert meeting_uuid not in manager._sessions


async def test_end_flow_no_transcript_injects_nothing(harness):
    """A meeting with zero transcript emits no chunk and no meeting_ended."""
    harness.make_anima("kotoha")
    harness.configure(
        meeting_mapping={"1000000003": "kotoha"},
        chunk_interval_seconds=3600,
        chunk_max_chars=100_000,
    )
    server = await harness.make_server()
    manager = await harness.make_manager()

    meeting_uuid = "uuid-silent-001"
    await manager.handle_rtms_started(
        _started_payload(server, meeting_uuid=meeting_uuid, meeting_id="1000000003", topic="無言会議")
    )
    session = await _wait_for(lambda: manager._sessions.get(meeting_uuid))
    await _wait_for(lambda: session.ready)

    await manager.handle_rtms_stopped({"meeting_uuid": meeting_uuid})

    assert _read_inbox(harness.shared_dir, "kotoha") == []
    assert meeting_uuid not in manager._sessions


# ── scenario 3: disconnect / reconnect + dedup ────────────────────────────


async def test_reconnect_recovers_and_dedup_prevents_duplicates(harness):
    """Media drop → re-handshake recovery; source_message_id dedup holds."""
    harness.make_anima("kotoha")
    # chunk_max_chars=1 → each utterance flushes as its own chunk, so the
    # sequence numbers used as dedup keys are easy to reason about.
    harness.configure(
        meeting_mapping={"1000000004": "kotoha"},
        chunk_interval_seconds=3600,
        chunk_max_chars=1,
    )
    server = await harness.make_server()
    manager = await harness.make_manager()

    meeting_uuid = "uuid-reconnect-001"
    await manager.handle_rtms_started(
        _started_payload(server, meeting_uuid=meeting_uuid, meeting_id="1000000004", topic="復旧テスト")
    )
    session = await _wait_for(lambda: manager._sessions.get(meeting_uuid))
    await _wait_for(lambda: session.ready)

    server.inject_transcript(meeting_uuid, "田中", "発話イチ")
    await _wait_for(lambda: any("発話イチ" in m["content"] for m in _transcripts(harness.shared_dir, "kotoha")))

    first_chunk = _transcripts(harness.shared_dir, "kotoha")[0]
    assert first_chunk["source_message_id"] == f"{meeting_uuid}:1"

    # Drop the media plane; the manager must re-handshake and resume.
    server.drop_media(meeting_uuid)
    await _wait_for(lambda: server.media_connect_count.get(meeting_uuid, 0) >= 2, timeout=15.0)
    await _wait_for(lambda: manager._sessions[meeting_uuid].ready, timeout=15.0)

    server.inject_transcript(meeting_uuid, "佐藤", "発話ニ")
    await _wait_for(lambda: any("発話ニ" in m["content"] for m in _transcripts(harness.shared_dir, "kotoha")))

    transcripts = _transcripts(harness.shared_dir, "kotoha")
    # Recovery delivered the post-reconnect utterance exactly once, and the
    # pre-reconnect utterance was not duplicated by the replay.
    assert sum("発話イチ" in m["content"] for m in transcripts) == 1
    assert sum("発話ニ" in m["content"] for m in transcripts) == 1
    ids = [m["source_message_id"] for m in transcripts]
    assert len(ids) == len(set(ids))  # unique dedup keys

    # A re-delivered chunk (identical source_message_id) is dropped by the
    # dedup guard the gateway relies on — no new inbox file appears.
    before = len(_read_inbox(harness.shared_dir, "kotoha"))
    from core.messenger import Messenger
    from core.paths import get_shared_dir

    duplicate = Messenger(get_shared_dir(), "kotoha").receive_external(
        content="[Zoom会議実況 チャンク#1 | replay]",
        source="zoom",
        source_message_id=f"{meeting_uuid}:1",
        intent="meeting_transcript",
    )
    assert duplicate is None
    assert len(_read_inbox(harness.shared_dir, "kotoha")) == before


# ── scenario 4: concurrent meetings ───────────────────────────────────────


async def test_concurrent_meetings_no_crosstalk(harness):
    """Two simultaneous meetings keep separate buffers and sequence numbers."""
    harness.make_anima("kotoha")
    harness.make_anima("aoi")
    harness.configure(
        meeting_mapping={"2000000001": "kotoha", "2000000002": "aoi"},
        chunk_interval_seconds=3600,
        chunk_max_chars=1,  # one chunk per utterance for clean sequence checks
    )
    server = await harness.make_server()
    manager = await harness.make_manager()

    uuid_a = "uuid-concurrent-A"
    uuid_b = "uuid-concurrent-B"
    await manager.handle_rtms_started(
        _started_payload(server, meeting_uuid=uuid_a, meeting_id="2000000001", topic="会議A", stream_id="stream-A")
    )
    await manager.handle_rtms_started(
        _started_payload(server, meeting_uuid=uuid_b, meeting_id="2000000002", topic="会議B", stream_id="stream-B")
    )
    await _wait_for(lambda: manager._sessions.get(uuid_a) and manager._sessions[uuid_a].ready)
    await _wait_for(lambda: manager._sessions.get(uuid_b) and manager._sessions[uuid_b].ready)

    server.inject_transcript(uuid_a, "田中", "エーいち")
    server.inject_transcript(uuid_b, "鈴木", "ビーいち")
    server.inject_transcript(uuid_a, "田中", "エーに")

    await _wait_for(lambda: len(_transcripts(harness.shared_dir, "kotoha")) == 2)
    await _wait_for(lambda: len(_transcripts(harness.shared_dir, "aoi")) == 1)

    kotoha_chunks = _transcripts(harness.shared_dir, "kotoha")
    aoi_chunks = _transcripts(harness.shared_dir, "aoi")

    # No cross-talk between the two meetings' inboxes.
    kotoha_text = "\n".join(m["content"] for m in kotoha_chunks)
    aoi_text = "\n".join(m["content"] for m in aoi_chunks)
    assert "エーいち" in kotoha_text and "エーに" in kotoha_text
    assert "ビー" not in kotoha_text
    assert "ビーいち" in aoi_text
    assert "エー" not in aoi_text

    # Independent, non-mixed sequence numbering per meeting.
    assert [m["source_message_id"] for m in kotoha_chunks] == [f"{uuid_a}:1", f"{uuid_a}:2"]
    assert [m["source_message_id"] for m in aoi_chunks] == [f"{uuid_b}:1"]

    # Correct routing metadata on each side.
    assert all(m["external_channel_id"] == "2000000001" for m in kotoha_chunks)
    assert all(m["external_thread_ts"] == f"zoom-{uuid_a}" for m in kotoha_chunks)
    assert all(m["external_channel_id"] == "2000000002" for m in aoi_chunks)
    assert all(m["external_thread_ts"] == f"zoom-{uuid_b}" for m in aoi_chunks)


# ── scenario 5: bad-signature handshake is rejected ───────────────────────


async def test_bad_signature_handshake_rejected(harness):
    """A signature computed with the wrong secret yields no ingestion."""
    harness.make_anima("kotoha")
    harness.configure(
        meeting_mapping={"3000000001": "kotoha"},
        chunk_interval_seconds=3600,
        chunk_max_chars=1,
    )
    # Server expects a different client_secret than the manager signs with.
    server = await harness.make_server(expected_secret="a-completely-different-secret")
    manager = await harness.make_manager()

    meeting_uuid = "uuid-badsig-001"
    await manager.handle_rtms_started(
        _started_payload(server, meeting_uuid=meeting_uuid, meeting_id="3000000001", topic="不正署名")
    )

    # Signaling handshake attempted and rejected as invalid.
    await _wait_for(lambda: any(u == meeting_uuid for u, _ in server.signaling_handshakes))
    assert all(not ok for u, ok in server.signaling_handshakes if u == meeting_uuid)

    # The session gives up (no reconnect on auth failure) and streams nothing.
    session = manager._sessions.get(meeting_uuid)
    if session is not None:
        await _wait_for(lambda: session.stopping)
    server.inject_transcript(meeting_uuid, "田中", "この発話は届かない")
    await asyncio.sleep(0.3)
    assert _read_inbox(harness.shared_dir, "kotoha") == []
    # Media plane never reached the streaming handshake.
    assert server.media_connect_count.get(meeting_uuid, 0) == 0
