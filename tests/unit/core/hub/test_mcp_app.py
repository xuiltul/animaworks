from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Mount

from core.hub import timeline
from core.hub.mcp_app import build_hub_mcp_app, run_hub_command
from core.hub.members import HubMember, last_seen, save_member, token_digest
from core.messaging.board import clear_post_hooks
from core.messaging.messenger import ChannelMeta, save_channel_meta
from core.time_utils import now_iso


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    shared = tmp_path / "shared"
    (shared / "channels").mkdir(parents=True)
    return shared


@pytest.fixture(autouse=True)
def _clear_hooks():
    clear_post_hooks()
    yield
    clear_post_hooks()


@pytest.fixture
def member() -> HubMember:
    return HubMember(
        id="grok",
        display_name="Grok",
        company="a",
        token_sha256=token_digest("token"),
        message_route={"type": "poll"},
    )


def _hub_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    timeline_enabled: bool = True,
    board: str = "hub",
    max_posts_per_hour: int = 20,
) -> None:
    config = SimpleNamespace(
        hub=SimpleNamespace(
            enabled=enabled,
            timeline_enabled=timeline_enabled,
            company="a",
            board=board,
            public_hosts=[],
            max_posts_per_hour=max_posts_per_hour,
            max_round_trips=4,
        )
    )
    monkeypatch.setattr("core.hub.mcp_app.load_config", lambda: config)
    monkeypatch.setattr("core.config.models.load_config", lambda: config)


def _make_board(shared_dir: Path, name: str = "hub", *, company: str | None = "a") -> None:
    (shared_dir / "channels" / f"{name}.jsonl").touch()
    save_channel_meta(shared_dir, name, ChannelMeta(members=[], company=company))


def _append_board(shared_dir: Path, records: list[dict[str, object]], name: str = "hub") -> None:
    path = shared_dir / "channels" / f"{name}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_help_and_argument_errors(shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_config(monkeypatch)
    help_text = run_hub_command(shared_dir, member, "help")
    assert "commands:" in help_text
    assert "timeline" in help_text and "read" in help_text and "post" in help_text

    assert run_hub_command(shared_dir, member, "").startswith("error: missing command")
    assert run_hub_command(shared_dir, member, "nope").startswith("error: unknown command")
    assert run_hub_command(shared_dir, member, "timeline", "body").startswith("error: text is only accepted")
    assert run_hub_command(shared_dir, member, "help", "body").startswith("error: text is only accepted")
    assert run_hub_command(shared_dir, member, "post").startswith("error: post requires text")
    assert run_hub_command(shared_dir, member, "post unexpected", "body").startswith("error: unexpected argument")


def test_timeline_marks_new_and_human_read(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    item = timeline.append_item(
        shared_dir,
        from_name="alice",
        company="a",
        kind="done",
        text="finished the report",
        source="anima",
    )

    first = run_hub_command(shared_dir, member, "timeline")
    assert "[新]" in first
    assert item["id"] in first
    assert "human_read=no" in first

    second = run_hub_command(shared_dir, member, "timeline")
    assert "[新]" not in second

    timeline.advance_timeline_read(shared_dir, "human:operator", str(item["id"]))
    third = run_hub_command(shared_dir, member, "timeline")
    assert "human_read=yes" in third


def test_timeline_disabled_and_empty(shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_config(monkeypatch, timeline_enabled=False)
    assert "無効" in run_hub_command(shared_dir, member, "timeline")

    _hub_config(monkeypatch)
    assert "項目はありません" in run_hub_command(shared_dir, member, "timeline")


def test_read_lists_new_posts_mentions_first_and_advances(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    _make_board(shared_dir)
    save_member(shared_dir, member)
    _append_board(
        shared_dir,
        [
            {"id": "p1", "ts": "2026-01-01T00:00:01+00:00", "from": "bob", "text": "hello @grok", "kind": "anima"},
            {"id": "p2", "ts": "2026-01-01T00:00:02+00:00", "from": "carol", "text": "general news", "kind": "anima"},
            {"id": "p3", "ts": "2026-01-01T00:00:03+00:00", "from": "grok", "text": "my own", "kind": "external_agent"},
        ],
    )

    output = run_hub_command(shared_dir, member, "read")
    assert output.splitlines()[0].startswith("[to you] p1")
    assert "p2" in output
    assert "p3" not in output

    assert "新着なし" in run_hub_command(shared_dir, member, "read")


def test_post_writes_to_hub_and_supports_idempotency(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    _make_board(shared_dir)
    save_member(shared_dir, member)

    first = run_hub_command(shared_dir, member, "post --request-id req-1", "hello hub")
    assert first.startswith("posted ")
    post_id = first.split()[1]

    duplicate = run_hub_command(shared_dir, member, "post --request-id req-1", "hello hub")
    assert duplicate == f"duplicate post: {post_id}"

    record = json.loads((shared_dir / "channels" / "hub.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["source"] == "hub"
    assert record["kind"] == "external_agent"
    assert record["request_id"] == "req-1"


def test_post_validates_reply_target_and_rate_limit(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch, max_posts_per_hour=1)
    _make_board(shared_dir)
    save_member(shared_dir, member)
    _append_board(
        shared_dir,
        [{"id": "p1", "ts": now_iso(), "from": "bob", "text": "parent", "kind": "anima"}],
    )

    assert "reply target not found" in run_hub_command(shared_dir, member, "post --reply-to missing", "x")
    first = run_hub_command(shared_dir, member, "post --reply-to p1", "reply")
    assert first.startswith("posted ")
    second = run_hub_command(shared_dir, member, "post", "one more")
    assert "post limit reached" in second


async def test_mcp_endpoint_authenticates_and_runs_tool(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    save_member(shared_dir, member)
    _make_board(shared_dir)
    _append_board(
        shared_dir,
        [{"id": "b1", "ts": now_iso(), "from": "bob", "text": "board message", "kind": "anima"}],
    )
    timeline.append_item(
        shared_dir,
        from_name="alice",
        company="a",
        kind="done",
        text="timeline item",
        source="anima",
    )

    hub_app, session_manager = build_hub_mcp_app(lambda: shared_dir)
    root = Starlette(routes=[Mount("/hub", app=hub_app)])
    url = "http://127.0.0.1:8000/hub/mcp"

    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=root),
            base_url="http://127.0.0.1:8000",
            headers=headers,
            timeout=timeout or httpx.Timeout(10.0),
            auth=auth,
        )

    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with session_manager.run():
        async with streamablehttp_client(
            url,
            headers={"Authorization": "Bearer token"},
            httpx_client_factory=factory,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert [tool.name for tool in tools.tools] == ["hub"]

                help_result = await session.call_tool("hub", {"command": "help"})
                assert help_result.isError is False
                assert "commands:" in help_result.content[0].text
                assert "日本語:" in help_result.content[0].text

                timeline_result = await session.call_tool("hub", {"command": "timeline"})
                assert "timeline item" in timeline_result.content[0].text

                read_result = await session.call_tool("hub", {"command": "read"})
                assert "board message" in read_result.content[0].text

                post_result = await session.call_tool(
                    "hub", {"command": "post", "text": "hello from an external agent"}
                )
                assert post_result.content[0].text.startswith("posted ")

    assert last_seen(shared_dir, "grok") is not None


async def test_mcp_endpoint_returns_404_and_401(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    def request(app, headers: dict[str, str] | None = None):
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/mcp",
            "raw_path": b"/mcp",
            "root_path": "",
            "query_string": b"",
            "headers": [
                (b"host", b"127.0.0.1:8000"),
                (b"content-type", b"application/json"),
                *[(name.encode(), value.encode()) for name, value in (headers or {}).items()],
            ],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8000),
        }
        messages: list[dict] = []

        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message):
            messages.append(message)

        return scope, receive, send, messages

    _hub_config(monkeypatch, enabled=False)
    hub_app, _session_manager = build_hub_mcp_app(lambda: shared_dir)
    scope, receive, send, messages = request(hub_app)
    await hub_app(scope, receive, send)
    assert messages[0]["status"] == 404

    _hub_config(monkeypatch)
    save_member(shared_dir, member)
    hub_app, session_manager = build_hub_mcp_app(lambda: shared_dir)
    scope, receive, send, messages = request(hub_app)
    await hub_app(scope, receive, send)
    assert messages[0]["status"] == 401

    scope, receive, send, messages = request(hub_app, {"Authorization": "Bearer wrong"})
    await hub_app(scope, receive, send)
    assert messages[0]["status"] == 401

    scope, receive, send, messages = request(hub_app, {"authorization": "bearer token"})
    async with session_manager.run():
        await hub_app(scope, receive, send)
    assert messages[0]["status"] != 401
    assert last_seen(shared_dir, "grok") is not None


def test_other_company_member_cannot_read_timeline_or_hub(shared_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A member whose company differs from hub.company gets no data back."""
    _hub_config(monkeypatch)
    _make_board(shared_dir)
    outsider = HubMember(id="outsider", display_name="Outsider", company="b", token_sha256=token_digest("x"))
    save_member(shared_dir, outsider)
    for command in ("timeline", "read"):
        assert run_hub_command(shared_dir, outsider, command).startswith("error: this hub belongs to another company")
