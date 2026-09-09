# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from cli.tui.app import AnimaChatApp
from cli.tui.widgets.sidebar import Sidebar


class FakeClient:
    """A stub AnimaWorksClient with Phase 2 methods."""

    def __init__(
        self,
        *,
        animas=None,
        history=None,
        skills=None,
        active_refs=None,
    ):
        self.animas = animas or [
            {"name": "sora", "status": "running", "busy": None},
            {"name": "rin", "status": "running", "busy": None},
            {"name": "mei", "status": "running", "busy": None},
        ]
        self.history = history or {"sessions": []}
        self.skills = skills or []
        self.active_refs = active_refs or []
        self._ws_queue: asyncio.Queue = asyncio.Queue()
        self.history_calls: list[str] = []
        self.history_threads: list[str] = []
        self.threads = {}
        self.thread_calls: list[str] = []
        self.skill_calls: list[str] = []
        self.active_calls: list[str] = []
        self.set_active_calls: list = []
        self.resolve_calls: list = []
        self.chat_queue: asyncio.Queue = asyncio.Queue()
        self.messages: list = []
        self.post_calls: list = []
        self.board_calls: list = []
        self.tasks_calls: list = []
        self.compact_calls: list = []
        self.active_stream = {"active": False}

    def push_ws(self, event_type: str, data: dict) -> None:
        self._ws_queue.put_nowait({"type": event_type, "data": data})

    async def list_animas(self):
        return self.animas

    async def get_history(self, anima, *, thread_id="default", limit=50, before=None):
        self.history_calls.append(anima)
        self.history_threads.append(thread_id)
        return self.history

    async def list_threads(self, anima):
        self.thread_calls.append(anima)
        return self.threads

    async def get_active_stream(self, anima, *, thread_id="default"):
        return self.active_stream

    async def interrupt(self, anima, *, thread_id):
        return {"status": "interrupted"}

    async def chat_stream(self, anima, message, *, thread_id="default", resume=None, last_event_id=None, model=None):
        self.messages.append((anima, message))
        while True:
            ev = await self.chat_queue.get()
            if ev is None:
                return
            yield ev

    async def ws_events(self):
        while True:
            ev = await self._ws_queue.get()
            if ev is None:
                return
            yield ev

    async def list_skills(self, anima, thread_id="default"):
        self.skill_calls.append(anima)
        return {"anima": anima, "thread_id": thread_id, "skills": self.skills}

    async def get_active_skills(self, anima, thread_id="default"):
        self.active_calls.append(anima)
        return {
            "anima": anima,
            "thread_id": thread_id,
            "accepted": [{"ref": r, "name": r, "active": True} for r in self.active_refs],
            "rejections": [],
            "warnings": [],
        }

    async def set_active_skills(self, anima, thread_id="default", refs=None, confirm_risk=False):
        self.set_active_calls.append((anima, thread_id, list(refs or []), confirm_risk))
        return {
            "accepted": [{"ref": r, "name": r, "active": True} for r in (refs or [])],
            "rejections": [],
            "warnings": [],
        }

    async def list_channels(self):
        return [{"name": "dev", "message_count": 2}]

    async def read_channel(self, name, limit=50):
        self.board_calls.append((name, limit))
        return {"channel": name, "messages": [{"from": "mei", "ts": "2026-01-01T10:00:00", "text": "hello"}]}

    async def post_channel(self, name, text):
        self.post_calls.append((name, text))
        return {"status": "ok", "channel": name}

    async def list_tasks(self, assignee=None):
        self.tasks_calls.append(assignee)
        return {"tasks": [{"title": "a task", "status": "todo"}]}

    async def resolve_interaction(self, anima, callback_id, decision, comment=""):
        self.resolve_calls.append((anima, callback_id, decision))
        return {"status": "ok", "decision": decision}

    async def compact_session(self, anima, thread_id="default"):
        self.compact_calls.append((anima, thread_id))
        return {"status": "ok", "thread_id": thread_id, "mode": "s"}


async def _pump(n=80):
    for _ in range(n):
        await asyncio.sleep(0)


def _app(client, anima="sora"):
    return AnimaChatApp(client=client, anima_name=anima)


# ── (a) sidebar shows all animas, ws status changes badge ──
@pytest.mark.asyncio
async def test_sidebar_lists_all_animas_and_ws_status_changes_badge():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        assert set(app.state.animas) >= {"sora", "rin", "mei"}
        client.push_ws("anima.status", {"name": "rin", "status": "thinking"})
        await _pump()
        assert app.state.animas["rin"].status == "thinking"


# ── (b)(c) palette opens with skills and filters ──
@pytest.mark.asyncio
async def test_palette_opens_and_lists_skills_then_filters():
    skills = [
        {
            "ref": "pr",
            "name": "pr-review",
            "description": "review a PR",
            "active": False,
            "is_common": False,
            "is_procedure": False,
        },
    ]
    client = FakeClient(skills=skills)
    app = _app(client)
    async with app.run_test() as pilot:
        await _pump()
        app.input_container.focus_input()
        await pilot.press("/")
        await _pump()
        assert app.palette.is_open
        items = app._palette_items()
        values = [it.value for it in items]
        assert "/pr-review " in values
        # Skills are the main content: the skill is the first candidate and
        # built-in commands come after it.
        assert items and items[0].value == "/pr-review "
        first_builtin = next(i for i, it in enumerate(items) if it.value in ("/help", "/animas"))
        assert first_builtin > 0

        # Narrow to the skill — candidate count must drop and the skill remain.
        await pilot.press("p", "r")
        await _pump()
        assert app.palette.is_open
        items = app._palette_items()
        filtered = [it for it in items if it.value.startswith("/pr-review")]
        assert filtered
        # every remaining item matches the query
        for it in app.palette._items:
            assert it.matches("pr")
        # clean up the ws worker
        client.push_ws("anima.status", {"name": "rin", "status": "idle"})


@pytest.mark.asyncio
async def test_palette_labels_are_single_line():
    skills = [
        {
            "ref": "pr",
            "name": "pr-review",
            "description": "review a PR",
            "active": False,
            "is_common": False,
            "is_procedure": False,
        },
    ]
    client = FakeClient(skills=skills)
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        for item in app._palette_items():
            assert item.label.no_wrap is True


# ── (d) /skill foo sets active skills with existing refs + foo ──
@pytest.mark.asyncio
async def test_skill_command_replaces_with_existing_plus_new():
    skills = [
        {
            "ref": "foo-ref",
            "name": "foo",
            "description": "d",
            "active": False,
            "is_common": False,
            "is_procedure": False,
        },
    ]
    client = FakeClient(skills=skills, active_refs=["existing-ref"])
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/skill foo")
        await _pump()
        assert client.set_active_calls, "set_active_skills was not called"
        anima, thread_id, refs, confirm = client.set_active_calls[-1]
        assert anima == "sora"
        assert set(refs) == {"existing-ref", "foo-ref"}
        assert confirm is False


# ── (e) /anima rin switches, history fetched for rin ──
@pytest.mark.asyncio
async def test_switch_anima_loads_rin_history():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        assert "sora" in client.history_calls
        await app._handle_message("/anima rin")
        await _pump()
        assert app.anima_name == "rin"
        assert app.state.current == "rin"
        assert client.history_calls[-1] == "rin"


# ── (f) switching is rejected while busy ──
@pytest.mark.asyncio
async def test_switch_rejected_while_busy():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        app.busy = True
        await app._handle_message("/anima rin")
        await _pump()
        assert app.anima_name == "sora"
        assert app.state.current == "sora"
        assert app.busy is True


# ── (g) proactive: unread for other anima, transcript for current ──
@pytest.mark.asyncio
async def test_proactive_other_anima_increments_unread():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        client.push_ws(
            "anima.proactive_message",
            {"anima": "rin", "subject": "subj", "body": "need you"},
        )
        await _pump()
        assert app.state.animas["rin"].unread == 1
        assert app.state.animas["sora"].unread == 0


@pytest.mark.asyncio
async def test_proactive_current_anima_renders_in_transcript():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        client.push_ws(
            "anima.proactive_message",
            {"anima": "sora", "subject": "subj", "body": "hello there"},
        )
        await _pump()
        from cli.tui.widgets import AssistantBlock

        blocks = list(app.query(AssistantBlock))
        assert blocks, "no assistant block rendered"
        assert any("hello there" in b._body for b in blocks)


# ── (h) activity feed rows are single-line and stay inside the sidebar ──
@pytest.mark.asyncio
async def test_activity_feed_rows_single_line_and_inside_sidebar():
    from cli.tui.widgets.sidebar import _FeedLine

    client = FakeClient()
    app = _app(client)
    async with app.run_test(size=(120, 40)) as _:
        await _pump()
        long = "x" * 300
        for _ in range(10):
            client.push_ws("anima.heartbeat", {"name": "rin", "result": {"summary": long}})
        await _pump()
        await asyncio.sleep(0.2)  # let the incremental mounts be laid out
        await _pump()
        feed = app.query_one("#sidebar", Sidebar).feed
        lines = list(feed.query(_FeedLine))
        assert lines, "no feed lines rendered"
        for line in lines:
            assert line.region.height == 1, f"feed row wrapped ({line.region.height} lines)"
            assert app.sidebar.region.contains_region(line.region)


# ── (h2) activity feed updates incrementally (no full rebuild per event) ──
@pytest.mark.asyncio
async def test_activity_feed_updates_incrementally():
    from cli.tui.widgets.sidebar import _FeedLine

    client = FakeClient()
    app = _app(client)
    async with app.run_test(size=(120, 40)) as _:
        await _pump()
        feed = app.query_one("#sidebar", Sidebar).feed
        for i in range(3):
            client.push_ws("anima.heartbeat", {"name": "rin", "result": {"summary": f"hb{i}"}})
        await _pump()
        await _pump()
        first = list(feed.query(_FeedLine))
        assert len(first) == 3
        client.push_ws("anima.heartbeat", {"name": "rin", "result": {"summary": "hb3"}})
        await _pump()
        await _pump()
        after = list(feed.query(_FeedLine))
        assert len(after) == 4
        # Existing rows are reused, not rebuilt.
        assert [id(w) for w in after[:3]] == [id(w) for w in first]
        assert "hb3" in str(after[-1].content)
        for i in range(70):
            client.push_ws("anima.heartbeat", {"name": "rin", "result": {"summary": f"x{i}"}})
        await _pump()
        await asyncio.sleep(0.2)
        await _pump()
        rows = list(feed.query(_FeedLine))
        assert len(rows) <= feed.MAX_ROWS
        assert "x69" in str(rows[-1].content)


# ── (i) current chat partner is highlighted in the sidebar ──
@pytest.mark.asyncio
async def test_current_anima_row_is_highlighted():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        animas = app.sidebar.animas
        assert animas._rows["sora"].has_class("current")
        await app._handle_message("/anima rin")
        await _pump()
        assert animas._rows["rin"].has_class("current")
        assert not animas._rows["sora"].has_class("current")


# ── (j) one blank line (max) between transcript turns ──
@pytest.mark.asyncio
async def test_transcript_turn_spacing_is_at_most_one_line():
    client = FakeClient()
    app = _app(client)
    async with app.run_test(size=(120, 40)) as pilot:
        await _pump()
        await app.transcript.add_human("You", "hello")
        block = app.transcript.new_assistant("sora")
        block.set_final("hi there")
        await app.transcript.mount_assistant(block)
        await app.transcript.add_human("You", "second")
        await pilot.pause()
        await pilot.pause()
        blocks = list(app.transcript.children)
        for prev, nxt in zip(blocks, blocks[1:], strict=False):
            gap = nxt.region.y - (prev.region.y + prev.region.height)
            assert gap <= 1, f"too much blank space between turns: {gap} rows"


# ── scrolling the transcript drives the scrollbar and repaints ──
@pytest.mark.asyncio
async def test_transcript_scroll_moves_scrollbar_and_refreshes():
    """`Transcript.watch_scroll_y` must not swallow the base behaviour.

    Overriding it without calling ``super()`` left the scrollbar thumb
    frozen at the top (so it could not be dragged) and skipped the
    repaint, which looked exactly like a hung UI.
    """
    client = FakeClient()
    app = _app(client)
    async with app.run_test(size=(120, 40)) as pilot:
        await _pump()
        for index in range(40):
            await app.transcript.add_human("You", f"line {index}")
        await pilot.pause()
        assert app.transcript.max_scroll_y > 0
        scrollbar = app.transcript.vertical_scrollbar
        assert scrollbar.region.width == 1
        assert scrollbar.region.right == app.transcript.region.right
        assert app.transcript.scrollable_content_region.right == scrollbar.region.x
        app.transcript.scroll_to(y=0, animate=False, force=True)
        await pilot.pause()
        assert app.transcript.vertical_scrollbar.position == 0
        target = app.transcript.max_scroll_y
        app.transcript.scroll_to(y=target, animate=False, force=True)
        await pilot.pause()
        assert app.transcript.vertical_scrollbar.position == pytest.approx(target)


# ── (k) palette Enter runs an exact command; partials only complete ──
@pytest.mark.asyncio
async def test_palette_enter_runs_exact_command():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as pilot:
        await _pump()
        app.input_container.focus_input()
        await pilot.press(*"/tasks")
        await _pump()
        assert app.palette.is_open
        await pilot.press("enter")
        await _pump()
        assert client.tasks_calls, "list_tasks was not called"


@pytest.mark.asyncio
async def test_palette_enter_partial_completes_without_running():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as pilot:
        await _pump()
        app.input_container.focus_input()
        await pilot.press("/", "t", "a")
        await _pump()
        assert app.palette.is_open
        await pilot.press("enter")
        await _pump()
        assert app.input_container.input.text == "/tasks "
        assert client.tasks_calls == []


@pytest.mark.asyncio
async def test_palette_enter_skill_completes_first_candidate():
    skills = [
        {
            "ref": "pr",
            "name": "pr-review",
            "description": "review a PR",
            "active": False,
            "is_common": False,
            "is_procedure": False,
        },
    ]
    client = FakeClient(skills=skills)
    app = _app(client)
    async with app.run_test() as pilot:
        await _pump()
        app.input_container.focus_input()
        # `/pr` uniquely narrows to the skill candidate (bare-name token).
        await pilot.press("/", "p", "r")
        await _pump()
        assert app.palette.is_open
        # the skill candidate is the top (prefix-ranked) result
        assert app.palette._items and app.palette._items[0].value == "/pr-review "
        await pilot.press("tab")
        await _pump()
        assert app.input_container.input.text == "/pr-review "


# ── (l) switching to an anima with empty history shows a hint ──
@pytest.mark.asyncio
async def test_switch_to_empty_history_shows_hint():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/anima rin")
        await _pump()
        await _pump()
        transients = [t for t in app.query("Static.transient") if "no conversation history yet" in str(t.content)]
        assert transients, "expected a 'no conversation history yet' hint"


# ── (m) direct skill invocation via /<skill> ──


def _skill(
    ref,
    name,
    description="desc",
    active=False,
    is_common=False,
    is_procedure=False,
):
    return {
        "ref": ref,
        "name": name,
        "description": description,
        "active": active,
        "is_common": is_common,
        "is_procedure": is_procedure,
    }


@pytest.mark.asyncio
async def test_skill_direct_invoke_activates_without_message():
    skills = [_skill("pr", "pr-review")]
    client = FakeClient(skills=skills)
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/pr-review")
        await _pump()
        await _pump()
        assert client.set_active_calls, "set_active_skills was not called"
        anima, thread_id, refs, confirm = client.set_active_calls[-1]
        assert anima == "sora"
        assert refs == ["pr"]
        assert confirm is False
        assert client.messages == [], "no message should be sent on bare activation"


@pytest.mark.asyncio
async def test_skill_direct_invoke_with_message_sends_after_activation():
    skills = [_skill("pr", "pr-review")]
    client = FakeClient(skills=skills)
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/pr-review レビューして")
        await _pump()
        await _pump()
        assert client.set_active_calls, "set_active_skills was not called"
        anima, thread_id, refs, confirm = client.set_active_calls[-1]
        assert refs == ["pr"]
        assert client.messages == [("sora", "レビューして")]


@pytest.mark.asyncio
async def test_skill_direct_invoke_off_deactivates():
    skills = [_skill("pr", "pr-review")]
    client = FakeClient(skills=skills, active_refs=["pr"])
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/pr-review --off")
        await _pump()
        await _pump()
        assert client.set_active_calls, "set_active_skills was not called"
        anima, thread_id, refs, confirm = client.set_active_calls[-1]
        assert refs == []
        assert client.messages == []


@pytest.mark.asyncio
async def test_unknown_skill_shows_unknown_command():
    client = FakeClient()  # no skills
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/nosuchskill")
        await _pump()
        assert client.set_active_calls == [], "set_active_skills should not be called"
        transients = [t for t in app.query("Static.transient") if "Unknown command" in str(t.content)]
        assert transients, "expected an Unknown command message"


# ── (n) /compact ──
@pytest.mark.asyncio
async def test_compact_calls_client():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/compact")
        await _pump()
        await _pump()
        assert client.compact_calls, "compact_session was not called"
        assert client.compact_calls == [("sora", "default")]


@pytest.mark.asyncio
async def test_compact_skipped_when_busy():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        app.busy = True
        await app._handle_message("/compact")
        await _pump()
        assert client.compact_calls == [], "compact_session should not be called while busy"


# ── (o) thread switching: /clear, /thread, /threads ──


def _threads_data(active=None, threads=None):
    return {
        "anima": "sora",
        "active_conversation": active,
        "threads": threads or [],
        "archived_sessions": [],
        "episodes": [],
        "transcripts": [],
    }


@pytest.mark.asyncio
async def test_clear_starts_new_thread():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        first_thread = app.thread_id
        await app._handle_message("/clear")
        await _pump()
        await _pump()
        assert app.thread_id != first_thread
        # new thread id is 8 hex chars
        assert len(app.thread_id) == 8
        assert all(c in "0123456789abcdef" for c in app.thread_id)
        # history was fetched for the new thread
        assert client.history_threads[-1] == app.thread_id
        # skills reloaded for the new thread
        assert client.skill_calls[-1] == "sora"
        # session tracked the switch
        assert app.session.thread_id == app.thread_id
        transients = [t for t in app.query("Static.transient") if "new thread" in str(t.content)]
        assert transients, "expected a 'new thread' transient"


@pytest.mark.asyncio
async def test_thread_without_args_matches_clear():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/thread")
        await _pump()
        await _pump()
        assert len(app.thread_id) == 8
        assert client.history_threads[-1] == app.thread_id


@pytest.mark.asyncio
async def test_thread_with_id_switches():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/thread abc123")
        await _pump()
        await _pump()
        assert app.thread_id == "abc123"
        assert app.session.thread_id == "abc123"
        assert client.history_threads[-1] == "abc123"


@pytest.mark.asyncio
async def test_thread_same_id_is_noop():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        calls_before = len(client.history_threads)
        await app._handle_message("/thread default")
        await _pump()
        transients = [t for t in app.query("Static.transient") if "Already on thread" in str(t.content)]
        assert transients, "expected an 'Already on thread' transient"
        assert app.thread_id == "default"
        assert len(client.history_threads) == calls_before


@pytest.mark.asyncio
async def test_thread_invalid_id_rejected():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/thread x/y")
        await _pump()
        transients = [t for t in app.query("Static.transient") if "Invalid thread id" in str(t.content)]
        assert transients, "expected an 'Invalid thread id' transient"
        assert app.thread_id == "default"


@pytest.mark.asyncio
async def test_thread_rejected_while_busy():
    client = FakeClient()
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        app.busy = True
        await app._handle_message("/thread xyz")
        await _pump()
        assert app.thread_id == "default"


@pytest.mark.asyncio
async def test_threads_lists_default_and_threads():
    client = FakeClient()
    client.threads = _threads_data(
        active={"exists": True, "turn_count": 12, "total_turn_count": 40, "last_timestamp": "2026-09-07T12:00:00"},
        threads=[
            {"thread_id": "1a2b3c4d", "turn_count": 3, "total_turn_count": 3, "last_timestamp": "2026-09-07T11:30:00"},
            {"thread_id": "9f8e7d6c", "turn_count": 0, "total_turn_count": 0},
        ],
    )
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/threads")
        await _pump()
        await _pump()
        assert client.thread_calls == ["sora"]
        text = "\n".join(str(t.content) for t in app.query("Static.transient"))
        assert "Threads for sora" in text
        assert "default" in text
        assert "40 turns" in text
        assert "1a2b3c4d" in text
        assert "9f8e7d6c" in text
        # current thread (default) is marked with *
        assert "* default" in text


@pytest.mark.asyncio
async def test_threads_shows_default_when_active_conversation_null():
    client = FakeClient()
    client.threads = _threads_data(
        active=None,
        threads=[{"thread_id": "1a2b3c4d", "turn_count": 3, "total_turn_count": 3}],
    )
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/threads")
        await _pump()
        await _pump()
        text = "\n".join(str(t.content) for t in app.query("Static.transient"))
        assert "default" in text
        assert "1a2b3c4d" in text


@pytest.mark.asyncio
async def test_threads_marks_current_thread_not_on_server():
    client = FakeClient()
    client.threads = _threads_data(active=None, threads=[])
    app = _app(client)
    async with app.run_test() as _:
        await _pump()
        # switch to a brand-new id first, then list
        await app._handle_message("/thread abcdef01")
        await _pump()
        await _pump()
        await app._handle_message("/threads")
        await _pump()
        await _pump()
        text = "\n".join(str(t.content) for t in app.query("Static.transient"))
        assert "abcdef01" in text
        assert "* abcdef01" in text
