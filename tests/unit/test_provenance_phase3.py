from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for provenance Phase 3: origin_chain propagation in Anima-to-Anima messaging."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.execution._sanitize import (
    MAX_ORIGIN_CHAIN_LENGTH,
    ORIGIN_ANIMA,
    ORIGIN_EXTERNAL_PLATFORM,
    ORIGIN_HUMAN,
    ORIGIN_SYSTEM,
)


# ── Messenger.send() origin_chain ─────────────────────────────


class TestMessengerSendOriginChain:
    """Messenger.send() propagates origin_chain to the Message."""

    @pytest.fixture
    def messenger(self, tmp_path: Path) -> Any:
        from core.messenger import Messenger

        shared = tmp_path / "shared"
        shared.mkdir(parents=True)
        return Messenger(shared, "sender-anima")

    def test_send_without_origin_chain_backward_compat(self, messenger: Any) -> None:
        """Calling send() without origin_chain still works (empty list)."""
        msg = messenger.send(
            to="target-anima",
            content="hello",
            skip_logging=True,
        )
        assert msg.origin_chain == []

    def test_send_with_origin_chain_human(self, messenger: Any) -> None:
        msg = messenger.send(
            to="target-anima",
            content="relayed from human",
            skip_logging=True,
            origin_chain=["human", "anima"],
        )
        assert msg.origin_chain == ["human", "anima"]

    def test_send_with_origin_chain_external(self, messenger: Any) -> None:
        msg = messenger.send(
            to="target-anima",
            content="relayed from slack",
            skip_logging=True,
            origin_chain=["external_platform", "anima"],
        )
        assert msg.origin_chain == ["external_platform", "anima"]

    def test_origin_chain_persisted_to_inbox_json(self, messenger: Any) -> None:
        """origin_chain is written to the inbox JSON file."""
        chain = ["external_platform", "anima"]
        msg = messenger.send(
            to="target-anima",
            content="test persistence",
            skip_logging=True,
            origin_chain=chain,
        )
        inbox_file = messenger.shared_dir / "inbox" / "target-anima" / f"{msg.id}.json"
        assert inbox_file.exists()
        data = json.loads(inbox_file.read_text(encoding="utf-8"))
        assert data["origin_chain"] == chain

    def test_send_none_origin_chain_becomes_empty_list(self, messenger: Any) -> None:
        msg = messenger.send(
            to="target-anima",
            content="hi",
            skip_logging=True,
            origin_chain=None,
        )
        assert msg.origin_chain == []

    def test_origin_chain_roundtrip_receive(self, messenger: Any) -> None:
        """Message sent with origin_chain can be received with chain intact."""
        chain = ["external_platform", "anima"]
        messenger.send(
            to="sender-anima",
            content="self-message for test",
            skip_logging=True,
            origin_chain=chain,
        )
        received = messenger.receive()
        assert len(received) == 1
        assert received[0].origin_chain == chain


# ── ToolHandler.set_session_origin() ──────────────────────────


class TestToolHandlerSessionOrigin:
    """ToolHandler session origin management."""

    @pytest.fixture
    def handler(self, tmp_path: Path) -> Any:
        from core.memory import MemoryManager
        from core.tooling.handler import ToolHandler

        anima_dir = tmp_path / "animas" / "test-anima"
        (anima_dir / "activity_log").mkdir(parents=True)
        (anima_dir / "episodes").mkdir(parents=True)
        (anima_dir / "knowledge").mkdir(parents=True)
        (anima_dir / "procedures").mkdir(parents=True)
        (anima_dir / "skills").mkdir(parents=True)
        (anima_dir / "state").mkdir(parents=True)
        memory = MemoryManager(anima_dir)
        return ToolHandler(anima_dir, memory)

    def test_initial_session_origin_empty(self, handler: Any) -> None:
        assert handler._session_origin == ""
        assert handler._session_origin_chain == []

    def test_set_session_origin_human(self, handler: Any) -> None:
        handler.set_session_origin(ORIGIN_HUMAN)
        assert handler._session_origin == ORIGIN_HUMAN
        assert handler._session_origin_chain == []

    def test_set_session_origin_with_chain(self, handler: Any) -> None:
        handler.set_session_origin(
            ORIGIN_EXTERNAL_PLATFORM,
            ["external_platform"],
        )
        assert handler._session_origin == ORIGIN_EXTERNAL_PLATFORM
        assert handler._session_origin_chain == ["external_platform"]

    def test_set_session_origin_none_chain(self, handler: Any) -> None:
        handler.set_session_origin(ORIGIN_SYSTEM, None)
        assert handler._session_origin == ORIGIN_SYSTEM
        assert handler._session_origin_chain == []

    def test_set_session_origin_replaces_previous(self, handler: Any) -> None:
        handler.set_session_origin(ORIGIN_HUMAN)
        handler.set_session_origin(ORIGIN_SYSTEM)
        assert handler._session_origin == ORIGIN_SYSTEM
        assert handler._session_origin_chain == []


# ── _handle_send_message origin_chain propagation ─────────────


class TestSendMessageOriginPropagation:
    """_handle_send_message() builds and passes origin_chain."""

    @pytest.fixture
    def setup(self, tmp_path: Path) -> dict[str, Any]:
        from core.memory import MemoryManager
        from core.tooling.handler import ToolHandler

        # Create sender anima dir
        anima_dir = tmp_path / "animas" / "sender"
        for sub in ("activity_log", "episodes", "knowledge", "procedures",
                     "skills", "state", "run"):
            (anima_dir / sub).mkdir(parents=True)

        # Create target anima dir (so it's recognized as internal)
        target_dir = tmp_path / "animas" / "target"
        target_dir.mkdir(parents=True)

        # Create shared dir with inbox
        shared = tmp_path / "shared"
        shared.mkdir(parents=True)

        from core.messenger import Messenger

        messenger = Messenger(shared, "sender")
        memory = MemoryManager(anima_dir)

        handler = ToolHandler(anima_dir, memory, messenger=messenger)
        return {
            "handler": handler,
            "messenger": messenger,
            "anima_dir": anima_dir,
            "target_dir": target_dir,
            "shared": shared,
            "tmp_path": tmp_path,
        }

    @staticmethod
    def _mock_resolve_recipient(name: str, *args: Any, **kwargs: Any) -> Any:
        """Create a mock resolved recipient that is internal."""
        mock = MagicMock()
        mock.is_internal = True
        mock.name = name
        return mock

    def test_chat_session_origin_chain_human_anima(self, setup: dict[str, Any]) -> None:
        """Chat (human) -> send_message -> target gets origin_chain=['human', 'anima']."""
        handler = setup["handler"]
        handler.set_session_origin(ORIGIN_HUMAN)

        with patch("core.outbound.resolve_recipient", side_effect=self._mock_resolve_recipient), \
             patch("core.config.models.load_config"):
            result = handler._handle_send_message({
                "to": "target",
                "content": "hello from human session",
                "intent": "report",
            })

        assert "Message sent" in result
        inbox_file = list((setup["shared"] / "inbox" / "target").glob("*.json"))
        assert len(inbox_file) == 1
        data = json.loads(inbox_file[0].read_text(encoding="utf-8"))
        assert data["origin_chain"] == ["human", "anima"]

    def test_inbox_external_origin_chain_propagation(self, setup: dict[str, Any]) -> None:
        """Inbox from external -> send_message -> target gets chain with external_platform."""
        handler = setup["handler"]
        handler.set_session_origin(
            ORIGIN_EXTERNAL_PLATFORM,
            ["external_platform"],
        )

        with patch("core.outbound.resolve_recipient", side_effect=self._mock_resolve_recipient), \
             patch("core.config.models.load_config"):
            result = handler._handle_send_message({
                "to": "target",
                "content": "relayed from slack",
                "intent": "report",
            })

        assert "Message sent" in result
        inbox_file = list((setup["shared"] / "inbox" / "target").glob("*.json"))
        assert len(inbox_file) == 1
        data = json.loads(inbox_file[0].read_text(encoding="utf-8"))
        assert "external_platform" in data["origin_chain"]
        assert "anima" in data["origin_chain"]

    def test_heartbeat_origin_chain_system_anima(self, setup: dict[str, Any]) -> None:
        """Heartbeat (system) -> send_message -> target gets ['system', 'anima']."""
        handler = setup["handler"]
        handler.set_session_origin(ORIGIN_SYSTEM)

        with patch("core.outbound.resolve_recipient", side_effect=self._mock_resolve_recipient), \
             patch("core.config.models.load_config"):
            result = handler._handle_send_message({
                "to": "target",
                "content": "heartbeat observation",
                "intent": "report",
            })

        assert "Message sent" in result
        inbox_file = list((setup["shared"] / "inbox" / "target").glob("*.json"))
        assert len(inbox_file) == 1
        data = json.loads(inbox_file[0].read_text(encoding="utf-8"))
        assert data["origin_chain"] == ["system", "anima"]

    def test_multi_hop_chain_accumulation(self, setup: dict[str, Any]) -> None:
        """A->B->C: origin_chain grows with each hop."""
        handler = setup["handler"]
        handler.set_session_origin(
            ORIGIN_ANIMA,
            ["external_platform", "anima"],
        )

        with patch("core.outbound.resolve_recipient", side_effect=self._mock_resolve_recipient), \
             patch("core.config.models.load_config"):
            result = handler._handle_send_message({
                "to": "target",
                "content": "multi-hop relay",
                "intent": "report",
            })

        assert "Message sent" in result
        inbox_file = list((setup["shared"] / "inbox" / "target").glob("*.json"))
        assert len(inbox_file) == 1
        data = json.loads(inbox_file[0].read_text(encoding="utf-8"))
        chain = data["origin_chain"]
        assert chain[0] == "external_platform"
        assert "anima" in chain

    def test_origin_chain_max_length_truncation(self, setup: dict[str, Any]) -> None:
        """origin_chain is truncated at MAX_ORIGIN_CHAIN_LENGTH."""
        handler = setup["handler"]
        long_chain = ["external_platform"] + ["anima"] * (MAX_ORIGIN_CHAIN_LENGTH + 5)
        handler.set_session_origin(ORIGIN_ANIMA, long_chain)

        with patch("core.outbound.resolve_recipient", side_effect=self._mock_resolve_recipient), \
             patch("core.config.models.load_config"):
            handler._handle_send_message({
                "to": "target",
                "content": "long chain",
                "intent": "report",
            })

        inbox_file = list((setup["shared"] / "inbox" / "target").glob("*.json"))
        assert len(inbox_file) == 1
        data = json.loads(inbox_file[0].read_text(encoding="utf-8"))
        assert len(data["origin_chain"]) <= MAX_ORIGIN_CHAIN_LENGTH

    def test_empty_session_origin_defaults_to_anima(self, setup: dict[str, Any]) -> None:
        """When session origin is not set, only 'anima' appears in chain."""
        handler = setup["handler"]

        with patch("core.outbound.resolve_recipient", side_effect=self._mock_resolve_recipient), \
             patch("core.config.models.load_config"):
            handler._handle_send_message({
                "to": "target",
                "content": "no origin set",
                "intent": "report",
            })

        inbox_file = list((setup["shared"] / "inbox" / "target").glob("*.json"))
        assert len(inbox_file) == 1
        data = json.loads(inbox_file[0].read_text(encoding="utf-8"))
        assert data["origin_chain"] == ["anima"]


# ── delegate_task origin_chain propagation ────────────────────


class TestDelegateTaskOriginPropagation:
    """_handle_delegate_task() passes origin_chain to the DM."""

    @pytest.fixture
    def setup(self, tmp_path: Path) -> dict[str, Any]:
        from core.memory import MemoryManager
        from core.tooling.handler import ToolHandler

        # Create supervisor anima dir
        anima_dir = tmp_path / "animas" / "supervisor"
        for sub in ("activity_log", "episodes", "knowledge", "procedures",
                     "skills", "state", "run"):
            (anima_dir / sub).mkdir(parents=True)

        # Create subordinate anima dir with task queue
        sub_dir = tmp_path / "animas" / "worker"
        (sub_dir / "state").mkdir(parents=True)
        (sub_dir / "activity_log").mkdir(parents=True)

        shared = tmp_path / "shared"
        shared.mkdir(parents=True)

        from core.messenger import Messenger

        messenger = Messenger(shared, "supervisor")
        memory = MemoryManager(anima_dir)
        handler = ToolHandler(anima_dir, memory, messenger=messenger)

        return {
            "handler": handler,
            "tmp_path": tmp_path,
            "shared": shared,
        }

    def test_delegate_task_propagates_origin_chain(self, setup: dict[str, Any]) -> None:
        handler = setup["handler"]
        handler.set_session_origin(ORIGIN_HUMAN)

        # Mock descendant check to pass
        with patch.object(handler, "_check_subordinate", return_value=None), \
             patch("core.paths.get_animas_dir",
                   return_value=setup["tmp_path"] / "animas"):
            result = handler._handle_delegate_task({
                "name": "worker",
                "instruction": "do the thing",
                "deadline": "2h",
            })

        assert "worker" in result
        # Check inbox for the DM
        inbox_file = list((setup["shared"] / "inbox" / "worker").glob("*.json"))
        assert len(inbox_file) >= 1
        data = json.loads(inbox_file[0].read_text(encoding="utf-8"))
        assert "human" in data["origin_chain"]
        assert "anima" in data["origin_chain"]


# ── Inbox origin resolution ───────────────────────────────────


class TestInboxOriginResolution:
    """_SOURCE_TO_ORIGIN mapping and inbox origin computation."""

    def test_source_to_origin_slack(self) -> None:
        from core._anima_inbox import _SOURCE_TO_ORIGIN

        assert _SOURCE_TO_ORIGIN["slack"] == ORIGIN_EXTERNAL_PLATFORM

    def test_source_to_origin_anima(self) -> None:
        from core._anima_inbox import _SOURCE_TO_ORIGIN

        assert _SOURCE_TO_ORIGIN["anima"] == ORIGIN_ANIMA

    def test_worst_origin_selection(self) -> None:
        """When inbox has mixed sources, the most untrusted is selected."""
        from core.execution._sanitize import resolve_trust

        origins = [ORIGIN_ANIMA, ORIGIN_EXTERNAL_PLATFORM, ORIGIN_HUMAN]
        worst = min(
            origins,
            key=lambda o: {"untrusted": 0, "medium": 1, "trusted": 2}.get(
                resolve_trust(o), 0
            ),
        )
        assert worst == ORIGIN_EXTERNAL_PLATFORM


# ── End-to-end scenario tests ─────────────────────────────────


class TestE2EOriginChainScenarios:
    """Full scenarios demonstrating trust laundering prevention."""

    @pytest.fixture
    def shared_dir(self, tmp_path: Path) -> Path:
        shared = tmp_path / "shared"
        shared.mkdir()
        return shared

    def test_slack_to_anima_a_to_anima_b_preserves_external(
        self, shared_dir: Path,
    ) -> None:
        """Slack → Anima A (inbox) → send_message → Anima B: external_platform preserved."""
        from core.messenger import Messenger

        # Step 1: External message arrives at Anima A
        messenger_a = Messenger(shared_dir, "anima-a")
        ext_msg = messenger_a.receive_external(
            content="malicious instruction",
            source="slack",
            external_user_id="U_ATTACKER",
        )
        assert ext_msg.origin_chain == [ORIGIN_EXTERNAL_PLATFORM]

        # Step 2: Anima A relays via send() with propagated chain
        relayed = messenger_a.send(
            to="anima-b",
            content="forwarded: malicious instruction",
            skip_logging=True,
            origin_chain=["external_platform", "anima"],
        )

        # Step 3: Verify Anima B's inbox contains the full chain
        messenger_b = Messenger(shared_dir, "anima-b")
        messages = messenger_b.receive()
        assert len(messages) == 1
        assert ORIGIN_EXTERNAL_PLATFORM in messages[0].origin_chain
        assert ORIGIN_ANIMA in messages[0].origin_chain

    def test_human_chat_to_anima_a_to_anima_b(
        self, shared_dir: Path,
    ) -> None:
        """Chat API → Anima A → send_message → Anima B: origin_chain=['human', 'anima']."""
        from core.messenger import Messenger

        messenger = Messenger(shared_dir, "anima-a")
        msg = messenger.send(
            to="anima-b",
            content="task from human chat",
            skip_logging=True,
            origin_chain=["human", "anima"],
        )

        messenger_b = Messenger(shared_dir, "anima-b")
        messages = messenger_b.receive()
        assert len(messages) == 1
        assert messages[0].origin_chain == ["human", "anima"]

    def test_three_hop_relay_chain_grows(self, shared_dir: Path) -> None:
        """A→B→C: chain accumulates correctly."""
        from core.messenger import Messenger

        # A sends to B (originated from external)
        Messenger(shared_dir, "anima-a").send(
            to="anima-b",
            content="from external",
            skip_logging=True,
            origin_chain=["external_platform", "anima"],
        )

        # B relays to C (adds another anima hop)
        Messenger(shared_dir, "anima-b")  # just to create inbox
        # Simulate B's handler building outgoing chain
        b_session_chain = ["external_platform", "anima"]
        b_session_origin = ORIGIN_ANIMA
        outgoing = list(b_session_chain)
        if b_session_origin not in outgoing:
            outgoing.append(b_session_origin)
        if ORIGIN_ANIMA not in outgoing:
            outgoing.append(ORIGIN_ANIMA)

        Messenger(shared_dir, "anima-b").send(
            to="anima-c",
            content="relayed again",
            skip_logging=True,
            origin_chain=outgoing,
        )

        # C receives
        messenger_c = Messenger(shared_dir, "anima-c")
        messages = messenger_c.receive()
        assert len(messages) == 1
        chain = messages[0].origin_chain
        assert chain[0] == "external_platform"
        assert "anima" in chain

    def test_origin_chain_truncated_at_max(self, shared_dir: Path) -> None:
        """Chain longer than MAX_ORIGIN_CHAIN_LENGTH is truncated."""
        from core.messenger import Messenger

        long_chain = [f"hop_{i}" for i in range(MAX_ORIGIN_CHAIN_LENGTH + 5)]
        msg = Messenger(shared_dir, "anima-a").send(
            to="anima-b",
            content="long chain test",
            skip_logging=True,
            origin_chain=long_chain,
        )
        # Messenger.send() passes the chain as-is; truncation is at the
        # handler level. But the message stores whatever is given.
        # The handler test above validates truncation at build time.
        assert len(msg.origin_chain) == MAX_ORIGIN_CHAIN_LENGTH + 5

    def test_backward_compat_old_message_without_chain(
        self, shared_dir: Path,
    ) -> None:
        """Pre-Phase 3 inbox JSON without origin_chain loads with empty list."""
        inbox_dir = shared_dir / "inbox" / "anima-x"
        inbox_dir.mkdir(parents=True)
        old_msg = {
            "id": "20260228_120000_000001",
            "from_person": "anima-y",
            "to_person": "anima-x",
            "content": "old format",
            "source": "anima",
        }
        (inbox_dir / "20260228_120000_000001.json").write_text(
            json.dumps(old_msg), encoding="utf-8",
        )

        from core.messenger import Messenger

        messenger = Messenger(shared_dir, "anima-x")
        messages = messenger.receive()
        assert len(messages) == 1
        assert messages[0].origin_chain == []


# ── build_outgoing_origin_chain helper ───────────────────────


class TestBuildOutgoingOriginChain:
    """Tests for the extracted build_outgoing_origin_chain helper."""

    def test_empty_session_appends_anima(self):
        from core.tooling.handler_base import build_outgoing_origin_chain

        chain = build_outgoing_origin_chain("", [])
        assert chain == [ORIGIN_ANIMA]

    def test_human_origin_appends_both(self):
        from core.tooling.handler_base import build_outgoing_origin_chain

        chain = build_outgoing_origin_chain(ORIGIN_HUMAN, [])
        assert chain == [ORIGIN_HUMAN, ORIGIN_ANIMA]

    def test_existing_chain_preserved(self):
        from core.tooling.handler_base import build_outgoing_origin_chain

        chain = build_outgoing_origin_chain(
            ORIGIN_HUMAN, [ORIGIN_EXTERNAL_PLATFORM],
        )
        assert chain == [ORIGIN_EXTERNAL_PLATFORM, ORIGIN_HUMAN, ORIGIN_ANIMA]

    def test_dedup_existing_origin(self):
        from core.tooling.handler_base import build_outgoing_origin_chain

        chain = build_outgoing_origin_chain(
            ORIGIN_ANIMA, [ORIGIN_EXTERNAL_PLATFORM, ORIGIN_ANIMA],
        )
        # ORIGIN_ANIMA already in chain, should not be duplicated
        assert chain == [ORIGIN_EXTERNAL_PLATFORM, ORIGIN_ANIMA]

    def test_truncation_at_max_length(self):
        from core.tooling.handler_base import build_outgoing_origin_chain

        long_chain = [f"hop_{i}" for i in range(MAX_ORIGIN_CHAIN_LENGTH + 5)]
        chain = build_outgoing_origin_chain("extra", long_chain)
        assert len(chain) <= MAX_ORIGIN_CHAIN_LENGTH
