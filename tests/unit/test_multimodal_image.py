# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multimodal image input feature.

Tests cover:
1. ImageAttachment model validation
2. Image validation (_validate_images)
3. save_images function
4. build_content_blocks function
5. ConversationTurn attachments
6. ConversationMemory.append_turn with attachments
7. BaseExecutor images parameter
8. ChatRequest model with images
"""

from __future__ import annotations

import base64
import inspect
import json
from pathlib import Path

import pytest

from server.routes.chat import (
    SUPPORTED_IMAGE_TYPES,
    ImageAttachment,
    _validate_images,
    build_content_blocks,
    save_images,
    ChatRequest,
)


# ── 1. ImageAttachment model validation ────────────────────────


class TestImageAttachment:
    def test_valid_image(self) -> None:
        img = ImageAttachment(data="dGVzdA==", media_type="image/png")
        assert img.media_type == "image/png"
        assert img.data == "dGVzdA=="

    def test_all_supported_types(self) -> None:
        for media_type in SUPPORTED_IMAGE_TYPES:
            img = ImageAttachment(data="dGVzdA==", media_type=media_type)
            assert img.media_type == media_type


# ── 2. Image validation (_validate_images) ─────────────────────


class TestImageValidation:
    def test_empty_list_valid(self) -> None:
        assert _validate_images([]) is None

    def test_valid_images_pass(self) -> None:
        imgs = [ImageAttachment(data="dGVzdA==", media_type="image/png")]
        assert _validate_images(imgs) is None

    def test_unsupported_type_rejected(self) -> None:
        imgs = [ImageAttachment(data="dGVzdA==", media_type="image/bmp")]
        error = _validate_images(imgs)
        assert error is not None
        assert "未対応" in error

    def test_oversized_payload_rejected(self) -> None:
        # Create a large base64 string (~21MB)
        large_data = "A" * (21 * 1024 * 1024)
        imgs = [ImageAttachment(data=large_data, media_type="image/png")]
        error = _validate_images(imgs)
        assert error is not None
        assert "大きすぎ" in error


# ── 3. save_images function ────────────────────────────────────


class TestSaveImages:
    def test_save_creates_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_images writes decoded base64 to attachments/ dir."""
        # Prepare the anima directory under the fake data dir
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)

        test_data = base64.b64encode(b"fake image content").decode()
        imgs = [ImageAttachment(data=test_data, media_type="image/png")]

        # save_images does a lazy `from core.paths import get_data_dir`.
        # We patch get_data_dir() to return tmp_path.
        import core.paths as paths_module

        monkeypatch.setattr(paths_module, "get_data_dir", lambda: tmp_path)

        paths = save_images("test-anima", imgs)

        assert len(paths) == 1
        assert paths[0].startswith("attachments/")
        assert paths[0].endswith(".png")

        # Verify file was actually written
        full_path = anima_dir / paths[0]
        assert full_path.exists()
        assert full_path.read_bytes() == b"fake image content"

    def test_save_empty_list(self) -> None:
        paths = save_images("test-anima", [])
        assert paths == []


# ── 4. build_content_blocks function ───────────────────────────


class TestBuildContentBlocks:
    def test_no_images_returns_string(self) -> None:
        result = build_content_blocks("hello", [])
        assert result == "hello"

    def test_with_images_returns_blocks(self) -> None:
        imgs = [ImageAttachment(data="dGVzdA==", media_type="image/png")]
        result = build_content_blocks("hello", imgs)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "hello"

    def test_images_only_no_text(self) -> None:
        imgs = [ImageAttachment(data="dGVzdA==", media_type="image/jpeg")]
        result = build_content_blocks("", imgs)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image"

    def test_multiple_images(self) -> None:
        imgs = [
            ImageAttachment(data="img1", media_type="image/png"),
            ImageAttachment(data="img2", media_type="image/jpeg"),
        ]
        result = build_content_blocks("caption", imgs)
        assert len(result) == 3
        assert result[0]["type"] == "image"
        assert result[1]["type"] == "image"
        assert result[2]["type"] == "text"


# ── 5. ConversationTurn attachments ────────────────────────────


class TestConversationTurnAttachments:
    def test_default_empty_attachments(self) -> None:
        from core.memory.conversation import ConversationTurn

        turn = ConversationTurn(role="human", content="test")
        assert turn.attachments == []

    def test_attachments_stored(self) -> None:
        from core.memory.conversation import ConversationTurn

        turn = ConversationTurn(
            role="human",
            content="test",
            attachments=["attachments/img.png"],
        )
        assert turn.attachments == ["attachments/img.png"]


# ── 6. ConversationMemory.append_turn with attachments ─────────


class TestConversationMemoryAttachments:
    def test_append_turn_with_attachments(self, tmp_path: Path) -> None:
        """append_turn stores attachment paths in the turn."""
        from core.memory.conversation import ConversationMemory
        from core.schemas import ModelConfig

        # Setup minimal anima dir
        anima_dir = tmp_path / "anima"
        state_dir = anima_dir / "state"
        state_dir.mkdir(parents=True)
        transcripts_dir = anima_dir / "transcripts"
        transcripts_dir.mkdir(parents=True)

        config = ModelConfig(model="claude-sonnet-4-20250514")
        conv = ConversationMemory(anima_dir, config)

        conv.append_turn("human", "look at this", attachments=["attachments/img.png"])
        conv.save()

        state = conv.load()
        assert len(state.turns) == 1
        assert state.turns[0].attachments == ["attachments/img.png"]

    def test_append_turn_without_attachments_backward_compat(
        self, tmp_path: Path,
    ) -> None:
        """append_turn without attachments still works (backward compat)."""
        from core.memory.conversation import ConversationMemory
        from core.schemas import ModelConfig

        anima_dir = tmp_path / "anima"
        (anima_dir / "state").mkdir(parents=True)
        (anima_dir / "transcripts").mkdir(parents=True)

        config = ModelConfig(model="claude-sonnet-4-20250514")
        conv = ConversationMemory(anima_dir, config)

        conv.append_turn("human", "hello")
        conv.save()

        state = conv.load()
        assert len(state.turns) == 1
        assert state.turns[0].attachments == []

    def test_conversation_state_includes_attachments(self, tmp_path: Path) -> None:
        """conversation.json includes attachments when present.

        append_turn() no longer writes to transcript JSONL (replaced by
        unified activity log).  Attachments are stored in the in-memory
        conversation state and persisted to conversation.json.
        """
        from core.memory.conversation import ConversationMemory
        from core.schemas import ModelConfig

        anima_dir = tmp_path / "anima"
        (anima_dir / "state").mkdir(parents=True)
        transcripts_dir = anima_dir / "transcripts"
        transcripts_dir.mkdir(parents=True)

        config = ModelConfig(model="claude-sonnet-4-20250514")
        conv = ConversationMemory(anima_dir, config)

        conv.append_turn(
            "human", "check image", attachments=["attachments/photo.jpg"],
        )
        conv.save()

        # Verify attachments in conversation state
        state = conv.load()
        assert len(state.turns) == 1
        assert state.turns[0].attachments == ["attachments/photo.jpg"]

        # Transcript JSONL should NOT be created (replaced by activity log)
        transcript_files = list(transcripts_dir.glob("*.jsonl"))
        assert len(transcript_files) == 0

    def test_conversation_state_omits_attachments_when_empty(
        self, tmp_path: Path,
    ) -> None:
        """conversation.json stores empty attachments list when none provided.

        append_turn() no longer writes to transcript JSONL.
        """
        from core.memory.conversation import ConversationMemory
        from core.schemas import ModelConfig

        anima_dir = tmp_path / "anima"
        (anima_dir / "state").mkdir(parents=True)
        transcripts_dir = anima_dir / "transcripts"
        transcripts_dir.mkdir(parents=True)

        config = ModelConfig(model="claude-sonnet-4-20250514")
        conv = ConversationMemory(anima_dir, config)

        conv.append_turn("human", "hello")
        conv.save()

        # Verify no attachments in conversation state
        state = conv.load()
        assert len(state.turns) == 1
        assert state.turns[0].attachments == []

        # Transcript JSONL should NOT be created
        transcript_files = list(transcripts_dir.glob("*.jsonl"))
        assert len(transcript_files) == 0


# ── 7. BaseExecutor images parameter ──────────────────────────


class TestBaseExecutorSignature:
    def test_execute_accepts_images_param(self) -> None:
        """Verify BaseExecutor.execute signature includes images parameter."""
        from core.execution.base import BaseExecutor

        sig = inspect.signature(BaseExecutor.execute)
        assert "images" in sig.parameters

    def test_execute_streaming_accepts_images_param(self) -> None:
        """Verify BaseExecutor.execute_streaming signature includes images parameter."""
        from core.execution.base import BaseExecutor

        sig = inspect.signature(BaseExecutor.execute_streaming)
        assert "images" in sig.parameters


# ── 8. ChatRequest model with images ──────────────────────────


class TestChatRequest:
    def test_without_images(self) -> None:
        req = ChatRequest(message="hello")
        assert req.images == []

    def test_with_images(self) -> None:
        req = ChatRequest(
            message="look",
            images=[{"data": "dGVzdA==", "media_type": "image/png"}],
        )
        assert len(req.images) == 1
        assert req.images[0].media_type == "image/png"
