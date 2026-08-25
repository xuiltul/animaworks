# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the voice front lane (PR-2).

Covers:
  - front_model configurede -> VoiceFrontLane is used
  - front_model unset        -> legacy process_message path
  - front health failure     -> automatic fallback to process_message
  - front system prompt is fixed across turns (prefix-cache friendly)
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.schemas import VoiceConfig
from core.prompt.builder import build_voice_front_prompt
from core.voice.front import READ_MEMORY_TOOL, VoiceFrontLane, extract_emotion
from core.voice.session import VoiceSession, read_memory_snippets
from core.voice.tts_base import TTSConfig

# ── Helpers ──────────────────────────────────────────────────────


def _audio_frames() -> bytes:
    """Return a PCM16 buffer with enough energy to pass the voice gate (>350ms)."""
    return struct.pack(f"<{8000}h", *([5000] * 8000))


def _done_response() -> SimpleNamespace:
    return SimpleNamespace(
        done=True,
        chunk=None,
        result={"cycle_result": {"emotion": "neutral"}},
    )


def _legacy_supervisor() -> MagicMock:
    """Supervisor whose send_request_stream completes immediately (legacy path)."""

    async def _stream(**kwargs):  # type: ignore[no-untyped-def]
        yield _done_response()

    supervisor = MagicMock()
    supervisor.send_request_stream = MagicMock(side_effect=_stream)
    return supervisor


def _make_voice_session(
    *,
    front_model: str | None = None,
    front_api_base: str | None = None,
    supervisor: MagicMock | None = None,
) -> VoiceSession:
    stt = AsyncMock()
    stt.transcribe_buffer_async = AsyncMock(return_value={"raw_text": "こんにちは", "language": "ja"})
    tts = AsyncMock()
    tts.health_check = AsyncMock(return_value=True)

    async def _synthesize(text: str, config: object) -> AsyncMock:
        yield b"\x00\x01"

    tts.synthesize = _synthesize
    ws = AsyncMock()
    voice_config = VoiceConfig(stt_refine_enabled=False)
    sess = VoiceSession(
        "test",
        ws,
        stt,
        tts,
        TTSConfig(provider="voicevox"),
        supervisor or _legacy_supervisor(),
        voice_config,
        front_model=front_model,
        front_api_base=front_api_base,
    )
    return sess


def _front_lane_stub(*, healthy: bool = True) -> AsyncMock:
    lane = AsyncMock()
    lane.check_health = AsyncMock(return_value=healthy)
    lane.reset_turn = MagicMock()

    async def _stream(user_text: str, **kwargs):  # type: ignore[no-untyped-def]
        yield "こんにちは！"
        yield ' はい。 <!-- emotion: {"emotion": "smile"} -->'

    lane.stream = _stream
    return lane


# ── Prompt building (prefix-fixed) ─────────────────────────────


class TestBuildVoiceFrontPrompt:
    def test_deterministic_across_turns(self, tmp_path: Path) -> None:
        (tmp_path / "identity.md").write_text("## あなたは元気な助手です", encoding="utf-8")
        first = build_voice_front_prompt(tmp_path, anima_name="taro")
        second = build_voice_front_prompt(tmp_path, anima_name="taro")
        assert first == second  # fixed system prompt → prefix cache friendly

    def test_includes_persona_identity_and_rules(self, tmp_path: Path) -> None:
        (tmp_path / "identity.md").write_text("IDENTITY_TEXT", encoding="utf-8")
        (tmp_path / "specialty_prompt.md").write_text("SPECIALTY_TEXT", encoding="utf-8")
        prompt = build_voice_front_prompt(tmp_path, anima_name="taro")
        assert "IDENTITY_TEXT" in prompt
        assert "SPECIALTY_TEXT" in prompt
        assert "taro" in prompt
        assert "emotion" in prompt
        assert "60 characters" in prompt
        assert "read_memory is read-only" in prompt

    def test_reads_specialty_prompt_file(self, tmp_path: Path) -> None:
        # Regression: the real anima file is ``specialty_prompt.md`` (no "i") —
        # see core/anima_factory.py and core/memory/manager.py.
        (tmp_path / "specialty_prompt.md").write_text("実務の専門性: 図書館管理", encoding="utf-8")
        prompt = build_voice_front_prompt(tmp_path, anima_name="taro")
        assert "実務の専門性: 図書館管理" in prompt
        # the misspelled variant must not be consulted
        (tmp_path / "speciality_prompt.md").write_text("WRONG_FILE", encoding="utf-8")
        assert "WRONG_FILE" not in build_voice_front_prompt(tmp_path, anima_name="taro")

    def test_no_turn_dependent_state(self, tmp_path: Path) -> None:
        prompt = build_voice_front_prompt(tmp_path, anima_name="taro")
        # No dynamic timestamp / current-state style content that would break
        # prefix caching if it changed between requests.
        assert "current_state" not in prompt.lower()
        # The ask_anima tool instructions are static text, not per-turn state.
        assert "ask_anima" in prompt


# ── Emotion extraction ─────────────────────────────────────────


class TestExtractEmotion:
    def test_parses_tag(self) -> None:
        assert extract_emotion('こんにちは。 <!-- emotion: {"emotion": "smile"} -->') == "smile"

    def test_defaults_neutral(self) -> None:
        assert extract_emotion("プレーンテキスト") == "neutral"

    def test_invalid_emotion_defaults_neutral(self) -> None:
        assert extract_emotion('<!-- emotion: {"emotion": "angry"} -->') == "neutral"


# ── VoiceFrontLane behaviour ────────────────────────────────────


class TestVoiceFrontLane:
    @pytest.mark.asyncio
    async def test_stream_appends_history_and_calls_litellm(self) -> None:
        lane = VoiceFrontLane(
            model="openai/qwen3.6-35b-a3b",
            api_base="http://x:8000/v1",
            system_prompt="SYSTEM",
        )
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="こんにちは"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="。"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
        ]

        async def _mc(**kwargs):  # type: ignore[no-untyped-def]
            for c in chunks:
                yield c

        with patch("core.voice.front.litellm.acompletion", side_effect=_mc) as mock_ac:
            got = [d async for d in lane.stream("こんにちは")]
        assert "".join(got) == "こんにちは。"
        mock_ac.assert_awaited_once()
        call = mock_ac.await_args.kwargs
        assert call["model"] == "openai/qwen3.6-35b-a3b"
        assert call["api_base"] == "http://x:8000/v1"
        assert call["stream"] is True
        assert call["messages"][0] == {"role": "system", "content": "SYSTEM"}
        assert call["messages"][-1] == {"role": "user", "content": "こんにちは"}
        # history now contains the user + assistant pair
        assert lane.history == [
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "こんにちは。"},
        ]

    @pytest.mark.asyncio
    async def test_check_health_returns_false_on_error(self) -> None:
        lane = VoiceFrontLane(
            model="openai/q",
            api_base="http://x:8000/v1",
            system_prompt="S",
        )
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=ConnectionError("down"))
            assert await lane.check_health() is False
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=SimpleNamespace(status_code=200)
            )
            assert await lane.check_health() is True

    @pytest.mark.asyncio
    async def test_empty_api_base_is_unhealthy(self) -> None:
        lane = VoiceFrontLane(model="openai/q", api_base="", system_prompt="S")
        assert await lane.check_health() is False

    @pytest.mark.asyncio
    async def test_named_tool_executor_receives_parsed_arguments(self) -> None:
        lane = VoiceFrontLane(model="openai/q", api_base="", system_prompt="S")

        async def _tool_response(**kwargs):  # type: ignore[no-untyped-def]
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="memory_1",
                                    function=SimpleNamespace(
                                        name="read_memory",
                                        arguments='{"query":"散歩"}',
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            )

        async def _final_response(**kwargs):  # type: ignore[no-untyped-def]
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="思い出した", tool_calls=None),
                        finish_reason="stop",
                    )
                ]
            )

        executor = MagicMock(return_value="散歩の記憶")
        with patch(
            "core.voice.front.litellm.acompletion",
            side_effect=[_tool_response(), _final_response()],
        ) as mock_ac:
            got = [
                delta
                async for delta in lane.stream(
                    "思い出して",
                    tools=[READ_MEMORY_TOOL],
                    tool_executors={"read_memory": executor},
                )
            ]

        assert "".join(got) == "思い出した"
        executor.assert_called_once_with({"query": "散歩"})
        tool_message = mock_ac.await_args_list[1].kwargs["messages"][-1]
        assert tool_message == {
            "role": "tool",
            "tool_call_id": "memory_1",
            "content": "散歩の記憶",
        }

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_and_continues(self) -> None:
        lane = VoiceFrontLane(model="openai/q", api_base="", system_prompt="S")

        async def _tool_response(**kwargs):  # type: ignore[no-untyped-def]
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="unknown_1",
                                    function=SimpleNamespace(name="write_memory", arguments="{}"),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            )

        async def _final_response(**kwargs):  # type: ignore[no-untyped-def]
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="続行", tool_calls=None),
                        finish_reason="stop",
                    )
                ]
            )

        with patch(
            "core.voice.front.litellm.acompletion",
            side_effect=[_tool_response(), _final_response()],
        ) as mock_ac:
            got = [delta async for delta in lane.stream("test")]

        assert "".join(got) == "続行"
        assert mock_ac.await_args_list[1].kwargs["messages"][-1]["content"] == ("error: unknown tool write_memory")

    @pytest.mark.asyncio
    async def test_history_is_limited_to_thirty_messages(self) -> None:
        lane = VoiceFrontLane(model="openai/q", api_base="", system_prompt="S")
        lane.set_history([{"role": "user", "content": str(index)} for index in range(30)])

        async def _response(**kwargs):  # type: ignore[no-untyped-def]
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="reply", tool_calls=None),
                        finish_reason="stop",
                    )
                ]
            )

        with patch("core.voice.front.litellm.acompletion", side_effect=_response):
            assert [delta async for delta in lane.stream("new")] == ["reply"]

        assert len(lane.history) == 30
        assert lane.history[0]["content"] == "2"


class TestReadMemorySnippets:
    @staticmethod
    def _create_memory_tree(anima_dir: Path) -> None:
        (anima_dir / "episodes").mkdir(parents=True)
        (anima_dir / "knowledge" / "archive").mkdir(parents=True)
        (anima_dir / "procedures").mkdir(parents=True)
        (anima_dir / "episodes" / "2026-08-24.md").write_text("古い出来事", encoding="utf-8")
        (anima_dir / "episodes" / "2026-08-25.md").write_text(
            "---\ntitle: hidden\n---\n今日の前半。最新エピソードの末尾。",
            encoding="utf-8",
        )
        (anima_dir / "knowledge" / "a.md").write_text("散歩すると発想が整う。", encoding="utf-8")
        (anima_dir / "knowledge" / "archive" / "old.md").write_text("アーカイブの秘密", encoding="utf-8")
        (anima_dir / "procedures" / "p.md").write_text(
            "朝の手順では珈琲を淹れる。珈琲の香りを楽しむ。", encoding="utf-8"
        )

    def test_empty_query_returns_latest_episode_and_active_knowledge(self, tmp_path: Path) -> None:
        self._create_memory_tree(tmp_path)

        result = read_memory_snippets(tmp_path, "")

        assert "最近の出来事 (2026-08-25.md)" in result
        assert "最新エピソードの末尾" in result
        assert "知っていること (a.md)" in result
        assert "散歩すると発想が整う" in result
        assert "2026-08-24.md" not in result
        assert "アーカイブの秘密" not in result
        assert "title: hidden" not in result

    def test_query_returns_matching_snippet_and_no_hit_message(self, tmp_path: Path) -> None:
        self._create_memory_tree(tmp_path)

        result = read_memory_snippets(tmp_path, "珈琲")

        assert "## procedures/p.md" in result
        assert "珈琲の香り" in result
        assert read_memory_snippets(tmp_path, "宇宙船") == ("（「宇宙船」に関する記憶は見つからなかった）")

    def test_result_is_truncated_to_max_chars(self, tmp_path: Path) -> None:
        self._create_memory_tree(tmp_path)

        result = read_memory_snippets(tmp_path, "", max_chars=80)

        assert len(result) <= 80


# ── VoiceSession routing ────────────────────────────────────────


class TestVoiceSessionFrontRouting:
    @pytest.mark.asyncio
    async def test_front_model_unset_uses_legacy_path(self) -> None:
        supervisor = _legacy_supervisor()
        sess = _make_voice_session(supervisor=supervisor)
        sess._audio_buffer.extend(_audio_frames())
        await sess.handle_speech_end()
        supervisor.send_request_stream.assert_called()
        assert sess._front_lane is None

    @pytest.mark.asyncio
    async def test_front_model_set_uses_front_lane(self) -> None:
        supervisor = _legacy_supervisor()
        sess = _make_voice_session(
            front_model="openai/qwen3.6-35b-a3b",
            front_api_base="http://x:8000/v1",
            supervisor=supervisor,
        )
        sess._front_lane = _front_lane_stub(healthy=True)
        sess._audio_buffer.extend(_audio_frames())
        await sess.handle_speech_end()
        # Front lane handled the turn → legacy path must NOT run.
        supervisor.send_request_stream.assert_not_called()
        sended = [c.args for c in sess._ws.send_json.call_args_list]
        texts = [a[0]["text"] for a in sended if a[0].get("type") == "response_text"]
        assert any("こんにちは！" in t for t in texts)
        # emotion parsed from the front output
        emotions = [a[0]["emotion"] for a in sended if a[0].get("type") == "response_done"]
        assert emotions and emotions[-1] == "smile"

    @pytest.mark.asyncio
    async def test_front_health_failure_falls_back(self) -> None:
        supervisor = _legacy_supervisor()
        sess = _make_voice_session(
            front_model="openai/qwen3.6-35b-a3b",
            front_api_base="http://x:8000/v1",
            supervisor=supervisor,
        )
        sess._front_lane = _front_lane_stub(healthy=False)
        sess._audio_buffer.extend(_audio_frames())
        await sess.handle_speech_end()
        # Health failed → fall back to the full agent loop.
        supervisor.send_request_stream.assert_called()

    @pytest.mark.asyncio
    async def test_front_config_from_voice_config_schema(self) -> None:
        # front_model / front_api_base round-trip through the config schema.
        vc = VoiceConfig(
            front_model="openai/qwen3.6-35b-a3b",
            front_api_base="http://x:8000/v1",
        )
        assert vc.front_model == "openai/qwen3.6-35b-a3b"
        assert vc.front_api_base == "http://x:8000/v1"
        assert VoiceConfig().front_model is None

    @pytest.mark.asyncio
    async def test_front_records_conversation_with_from_person_role(self) -> None:
        """The user turn is recorded with the ``from_person`` role (like the
        existing chat path), and the assistant turn with ``assistant``."""
        sess = _make_voice_session(
            front_model="openai/qwen3.6-35b-a3b",
            front_api_base="http://x:8000/v1",
        )
        sess._front_lane = _front_lane_stub(healthy=True)
        mock_conv = MagicMock()
        with patch("core.memory.conversation.ConversationMemory", return_value=mock_conv):
            sess._audio_buffer.extend(_audio_frames())
            await sess.handle_speech_end()
        roles = [c.args[0] for c in mock_conv.append_turn.call_args_list]
        assert roles == ["human", "assistant"]
        mock_conv.save.assert_called_once()


def test_read_memory_page_rotates_material(tmp_path) -> None:
    (tmp_path / "episodes").mkdir()
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "episodes" / "2026-08-25.md").write_text("A" * 100 + "B" * 100, encoding="utf-8")
    for name in ("k1", "k2", "k3", "k4"):
        (tmp_path / "knowledge" / f"{name}.md").write_text(f"note {name}", encoding="utf-8")

    page0 = read_memory_snippets(tmp_path, "", max_chars=160, page=0)
    page1 = read_memory_snippets(tmp_path, "", max_chars=160, page=1)
    # page 0 is the tail of the episode, page 1 the window before it
    assert "B" * 20 in page0 and "A" * 20 not in page0
    assert "A" * 20 in page1
    # knowledge picks rotate instead of always being the newest three
    assert page0 != page1


def test_is_repeating_detects_looped_paragraph() -> None:
    from core.voice.front import _is_repeating

    para = "「豆知識」の時間ね！今日の手帳、マステで星デコしてたんだよね✨ "
    assert _is_repeating(para) is False
    assert _is_repeating(para + para) is True
    assert _is_repeating("短い") is False
