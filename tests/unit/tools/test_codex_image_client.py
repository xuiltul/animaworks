# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codex-first image generation clients."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from core.config.models import ImageGenConfig
from core.tools.image.codex import (
    CodexFirstClient,
    CodexImageClient,
    codex_available,
)
from core.tools.image_gen import _build_fullbody_client, _build_reference_client


def _png_bytes(width: int, height: int, color: tuple[int, int, int, int] = (10, 20, 30, 255)) -> bytes:
    img = Image.new("RGBA", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _decode_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        return img.size


def _mock_codex_writes_png(
    size: tuple[int, int] = (64, 64),
    *,
    returncode: int = 0,
    missing_out: bool = False,
) -> Any:
    """Return a subprocess.run side_effect that writes out.png under -C dir."""

    def _run(cmd: list[str], **kwargs: Any) -> MagicMock:
        tmpdir: Path | None = None
        for i, part in enumerate(cmd):
            if part == "-C" and i + 1 < len(cmd):
                tmpdir = Path(cmd[i + 1])
                break
        assert tmpdir is not None, f"no -C in cmd: {cmd}"
        if not missing_out and returncode == 0:
            (tmpdir / "out.png").write_bytes(_png_bytes(*size))
        result = MagicMock()
        result.returncode = returncode
        result.stderr = b"stderr tail from mock"
        result.stdout = b""
        return result

    return _run


class TestCodexAvailable:
    def test_true_when_which_finds_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("core.tools.image.codex.shutil.which", lambda _n: "/usr/bin/codex")
        assert codex_available() is True

    def test_false_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("core.tools.image.codex.shutil.which", lambda _n: None)
        assert codex_available() is False


class TestCodexImageClient:
    def test_generate_fullbody_resizes_to_target(self) -> None:
        client = CodexImageClient(ImageGenConfig(image_style="anime"))
        with patch("core.tools.image.codex.subprocess.run", side_effect=_mock_codex_writes_png((80, 120))):
            out = client.generate_fullbody(prompt="1girl, black hair", width=1024, height=1536)
        assert isinstance(out, bytes)
        assert _decode_size(out) == (1024, 1536)

    def test_cover_crop_from_square_to_portrait(self) -> None:
        """Square mock output must become 1024x1536 without stretch."""
        client = CodexImageClient()
        with patch("core.tools.image.codex.subprocess.run", side_effect=_mock_codex_writes_png((200, 200))):
            out = client.generate_fullbody(prompt="subject", width=1024, height=1536)
        assert _decode_size(out) == (1024, 1536)

    def test_generate_from_reference_aspect_1_1(self) -> None:
        client = CodexImageClient(ImageGenConfig(image_style="realistic"))
        ref = _png_bytes(32, 32)
        with patch("core.tools.image.codex.subprocess.run", side_effect=_mock_codex_writes_png((90, 90))):
            out = client.generate_from_reference(reference_image=ref, prompt="icon", aspect_ratio="1:1")
        assert _decode_size(out) == (1024, 1024)

    def test_generate_from_reference_aspect_3_4(self) -> None:
        client = CodexImageClient()
        ref = _png_bytes(32, 48)
        with patch("core.tools.image.codex.subprocess.run", side_effect=_mock_codex_writes_png((100, 100))):
            out = client.generate_from_reference(reference_image=ref, prompt="bust", aspect_ratio="3:4")
        assert _decode_size(out) == (1024, 1365)

    def test_missing_out_png_raises_runtime_error(self) -> None:
        client = CodexImageClient()
        with (
            patch(
                "core.tools.image.codex.subprocess.run",
                side_effect=_mock_codex_writes_png(missing_out=True),
            ),
            pytest.raises(RuntimeError, match="out.png"),
        ):
            client.generate_fullbody(prompt="x")

    def test_nonzero_returncode_raises_runtime_error(self) -> None:
        client = CodexImageClient()
        with (
            patch(
                "core.tools.image.codex.subprocess.run",
                side_effect=_mock_codex_writes_png(returncode=1),
            ),
            pytest.raises(RuntimeError, match="failed"),
        ):
            client.generate_fullbody(prompt="x")

    def test_error_reason_prefers_last_distinct_error_line(self) -> None:
        client = CodexImageClient()

        def _run(_cmd: list[str], **_kwargs: Any) -> MagicMock:
            result = MagicMock()
            result.returncode = 1
            result.stdout = b""
            result.stderr = (
                b"debug output\n"
                b"ERROR: an earlier error\n"
                b"ERROR: You've hit your usage limit; try again at 12:06 PM.\n"
                b"ERROR: You've hit your usage limit; try again at 12:06 PM.\n"
            )
            return result

        with (
            patch("core.tools.image.codex.subprocess.run", side_effect=_run),
            pytest.raises(RuntimeError, match="ERROR: You've hit your usage limit; try again at 12:06 PM."),
        ):
            client.generate_fullbody(prompt="x")

    def test_vibe_and_face_refs_attached(self) -> None:
        client = CodexImageClient()
        captured: dict[str, Any] = {}

        def _run(cmd: list[str], **kwargs: Any) -> MagicMock:
            captured["cmd"] = cmd
            return _mock_codex_writes_png((32, 32))(cmd, **kwargs)

        vibe = _png_bytes(8, 8, (1, 0, 0, 255))
        face = _png_bytes(8, 8, (0, 1, 0, 255))
        with patch("core.tools.image.codex.subprocess.run", side_effect=_run):
            client.generate_fullbody(
                prompt="char",
                vibe_image=vibe,
                face_reference_image=face,
                width=64,
                height=64,
            )
        cmd = captured["cmd"]
        assert cmd.count("-i") == 2
        # -i paths must be absolute (resolved against invoking cwd, not -C)
        img_paths = [cmd[i + 1] for i, part in enumerate(cmd) if part == "-i"]
        assert all(Path(p).is_absolute() for p in img_paths)
        assert [Path(p).name for p in img_paths] == ["ref_0.png", "ref_1.png"]
        # "--" must precede the prompt so variadic -i does not swallow it
        assert cmd[-2] == "--"


class TestCodexFirstClient:
    def test_fallback_on_missing_out(self) -> None:
        fallback = MagicMock()
        fallback.generate_fullbody.return_value = b"FALLBACK-FULL"
        client = CodexFirstClient(fallback_factory=lambda: fallback)

        with patch(
            "core.tools.image.codex.subprocess.run",
            side_effect=_mock_codex_writes_png(missing_out=True),
        ):
            out = client.generate_fullbody(prompt="p", width=100, height=200)

        assert out == b"FALLBACK-FULL"
        fallback.generate_fullbody.assert_called_once()
        kwargs = fallback.generate_fullbody.call_args.kwargs
        assert kwargs["prompt"] == "p"
        assert kwargs["width"] == 100
        assert kwargs["height"] == 200

    def test_fallback_on_nonzero_rc_for_reference(self) -> None:
        fallback = MagicMock()
        fallback.generate_from_reference.return_value = b"FALLBACK-REF"
        client = CodexFirstClient(fallback_factory=lambda: fallback)
        ref = _png_bytes(16, 16)

        with patch(
            "core.tools.image.codex.subprocess.run",
            side_effect=_mock_codex_writes_png(returncode=2),
        ):
            out = client.generate_from_reference(reference_image=ref, prompt="icon", aspect_ratio="1:1")

        assert out == b"FALLBACK-REF"
        fallback.generate_from_reference.assert_called_once()
        call_kw = fallback.generate_from_reference.call_args.kwargs
        assert call_kw["prompt"] == "icon"
        assert call_kw["aspect_ratio"] == "1:1"
        assert call_kw["reference_image"] == ref

    def test_success_does_not_call_fallback_factory(self) -> None:
        factory = MagicMock(side_effect=RuntimeError("factory should not run"))
        client = CodexFirstClient(fallback_factory=factory)
        with patch("core.tools.image.codex.subprocess.run", side_effect=_mock_codex_writes_png((40, 40))):
            out = client.generate_fullbody(prompt="ok", width=64, height=64)
        assert _decode_size(out) == (64, 64)
        factory.assert_not_called()

    def test_fallback_failure_preserves_codex_reason(self) -> None:
        fallback = MagicMock()
        fallback.generate_fullbody.side_effect = RuntimeError("FAL_KEY required")
        client = CodexFirstClient(fallback_factory=lambda: fallback)

        def _run(_cmd: list[str], **_kwargs: Any) -> MagicMock:
            result = MagicMock()
            result.returncode = 1
            result.stdout = b""
            result.stderr = b"ERROR: You've hit your usage limit; try again at 12:06 PM.\n"
            return result

        with (
            patch("core.tools.image.codex.subprocess.run", side_effect=_run),
            pytest.raises(RuntimeError) as caught,
        ):
            client.generate_fullbody(prompt="x")

        assert str(caught.value) == (
            "FAL_KEY required (codex: ERROR: You've hit your usage limit; try again at 12:06 PM.)"
        )
        assert isinstance(caught.value.__cause__, RuntimeError)


class TestClientBuilders:
    def test_codex_unavailable_uses_api_clients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("core.tools.image_gen.codex_available", lambda: False)
        monkeypatch.setenv("NOVELAI_TOKEN", "tok")
        monkeypatch.setenv("FAL_KEY", "fal-tok")
        cfg = ImageGenConfig(image_style="anime", backend="api", prefer_codex=True)
        fullbody = _build_fullbody_client(cfg)
        ref = _build_reference_client(cfg)
        from core.tools.image.fal import FluxKontextClient
        from core.tools.image.novelai import NovelAIClient

        assert isinstance(fullbody, NovelAIClient)
        assert isinstance(ref, FluxKontextClient)

    def test_prefer_codex_false_skips_codex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("core.tools.image_gen.codex_available", lambda: True)
        monkeypatch.setenv("NOVELAI_TOKEN", "tok")
        monkeypatch.setenv("FAL_KEY", "fal-tok")
        cfg = ImageGenConfig(image_style="anime", backend="api", prefer_codex=False)
        fullbody = _build_fullbody_client(cfg)
        ref = _build_reference_client(cfg)
        from core.tools.image.fal import FluxKontextClient
        from core.tools.image.novelai import NovelAIClient

        assert isinstance(fullbody, NovelAIClient)
        assert isinstance(ref, FluxKontextClient)
        assert not isinstance(fullbody, CodexFirstClient)

    def test_prefer_codex_true_and_available_wraps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("core.tools.image_gen.codex_available", lambda: True)
        cfg = ImageGenConfig(image_style="anime", backend="api", prefer_codex=True)
        fullbody = _build_fullbody_client(cfg)
        ref = _build_reference_client(cfg)
        assert isinstance(fullbody, CodexFirstClient)
        assert isinstance(ref, CodexFirstClient)

    def test_diffusers_backend_ignores_codex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("core.tools.image_gen.codex_available", lambda: True)
        cfg = ImageGenConfig(backend="diffusers", prefer_codex=True)

        with patch("core.tools.image_gen.LocalDiffusersClient") as client:
            assert _build_fullbody_client(cfg) is client.return_value
            assert _build_reference_client(cfg) is client.return_value
        assert client.call_count == 2
