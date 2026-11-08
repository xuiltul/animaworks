from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from core.hub.avatar import (
    AvatarError,
    avatar_path,
    clear_avatar,
    fetch_avatar_url,
    normalize_avatar,
    save_avatar,
)


def _png(size: tuple[int, int] = (300, 200)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, (200, 20, 20)).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize(
    ("image_format", "mode"),
    [("PNG", "RGB"), ("JPEG", "RGB"), ("WEBP", "RGBA"), ("GIF", "P")],
)
def test_normalize_avatar_converts_supported_formats(image_format: str, mode: str) -> None:
    output = BytesIO()
    Image.new(mode, (300, 200), (200, 20, 20)).save(output, format=image_format)

    result = normalize_avatar(output.getvalue())
    with Image.open(BytesIO(result)) as image:
        assert image.format == "PNG"
        assert image.size == (256, 256)
        assert image.mode == "RGBA"


def test_normalize_avatar_rejects_unsupported_image_format() -> None:
    output = BytesIO()
    Image.new("RGB", (10, 10), (0, 0, 0)).save(output, format="BMP")

    with pytest.raises(AvatarError, match="unsupported"):
        normalize_avatar(output.getvalue())


def test_normalize_avatar_rejects_invalid_data_and_large_images() -> None:
    with pytest.raises(AvatarError, match="unsupported"):
        normalize_avatar(b"not an image")
    with pytest.raises(AvatarError, match="unsupported"):
        normalize_avatar(b"")
    with pytest.raises(AvatarError, match="too large"):
        normalize_avatar(b"x" * (5 * 1024 * 1024 + 1))

    huge = BytesIO()
    Image.new("1", (6000, 5000)).save(huge, format="PNG")
    with pytest.raises(AvatarError, match="too large"):
        normalize_avatar(huge.getvalue())


def test_save_and_clear_avatar(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    path = save_avatar(shared, "muse", b"png")

    assert path == avatar_path(shared, "muse")
    assert path.read_bytes() == b"png"
    clear_avatar(shared, "muse")
    assert not path.exists()
    clear_avatar(shared, "muse")


class _FakeResponse:
    def __init__(self, status: int, *, content_type: str = "image/png", body: bytes = b"", location: str = "") -> None:
        self.status_code = status
        self.headers = {"content-type": content_type}
        if location:
            self.headers["location"] = location
        self._body = body

    def iter_bytes(self):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.urls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method: str, url: str):
        self.urls.append(url)
        return self.responses.pop(0)


def _patch_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.hub.avatar._public_addresses", lambda host, port: ["93.184.216.34"])


def test_fetch_avatar_url_follows_valid_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_public(monkeypatch)
    client = _FakeClient(
        [
            _FakeResponse(302, location="https://cdn.example.com/a.png"),
            _FakeResponse(200, body=b"png"),
        ]
    )
    monkeypatch.setattr("core.hub.avatar.httpx.Client", lambda **kwargs: client)

    assert fetch_avatar_url("https://hub.example.com/a.png") == b"png"
    assert client.urls == ["https://hub.example.com/a.png", "https://cdn.example.com/a.png"]


def test_fetch_avatar_url_rejects_too_many_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_public(monkeypatch)
    client = _FakeClient(
        [
            _FakeResponse(301, location="https://a.example.com/1"),
            _FakeResponse(301, location="https://a.example.com/2"),
            _FakeResponse(301, location="https://a.example.com/3"),
            _FakeResponse(301, location="https://a.example.com/4"),
        ]
    )
    monkeypatch.setattr("core.hub.avatar.httpx.Client", lambda **kwargs: client)

    with pytest.raises(AvatarError, match="fetch failed"):
        fetch_avatar_url("https://hub.example.com/a.png")


def test_fetch_avatar_url_rejects_non_image_and_oversize(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_public(monkeypatch)
    non_image = _FakeClient([_FakeResponse(200, content_type="text/html", body=b"x")])
    monkeypatch.setattr("core.hub.avatar.httpx.Client", lambda **kwargs: non_image)
    with pytest.raises(AvatarError, match="unsupported"):
        fetch_avatar_url("https://hub.example.com/a.png")

    oversize = _FakeClient([_FakeResponse(200, body=b"x" * (5 * 1024 * 1024 + 1))])
    monkeypatch.setattr("core.hub.avatar.httpx.Client", lambda **kwargs: oversize)
    with pytest.raises(AvatarError, match="too large"):
        fetch_avatar_url("https://hub.example.com/a.png")


def test_fetch_avatar_url_revalidates_redirect_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def public(host: str, port: int) -> list[str]:
        calls.append((host, port))
        if host == "private.example.com":
            raise AvatarError("url not allowed")
        return ["93.184.216.34"]

    monkeypatch.setattr("core.hub.avatar._public_addresses", public)
    client = _FakeClient([_FakeResponse(302, location="https://private.example.com/a.png")])
    monkeypatch.setattr("core.hub.avatar.httpx.Client", lambda **kwargs: client)

    with pytest.raises(AvatarError, match="url not allowed"):
        fetch_avatar_url("https://hub.example.com/a.png")
    assert calls == [("hub.example.com", 443), ("private.example.com", 443)]


def test_fetch_avatar_url_rejects_unsafe_urls_and_private_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    for url in (
        "http://hub.example.com/a.png",
        "https://user@hub.example.com/a.png",
        "https://hub.example.com:8443/a.png",
    ):
        with pytest.raises(AvatarError, match="url not allowed"):
            fetch_avatar_url(url)

    for address in ("10.0.0.1", "127.0.0.1", "169.254.1.1", "::1"):
        monkeypatch.setattr(
            "core.hub.avatar.socket.getaddrinfo",
            lambda host, port, address=address, **kwargs: [(2, 1, 6, "", (address, 443, 0, 0))],
        )
        with pytest.raises(AvatarError, match="url not allowed"):
            fetch_avatar_url("https://hub.example.com/a.png")
