from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""External-AI avatar normalization, storage, and safe URL fetching."""

import ipaddress
import logging
import socket
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image, ImageOps

from core.platform.atomic_io import atomic_write_bytes

logger = logging.getLogger(__name__)

MAX_AVATAR_BYTES = 5 * 1024 * 1024
MAX_AVATAR_PIXELS = 25_000_000
AVATAR_SIZE = 256
ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "WEBP", "GIF"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 3


class AvatarError(ValueError):
    """A short, user-safe explanation of why avatar handling failed."""


def normalize_avatar(data: bytes) -> bytes:
    """Return a 256x256 PNG made from *data*, or raise :class:`AvatarError`."""
    if not data:
        raise AvatarError("unsupported image")
    if len(data) > MAX_AVATAR_BYTES:
        raise AvatarError("too large")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise AvatarError("unsupported image")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_AVATAR_PIXELS:
                raise AvatarError("too large")
            image.load()
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGBA")
            side = min(image.size)
            left = (image.width - side) // 2
            top = (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize((AVATAR_SIZE, AVATAR_SIZE), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except AvatarError:
        raise
    except Exception as exc:  # corrupt data, truncated files, decompression bombs
        raise AvatarError("unsupported image") from exc


def avatar_path(shared_dir: Path, member_id: str) -> Path:
    """Return the stored PNG path for one member."""
    return Path(shared_dir) / "hub" / "avatars" / f"{member_id}.png"


def save_avatar(shared_dir: Path, member_id: str, png: bytes) -> Path:
    """Atomically store already-normalized avatar bytes."""
    path = avatar_path(shared_dir, member_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, png)
    return path


def clear_avatar(shared_dir: Path, member_id: str) -> None:
    """Remove a stored avatar if it exists."""
    avatar_path(shared_dir, member_id).unlink(missing_ok=True)


def _validate_avatar_url(url: str) -> tuple[str, int]:
    """Validate an https avatar URL and return its host and port."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise AvatarError("url not allowed") from exc
    if parts.scheme != "https" or not parts.hostname:
        raise AvatarError("url not allowed")
    if parts.username is not None or parts.password is not None:
        raise AvatarError("url not allowed")
    if port is not None and port != 443:
        raise AvatarError("url not allowed")
    return parts.hostname, port or 443


def _public_addresses(host: str, port: int) -> list[str]:
    """Resolve *host* and require every address to be global unicast."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise AvatarError("url not allowed") from exc
    addresses: list[str] = []
    for info in infos:
        address = str(info[4][0]).split("%", 1)[0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise AvatarError("url not allowed") from exc
        if not parsed.is_global or parsed.is_multicast:
            raise AvatarError("url not allowed")
        addresses.append(address)
    if not addresses:
        raise AvatarError("url not allowed")
    return addresses


def fetch_avatar_url(url: str) -> bytes:
    """Fetch an avatar over https without following private-network redirects.

    Every redirect target is validated again.  DNS is deliberately re-resolved
    by httpx after this check, so a rebinding attacker can still win the race;
    the public-address check removes the normal private-network SSRF cases.
    """
    current = url
    redirects = 0
    try:
        with httpx.Client(follow_redirects=False, timeout=10, trust_env=False) as client:
            while True:
                host, port = _validate_avatar_url(current)
                _public_addresses(host, port)
                logger.debug("Fetching hub avatar from %s", host)
                with client.stream("GET", current) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location or redirects >= _MAX_REDIRECTS:
                            raise AvatarError("fetch failed")
                        redirects += 1
                        current = urljoin(current, location)
                        continue
                    if not 200 <= response.status_code < 300:
                        raise AvatarError("fetch failed")
                    content_type = response.headers.get("content-type", "").lower()
                    if not content_type.startswith("image/"):
                        raise AvatarError("unsupported image")
                    chunks = bytearray()
                    for chunk in response.iter_bytes():
                        chunks.extend(chunk)
                        if len(chunks) > MAX_AVATAR_BYTES:
                            raise AvatarError("too large")
                    if not chunks:
                        raise AvatarError("fetch failed")
                    return bytes(chunks)
    except AvatarError:
        raise
    except httpx.HTTPError as exc:
        raise AvatarError("fetch failed") from exc
