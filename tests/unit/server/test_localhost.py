"""Unit tests for localhost trust detection utilities in server/localhost.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from server.localhost import _extract_host, _is_localhost, _is_safe_localhost_request


# ── Helpers ──────────────────────────────────────────────


@dataclass
class _FakeAddress:
    """Minimal stand-in for ``starlette.datastructures.Address``."""

    host: str
    port: int = 50000


def _make_request(
    *,
    client_host: str | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock ``starlette.requests.Request``.

    Parameters
    ----------
    client_host:
        IP address of the connecting client.  ``None`` means
        ``request.client`` is ``None`` (e.g. Unix domain socket).
    headers:
        HTTP headers to attach to the request.
    """
    request = MagicMock()
    if client_host is None:
        request.client = None
    else:
        request.client = _FakeAddress(host=client_host)

    raw_headers = headers or {}
    request.headers = raw_headers
    # MagicMock.get needs to behave like a dict
    request.headers = MagicMock()
    request.headers.get = lambda key, default="": raw_headers.get(key, default)
    return request


# ── _is_localhost ────────────────────────────────────────


class TestIsLocalhost:
    """Tests for ``_is_localhost()``."""

    def test_client_none_returns_false(self):
        req = _make_request(client_host=None)
        assert _is_localhost(req) is False

    def test_ipv4_loopback(self):
        req = _make_request(client_host="127.0.0.1")
        assert _is_localhost(req) is True

    def test_ipv6_loopback(self):
        req = _make_request(client_host="::1")
        assert _is_localhost(req) is True

    def test_ipv4_mapped_ipv6_loopback(self):
        req = _make_request(client_host="::ffff:127.0.0.1")
        assert _is_localhost(req) is True

    def test_private_network_is_not_localhost(self):
        req = _make_request(client_host="192.168.1.1")
        assert _is_localhost(req) is False

    def test_public_ip_is_not_localhost(self):
        req = _make_request(client_host="8.8.8.8")
        assert _is_localhost(req) is False

    def test_invalid_host_string_returns_false(self):
        req = _make_request(client_host="not-an-ip")
        assert _is_localhost(req) is False


# ── _extract_host ────────────────────────────────────────


class TestExtractHost:
    """Tests for ``_extract_host()``."""

    def test_full_url_with_scheme(self):
        assert _extract_host("http://localhost:8000/path") == "localhost"

    def test_https_url(self):
        assert _extract_host("https://example.com:443/foo") == "example.com"

    def test_host_port_no_scheme(self):
        assert _extract_host("localhost:8000") == "localhost"

    def test_ipv6_bracket_notation(self):
        assert _extract_host("[::1]:8000") == "::1"

    def test_evil_host_port(self):
        assert _extract_host("evil.com:8000") == "evil.com"

    def test_plain_hostname(self):
        assert _extract_host("localhost") == "localhost"

    def test_ip_with_port(self):
        assert _extract_host("127.0.0.1:3000") == "127.0.0.1"

    def test_url_with_path_only(self):
        assert _extract_host("http://127.0.0.1/api/test") == "127.0.0.1"

    def test_empty_string(self):
        assert _extract_host("") == ""


# ── _is_safe_localhost_request ───────────────────────────


class TestIsSafeLocalhostRequest:
    """Tests for ``_is_safe_localhost_request()``."""

    def test_localhost_with_localhost_host_header(self):
        """Localhost client + Host: localhost -> safe."""
        req = _make_request(
            client_host="127.0.0.1",
            headers={"host": "localhost:8000"},
        )
        assert _is_safe_localhost_request(req) is True

    def test_localhost_with_127_host_header(self):
        """Localhost client + Host: 127.0.0.1 -> safe."""
        req = _make_request(
            client_host="127.0.0.1",
            headers={"host": "127.0.0.1:8000"},
        )
        assert _is_safe_localhost_request(req) is True

    def test_localhost_with_ipv6_host_header(self):
        """IPv6 loopback client + Host: [::1] -> safe."""
        req = _make_request(
            client_host="::1",
            headers={"host": "[::1]:8000"},
        )
        assert _is_safe_localhost_request(req) is True

    def test_localhost_with_localhost_origin(self):
        """Localhost + Origin=http://localhost -> safe."""
        req = _make_request(
            client_host="127.0.0.1",
            headers={
                "host": "localhost:8000",
                "origin": "http://localhost:8000",
            },
        )
        assert _is_safe_localhost_request(req) is True

    def test_localhost_with_evil_origin_blocked(self):
        """Localhost + Origin=http://evil.com -> CSRF, blocked."""
        req = _make_request(
            client_host="127.0.0.1",
            headers={
                "host": "localhost:8000",
                "origin": "http://evil.com",
            },
        )
        assert _is_safe_localhost_request(req) is False

    def test_localhost_with_evil_host_blocked(self):
        """Localhost + Host=evil.com -> DNS rebinding, blocked."""
        req = _make_request(
            client_host="127.0.0.1",
            headers={"host": "evil.com:8000"},
        )
        assert _is_safe_localhost_request(req) is False

    def test_remote_ip_blocked(self):
        """Remote IP -> blocked regardless of headers."""
        req = _make_request(
            client_host="192.168.1.1",
            headers={"host": "localhost:8000"},
        )
        assert _is_safe_localhost_request(req) is False

    def test_client_none_blocked(self):
        """No client info -> blocked."""
        req = _make_request(
            client_host=None,
            headers={"host": "localhost:8000"},
        )
        assert _is_safe_localhost_request(req) is False

    def test_no_origin_header_with_localhost_host(self):
        """No Origin header + localhost Host -> safe (same-origin GET)."""
        req = _make_request(
            client_host="127.0.0.1",
            headers={"host": "localhost:8000"},
        )
        assert _is_safe_localhost_request(req) is True

    def test_empty_host_header_blocked(self):
        """Empty Host header -> blocked (not in _LOCALHOST_HOSTS)."""
        req = _make_request(
            client_host="127.0.0.1",
            headers={"host": ""},
        )
        assert _is_safe_localhost_request(req) is False

    def test_ipv4_mapped_ipv6_with_localhost_host(self):
        """IPv4-mapped IPv6 loopback + localhost Host -> safe."""
        req = _make_request(
            client_host="::ffff:127.0.0.1",
            headers={"host": "localhost:8000"},
        )
        assert _is_safe_localhost_request(req) is True
