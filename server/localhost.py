from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Localhost trust detection and CSRF validation utilities.

Used by auth_guard middleware and WebSocket authentication to allow
localhost connections to bypass password authentication when
``trust_localhost`` is enabled.
"""

import ipaddress
from urllib.parse import urlparse

from starlette.requests import Request

_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_localhost(request: Request) -> bool:
    """Check if the request originates from a loopback address."""
    if request.client is None:
        return False
    try:
        addr = ipaddress.ip_address(request.client.host)
    except ValueError:
        return False
    return addr.is_loopback


def _extract_host(url_or_host: str) -> str:
    """Extract host portion from URL or host:port string."""
    # Handle full URLs: http://localhost:8000/path
    if "://" in url_or_host:
        return urlparse(url_or_host).hostname or ""
    # Handle [::1]:port (IPv6 bracket notation)
    if url_or_host.startswith("["):
        return url_or_host.split("]")[0].strip("[]")
    # Handle host:port
    if ":" in url_or_host:
        return url_or_host.rsplit(":", 1)[0]
    return url_or_host


def _is_safe_localhost_request(request: Request) -> bool:
    """Check if request is from localhost AND safe from CSRF.

    Returns True only when:
    1. The connection originates from a loopback IP address
    2. The Origin header (if present) points to localhost
    3. The Host header points to localhost
    """
    if not _is_localhost(request):
        return False

    # Origin header check (browsers always send this for cross-origin requests)
    origin = request.headers.get("origin")
    if origin:
        origin_host = _extract_host(origin)
        if origin_host not in _LOCALHOST_HOSTS:
            return False

    # Host header check (DNS rebinding protection)
    host = request.headers.get("host", "")
    host_name = _extract_host(host)
    if host_name not in _LOCALHOST_HOSTS:
        return False

    return True
