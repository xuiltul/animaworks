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


def _is_private_network(host: str) -> bool:
    """Check if a host string is a private/loopback IP address."""
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback
    except ValueError:
        return host in _LOCALHOST_HOSTS


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
       (direct connection or via trusted reverse proxy)
    2. The Origin header (if present) points to a trusted host
    3. The Host header points to a trusted host

    When behind a reverse proxy, the direct client is the proxy (loopback).
    The Host header will be the external address (e.g. 192.168.x.x) which
    is still safe for private-network deployments.
    """
    if not _is_localhost(request):
        return False

    # Origin header check (browsers always send this for cross-origin requests)
    origin = request.headers.get("origin")
    if origin:
        origin_host = _extract_host(origin)
        if origin_host not in _LOCALHOST_HOSTS and not _is_private_network(origin_host):
            return False

    # Host header check: allow localhost and private network IPs
    # (reverse proxy rewrites Host to the external LAN address)
    host = request.headers.get("host", "")
    host_name = _extract_host(host)
    if host_name not in _LOCALHOST_HOSTS and not _is_private_network(host_name):
        return False

    return True
