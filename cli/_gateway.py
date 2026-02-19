# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────

_DEFAULT_GATEWAY_URL = "http://localhost:18500"
_ENV_GATEWAY_URL = "ANIMAWORKS_GATEWAY_URL"


# ── Public API ────────────────────────────────────────────


def resolve_gateway_url(args: argparse.Namespace) -> str:
    """Resolve the gateway URL from CLI args or environment variable.

    Priority: ``--gateway-url`` flag > ``ANIMAWORKS_GATEWAY_URL`` env > default.
    """
    return getattr(args, "gateway_url", None) or os.environ.get(
        _ENV_GATEWAY_URL, _DEFAULT_GATEWAY_URL
    )


def gateway_request(
    args: argparse.Namespace,
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any] | list[Any] | None:
    """Make an HTTP request to the gateway, handling connection errors.

    Args:
        args: Parsed CLI namespace (used to resolve gateway URL).
        method: HTTP method (``GET``, ``POST``, etc.).
        path: URL path appended to the gateway base (e.g. ``/api/animas``).
        json: Optional JSON body for the request.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response body, or ``None`` on empty response.

    Raises:
        SystemExit: On connection error, timeout, or HTTP error.
    """
    import httpx

    gateway = resolve_gateway_url(args)
    url = f"{gateway}{path}"
    logger.debug("Gateway %s %s (timeout=%.1fs)", method, url, timeout)

    try:
        resp = httpx.request(method, url, json=json, timeout=timeout)
        return resp.json()
    except httpx.ConnectError:
        print(
            f"Cannot connect to gateway at {gateway}. "
            "Use --local for direct mode."
        )
        sys.exit(1)
    except httpx.TimeoutException:
        print(f"Request timed out after {timeout}s.")
        sys.exit(1)
    except httpx.HTTPError as exc:
        print(f"HTTP error: {exc}")
        sys.exit(1)


def gateway_request_or_none(
    args: argparse.Namespace,
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any] | list[Any] | None:
    """Make a gateway request, returning ``None`` on connection failure.

    Unlike :func:`gateway_request`, this variant does **not** call
    ``sys.exit`` on ``ConnectError`` -- it returns ``None`` so the caller
    can implement its own fallback logic.

    Other errors (timeout, general HTTP) still cause ``sys.exit(1)``.
    """
    import httpx

    gateway = resolve_gateway_url(args)
    url = f"{gateway}{path}"
    logger.debug("Gateway %s %s (timeout=%.1fs, soft-fail)", method, url, timeout)

    try:
        resp = httpx.request(method, url, json=json, timeout=timeout)
        return resp.json()
    except httpx.ConnectError:
        logger.debug("Gateway unreachable at %s", gateway)
        return None
    except httpx.TimeoutException:
        print(f"Request timed out after {timeout}s.")
        sys.exit(1)
    except httpx.HTTPError as exc:
        print(f"HTTP error: {exc}")
        sys.exit(1)
