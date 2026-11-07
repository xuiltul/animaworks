from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Client for Board writes from non-server processes."""

from typing import Any


class BoardPostClientError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def post_via_server(
    channel: str,
    text: str,
    *,
    from_anima: str,
    source: str = "anima",
    from_name: str | None = None,
    request_id: str | None = None,
    suppress_mentions: bool = False,
    triggers: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Post using the authenticated internal API and return its JSON body."""
    from core.host_api import response_detail
    from core.internal_api import host_api

    if triggers is None:
        from core.execution.session.board_reply import current_run_triggers

        triggers = [dict(trigger) for trigger in current_run_triggers()]

    payload: dict[str, Any] = {
        "from_anima": from_anima,
        "channel": channel,
        "text": text,
        "source": source,
        "from_name": from_name,
        "request_id": request_id,
        "suppress_mentions": suppress_mentions,
        "triggers": triggers,
    }
    response = host_api.post("/api/internal/post-channel", json=payload, timeout=30.0)
    if response.status_code < 200 or response.status_code >= 300:
        raise BoardPostClientError(response.status_code, response_detail(response))
    data = response.json()
    if not isinstance(data, dict):
        raise BoardPostClientError(502, "Invalid Board post response")
    return data
