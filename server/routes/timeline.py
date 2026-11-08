from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Company timeline API (human view and Anima internal API)."""

import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.i18n import t

logger = logging.getLogger(__name__)


class TimelineReadRequest(BaseModel):
    upto_id: str = Field(min_length=1)


class InternalTimelineAddRequest(BaseModel):
    from_anima: str | None = None
    kind: Literal["done", "decided"]
    text: str
    about: str | None = None
    ref: str | None = None
    urgent: bool = False


def _request_username(request: Request) -> str:
    """Return the authenticated username, falling back to "human"."""
    user = getattr(request.state, "user", None)
    username = getattr(user, "username", "") if user is not None else ""
    return str(username or "human")


def _shared_dir(request: Request) -> Path:
    from core.paths import get_shared_dir

    return Path(getattr(request.app.state, "shared_dir", None) or get_shared_dir())


def _animas_dir(request: Request) -> Path:
    from core.paths import get_animas_dir

    return Path(getattr(request.app.state, "animas_dir", None) or get_animas_dir())


def _hub():
    from core.config.models import load_config

    return load_config().hub


def _reader_record(reads: dict[str, Any], reader: str) -> dict[str, str]:
    record = reads.get("readers", {}).get(reader)
    if not isinstance(record, dict):
        return {"upto_id": "", "upto_ts": ""}
    return {
        "upto_id": str(record.get("upto_id") or ""),
        "upto_ts": str(record.get("upto_ts") or ""),
    }


def create_timeline_router() -> APIRouter:
    from server.internal_auth import ensure_self, require_internal_caller

    router = APIRouter()
    internal = APIRouter(dependencies=[Depends(require_internal_caller)])

    @router.get("/timeline")
    def get_timeline(request: Request, limit: int = 50, since: str | None = None):
        """Return the company timeline newest first, with the caller's read flags."""
        if limit < 1:
            return JSONResponse(
                status_code=400,
                content={"detail": t("server.timeline_invalid_request", detail="limit must be positive")},
            )
        hub = _hub()
        if not hub.timeline_enabled:
            return {"enabled": False, "items": [], "unread": 0}

        from core.hub.timeline import (
            decorate_items,
            is_read_for_reader,
            list_items,
            load_timeline_reads,
        )

        shared_dir = _shared_dir(request)
        reader = f"human:{_request_username(request)}"
        all_items = list_items(shared_dir, company=hub.company, since=since, limit=100_000)
        reads = load_timeline_reads(shared_dir)
        items = decorate_items(
            all_items[:limit],
            reads=reads,
            reader=reader,
            reference_items=all_items,
        )
        unread = sum(1 for item in all_items if not is_read_for_reader(item, reads, reader))
        return {
            "enabled": True,
            "company": hub.company,
            "items": items,
            "unread": unread,
            "reader": _reader_record(reads, reader),
        }

    @router.post("/timeline/read")
    def mark_timeline_read(request: Request, body: TimelineReadRequest):
        """Advance the requesting human's read position in the timeline."""
        hub = _hub()
        if not hub.timeline_enabled:
            return JSONResponse(status_code=409, content={"detail": t("server.timeline_disabled")})

        from core.hub.timeline import advance_timeline_read, find_item, load_timeline_reads

        shared_dir = _shared_dir(request)
        item = find_item(shared_dir, body.upto_id)
        if item is None or item.get("company") != hub.company:
            return JSONResponse(
                status_code=404,
                content={"detail": t("server.timeline_item_not_found", item_id=body.upto_id)},
            )

        reader = f"human:{_request_username(request)}"
        changed = advance_timeline_read(shared_dir, reader, body.upto_id)
        current = _reader_record(load_timeline_reads(shared_dir), reader)
        return {
            "changed": changed,
            "upto_id": current["upto_id"] or body.upto_id,
            "upto_ts": current["upto_ts"] or str(item.get("ts") or ""),
        }

    @internal.post("/internal/timeline/add")
    async def internal_timeline_add(request: Request, body: InternalTimelineAddRequest):
        """Append a timeline item on behalf of the verified caller."""
        caller = getattr(request.state, "internal_caller", None)
        claimed = body.from_anima or (caller.name if caller is not None and caller.kind == "anima" else None)
        denied = ensure_self(caller, claimed, path=request.url.path)
        if denied is not None:
            return denied

        hub = _hub()
        if not hub.timeline_enabled:
            return JSONResponse(status_code=409, content={"detail": t("server.timeline_disabled")})

        from core.config.models import read_anima_company_checked
        from core.hub.timeline import append_item
        from server.events import emit_direct

        company = hub.company
        if caller is not None and caller.kind == "anima":
            from_name = caller.name
            readable, company_value = read_anima_company_checked(_animas_dir(request) / from_name)
            if not readable or company_value != hub.company:
                return JSONResponse(
                    status_code=403,
                    content={"detail": t("server.timeline_company_mismatch")},
                )
            company = company_value
        else:
            from_name = body.from_anima or "operator"
            if body.from_anima:
                readable, company_value = read_anima_company_checked(_animas_dir(request) / body.from_anima)
                if not readable or company_value != hub.company:
                    return JSONResponse(
                        status_code=403,
                        content={"detail": t("server.timeline_company_mismatch")},
                    )
                company = company_value

        try:
            item = append_item(
                _shared_dir(request),
                from_name=from_name,
                company=company,
                kind=body.kind,
                text=body.text,
                source="anima",
                about=body.about,
                urgent=body.urgent,
                ref=body.ref,
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"detail": t("server.timeline_invalid_request", detail=str(exc))},
            )

        await emit_direct(getattr(request.app.state, "ws_manager", None), "timeline.item", item)
        return {"ok": True, "item": item}

    @internal.get("/internal/timeline")
    def internal_timeline(request: Request, limit: int = 50, since: str | None = None):
        """Return the caller company's timeline for the Anima CLI."""
        if limit < 1:
            return JSONResponse(
                status_code=400,
                content={"detail": t("server.timeline_invalid_request", detail="limit must be positive")},
            )
        hub = _hub()
        if not hub.timeline_enabled:
            return {"enabled": False, "items": [], "unread": 0}

        from core.config.models import read_anima_company_checked
        from core.hub.timeline import decorate_items, is_read_by_human, list_items, load_timeline_reads

        caller = getattr(request.state, "internal_caller", None)
        company = hub.company
        if caller is not None and caller.kind == "anima":
            readable, company_value = read_anima_company_checked(_animas_dir(request) / caller.name)
            if not readable or company_value != hub.company:
                return JSONResponse(
                    status_code=403,
                    content={"detail": t("server.timeline_company_mismatch")},
                )
            company = company_value

        shared_dir = _shared_dir(request)
        items = list_items(shared_dir, company=company, since=since, limit=limit)
        reads = load_timeline_reads(shared_dir)
        decorated = decorate_items(items, reads=reads, reader="")
        for item in decorated:
            item["read"] = bool(item.get("read_by_human"))
        all_items = list_items(shared_dir, company=company, since=since, limit=100_000)
        unread = sum(1 for item in all_items if not is_read_by_human(item, reads))
        return {
            "enabled": True,
            "company": company,
            "items": decorated,
            "unread": unread,
        }

    router.include_router(internal)
    return router
