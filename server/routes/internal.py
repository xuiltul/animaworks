from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.events import emit

logger = logging.getLogger("animaworks.routes.internal")

_native_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="native-ops",
)


class MessageSentNotification(BaseModel):
    from_person: str
    to_person: str
    content: str = ""
    message_id: str = ""


class EmbedRequest(BaseModel):
    texts: list[str]
    purpose: Literal["document", "query"] = "document"
    priority: Literal["interactive", "bulk"] = "interactive"


class VectorQueryRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    embedding: list[float]
    top_k: int = 10
    filter_metadata: dict[str, str | int | float] | None = None


class VectorUpsertRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    documents: list[dict[str, Any]]


class VectorUpdateMetadataRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    ids: list[str]
    metadatas: list[dict[str, str | int | float]]


class VectorDeleteDocumentsRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    ids: list[str]


class VectorGetByMetadataRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    where: dict[str, str | int | float] = {}
    limit: int = 20


class VectorGetByIdsRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    ids: list[str]


class VectorCollectionRequest(BaseModel):
    anima_name: str | None = None
    collection: str


class VectorListCollectionsRequest(BaseModel):
    anima_name: str | None = None


class VectorQuickCheckRequest(BaseModel):
    anima_name: str
    timeout_seconds: float = 10.0
    source: str = "internal_vector_quick_check"
    record_repair: bool = True


class NotificationMappingRequest(BaseModel):
    ts: str
    channel: str
    anima_name: str
    notification_text: str = ""
    callback_id: str = ""


class InteractionCreateRequest(BaseModel):
    anima_name: str
    category: str = "approval"
    options: list[str]
    allowed_users: dict[str, list[str]] | None = None
    callback_id: str = ""


class InteractionMessageTsRequest(BaseModel):
    callback_id: str
    platform: str = "slack"
    ts: str


class AnimaCreateRequest(BaseModel):
    character_sheet_content: str | None = None
    character_sheet_path: str | None = None
    name: str | None = None
    supervisor: str | None = None
    calling_anima: str = ""  # supervisor fallback when status.json has none


class DelegateTaskPersistRequest(BaseModel):
    delegator: str  # source anima name
    target: str  # destination anima name
    instruction: str  # full delegation text
    summary: str
    deadline: str  # relative ('30m','2h','1d') or ISO8601
    sub_task_id: str  # client-assigned 12hex id
    tracking_task_id: str  # client-assigned 12hex id
    workspace: str = ""  # resolve_workspace absolute path string
    persist_sub: bool = True  # write to subordinate queue
    persist_tracking: bool = True  # write delegated entry on delegator queue
    persist_pending: bool = True  # create state/pending/<id>.json


def create_internal_router() -> APIRouter:
    router = APIRouter()

    @router.get("/internal/company/boundary")
    async def internal_company_boundary(from_anima: str, to_anima: str):
        """Resolve company membership on the host for sandboxed handlers."""
        from core.anima_factory import validate_anima_name
        from core.company import get_company_display_name
        from core.config.models import read_anima_company_checked
        from core.paths import get_animas_dir

        if validate_anima_name(from_anima) or validate_anima_name(to_anima):
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid anima name"},
            )

        animas_dir = get_animas_dir()
        from_readable, from_company = read_anima_company_checked(animas_dir / from_anima)
        to_readable, to_company = read_anima_company_checked(animas_dir / to_anima)
        if not from_readable or not to_readable:
            unreadable = [
                name for name, readable in ((from_anima, from_readable), (to_anima, to_readable)) if not readable
            ]
            return JSONResponse(
                status_code=503,
                content={
                    "detail": f"Company membership unreadable for: {', '.join(unreadable)}",
                },
            )

        cross_company = from_company is not None and to_company is not None and from_company != to_company
        try:
            display_name = get_company_display_name(to_company or "")
        except Exception:
            logger.warning(
                "Failed to resolve company display name for %r",
                to_company,
                exc_info=True,
            )
            display_name = to_company or ""

        return {
            "from_company": from_company,
            "to_company": to_company,
            "cross_company": cross_company,
            "to_display_name": display_name,
        }

    @router.post("/internal/message-sent")
    async def internal_message_sent(body: MessageSentNotification, request: Request):
        """Notify the server that a message was sent via CLI.

        Triggers WebSocket broadcast and updates reply tracking so that
        selective archival (Fix 2) works for CLI-sent messages too.
        """
        await emit(
            request,
            "anima.interaction",
            {
                "from_person": body.from_person,
                "to_person": body.to_person,
                "type": "message",
                "summary": body.content[:200],
                "message_id": body.message_id,
            },
        )

        # Note: replied_to tracking is now managed by each Anima process.
        # The server no longer holds live DigitalAnima objects.

        return {"status": "ok"}

    @router.get("/messages/{message_id}")
    async def get_message(message_id: str, request: Request):
        """Return the full JSON of a stored message by its ID."""
        # Sanitize to prevent path traversal
        if "/" in message_id or "\\" in message_id or ".." in message_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid message_id"},
            )

        shared_dir: Path = request.app.state.shared_dir
        inbox_root = shared_dir / "inbox"
        if not inbox_root.is_dir():
            return JSONResponse(
                status_code=404,
                content={"detail": "Message not found"},
            )

        filename = f"{message_id}.json"
        for anima_inbox in sorted(inbox_root.iterdir()):
            if not anima_inbox.is_dir():
                continue
            # Check processed first, then inbox root
            for candidate in [
                anima_inbox / "processed" / filename,
                anima_inbox / filename,
            ]:
                if candidate.is_file():
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    return data

        return JSONResponse(
            status_code=404,
            content={"detail": "Message not found"},
        )

    # ── Embedding inference endpoint ────────────────────────────

    @router.post("/internal/embed")
    async def internal_embed(body: EmbedRequest):
        """Centralized embedding inference for child processes.

        Child processes call this endpoint via HTTP instead of loading
        the SentenceTransformer model on their own GPU, reducing total
        VRAM usage from ~22 GB to ~800 MB.
        """
        if len(body.texts) > 1000:
            return JSONResponse(
                status_code=400,
                content={"detail": "Max 1000 texts per request"},
            )
        if not body.texts:
            return {"embeddings": []}

        from functools import partial

        from core.memory.rag.singleton import thread_safe_encode

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            _native_executor,
            partial(thread_safe_encode, body.texts, purpose=body.purpose, priority=body.priority),
        )
        return {"embeddings": embeddings}

    # ── Vector store endpoints (ChromaDB process separation) ───────

    def _body_payload(body: BaseModel) -> dict[str, Any]:
        if hasattr(body, "model_dump"):
            return body.model_dump()
        return body.dict()

    async def _require_vector_worker(request: Request, path: str, body: BaseModel) -> dict[str, Any] | JSONResponse:
        manager = getattr(request.app.state, "vector_worker", None)
        if manager is None or not getattr(manager, "enabled", False):
            logger.warning("Vector worker unavailable for %s: manager disabled or missing", path)
            return JSONResponse(status_code=503, content={"detail": "Vector worker unavailable"})
        try:
            response = await manager.post(path, _body_payload(body))
        except Exception as exc:
            from core.memory.rag.vector_worker_client import VectorWorkerUnavailable

            if not isinstance(exc, VectorWorkerUnavailable):
                logger.exception("Vector worker request failed unexpectedly: %s", path)
            else:
                logger.warning("Vector worker unavailable for %s: %s", path, exc)
            return JSONResponse(status_code=503, content={"detail": "Vector worker unavailable"})
        if response.status_code >= 400:
            source_headers = dict(getattr(response, "headers", None) or {})
            headers = {}
            retry_after = source_headers.get("Retry-After") or source_headers.get("retry-after")
            if retry_after:
                headers["Retry-After"] = retry_after
            return JSONResponse(status_code=response.status_code, content=response.data, headers=headers)
        return response.data

    @router.post("/internal/vector/query")
    async def vector_query(body: VectorQueryRequest, request: Request):
        return await _require_vector_worker(request, "/query", body)

    @router.post("/internal/vector/upsert")
    async def vector_upsert(body: VectorUpsertRequest, request: Request):
        return await _require_vector_worker(request, "/upsert", body)

    @router.post("/internal/vector/update-metadata")
    async def vector_update_metadata(body: VectorUpdateMetadataRequest, request: Request):
        return await _require_vector_worker(request, "/update-metadata", body)

    @router.post("/internal/vector/delete-documents")
    async def vector_delete_documents(body: VectorDeleteDocumentsRequest, request: Request):
        return await _require_vector_worker(request, "/delete-documents", body)

    @router.post("/internal/vector/get-by-metadata")
    async def vector_get_by_metadata(body: VectorGetByMetadataRequest, request: Request):
        return await _require_vector_worker(request, "/get-by-metadata", body)

    @router.post("/internal/vector/get-by-ids")
    async def vector_get_by_ids(body: VectorGetByIdsRequest, request: Request):
        return await _require_vector_worker(request, "/get-by-ids", body)

    @router.post("/internal/vector/create-collection")
    async def vector_create_collection(body: VectorCollectionRequest, request: Request):
        return await _require_vector_worker(request, "/create-collection", body)

    @router.post("/internal/vector/delete-collection")
    async def vector_delete_collection(body: VectorCollectionRequest, request: Request):
        return await _require_vector_worker(request, "/delete-collection", body)

    @router.post("/internal/vector/list-collections")
    async def vector_list_collections(body: VectorListCollectionsRequest, request: Request):
        return await _require_vector_worker(request, "/list-collections", body)

    @router.post("/internal/vector/quick-check")
    async def vector_quick_check(body: VectorQuickCheckRequest, request: Request):
        return await _require_vector_worker(request, "/quick-check", body)

    # ── Notification / interaction persistence for sandboxed CLIs ──
    #
    # ``animaworks-tool call_human`` runs inside execution sandboxes where
    # ``{data_dir}/run/`` is read-only.  These endpoints let the sandboxed
    # process delegate the run-state writes to the server so that Slack
    # thread replies and interactive approvals still route back correctly.

    @router.post("/internal/notification-mapping")
    async def internal_notification_mapping(body: NotificationMappingRequest):
        from core.notification.reply_routing import save_notification_mapping

        ok = await asyncio.to_thread(
            save_notification_mapping,
            body.ts,
            body.channel,
            body.anima_name,
            notification_text=body.notification_text,
            callback_id=body.callback_id,
        )
        return {"ok": ok}

    @router.post("/internal/interaction/create")
    async def internal_interaction_create(body: InteractionCreateRequest):
        from core.notification.interactive import get_interaction_router

        try:
            req = await get_interaction_router().create(
                body.anima_name,
                body.category,
                body.options,
                allowed_users=body.allowed_users,
                callback_id=body.callback_id or None,
            )
        except ValueError as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        return {"ok": True, "request": req.model_dump(mode="json")}

    @router.post("/internal/interaction/message-ts")
    async def internal_interaction_message_ts(body: InteractionMessageTsRequest):
        from core.notification.interactive import get_interaction_router

        await get_interaction_router().update_message_ts(
            body.callback_id,
            body.platform,
            body.ts,
        )
        return {"ok": True}

    @router.post("/internal/anima/create")
    async def internal_anima_create(body: AnimaCreateRequest):
        """Create an anima outside sandbox EROFS constraints.

        Sandboxed Mode C MCP subprocesses cannot write to animas/ root.
        They fall back here so create_from_md runs on the host server.
        """
        if not body.character_sheet_content and not body.character_sheet_path:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": ("Either character_sheet_content or character_sheet_path is required"),
                },
            )

        def _create() -> Path:
            from core.anima_factory import create_from_md
            from core.paths import get_animas_dir, get_data_dir

            md_path = Path(body.character_sheet_path) if body.character_sheet_path else None
            anima_dir = create_from_md(
                get_animas_dir(),
                md_path,
                name=body.name,
                content=body.character_sheet_content,
                supervisor=body.supervisor,
            )

            # Supervisor fallback (same as _handle_create_anima local path)
            status_path = anima_dir / "status.json"
            if status_path.exists() and body.calling_anima:
                try:
                    status_data = json.loads(status_path.read_text(encoding="utf-8"))
                    if not status_data.get("supervisor"):
                        status_data["supervisor"] = body.calling_anima
                        status_path.write_text(
                            json.dumps(status_data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                except (OSError, json.JSONDecodeError):
                    logger.warning(
                        "Failed to set fallback supervisor for '%s'",
                        anima_dir.name,
                        exc_info=True,
                    )

            try:
                from cli.commands.init_cmd import _register_anima_in_config

                _register_anima_in_config(get_data_dir(), anima_dir.name)
            except Exception:
                logger.warning(
                    "Failed to register anima '%s' in config.json",
                    anima_dir.name,
                    exc_info=True,
                )

            return anima_dir

        try:
            loop = asyncio.get_running_loop()
            anima_dir = await loop.run_in_executor(_native_executor, _create)
        except FileExistsError as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        except Exception as exc:
            logger.exception("internal anima create failed")
            return JSONResponse(status_code=500, content={"detail": str(exc)})

        return {"status": "ok", "anima_dir": str(anima_dir)}

    @router.post("/internal/delegate-task")
    async def internal_delegate_task(body: DelegateTaskPersistRequest):
        """Persist a delegated task outside sandbox EROFS constraints.

        Sandboxed ``delegate_task`` cannot append to another anima's
        ``task_queue.jsonl`` or write ``state/pending/``.  Mode C handlers
        fall back here so host-side TaskQueueManager / TaskBoard writes succeed.
        """
        from core.anima_factory import validate_anima_name
        from core.company import check_company_boundary
        from core.i18n import t
        from core.memory._io import atomic_write_text
        from core.memory.task_queue import TaskQueueManager
        from core.paths import get_animas_dir
        from core.tooling.handler_delegation import _record_taskboard_delegation

        if validate_anima_name(body.delegator) or validate_anima_name(body.target):
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid anima name"},
            )

        animas_dir = get_animas_dir()
        target_dir = animas_dir / body.target
        delegator_dir = animas_dir / body.delegator
        if not target_dir.is_dir() or not delegator_dir.is_dir():
            missing = body.target if not target_dir.is_dir() else body.delegator
            return JSONResponse(
                status_code=404,
                content={"detail": f"Anima directory not found: {missing}"},
            )

        boundary = check_company_boundary(
            body.delegator,
            body.target,
            animas_dir=animas_dir,
        )
        if boundary.cross_company:
            if boundary.resolved_via == "fail_closed":
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Company membership unreadable"},
                )
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        f"Cross-company delegation blocked: "
                        f"{body.delegator} -> {body.target} "
                        f"({boundary.display_name})"
                    ),
                },
            )

        def _persist() -> dict[str, str]:
            from datetime import UTC, datetime

            if body.persist_sub:
                TaskQueueManager(target_dir).add_task(
                    source="anima",
                    original_instruction=body.instruction,
                    assignee=body.target,
                    summary=body.summary,
                    deadline=body.deadline,
                    relay_chain=[body.delegator],
                    task_id=body.sub_task_id,
                )
            if body.persist_tracking:
                TaskQueueManager(delegator_dir).add_delegated_task(
                    original_instruction=body.instruction,
                    assignee=body.target,
                    summary=t("handler.delegation_summary", summary=body.summary),
                    deadline=body.deadline,
                    relay_chain=[body.delegator, body.target],
                    task_id=body.tracking_task_id,
                    meta={
                        "delegated_to": body.target,
                        "delegated_task_id": body.sub_task_id,
                    },
                )
            if body.persist_pending:
                task_desc = {
                    "task_type": "llm",
                    "task_id": body.sub_task_id,
                    "title": body.summary,
                    "description": body.instruction,
                    "context": "",
                    "acceptance_criteria": [],
                    "constraints": [],
                    "file_paths": [],
                    "submitted_by": body.delegator,
                    "submitted_at": datetime.now(UTC).isoformat(),
                    "reply_to": body.delegator,
                    "source": "delegation",
                    "working_directory": body.workspace,
                }
                pending_dir = target_dir / "state" / "pending"
                pending_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_text(
                    pending_dir / f"{body.sub_task_id}.json",
                    json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
                )
            try:
                _record_taskboard_delegation(
                    delegated_to=body.target,
                    delegated_task_id=body.sub_task_id,
                    delegator=body.delegator,
                    tracking_task_id=(body.tracking_task_id if body.persist_tracking else None),
                )
            except Exception:
                logger.warning(
                    "TaskBoard write failed in internal delegate-task; "
                    "queue entries remain authoritative",
                    exc_info=True,
                )
            return {
                "sub_task_id": body.sub_task_id,
                "tracking_task_id": body.tracking_task_id,
            }

        try:
            loop = asyncio.get_running_loop()
            ids = await loop.run_in_executor(_native_executor, _persist)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        except Exception as exc:
            logger.exception("internal delegate-task failed")
            return JSONResponse(status_code=500, content={"detail": str(exc)})

        return {
            "ok": True,
            "sub_task_id": ids["sub_task_id"],
            "tracking_task_id": ids["tracking_task_id"],
        }

    @router.post("/internal/vector/reset-store")
    async def vector_reset_store(body: VectorListCollectionsRequest, request: Request):
        # Forwarded to the worker so repair/quarantine can drop the worker's
        # cached (and possibly stale or corrupt) ChromaVectorStore. Without this
        # route the proxy returned 405 and the reset never reached the worker,
        # leaving stale handles and latched init-failures in place.
        return await _require_vector_worker(request, "/reset-store", body)

    return router
