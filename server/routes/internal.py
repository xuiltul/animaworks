from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import concurrent.futures
import json
import logging
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.anima.workspace_files import AGENTS_FILE, IDENTITY_FILE, SOUL_FILE
from server.events import emit, emit_notification

logger = logging.getLogger("animaworks.routes.internal")

# Owner-unavailable 503s tell vector clients how long to wait before their single retry.
_ROOT_RETRY_AFTER_MS = 250

_native_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="native-ops",
)


def _resolve_workspace_path(raw_path: str) -> Path:
    """Resolve an untrusted workspace path off the async route event loop."""
    return Path(raw_path).expanduser().resolve()


class MessageSentNotification(BaseModel):
    from_person: str
    to_person: str
    content: str = ""
    message_id: str = ""


class PhoneAlertRequest(BaseModel):
    anima: str
    subject: str
    body: str


class EmbedRequest(BaseModel):
    texts: list[str]
    purpose: Literal["document", "query"] = "document"
    priority: Literal["interactive", "bulk"] = "interactive"


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int | None = None


class VectorQueryRequest(BaseModel):
    anima_name: str
    collection: str
    embedding: list[float]
    top_k: int = 10
    filter_metadata: dict[str, str | int | float] | None = None


class VectorUpsertRequest(BaseModel):
    anima_name: str
    collection: str
    documents: list[dict[str, Any]]


class VectorUpdateMetadataRequest(BaseModel):
    anima_name: str
    collection: str
    ids: list[str]
    # Same shape as upsert metadata: the store serialises list values such as
    # fact ``entities``, so rejecting them here 422'd every fact refresh.
    metadatas: list[dict[str, Any]]


class VectorDeleteDocumentsRequest(BaseModel):
    anima_name: str
    collection: str
    ids: list[str]


class VectorGetByMetadataRequest(BaseModel):
    anima_name: str
    collection: str
    where: dict[str, str | int | float] = {}
    limit: int = 20


class VectorGetAllRequest(BaseModel):
    anima_name: str
    collection: str
    limit: int = 100_000


class VectorGetByIdsRequest(BaseModel):
    anima_name: str
    collection: str
    ids: list[str]


class VectorCollectionRequest(BaseModel):
    anima_name: str
    collection: str


class VectorListCollectionsRequest(BaseModel):
    anima_name: str


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


class CallHumanConfirmRequest(BaseModel):
    anima_name: str
    session_id: str
    sha: str = ""


class InteractionMessageTsRequest(BaseModel):
    callback_id: str
    platform: str = "slack"
    ts: str


class AnimaCreateRequest(BaseModel):
    character_sheet_content: str | None = None
    character_sheet_path: str | None = None
    name: str | None = None
    supervisor: str | None = None
    role: str | None = None
    calling_anima: str = ""  # supervisor fallback when status.json has none
    creation_type: Literal["character_sheet", "template", "blank"] = "character_sheet"
    template: str | None = None


class AnimaControlRequest(BaseModel):
    action: Literal["enable", "disable", "set_model", "set_background_model", "request_restart"]
    model: str = ""
    credential: str = ""
    background_model: str = ""
    background_credential: str = ""


class AnimaPromptSettingsRequest(BaseModel):
    setting: Literal["identity", "soul", "agents", "injection"]
    content: str
    mode: Literal["overwrite", "append"] = "overwrite"


class WorkspaceGrantRequest(BaseModel):
    alias: str
    path: str
    target_anima: str
    caller_anima: str = ""
    make_default: bool = True
    human_origin: bool = False


class CompanyAssignRequest(BaseModel):
    anima_names: list[str]
    company_name: str | None = None
    unassign: bool = False


class CompanySplitRequest(BaseModel):
    manifest_path: str
    execute: bool = False


class DelegateTaskPersistRequest(BaseModel):
    delegator: str  # source anima name
    target: str  # destination anima name
    instruction: str  # full delegation text
    summary: str
    sub_task_id: str  # client-assigned 12hex id
    tracking_task_id: str  # client-assigned 12hex id
    workspace: str = ""  # resolve_workspace absolute path string
    acceptance_criteria: list[str] = []  # verifiable acceptance criteria for pending JSON
    model: str = ""  # optional per-task LLM model override
    subject: str = ""  # optional deduplication key: target (e.g. github:owner/repo#123)
    version: str = ""  # optional deduplication key: revision (e.g. head SHA)
    execution_input: dict[str, Any] | None = None  # complete canonical input from sandbox callers
    attempt_identity: dict[str, str] | None = None


class InternalSendMessageRequest(BaseModel):
    message: dict[str, Any]  # full core.schemas.Message dump from the sandboxed sender


class InternalPostChannelRequest(BaseModel):
    from_anima: str
    channel: str
    text: str
    source: str = "anima"
    from_name: str | None = None


class InternalNotifyWebRequest(BaseModel):
    anima: str
    subject: str
    body: str
    priority: str = "normal"
    timestamp: str = ""


class UpdateTaskPersistRequest(BaseModel):
    anima_name: str
    task_id: str
    status: Literal["pending", "in_progress", "done", "cancelled"] | None = None
    meta: dict[str, Any] = {}
    summary: str | None = None
    attempt_identity: dict[str, str] | None = None
    resume: bool = False
    waiting_for: dict[str, Any] | None = None
    scope_note: str | None = None


class TaskBoardActionRequest(BaseModel):
    actor: str
    action: Literal["claim", "release", "done", "cancel", "note"]
    task_id: str
    ttl_seconds: int | None = None
    text: str | None = None


class SubmitTasksPersistRequest(BaseModel):
    anima_name: str
    tasks: list[dict[str, Any]]
    source: Literal["human", "anima"] = "anima"
    meta: dict[str, Any] = {}
    attempt_identity: dict[str, str] | None = None


def create_internal_router() -> APIRouter:
    from core.notification import CallHumanKeys
    from server.internal_auth import (
        ensure_self,
        ensure_self_or_descendant,
        internal_authz_denied,
        require_internal_caller,
    )

    router = APIRouter()
    internal = APIRouter(dependencies=[Depends(require_internal_caller)])
    _call_human_keys = CallHumanKeys()

    def _verified_internal_caller(request: Request):
        caller = getattr(request.state, "internal_caller", None)
        if caller is not None:
            return caller
        auth = getattr(request.app.state, "internal_auth", None)
        return auth.verify(request.headers.get("X-AnimaWorks-Internal-Auth")) if auth is not None else None

    def _require_settings_caller(request: Request):
        caller = _verified_internal_caller(request)
        if caller is None:
            return None, JSONResponse(
                status_code=401, content={"detail": "Internal settings authentication is required"}
            )
        if caller.kind not in {"anima", "operator"}:
            return None, JSONResponse(status_code=403, content={"detail": "Trusted internal caller required"})
        return caller, None

    @internal.post("/internal/company/assign")
    async def internal_company_assign(body: CompanyAssignRequest, request: Request):
        """Apply CLI company assignments through the root-owned status writer."""
        caller = _verified_internal_caller(request)
        if caller is None or caller.kind != "operator":
            return JSONResponse(status_code=403, content={"detail": "Operator authentication required"})
        from core.org.company import CompanyError, assign_animas
        from core.paths import get_data_dir

        try:
            lines = await asyncio.to_thread(
                assign_animas,
                body.anima_names,
                company_name=body.company_name,
                unassign=body.unassign,
                data_dir=get_data_dir(),
            )
        except CompanyError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        except Exception as exc:
            logger.exception("internal company assignment failed")
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        return {"lines": lines}

    @internal.post("/internal/company/split")
    async def internal_company_split(body: CompanySplitRequest, request: Request):
        """Run a CLI company split on root when it mutates status/settings."""
        caller = _verified_internal_caller(request)
        if caller is None or caller.kind != "operator":
            return JSONResponse(status_code=403, content={"detail": "Operator authentication required"})
        from core.org.company import CompanyError, SplitExecutionError, split_companies
        from core.paths import get_data_dir

        try:
            lines = await asyncio.to_thread(
                split_companies,
                body.manifest_path,
                execute=body.execute,
                data_dir=get_data_dir(),
            )
        except SplitExecutionError as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc), "completed_lines": exc.completed_lines})
        except CompanyError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        except Exception as exc:
            logger.exception("internal company split failed")
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        return {"lines": lines}

    @internal.get("/internal/company/boundary")
    async def internal_company_boundary(from_anima: str, to_anima: str, request: Request):
        """Resolve company membership on the host for sandboxed handlers."""
        denied = ensure_self(getattr(request.state, "internal_caller", None), from_anima, path=request.url.path)
        if denied is not None:
            return denied

        from core.anima.factory import validate_anima_name
        from core.config.models import read_anima_company_checked
        from core.org.company import get_company_display_name
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

    @internal.post("/internal/message-sent")
    async def internal_message_sent(body: MessageSentNotification, request: Request):
        """Notify the server that a message was sent via CLI.

        Triggers WebSocket broadcast and updates reply tracking so that
        selective archival (Fix 2) works for CLI-sent messages too.
        """
        denied = ensure_self(getattr(request.state, "internal_caller", None), body.from_person, path=request.url.path)
        if denied is not None:
            return denied

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

    @internal.post("/internal/embed")
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

        from core.memory.rag.embedding import thread_safe_encode

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            _native_executor,
            partial(thread_safe_encode, body.texts, purpose=body.purpose, priority=body.priority),
        )
        return {"embeddings": embeddings}

    # ── Cross-encoder rerank endpoint ─────────────────────────────

    @internal.post("/internal/rerank")
    async def internal_rerank(body: RerankRequest):
        """Centralized cross-encoder reranking for child processes.

        Child processes call this endpoint via HTTP instead of loading
        the CrossEncoder model on their own, eliminating per-MCP-subprocess
        model loads (~130 MB each).
        """
        if len(body.documents) > 1000:
            return JSONResponse(
                status_code=400,
                content={"detail": "Max 1000 documents per request"},
            )
        if not body.documents:
            return {"scores": []}

        from functools import partial

        from core.config import load_config
        from core.memory.retrieval.reranker import get_reranker

        started = perf_counter()
        init_started = perf_counter()
        model_name = load_config().rag.cross_encoder_model
        reranker = get_reranker(model_name) if model_name else get_reranker()
        init_elapsed = perf_counter() - init_started
        loop = asyncio.get_running_loop()
        score_started = perf_counter()
        scores = await loop.run_in_executor(
            _native_executor,
            partial(reranker.score_sync, body.query, body.documents),
        )
        logger.info(
            "Internal rerank complete: query_chars=%d documents=%d elapsed=%.3fs init=%.3fs score=%.3fs",
            len(body.query),
            len(body.documents),
            perf_counter() - started,
            init_elapsed,
            perf_counter() - score_started,
        )
        if scores is None:
            # Explicit failure — never collapse to empty success (caller must
            # treat as unavailable and keep original order).
            return JSONResponse(
                status_code=503,
                content={"detail": "Reranker unavailable"},
            )

        if body.top_k is not None and body.top_k >= 0:
            # Optional ranked view; primary contract remains full scores array.
            indexed = sorted(
                enumerate(scores),
                key=lambda pair: pair[1],
                reverse=True,
            )[: body.top_k]
            return {
                "scores": scores,
                "results": [{"index": i, "score": s} for i, s in indexed],
            }
        return {"scores": scores}

    # ── Vector store endpoints (ChromaDB process separation) ───────

    def _body_payload(body: BaseModel) -> dict[str, Any]:
        if hasattr(body, "model_dump"):
            return body.model_dump()
        return body.dict()

    async def _forward_to_root(request: Request, path: str, body: BaseModel) -> dict[str, Any] | JSONResponse:
        from core.anima.factory import validate_anima_name
        from core.i18n import t
        from core.memory.rag.vector_ops import UnsupportedVectorPath, to_owner_interaction

        anima_name = getattr(body, "anima_name", "")
        caller = getattr(request.state, "internal_caller", None)
        if isinstance(anima_name, str) and anima_name:
            denied = ensure_self(caller, anima_name, path=request.url.path)
            if denied is not None:
                return denied
        if not isinstance(anima_name, str) or validate_anima_name(anima_name) is not None:
            return JSONResponse(status_code=422, content={"detail": t("rag.invalid_anima_name")})

        try:
            method, params = to_owner_interaction(path, _body_payload(body))
        except UnsupportedVectorPath:
            return JSONResponse(status_code=409, content={"detail": t("rag.unsupported_vector_path")})

        supervisor = getattr(request.app.state, "supervisor", None)
        if supervisor is None:
            return JSONResponse(
                status_code=503,
                content={"detail": t("rag.root_unavailable"), "retry_after_ms": _ROOT_RETRY_AFTER_MS},
                headers={"Retry-After": "1"},
            )
        try:
            result = await supervisor.send_request(
                anima_name,
                "memory",
                {"method": method, "params": params},
                timeout=120.0,
            )
        except Exception:
            logger.warning("Root memory proxy unavailable: anima=%s method=%s", anima_name, method, exc_info=True)
            return JSONResponse(
                status_code=503,
                content={"detail": t("rag.root_unavailable"), "retry_after_ms": _ROOT_RETRY_AFTER_MS},
                headers={"Retry-After": "1"},
            )
        if not isinstance(result, dict) or result.get("ok") is False:
            return JSONResponse(
                status_code=503,
                content={"detail": t("rag.root_operation_failed"), "retry_after_ms": _ROOT_RETRY_AFTER_MS},
                headers={"Retry-After": "1"},
            )
        return result

    @internal.post("/internal/vector/query")
    async def vector_query(body: VectorQueryRequest, request: Request):
        return await _forward_to_root(request, "/query", body)

    @internal.post("/internal/vector/upsert")
    async def vector_upsert(body: VectorUpsertRequest, request: Request):
        return await _forward_to_root(request, "/upsert", body)

    @internal.post("/internal/vector/update-metadata")
    async def vector_update_metadata(body: VectorUpdateMetadataRequest, request: Request):
        return await _forward_to_root(request, "/update-metadata", body)

    @internal.post("/internal/vector/delete-documents")
    async def vector_delete_documents(body: VectorDeleteDocumentsRequest, request: Request):
        return await _forward_to_root(request, "/delete-documents", body)

    @internal.post("/internal/vector/get-by-metadata")
    async def vector_get_by_metadata(body: VectorGetByMetadataRequest, request: Request):
        return await _forward_to_root(request, "/get-by-metadata", body)

    @internal.post("/internal/vector/get-all")
    async def vector_get_all(body: VectorGetAllRequest, request: Request):
        return await _forward_to_root(request, "/get-all", body)

    @internal.post("/internal/vector/count")
    async def vector_count(body: VectorCollectionRequest, request: Request):
        return await _forward_to_root(request, "/count", body)

    @internal.post("/internal/vector/get-by-ids")
    async def vector_get_by_ids(body: VectorGetByIdsRequest, request: Request):
        return await _forward_to_root(request, "/get-by-ids", body)

    @internal.post("/internal/vector/create-collection")
    async def vector_create_collection(body: VectorCollectionRequest, request: Request):
        return await _forward_to_root(request, "/create-collection", body)

    @internal.post("/internal/vector/delete-collection")
    async def vector_delete_collection(body: VectorCollectionRequest, request: Request):
        return await _forward_to_root(request, "/delete-collection", body)

    @internal.post("/internal/vector/list-collections")
    async def vector_list_collections(body: VectorListCollectionsRequest, request: Request):
        return await _forward_to_root(request, "/list-collections", body)

    # ── Notification / interaction persistence for sandboxed CLIs ──
    #
    # ``animaworks-tool call_human`` runs inside execution sandboxes where
    # ``{data_dir}/run/`` is read-only.  These endpoints let the sandboxed
    # process delegate the run-state writes to the server so that Slack
    # thread replies and interactive approvals still route back correctly.

    @internal.post("/internal/notification-mapping")
    async def internal_notification_mapping(body: NotificationMappingRequest, request: Request):
        denied = ensure_self(getattr(request.state, "internal_caller", None), body.anima_name, path=request.url.path)
        if denied is not None:
            return denied

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

    @internal.post("/internal/phone/alert")
    async def internal_phone_alert(body: PhoneAlertRequest, request: Request):
        """Start an urgent phone alert for the configured Anima."""
        denied = ensure_self(getattr(request.state, "internal_caller", None), body.anima, path=request.url.path)
        if denied is not None:
            return denied

        from core.config import load_config
        from core.phone.alert import start_alert

        phone_config = load_config().phone
        if not phone_config.enabled:
            return {"status": "skipped", "reason": "phone channel is disabled"}
        if body.anima != phone_config.anima:
            return {"status": "skipped", "reason": "Anima is not the configured phone target"}
        try:
            await start_alert(body.anima, body.subject, body.body, phone_config=phone_config)
        except Exception:
            logger.warning("Unable to queue phone alert for anima=%s", body.anima)
            return JSONResponse(
                status_code=503,
                content={"status": "error", "reason": "phone alert could not be started"},
            )
        return {"status": "calling"}

    @internal.post("/internal/call-human/confirm")
    async def internal_call_human_confirm(body: CallHumanConfirmRequest, request: Request):
        """Check the CLI ``call_human`` confirmation key; keys live in server memory only."""
        denied = ensure_self(getattr(request.state, "internal_caller", None), body.anima_name, path=request.url.path)
        if denied is not None:
            return denied
        issued_key = _call_human_keys.check(body.anima_name, body.session_id, body.sha)
        return {"ok": issued_key is None, "sha": issued_key or ""}

    @internal.post("/internal/interaction/create")
    async def internal_interaction_create(body: InteractionCreateRequest, request: Request):
        denied = ensure_self(getattr(request.state, "internal_caller", None), body.anima_name, path=request.url.path)
        if denied is not None:
            return denied

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

    @internal.post("/internal/interaction/message-ts")
    async def internal_interaction_message_ts(body: InteractionMessageTsRequest):
        from core.notification.interactive import get_interaction_router

        await get_interaction_router().update_message_ts(
            body.callback_id,
            body.platform,
            body.ts,
        )
        return {"ok": True}

    @internal.post("/internal/anima/create")
    async def internal_anima_create(body: AnimaCreateRequest, request: Request):
        """Create an anima outside sandbox EROFS constraints.

        Sandboxed Mode C MCP subprocesses cannot write to animas/ root.
        They fall back here so create_from_md runs on the host server.
        """
        caller = _verified_internal_caller(request)
        denied = ensure_self(caller, body.calling_anima, path=request.url.path)
        if denied is not None:
            return denied
        if caller is not None and caller.kind == "anima":
            from core.anima.skills_check import has_newstaff_skill
            from core.paths import get_animas_dir

            if not has_newstaff_skill(get_animas_dir() / caller.name):
                denied = internal_authz_denied(
                    caller,
                    body.calling_anima,
                    "server.internal_newstaff_required",
                    path=request.url.path,
                )
                if denied is not None:
                    return denied
            if body.supervisor is not None:
                denied = ensure_self_or_descendant(caller, body.supervisor, path=request.url.path)
                if denied is not None:
                    return denied

        if body.name is not None:
            from core.anima.factory import validate_anima_name

            name_error = validate_anima_name(body.name)
            if name_error:
                return JSONResponse(status_code=422, content={"detail": name_error})
        if body.creation_type == "blank" and not body.name:
            return JSONResponse(status_code=422, content={"detail": "name is required for blank creation"})
        if body.creation_type == "template":
            if not body.template or Path(body.template).name != body.template or ".." in body.template:
                return JSONResponse(status_code=422, content={"detail": "valid template is required"})
        if (
            body.creation_type == "character_sheet"
            and not body.character_sheet_content
            and not body.character_sheet_path
        ):
            return JSONResponse(
                status_code=422,
                content={"detail": "Either character_sheet_content or character_sheet_path is required"},
            )

        def _create() -> Path:
            from core.anima.factory import create_blank, create_from_md, create_from_template
            from core.config import register_anima_in_config
            from core.paths import get_animas_dir, get_data_dir

            md_path = Path(body.character_sheet_path) if body.character_sheet_path else None
            if body.creation_type == "blank":
                anima_dir = create_blank(get_animas_dir(), body.name or "")
            elif body.creation_type == "template":
                anima_dir = create_from_template(get_animas_dir(), body.template or "", anima_name=body.name)
            else:
                anima_dir = create_from_md(
                    get_animas_dir(),
                    md_path,
                    name=body.name,
                    content=body.character_sheet_content,
                    supervisor=body.supervisor,
                    role=body.role,
                )

            # Supervisor fallback (same as _handle_create_anima local path)
            status_path = anima_dir / "status.json"
            if status_path.exists() and body.calling_anima:
                from core.anima.settings_store import update_status

                def set_fallback_supervisor(status_data: dict[str, Any]) -> None:
                    if not status_data.get("supervisor"):
                        status_data["supervisor"] = body.calling_anima

                try:
                    update_status(anima_dir, set_fallback_supervisor)
                except (OSError, ValueError, json.JSONDecodeError):
                    logger.warning(
                        "Failed to set fallback supervisor for '%s'",
                        anima_dir.name,
                        exc_info=True,
                    )

            try:
                register_anima_in_config(get_data_dir(), anima_dir.name)
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

    @internal.post("/internal/animas/{target}/control")
    async def internal_anima_control(target: str, body: AnimaControlRequest, request: Request):
        """Persist subordinate control settings after verifying the token owner."""
        caller, denied = _require_settings_caller(request)
        if denied is not None:
            return denied

        from core.anima.factory import validate_anima_name
        from core.config.models import load_config
        from core.org.hierarchy import descendants_of
        from core.paths import get_animas_dir

        if validate_anima_name(target):
            return JSONResponse(status_code=400, content={"detail": "Invalid anima name"})
        try:
            config = load_config()
        except Exception as exc:
            logger.warning("Unable to load org hierarchy for anima control", exc_info=True)
            return JSONResponse(status_code=503, content={"detail": str(exc)})
        if caller.kind == "anima":
            descendants = descendants_of(config.animas, caller.name)
            if target == caller.name or target not in descendants:
                logger.warning("internal_anima_control_denied caller=%s target=%s", caller.name, target)
                return JSONResponse(status_code=403, content={"detail": "Target must be a descendant of the caller"})

        target_dir = get_animas_dir() / target
        if not target_dir.is_dir() or not (target_dir / IDENTITY_FILE).is_file():
            return JSONResponse(status_code=404, content={"detail": f"Anima not found: {target}"})

        def _persist() -> dict[str, Any]:
            from core.anima.settings_store import update_status

            changed = False
            result: dict[str, Any] = {}
            if body.action in {"enable", "disable"}:
                enabled = body.action == "enable"

                def set_enabled(status: dict[str, Any]) -> None:
                    nonlocal changed
                    if status.get("enabled", True) != enabled:
                        status["enabled"] = enabled
                        changed = True

                update_status(target_dir, set_enabled)
            elif body.action == "request_restart":

                def request_restart(status: dict[str, Any]) -> None:
                    nonlocal changed
                    changed = not bool(status.get("restart_requested"))
                    status["restart_requested"] = True

                update_status(target_dir, request_restart)
                changed = True
            elif body.action == "set_model":
                if not body.model.strip():
                    raise ValueError("model is required")
                from core.config.model_config import smart_update_model

                result = smart_update_model(
                    target_dir,
                    model=body.model.strip(),
                    credential=body.credential.strip() or None,
                )
                changed = True
            elif body.action == "set_background_model":
                from core.config.model_config import update_status_model

                update_status_model(
                    target_dir,
                    background_model=body.background_model.strip(),
                    background_credential=body.background_credential.strip(),
                )
                result = {
                    "background_model": body.background_model.strip(),
                    "background_credential": body.background_credential.strip(),
                }
                changed = True
            return {"changed": changed, "result": result}

        try:
            persisted = await asyncio.to_thread(_persist)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"detail": str(exc)})
        except (OSError, json.JSONDecodeError) as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        except Exception as exc:
            logger.exception("internal anima control failed for %s", target)
            return JSONResponse(status_code=500, content={"detail": str(exc)})

        if body.action in {"set_model", "set_background_model"}:
            supervisor = getattr(request.app.state, "supervisor", None)
            if supervisor is not None and target in getattr(supervisor, "processes", {}):
                try:
                    await supervisor.send_request(target, "reload_config", {}, timeout=10.0)
                except Exception:
                    logger.info("Model reload deferred until next start for anima=%s", target, exc_info=True)
        return {
            "ok": True,
            "action": body.action,
            "target_anima": target,
            "changed": persisted["changed"],
            "result": persisted["result"],
        }

    @internal.post("/internal/animas/{target}/prompt-settings")
    async def internal_update_anima_prompt_setting(
        target: str,
        body: AnimaPromptSettingsRequest,
        request: Request,
    ):
        """Apply an authorized IDENTITY/SOUL/AGENTS change through the root owner."""
        caller, denied = _require_settings_caller(request)
        if denied is not None:
            return denied
        from core.anima.factory import validate_anima_name
        from core.paths import get_animas_dir

        if validate_anima_name(target):
            return JSONResponse(status_code=400, content={"detail": "Invalid anima name"})
        target_dir = get_animas_dir() / target
        if not target_dir.is_dir() or not (target_dir / IDENTITY_FILE).is_file():
            return JSONResponse(status_code=404, content={"detail": f"Anima not found: {target}"})

        if caller.kind == "anima":
            if target == caller.name and body.setting in {"injection", "agents", "soul"}:
                # Animas maintain their own AGENTS.md / SOUL.md through the
                # root owner (the former identity/injection split).
                pass
            elif target == caller.name:
                from core.anima.bootstrap_state import get_bootstrap_status

                bootstrap = get_bootstrap_status(target_dir)
                if not (bootstrap.get("needs_user_input") or bootstrap.get("needs_repair")):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Anima prompt settings are root-owned outside bootstrap/repair"},
                    )
            elif body.setting in {"injection", "agents"}:
                from core.config.models import load_config
                from core.org.hierarchy import descendants_of

                config = load_config()
                if target not in descendants_of(config.animas, caller.name):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Only an ancestor may request subordinate injection changes"},
                    )
            else:
                return JSONResponse(status_code=403, content={"detail": "Anima identity is root-owned"})

        filename = {
            "identity": IDENTITY_FILE,
            "soul": SOUL_FILE,
            "agents": AGENTS_FILE,
            "injection": AGENTS_FILE,
        }[body.setting]
        path = target_dir / filename
        content = body.content
        if body.mode == "append":
            try:
                existing = await asyncio.to_thread(path.read_text, encoding="utf-8") if path.is_file() else ""
            except OSError as exc:
                return JSONResponse(status_code=409, content={"detail": str(exc)})
            content = existing + content
        try:
            from core.anima.settings_store import write_agents, write_identity, write_soul

            writer = {
                "identity": write_identity,
                "soul": write_soul,
                "agents": write_agents,
                "injection": write_agents,
            }[body.setting]
            await asyncio.to_thread(writer, target_dir, content)
        except (OSError, ValueError) as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        except Exception as exc:
            logger.exception("internal prompt setting update failed for %s/%s", target, body.setting)
            return JSONResponse(status_code=500, content={"detail": str(exc)})
        return {"ok": True, "target_anima": target, "setting": body.setting, "length": len(content)}

    @internal.post("/internal/workspace/grant")
    async def internal_workspace_grant(body: WorkspaceGrantRequest, request: Request):
        """Apply a human-origin workspace grant through root-owned writers."""
        caller, denied = _require_settings_caller(request)
        if denied is not None:
            return denied
        if not body.human_origin:
            return JSONResponse(status_code=403, content={"detail": "Human-origin instruction required"})
        if not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$", body.alias):
            return JSONResponse(status_code=400, content={"detail": "Invalid workspace alias"})

        from core.config.file_access_policy import effective_write_roots
        from core.config.models import load_config, load_permissions
        from core.org.hierarchy import descendants_of
        from core.org.workspace import qualified_alias
        from core.paths import get_animas_dir

        config = load_config()
        caller_name = caller.name if caller.kind == "anima" else body.caller_anima
        if caller.kind == "operator" and not caller_name:
            return JSONResponse(status_code=400, content={"detail": "caller_anima is required for operator callers"})
        caller_config = config.animas.get(caller_name)
        if caller_config is None or caller_config.supervisor is not None:
            return JSONResponse(status_code=403, content={"detail": "Only top-level Animas can grant workspaces"})
        target = body.target_anima
        target_config = config.animas.get(target)
        if target_config is None:
            return JSONResponse(status_code=404, content={"detail": f"Target Anima not found: {target}"})
        if target != caller_name and target not in descendants_of(config.animas, caller_name):
            return JSONResponse(
                status_code=403,
                content={"detail": "Top-level Animas can grant workspaces only to themselves or descendants"},
            )

        try:
            workspace_path = await asyncio.to_thread(_resolve_workspace_path, body.path)
        except (OSError, RuntimeError) as exc:
            return JSONResponse(status_code=400, content={"detail": f"Failed to resolve path: {exc}"})
        if not workspace_path.is_dir():
            return JSONResponse(
                status_code=400, content={"detail": f"Workspace path is not an existing directory: {workspace_path}"}
            )
        if workspace_path == Path("/"):
            return JSONResponse(status_code=403, content={"detail": "Filesystem root cannot be granted"})
        try:
            home = Path.home().resolve()
            if workspace_path == home:
                return JSONResponse(status_code=403, content={"detail": "Home directory root cannot be granted"})
        except OSError:
            pass
        for root in (
            Path("/etc"),
            Path("/proc"),
            Path("/dev"),
            Path("/sys"),
            Path("/run"),
            Path("/boot"),
            Path("/root"),
        ):
            try:
                protected_root = root.resolve()
            except OSError:
                protected_root = root
            if workspace_path == protected_root or workspace_path.is_relative_to(protected_root):
                return JSONResponse(
                    status_code=403, content={"detail": f"Protected system directory: {workspace_path}"}
                )
        animas_root = get_animas_dir().resolve()
        if workspace_path == animas_root or workspace_path.is_relative_to(animas_root):
            return JSONResponse(status_code=403, content={"detail": "Anima home directories cannot be workspaces"})

        if not body.path.strip():
            return JSONResponse(status_code=400, content={"detail": "path is required"})
        target_dir = (animas_root / target).resolve()
        if not target_dir.is_dir() or not (target_dir / IDENTITY_FILE).is_file():
            return JSONResponse(status_code=404, content={"detail": f"Target Anima directory not found: {target}"})

        alias = body.alias.strip()
        qualified = qualified_alias(alias, str(workspace_path))
        try:
            from core.anima.settings_store import update_config, update_status, write_permissions

            update_config(lambda current: current.workspaces.__setitem__(alias, str(workspace_path)))
            permissions = load_permissions(target_dir)
            current_roots = effective_write_roots(target_dir, permissions.file_roots)
            permissions_unrestricted = permissions.file_roots == ["/"]
            permissions_changed = False
            if not permissions_unrestricted:
                allowed = any(workspace_path == root or workspace_path.is_relative_to(root) for root in current_roots)
                if not allowed:
                    effective_write_roots(target_dir, [str(workspace_path)])
                    permissions.file_roots.append(str(workspace_path))
                    write_permissions(target_dir, permissions)
                    permissions_changed = True

            status_changed = False
            if body.make_default:

                def set_workspace(status: dict[str, Any]) -> None:
                    nonlocal status_changed
                    if status.get("default_workspace") != qualified:
                        status["default_workspace"] = qualified
                        status_changed = True

                update_status(target_dir, set_workspace)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("internal workspace grant failed for %s", target, exc_info=True)
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        except Exception as exc:
            logger.exception("internal workspace grant failed for %s", target)
            return JSONResponse(status_code=500, content={"detail": str(exc)})

        return {
            "status": "ok",
            "qualified_alias": qualified,
            "alias": alias,
            "path": str(workspace_path),
            "target_anima": target,
            "global_workspace_registered": True,
            "permissions_changed": permissions_changed,
            "permissions_unrestricted": permissions_unrestricted,
            "default_workspace_changed": status_changed,
            "effective_next_codex_run": True,
        }

    @internal.post("/internal/settings/anima-icon-template")
    async def internal_persist_anima_icon_template(request: Request):
        """Persist icon-template defaults on behalf of an authenticated worker."""
        caller, denied = _require_settings_caller(request)
        if denied is not None:
            return denied
        try:
            from core.integrations._anima_icon_url import persist_anima_icon_path_template

            await asyncio.to_thread(persist_anima_icon_path_template)
        except Exception as exc:
            logger.exception("internal icon-template persistence failed for caller=%s", caller.name)
            return JSONResponse(status_code=500, content={"detail": str(exc)})
        return {"ok": True}

    @internal.post("/internal/send-message")
    async def internal_send_message(body: InternalSendMessageRequest, request: Request):
        """Persist a DM outside sandbox EROFS constraints.

        Sandboxed Messenger.send cannot write shared/inbox (write-access
        charter: only company shared + work dirs are writable), so it falls
        back here and the host writes the exact message file.
        """
        from core.anima.factory import validate_anima_name
        from core.paths import get_shared_dir
        from core.schemas import Message

        try:
            msg = Message(**body.message)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": f"Invalid message: {exc}"})
        denied = ensure_self(getattr(request.state, "internal_caller", None), msg.from_person, path=request.url.path)
        if denied is not None:
            return denied
        if validate_anima_name(msg.from_person) or not re.fullmatch(r"[A-Za-z0-9_-]+", msg.to_person):
            return JSONResponse(status_code=400, content={"detail": "Invalid sender/recipient name"})
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", msg.id):
            return JSONResponse(status_code=400, content={"detail": "Invalid message ID"})

        target_dir = get_shared_dir() / "inbox" / msg.to_person
        target_dir.mkdir(parents=True, exist_ok=True)
        from core.platform.atomic_io import atomic_write_text

        if not (target_dir / "processed" / f"{msg.id}.json").exists():
            atomic_write_text(target_dir / f"{msg.id}.json", msg.model_dump_json(indent=2))
        logger.info("internal send-message: %s -> %s (%s)", msg.from_person, msg.to_person, msg.id)
        return {"ok": True, "message_id": msg.id, "thread_id": msg.thread_id}

    @internal.post("/internal/post-channel")
    async def internal_post_channel(body: InternalPostChannelRequest, request: Request):
        """Append a channel post outside sandbox EROFS constraints."""
        denied = ensure_self(getattr(request.state, "internal_caller", None), body.from_anima, path=request.url.path)
        if denied is not None:
            return denied
        from core.anima.factory import validate_anima_name
        from core.exceptions import ChannelAccessDeniedError, ChannelNotFoundError
        from core.messaging.messenger import Messenger
        from core.paths import get_shared_dir

        if validate_anima_name(body.from_anima):
            return JSONResponse(status_code=400, content={"detail": "Invalid anima name"})
        messenger = Messenger(get_shared_dir(), body.from_anima)
        try:
            messenger.post_channel(
                body.channel,
                body.text,
                source=body.source,
                from_name=body.from_name,
            )
        except ChannelNotFoundError as exc:
            return JSONResponse(status_code=404, content={"detail": str(exc)})
        except ChannelAccessDeniedError as exc:
            return JSONResponse(status_code=403, content={"detail": str(exc)})
        logger.info("internal post-channel: %s -> #%s", body.from_anima, body.channel)
        return {"ok": True}

    @internal.post("/internal/notify-web")
    async def internal_notify_web(body: InternalNotifyWebRequest, request: Request):
        """Push a call_human notification into connected Web UI clients."""
        denied = ensure_self(getattr(request.state, "internal_caller", None), body.anima, path=request.url.path)
        if denied is not None:
            return denied
        from core.anima.factory import validate_anima_name

        if validate_anima_name(body.anima):
            return JSONResponse(status_code=400, content={"detail": "Invalid anima name"})
        await emit_notification(request, body.model_dump())
        logger.info("internal notify-web: %s (%s)", body.anima, body.priority)
        return {"ok": True}

    @internal.get("/internal/tasks")
    async def internal_tasks(
        anima_name: str,
        request: Request,
        include_archived: bool = False,
        task_id: str | None = None,
    ):
        """Read a task snapshot for workers without direct database access."""
        denied = ensure_self_or_descendant(
            getattr(request.state, "internal_caller", None), anima_name, path=request.url.path
        )
        if denied is not None:
            return denied
        from core.anima.factory import validate_anima_name
        from core.paths import get_animas_dir
        from core.tasks.queue import TaskQueueManager

        if validate_anima_name(anima_name):
            return JSONResponse(status_code=400, content={"detail": "Invalid anima name"})
        anima_dir = get_animas_dir() / anima_name
        if not anima_dir.is_dir():
            return JSONResponse(status_code=404, content={"detail": "Anima directory not found"})

        def _read():
            store = TaskQueueManager(anima_dir, read_only=True).store
            if not store.has_database:
                return {"tasks": [], "input_ids": []}
            with store.reader():
                if task_id is not None:
                    entry = store.get(anima_name, task_id)
                    return {"tasks": [entry.model_dump(mode="json")] if entry else [], "input_ids": []}
                return {
                    "tasks": [
                        entry.model_dump(mode="json")
                        for entry in store.read(anima_name, archived=include_archived).values()
                    ],
                    "input_ids": sorted(store.executable_ids(anima_name)),
                }

        return await asyncio.get_running_loop().run_in_executor(_native_executor, _read)

    @internal.post("/internal/submit-tasks")
    async def internal_submit_tasks(body: SubmitTasksPersistRequest, request: Request):
        """Publish a complete batch on the host; no sandbox DB grant is needed."""
        denied = ensure_self(getattr(request.state, "internal_caller", None), body.anima_name, path=request.url.path)
        if denied is not None:
            return denied
        from core.anima.factory import validate_anima_name
        from core.paths import get_animas_dir
        from core.tasks.dispatch import publish_tasks

        if validate_anima_name(body.anima_name):
            return JSONResponse(status_code=400, content={"detail": "Invalid anima name"})
        anima_dir = get_animas_dir() / body.anima_name
        if not anima_dir.is_dir():
            return JSONResponse(status_code=404, content={"detail": "Anima directory not found"})

        def _publish():
            from core.tasks.board.tasks import attempt_scope

            with attempt_scope(body.attempt_identity):
                return publish_tasks(anima_dir, body.tasks, source=body.source, meta=body.meta, host_fallback=False)

        try:
            entries = await asyncio.get_running_loop().run_in_executor(_native_executor, _publish)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        except Exception as exc:
            logger.exception("internal submit-tasks failed")
            return JSONResponse(status_code=500, content={"detail": str(exc)})
        return {"ok": True, "tasks": [entry.model_dump(mode="json") for entry in entries]}

    @internal.post("/internal/delegate-task")
    async def internal_delegate_task(body: DelegateTaskPersistRequest, request: Request):
        """Persist a delegated task outside sandbox EROFS constraints.

        Sandboxed ``delegate_task`` cannot write the shared task database.
        Mode C handlers use this endpoint for atomic host-side publication.
        """
        caller = getattr(request.state, "internal_caller", None)
        denied = ensure_self(caller, body.delegator, path=request.url.path)
        if denied is not None:
            return denied
        if caller is not None and caller.kind == "anima":
            from core.config.io import load_config
            from core.org.hierarchy import is_direct_subordinate

            if not is_direct_subordinate(load_config().animas, body.delegator, body.target):
                denied = internal_authz_denied(
                    caller,
                    body.target,
                    "server.internal_not_subordinate",
                    path=request.url.path,
                )
                if denied is not None:
                    return denied

        from core.anima.factory import validate_anima_name
        from core.org.company import check_company_boundary
        from core.paths import get_animas_dir

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
                        f"Cross-company delegation blocked: {body.delegator} -> {body.target} ({boundary.display_name})"
                    ),
                },
            )

        def _persist() -> dict[str, Any]:
            from datetime import UTC, datetime

            from core.tasks.board.tasks import attempt_scope
            from core.tasks.dispatch import publish_delegation

            payload = {
                "task_type": "llm",
                "task_id": body.sub_task_id,
                "title": body.summary,
                "description": body.instruction,
                "context": "",
                "acceptance_criteria": list(body.acceptance_criteria or []),
                "constraints": [],
                "file_paths": [],
                "submitted_by": body.delegator,
                "submitted_at": datetime.now(UTC).isoformat(),
                "reply_to": body.delegator,
                "source": "delegation",
                "working_directory": body.workspace,
                "model": body.model,
                "subject": body.subject,
                "version": body.version,
            }
            if body.execution_input is not None:
                payload = dict(body.execution_input)
                if payload.get("task_id") != body.sub_task_id:
                    raise ValueError("Delegated execution input task_id must match sub_task_id")
                if payload.get("submitted_by", body.delegator) != body.delegator:
                    raise ValueError("Delegated execution input must retain its delegator")
                payload.setdefault("submitted_by", body.delegator)
                if body.subject and "subject" not in payload:
                    payload["subject"] = body.subject
                if body.version and "version" not in payload:
                    payload["version"] = body.version
            with attempt_scope(body.attempt_identity):
                published = publish_delegation(
                    target_dir,
                    payload,
                    delegator=body.delegator,
                    tracking_task_id=body.tracking_task_id,
                    host_fallback=False,
                )
            return {
                "task_id": published.task_id,
                "status": published.status,
                "deduplicated": published.deduplicated,
                "via_host": published.via_host,
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
            "sub_task_id": ids["task_id"],
            "tracking_task_id": ids["tracking_task_id"],
            "task_id": ids["task_id"],
            "status": ids["status"],
            "deduplicated": ids["deduplicated"],
            "via_host": ids["via_host"],
        }

    @internal.post("/internal/task-board-action")
    async def internal_task_board_action(body: TaskBoardActionRequest, request: Request):
        """Run a lease-guarded task board write for a sandboxed anima CLI."""
        caller = getattr(request.state, "internal_caller", None)
        if body.actor == "human":
            if caller is not None and caller.kind == "anima":
                denied = internal_authz_denied(
                    caller,
                    body.actor,
                    "server.internal_identity_mismatch",
                    path=request.url.path,
                )
                if denied is not None:
                    return denied
        else:
            denied = ensure_self(caller, body.actor, path=request.url.path)
            if denied is not None:
                return denied

        from core.anima.factory import validate_anima_name
        from core.tasks.board.board_actions import BoardActionError, run_board_action

        if body.actor != "human" and validate_anima_name(body.actor):
            return JSONResponse(status_code=400, content={"detail": "Invalid actor"})

        def _run() -> dict[str, Any]:
            try:
                result = run_board_action(**body.model_dump())
            except BoardActionError as exc:
                return {"ok": False, "error": exc.message, "exit_code": exc.exit_code, "payload": exc.payload}
            return {"ok": True, "result": result}

        try:
            return await asyncio.get_running_loop().run_in_executor(_native_executor, _run)
        except Exception as exc:
            logger.exception("internal task-board-action failed")
            return JSONResponse(status_code=500, content={"detail": str(exc)})

    @internal.post("/internal/update-task")
    async def internal_update_task(body: UpdateTaskPersistRequest, request: Request):
        """Persist a task update outside sandbox EROFS constraints."""
        denied = ensure_self(getattr(request.state, "internal_caller", None), body.anima_name, path=request.url.path)
        if denied is not None:
            return denied
        from core.anima.factory import validate_anima_name
        from core.paths import get_animas_dir
        from core.tasks.queue import TaskQueueManager

        if validate_anima_name(body.anima_name):
            return JSONResponse(status_code=400, content={"detail": "Invalid anima name"})
        anima_dir = get_animas_dir() / body.anima_name
        if not anima_dir.is_dir():
            return JSONResponse(
                status_code=404,
                content={"detail": f"Anima directory not found: {body.anima_name}"},
            )

        def _persist() -> Any:
            from core.tasks.board.tasks import attempt_scope
            from core.tasks.dispatch import persist_task_update

            manager = TaskQueueManager(anima_dir)
            with attempt_scope(body.attempt_identity):
                return persist_task_update(
                    manager,
                    body.task_id,
                    body.status,
                    summary=body.summary,
                    resume=body.resume,
                    meta_patch=body.meta,
                    waiting_for=body.waiting_for,
                    scope_note=body.scope_note,
                )

        try:
            loop = asyncio.get_running_loop()
            entry = await loop.run_in_executor(_native_executor, _persist)
        except Exception as exc:
            from core.exceptions import TaskOwnedByOther

            if isinstance(exc, TaskOwnedByOther):
                return JSONResponse(
                    status_code=409,
                    content={"error_type": "TaskOwnedByOther", "owner": exc.owner},
                )
            if isinstance(exc, ValueError) and (
                body.status in {None, "in_progress"}
                or (body.resume and body.status != "pending")
                or body.waiting_for is not None
                or body.scope_note is not None
            ):
                return JSONResponse(status_code=400, content={"error_type": "InvalidArguments", "detail": str(exc)})
            logger.exception("internal update-task failed")
            return JSONResponse(status_code=500, content={"detail": str(exc)})
        if entry is None:
            return JSONResponse(status_code=404, content={"detail": f"Task not found: {body.task_id}"})
        return {"ok": True, "task": entry.model_dump(mode="json")}

    router.include_router(internal)
    return router
