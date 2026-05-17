from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Interactive approval routing for call_human and related flows.

Persists pending approval requests, validates tokens, and delivers outcomes
to Anima inboxes via :meth:`core.messenger.Messenger.receive_external`.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from core.auth.manager import load_auth
from core.i18n import t
from core.paths import get_data_dir, get_shared_dir
from core.platform.locks import file_lock

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

TTL_DAYS = 7  # mirrored on :class:`InteractionRouter`

# ── Data models ───────────────────────────────────────────


class InteractionRequest(BaseModel):
    """A pending interactive approval or gate request."""

    callback_id: str
    anima_name: str
    category: str
    options: list[str]
    allowed_users: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    approval_token: str
    message_ts: dict[str, str] = Field(default_factory=dict)


class InteractionResult(BaseModel):
    """Outcome of resolving an :class:`InteractionRequest`."""

    callback_id: str
    decision: str
    actor: str
    source: str
    comment: str = ""
    resolved_at: datetime


# ── InteractionRouter ────────────────────────────────────


class InteractionRouter:
    """Load/store interactive requests and deliver results to Anima inboxes.

    Storage path: ``{data_dir}/run/interaction_map.json`` (see :attr:`STORAGE_FILE`).
    """

    STORAGE_FILE = "run/interaction_map.json"
    TTL_DAYS = TTL_DAYS

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    def _map_path(self) -> Path:
        return get_data_dir() / self.STORAGE_FILE

    def _hmac_secret_bytes(self) -> bytes:
        """Return HMAC key from :class:`~core.auth.models.AuthConfig` or path hash."""
        cfg = load_auth()
        sk = (cfg.secret_key or "").strip()
        if sk:
            return sk.encode("utf-8")
        path_bytes = str(get_data_dir().resolve()).encode("utf-8")
        return hashlib.sha256(path_bytes).digest()

    def _compute_approval_token(self, callback_id: str) -> str:
        key = self._hmac_secret_bytes()
        return hmac.new(key, callback_id.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_approval_token(self, callback_id: str, token: str) -> bool:
        """Return True if *token* matches the HMAC for *callback_id*."""
        if not token or not callback_id:
            return False
        expected = self._compute_approval_token(callback_id)
        return hmac.compare_digest(expected, token)

    def _read_all_entries(self) -> dict[str, Any]:
        path = self._map_path()
        if not path.exists():
            return {"entries": {}}
        try:
            with path.open("r", encoding="utf-8") as fd, file_lock(fd, exclusive=False):
                raw = fd.read()
            data = json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read interaction map from %s", path)
            return {"entries": {}}
        if "entries" not in data or not isinstance(data["entries"], dict):
            return {"entries": {}}
        return data

    def _write_all_entries(self, data: dict[str, Any]) -> None:
        path = self._map_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a+", encoding="utf-8") as fd, file_lock(fd, exclusive=True):
                fd.seek(0)
                fd.truncate()
                fd.write(json.dumps(data, ensure_ascii=False, indent=2))
                fd.flush()
        except OSError:
            logger.exception("Failed to write interaction map to %s", path)

    def _request_expired(self, req: InteractionRequest, *, now: datetime) -> bool:
        if req.created_at.tzinfo is None:
            created = req.created_at.replace(tzinfo=UTC)
        else:
            created = req.created_at.astimezone(UTC)
        age_days = (now - created).total_seconds() / 86400.0
        return age_days > float(self.TTL_DAYS)

    def _dump_request(self, req: InteractionRequest) -> dict[str, Any]:
        return req.model_dump(mode="json")

    def _parse_request(self, blob: dict[str, Any]) -> InteractionRequest:
        return InteractionRequest.model_validate(blob)

    async def create(
        self,
        anima_name: str,
        category: str,
        options: list[str],
        allowed_users: dict[str, list[str]] | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        callback_id: str | None = None,
    ) -> InteractionRequest:
        """Create and persist a new interaction request.

        Args:
            anima_name: Target Anima that will receive the outcome.
            category: Logical category (e.g. ``approval``, ``design_gate``).
            options: Allowed decision labels shown to the user.
            allowed_users: Optional per-platform user IDs allowed to act.
            metadata: Arbitrary context stored with the request.
            callback_id: Optional stable id (e.g. CLI ``--callback-id``); auto-generated if omitted.

        Returns:
            The persisted :class:`InteractionRequest` including ``approval_token``.
        """
        async with self._lock:
            data = self._read_all_entries()
            entries: dict[str, Any] = data.setdefault("entries", {})

            explicit = (callback_id or "").strip()
            if explicit:
                cid = explicit
                if cid in entries:
                    raise ValueError(f"callback_id already in use: {cid!r}")
            else:
                cid = secrets.token_urlsafe(16)
                while cid in entries:
                    cid = secrets.token_urlsafe(16)
            callback_id = cid
            approval_token = self._compute_approval_token(callback_id)
            created_at = datetime.now(UTC)
            req = InteractionRequest(
                callback_id=callback_id,
                anima_name=anima_name,
                category=category,
                options=list(options),
                allowed_users=dict(allowed_users or {}),
                metadata=dict(metadata or {}),
                created_at=created_at,
                approval_token=approval_token,
                message_ts={},
            )
            entries[callback_id] = {
                "resolved": False,
                "request": self._dump_request(req),
                "result": None,
            }
            self._write_all_entries(data)
            return req

    async def lookup(self, callback_id: str) -> InteractionRequest | None:
        """Return the pending request for *callback_id* if active and not expired."""
        async with self._lock:
            data = self._read_all_entries()
            entry = data.get("entries", {}).get(callback_id)
            if entry is None:
                return None
            if entry.get("resolved"):
                return None
            req_blob = entry.get("request")
            if not isinstance(req_blob, dict):
                return None
            req = self._parse_request(req_blob)
            now = datetime.now(UTC)
            if self._request_expired(req, now=now):
                return None
            return req

    async def lookup_resolved(self, callback_id: str) -> tuple[InteractionRequest, InteractionResult] | None:
        """Return a resolved request and result, or ``None`` if it is not approved yet."""
        async with self._lock:
            data = self._read_all_entries()
            entry = data.get("entries", {}).get(callback_id)
            if entry is None or not entry.get("resolved"):
                return None
            req_blob = entry.get("request")
            result_blob = entry.get("result")
            if not isinstance(req_blob, dict) or not isinstance(result_blob, dict):
                return None
            try:
                req = self._parse_request(req_blob)
                result = InteractionResult.model_validate(result_blob)
            except Exception:
                logger.exception("Failed to parse resolved interaction callback_id=%s", callback_id)
                return None
            return req, result

    async def resolve(
        self,
        callback_id: str,
        decision: str,
        actor: str,
        source: str,
        comment: str = "",
    ) -> InteractionResult | None:
        """Mark a request resolved, persist, and inject the result into the Anima inbox.

        Args:
            callback_id: Request identifier.
            decision: Chosen option or free-text label.
            actor: Who took the action (display id or name).
            source: Channel id (e.g. ``slack``, ``web``).
            comment: Optional comment text.

        Returns:
            :class:`InteractionResult` on success, or ``None`` if missing or already resolved.
        """
        async with self._lock:
            data = self._read_all_entries()
            entries: dict[str, Any] = data.setdefault("entries", {})
            entry = entries.get(callback_id)
            if entry is None:
                return None
            if entry.get("resolved"):
                return None
            req_blob = entry.get("request")
            if not isinstance(req_blob, dict):
                return None
            req = self._parse_request(req_blob)
            now = datetime.now(UTC)
            if self._request_expired(req, now=now):
                return None

            resolved_at = now
            result = InteractionResult(
                callback_id=callback_id,
                decision=decision,
                actor=actor,
                source=source,
                comment=comment or "",
                resolved_at=resolved_at,
            )
            entry["resolved"] = True
            entry["result"] = result.model_dump(mode="json")
            entries[callback_id] = entry
            self._write_all_entries(data)

        summary = t("interactive.resolved_by", actor=actor, decision=decision)
        lines = [
            f"[Interactive / {req.category}]",
            summary,
            f"callback_id: {callback_id}",
            f"source: {source}",
        ]
        if comment:
            lines.append(f"comment: {comment}")
        content = "\n".join(lines)

        from core.messenger import Messenger

        messenger = Messenger(get_shared_dir(), req.anima_name)
        messenger.receive_external(
            content=content,
            source=source,
            source_message_id=f"interaction:{callback_id}",
            external_user_id=actor,
            intent="question",
        )
        return result

    async def update_message_ts(self, callback_id: str, platform: str, ts: str) -> None:
        """Attach a platform message id (e.g. Slack ``ts``) for button invalidation."""
        async with self._lock:
            data = self._read_all_entries()
            entries: dict[str, Any] = data.setdefault("entries", {})
            entry = entries.get(callback_id)
            if entry is None:
                return
            req_blob = entry.get("request")
            if not isinstance(req_blob, dict):
                return
            req = self._parse_request(req_blob)
            mt = dict(req.message_ts)
            mt[platform] = ts
            req = req.model_copy(update={"message_ts": mt})
            entry["request"] = self._dump_request(req)
            entries[callback_id] = entry
            self._write_all_entries(data)

    async def prune(self, max_age_days: int = 7) -> int:
        """Remove entries older than *max_age_days*; returns number removed."""
        async with self._lock:
            data = self._read_all_entries()
            entries: dict[str, Any] = data.setdefault("entries", {})
            now = datetime.now(UTC)
            to_delete: list[str] = []
            for cid, entry in list(entries.items()):
                if not isinstance(entry, dict):
                    to_delete.append(cid)
                    continue
                req_blob = entry.get("request")
                if not isinstance(req_blob, dict):
                    to_delete.append(cid)
                    continue
                try:
                    req = self._parse_request(req_blob)
                except (ValueError, TypeError, ValidationError):
                    to_delete.append(cid)
                    continue

                cutpoint: datetime | None = None
                if entry.get("resolved"):
                    res_blob = entry.get("result")
                    if isinstance(res_blob, dict) and res_blob.get("resolved_at"):
                        try:
                            ra = datetime.fromisoformat(str(res_blob["resolved_at"]))
                            if ra.tzinfo is None:
                                cutpoint = ra.replace(tzinfo=UTC)
                            else:
                                cutpoint = ra.astimezone(UTC)
                        except (ValueError, TypeError):
                            cutpoint = None
                    if cutpoint is None:
                        created = req.created_at
                        if created.tzinfo is None:
                            cutpoint = created.replace(tzinfo=UTC)
                        else:
                            cutpoint = created.astimezone(UTC)
                else:
                    created = req.created_at
                    if created.tzinfo is None:
                        cutpoint = created.replace(tzinfo=UTC)
                    else:
                        cutpoint = created.astimezone(UTC)

                if cutpoint is None:
                    to_delete.append(cid)
                    continue
                age_days = (now - cutpoint).total_seconds() / 86400.0
                if age_days > float(max_age_days):
                    to_delete.append(cid)

            for cid in to_delete:
                entries.pop(cid, None)
            removed = len(to_delete)
            if removed:
                self._write_all_entries(data)
            return removed


# ── Singleton ────────────────────────────────────────────

_router: InteractionRouter | None = None


def get_interaction_router() -> InteractionRouter:
    """Return the process-wide :class:`InteractionRouter` instance."""
    global _router
    if _router is None:
        _router = InteractionRouter()
    return _router


def build_text_fallback(interaction: InteractionRequest, *, web_base_url: str = "") -> str:
    """Build text-based fallback for channels without native button support."""
    lines = ["\n▶ " + t("interactive.fallback_header")]
    for i, opt in enumerate(interaction.options, 1):
        emoji = {"approve": "✅", "reject": "❌", "comment": "💬"}.get(opt, "▶️")
        lines.append(f"[{i}] {emoji} {opt.capitalize()}")
    lines.append("")
    lines.append(t("interactive.fallback_instruction"))

    if web_base_url:
        url = f"{web_base_url.rstrip('/')}/api/approve/{interaction.callback_id}?token={interaction.approval_token}"
        lines.append(t("interactive.fallback_url_or", url=url))

    return "\n".join(lines)


__all__ = [
    "InteractionRequest",
    "InteractionResult",
    "InteractionRouter",
    "TTL_DAYS",
    "build_text_fallback",
    "get_interaction_router",
]
