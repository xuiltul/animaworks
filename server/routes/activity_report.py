from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Activity Report API — organisation-wide audit + LLM narrative generation."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.i18n import t
from core.paths import get_data_dir
from core.time_utils import now_jst

logger = logging.getLogger("animaworks.routes.activity_report")

_CACHE_DIR_NAME = "activity-reports"


# ── Request / Response models ─────────────────────────────────


class GenerateRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    model: str = ""
    force_regenerate: bool = False


# ── Helpers ───────────────────────────────────────────────────


def _cache_dir() -> Path:
    d = get_data_dir() / "cache" / _CACHE_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(report_date: str, model: str) -> Path:
    model_hash = hashlib.md5(model.encode()).hexdigest()[:8]
    return _cache_dir() / f"{report_date}_{model_hash}.json"


def _read_cache(report_date: str, model: str) -> dict[str, Any] | None:
    p = _cache_path(report_date, model)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_cache(report_date: str, model: str, data: dict[str, Any]) -> None:
    p = _cache_path(report_date, model)
    try:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("Failed to write cache: %s", p)


def _resolve_model() -> str:
    """Resolve default model from consolidation config."""
    try:
        from core.config import load_config

        return load_config().consolidation.llm_model
    except Exception:
        return ""


def _available_models() -> list[dict[str, str]]:
    """Build list of available models for the UI dropdown."""
    try:
        from core.config import load_config

        cfg = load_config()
    except Exception:
        return []

    models: list[dict[str, str]] = []
    seen: set[str] = set()

    default = cfg.consolidation.llm_model
    if default and default not in seen:
        label = default.split("/")[-1] if "/" in default else default
        models.append({"id": default, "label": label})
        seen.add(default)

    for provider, cred in cfg.credentials.items():
        if not cred.api_key:
            continue
        if provider == "anthropic":
            for m in ("anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-4-5"):
                if m not in seen:
                    models.append({"id": m, "label": m.split("/")[-1]})
                    seen.add(m)
        elif provider == "openai":
            for m in ("openai/gpt-4.1-mini", "openai/gpt-4.1-nano"):
                if m not in seen:
                    models.append({"id": m, "label": m.split("/")[-1]})
                    seen.add(m)
        elif provider in ("google", "gemini"):
            for m in ("gemini/gemini-2.5-flash",):
                if m not in seen:
                    models.append({"id": m, "label": m.split("/")[-1]})
                    seen.add(m)

    return models


_MAX_TIMELINE_CHARS = 280_000


def _truncate_timeline(text: str) -> str:
    """Last-resort truncation if budget-aware thinning was insufficient."""
    if len(text) <= _MAX_TIMELINE_CHARS:
        return text
    truncated = text[:_MAX_TIMELINE_CHARS]
    last_nl = truncated.rfind("\n")
    if last_nl > 0:
        truncated = truncated[:last_nl]
    return truncated + "\n\n... (truncated) ..."


async def _generate_narrative(timeline_text: str, model: str) -> str | None:
    """Generate LLM narrative from unified timeline text."""
    try:
        from core.memory._llm_utils import one_shot_completion
    except ImportError:
        logger.info("one_shot_completion not available; skipping narrative generation")
        return None

    safe_text = _truncate_timeline(timeline_text)
    system_prompt = t("activity_report.llm_system_prompt")
    user_prompt = t("activity_report.llm_user_prompt", data=safe_text)

    try:
        return await one_shot_completion(
            user_prompt,
            system_prompt=system_prompt,
            model=model,
            max_tokens=4096,
        )
    except Exception as e:
        logger.warning("Narrative generation failed: %s", e)
        return None


# ── Router ────────────────────────────────────────────────────


def create_activity_report_router() -> APIRouter:
    router = APIRouter(prefix="/activity-report", tags=["activity-report"])

    @router.get("/models")
    async def get_models() -> JSONResponse:
        default_model = _resolve_model()
        models = _available_models()
        return JSONResponse(
            {
                "default_model": default_model,
                "available_models": models,
            }
        )

    @router.post("/generate")
    async def generate_report(req: GenerateRequest) -> JSONResponse:
        try:
            report_date = datetime.strptime(req.date, "%Y-%m-%d").date()
        except ValueError:
            return JSONResponse(
                {"error": t("activity_report.invalid_date")},
                status_code=400,
            )

        today = now_jst().date()
        if report_date > today:
            return JSONResponse(
                {"error": t("activity_report.future_date")},
                status_code=400,
            )

        model = req.model or _resolve_model()

        if req.model:
            allowed = {m["id"] for m in _available_models()}
            if req.model not in allowed:
                return JSONResponse(
                    {"error": t("activity_report.invalid_model")},
                    status_code=400,
                )

        if not req.force_regenerate:
            cached = _read_cache(req.date, model)
            if cached:
                cached["cached"] = True
                return JSONResponse(cached)

        from core.audit import collect_org_audit, generate_org_timeline

        def _gen_timeline() -> str:
            return generate_org_timeline(req.date, max_chars=_MAX_TIMELINE_CHARS)

        report, timeline_text = await asyncio.gather(
            collect_org_audit(req.date),
            asyncio.get_event_loop().run_in_executor(None, _gen_timeline),
        )
        structured = report.to_dict()

        narrative_md: str | None = None
        if timeline_text.strip() and model:
            narrative_md = await _generate_narrative(timeline_text, model)

        result: dict[str, Any] = {
            "date": req.date,
            "structured": structured,
            "timeline": timeline_text,
            "narrative_md": narrative_md,
            "model_used": model,
            "cached": False,
            "generated_at": now_jst().isoformat(),
        }

        _write_cache(req.date, model, result)
        return JSONResponse(result)

    @router.get("/{report_date}")
    async def get_cached_report(report_date: str) -> JSONResponse:
        try:
            datetime.strptime(report_date, "%Y-%m-%d")
        except ValueError:
            return JSONResponse(
                {"error": t("activity_report.invalid_date")},
                status_code=400,
            )

        results: list[tuple[Path, dict]] = []
        for p in _cache_dir().glob(f"{report_date}_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                results.append((p, data))
            except (json.JSONDecodeError, OSError):
                continue

        if results:
            results.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
            data = results[0][1]
            data["cached"] = True
            return JSONResponse(data)

        return JSONResponse(
            {"error": t("activity_report.not_found")},
            status_code=404,
        )

    return router
