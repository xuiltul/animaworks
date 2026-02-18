from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("animaworks.routes.logs")

# Directories to search for log files
_LOG_SEARCH_DIRS = [
    Path.home() / ".animaworks" / "logs",
]


def _validate_filename(filename: str) -> None:
    """Reject filenames with path traversal attempts."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")


def _collect_log_files() -> list[dict]:
    """Collect log files from known directories."""
    files: list[dict] = []
    seen: set[str] = set()

    for log_dir in _LOG_SEARCH_DIRS:
        if not log_dir.exists():
            continue
        for log_file in log_dir.rglob("*.log"):
            if log_file.name in seen:
                continue
            seen.add(log_file.name)
            stat = log_file.stat()
            files.append({
                "name": log_file.name,
                "path": str(log_file.relative_to(log_dir)),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            })

    # Sort by modification time descending
    files.sort(key=lambda f: f["modified"], reverse=True)
    return files


def _resolve_log_path(filename: str) -> Path | None:
    """Find a log file by name in known log directories."""
    for log_dir in _LOG_SEARCH_DIRS:
        candidate = log_dir / filename
        if candidate.exists() and candidate.is_file():
            # Ensure the resolved path is within the log directory
            try:
                candidate.resolve().relative_to(log_dir.resolve())
            except ValueError:
                return None
            return candidate
    return None


def create_logs_router() -> APIRouter:
    router = APIRouter()

    @router.get("/system/logs")
    async def list_logs(request: Request):
        """List available log files."""
        return {"files": _collect_log_files()}

    @router.get("/system/logs/stream")
    async def stream_logs(
        request: Request,
        file: str = Query(default="animaworks.log"),
    ):
        """SSE endpoint for real-time log streaming (tail -f style)."""
        _validate_filename(file)
        log_path = _resolve_log_path(file)
        if log_path is None:
            raise HTTPException(status_code=404, detail=f"Log file not found: {file}")

        async def log_stream_generator():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    # Seek to end
                    f.seek(0, 2)
                    while True:
                        line = f.readline()
                        if line:
                            yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
                        else:
                            await asyncio.sleep(0.5)
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            log_stream_generator(),
            media_type="text/event-stream",
        )

    @router.get("/system/logs/{filename}")
    async def read_log(
        request: Request,
        filename: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=5000),
    ):
        """Read log file content with pagination."""
        _validate_filename(filename)
        log_path = _resolve_log_path(filename)
        if log_path is None:
            raise HTTPException(
                status_code=404, detail=f"Log file not found: {filename}"
            )

        try:
            all_lines = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read log: {exc}")

        total_lines = len(all_lines)
        paginated = all_lines[offset : offset + limit]

        return {
            "filename": filename,
            "total_lines": total_lines,
            "offset": offset,
            "limit": limit,
            "lines": paginated,
        }

    return router
