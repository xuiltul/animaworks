# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Data models for the external tasks snapshot store."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExternalTask(BaseModel):
    """One external task item collected from a remote source.

    ``id`` is assigned by each source collector in the deterministic form
    ``"{source_type}-{stable_external_id}"`` (e.g. ``github-pr-42``).
    """

    id: str
    title: str
    status: str
    source_type: str
    source_icon: str
    source_url: str | None = None
    created_at: str
    last_updated_at: str
    priority: int


class SourceHealth(BaseModel):
    """Per-source collection health recorded in a snapshot."""

    status: str  # "ok" | "unavailable"
    collected_at: str | None = None
    error: str | None = None


class Snapshot(BaseModel):
    """JSON snapshot of collected external tasks and source health."""

    version: int = 1
    last_collected_at: str | None = None
    sources: dict[str, SourceHealth] = Field(default_factory=dict)
    tasks: list[ExternalTask] = Field(default_factory=list)
