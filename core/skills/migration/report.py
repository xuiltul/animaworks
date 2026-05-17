from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared report models for external agent migration imports."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.time_utils import now_iso


@dataclass(slots=True)
class MigrationItem:
    """One planned or applied migration action."""

    source_type: str
    source_path: str
    target_path: str
    action: str
    status: str
    source_fingerprint: str = ""
    message: str = ""
    scan_verdict: str | None = None
    manual_action: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MigrationReport:
    """Dry-run/apply report returned by migration importers."""

    source_system: str
    source_path: str
    mode: str
    batch_id: str
    target_anima: str | None = None
    common_skills: bool = False
    generated_at: str = field(default_factory=now_iso)
    items: list[MigrationItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manual_actions: list[str] = field(default_factory=list)
    backup_manifest_path: str | None = None
    report_path: str | None = None

    def add_item(self, item: MigrationItem) -> None:
        self.items.append(item)
        if item.manual_action:
            self.manual_actions.append(item.manual_action)

    def add_warning(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "source_path": self.source_path,
            "mode": self.mode,
            "batch_id": self.batch_id,
            "target_anima": self.target_anima,
            "common_skills": self.common_skills,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "warnings": list(self.warnings),
            "manual_actions": list(self.manual_actions),
            "backup_manifest_path": self.backup_manifest_path,
            "report_path": self.report_path,
            "items": [item.to_dict() for item in self.items],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.source_system.title()} Migration Report",
            "",
            f"- source_system: {self.source_system}",
            f"- source_path: `{self.source_path}`",
            f"- mode: {self.mode}",
            f"- batch_id: `{self.batch_id}`",
            f"- target_anima: {self.target_anima or '(none)'}",
            f"- common_skills: {str(self.common_skills).lower()}",
            f"- generated_at: {self.generated_at}",
        ]
        if self.backup_manifest_path:
            lines.append(f"- backup_manifest: `{self.backup_manifest_path}`")
        lines.extend(["", "## Summary", ""])
        if self.summary:
            for status, count in sorted(self.summary.items()):
                lines.append(f"- {status}: {count}")
        else:
            lines.append("- no_items: 0")
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        if self.manual_actions:
            lines.extend(["", "## Manual Actions", ""])
            lines.extend(f"- {action}" for action in self.manual_actions)
        lines.extend(["", "## Items", ""])
        for item in self.items:
            detail = f"- {item.status}: {item.action} `{item.source_path}` -> `{item.target_path}`"
            if item.scan_verdict:
                detail += f" scan={item.scan_verdict}"
            if item.message:
                detail += f" - {item.message}"
            lines.append(detail)
        return "\n".join(lines).rstrip() + "\n"

    def write_markdown(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        self.report_path = str(path)
        return path
