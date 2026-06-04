from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Usage instrumentation helpers for the completion_gate tool."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.paths import get_common_skills_dir
from core.skills.models import SkillUsageEventType
from core.skills.usage import SkillUsageTracker


@dataclass(slots=True)
class CompletionGateUsageResult:
    """Result summary returned to the completion_gate handler."""

    recorded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _ref_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _bad_ref(ref: str) -> str | None:
    if not ref:
        return "empty ref"
    if "\\" in ref:
        return "backslashes are not allowed"
    if "//" in ref:
        return "empty path segments are not allowed"
    if any(part in {"", ".", ".."} for part in ref.split("/")):
        return "path traversal is not allowed"
    path = Path(ref)
    if path.is_absolute():
        return "absolute paths are not allowed"
    parts = path.parts
    if not parts:
        return "path traversal is not allowed"
    return None


def _resolve_skill_ref(anima_dir: Path, ref: str) -> tuple[str, bool, str] | str:
    reason = _bad_ref(ref)
    if reason:
        return reason

    parts = Path(ref).parts
    if len(parts) == 2 and parts[0] in {"skills", "common_skills"}:
        root, name = parts
        rel_parts = (root, name, "SKILL.md")
    elif len(parts) == 3 and parts[0] == "skills" and parts[2] == "SKILL.md":
        root, name, _ = parts
        rel_parts = parts
    elif len(parts) >= 3 and parts[0] == "common_skills" and parts[-1] == "SKILL.md":
        root = parts[0]
        name = parts[-2]
        rel_parts = parts
    else:
        return "expected skills/{name}/SKILL.md or common_skills/{name}/SKILL.md"

    if not name or name == "quarantine":
        return "quarantine or empty skill names are not active skills"

    if root == "common_skills":
        skill_md = get_common_skills_dir().joinpath(*rel_parts[1:])
        is_common = True
    else:
        skill_md = anima_dir / "skills" / name / "SKILL.md"
        is_common = False

    if not skill_md.exists():
        return "skill file not found"
    return name, is_common, str(Path(*rel_parts))


def _resolve_procedure_ref(anima_dir: Path, ref: str) -> tuple[str, Path] | str:
    reason = _bad_ref(ref)
    if reason:
        return reason

    path = Path(ref)
    parts = path.parts
    if len(parts) != 2 or parts[0] != "procedures" or path.suffix != ".md":
        return "expected procedures/{name}.md"

    target = anima_dir / path
    if not target.exists():
        return "procedure file not found"
    return path.stem, target


def record_completion_gate_usage(anima_dir: Path, args: dict[str, Any]) -> CompletionGateUsageResult:
    """Validate completion_gate refs and append usage events for valid refs."""
    result = CompletionGateUsageResult()
    tracker = SkillUsageTracker(anima_dir)
    seen: set[tuple[str, str, bool]] = set()

    for ref in _ref_list(args.get("applied_skill_refs")):
        resolved = _resolve_skill_ref(anima_dir, ref)
        if isinstance(resolved, str):
            result.warnings.append(f"{ref}: {resolved}")
            continue
        skill_name, is_common, canonical_ref = resolved
        key = ("skill", skill_name, is_common)
        if key in seen:
            continue
        seen.add(key)
        tracker.record(
            skill_name,
            SkillUsageEventType.use,
            is_common=is_common,
            ref=canonical_ref,
            notes="completion_gate",
        )
        result.recorded.append(canonical_ref.removesuffix("/SKILL.md"))

    for ref in _ref_list(args.get("applied_procedure_refs")):
        resolved_proc = _resolve_procedure_ref(anima_dir, ref)
        if isinstance(resolved_proc, str):
            result.warnings.append(f"{ref}: {resolved_proc}")
            continue
        proc_name, _proc_path = resolved_proc
        key = ("procedure", proc_name, False)
        if key in seen:
            continue
        seen.add(key)
        tracker.record(
            proc_name,
            SkillUsageEventType.use,
            is_common=False,
            is_procedure=True,
            ref=ref,
            notes="completion_gate",
        )
        result.recorded.append(f"procedures/{proc_name}.md")

    creation = args.get("skill_creation")
    if creation is not None:
        if not isinstance(creation, dict):
            result.warnings.append("skill_creation: expected object")
        else:
            status = str(creation.get("status") or "").strip()
            if status and status not in {"created", "candidate_only", "not_needed"}:
                result.warnings.append(f"skill_creation.status: unexpected value '{status}'")
            created_refs = _ref_list(creation.get("created_skill_refs"))
            if status == "created" and not created_refs:
                result.warnings.append("skill_creation.created_skill_refs: required when status is created")
            for ref in created_refs:
                resolved = _resolve_skill_ref(anima_dir, ref)
                if isinstance(resolved, str):
                    result.warnings.append(f"{ref}: {resolved}")

    return result
