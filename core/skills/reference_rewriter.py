from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate human-reviewed skill reference rewrite proposals."""

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.time_utils import now_iso


@dataclass(slots=True)
class ReferenceRewriteChange:
    path: str
    before: str
    after: str


@dataclass(slots=True)
class ReferenceRewriteProposal:
    old_skill: str
    absorbed_into: str | None
    proposal_path: Path
    changes: list[ReferenceRewriteChange] = field(default_factory=list)


def generate_reference_rewrite_proposal(
    anima_dir: Path,
    skill_name: str,
    *,
    absorbed_into: str | None = None,
    actor: str = "curator",
) -> Path | None:
    """Write a Markdown proposal for references affected by archive/block/delete."""
    changes = collect_reference_rewrite_changes(
        anima_dir,
        skill_name,
        absorbed_into=absorbed_into,
    )
    proposal_dir = anima_dir / "state" / "skill_curator" / "reference_rewrites"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = proposal_dir / f"{_safe_stamp()}_{_safe_name(skill_name)}.md"
    proposal = ReferenceRewriteProposal(skill_name, absorbed_into, proposal_path, changes)
    proposal_path.write_text(_render_proposal(proposal, actor=actor), encoding="utf-8")
    return proposal_path


def collect_reference_rewrite_changes(
    anima_dir: Path,
    skill_name: str,
    *,
    absorbed_into: str | None = None,
) -> list[ReferenceRewriteChange]:
    """Return proposed text changes without modifying source files."""
    changes: list[ReferenceRewriteChange] = []
    for path in _candidate_files(anima_dir):
        if not path.is_file():
            continue
        before = path.read_text(encoding="utf-8")
        after = rewrite_skill_references_in_text(before, skill_name, absorbed_into=absorbed_into)
        if after != before:
            changes.append(
                ReferenceRewriteChange(
                    path=str(path.relative_to(anima_dir)),
                    before=before,
                    after=after,
                )
            )
    return changes


def rewrite_skill_references_in_text(text: str, skill_name: str, *, absorbed_into: str | None = None) -> str:
    """Rewrite YAML-ish and JSON skill references in a text blob."""
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _rewrite_yamlish_text(text, skill_name, absorbed_into=absorbed_into)
        changed, rewritten = _rewrite_json_value(data, skill_name, absorbed_into=absorbed_into)
        return json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n" if changed else text
    if _looks_jsonl(text):
        out: list[str] = []
        changed = False
        for line in text.splitlines():
            if not line.strip():
                out.append(line)
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                out.append(line)
                continue
            line_changed, rewritten = _rewrite_json_value(data, skill_name, absorbed_into=absorbed_into)
            changed = changed or line_changed
            out.append(json.dumps(rewritten, ensure_ascii=False) if line_changed else line)
        return "\n".join(out) + ("\n" if text.endswith("\n") else "") if changed else text
    return _rewrite_yamlish_text(text, skill_name, absorbed_into=absorbed_into)


def _candidate_files(anima_dir: Path) -> list[Path]:
    candidates = [anima_dir / "cron.md"]
    goals_dir = anima_dir / "goals"
    if goals_dir.is_dir():
        candidates.extend(
            path for path in goals_dir.rglob("*") if path.suffix.lower() in {".md", ".json", ".jsonl", ".yaml", ".yml"}
        )
    state_dir = anima_dir / "state"
    candidates.extend(
        path
        for path in [
            state_dir / "task_queue.jsonl",
            state_dir / "goal_state.jsonl",
            state_dir / "taskboard.json",
            state_dir / "taskboard.jsonl",
            state_dir / "taskboard.yaml",
            state_dir / "taskboard.yml",
        ]
        if path.is_file()
    )
    return sorted(set(candidates))


def _rewrite_yamlish_text(text: str, skill_name: str, *, absorbed_into: str | None) -> str:
    lines = text.splitlines()
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if _is_inline_skills_line(stripped):
            new_line = _rewrite_inline_skills_line(line, skill_name, absorbed_into)
            changed = changed or new_line != line
            if new_line:
                out.append(new_line)
            i += 1
            continue
        if _is_scalar_skills_line(stripped):
            new_line = _rewrite_scalar_skills_line(line, skill_name, absorbed_into)
            changed = changed or new_line != line
            if new_line:
                out.append(new_line)
            i += 1
            continue
        if re.match(r"^\s*skills:\s*$", line):
            block, next_i = _consume_skills_block(lines, i)
            new_block = _rewrite_skills_block(block, skill_name, absorbed_into)
            changed = changed or new_block != block
            out.extend(new_block)
            i = next_i
            continue
        if _is_scalar_skill_line(stripped):
            new_line = _rewrite_scalar_skill_line(line, skill_name, absorbed_into)
            changed = changed or new_line != line
            if new_line:
                out.append(new_line)
            i += 1
            continue
        out.append(line)
        i += 1
    if not changed:
        return text
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + suffix


def _rewrite_json_value(value: Any, skill_name: str, *, absorbed_into: str | None) -> tuple[bool, Any]:
    if isinstance(value, dict):
        changed = False
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"skill", "skill_name", "skill_pointer"} and item == skill_name:
                changed = True
                if absorbed_into:
                    result[key] = absorbed_into
                continue
            if key == "skills" and isinstance(item, list):
                new_list = _rewrite_skill_list([str(v) for v in item], skill_name, absorbed_into)
                changed = changed or new_list != item
                if new_list:
                    result[key] = new_list
                continue
            item_changed, new_item = _rewrite_json_value(item, skill_name, absorbed_into=absorbed_into)
            changed = changed or item_changed
            result[key] = new_item
        return changed, result
    if isinstance(value, list):
        changed = False
        result = []
        for item in value:
            item_changed, new_item = _rewrite_json_value(item, skill_name, absorbed_into=absorbed_into)
            changed = changed or item_changed
            result.append(new_item)
        return changed, result
    return False, value


def _is_inline_skills_line(stripped: str) -> bool:
    return stripped.startswith("skills:") and "[" in stripped and "]" in stripped


def _is_scalar_skills_line(stripped: str) -> bool:
    return stripped.startswith("skills:") and "[" not in stripped and stripped != "skills:"


def _is_scalar_skill_line(stripped: str) -> bool:
    return stripped.startswith(("skill:", "skill_name:", "skill_pointer:"))


def _rewrite_inline_skills_line(line: str, skill_name: str, absorbed_into: str | None) -> str:
    prefix, rest = line.split(":", 1)
    before, _, after_bracket = rest.partition("[")
    inner, _, suffix = after_bracket.partition("]")
    values = [_strip_quotes(v.strip()) for v in inner.split(",") if v.strip()]
    rewritten = _rewrite_skill_list(values, skill_name, absorbed_into)
    if not rewritten:
        return ""
    quote = '"'
    joined = ", ".join(f"{quote}{v}{quote}" for v in rewritten)
    return f"{prefix}:{before}[{joined}]{suffix}"


def _consume_skills_block(lines: list[str], start: int) -> tuple[list[str], int]:
    block = [lines[start]]
    i = start + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("- "):
            block.append(lines[i])
            i += 1
            continue
        if not stripped:
            block.append(lines[i])
            i += 1
            continue
        break
    return block, i


def _rewrite_skills_block(block: list[str], skill_name: str, absorbed_into: str | None) -> list[str]:
    header = block[0]
    indent = re.match(r"^(\s*)", header).group(1)  # type: ignore[union-attr]
    items = [_strip_quotes(line.strip()[2:].strip()) for line in block[1:] if line.strip().startswith("- ")]
    rewritten = _rewrite_skill_list(items, skill_name, absorbed_into)
    if not rewritten:
        return []
    return [header] + [f"{indent}  - {item}" for item in rewritten]


def _rewrite_scalar_skill_line(line: str, skill_name: str, absorbed_into: str | None) -> str:
    prefix, value = line.split(":", 1)
    current = _strip_quotes(value.strip())
    if current != skill_name:
        return line
    if not absorbed_into:
        return ""
    return f"{prefix}: {absorbed_into}"


def _rewrite_scalar_skills_line(line: str, skill_name: str, absorbed_into: str | None) -> str:
    prefix, value = line.split(":", 1)
    raw = value.strip()
    values = [_strip_quotes(v.strip()) for v in raw.split(",") if v.strip()] if "," in raw else [_strip_quotes(raw)]
    rewritten = _rewrite_skill_list(values, skill_name, absorbed_into)
    if rewritten == values:
        return line
    if not rewritten:
        return ""
    return f"{prefix}: {', '.join(rewritten)}"


def _rewrite_skill_list(values: list[str], skill_name: str, absorbed_into: str | None) -> list[str]:
    result: list[str] = []
    for value in values:
        if _skill_ref_matches(value, skill_name):
            if absorbed_into and absorbed_into not in result:
                result.append(absorbed_into)
            continue
        if value not in result:
            result.append(value)
    return result


def _skill_ref_matches(value: str, skill_name: str) -> bool:
    if value == skill_name:
        return True
    path = Path(value)
    if path.name != "SKILL.md" or ".." in path.parts:
        return False
    return path.parent.name == skill_name


def _looks_jsonl(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("{") and line.endswith("}") for line in lines)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _render_proposal(proposal: ReferenceRewriteProposal, *, actor: str) -> str:
    lines = [
        f"# Skill Reference Rewrite Proposal: {proposal.old_skill}",
        "",
        f"- old_skill: {proposal.old_skill}",
        f"- absorbed_into: {proposal.absorbed_into or 'null'}",
        f"- actor: {actor}",
        f"- generated_at: {now_iso()}",
        "",
    ]
    if not proposal.changes:
        lines.append("No references were found.")
        return "\n".join(lines) + "\n"
    for change in proposal.changes:
        lines.extend(
            [
                f"## {change.path}",
                "",
                "```diff",
                *_diff_lines(change.path, change.before, change.after),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _diff_lines(path: str, before: str, after: str) -> list[str]:
    return list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )


def _safe_stamp() -> str:
    return now_iso().replace(":", "").replace("+", "_").replace("-", "").replace(".", "")


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-._")
    return safe or "skill"
