from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Session-scoped action memory gate for external side-effect tools."""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.action_memory_gate")

_REQUIRED_READ_RE = re.compile(
    r"""read_memory_file\s*\(\s*path\s*=\s*(?P<quote>['"])(?P<path>.+?)(?P=quote)\s*\)"""
)
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_HANDLER_ACTION_TOOLS: frozenset[str] = frozenset(
    {
        "send_message",
        "post_channel",
        "call_human",
        "write_memory_file",
        "gmail_draft",
        "gmail_send",
        "chatwork_send",
        "slack_send",
        "discord_send",
    }
)

_CLI_ACTION_MAP: dict[tuple[str, str], str] = {
    ("gmail", "draft"): "gmail_draft",
    ("gmail", "send"): "gmail_send",
    ("chatwork", "send"): "chatwork_send",
    ("slack", "send"): "slack_send",
    ("discord", "send"): "discord_send",
}


ACTION_TOOL_NAMES: frozenset[str] = _HANDLER_ACTION_TOOLS


@dataclass(frozen=True, slots=True)
class ActionMemoryGateDecision:
    """Result of checking whether an action can execute."""

    allowed: bool
    tool: str
    reason: str = ""
    rule_id: str = ""
    rule: str = ""
    required_paths: list[str] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        """Return a structured payload suitable for tool output or stderr."""
        if self.allowed:
            return {
                "status": "ok",
                "tool": self.tool,
                "message": "Action allowed by memory gate",
            }
        return {
            "status": "error",
            "error_type": "ActionMemoryGate",
            "message": "Action paused by memory gate. Read required memory before retrying.",
            "tool": self.tool,
            "reason": self.reason,
            "rule_id": self.rule_id,
            "required_paths": self.required_paths,
            "missing_paths": self.missing_paths,
            "rule": self.rule,
            "score": self.score,
        }

    def to_json(self) -> str:
        """Return the decision payload as JSON."""
        return json.dumps(self.to_payload(), ensure_ascii=False)


def action_tool_name_for_handler(name: str) -> str | None:
    """Return the action-rule tool name for a ToolHandler schema name."""
    return name if name in _HANDLER_ACTION_TOOLS else None


def action_tool_name_for_sdk(name: str) -> str | None:
    """Return the canonical action-rule name for SDK/MCP PreToolUse names."""
    if name.startswith("mcp__aw__"):
        name = name[len("mcp__aw__") :]
    return action_tool_name_for_handler(name)


def action_tool_name_from_cli_argv(argv: list[str]) -> str | None:
    """Map ``animaworks-tool`` argv to an action-rule tool name."""
    if not argv:
        return None
    tool_name = argv[0]
    if tool_name == "submit":
        return None
    if tool_name == "call_human":
        return "call_human"

    subcommand = ""
    for arg in argv[1:]:
        if not arg.startswith("-"):
            subcommand = arg
            break
    if not subcommand:
        return None
    return _CLI_ACTION_MAP.get((tool_name, subcommand))


def _normalize_memory_path(raw: str, anima_dir: Path) -> str:
    """Normalize memory paths for comparing required reads with reads performed."""
    path_text = str(raw).strip()
    path_text = re.sub(r"/+", "/", path_text)
    while path_text.startswith("./"):
        path_text = path_text[2:]
    if path_text.endswith("/") and path_text != "/":
        path_text = path_text[:-1]

    if not path_text:
        return ""
    if not path_text.startswith("/") and ".." not in path_text:
        return path_text

    try:
        resolved = Path(path_text).resolve() if path_text.startswith("/") else (anima_dir / path_text).resolve()
        anima_resolved = anima_dir.resolve()
        try:
            return str(resolved.relative_to(anima_resolved))
        except ValueError:
            pass

        from core.paths import get_common_knowledge_dir, get_common_skills_dir, get_reference_dir

        shared_roots = (
            ("common_knowledge", get_common_knowledge_dir().resolve()),
            ("reference", get_reference_dir().resolve()),
            ("common_skills", get_common_skills_dir().resolve()),
        )
        for prefix, root in shared_roots:
            try:
                return f"{prefix}/{resolved.relative_to(root)}"
            except ValueError:
                continue
    except Exception:
        logger.debug("Failed to normalize memory path for action gate: %r", raw, exc_info=True)
    return path_text.lstrip("/") if path_text.startswith("/") else path_text


def extract_required_memory_paths(rule_content: str, anima_dir: Path) -> list[str]:
    """Extract required ``read_memory_file(path=...)`` pointers from a rule."""
    paths: list[str] = []
    seen: set[str] = set()
    for match in _REQUIRED_READ_RE.finditer(rule_content):
        rel = _normalize_memory_path(match.group("path"), anima_dir)
        if rel and rel not in seen:
            seen.add(rel)
            paths.append(rel)
    return paths


def _state_dir(anima_dir: Path) -> Path:
    return anima_dir / "run" / "action_memory_gate"


def _session_key(explicit: str | None = None) -> str:
    if explicit:
        return _SAFE_SESSION_RE.sub("_", explicit)[:120] or "session"

    try:
        from core.execution.session_context import current_runtime_session

        ctx = current_runtime_session()
        if ctx is not None:
            return _SAFE_SESSION_RE.sub("_", ctx.tool_session_id or ctx.request_id)[:120] or "session"
    except Exception:
        logger.debug("Failed to read runtime session context for action gate", exc_info=True)

    env_key = os.environ.get("ANIMAWORKS_TOOL_SESSION_ID") or os.environ.get("ANIMAWORKS_REQUEST_ID")
    if env_key:
        return _SAFE_SESSION_RE.sub("_", env_key)[:120] or "session"
    return f"pid-{os.getpid()}"


def _state_path(anima_dir: Path, session_key: str | None = None) -> Path:
    return _state_dir(anima_dir) / f"{_session_key(session_key)}.json"


def _empty_state() -> dict[str, list[str]]:
    return {"read_paths": [], "shown_rules": []}


def _load_state(anima_dir: Path, session_key: str | None = None) -> dict[str, list[str]]:
    path = _state_path(anima_dir, session_key)
    try:
        if not path.is_file():
            return _empty_state()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_state()
        read_paths = data.get("read_paths", [])
        shown_rules = data.get("shown_rules", [])
        return {
            "read_paths": [str(p) for p in read_paths] if isinstance(read_paths, list) else [],
            "shown_rules": [str(r) for r in shown_rules] if isinstance(shown_rules, list) else [],
        }
    except Exception:
        logger.debug("Failed to load action gate state: %s", path, exc_info=True)
        return _empty_state()


def _save_state(anima_dir: Path, state: dict[str, list[str]], session_key: str | None = None) -> None:
    path = _state_path(anima_dir, session_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to save action gate state: %s", path, exc_info=True)


def record_memory_read(anima_dir: Path, path: str, *, session_key: str | None = None) -> str:
    """Record that a memory file was read in the current action-gate session."""
    rel = _normalize_memory_path(path, anima_dir)
    if not rel:
        return rel
    state = _load_state(anima_dir, session_key)
    read_paths = state.setdefault("read_paths", [])
    if rel not in read_paths:
        read_paths.append(rel)
        _save_state(anima_dir, state, session_key)
    return rel


def _get_retriever(anima_dir: Path) -> Any | None:
    knowledge_dir = anima_dir / "knowledge"
    if not knowledge_dir.is_dir():
        return None
    try:
        from core.memory.rag import MemoryRetriever
        from core.memory.rag.indexer import MemoryIndexer
        from core.memory.rag.singleton import get_vector_store

        vector_store = get_vector_store(anima_dir.name)
        if vector_store is None:
            return None
        indexer = MemoryIndexer(vector_store, anima_dir.name, anima_dir)
        return MemoryRetriever(vector_store, indexer, knowledge_dir)
    except Exception:
        logger.debug("Action memory gate retriever init failed", exc_info=True)
        return None


def _search_action_rules(anima_dir: Path, tool_name: str, query: str) -> list[Any]:
    retriever = _get_retriever(anima_dir)
    if retriever is None:
        return []
    return retriever.search_action_rules(tool_name, query, anima_dir.name)


def _json_query(tool_name: str, args: dict[str, Any] | None) -> str:
    try:
        args_text = json.dumps(args or {}, ensure_ascii=False, default=str)
    except TypeError:
        args_text = str(args or {})
    return f"{tool_name} {args_text[:500]}"


def check_action(
    anima_dir: Path,
    tool_name: str,
    args: dict[str, Any] | None = None,
    *,
    session_key: str | None = None,
) -> ActionMemoryGateDecision:
    """Check whether a side-effect action should pause for action memory."""
    if not tool_name:
        return ActionMemoryGateDecision(allowed=True, tool=tool_name)

    try:
        results = _search_action_rules(anima_dir, tool_name, _json_query(tool_name, args))
    except Exception:
        logger.debug("Action memory gate search failed", exc_info=True)
        return ActionMemoryGateDecision(allowed=True, tool=tool_name, reason="search_failed")

    if not results:
        return ActionMemoryGateDecision(allowed=True, tool=tool_name, reason="no_matching_rule")

    top_score = float(getattr(results[0], "score", 0.0) or 0.0)
    matching_rules: list[tuple[str, str, list[str], float]] = []
    for result in results:
        score = float(getattr(result, "score", 0.0) or 0.0)
        if score < 0.80:
            continue
        rule_id = str(getattr(result, "doc_id", "") or "")
        rule_content = str(getattr(result, "content", "") or "")
        required_paths = extract_required_memory_paths(rule_content, anima_dir)
        matching_rules.append((rule_id, rule_content, required_paths, score))

    if not matching_rules:
        return ActionMemoryGateDecision(allowed=True, tool=tool_name, reason="below_threshold", score=top_score)

    state = _load_state(anima_dir, session_key)
    read_paths = set(state.get("read_paths", []))
    for rule_id, rule_content, required_paths, score in matching_rules:
        missing_paths = [p for p in required_paths if p not in read_paths]
        if missing_paths:
            return ActionMemoryGateDecision(
                allowed=False,
                tool=tool_name,
                reason="missing_required_memory",
                rule_id=rule_id,
                rule=rule_content,
                required_paths=required_paths,
                missing_paths=missing_paths,
                score=score,
            )

    shown_rules = state.setdefault("shown_rules", [])
    for rule_id, rule_content, required_paths, score in matching_rules:
        rule_key = f"{tool_name}:{rule_id or rule_content[:80]}"
        if not required_paths and rule_key not in shown_rules:
            shown_rules.append(rule_key)
            _save_state(anima_dir, state, session_key)
            return ActionMemoryGateDecision(
                allowed=False,
                tool=tool_name,
                reason="review_rule_before_retry",
                rule_id=rule_id,
                rule=rule_content,
                required_paths=[],
                missing_paths=[],
                score=score,
            )

    all_required_paths: list[str] = []
    seen_paths: set[str] = set()
    for _, _, required_paths, _ in matching_rules:
        for path in required_paths:
            if path not in seen_paths:
                seen_paths.add(path)
                all_required_paths.append(path)

    first_rule_id, _, _, first_score = matching_rules[0]

    return ActionMemoryGateDecision(
        allowed=True,
        tool=tool_name,
        reason="required_memory_satisfied" if all_required_paths else "rule_already_shown",
        rule_id=first_rule_id,
        required_paths=all_required_paths,
        score=first_score,
    )
