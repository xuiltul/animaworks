from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Session-scoped action memory gate for external side-effect tools.

Failure mode (``config.action_gate.fail_mode``):

- ``open`` (default): legacy fail-open. All three soft-fail cases still
  allow execution, but always emit structured observability logs
  (action, anima, fail kind, fail_mode, would_block).
- ``middle``: block only on ``search_failed``; ``no_matching_rule`` warns
  and passes; below-threshold rules go through the normal read/review
  flow instead of pass-through.
- ``close``: block on ``search_failed``; hold ``no_matching_rule`` unless
  explicitly allowed; below-threshold uses the read/review flow.

Default is ``open`` because this fleet has frequent vector-search
outages (FD exhaustion, repair loops, CUDA failures). Immediate
fail-close would halt external sends during infrastructure incidents.
Observe logs, then migrate open → middle → close.
See ``docs/specs/pi-fix3-action-gate-fail-mode.md``.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("animaworks.action_memory_gate")

_REQUIRED_READ_RE = re.compile(r"""read_memory_file\s*\(\s*path\s*=\s*(?P<quote>['"])(?P<path>.+?)(?P=quote)\s*\)""")
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9_.-]+")

FailMode = Literal["open", "middle", "close"]
FailKind = Literal["no_matching_rule", "search_failed", "below_threshold"]

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

# Score threshold for treating a retrieved ACTION-RULE as matching.
_MATCH_SCORE_THRESHOLD = 0.80


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
    fail_mode: str = ""
    would_block: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Return a structured payload suitable for tool output or stderr."""
        if self.allowed:
            return {
                "status": "ok",
                "tool": self.tool,
                "message": "Action allowed by memory gate",
                "reason": self.reason,
                "fail_mode": self.fail_mode,
                "would_block": self.would_block,
            }
        payload: dict[str, Any] = {
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
            "fail_mode": self.fail_mode,
            "would_block": self.would_block,
        }
        if self.reason == "no_matching_rule":
            payload["message"] = (
                "Action held by memory gate: no matching ACTION-RULE knowledge. "
                "Add an [ACTION-RULE] for this tool, or call grant_no_rule_allow "
                "after human review to release the hold for this session."
            )
        elif self.reason == "search_failed":
            payload["message"] = (
                "Action blocked by memory gate: action-rule search failed "
                "(infrastructure). Retry after RAG/vector store recovers."
            )
        return payload

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


def _notify_state_path(anima_dir: Path) -> Path:
    """Cross-session notify dedup state (per anima, not per tool session)."""
    return _state_dir(anima_dir) / "no_rule_notify.json"


def _empty_state() -> dict[str, Any]:
    return {
        "read_paths": [],
        "shown_rules": [],
        "no_rule_allows": [],
    }


def _load_state(anima_dir: Path, session_key: str | None = None) -> dict[str, Any]:
    path = _state_path(anima_dir, session_key)
    try:
        if not path.is_file():
            return _empty_state()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_state()
        read_paths = data.get("read_paths", [])
        shown_rules = data.get("shown_rules", [])
        no_rule_allows = data.get("no_rule_allows", [])
        return {
            "read_paths": [str(p) for p in read_paths] if isinstance(read_paths, list) else [],
            "shown_rules": [str(r) for r in shown_rules] if isinstance(shown_rules, list) else [],
            "no_rule_allows": ([str(t) for t in no_rule_allows] if isinstance(no_rule_allows, list) else []),
        }
    except Exception:
        logger.debug("Failed to load action gate state: %s", path, exc_info=True)
        return _empty_state()


def _save_state(anima_dir: Path, state: dict[str, Any], session_key: str | None = None) -> None:
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


def grant_no_rule_allow(
    anima_dir: Path,
    tool_name: str,
    *,
    session_key: str | None = None,
) -> None:
    """Explicitly allow ``tool_name`` when no ACTION-RULE matches (close mode).

    This is the hold-release path for ``no_matching_rule`` under
    ``fail_mode=close``. Prefer adding real ``[ACTION-RULE]`` knowledge;
    use this only after human review for operational continuity.
    """
    if not tool_name:
        return
    state = _load_state(anima_dir, session_key)
    allows = state.setdefault("no_rule_allows", [])
    if tool_name not in allows:
        allows.append(tool_name)
        _save_state(anima_dir, state, session_key)
        logger.info(
            "action_gate_no_rule_allow_granted anima=%s tool=%s session=%s",
            anima_dir.name,
            tool_name,
            _session_key(session_key),
        )


def revoke_no_rule_allow(
    anima_dir: Path,
    tool_name: str,
    *,
    session_key: str | None = None,
) -> None:
    """Remove a previous :func:`grant_no_rule_allow` for ``tool_name``."""
    state = _load_state(anima_dir, session_key)
    allows = state.get("no_rule_allows", [])
    if tool_name in allows:
        allows.remove(tool_name)
        state["no_rule_allows"] = allows
        _save_state(anima_dir, state, session_key)


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


def _resolve_fail_mode() -> tuple[FailMode, int]:
    """Load fail_mode and notify cooldown from config.

    Default is ``open`` (see module docstring / ActionGateConfig).
    """
    try:
        from core.config import load_config

        cfg = load_config().action_gate
        mode = getattr(cfg, "fail_mode", "open") or "open"
        if mode not in ("open", "middle", "close"):
            mode = "open"
        cooldown = int(getattr(cfg, "no_rule_notify_cooldown_seconds", 21600) or 0)
        return mode, max(0, cooldown)  # type: ignore[return-value]
    except Exception:
        logger.debug("Failed to load action_gate config; defaulting to open", exc_info=True)
        return "open", 21600


def _log_soft_fail(
    *,
    anima_name: str,
    tool: str,
    fail_kind: FailKind,
    fail_mode: FailMode,
    would_block: bool,
    blocked: bool,
    score: float = 0.0,
    extra: str = "",
) -> None:
    """Emit structured observability for soft-fail cases (always, all modes)."""
    logger.warning(
        "action_gate_soft_fail anima=%s tool=%s fail_kind=%s fail_mode=%s would_block=%s blocked=%s score=%.4f %s",
        anima_name,
        tool,
        fail_kind,
        fail_mode,
        would_block,
        blocked,
        score,
        extra,
    )


def _would_block_in_close(fail_kind: FailKind) -> bool:
    """Whether close mode would block/hold this fail kind."""
    return fail_kind in ("no_matching_rule", "search_failed", "below_threshold")


def _load_notify_state(anima_dir: Path) -> dict[str, float]:
    path = _notify_state_path(anima_dir)
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float, str))}
    except Exception:
        logger.debug("Failed to load no_rule notify state", exc_info=True)
        return {}


def _save_notify_state(anima_dir: Path, state: dict[str, float]) -> None:
    path = _notify_state_path(anima_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to save no_rule notify state", exc_info=True)


def _maybe_notify_no_matching_rule(
    anima_dir: Path,
    tool_name: str,
    *,
    cooldown_seconds: int,
) -> bool:
    """Send a human-facing notification for no_matching_rule (deduped).

    Returns True if a notification was attempted (not suppressed by cooldown).
    Dedup is per (anima, tool). Cooldown ensures holds are not silent forever
    if the first alert was missed (IaC freeze precedent 2026-07-12).
    """
    now = time.time()
    state = _load_notify_state(anima_dir)
    last = state.get(tool_name, 0.0)
    if cooldown_seconds > 0 and last and (now - last) < cooldown_seconds:
        logger.info(
            "action_gate_no_rule_notify_suppressed anima=%s tool=%s cooldown_remaining=%.0fs",
            anima_dir.name,
            tool_name,
            cooldown_seconds - (now - last),
        )
        return False

    state[tool_name] = now
    _save_notify_state(anima_dir, state)

    subject = f"[ActionGate] No ACTION-RULE for {tool_name} ({anima_dir.name})"
    body = (
        f"Anima: {anima_dir.name}\n"
        f"Tool: {tool_name}\n"
        f"Reason: no_matching_rule (fail_mode=close)\n"
        f"Action held until an [ACTION-RULE] is added for this tool, "
        f"or grant_no_rule_allow is used after human review.\n"
        f"Release path: core.memory.action_gate.grant_no_rule_allow("
        f"anima_dir, {tool_name!r}) or add knowledge with trigger_tools: {tool_name}."
    )
    # Always leave a durable trail on disk (works even without HumanNotifier).
    try:
        trail = _state_dir(anima_dir) / "no_rule_holds.jsonl"
        trail.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": now,
            "anima": anima_dir.name,
            "tool": tool_name,
            "subject": subject,
        }
        with trail.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to append no_rule hold trail", exc_info=True)

    try:
        from core.config import load_config
        from core.notification.notifier import HumanNotifier

        config = load_config()
        hn = getattr(config, "human_notification", None)
        if hn is not None and getattr(hn, "enabled", False):
            notifier = HumanNotifier.from_config(hn)
            if notifier.channel_count > 0:
                from core.skills.promotion_approval import run_coroutine_sync

                run_coroutine_sync(
                    notifier.notify(
                        subject,
                        body,
                        "high",
                        anima_name=anima_dir.name,
                    )
                )
                logger.info(
                    "action_gate_no_rule_notified anima=%s tool=%s channels=%d",
                    anima_dir.name,
                    tool_name,
                    notifier.channel_count,
                )
                return True
    except Exception:
        logger.debug("HumanNotifier path failed for no_matching_rule", exc_info=True)

    logger.warning(
        "action_gate_no_rule_notify_logged_only anima=%s tool=%s subject=%s",
        anima_dir.name,
        tool_name,
        subject,
    )
    return True


def _rules_from_results(
    results: list[Any],
    anima_dir: Path,
    *,
    enforce_threshold: bool,
) -> tuple[list[tuple[str, str, list[str], float]], float]:
    """Build matching rule tuples from search results.

    When ``enforce_threshold`` is True, only rules with score >= threshold
    are included. When False (middle/close below-threshold path), all
    results are included so the existing read/review flow applies.
    """
    top_score = float(getattr(results[0], "score", 0.0) or 0.0) if results else 0.0
    matching_rules: list[tuple[str, str, list[str], float]] = []
    for result in results:
        score = float(getattr(result, "score", 0.0) or 0.0)
        if enforce_threshold and score < _MATCH_SCORE_THRESHOLD:
            continue
        rule_id = str(getattr(result, "doc_id", "") or "")
        rule_content = str(getattr(result, "content", "") or "")
        required_paths = extract_required_memory_paths(rule_content, anima_dir)
        matching_rules.append((rule_id, rule_content, required_paths, score))
    return matching_rules, top_score


def _evaluate_matching_rules(
    anima_dir: Path,
    tool_name: str,
    matching_rules: list[tuple[str, str, list[str], float]],
    *,
    session_key: str | None,
    fail_mode: FailMode,
) -> ActionMemoryGateDecision:
    """Apply required-memory / review-once logic to matching rules."""
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
                fail_mode=fail_mode,
                would_block=True,
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
                fail_mode=fail_mode,
                would_block=True,
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
        fail_mode=fail_mode,
        would_block=False,
    )


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

    fail_mode, notify_cooldown = _resolve_fail_mode()
    anima_name = anima_dir.name

    try:
        results = _search_action_rules(anima_dir, tool_name, _json_query(tool_name, args))
    except Exception:
        logger.debug("Action memory gate search failed", exc_info=True)
        would_block = _would_block_in_close("search_failed")
        # middle and close both block search_failed
        blocked = fail_mode in ("middle", "close")
        _log_soft_fail(
            anima_name=anima_name,
            tool=tool_name,
            fail_kind="search_failed",
            fail_mode=fail_mode,
            would_block=would_block,
            blocked=blocked,
            extra="trigger=exception",
        )
        if blocked:
            return ActionMemoryGateDecision(
                allowed=False,
                tool=tool_name,
                reason="search_failed",
                fail_mode=fail_mode,
                would_block=would_block,
            )
        return ActionMemoryGateDecision(
            allowed=True,
            tool=tool_name,
            reason="search_failed",
            fail_mode=fail_mode,
            would_block=would_block,
        )

    if not results:
        return _handle_no_matching_rule(
            anima_dir,
            tool_name,
            fail_mode=fail_mode,
            notify_cooldown=notify_cooldown,
            session_key=session_key,
        )

    matching_rules, top_score = _rules_from_results(
        results,
        anima_dir,
        enforce_threshold=True,
    )

    if not matching_rules:
        # Below threshold: open allows; middle/close continue read/review flow.
        would_block = _would_block_in_close("below_threshold")
        if fail_mode in ("middle", "close"):
            _log_soft_fail(
                anima_name=anima_name,
                tool=tool_name,
                fail_kind="below_threshold",
                fail_mode=fail_mode,
                would_block=would_block,
                blocked=True,  # will enter read/review (not pass-through)
                score=top_score,
                extra="trigger=enforce_read_review",
            )
            soft_rules, _ = _rules_from_results(
                results,
                anima_dir,
                enforce_threshold=False,
            )
            if soft_rules:
                return _evaluate_matching_rules(
                    anima_dir,
                    tool_name,
                    soft_rules,
                    session_key=session_key,
                    fail_mode=fail_mode,
                )
            # No usable content — treat as no matching rule
            return _handle_no_matching_rule(
                anima_dir,
                tool_name,
                fail_mode=fail_mode,
                notify_cooldown=notify_cooldown,
                session_key=session_key,
            )

        _log_soft_fail(
            anima_name=anima_name,
            tool=tool_name,
            fail_kind="below_threshold",
            fail_mode=fail_mode,
            would_block=would_block,
            blocked=False,
            score=top_score,
            extra="trigger=pass_through",
        )
        return ActionMemoryGateDecision(
            allowed=True,
            tool=tool_name,
            reason="below_threshold",
            score=top_score,
            fail_mode=fail_mode,
            would_block=would_block,
        )

    return _evaluate_matching_rules(
        anima_dir,
        tool_name,
        matching_rules,
        session_key=session_key,
        fail_mode=fail_mode,
    )


def _handle_no_matching_rule(
    anima_dir: Path,
    tool_name: str,
    *,
    fail_mode: FailMode,
    notify_cooldown: int,
    session_key: str | None,
) -> ActionMemoryGateDecision:
    """Handle the no_matching_rule soft-fail case per fail_mode."""
    anima_name = anima_dir.name
    would_block = _would_block_in_close("no_matching_rule")

    if fail_mode == "close":
        state = _load_state(anima_dir, session_key)
        allows = set(state.get("no_rule_allows", []))
        if tool_name in allows:
            _log_soft_fail(
                anima_name=anima_name,
                tool=tool_name,
                fail_kind="no_matching_rule",
                fail_mode=fail_mode,
                would_block=would_block,
                blocked=False,
                extra="trigger=explicit_allow",
            )
            return ActionMemoryGateDecision(
                allowed=True,
                tool=tool_name,
                reason="no_matching_rule_allowed",
                fail_mode=fail_mode,
                would_block=would_block,
            )

        _log_soft_fail(
            anima_name=anima_name,
            tool=tool_name,
            fail_kind="no_matching_rule",
            fail_mode=fail_mode,
            would_block=would_block,
            blocked=True,
            extra="trigger=hold",
        )
        _maybe_notify_no_matching_rule(
            anima_dir,
            tool_name,
            cooldown_seconds=notify_cooldown,
        )
        return ActionMemoryGateDecision(
            allowed=False,
            tool=tool_name,
            reason="no_matching_rule",
            fail_mode=fail_mode,
            would_block=would_block,
        )

    # open and middle: allow with structured log (middle warns via same log)
    _log_soft_fail(
        anima_name=anima_name,
        tool=tool_name,
        fail_kind="no_matching_rule",
        fail_mode=fail_mode,
        would_block=would_block,
        blocked=False,
        extra="trigger=pass_through",
    )
    return ActionMemoryGateDecision(
        allowed=True,
        tool=tool_name,
        reason="no_matching_rule",
        fail_mode=fail_mode,
        would_block=would_block,
    )
