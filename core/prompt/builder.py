from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


import logging
from pathlib import Path

from core.memory import MemoryManager
from core.paths import PROJECT_DIR, get_data_dir, load_prompt
from core.memory.shortterm import ShortTermMemory

logger = logging.getLogger("animaworks.prompt_builder")

# ── Emotion Instruction ───────────────────────────────────

def _build_emotion_instruction() -> str:
    """Build EMOTION_INSTRUCTION with the canonical emotion list."""
    from core.schemas import VALID_EMOTIONS
    emotion_list = ", ".join(sorted(VALID_EMOTIONS))
    return f"""\
## 表情メタデータ

応答の最後の行に、あなたの今の感情を以下の形式で付加してください。
この行はユーザーには表示されません。

<!-- emotion: {{"emotion": "<感情名>"}} -->

使える感情名: {emotion_list}

例:
- 嬉しい話題、褒められた時 → smile または laugh
- 難しい質問、困った時 → troubled
- 予想外の情報を聞いた時 → surprised
- じっくり考える必要がある時 → thinking
- 恥ずかしい話題、照れる時 → embarrassed
- 通常の会話 → neutral
"""


EMOTION_INSTRUCTION = _build_emotion_instruction()


def _discover_other_persons(person_dir: Path) -> list[str]:
    """List sibling person directories."""
    persons_root = person_dir.parent
    self_name = person_dir.name
    others = []
    for d in sorted(persons_root.iterdir()):
        if d.is_dir() and d.name != self_name and (d / "identity.md").exists():
            others.append(d.name)
    return others


# ── Organisation Context ─────────────────────────────────


def _format_person_entry(name: str, speciality: str | None) -> str:
    """Format a person name with optional speciality annotation."""
    if speciality:
        return f"{name} ({speciality})"
    return name


def _build_org_context(person_name: str, other_persons: list[str]) -> str:
    """Build organisation context section from supervisor chain.

    Reads config.json to derive each Person's relationship
    (supervisor / subordinate / peer) relative to *person_name*
    and returns a formatted prompt section.
    """
    from core.config import load_config

    try:
        config = load_config()
    except Exception:
        logger.debug("Could not load config for org context", exc_info=True)
        return ""

    all_persons = config.persons
    my_config = all_persons.get(person_name)
    my_supervisor = my_config.supervisor if my_config else None
    my_speciality = my_config.speciality if my_config else None

    # Supervisor
    if my_supervisor:
        sup_spec = None
        if my_supervisor in all_persons:
            sup_spec = all_persons[my_supervisor].speciality
        supervisor_line = _format_person_entry(my_supervisor, sup_spec)
    else:
        supervisor_line = "(なし — あなたがトップです)"

    # Subordinates: persons whose supervisor is me
    subordinates: list[str] = []
    for name in sorted(all_persons):
        if name == person_name:
            continue
        pcfg = all_persons[name]
        if pcfg.supervisor == person_name:
            subordinates.append(_format_person_entry(name, pcfg.speciality))

    # Peers: persons with the same supervisor (excluding self)
    peers: list[str] = []
    if my_supervisor is not None:
        for name in sorted(all_persons):
            if name == person_name:
                continue
            pcfg = all_persons[name]
            if pcfg.supervisor == my_supervisor:
                peers.append(_format_person_entry(name, pcfg.speciality))

    subordinates_line = ", ".join(subordinates) if subordinates else "(なし)"
    peers_line = ", ".join(peers) if peers else "(なし)"
    person_speciality = my_speciality or "(未設定)"

    parts = [
        load_prompt(
            "org_context",
            supervisor_line=supervisor_line,
            subordinates_line=subordinates_line,
            peers_line=peers_line,
            person_speciality=person_speciality,
        ),
    ]

    # Communication rules: only when there are other persons
    if other_persons:
        parts.append(load_prompt("communication_rules"))

    return "\n\n".join(parts)


def _build_messaging_section(
    person_dir: Path,
    other_persons: list[str],
) -> str:
    """Build the messaging instructions with resolved paths."""
    self_name = person_dir.name
    main_py = PROJECT_DIR / "main.py"
    persons_line = (
        ", ".join(other_persons) if other_persons else "(まだ他の社員はいません)"
    )

    return load_prompt(
        "messaging",
        persons_line=persons_line,
        main_py=main_py,
        self_name=self_name,
    )


def _load_a2_reflection() -> str:
    """Load the A2 reflection/retry prompt template."""
    try:
        return load_prompt("a2_reflection")
    except Exception:
        logger.debug("a2_reflection template not found, skipping")
        return ""


def build_system_prompt(
    memory: MemoryManager,
    tool_registry: list[str] | None = None,
    personal_tools: dict[str, str] | None = None,
    priming_section: str = "",
    execution_mode: str = "a1",
) -> str:
    """Construct the full system prompt from Markdown files.

    System prompt =
        environment (guardrails, folder structure, boundaries)
        + company vision
        + identity.md (who you are)
        + injection.md (role/philosophy)
        + permissions.md (what you can do)
        + state/current_task.md (what you're doing now)
        + priming memories (automatic recall) ← NEW
        + memory directory guide
        + personal skills + common skills
        + behavior rules (search-before-decide)
        + messaging instructions
    """
    parts: list[str] = []

    # Environment guardrails (always first)
    pd = memory.person_dir
    data_dir = get_data_dir()
    parts.append(load_prompt(
        "environment",
        data_dir=data_dir,
        person_name=pd.name,
    ))

    # Bootstrap instructions (highest priority after environment)
    bootstrap = memory.read_bootstrap()
    if bootstrap:
        parts.append(bootstrap)

    company_vision = memory.read_company_vision()
    if company_vision:
        parts.append(company_vision)

    identity = memory.read_identity()
    if identity:
        parts.append(identity)

    injection = memory.read_injection()
    if injection:
        parts.append(injection)

    permissions = memory.read_permissions()
    if permissions:
        parts.append(permissions)

    state = memory.read_current_state()
    if state:
        parts.append(f"## 現在の状態\n\n{state}")

    pending = memory.read_pending()
    if pending:
        parts.append(f"## 未完了タスク\n\n{pending}")

    # Priming section (automatic memory recall)
    if priming_section:
        parts.append(priming_section)

    # Memory directory guide
    knowledge_list = ", ".join(memory.list_knowledge_files()) or "(なし)"
    episode_list = ", ".join(memory.list_episode_files()[:7]) or "(なし)"
    procedure_list = ", ".join(memory.list_procedure_files()) or "(なし)"
    skill_summaries = memory.list_skill_summaries()
    common_skill_summaries = memory.list_common_skill_summaries()
    all_skill_names = [s[0] for s in skill_summaries] + [
        f"{s[0]}(共通)" for s in common_skill_summaries
    ]
    skill_names = ", ".join(all_skill_names) or "(なし)"

    shared_users_list = ", ".join(memory.list_shared_users()) or "(なし)"

    parts.append(load_prompt(
        "memory_guide",
        person_dir=pd,
        knowledge_list=knowledge_list,
        episode_list=episode_list,
        procedure_list=procedure_list,
        skill_names=skill_names,
        shared_users_list=shared_users_list,
    ))

    # Personal skills
    if skill_summaries:
        skill_lines = "\n".join(
            f"| {name} | {desc} |" for name, desc in skill_summaries
        )
        parts.append(load_prompt(
            "skills_guide",
            person_dir=pd,
            skill_lines=skill_lines,
        ))

    # Common skills (shared across all persons)
    if common_skill_summaries:
        common_skill_lines = "\n".join(
            f"| {name} | {desc} |" for name, desc in common_skill_summaries
        )
        common_skills_dir = memory.common_skills_dir
        parts.append(
            f"## 共通スキル\n\n"
            f"以下は全社員共通のスキルです。使用する際は "
            f"`{common_skills_dir}/{{スキル名}}.md` をReadで読んでから実行してください。\n\n"
            f"| スキル名 | 概要 |\n|---------|------|\n{common_skill_lines}"
        )

    # Inject dynamically generated external tools guide (filtered by registry)
    if permissions and "外部ツール" in permissions and (tool_registry or personal_tools):
        if execution_mode == "a2":
            # A2: guide users to discover_tools instead of CLI guide
            categories = ", ".join(sorted(tool_registry or []))
            if personal_tools:
                personal_cats = ", ".join(sorted(personal_tools.keys()))
                categories = f"{categories}, {personal_cats}" if categories else personal_cats
            parts.append(
                f"## 外部ツール\n\n"
                f"外部ツールを使うには `discover_tools` を呼んでください。\n"
                f"利用可能なカテゴリ: {categories}\n"
                f"カテゴリを指定して呼ぶとそのツール群が使えるようになります。"
            )
        else:
            # A1/B: CLI guide via animaworks-tool
            from core.tooling.guide import build_tools_guide
            tools_guide = build_tools_guide(tool_registry or [], personal_tools)
            if tools_guide:
                parts.append(tools_guide)

    # A2 reflection prompt for self-correction
    if execution_mode == "a2":
        reflection = _load_a2_reflection()
        if reflection:
            parts.append(reflection)

    # Emotion metadata instruction for bustup expression
    parts.append(EMOTION_INSTRUCTION)

    parts.append(load_prompt("behavior_rules"))

    # Organisation context (supervisor / subordinates / peers)
    other_persons = _discover_other_persons(pd)
    org_context = _build_org_context(pd.name, other_persons)
    if org_context:
        parts.append(org_context)

    # Messaging instructions
    parts.append(_build_messaging_section(pd, other_persons))

    # Hiring context: suggest team building when top-level person has no peers
    if not other_persons:
        try:
            model_config = memory.read_model_config()
            if model_config.supervisor is None:
                parts.append(load_prompt("hiring_context"))
        except Exception:
            logger.debug("Skipped hiring context injection", exc_info=True)

    logger.debug(
        "System prompt built: %d sections, total_len=%d",
        len(parts), sum(len(p) for p in parts),
    )
    return "\n\n---\n\n".join(parts)


def inject_shortterm(
    base_prompt: str,
    shortterm: ShortTermMemory,
) -> str:
    """Append short-term memory content to the system prompt.

    If the shortterm folder has a ``session_state.md``, its content is
    appended after a separator so the agent can pick up where it left off.
    """
    md_content = shortterm.load_markdown()
    if not md_content:
        return base_prompt
    return base_prompt + "\n\n---\n\n" + md_content