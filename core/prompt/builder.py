from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.memory import MemoryManager
from core.memory.manager import match_skills_by_description
from core.paths import PROJECT_DIR, get_data_dir, load_prompt
from core.memory.shortterm import ShortTermMemory
from core.schemas import SkillMeta

logger = logging.getLogger("animaworks.prompt_builder")


@dataclass
class BuildResult:
    """Result of system prompt building."""

    system_prompt: str
    injected_procedures: list[Path] = field(default_factory=list)

    def __str__(self) -> str:
        """Backward compatibility: str() returns prompt."""
        return self.system_prompt

    def __len__(self) -> int:
        """Backward compatibility: len() returns prompt length."""
        return len(self.system_prompt)

    def encode(self, encoding: str = "utf-8") -> bytes:
        """Backward compatibility: encode() encodes prompt."""
        return self.system_prompt.encode(encoding)

    def __contains__(self, item: str) -> bool:
        """Backward compatibility: 'x in result' checks prompt."""
        return item in self.system_prompt

    def __add__(self, other: str) -> str:
        """Backward compatibility: result + str concatenates prompt."""
        return self.system_prompt + other

    def __radd__(self, other: str) -> str:
        """Backward compatibility: str + result concatenates prompt."""
        return other + self.system_prompt

    def index(self, sub: str, *args: int) -> int:
        """Backward compatibility: result.index(x) searches prompt."""
        return self.system_prompt.index(sub, *args)

    def count(self, sub: str, *args: int) -> int:
        """Backward compatibility: result.count(x) counts in prompt."""
        return self.system_prompt.count(sub, *args)


# ── Skill Injection Budget ────────────────────────────────

SKILL_INJECTION_BUDGET: dict[str, int] = {
    "greeting": 1000,
    "question": 3000,
    "request": 5000,
    "heartbeat": 2000,
}


def _classify_message_for_skill_budget(message: str) -> str:
    """Classify message type for skill injection budget."""
    if not message:
        return "question"  # default
    msg_lower = message.lower()

    greeting_patterns = [
        "こんにちは", "おはよう", "こんばんは", "よろしく",
        "hello", "hi ", "hey", "good morning", "good evening",
    ]
    if any(p in msg_lower for p in greeting_patterns) and len(message) < 50:
        return "greeting"

    if len(message) > 100:
        return "request"

    question_patterns = [
        "?", "？", "教えて", "どう", "なぜ", "いつ", "どこ", "誰",
        "what", "why", "when", "where", "who", "how", "can you",
    ]
    if any(p in msg_lower for p in question_patterns):
        return "question"

    return "request"


def _build_skill_body(path: Path) -> str:
    """Read a skill file and return its body (after frontmatter)."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()

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

**重要**: 会話の内容に合わせて表情を積極的に変えてください。neutral以外の表情を優先的に選びましょう。

表情の選び方:
- smile: 相手の話に共感した時、良いニュースを聞いた時、挨拶の時、感謝された時
- laugh: 面白い話、ジョークを言う/聞いた時、楽しい雰囲気の時、嬉しい成果を報告する時
- troubled: 難しい問題に直面した時、相手の悩みを聞いている時、判断に迷う時、トラブル報告の時
- surprised: 予想外の情報、意外な展開、新しい発見をした時、驚くべき結果が出た時
- thinking: 分析・検討中の時、質問の意図を考えている時、計画を練っている時、比較検討中の時
- embarrassed: 褒められた時、失敗を認める時、個人的な話題の時、照れる内容の時
- neutral: 淡々とした事実伝達、定型的な確認応答のみ（迷ったらneutral以外を選ぶ）
"""


EMOTION_INSTRUCTION = _build_emotion_instruction()


def _discover_other_animas(anima_dir: Path) -> list[str]:
    """List sibling anima directories."""
    animas_root = anima_dir.parent
    self_name = anima_dir.name
    others = []
    for d in sorted(animas_root.iterdir()):
        if d.is_dir() and d.name != self_name and (d / "identity.md").exists():
            others.append(d.name)
    return others


# ── Organisation Context ─────────────────────────────────


def _format_anima_entry(name: str, speciality: str | None) -> str:
    """Format an anima name with optional speciality annotation."""
    if speciality:
        return f"{name} ({speciality})"
    return name


def _build_full_org_tree(
    anima_name: str,
    all_animas: dict[str, Any],
) -> str:
    """Build an indented full organization tree for top-level animas."""
    # Build children map: parent -> list of children
    children: dict[str | None, list[str]] = {}
    for name, pcfg in all_animas.items():
        parent = pcfg.supervisor
        children.setdefault(parent, []).append(name)
    for k in children:
        children[k].sort()

    lines: list[str] = []

    def _render(name: str, prefix: str, is_last: bool, is_root: bool) -> None:
        spec = all_animas[name].speciality if name in all_animas else None
        label = _format_anima_entry(name, spec)
        if is_root:
            marker = ""
            suffix = "  ← あなた" if name == anima_name else ""
        else:
            marker = "└── " if is_last else "├── "
            suffix = "  ← あなた" if name == anima_name else ""
        lines.append(f"{prefix}{marker}{label}{suffix}")
        kids = children.get(name, [])
        for i, child in enumerate(kids):
            child_is_last = (i == len(kids) - 1)
            if is_root:
                child_prefix = prefix
            else:
                child_prefix = prefix + ("    " if is_last else "│   ")
            _render(child, child_prefix, child_is_last, False)

    roots = children.get(None, [])
    for i, root in enumerate(roots):
        _render(root, "", i == len(roots) - 1, True)

    return "\n".join(lines)


def _build_org_context(anima_name: str, other_animas: list[str]) -> str:
    """Build organisation context section from supervisor chain.

    Reads config.json to derive each Anima's relationship
    (supervisor / subordinate / peer) relative to *anima_name*
    and returns a formatted prompt section.
    """
    from core.config import load_config

    try:
        config = load_config()
    except Exception:
        logger.debug("Could not load config for org context", exc_info=True)
        return ""

    all_animas = config.animas
    my_config = all_animas.get(anima_name)
    my_supervisor = my_config.supervisor if my_config else None
    my_speciality = my_config.speciality if my_config else None
    is_top_level = my_supervisor is None

    # Top-level anima with subordinates: show full org tree
    if is_top_level and len(all_animas) > 1:
        anima_speciality = my_speciality or "(未設定)"
        tree_text = _build_full_org_tree(anima_name, all_animas)
        parts = [
            f"## あなたの組織上の位置\n\n"
            f"あなたの専門: {anima_speciality}\n\n"
            f"あなたはトップレベルです（上司なし）。以下が組織全体の構成です：\n\n"
            f"```\n{tree_text}\n```",
        ]
        if other_animas:
            parts.append(load_prompt("communication_rules"))
        return "\n\n".join(parts)

    # Non-top-level: existing logic
    # Supervisor
    if my_supervisor:
        sup_spec = None
        if my_supervisor in all_animas:
            sup_spec = all_animas[my_supervisor].speciality
        supervisor_line = _format_anima_entry(my_supervisor, sup_spec)
    else:
        supervisor_line = "(なし — あなたがトップです)"

    # Subordinates: animas whose supervisor is me
    subordinates: list[str] = []
    for name in sorted(all_animas):
        if name == anima_name:
            continue
        pcfg = all_animas[name]
        if pcfg.supervisor == anima_name:
            subordinates.append(_format_anima_entry(name, pcfg.speciality))

    # Peers: animas with the same supervisor (excluding self)
    peers: list[str] = []
    if my_supervisor is not None:
        for name in sorted(all_animas):
            if name == anima_name:
                continue
            pcfg = all_animas[name]
            if pcfg.supervisor == my_supervisor:
                peers.append(_format_anima_entry(name, pcfg.speciality))

    subordinates_line = ", ".join(subordinates) if subordinates else "(なし)"
    peers_line = ", ".join(peers) if peers else "(なし)"
    anima_speciality = my_speciality or "(未設定)"

    parts = [
        load_prompt(
            "org_context",
            supervisor_line=supervisor_line,
            subordinates_line=subordinates_line,
            peers_line=peers_line,
            anima_speciality=anima_speciality,
        ),
    ]

    # Communication rules: only when there are other animas
    if other_animas:
        parts.append(load_prompt("communication_rules"))

    return "\n\n".join(parts)


def _build_messaging_section(
    anima_dir: Path,
    other_animas: list[str],
    execution_mode: str = "a1",
) -> str:
    """Build the messaging instructions with resolved paths."""
    self_name = anima_dir.name
    main_py = PROJECT_DIR / "main.py"
    animas_line = (
        ", ".join(other_animas) if other_animas else "(まだ他の社員はいません)"
    )

    template_name = "messaging_a1" if execution_mode == "a1" else "messaging"
    return load_prompt(
        template_name,
        animas_line=animas_line,
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


def _build_human_notification_guidance() -> str:
    """Build the human notification instruction for top-level Animas."""
    return """\
## 人間への連絡

あなたはトップレベルのPersonです（上司なし）。
重要な事項は `call_human` ツールで人間の管理者に連絡してください。
連絡内容はチャット画面と外部通知チャネル（Slack等）の両方に届きます。
部下からのエスカレーションを受けた場合、内容を判断し、重要であれば人間に連絡してください。
大したことがなければ自分の判断で対応を完了して構いません。

**連絡すべき場合:**
- 問題・エラー・障害の検出
- 判断が必要な事項
- 重要なタスクの完了報告
- 部下からのエスカレーション

**連絡不要な場合:**
- 定常的な巡回で特に問題がなかった場合
- 軽微な自動修復が完了した場合

判断に迷う場合は連絡してください。"""


def build_system_prompt(
    memory: MemoryManager,
    tool_registry: list[str] | None = None,
    personal_tools: dict[str, str] | None = None,
    priming_section: str = "",
    execution_mode: str = "a1",
    message: str = "",
    retriever: object | None = None,
) -> BuildResult:
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
    pd = memory.anima_dir
    data_dir = get_data_dir()
    parts.append(load_prompt(
        "environment",
        data_dir=data_dir,
        anima_name=pd.name,
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

    # Role-specific specialty prompt (injected between injection and permissions)
    specialty = memory.read_specialty_prompt()
    if specialty:
        parts.append(specialty)

    permissions = memory.read_permissions()
    if permissions:
        parts.append(permissions)

    state = memory.read_current_state()
    if state and state.strip() != "status: idle":
        parts.append(
            "## ⚠️ 進行中タスク（MUST: 最優先で確認すること）\n\n"
            "以下のタスクが進行中です。状態を確認し、このタスクの続きから開始してください。\n"
            "「idle」「待機中」と判定する前に、必ずこの内容を確認すること。\n\n"
            f"{state}"
        )
    elif state:
        parts.append(f"## 現在の状態\n\n{state}")

    pending = memory.read_pending()
    if pending:
        parts.append(f"## 未完了タスク\n\n{pending}")

    # Resolution registry injection (cross-org resolved issues)
    try:
        resolutions = memory.read_resolutions(days=7)
        if resolutions:
            res_lines = []
            for r in resolutions[-10:]:  # Latest 10 entries
                ts_short = r.get("ts", "")[:16]  # YYYY-MM-DDTHH:MM
                resolver = r.get("resolver", "unknown")
                issue = r.get("issue", "")
                res_lines.append(f"- [{ts_short}] {resolver}: {issue}")
            parts.append(
                "## 解決済み案件（組織横断）\n\n"
                "以下は直近7日間に解決された案件です。"
                "これらの問題については再調査・再報告は不要です。\n\n"
                + "\n".join(res_lines)
            )
    except Exception:
        logger.debug("Failed to inject resolution registry", exc_info=True)

    # Priming section (automatic memory recall)
    if priming_section:
        parts.append(priming_section)

    # Memory directory guide
    knowledge_list = ", ".join(memory.list_knowledge_files()) or "(なし)"
    episode_list = ", ".join(memory.list_episode_files()[:7]) or "(なし)"
    procedure_list = ", ".join(memory.list_procedure_files()) or "(なし)"
    skill_metas = memory.list_skill_metas()
    common_skill_metas = memory.list_common_skill_metas()
    all_metas = skill_metas + common_skill_metas
    all_skill_names = [m.name for m in skill_metas] + [
        f"{m.name}(共通)" for m in common_skill_metas
    ]
    skill_names = ", ".join(all_skill_names) or "(なし)"

    shared_users_list = ", ".join(memory.list_shared_users()) or "(なし)"

    parts.append(load_prompt(
        "memory_guide",
        anima_dir=pd,
        knowledge_list=knowledge_list,
        episode_list=episode_list,
        procedure_list=procedure_list,
        skill_names=skill_names,
        shared_users_list=shared_users_list,
    ))

    # Common knowledge reference hint
    common_knowledge_dir = data_dir / "common_knowledge"
    if common_knowledge_dir.exists() and any(common_knowledge_dir.rglob("*.md")):
        parts.append(
            "## 共有リファレンス\n\n"
            "困ったとき・手順が不明なときは `common_knowledge/` を "
            "`search_memory` で検索するか、`read_memory_file` で直接読んでください。\n"
            "目次: `common_knowledge/00_index.md`"
        )

    # ── Skill + procedure injection (description-based matching) ──────
    procedure_metas = memory.list_procedure_metas()
    anima_name = memory.anima_dir.name if hasattr(memory, "anima_dir") else ""
    all_metas_with_procedures = all_metas + procedure_metas
    matched_skills = match_skills_by_description(
        message, all_metas_with_procedures, retriever=retriever, anima_name=anima_name,
    )
    matched_names: set[str] = set()
    msg_type = _classify_message_for_skill_budget(message)
    budget = SKILL_INJECTION_BUDGET.get(msg_type, 3000)
    used_tokens = 0

    # Track injected procedures for outcome tracking (Phase 3)
    injected_procedure_paths: list[Path] = []

    # Inject matched skill/procedure full text (within budget)
    procedure_name_set = {m.name for m in procedure_metas}
    for skill in matched_skills:
        body = _build_skill_body(skill.path)
        # Rough token estimate: 1 char ≈ 1 token for CJK text
        body_len = len(body)
        if used_tokens + body_len > budget:
            break
        is_procedure = skill.name in procedure_name_set
        if is_procedure:
            label = "(手順)"
            injected_procedure_paths.append(skill.path)
        elif skill.is_common:
            label = "(共通スキル)"
        else:
            label = "(個人スキル)"
        section_title = "手順" if is_procedure else "スキル"
        parts.append(
            f"## {section_title}: {skill.name} {label}\n\n"
            f"以下の{section_title}がこのメッセージに該当します。手順に従って実行してください。\n\n"
            f"{body}"
        )
        matched_names.add(skill.name)
        used_tokens += body_len

    # injected_procedure_paths is returned in BuildResult for the caller
    # to pass to finalize_session() (replaces former module-level dict).

    # Non-matched personal skills → table
    unmatched_personal = [
        m for m in skill_metas if m.name not in matched_names
    ]
    if unmatched_personal:
        skill_lines = "\n".join(
            f"| {m.name} | {m.description} |" for m in unmatched_personal
        )
        parts.append(load_prompt(
            "skills_guide",
            anima_dir=pd,
            skill_lines=skill_lines,
        ))

    # Non-matched common skills → table
    unmatched_common = [
        m for m in common_skill_metas if m.name not in matched_names
    ]
    if unmatched_common:
        common_skill_lines = "\n".join(
            f"| {m.name} | {m.description} |" for m in unmatched_common
        )
        common_skills_dir = memory.common_skills_dir
        parts.append(
            f"## 共通スキル\n\n"
            f"以下は全社員共通のスキルです。使用する際は "
            f"`{common_skills_dir}/{{スキル名}}.md` をReadで読んでから実行してください。\n\n"
            f"| スキル名 | 概要 |\n|---------|------|\n{common_skill_lines}"
        )

    # Non-matched procedures → table
    unmatched_procedures = [
        m for m in procedure_metas if m.name not in matched_names
    ]
    if unmatched_procedures:
        proc_lines = "\n".join(
            f"| {m.name} | {m.description} |" for m in unmatched_procedures
        )
        parts.append(
            f"## 手順書\n\n"
            f"以下は個人の手順書です。使用する際は `procedures/{{手順名}}.md` をReadで読んでから実行してください。\n\n"
            f"| 手順名 | 概要 |\n|---------|------|\n{proc_lines}"
        )

    # Commander hiring guardrail: force create_anima tool/CLI usage
    has_newstaff = any(m.name == "newstaff" for m in skill_metas)
    if has_newstaff:
        if execution_mode == "a1":
            parts.append(
                "## 雇用ルール\n\n"
                "新しいAnimaを雇用する際は、以下の手順に従ってください。\n"
                "手動で identity.md 等のファイルを個別に作成してはいけません。\n\n"
                "1. キャラクターシートを1ファイルのMarkdownとして作成する\n"
                "   - 必須セクション: `## 基本情報`, `## 人格`, `## 役割・行動方針`\n"
                "2. Bashで以下のコマンドを実行する:\n"
                "   ```\n"
                "   animaworks create-anima --from-md <キャラクターシートのパス>"
                " --supervisor $(basename $ANIMAWORKS_ANIMA_DIR)\n"
                "   ```\n"
                "3. サーバーのReconciliationが自動的に新Animaを検出・起動します"
            )
        else:
            parts.append(
                "## 雇用ルール\n\n"
                "新しいAnimaを雇用する際は、必ず `create_anima` ツールを使用してください。\n"
                "手動で identity.md 等のファイルを個別に作成してはいけません。\n"
                "キャラクターシートを1ファイルで作成し、create_anima に渡してください。"
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

    # Organisation context (supervisor / subordinates / peers)
    other_animas = _discover_other_animas(pd)

    # Hiring context: suggest team building when top-level anima has no peers
    # Placed before behavior_rules so the directive is not buried at the end.
    if not other_animas:
        try:
            model_config = memory.read_model_config()
            if model_config.supervisor is None:
                parts.append(load_prompt("hiring_context"))
        except Exception:
            logger.debug("Skipped hiring context injection", exc_info=True)

    parts.append(load_prompt("behavior_rules"))

    org_context = _build_org_context(pd.name, other_animas)
    if org_context:
        parts.append(org_context)

    # Messaging instructions
    parts.append(_build_messaging_section(pd, other_animas, execution_mode))

    # Human notification guidance for top-level Animas
    try:
        from core.config import load_config as _load_cfg
        _cfg = _load_cfg()
        _my_pcfg = _cfg.animas.get(pd.name)
        _is_top_level = _my_pcfg is None or _my_pcfg.supervisor is None
        if _is_top_level and _cfg.human_notification.enabled:
            parts.append(_build_human_notification_guidance())
    except Exception:
        logger.debug("Skipped human notification guidance injection", exc_info=True)

    prompt = "\n\n---\n\n".join(parts)
    logger.debug(
        "System prompt built: %d sections, total_len=%d",
        len(parts), len(prompt),
    )
    return BuildResult(
        system_prompt=prompt,
        injected_procedures=injected_procedure_paths,
    )


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