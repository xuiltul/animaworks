from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


"""Mode B executor: assisted (1-shot, framework handles memory I/O).

The framework reads memory, injects context, calls the LLM once without
tools, then records the episode, extracts knowledge, and optionally
sends reply/report messages on behalf of the anima.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.prompt.context import ContextTracker
from core.execution.base import BaseExecutor, ExecutionResult
from core.memory import MemoryManager
from core.messenger import Messenger
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory

logger = logging.getLogger("animaworks.execution.assisted")


# ── Post-call prompts ──────────────────────────────────────

POST_CALL_PROMPT_MESSAGE = """\
以下のやりとりを振り返り、回答してください。

## 知識抽出
このやりとりから学ぶべきことはありますか？
あれば簡潔に書いてください。なければ「なし」と書いてください。

## 返信判定
受信メッセージに対して返信が必要ですか？
- 質問・指示・依頼・確認事項がある → 「返信: （内容）」
- お礼・挨拶・了解など会話の終了 → 「返信不要」
"""

POST_CALL_PROMPT_HEARTBEAT = """\
以下のやりとりを振り返り、回答してください。

## 知識抽出
このやりとりから学ぶべきことはありますか？
あれば簡潔に書いてください。なければ「なし」と書いてください。

## 報告判定
この結果を上司（{supervisor}）に報告する必要がありますか？
- 異常・問題・重要な変化があった場合 → 「報告: （内容）」
- 特に報告すべきことがない場合 → 「報告不要」
"""


@dataclass
class PostCallResult:
    """Result of the post-call LLM judgement."""

    knowledge: str | None = None
    send_needed: bool = False
    send_content: str = ""


class AssistedExecutor(BaseExecutor):
    """Execute in assisted mode (Mode B).

    Flow:
      1. Pre-call:  inject identity + recent episodes + keyword-matched knowledge
      2. LLM 1-shot call (no tools)
      3. Post-call: record episode
      4. Post-call: knowledge extraction + send judgement (unified 1-shot)
      5. If send needed: framework sends message on behalf of the anima
    """

    def __init__(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
        memory: MemoryManager,
        messenger: Messenger | None = None,
    ) -> None:
        super().__init__(model_config, anima_dir)
        self._memory = memory
        self._messenger = messenger

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
    ) -> Any:
        """Call LiteLLM ``acompletion`` and return the raw response."""
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._model_config.model,
            "messages": messages,
            "max_tokens": self._model_config.max_tokens,
        }

        if system:
            kwargs["messages"] = [
                {"role": "system", "content": system}
            ] + messages

        api_key = self._resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        if self._model_config.api_base_url:
            kwargs["api_base"] = self._model_config.api_base_url

        return await litellm.acompletion(**kwargs)

    async def _post_call(
        self, trigger: str, prompt: str, reply: str,
    ) -> PostCallResult:
        """Post-call: knowledge extraction + send judgement in 1 LLM call."""
        is_message = trigger.startswith("message:")
        supervisor = self._model_config.supervisor or ""

        if is_message:
            post_prompt = POST_CALL_PROMPT_MESSAGE
        else:
            post_prompt = POST_CALL_PROMPT_HEARTBEAT.format(supervisor=supervisor)

        extract_messages = [
            {
                "role": "user",
                "content": (
                    f"{post_prompt}\n\n"
                    f"## やりとり\n質問: {prompt[:1000]}\n\n回答: {reply[:1000]}"
                ),
            }
        ]
        extract_resp = await self._call_llm(extract_messages)
        text = extract_resp.choices[0].message.content or ""

        result = PostCallResult()

        # Parse knowledge extraction
        knowledge_match = re.search(
            r"## 知識抽出\s*\n(.+?)(?=\n## |\Z)", text, re.DOTALL,
        )
        if knowledge_match:
            knowledge_text = knowledge_match.group(1).strip()
            if knowledge_text and knowledge_text != "なし":
                result.knowledge = knowledge_text

        # Parse send judgement
        reply_match = re.search(r"返信:\s*(.+)", text)
        report_match = re.search(r"報告:\s*(.+)", text)
        if reply_match:
            result.send_needed = True
            result.send_content = reply_match.group(1).strip()
        elif report_match:
            result.send_needed = True
            result.send_content = report_match.group(1).strip()

        return result

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
    ) -> ExecutionResult:
        """Run the assisted execution flow."""
        logger.info("_run_assisted START prompt_len=%d trigger=%s", len(prompt), trigger)

        # ── 1. Pre-call: gather context ──────────────────
        identity = self._memory.read_identity()
        injection = self._memory.read_injection()
        recent_episodes = self._memory.read_recent_episodes(days=7)

        # Simple keyword extraction for knowledge search
        keywords = set(re.findall(r"[\w]{3,}", prompt))
        knowledge_hits: list[str] = []
        for kw in list(keywords)[:10]:
            for fname, line in self._memory.search_memory_text(kw, scope="knowledge"):
                knowledge_hits.append(f"[{fname}] {line}")
        knowledge_context = "\n".join(dict.fromkeys(knowledge_hits))  # dedupe

        # Build enriched system prompt
        system_parts = [identity, injection]
        if recent_episodes:
            system_parts.append(f"## 直近の行動ログ\n\n{recent_episodes[:4000]}")
        if knowledge_context:
            system_parts.append(f"## 関連知識\n\n{knowledge_context[:4000]}")

        # Skills injection (personal + common)
        skill_summaries = self._memory.list_skill_summaries()
        common_skill_summaries = self._memory.list_common_skill_summaries()
        if skill_summaries:
            skill_lines = "\n".join(
                f"| {name} | {desc} |" for name, desc in skill_summaries
            )
            system_parts.append(
                f"## 個人スキル\n\n"
                f"使用する際は skills/{{スキル名}}.md をReadで読んでから実行してください。\n\n"
                f"| スキル名 | 概要 |\n|---------|------|\n{skill_lines}"
            )
        if common_skill_summaries:
            common_skill_lines = "\n".join(
                f"| {name} | {desc} |" for name, desc in common_skill_summaries
            )
            common_skills_dir = self._memory.common_skills_dir
            system_parts.append(
                f"## 共通スキル\n\n"
                f"以下は全社員共通のスキルです。使用する際は "
                f"`{common_skills_dir}/{{スキル名}}.md` をReadで読んでから実行してください。\n\n"
                f"| スキル名 | 概要 |\n|---------|------|\n{common_skill_lines}"
            )

        system = "\n\n---\n\n".join(p for p in system_parts if p)

        # ── 2. LLM 1-shot call ───────────────────────────
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages, system=system)
        reply = response.choices[0].message.content or ""
        logger.info("_run_assisted LLM replied, len=%d", len(reply))

        # ── 3. Post-call: record episode ─────────────────
        ts = datetime.now().strftime("%H:%M")
        episode = f"- {ts} [assisted] prompt: {prompt[:200]}… → reply: {reply[:200]}…"
        self._memory.append_episode(episode)

        # ── 4. Post-call: knowledge extraction + send judgement ─
        try:
            post = await self._post_call(trigger, prompt, reply)

            if post.knowledge:
                topic = datetime.now().strftime("learned_%Y%m%d_%H%M%S")
                self._memory.write_knowledge(topic, post.knowledge)
                logger.info("Knowledge extracted: %s", post.knowledge[:100])

            # ── 5. Send message on behalf of the anima ──
            if post.send_needed and self._messenger:
                if trigger.startswith("message:"):
                    sender = trigger.split(":", 1)[1]
                    self._messenger.send(to=sender, content=post.send_content)
                    logger.info("Mode B auto-reply sent to %s", sender)
                elif self._model_config.supervisor:
                    self._messenger.send(
                        to=self._model_config.supervisor,
                        content=post.send_content,
                    )
                    logger.info(
                        "Mode B auto-report sent to %s",
                        self._model_config.supervisor,
                    )
        except Exception:
            logger.warning("Post-call processing failed", exc_info=True)

        return ExecutionResult(text=reply)
