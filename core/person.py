from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from core.agent import AgentCore
from core.memory import MemoryManager
from core.messenger import Messenger
from core.schemas import CycleResult, PersonStatus

logger = logging.getLogger("animaworks.person")


class DigitalPerson:
    """A Digital Person: encapsulates identity, memory, agent, and communication.

    1 person = 1 directory.
    """

    def __init__(self, person_dir: Path, shared_dir: Path) -> None:
        self.person_dir = person_dir
        self.name = person_dir.name

        self.memory = MemoryManager(person_dir)
        self.model_config = self.memory.read_model_config()
        self.messenger = Messenger(shared_dir, self.name)
        self.agent = AgentCore(
            person_dir, self.memory, self.model_config, self.messenger
        )

        self._lock = asyncio.Lock()
        self._status = "idle"
        self._current_task = ""
        self._last_heartbeat: datetime | None = None
        self._last_activity: datetime | None = None

        logger.info("DigitalPerson '%s' initialized from %s", self.name, person_dir)

    @property
    def status(self) -> PersonStatus:
        return PersonStatus(
            name=self.name,
            status=self._status,
            current_task=self._current_task,
            last_heartbeat=self._last_heartbeat,
            last_activity=self._last_activity,
            pending_messages=self.messenger.unread_count(),
        )

    async def process_message(
        self, content: str, from_person: str = "human"
    ) -> str:
        async with self._lock:
            self._status = "thinking"
            self._current_task = f"Responding to {from_person}"

            prompt = (
                f"あなたに{from_person}からメッセージが届きました:\n\n"
                f"{content}"
            )

            try:
                result = await self.agent.run_cycle(
                    prompt, trigger=f"message:{from_person}"
                )
                self._last_activity = datetime.now()
                return result.summary
            finally:
                self._status = "idle"
                self._current_task = ""

    async def run_heartbeat(self) -> CycleResult:
        async with self._lock:
            self._status = "checking"
            self._last_heartbeat = datetime.now()

            hb_config = self.memory.read_heartbeat_config()
            parts = [
                "ハートビートです。以下を確認してください:",
                hb_config
                or (
                    "- Inboxに未読メッセージがあるか\n"
                    "- 進行中タスクにブロッカーがないか\n"
                    "- 何もなければ何もしない（HEARTBEAT_OK）"
                ),
            ]

            if self.messenger.has_unread():
                messages = self.messenger.receive_and_archive()
                summary = "\n".join(
                    f"- {m.from_person}: {m.content[:100]}" for m in messages
                )
                parts.append(f"\n## 未読メッセージ\n{summary}")

            try:
                result = await self.agent.run_cycle(
                    "\n\n".join(parts), trigger="heartbeat"
                )
                self._last_activity = datetime.now()
                return result
            finally:
                self._status = "idle"
                self._current_task = ""

    async def run_cron_task(
        self, task_name: str, description: str
    ) -> CycleResult:
        async with self._lock:
            self._status = "working"
            self._current_task = task_name

            prompt = (
                f"定時タスク「{task_name}」を実行してください。\n\n"
                f"{description}\n\n"
                "必ず結果を出力してください。"
            )

            try:
                result = await self.agent.run_cycle(
                    prompt, trigger=f"cron:{task_name}"
                )
                self._last_activity = datetime.now()
                return result
            finally:
                self._status = "idle"
                self._current_task = ""
