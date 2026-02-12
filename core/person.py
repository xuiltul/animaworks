from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

from core.agent import AgentCore
from core.memory import MemoryManager
from core.messenger import Messenger
from core.paths import load_prompt
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
        logger.info(
            "[%s] process_message START from=%s content_len=%d",
            self.name, from_person, len(content),
        )
        async with self._lock:
            self._status = "thinking"
            self._current_task = f"Responding to {from_person}"

            prompt = load_prompt(
                "chat_message", from_person=from_person, content=content
            )

            try:
                result = await self.agent.run_cycle(
                    prompt, trigger=f"message:{from_person}"
                )
                self._last_activity = datetime.now()
                logger.info(
                    "[%s] process_message END duration_ms=%d",
                    self.name, result.duration_ms,
                )
                return result.summary
            except Exception:
                logger.exception("[%s] process_message FAILED", self.name)
                raise
            finally:
                self._status = "idle"
                self._current_task = ""

    async def process_message_stream(
        self, content: str, from_person: str = "human"
    ) -> AsyncGenerator[dict, None]:
        """Streaming version of process_message.

        Yields stream event dicts. The lock is held for the entire duration.
        """
        logger.info(
            "[%s] process_message_stream START from=%s content_len=%d",
            self.name, from_person, len(content),
        )
        async with self._lock:
            self._status = "thinking"
            self._current_task = f"Responding to {from_person}"

            prompt = load_prompt(
                "chat_message", from_person=from_person, content=content
            )

            try:
                async for chunk in self.agent.run_cycle_streaming(
                    prompt, trigger=f"message:{from_person}"
                ):
                    if chunk.get("type") == "cycle_done":
                        self._last_activity = datetime.now()
                        logger.info(
                            "[%s] process_message_stream END",
                            self.name,
                        )
                    yield chunk
            except Exception:
                logger.exception("[%s] process_message_stream FAILED", self.name)
                yield {"type": "error", "message": "Internal error"}
            finally:
                self._status = "idle"
                self._current_task = ""

    async def run_heartbeat(self) -> CycleResult:
        logger.info("[%s] run_heartbeat START", self.name)
        async with self._lock:
            self._status = "checking"
            self._last_heartbeat = datetime.now()

            hb_config = self.memory.read_heartbeat_config()
            checklist = hb_config or load_prompt("heartbeat_default_checklist")
            parts = [load_prompt("heartbeat", checklist=checklist)]

            unread_count = 0
            if self.messenger.has_unread():
                messages = self.messenger.receive_and_archive()
                unread_count = len(messages)
                logger.info(
                    "[%s] Processing %d unread messages in heartbeat",
                    self.name, unread_count,
                )
                summary = "\n".join(
                    f"- {m.from_person}: {m.content[:100]}" for m in messages
                )
                parts.append(load_prompt("unread_messages", summary=summary))

            try:
                result = await self.agent.run_cycle(
                    "\n\n".join(parts), trigger="heartbeat"
                )
                self._last_activity = datetime.now()
                logger.info(
                    "[%s] run_heartbeat END duration_ms=%d unread_processed=%d",
                    self.name, result.duration_ms, unread_count,
                )
                return result
            except Exception:
                logger.exception("[%s] run_heartbeat FAILED", self.name)
                raise
            finally:
                self._status = "idle"
                self._current_task = ""

    async def run_cron_task(
        self, task_name: str, description: str
    ) -> CycleResult:
        logger.info("[%s] run_cron_task START task=%s", self.name, task_name)
        async with self._lock:
            self._status = "working"
            self._current_task = task_name

            prompt = load_prompt(
                "cron_task", task_name=task_name, description=description
            )

            try:
                result = await self.agent.run_cycle(
                    prompt, trigger=f"cron:{task_name}"
                )
                self._last_activity = datetime.now()
                logger.info(
                    "[%s] run_cron_task END task=%s duration_ms=%d",
                    self.name, task_name, result.duration_ms,
                )
                return result
            except Exception:
                logger.exception(
                    "[%s] run_cron_task FAILED task=%s", self.name, task_name,
                )
                raise
            finally:
                self._status = "idle"
                self._current_task = ""
