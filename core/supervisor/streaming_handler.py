# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Streaming IPC message handler.

Handles streaming process_message requests with queue-based merge
of Agent SDK stream chunks and periodic keep-alive chunks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from core.supervisor.ipc import IPCRequest, IPCResponse

if TYPE_CHECKING:
    from core.anima import DigitalAnima

logger = logging.getLogger(__name__)

_DEFAULT_KEEPALIVE_INTERVAL = 30  # fallback keep-alive interval in seconds


class _Sentinel:
    """Queue termination marker for keep-alive merge."""

    __slots__ = ()


_SENTINEL = _Sentinel()


class StreamingIPCHandler:
    """Streaming message processing with keep-alive merge."""

    def __init__(
        self,
        anima: DigitalAnima,
        anima_name: str,
        anima_dir: Any,
    ) -> None:
        self._anima = anima
        self._anima_name = anima_name
        self._anima_dir = anima_dir

    async def handle_stream(
        self, request: IPCRequest
    ) -> AsyncIterator[IPCResponse]:
        """Handle streaming process_message request.

        Yields IPCResponse chunks with stream=True, followed by
        a final response with done=True containing the full result.

        Uses an asyncio.Queue to merge Agent SDK stream chunks and
        periodic keep-alive chunks so that the IPC layer's per-chunk
        timeout is reset even during long tool executions.
        """
        if not self._anima:
            yield IPCResponse(
                id=request.id,
                error={
                    "code": "NOT_INITIALIZED",
                    "message": "Anima not initialized"
                }
            )
            return

        # ── Resolve keep-alive interval from config ──────────────
        try:
            from core.config import load_config
            keepalive_interval: int = load_config().server.keepalive_interval
        except Exception:
            keepalive_interval = _DEFAULT_KEEPALIVE_INTERVAL

        message = request.params.get("message", "")
        from_person = request.params.get("from_person", "human")
        intent = request.params.get("intent") or ""
        images = request.params.get("images") or None
        attachment_paths = request.params.get("attachment_paths") or None
        full_response = ""

        # Track bootstrap state to detect completion
        was_bootstrapping = self._anima.needs_bootstrap

        # ── Queue-based merge of SDK stream + keep-alive ─────────
        queue: asyncio.Queue[IPCResponse | _Sentinel] = asyncio.Queue()
        last_chunk_time = time.monotonic()
        stream_start_time = time.monotonic()

        async def _enqueue(resp: IPCResponse) -> None:
            """Put response on queue and update last-chunk timestamp."""
            nonlocal last_chunk_time
            last_chunk_time = time.monotonic()
            await queue.put(resp)

        async def _stream_producer() -> None:
            """Read Agent SDK stream and enqueue IPCResponse chunks."""
            nonlocal full_response
            try:
                # Emit bootstrap_start if anima needs bootstrap
                if was_bootstrapping:
                    await _enqueue(IPCResponse(
                        id=request.id,
                        stream=True,
                        chunk=json.dumps(
                            {"type": "bootstrap_start"}, ensure_ascii=False,
                        ),
                    ))

                async for chunk in self._anima.process_message_stream(
                    message, from_person=from_person,
                    intent=intent,
                    images=images, attachment_paths=attachment_paths,
                ):
                    event_type = chunk.get("type", "unknown")

                    if event_type == "text_delta":
                        text = chunk.get("text", "")
                        full_response += text
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(chunk, ensure_ascii=False),
                        ))

                    elif event_type == "cycle_done":
                        cycle_result = chunk.get("cycle_result", {})
                        full_response = cycle_result.get(
                            "summary", full_response,
                        )

                        # Emit bootstrap_complete if bootstrap just finished
                        if (
                            was_bootstrapping
                            and not self._anima.needs_bootstrap
                        ):
                            await _enqueue(IPCResponse(
                                id=request.id,
                                stream=True,
                                chunk=json.dumps(
                                    {"type": "bootstrap_complete"},
                                    ensure_ascii=False,
                                ),
                            ))

                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            done=True,
                            result={
                                "response": full_response,
                                "replied_to": [],
                                "cycle_result": cycle_result,
                            },
                        ))
                        return

                    elif event_type == "bootstrap_busy":
                        # Anima is already bootstrapping — forward as-is
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(chunk, ensure_ascii=False),
                        ))

                    elif event_type == "error":
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(chunk, ensure_ascii=False),
                        ))

                    else:
                        # Forward other event types (tool_start, tool_end,
                        # chain_start, etc.) as stream chunks
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(chunk, ensure_ascii=False),
                        ))

                # Stream ended without cycle_done — send final done
                await _enqueue(IPCResponse(
                    id=request.id,
                    stream=True,
                    done=True,
                    result={
                        "response": full_response,
                        "replied_to": [],
                    },
                ))

            except TimeoutError as e:
                logger.error("Timeout in streaming process_message: %s", e)
                await queue.put(IPCResponse(
                    id=request.id,
                    error={
                        "code": "IPC_TIMEOUT",
                        "message": str(e) or "Stream processing timed out",
                    },
                ))
            except Exception as e:
                logger.exception(
                    "Error in streaming process_message: %s", e,
                )
                await queue.put(IPCResponse(
                    id=request.id,
                    error={
                        "code": "STREAM_ERROR",
                        "message": str(e),
                    },
                ))
            finally:
                await queue.put(_SENTINEL)

        async def _keepalive_producer() -> None:
            """Emit keep-alive chunks when Agent SDK stream is silent.

            Stops automatically when *producer_task* finishes (crash or
            normal exit), so that stale keep-alives do not mask a dead
            Agent SDK subprocess.
            """
            try:
                while True:
                    await asyncio.sleep(keepalive_interval)
                    # Stop if producer finished (SENTINEL already queued)
                    if producer_task.done():
                        logger.debug(
                            "Keepalive stopping: producer finished for %s",
                            self._anima_name,
                        )
                        return
                    elapsed_since_chunk = time.monotonic() - last_chunk_time
                    if elapsed_since_chunk >= keepalive_interval:
                        elapsed = round(
                            time.monotonic() - stream_start_time, 1,
                        )
                        logger.debug(
                            "Keep-alive sent for %s (elapsed=%.1fs)",
                            self._anima_name, elapsed,
                        )
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(
                                {"type": "keepalive", "elapsed_s": elapsed},
                                ensure_ascii=False,
                            ),
                        ))
            except asyncio.CancelledError:
                return

        # Launch producer tasks
        logger.debug(
            "Starting queue-based stream merge for %s (keepalive=%ds)",
            self._anima_name, keepalive_interval,
        )
        producer_task = asyncio.create_task(
            _stream_producer(),
            name=f"stream-producer-{self._anima_name}",
        )
        keepalive_task = asyncio.create_task(
            _keepalive_producer(),
            name=f"keepalive-{self._anima_name}",
        )

        try:
            # ── Main loop: drain queue and yield ─────────────────
            while True:
                item = await queue.get()
                if isinstance(item, _Sentinel):
                    break
                yield item
                # If this was a terminal response, stop immediately
                if item.done or item.error:
                    break
        finally:
            keepalive_task.cancel()
            # Ensure producer finishes; suppress CancelledError
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
            logger.debug(
                "Stream merge completed for %s (%.1fs)",
                self._anima_name,
                time.monotonic() - stream_start_time,
            )
