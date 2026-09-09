"""
IPC communication layer using JSON Lines over a platform-specific transport.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.exceptions import IPCConnectionError
from core.supervisor.transport import (
    cleanup_ipc_endpoint,
    open_ipc_connection,
    resolve_client_endpoint,
    start_ipc_server,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
IPC_BUFFER_LIMIT = 16 * 1024 * 1024  # 16MB — default asyncio limit is 64KB
IPC_CHUNK_MAX = 1 * 1024 * 1024  # 1MB — max chunk for socket writes


# ── Protocol Types ──────────────────────────────────────────────────


@dataclass
class IPCRequest:
    """IPC request from parent to child process."""

    id: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON line."""
        return json.dumps({"id": self.id, "method": self.method, "params": self.params}, default=str)

    @classmethod
    def from_json(cls, line: str) -> IPCRequest:
        """Deserialize from JSON line."""
        data = json.loads(line)
        return cls(id=data["id"], method=data["method"], params=data.get("params", {}))


@dataclass
class IPCResponse:
    """IPC response from child to parent process."""

    id: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    stream: bool = False
    chunk: str | None = None
    done: bool = False

    def to_json(self) -> str:
        """Serialize to JSON line."""
        data: dict[str, Any] = {"id": self.id}

        if self.stream:
            data["stream"] = True
            if self.chunk is not None:
                data["chunk"] = self.chunk
            if self.done:
                data["done"] = True
                if self.result is not None:
                    data["result"] = self.result
        elif self.error is not None:
            data["error"] = self.error
        elif self.result is not None:
            data["result"] = self.result

        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, line: str) -> IPCResponse:
        """Deserialize from JSON line."""
        data = json.loads(line)
        return cls(
            id=data["id"],
            result=data.get("result"),
            error=data.get("error"),
            stream=data.get("stream", False),
            chunk=data.get("chunk"),
            done=data.get("done", False),
        )


@dataclass
class IPCEvent:
    """Asynchronous event from child to parent (no request ID)."""

    event: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON line."""
        return json.dumps({"event": self.event, "data": self.data}, default=str)

    @classmethod
    def from_json(cls, line: str) -> IPCEvent:
        """Deserialize from JSON line."""
        data = json.loads(line)
        return cls(event=data["event"], data=data.get("data", {}))


# ── IPC Server (Child Process) ────────────────────────────────────────

# Handler may return a single IPCResponse OR an AsyncIterator of IPCResponse
# for streaming.
RequestHandler = Callable[[IPCRequest], Awaitable[IPCResponse | AsyncIterator[IPCResponse]]]


class IPCServer:
    """
    IPC server for child process.

    Listens on a platform-specific endpoint resolved from ``socket_path`` and
    handles incoming requests by dispatching to registered handlers.

    Supports both single-response and streaming-response handlers:
    - If handler returns an IPCResponse, a single JSON line is sent.
    - If handler returns an AsyncIterator[IPCResponse], each item is sent
      as a separate JSON line (streaming protocol).
    """

    def __init__(self, socket_path: Path, request_handler: RequestHandler):
        self.socket_path = socket_path
        self.request_handler = request_handler
        self.server: asyncio.Server | None = None
        self.endpoint_description = str(socket_path)

    async def start(self) -> None:
        """Start the IPC server."""
        self.server, endpoint = await start_ipc_server(
            self.socket_path,
            self._handle_connection,
            limit=IPC_BUFFER_LIMIT,
        )
        self.endpoint_description = endpoint.describe()
        logger.info("IPC server started on %s", self.endpoint_description)

    @staticmethod
    async def _chunked_write(
        writer: asyncio.StreamWriter,
        data: bytes,
    ) -> None:
        """Write *data* to *writer*, splitting into IPC_CHUNK_MAX-sized pieces.

        For large JSON lines (e.g. RAG search results that can be several MB),
        writing the entire payload in one ``writer.write()`` call may overflow
        the OS send buffer.  This helper writes in manageable pieces with an
        ``await writer.drain()`` between each to let the kernel flush.
        """
        if len(data) <= IPC_CHUNK_MAX:
            writer.write(data)
            await writer.drain()
            return

        offset = 0
        while offset < len(data):
            end = min(offset + IPC_CHUNK_MAX, len(data))
            writer.write(data[offset:end])
            await writer.drain()
            offset = end

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a single client connection."""
        peer = writer.get_extra_info("peername", "unknown")
        logger.debug("IPC connection from %s", peer)

        try:
            while True:
                line_bytes = await reader.readline()
                if not line_bytes:
                    # Connection closed
                    break

                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue

                request: IPCRequest | None = None
                try:
                    request = IPCRequest.from_json(line)
                    logger.debug("IPC request: %s (id=%s)", request.method, request.id)

                    handler_result = await self.request_handler(request)

                    # Check if result is an async iterator (streaming)
                    if isinstance(handler_result, AsyncIterator):
                        try:
                            async for response in handler_result:
                                response_data = (response.to_json() + "\n").encode("utf-8")
                                await self._chunked_write(writer, response_data)
                        finally:
                            aclose = getattr(handler_result, "aclose", None)
                            if callable(aclose):
                                await aclose()
                    else:
                        response_data = (handler_result.to_json() + "\n").encode("utf-8")
                        await self._chunked_write(writer, response_data)

                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON in IPC request: %s", e)
                    error_response = IPCResponse(id="unknown", error={"code": "INVALID_JSON", "message": str(e)})
                    if not self._try_send_error(writer, error_response):
                        break
                    try:
                        await writer.drain()
                    except (ConnectionError, OSError):
                        break
                except Exception as e:
                    logger.exception("Error handling IPC request: %s", e)
                    error_response = IPCResponse(
                        id=request.id if request is not None else "unknown",
                        error={"code": "HANDLER_ERROR", "message": str(e)},
                    )
                    # The usual cause is the peer vanishing mid-write; pushing an
                    # error frame down the same broken pipe just re-raises.
                    if not self._try_send_error(writer, error_response):
                        break
                    try:
                        await writer.drain()
                    except (ConnectionError, OSError):
                        break

        except asyncio.CancelledError:
            logger.info("IPC connection cancelled")
        except Exception as e:
            logger.exception("IPC connection error: %s", e)
        finally:
            # Never let cleanup escape: an exception here surfaces as
            # "Unhandled exception in client_connected_cb" in asyncio.
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                logger.debug("IPC connection cleanup failed: %s", peer, exc_info=True)
            logger.debug("IPC connection closed: %s", peer)

    @staticmethod
    def _try_send_error(writer: asyncio.StreamWriter, response: IPCResponse) -> bool:
        try:
            writer.write((response.to_json() + "\n").encode("utf-8"))
        except (ConnectionError, OSError):
            return False
        return True

    async def stop(self) -> None:
        """Stop the server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("IPC server stopped")

        cleanup_ipc_endpoint(self.socket_path)


# ── IPC Client (Parent Process) ────────────────────────────────────────


class IPCClient:
    """
    IPC client for parent process.

    Connects to a child process endpoint resolved from ``socket_path`` and
    sends requests.
    """

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path

    @staticmethod
    def _resolve_ipc_timeout() -> float:
        """Resolve IPC stream timeout from config, with fallback default."""
        try:
            from core.config import load_config

            config = load_config()
            return float(config.server.ipc_stream_timeout)
        except Exception:
            logger.debug("Config load failed for ipc_stream_timeout", exc_info=True)
            return 60.0

    async def connect(self, timeout: float = 5.0) -> None:
        """Test connectivity to the active IPC endpoint.

        Opens a temporary connection to verify the endpoint is reachable, then
        closes it immediately. Used by ``_wait_for_ready()`` during process
        startup to confirm the IPC server is listening.

        Actual request traffic uses per-request dedicated connections
        opened inside :meth:`send_request` and :meth:`send_request_stream`.
        """
        async with asyncio.timeout(timeout):
            _reader, writer = await open_ipc_connection(self.socket_path, limit=IPC_BUFFER_LIMIT)
            writer.close()
            await writer.wait_closed()
            endpoint = resolve_client_endpoint(self.socket_path)
            logger.debug("IPC client connectivity verified: %s", endpoint.describe())

    async def send_request(self, request: IPCRequest, timeout: float = 60.0) -> IPCResponse:
        """Send a request over a dedicated connection and wait for response.

        Opens a fresh dedicated connection for each request so that
        concurrent traffic cannot contaminate the response with stale
        buffered data (the same pattern used by :meth:`send_request_stream`).

        Args:
            request: The request to send
            timeout: Timeout in seconds

        Returns:
            The response

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            RuntimeError: If socket not available or response ID mismatch
        """
        request_line = request.to_json() + "\n"

        async with asyncio.timeout(timeout):
            try:
                reader, writer = await open_ipc_connection(self.socket_path, limit=IPC_BUFFER_LIMIT)
            except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
                raise IPCConnectionError(f"Not connected: {e}") from e

            try:
                writer.write(request_line.encode("utf-8"))
                await writer.drain()
                logger.debug("IPC request sent: %s (id=%s)", request.method, request.id)

                response_line_bytes = await reader.readline()
                if not response_line_bytes:
                    raise IPCConnectionError("Connection closed")

                response_line = response_line_bytes.decode("utf-8").strip()
                response = IPCResponse.from_json(response_line)
                logger.debug("IPC response received: id=%s", response.id)

                # Validate response ID matches request ID
                if response.id != request.id:
                    logger.warning(
                        "[IPC] ID_MISMATCH method=%s expected=%s got=%s",
                        request.method,
                        request.id,
                        response.id,
                    )
                    raise IPCConnectionError(
                        f"IPC protocol error: response ID mismatch (expected={request.id}, got={response.id})"
                    )

                return response
            finally:
                writer.close()
                await writer.wait_closed()

    async def send_request_stream(
        self,
        request: IPCRequest,
        timeout: float | None = None,
    ) -> AsyncIterator[IPCResponse]:
        """
        Send a request and yield streaming responses over a dedicated connection.

        Opens a fresh dedicated connection for each streaming request (the
        same per-request pattern used by :meth:`send_request`).

        The first readline uses a generous timeout (``max(timeout, 120s)``)
        because the initial response may carry large payloads such as RAG
        search results.  Subsequent chunks use the normal *timeout*.

        Args:
            request: The request to send
            timeout: Per-chunk timeout in seconds. Resets on each received chunk.
                If None, reads from ``config.json server.ipc_stream_timeout``
                (default 60s).

        Yields:
            IPCResponse objects (chunks and final result)

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            RuntimeError: If socket not available or response ID mismatch
        """
        if timeout is None:
            timeout = self._resolve_ipc_timeout()

        import time as _time

        _ipc_start = _time.monotonic()
        chunk_count = 0

        # The first chunk may include large RAG data, so allow extra time.
        first_chunk_timeout = max(timeout, 120.0)

        # Open a dedicated connection for this streaming request.
        logger.info(
            "[IPC-STREAM] opening dedicated connection endpoint=%s method=%s id=%s",
            resolve_client_endpoint(self.socket_path).describe(),
            request.method,
            request.id,
        )
        try:
            stream_reader, stream_writer = await open_ipc_connection(self.socket_path, limit=IPC_BUFFER_LIMIT)
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            raise IPCConnectionError(f"Not connected: {e}") from e

        try:
            # Send request
            request_line = request.to_json() + "\n"
            stream_writer.write(request_line.encode("utf-8"))
            await stream_writer.drain()
            logger.info(
                "[IPC-STREAM] request sent method=%s id=%s timeout=%.1fs first_chunk_timeout=%.1fs endpoint=%s",
                request.method,
                request.id,
                timeout,
                first_chunk_timeout,
                resolve_client_endpoint(self.socket_path).describe(),
            )

            # Read streaming responses until done
            while True:
                # Use generous timeout for the first chunk (large RAG payloads),
                # then normal per-chunk timeout for subsequent chunks.
                effective_timeout = first_chunk_timeout if chunk_count == 0 else timeout
                try:
                    response_line_bytes = await asyncio.wait_for(
                        stream_reader.readline(),
                        timeout=effective_timeout,
                    )
                except TimeoutError:
                    elapsed = _time.monotonic() - _ipc_start
                    logger.info(
                        "[IPC-STREAM] TIMEOUT method=%s id=%s chunks=%d elapsed=%.1fs effective_timeout=%.1fs",
                        request.method,
                        request.id,
                        chunk_count,
                        elapsed,
                        effective_timeout,
                    )
                    raise
                if not response_line_bytes:
                    elapsed = _time.monotonic() - _ipc_start
                    logger.info(
                        "[IPC-STREAM] CONNECTION_CLOSED method=%s id=%s chunks=%d elapsed=%.1fs",
                        request.method,
                        request.id,
                        chunk_count,
                        elapsed,
                    )
                    raise IPCConnectionError("Connection closed during stream")

                response_line = response_line_bytes.decode("utf-8").strip()
                if not response_line:
                    continue

                response = IPCResponse.from_json(response_line)
                chunk_count += 1

                # Validate response ID matches request ID
                if response.id != request.id:
                    elapsed = _time.monotonic() - _ipc_start
                    logger.warning(
                        "[IPC-STREAM] ID_MISMATCH method=%s expected=%s got=%s chunks=%d elapsed=%.1fs",
                        request.method,
                        request.id,
                        response.id,
                        chunk_count,
                        elapsed,
                    )
                    raise IPCConnectionError(
                        f"IPC protocol error: response ID mismatch (expected={request.id}, got={response.id})"
                    )

                # Non-streaming response (error or unexpected)
                if not response.stream:
                    elapsed = _time.monotonic() - _ipc_start
                    logger.info(
                        "[IPC-STREAM] non-stream response method=%s id=%s chunks=%d elapsed=%.1fs error=%s",
                        request.method,
                        request.id,
                        chunk_count,
                        elapsed,
                        response.error,
                    )
                    yield response
                    return

                yield response

                # Final chunk with done=True ends the stream
                if response.done:
                    elapsed = _time.monotonic() - _ipc_start
                    logger.info(
                        "[IPC-STREAM] done method=%s id=%s chunks=%d elapsed=%.1fs",
                        request.method,
                        request.id,
                        chunk_count,
                        elapsed,
                    )
                    return
        finally:
            elapsed = _time.monotonic() - _ipc_start
            stream_writer.close()
            await stream_writer.wait_closed()
            logger.info(
                "[IPC-STREAM] dedicated connection closed method=%s id=%s chunks=%d elapsed=%.1fs",
                request.method,
                request.id,
                chunk_count,
                elapsed,
            )

    async def close(self) -> None:
        """No-op — kept for backward compatibility.

        With per-request dedicated connections there is no persistent
        connection to close.  Each :meth:`send_request` and
        :meth:`send_request_stream` call manages its own connection
        lifecycle.
        """
