// ── Shared Chat Stream ──────────────────────────────────
// Common SSE chat streaming logic used across all chat modules.
// Extracts the duplicated fetch + ReadableStream + SSE parse loop
// into a single reusable function with callback-based UI injection.

import { parseConvSSE, getErrorMessage } from "./sse-parser.js";
import { createLogger } from "./logger.js";
import { api } from "../modules/api.js";
import { basePath } from "/shared/base-path.js";

const logger = createLogger("chat-stream");

/**
 * Idle watchdog for SSE reads (ms).
 *
 * The server tail emits a ": keepalive" comment every 30s while a stream is
 * alive, so 90s of complete silence (3 missed keepalives) means the connection
 * is dead or the producer hung without ever sending `done`. Without this
 * watchdog `reader.read()` can block forever, the promise never settles, and
 * the caller's `isStreaming` state is never cleared (history polling stalls).
 */
const SSE_IDLE_TIMEOUT_MS = 90_000;

/**
 * Wrap `reader.read()` with an idle timeout.
 *
 * Any bytes (including SSE keepalive comments) reset the timer because the
 * timer is re-armed on every read call.
 *
 * @param {ReadableStreamDefaultReader} reader
 * @param {number} timeoutMs - <= 0 disables the watchdog
 * @returns {Promise<{done: boolean, value?: Uint8Array}>}
 */
function _readWithIdleTimeout(reader, timeoutMs) {
  if (!(timeoutMs > 0)) return reader.read();
  let timer = null;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const err = new Error(`SSE idle timeout: no data received for ${timeoutMs}ms`);
      err.name = "SseIdleTimeoutError";
      reject(err);
    }, timeoutMs);
  });
  return Promise.race([reader.read(), timeout]).finally(() => clearTimeout(timer));
}

/**
 * Best-effort cancel of a reader (tears down the underlying fetch body).
 * Never throws; safe to call on partial/mock readers without `cancel()`.
 */
function _cancelReaderQuietly(reader, reason) {
  try {
    if (typeof reader.cancel === "function") {
      Promise.resolve(reader.cancel(reason)).catch(() => {});
    }
  } catch {
    /* ignore */
  }
}

/**
 * Fetch the active stream for an anima.
 * @param {string} animaName
 * @param {string} [threadId] - Optional thread ID to filter by
 * @returns {Promise<object|null>} Active stream info or null
 */
export async function fetchActiveStream(animaName, threadId) {
  try {
    let path = `/api/animas/${encodeURIComponent(animaName)}/stream/active`;
    if (threadId) path += `?thread_id=${encodeURIComponent(threadId)}`;
    const data = await api(path);
    return data.active ? data : null;
  } catch {
    return null;
  }
}

/**
 * Fetch progress of a specific stream.
 * @param {string} animaName
 * @param {string} responseId
 * @returns {Promise<object|null>}
 */
export async function fetchStreamProgress(animaName, responseId) {
  try {
    return await api(
      `/api/animas/${encodeURIComponent(animaName)}/stream/${encodeURIComponent(responseId)}/progress`
    );
  } catch {
    return null;
  }
}

/**
 * SSE chat stream processor.
 *
 * Handles the full lifecycle: fetch -> ReadableStream -> SSE parse -> callbacks.
 * UI-specific logic (bubble updates, Live2D expressions, state management)
 * is injected via the callbacks parameter.
 *
 * @param {string} animaName - Target Anima name
 * @param {string|FormData} body - Request body (JSON string or FormData)
 * @param {AbortSignal|null} signal - Optional AbortSignal for cancellation
 * @param {object} callbacks - Event callbacks
 * @param {function(string): void} [callbacks.onTextDelta] - Text delta received
 * @param {function(string): void} [callbacks.onToolStart] - Tool execution started (tool name)
 * @param {function(): void} [callbacks.onToolEnd] - Tool execution ended
 * @param {function({summary: string, emotion: string, images: Array}): void} [callbacks.onDone] - Stream completed
 * @param {function({message: string}): void} [callbacks.onError] - SSE error event received
 * @param {function(object): void} [callbacks.onBootstrap] - Bootstrap event (data object)
 * @param {function(): void} [callbacks.onChainStart] - Chain continuation event
 * @param {function({message: string}): void} [callbacks.onHeartbeatRelayStart] - Heartbeat relay started
 * @param {function({text: string}): void} [callbacks.onHeartbeatRelay] - Heartbeat relay text chunk
 * @param {function(): void} [callbacks.onHeartbeatRelayDone] - Heartbeat relay completed
 * @param {function(): void} [callbacks.onThinkingStart] - Thinking block started
 * @param {function(string): void} [callbacks.onThinkingDelta] - Thinking text delta
 * @param {function(): void} [callbacks.onThinkingEnd] - Thinking block ended
 * @param {function(): void} [callbacks.onReconnecting] - Reconnection attempt starting
 * @param {function(): void} [callbacks.onReconnected] - Reconnection successful
 * @param {object} [options]
 * @param {number} [options.idleTimeoutMs=SSE_IDLE_TIMEOUT_MS] - Reject when no bytes arrive for this long
 * @returns {Promise<void>}
 * @throws {Error} On HTTP error (non-ok response), network failure, or idle timeout
 *   (`err.name === "SseIdleTimeoutError"`) after reconnection attempts are exhausted
 */
export async function streamChat(animaName, body, signal, callbacks, options = {}) {
  const url = `${basePath}/api/animas/${encodeURIComponent(animaName)}/chat/stream`;
  const start = performance.now();
  logger.info(`[SSE-FE] streamChat START anima=${animaName} url=${url}`);

  // Track response ID and last event ID for reconnection
  let responseId = null;
  let lastEventId = null;

  const headers = body instanceof FormData ? {} : { "Content-Type": "application/json" };

  const res = await fetch(url, {
    method: "POST",
    headers,
    body,
    ...(signal ? { signal } : {}),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    logger.error(`[SSE-FE] fetch FAILED status=${res.status} body=${text.slice(0, 200)}`);
    throw new Error(`API ${res.status}: ${text}`);
  }

  try {
    await _processStream(res, callbacks, (id) => { responseId = id; }, (id) => { lastEventId = id; }, signal, options);
  } catch (err) {
    const elapsed = ((performance.now() - start) / 1000).toFixed(1);
    logger.info(`[SSE-FE] _processStream ERROR anima=${animaName} err=${err.name}:${err.message} elapsed=${elapsed}s responseId=${responseId} lastEventId=${lastEventId}`);
    if (err.name === "AbortError") throw err;

    // Attempt reconnection with exponential backoff
    if (responseId) {
      logger.info(`[SSE-FE] reconnect attempt anima=${animaName} responseId=${responseId} lastEventId=${lastEventId}`);
      const reconnected = await _reconnectWithBackoff(
        animaName, responseId, lastEventId, body, signal, callbacks, options,
      );
      if (reconnected) {
        logger.info(`[SSE-FE] reconnect SUCCESS anima=${animaName}`);
        return;
      }
      logger.info(`[SSE-FE] reconnect FAILED anima=${animaName}`);
    }

    throw err;
  }

  const elapsed = ((performance.now() - start) / 1000).toFixed(1);
  logger.info(`[SSE-FE] streamChat COMPLETE anima=${animaName} elapsed=${elapsed}s responseId=${responseId}`);
}

/**
 * Process a ReadableStream response, parsing SSE events.
 *
 * @param {object} [options]
 * @param {number} [options.idleTimeoutMs=SSE_IDLE_TIMEOUT_MS] - Idle watchdog; <= 0 disables
 * @throws {Error} `SseIdleTimeoutError` when no bytes arrive within the idle timeout
 */
async function _processStream(res, callbacks, setResponseId, setLastEventId, signal, options = {}) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const idleTimeoutMs = options.idleTimeoutMs ?? SSE_IDLE_TIMEOUT_MS;
  let buffer = "";
  let chunkCount = 0;
  let sseEventCount = 0;
  const streamStart = performance.now();

  logger.debug(`[SSE-FE] _processStream: reader opened idleTimeoutMs=${idleTimeoutMs}`);

  try {
    while (true) {
      if (signal?.aborted) throw new DOMException("Aborted", "AbortError");

      const { done, value } = await _readWithIdleTimeout(reader, idleTimeoutMs);

      if (done) {
        const totalElapsed = ((performance.now() - streamStart) / 1000).toFixed(1);
        logger.info(`[SSE-FE] reader DONE chunks=${chunkCount} sseEvents=${sseEventCount} elapsed=${totalElapsed}s bufferRemaining=${buffer.length}`);
        break;
      }
      chunkCount++;

      buffer += decoder.decode(value, { stream: true });
      const { parsed, remaining } = parseConvSSE(buffer);
      buffer = remaining;

      if (chunkCount % 200 === 0) {
        logger.debug(`[SSE-FE] chunk#${chunkCount} parsedEvents=${parsed.length} bufRemain=${remaining.length}`);
      }

      for (const { id, event, data } of parsed) {
        sseEventCount++;
        if (id) setLastEventId(id);

        switch (event) {
          case "stream_start":
            if (data.response_id) setResponseId(data.response_id);
            callbacks.onStreamStart?.(data);
            logger.info(`[SSE-FE] EVENT stream_start response_id=${data.response_id} id=${id}`);
            break;

          case "text_delta":
            callbacks.onTextDelta?.(data.text || "");
            break;

          case "tool_start":
            logger.debug(`[SSE-FE] EVENT tool_start tool=${data.tool_name} id=${id}`);
            callbacks.onToolStart?.(data.tool_name, { tool_id: data.tool_id });
            break;

          case "tool_detail":
            callbacks.onToolDetail?.(data.tool_name, data.detail, { tool_id: data.tool_id });
            break;

          case "tool_end":
            logger.debug(`[SSE-FE] EVENT tool_end tool=${data.tool_name || "?"} id=${id}`);
            callbacks.onToolEnd?.({
              tool_id: data.tool_id,
              tool_name: data.tool_name || "",
              result_summary: data.result_summary || "",
              input_summary: data.input_summary || "",
              is_error: !!data.is_error,
            });
            break;

          case "context_update":
            callbacks.onContextUpdate?.({
              contextUsageRatio: data.context_usage_ratio || 0,
              inputTokens: data.input_tokens || 0,
              contextWindow: data.context_window || 0,
              threshold: data.threshold || 0,
            });
            break;

          case "done": {
            const summaryLen = (data.summary || "").length;
            const totalElapsed = ((performance.now() - streamStart) / 1000).toFixed(1);
            logger.info(`[SSE-FE] EVENT done summary_len=${summaryLen} emotion=${data.emotion || "neutral"} totalEvents=${sseEventCount} elapsed=${totalElapsed}s id=${id}`);
            callbacks.onDone?.({
              summary: data.summary || null,
              emotion: data.emotion || "neutral",
              images: data.images || data.artifacts || [],
              files: data.files || [],
              thinkingSummary: data.thinking_summary || null,
              contextUsageRatio: data.context_usage_ratio || 0,
              inputTokens: data.input_tokens || 0,
              contextWindow: data.context_window || 0,
              contextThreshold: data.context_threshold || 0,
            });
            break;
          }

          case "error":
            logger.info(`[SSE-FE] EVENT error code=${data.code || "?"} msg=${getErrorMessage(data)} id=${id}`);
            callbacks.onError?.({ message: getErrorMessage(data) });
            break;

          case "bootstrap":
            logger.info(`[SSE-FE] EVENT bootstrap status=${data.status} id=${id}`);
            callbacks.onBootstrap?.(data);
            break;

          case "chain_start":
            logger.debug(`[SSE-FE] EVENT chain_start id=${id}`);
            callbacks.onChainStart?.();
            break;

          case "compression_start":
            logger.debug(`[SSE-FE] EVENT compression_start id=${id}`);
            callbacks.onCompressionStart?.();
            break;

          case "compression_end":
            logger.debug(`[SSE-FE] EVENT compression_end id=${id}`);
            callbacks.onCompressionEnd?.();
            break;

          case "heartbeat_relay_start":
            logger.debug(`[SSE-FE] EVENT heartbeat_relay_start msg=${data.message || ""} id=${id}`);
            callbacks.onHeartbeatRelayStart?.({ message: data.message || "" });
            break;

          case "heartbeat_relay":
            callbacks.onHeartbeatRelay?.({ text: data.text || "" });
            break;

          case "heartbeat_relay_done":
            logger.debug(`[SSE-FE] EVENT heartbeat_relay_done id=${id}`);
            callbacks.onHeartbeatRelayDone?.();
            break;

          case "thinking_start":
            logger.debug(`[SSE-FE] EVENT thinking_start id=${id}`);
            callbacks.onThinkingStart?.();
            break;

          case "thinking_delta":
            callbacks.onThinkingDelta?.(data.text || "");
            break;

          case "thinking_end":
            logger.debug(`[SSE-FE] EVENT thinking_end id=${id}`);
            callbacks.onThinkingEnd?.();
            break;

          default:
            logger.debug(`[SSE-FE] EVENT unknown event=${event} id=${id}`);
            break;
        }
      }
    }
  } catch (err) {
    if (err?.name === "SseIdleTimeoutError") {
      const totalElapsed = ((performance.now() - streamStart) / 1000).toFixed(1);
      logger.warn(`[SSE-FE] IDLE TIMEOUT chunks=${chunkCount} sseEvents=${sseEventCount} elapsed=${totalElapsed}s idleTimeoutMs=${idleTimeoutMs}`);
      // Tear down the underlying connection; the pending read() is settled by cancel().
      _cancelReaderQuietly(reader, err.message);
    }
    throw err;
  } finally {
    const totalElapsed = ((performance.now() - streamStart) / 1000).toFixed(1);
    logger.info(`[SSE-FE] reader.releaseLock chunks=${chunkCount} sseEvents=${sseEventCount} elapsed=${totalElapsed}s`);
    reader.releaseLock();
  }
}

/**
 * Reconnect with exponential backoff (1s -> 2s -> 4s -> ... max 30s, 5 attempts).
 */
async function _reconnectWithBackoff(animaName, responseId, lastEventId, originalBody, signal, callbacks, options = {}) {
  const MAX_RETRIES = 5;
  const MAX_DELAY = 30000;
  let delay = 1000;
  const reconnectStart = performance.now();

  logger.info(`[SSE-FE] reconnect START anima=${animaName} responseId=${responseId} lastEventId=${lastEventId}`);

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    if (signal?.aborted) {
      logger.info(`[SSE-FE] reconnect ABORTED attempt=${attempt}`);
      return false;
    }

    logger.info(`[SSE-FE] reconnect attempt=${attempt}/${MAX_RETRIES} delay=${delay}ms`);
    callbacks.onReconnecting?.();

    await new Promise((r) => setTimeout(r, delay));

    try {
      // Parse from_person from original body
      let fromPerson = "human";
      if (typeof originalBody === "string") {
        try { fromPerson = JSON.parse(originalBody).from_person || "human"; } catch { /* ignore */ }
      }

      const resumeBody = JSON.stringify({
        message: "",
        from_person: fromPerson,
        resume: responseId,
        last_event_id: lastEventId || "",
      });

      const url = `${basePath}/api/animas/${encodeURIComponent(animaName)}/chat/stream`;
      logger.info(`[SSE-FE] reconnect fetch POST ${url} lastEventId=${lastEventId}`);
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: resumeBody,
        ...(signal ? { signal } : {}),
      });

      logger.info(`[SSE-FE] reconnect fetch response status=${res.status} ok=${res.ok}`);

      if (!res.ok) {
        logger.warn(`[SSE-FE] reconnect FAILED status=${res.status} attempt=${attempt}`);
        delay = Math.min(delay * 2, MAX_DELAY);
        continue;
      }

      callbacks.onReconnected?.();
      logger.info(`[SSE-FE] reconnect SUCCESS attempt=${attempt} processing resumed stream`);
      await _processStream(res, callbacks, () => {}, (id) => { lastEventId = id; }, signal, options);
      const elapsed = ((performance.now() - reconnectStart) / 1000).toFixed(1);
      logger.info(`[SSE-FE] reconnect COMPLETE anima=${animaName} elapsed=${elapsed}s`);
      return true;
    } catch (err) {
      if (err.name === "AbortError") {
        logger.info(`[SSE-FE] reconnect ABORTED during attempt=${attempt}`);
        return false;
      }
      logger.warn(`[SSE-FE] reconnect attempt=${attempt} ERROR: ${err.name}:${err.message}`);
      delay = Math.min(delay * 2, MAX_DELAY);
    }
  }

  const elapsed = ((performance.now() - reconnectStart) / 1000).toFixed(1);
  logger.error(`[SSE-FE] reconnect ALL_FAILED anima=${animaName} elapsed=${elapsed}s`);
  return false;
}
