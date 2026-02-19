// ── Shared Chat Stream ──────────────────────────────────
// Common SSE chat streaming logic used across all chat modules.
// Extracts the duplicated fetch + ReadableStream + SSE parse loop
// into a single reusable function with callback-based UI injection.

import { parseConvSSE, getErrorMessage } from "./sse-parser.js";
import { createLogger } from "./logger.js";

const logger = createLogger("chat-stream");

/**
 * Fetch the active stream for an anima.
 * @param {string} animaName
 * @returns {Promise<object|null>} Active stream info or null
 */
export async function fetchActiveStream(animaName) {
  try {
    const res = await fetch(`/api/animas/${encodeURIComponent(animaName)}/stream/active`);
    if (!res.ok) return null;
    const data = await res.json();
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
    const res = await fetch(
      `/api/animas/${encodeURIComponent(animaName)}/stream/${encodeURIComponent(responseId)}/progress`
    );
    if (!res.ok) return null;
    return await res.json();
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
 * @param {function({summary: string, emotion: string}): void} [callbacks.onDone] - Stream completed
 * @param {function({message: string}): void} [callbacks.onError] - SSE error event received
 * @param {function(object): void} [callbacks.onBootstrap] - Bootstrap event (data object)
 * @param {function(): void} [callbacks.onChainStart] - Chain continuation event
 * @param {function({message: string}): void} [callbacks.onHeartbeatRelayStart] - Heartbeat relay started
 * @param {function({text: string}): void} [callbacks.onHeartbeatRelay] - Heartbeat relay text chunk
 * @param {function(): void} [callbacks.onHeartbeatRelayDone] - Heartbeat relay completed
 * @param {function(): void} [callbacks.onReconnecting] - Reconnection attempt starting
 * @param {function(): void} [callbacks.onReconnected] - Reconnection successful
 * @returns {Promise<void>}
 * @throws {Error} On HTTP error (non-ok response) or network failure
 */
export async function streamChat(animaName, body, signal, callbacks) {
  const url = `/api/animas/${encodeURIComponent(animaName)}/chat/stream`;
  const start = performance.now();
  logger.info(`[SSE-FE] streamChat START anima=${animaName} url=${url}`);

  // Track response ID and last event ID for reconnection
  let responseId = null;
  let lastEventId = null;

  const headers = body instanceof FormData ? {} : { "Content-Type": "application/json" };

  logger.info(`[SSE-FE] fetch POST ${url}`);
  const res = await fetch(url, {
    method: "POST",
    headers,
    body,
    ...(signal ? { signal } : {}),
  });

  logger.info(`[SSE-FE] fetch response status=${res.status} ok=${res.ok} type=${res.headers.get("content-type")}`);

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    logger.error(`[SSE-FE] fetch FAILED status=${res.status} body=${text.slice(0, 200)}`);
    throw new Error(`API ${res.status}: ${text}`);
  }

  try {
    logger.info(`[SSE-FE] _processStream starting anima=${animaName}`);
    await _processStream(res, callbacks, (id) => { responseId = id; logger.info(`[SSE-FE] response_id set: ${id}`); }, (id) => { lastEventId = id; }, signal);
    logger.info(`[SSE-FE] _processStream completed anima=${animaName} responseId=${responseId} lastEventId=${lastEventId}`);
  } catch (err) {
    const elapsed = ((performance.now() - start) / 1000).toFixed(1);
    logger.info(`[SSE-FE] _processStream ERROR anima=${animaName} err=${err.name}:${err.message} elapsed=${elapsed}s responseId=${responseId} lastEventId=${lastEventId}`);
    if (err.name === "AbortError") throw err;

    // Attempt reconnection with exponential backoff
    if (responseId) {
      logger.info(`[SSE-FE] reconnect attempt anima=${animaName} responseId=${responseId} lastEventId=${lastEventId}`);
      const reconnected = await _reconnectWithBackoff(
        animaName, responseId, lastEventId, body, signal, callbacks,
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
 */
async function _processStream(res, callbacks, setResponseId, setLastEventId, signal) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let chunkCount = 0;
  let sseEventCount = 0;
  let lastChunkTime = performance.now();
  const streamStart = performance.now();

  logger.info("[SSE-FE] _processStream: reader opened");

  try {
    while (true) {
      if (signal?.aborted) throw new DOMException("Aborted", "AbortError");

      const readStart = performance.now();
      const { done, value } = await reader.read();
      const readMs = (performance.now() - readStart).toFixed(0);
      const gapMs = (performance.now() - lastChunkTime).toFixed(0);

      if (done) {
        const totalElapsed = ((performance.now() - streamStart) / 1000).toFixed(1);
        logger.info(`[SSE-FE] reader DONE chunks=${chunkCount} sseEvents=${sseEventCount} elapsed=${totalElapsed}s bufferRemaining=${buffer.length}`);
        break;
      }
      chunkCount++;
      lastChunkTime = performance.now();

      const rawLen = value ? value.byteLength : 0;
      buffer += decoder.decode(value, { stream: true });
      const { parsed, remaining } = parseConvSSE(buffer);
      buffer = remaining;

      if (chunkCount % 50 === 0 || parsed.length > 0) {
        logger.info(`[SSE-FE] chunk#${chunkCount} rawBytes=${rawLen} parsedEvents=${parsed.length} bufRemain=${remaining.length} readMs=${readMs} gapMs=${gapMs}`);
      }

      for (const { id, event, data } of parsed) {
        sseEventCount++;
        // Track event IDs for reconnection
        if (id) setLastEventId(id);

        switch (event) {
          case "stream_start":
            if (data.response_id) setResponseId(data.response_id);
            logger.info(`[SSE-FE] EVENT stream_start response_id=${data.response_id} id=${id}`);
            break;

          case "text_delta":
            // text_delta is high-frequency; log periodically
            if (sseEventCount % 20 === 0) {
              logger.info(`[SSE-FE] EVENT text_delta (periodic) sseEvent#${sseEventCount} delta_len=${(data.text || "").length} id=${id}`);
            }
            callbacks.onTextDelta?.(data.text || "");
            break;

          case "tool_start":
            logger.info(`[SSE-FE] EVENT tool_start tool=${data.tool_name} id=${id}`);
            callbacks.onToolStart?.(data.tool_name);
            break;

          case "tool_end":
            logger.info(`[SSE-FE] EVENT tool_end tool=${data.tool_name || "?"} id=${id}`);
            callbacks.onToolEnd?.();
            break;

          case "done": {
            const summaryLen = (data.summary || "").length;
            const totalElapsed = ((performance.now() - streamStart) / 1000).toFixed(1);
            logger.info(`[SSE-FE] EVENT done summary_len=${summaryLen} emotion=${data.emotion || "neutral"} totalEvents=${sseEventCount} elapsed=${totalElapsed}s id=${id}`);
            callbacks.onDone?.({
              summary: data.summary || null,
              emotion: data.emotion || "neutral",
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
            logger.info(`[SSE-FE] EVENT chain_start id=${id}`);
            callbacks.onChainStart?.();
            break;

          case "heartbeat_relay_start":
            logger.info(`[SSE-FE] EVENT heartbeat_relay_start msg=${data.message || ""} id=${id}`);
            callbacks.onHeartbeatRelayStart?.({ message: data.message || "" });
            break;

          case "heartbeat_relay":
            logger.info(`[SSE-FE] EVENT heartbeat_relay len=${(data.text || "").length} id=${id}`);
            callbacks.onHeartbeatRelay?.({ text: data.text || "" });
            break;

          case "heartbeat_relay_done":
            logger.info(`[SSE-FE] EVENT heartbeat_relay_done id=${id}`);
            callbacks.onHeartbeatRelayDone?.();
            break;

          default:
            logger.info(`[SSE-FE] EVENT unknown event=${event} id=${id}`);
            break;
        }
      }
    }
  } finally {
    const totalElapsed = ((performance.now() - streamStart) / 1000).toFixed(1);
    logger.info(`[SSE-FE] reader.releaseLock chunks=${chunkCount} sseEvents=${sseEventCount} elapsed=${totalElapsed}s`);
    reader.releaseLock();
  }
}

/**
 * Reconnect with exponential backoff (1s -> 2s -> 4s -> ... max 30s, 5 attempts).
 */
async function _reconnectWithBackoff(animaName, responseId, lastEventId, originalBody, signal, callbacks) {
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

      const url = `/api/animas/${encodeURIComponent(animaName)}/chat/stream`;
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
      await _processStream(res, callbacks, () => {}, (id) => { lastEventId = id; }, signal);
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
