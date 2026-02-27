// ── Shared SSE Parser ──────────────────────────────────
// Common SSE parsing and error message utilities used across all chat modules.

import { t } from "/shared/i18n.js";
import { createLogger } from './logger.js';

const logger = createLogger('sse-parser');

/**
 * Parse SSE (Server-Sent Events) buffer into structured events.
 * Uses "\n\n" as the block delimiter (standard SSE format).
 * @param {string} buffer - Raw SSE text buffer
 * @returns {{parsed: Array<{id: string|null, event: string, data: object}>, remaining: string}}
 */
export function parseConvSSE(buffer) {
  const parsed = [];
  const parts = buffer.split("\n\n");
  const remaining = parts.pop() || "";

  for (const part of parts) {
    if (!part.trim()) continue;
    // Skip SSE comments (keepalive)
    if (part.trim().startsWith(":")) {
      logger.info(`[SSE-PARSE] keepalive comment received: "${part.trim().slice(0, 50)}"`);
      continue;
    }
    let eventName = "message";
    let eventId = null;
    const dataLines = [];
    for (const line of part.split("\n")) {
      if (line.startsWith("event: ")) {
        eventName = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        dataLines.push(line.slice(6));
      } else if (line.startsWith("id: ")) {
        eventId = line.slice(4).trim();
      }
    }
    if (dataLines.length > 0) {
      try {
        parsed.push({ id: eventId, event: eventName, data: JSON.parse(dataLines.join("\n")) });
      } catch {
        const raw = dataLines.join("\n");
        logger.warn(`[SSE-PARSE] JSON parse FAILED event='${eventName}': ${raw.slice(0, 100)}`);
      }
    }
  }
  if (parsed.length > 0) {
    const eventNames = parsed.map(e => e.event).join(",");
    logger.info(`[SSE-PARSE] parsed=${parsed.length} events=[${eventNames}] remaining=${remaining.length}`);
  }
  return { parsed, remaining };
}

/**
 * Get user-friendly error message for SSE error code.
 * @param {object} data - SSE error event data with code and message fields
 * @returns {string} User-friendly error message
 */
const SSE_ERROR_KEYS = {
  'IPC_TIMEOUT': 'sse.ipc_timeout',
  'TOOL_ERROR': 'sse.tool_error',
  'LLM_ERROR': 'sse.llm_error',
  'STREAM_ERROR': 'sse.stream_error',
  'ANIMA_NOT_FOUND': 'sse.anima_not_found',
};

export function getErrorMessage(data) {
  const key = SSE_ERROR_KEYS[data.code];
  if (key) return t(key);
  return data.error || data.message || t('sse.generic_error');
}
