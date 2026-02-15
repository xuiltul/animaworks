// ── Shared SSE Parser ──────────────────────────────────
// Common SSE parsing and error message utilities used across all chat modules.

/**
 * Parse SSE (Server-Sent Events) buffer into structured events.
 * Uses "\n\n" as the block delimiter (standard SSE format).
 * @param {string} buffer - Raw SSE text buffer
 * @returns {{parsed: Array<{event: string, data: object}>, remaining: string}}
 */
export function parseConvSSE(buffer) {
  const parsed = [];
  const parts = buffer.split("\n\n");
  const remaining = parts.pop() || "";

  for (const part of parts) {
    if (!part.trim()) continue;
    let eventName = "message";
    const dataLines = [];
    for (const line of part.split("\n")) {
      if (line.startsWith("event: ")) {
        eventName = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        dataLines.push(line.slice(6));
      }
    }
    if (dataLines.length > 0) {
      try {
        parsed.push({ event: eventName, data: JSON.parse(dataLines.join("\n")) });
      } catch { /* skip non-JSON data */ }
    }
  }
  return { parsed, remaining };
}

/**
 * Get user-friendly error message for SSE error code.
 * @param {object} data - SSE error event data with code and message fields
 * @returns {string} User-friendly error message
 */
export function getErrorMessage(data) {
  const messages = {
    'IPC_TIMEOUT': '応答がタイムアウトしました',
    'TOOL_ERROR': 'ツール実行中にエラーが発生しました',
    'LLM_ERROR': 'AIモデルとの通信でエラーが発生しました',
    'STREAM_ERROR': '通信エラーが発生しました',
    'PERSON_NOT_FOUND': 'Personが見つかりませんでした',
  };
  return messages[data.code] || data.error || data.message || 'エラーが発生しました';
}
