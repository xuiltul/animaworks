// ── Chat Work Indicator — live tool/thinking/bg-task strip above input ──
import { CONSTANTS } from "./ctx.js";
import { normalizeRunningTasks } from "../../shared/activity-context.js";

const TOOL_END_HOLD_MS = 1500;
const WS_TOOL_HOLD_MS = 3000;
const STREAM_STALE_MS = 30000;

/**
 * Mini indicator above the chat input that surfaces:
 * 1. Streaming tool activity (SSE)
 * 2. Thinking state
 * 3. Background running tasks (polled)
 * 4. Non-streaming WS tool_activity (brief flash)
 *
 * @param {ReturnType<import("./ctx.js").createChatContext>} ctx
 */
export function createWorkIndicatorController(ctx) {
  const $ = ctx.$;
  const { state, deps } = ctx;
  const { t, api, escapeHtml, getIcon } = deps;

  /** @type {{ name: string, detail: string, completed: boolean, isError: boolean } | null} */
  let _streamingTool = null;
  let _thinking = false;
  /** @type {{ name: string, detail: string } | null} */
  let _wsTool = null;
  /** @type {Array<Record<string, unknown>>} */
  let _bgTasks = [];

  let _toolEndTimer = null;
  let _wsToolTimer = null;
  let _staleTimer = null;
  let _pollTimer = null;
  let _destroyed = false;
  let _pollInFlight = false;

  function _container() {
    return $("chatWorkIndicator");
  }

  function _clearTimer(key) {
    if (key === "toolEnd" && _toolEndTimer) {
      clearTimeout(_toolEndTimer);
      _toolEndTimer = null;
    } else if (key === "wsTool" && _wsToolTimer) {
      clearTimeout(_wsToolTimer);
      _wsToolTimer = null;
    } else if (key === "stale" && _staleTimer) {
      clearTimeout(_staleTimer);
      _staleTimer = null;
    }
  }

  function _bumpStaleTimer() {
    _clearTimer("stale");
    if (!_streamingTool && !_thinking) return;
    _staleTimer = setTimeout(() => {
      _streamingTool = null;
      _thinking = false;
      _render();
    }, STREAM_STALE_MS);
  }

  function _isStreamingSelected() {
    const name = state.selectedAnima;
    if (!name) return false;
    return !!state.manager?.isStreamingForAnima?.(name);
  }

  function _hasActiveContent() {
    return !!(_streamingTool || _thinking || _wsTool || _bgTasks.length > 0);
  }

  function _toolLabel(toolName, detail) {
    const base = toolName || "tool";
    if (detail) return `${base}: ${detail}`;
    return base;
  }

  function _chipHtml({ iconHtml, text, className = "", title = "" }) {
    const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
    return `<span class="chat-work-chip ${className}"${titleAttr}>` +
      `<span class="chat-work-chip-icon">${iconHtml}</span>` +
      `<span class="chat-work-chip-text">${escapeHtml(text)}</span>` +
      `</span>`;
  }

  function _render() {
    const el = _container();
    if (!el || _destroyed) return;

    if (!_hasActiveContent()) {
      el.innerHTML = "";
      el.hidden = true;
      return;
    }

    el.hidden = false;
    const parts = [];

    // Priority 1: streaming tool
    if (_streamingTool) {
      const label = _toolLabel(_streamingTool.name, _streamingTool.detail);
      if (_streamingTool.completed) {
        const mark = _streamingTool.isError ? "✗" : "✓";
        const doneText = `${mark} ${t("chat.tool_done", { tool: _streamingTool.name || "tool" })}`;
        parts.push(_chipHtml({
          iconHtml: getIcon("tool_result"),
          text: doneText,
          className: _streamingTool.isError ? "chat-work-chip--error" : "chat-work-chip--done",
          title: label,
        }));
      } else {
        parts.push(_chipHtml({
          iconHtml: getIcon("tool_use"),
          text: t("chat.work_tool_running", { tool: label }),
          className: "chat-work-chip--running",
          title: label,
        }));
      }
    }

    // Priority 2: thinking
    if (_thinking) {
      parts.push(_chipHtml({
        iconHtml: getIcon("heartbeat_reflection"),
        text: t("chat.work_thinking"),
        className: "chat-work-chip--thinking",
      }));
    }

    // Priority 3: background tasks
    if (_bgTasks.length > 0) {
      const countLabel = t("chat.work_bg_tasks", { count: _bgTasks.length });
      parts.push(
        `<span class="chat-work-chip chat-work-chip--bg chat-work-bg-heading">` +
        `<span class="chat-work-chip-text">${escapeHtml(countLabel)}</span>` +
        `</span>`,
      );
      for (const task of _bgTasks.slice(0, 8)) {
        const taskTitle = String(task.title || task.task_id || "");
        const chipText = taskTitle;
        parts.push(
          `<span class="chat-work-chip chat-work-chip--bg running-task-chip" ` +
          `data-slot-id="${escapeHtml(String(task.slot_id ?? ""))}" ` +
          `data-task-id="${escapeHtml(String(task.task_id || ""))}" ` +
          `title="${escapeHtml(taskTitle)}">` +
          `<span class="chat-work-chip-text">${escapeHtml(chipText)}</span>` +
          `</span>`,
        );
      }
    }

    // Priority 4: WS tool activity (non-streaming)
    if (_wsTool && !_streamingTool) {
      const label = _toolLabel(_wsTool.name, _wsTool.detail);
      parts.push(_chipHtml({
        iconHtml: getIcon("tool_use"),
        text: t("chat.work_tool_running", { tool: label }),
        className: "chat-work-chip--ws",
        title: label,
      }));
    }

    el.innerHTML = parts.join("");
    if (window.lucide) {
      try {
        window.lucide.createIcons({ nodes: [el] });
      } catch { /* optional */ }
    }
  }

  // ── SSE / streaming callbacks ──────────────────────────

  function onToolStart(toolName, detail) {
    if (_destroyed) return;
    _clearTimer("toolEnd");
    const detailText = typeof detail === "string"
      ? detail
      : (detail?.detail || detail?.input_summary || "");
    _streamingTool = {
      name: toolName || "tool",
      detail: detailText || "",
      completed: false,
      isError: false,
    };
    _wsTool = null;
    _clearTimer("wsTool");
    _bumpStaleTimer();
    _render();
  }

  function onToolDetail(toolName, detailText, _info) {
    if (_destroyed || !_streamingTool) return;
    if (toolName) _streamingTool.name = toolName;
    if (detailText != null && detailText !== "") {
      _streamingTool.detail = String(detailText);
    }
    _bumpStaleTimer();
    _render();
  }

  function onToolEnd(detail) {
    if (_destroyed) return;
    if (!_streamingTool) {
      // Still show a brief done flash if we missed start
      _streamingTool = {
        name: detail?.tool_name || "tool",
        detail: "",
        completed: true,
        isError: !!detail?.is_error,
      };
    } else {
      _streamingTool.completed = true;
      _streamingTool.isError = !!detail?.is_error;
      if (detail?.tool_name) _streamingTool.name = detail.tool_name;
    }
    _bumpStaleTimer();
    _render();
    _clearTimer("toolEnd");
    _toolEndTimer = setTimeout(() => {
      _streamingTool = null;
      _toolEndTimer = null;
      _render();
    }, TOOL_END_HOLD_MS);
  }

  function onThinkingStart() {
    if (_destroyed) return;
    _thinking = true;
    _bumpStaleTimer();
    _render();
  }

  function onThinkingEnd() {
    if (_destroyed) return;
    _thinking = false;
    _clearTimer("stale");
    if (_streamingTool && !_streamingTool.completed) _bumpStaleTimer();
    _render();
  }

  function onStreamSettled() {
    if (_destroyed) return;
    // Keep completed-tool flash if mid-hold; otherwise clear streaming state
    if (_streamingTool && !_streamingTool.completed) {
      _streamingTool = null;
    }
    _thinking = false;
    _clearTimer("stale");
    _render();
  }

  // ── WS anima-tool-activity ─────────────────────────────

  function _onToolActivity(ev) {
    if (_destroyed) return;
    const data = ev?.detail || {};
    const animaName = data.name;
    if (!animaName || animaName !== state.selectedAnima) return;

    // Streaming path already covers in-chat tools
    if (_isStreamingSelected() || _streamingTool) return;

    const evtType = data.event || data.type || "";
    const toolName = data.tool_name || data.tool || "tool";

    if (evtType === "tool_start" || evtType === "tool_detail" || evtType === "tool_use") {
      _wsTool = {
        name: toolName,
        detail: data.detail || data.summary || "",
      };
      _render();
      _clearTimer("wsTool");
      _wsToolTimer = setTimeout(() => {
        _wsTool = null;
        _wsToolTimer = null;
        _render();
      }, WS_TOOL_HOLD_MS);
    } else if (evtType === "tool_end") {
      _wsTool = {
        name: toolName,
        detail: data.is_error ? t("common.error") : "",
      };
      _render();
      _clearTimer("wsTool");
      _wsToolTimer = setTimeout(() => {
        _wsTool = null;
        _wsToolTimer = null;
        _render();
      }, WS_TOOL_HOLD_MS);
    }
  }

  // ── Background running-tasks poll ──────────────────────

  async function _pollRunningTasks() {
    if (_destroyed || _pollInFlight) return;
    const anima = state.selectedAnima;
    if (!anima) {
      if (_bgTasks.length) {
        _bgTasks = [];
        _render();
      }
      return;
    }
    _pollInFlight = true;
    try {
      const data = await api(
        `/api/activity/running-tasks?anima=${encodeURIComponent(anima)}`,
      );
      if (_destroyed || state.selectedAnima !== anima) return;
      // Keep only selected anima's tasks (API already filters, normalize is defensive)
      const tasks = normalizeRunningTasks(data).filter(
        (task) => !task.anima || task.anima === anima,
      );
      _bgTasks = tasks;
      _render();
    } catch (err) {
      console.error("Failed to load running activity tasks:", err);
      // Hide bg chips only; leave other sources intact
      if (_bgTasks.length) {
        _bgTasks = [];
        _render();
      }
    } finally {
      _pollInFlight = false;
    }
  }

  function onAnimaChange() {
    if (_destroyed) return;
    _streamingTool = null;
    _thinking = false;
    _wsTool = null;
    _bgTasks = [];
    _clearTimer("toolEnd");
    _clearTimer("wsTool");
    _clearTimer("stale");
    _render();
    void _pollRunningTasks();
  }

  function destroy() {
    _destroyed = true;
    document.removeEventListener("anima-tool-activity", _onToolActivity);
    _clearTimer("toolEnd");
    _clearTimer("wsTool");
    _clearTimer("stale");
    if (_pollTimer) {
      clearInterval(_pollTimer);
      _pollTimer = null;
    }
    _streamingTool = null;
    _thinking = false;
    _wsTool = null;
    _bgTasks = [];
    const el = _container();
    if (el) {
      el.innerHTML = "";
      el.hidden = true;
    }
  }

  // Wire listeners / poll on create
  document.addEventListener("anima-tool-activity", _onToolActivity);
  _pollTimer = setInterval(() => {
    void _pollRunningTasks();
  }, CONSTANTS.CHAT_POLL_INTERVAL_MS);
  // Immediate first poll when anima already selected
  if (state.selectedAnima) void _pollRunningTasks();

  return {
    onToolStart,
    onToolDetail,
    onToolEnd,
    onThinkingStart,
    onThinkingEnd,
    onStreamSettled,
    onAnimaChange,
    destroy,
    /** @internal test/debug */
    _render,
    get _state() {
      return {
        streamingTool: _streamingTool,
        thinking: _thinking,
        wsTool: _wsTool,
        bgTasks: _bgTasks,
      };
    },
  };
}
