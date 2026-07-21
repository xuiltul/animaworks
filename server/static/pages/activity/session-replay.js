// ── Session replay for activity groups ──────────────────

import { api } from "../../modules/api.js";
import { escapeHtml, renderSafeMarkdown, smartTimestamp } from "../../modules/state.js";
import { getDisplaySummary, getIcon } from "../../shared/activity-types.js";
import { t } from "/shared/i18n.js";
import { isGroupInProgress } from "./swimlane-layout.js";

export const REPLAY_POLL_INTERVAL_MS = 5000;

let _activeReplay = null;
let _replayGeneration = 0;

function _refreshIcons(container) {
  if (typeof window !== "undefined" && window.lucide && container) {
    window.lucide.createIcons({ nodes: [container] });
  }
}

function _eventId(evt, index) {
  if (evt?.id) return String(evt.id);
  const toolUseId = evt?.meta?.tool_use_id || "";
  return `${evt?.ts || ""}:${evt?.type || ""}:${toolUseId}:${index}`;
}

function _eventSignature(evt) {
  try {
    return JSON.stringify(evt);
  } catch {
    return `${evt?.ts || ""}:${evt?.type || ""}:${evt?.content || ""}`;
  }
}

function _messageContent(evt) {
  return evt?.content || evt?.summary || "";
}

function _formatToolValue(value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function _formatDuration(evt) {
  const candidates = [
    evt?.meta?.duration_ms,
    evt?.meta?.elapsed_ms,
    evt?.tool_result?.duration_ms,
  ];
  const durationMs = candidates.map(Number).find(Number.isFinite);
  const startMs = Date.parse(evt?.ts || "");
  const resultMs = Date.parse(evt?.tool_result?.ts || "");
  const elapsedMs = durationMs ?? (
    Number.isFinite(startMs) && Number.isFinite(resultMs)
      ? Math.max(0, resultMs - startMs)
      : undefined
  );
  if (elapsedMs === undefined) return t("activity.replay_duration_unknown");
  if (elapsedMs < 1000) return `${Math.round(elapsedMs)}ms`;
  return `${(elapsedMs / 1000).toFixed(elapsedMs < 10_000 ? 1 : 0)}s`;
}

function _toolStatus(result) {
  if (!result) {
    return `<span class="replay-tool-status replay-tool-status--pending" aria-label="${escapeHtml(t("activity.replay_tool_pending"))}">…</span>`;
  }
  const isError = result.is_error === true;
  const label = isError ? t("activity.replay_tool_error") : t("activity.replay_tool_success");
  const icon = isError
    ? '<span class="replay-tool-status-emoji" aria-hidden="true">✗</span><i data-lucide="x" aria-hidden="true"></i>'
    : '<span class="replay-tool-status-emoji" aria-hidden="true">✓</span><i data-lucide="check" aria-hidden="true"></i>';
  return `<span class="replay-tool-status ${isError ? "is-error" : "is-success"}" aria-label="${escapeHtml(label)}">${icon}</span>`;
}

function _renderMessage(evt, direction) {
  const row = document.createElement("div");
  row.className = `replay-message replay-message--${direction}`;

  const bubble = document.createElement("div");
  bubble.className = "replay-bubble";

  if (direction === "outgoing" && evt?.meta?.thinking_text) {
    const thinking = document.createElement("details");
    thinking.className = "replay-thinking";
    thinking.innerHTML =
      `<summary>${getIcon("heartbeat_reflection")}<span>${escapeHtml(t("activity.replay_thinking"))}</span></summary>` +
      `<div class="replay-thinking-content">${escapeHtml(String(evt.meta.thinking_text))}</div>`;
    bubble.appendChild(thinking);
  }

  const content = document.createElement("div");
  content.className = "replay-bubble-content activity-markdown";
  content.innerHTML = renderSafeMarkdown(String(_messageContent(evt)));
  bubble.appendChild(content);

  const time = document.createElement("time");
  time.className = "replay-event-time";
  time.dateTime = evt?.ts || "";
  time.textContent = smartTimestamp(evt?.ts);
  bubble.appendChild(time);

  row.appendChild(bubble);
  return row;
}

function _renderTool(evt) {
  const row = document.createElement("div");
  row.className = "replay-tool-row";

  const details = document.createElement("details");
  details.className = `replay-tool${evt?.tool_result?.is_error ? " is-error" : ""}`;
  const toolName = evt?.tool || t("activity.replay_tool_unknown");
  const duration = _formatDuration(evt);
  details.innerHTML =
    `<summary class="replay-tool-chip">` +
    `<span class="replay-tool-icon">${getIcon("tool_use")}</span>` +
    `<span class="replay-tool-name">${escapeHtml(toolName)}</span>` +
    `<span class="replay-tool-duration">${escapeHtml(duration)}</span>` +
    _toolStatus(evt?.tool_result) +
    `<i class="replay-tool-chevron" data-lucide="chevron-down" aria-hidden="true"></i>` +
    `</summary>`;

  const expanded = document.createElement("div");
  expanded.className = "replay-tool-expanded";
  const args = _formatToolValue(evt?.meta?.args ?? (evt?.content || evt?.summary));
  const result = _formatToolValue(evt?.tool_result?.content);

  const argsBlock = document.createElement("div");
  argsBlock.className = "replay-tool-block";
  argsBlock.innerHTML = `<div class="replay-tool-block-label">${escapeHtml(t("activity.replay_tool_arguments"))}</div><pre>${escapeHtml(args || t("activity.replay_empty"))}</pre>`;
  expanded.appendChild(argsBlock);

  const resultBlock = document.createElement("div");
  resultBlock.className = "replay-tool-block";
  resultBlock.innerHTML = `<div class="replay-tool-block-label">${escapeHtml(t("activity.replay_tool_result"))}</div><pre>${escapeHtml(result || (evt?.tool_result ? t("activity.replay_empty") : t("activity.replay_tool_pending")))}</pre>`;
  expanded.appendChild(resultBlock);

  details.appendChild(expanded);
  row.appendChild(details);
  return row;
}

function _renderSystem(evt) {
  const row = document.createElement("div");
  row.className = `replay-system-row${evt?.type === "error" ? " is-error" : ""}`;
  const text = getDisplaySummary(evt) || evt?.content || evt?.type || "";
  row.innerHTML =
    `<span class="replay-system-line" aria-hidden="true"></span>` +
    `<span class="replay-system-content">` +
    `<span class="replay-system-icon">${getIcon(evt?.type)}</span>` +
    `<span>${escapeHtml(text)}</span>` +
    `<time datetime="${escapeHtml(evt?.ts || "")}">${escapeHtml(smartTimestamp(evt?.ts))}</time>` +
    `</span>` +
    `<span class="replay-system-line" aria-hidden="true"></span>`;
  return row;
}

function _renderEvent(evt) {
  if (evt?.type === "message_received") return _renderMessage(evt, "incoming");
  if (evt?.type === "response_sent") return _renderMessage(evt, "outgoing");
  if (evt?.type === "tool_use") return _renderTool(evt);
  return _renderSystem(evt);
}

function _isNearBottom(scroll) {
  return scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight <= 48;
}

function _appendEvents(replay, events) {
  for (const { evt, id, signature } of events) {
    const row = _renderEvent(evt);
    row.dataset.eventId = id;
    replay.list.appendChild(row);
    replay.eventSignatures.set(id, signature);
  }
  _refreshIcons(replay.list);
}

function _replaceEvents(replay, events) {
  const previousTop = replay.scroll.scrollTop;
  replay.list.innerHTML = "";
  replay.eventSignatures.clear();
  _appendEvents(replay, events);
  if (replay.followTail) replay.scroll.scrollTop = replay.scroll.scrollHeight;
  else replay.scroll.scrollTop = previousTop;
}

function _normalizedEvents(group) {
  return (group?.events || []).map((evt, index) => ({
    evt,
    id: _eventId(evt, index),
    signature: _eventSignature(evt),
  }));
}

function _updateLiveStatus(replay, group) {
  const live = isGroupInProgress(group, Date.now());
  replay.status.hidden = false;
  replay.status.classList.toggle("is-live", live);
  replay.status.innerHTML = live
    ? `<span class="replay-live-dot" aria-hidden="true"></span>${escapeHtml(t("activity.replay_live"))}`
    : escapeHtml(t("activity.replay_finished"));
  return live;
}

function _showError(replay, err) {
  const notFound = String(err?.message || "").includes("API 404");
  replay.scroll.innerHTML =
    `<div class="replay-error" role="alert">${getIcon("error")}<span>${escapeHtml(t(notFound ? "activity.replay_not_found" : "activity.replay_load_failed"))}</span></div>`;
  replay.status.hidden = true;
  _refreshIcons(replay.scroll);
}

function _schedulePoll(replay) {
  if (_activeReplay !== replay || replay.destroyed) return;
  replay.pollTimer = setTimeout(() => _poll(replay), REPLAY_POLL_INTERVAL_MS);
}

async function _poll(replay) {
  if (_activeReplay !== replay || replay.destroyed || !replay.container.isConnected) return;
  try {
    const data = await api(replay.url, { signal: replay.controller.signal });
    if (_activeReplay !== replay || replay.destroyed) return;
    const group = data?.group;
    const events = _normalizedEvents(group);
    const changedExisting = events.some(({ id, signature }) =>
      replay.eventSignatures.has(id) && replay.eventSignatures.get(id) !== signature,
    );
    const unseen = events.filter(({ id }) => !replay.eventSignatures.has(id));

    if (changedExisting) {
      _replaceEvents(replay, events);
    } else if (unseen.length > 0) {
      _appendEvents(replay, unseen);
      if (replay.followTail) replay.scroll.scrollTop = replay.scroll.scrollHeight;
    }

    replay.group = group;
    replay.onGroupUpdate?.(group, { changed: changedExisting || unseen.length > 0 });
    if (_updateLiveStatus(replay, group)) _schedulePoll(replay);
  } catch (err) {
    if (err?.name === "AbortError" || replay.destroyed) return;
    _showError(replay, err);
  }
}

/** Stop the current replay and its pending request/timer. */
export function destroySessionReplay() {
  _replayGeneration += 1;
  if (!_activeReplay) return;
  _activeReplay.destroyed = true;
  if (_activeReplay.pollTimer) clearTimeout(_activeReplay.pollTimer);
  _activeReplay.controller.abort();
  _activeReplay = null;
}

/**
 * Load and render one activity group as a conversation replay.
 * @param {HTMLElement} container
 * @param {{anima: string, groupId: string, onGroupUpdate?: Function}} options
 */
export async function renderSessionReplay(container, { anima, groupId, onGroupUpdate } = {}) {
  destroySessionReplay();
  if (!container) return;

  const generation = _replayGeneration;
  container.innerHTML = `<div class="replay-loading" role="status">${getIcon("status")}<span>${escapeHtml(t("activity.replay_loading"))}</span></div>`;
  _refreshIcons(container);

  const replay = {
    anima: anima || "",
    groupId: groupId || "",
    container,
    controller: new AbortController(),
    destroyed: false,
    pollTimer: null,
    followTail: true,
    eventSignatures: new Map(),
    onGroupUpdate,
  };
  replay.url = `/api/activity/group?anima=${encodeURIComponent(replay.anima)}&id=${encodeURIComponent(replay.groupId)}`;
  _activeReplay = replay;

  try {
    const data = await api(replay.url, { signal: replay.controller.signal });
    if (_activeReplay !== replay || replay.destroyed || generation !== _replayGeneration) return;

    container.innerHTML =
      `<div class="replay-status" aria-live="polite"></div>` +
      `<div class="replay-scroll" tabindex="0"><div class="replay-events" role="log" aria-live="polite"></div></div>`;
    replay.status = container.querySelector(".replay-status");
    replay.scroll = container.querySelector(".replay-scroll");
    replay.list = container.querySelector(".replay-events");
    replay.scroll.addEventListener("scroll", () => {
      replay.followTail = _isNearBottom(replay.scroll);
    }, { passive: true });

    replay.group = data?.group;
    _appendEvents(replay, _normalizedEvents(replay.group));
    replay.scroll.scrollTop = replay.scroll.scrollHeight;
    replay.onGroupUpdate?.(replay.group, { changed: true });
    if (_updateLiveStatus(replay, replay.group)) _schedulePoll(replay);
  } catch (err) {
    if (err?.name === "AbortError" || replay.destroyed) return;
    container.innerHTML = '<div class="replay-scroll"></div><div class="replay-status" hidden></div>';
    replay.scroll = container.querySelector(".replay-scroll");
    replay.status = container.querySelector(".replay-status");
    _showError(replay, err);
  }
}
