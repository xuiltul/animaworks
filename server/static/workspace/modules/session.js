// ── Session & Conversation History ──────────────────────
// Session list, active conversation, archived sessions, transcripts, episodes.

import { getState, setState } from "./state.js";
import * as api from "./api.js";
import { escapeHtml, renderSimpleMarkdown, timeStr } from "./utils.js";
import { t } from "/shared/i18n.js";

const TOOL_RESULT_TRUNCATE = 500;

// Container reference for async operations
let _container = null;

// ── DOM Queries ──────────────────────

function findNodes() {
  if (!_container) return {};
  return {
    listArea: _container.querySelector(".session-list"),
    detailArea: _container.querySelector(".session-detail"),
  };
}

function showList() {
  const { listArea, detailArea } = findNodes();
  if (listArea) listArea.style.display = "";
  if (detailArea) detailArea.style.display = "none";
}

function showDetail(title) {
  const { listArea, detailArea } = findNodes();
  if (listArea) listArea.style.display = "none";
  if (detailArea) {
    detailArea.style.display = "";
    const titleEl = detailArea.querySelector(".session-detail-title");
    if (titleEl) titleEl.textContent = title;
  }
}

function getDetailBody() {
  if (!_container) return null;
  return _container.querySelector(".session-detail-body");
}

// ── Render ──────────────────────

/** Build the session panel DOM inside the given container. */
export function renderSessionList(container) {
  _container = container;

  container.innerHTML = `
    <div class="session-panel">
      <div class="session-list"></div>
      <div class="session-detail" style="display:none">
        <button class="session-back-btn">&larr; ${t("ws.back")}</button>
        <div class="session-detail-title"></div>
        <div class="session-detail-body"></div>
      </div>
    </div>`;

  // Bind back button
  container.querySelector(".session-back-btn").addEventListener("click", () => {
    showList();
  });
}

/** Initialize: render + load sessions. */
export function initSession(container) {
  _container = container;
  renderSessionList(container);
  loadSessions();
}

// ── Data Loading ──────────────────────

/** Load session list from API and render items. */
export async function loadSessions() {
  const { listArea } = findNodes();
  if (!listArea) return;

  const { selectedAnima } = getState();

  if (!selectedAnima) {
    listArea.innerHTML = `<div class="loading-placeholder">${t("ws.select_anima")}</div>`;
    return;
  }

  listArea.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const data = await api.fetchSessions(selectedAnima);
    setState({ sessionList: data });
    renderItems(data);
  } catch (err) {
    console.error("Failed to load sessions:", err);
    listArea.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

// ── List Rendering ──────────────────────

function renderItems(data) {
  const { listArea } = findNodes();
  if (!listArea) return;

  let html = "";

  // 1. Active conversation
  if (data.active_conversation && data.active_conversation.exists !== false) {
    const ac = data.active_conversation;
    const lastTime = ac.last_timestamp ? timeStr(ac.last_timestamp) : "--:--";
    html += `
      <div class="session-section">
        <div class="section-header">${t("chat.history_current")}</div>
        <div class="session-item session-active" data-type="active">
          <div class="session-item-title">${t("ws.conversation_active")}</div>
          <div class="session-item-meta">
            ${ac.total_turn_count || 0}${t("ws.turns")} ${ac.has_summary ? t("chat.session_summary") : ""}
            | 最終: ${lastTime}
          </div>
        </div>
      </div>`;
  }

  // 2. Archived sessions
  if (data.archived_sessions && data.archived_sessions.length > 0) {
    html += `<div class="session-section"><div class="section-header">${t("chat.history_archive")}</div>`;
    for (const s of data.archived_sessions) {
      const ts = s.timestamp ? timeStr(s.timestamp) : s.id;
      html += `
        <div class="session-item" data-type="archive" data-id="${escapeHtml(s.id)}">
          <div class="session-item-title">${escapeHtml(s.trigger || t("ws.session_default"))} (${s.turn_count || 0}${t("ws.turns")})</div>
          <div class="session-item-meta">${ts}</div>
        </div>`;
    }
    html += "</div>";
  }

  // 3. Transcripts
  if (data.transcripts && data.transcripts.length > 0) {
    html += `<div class="session-section"><div class="section-header">${t("chat.history_transcript")}</div>`;
    for (const tr of data.transcripts) {
      html += `
        <div class="session-item" data-type="transcript" data-date="${escapeHtml(tr.date)}">
          <div class="session-item-title">${escapeHtml(tr.date)}</div>
          <div class="session-item-meta">${t("chat.messages_count", { count: tr.message_count || 0 })}</div>
        </div>`;
    }
    html += "</div>";
  }

  // 4. Episodes
  if (data.episodes && data.episodes.length > 0) {
    html += '<div class="session-section"><div class="section-header">エピソード</div>';
    for (const e of data.episodes) {
      html += `
        <div class="session-item" data-type="episode" data-date="${escapeHtml(e.date)}">
          <div class="session-item-title">${escapeHtml(e.date)}</div>
          ${e.preview ? `<div class="session-item-preview">${escapeHtml(e.preview)}</div>` : ""}
        </div>`;
    }
    html += "</div>";
  }

  if (!html) {
    html = `<div class="loading-placeholder">${t("chat.history_empty")}</div>`;
  }

  listArea.innerHTML = html;

  // Bind click handlers
  listArea.querySelectorAll(".session-item").forEach((item) => {
    item.addEventListener("click", () => {
      const type = item.dataset.type;
      if (type === "active") loadActiveConversation();
      else if (type === "archive") loadArchivedSession(item.dataset.id);
      else if (type === "transcript") loadTranscript(item.dataset.date);
      else if (type === "episode") loadEpisode(item.dataset.date);
    });
  });
}

// ── Detail Rendering ──────────────────────

/** Render conversation turns (shared by active conversation and transcripts). */
function renderTurns(data) {
  const body = getDetailBody();
  if (!body) return;

  let html = "";

  // Summary box
  if (data.has_summary && data.compressed_summary) {
    html += `
      <div class="history-summary">
        <div class="history-summary-label">${t("ws.summary_label", { count: data.compressed_turn_count || 0 })}</div>
        <div class="history-summary-body">${renderSimpleMarkdown(data.compressed_summary)}</div>
      </div>`;
  }

  // Turn list
  if (data.turns && data.turns.length > 0) {
    for (const turn of data.turns) {
      const ts = turn.timestamp ? timeStr(turn.timestamp) : "";
      const roleClass = turn.role === "assistant" ? "assistant" : "user";
      const roleLabel = turn.role === "human" ? t("chat.role_human") : turn.role;
      const content =
        turn.role === "assistant" ? renderSimpleMarkdown(turn.content || "") : escapeHtml(turn.content || "");
      html += `
        <div class="session-turn">
          <div class="session-turn-meta">${ts} - ${escapeHtml(roleLabel)}</div>
          <div class="session-turn-bubble ${roleClass}">${content}</div>
        </div>`;
    }
  }

  if (!html) {
    html = `<div class="loading-placeholder">${t("chat.history_no_data")}</div>`;
  }

  body.innerHTML = html;
  body.scrollTop = body.scrollHeight;
}

// ── Tool Call Rendering ──────────────────────

function _renderToolCalls(toolCalls) {
  if (!toolCalls || toolCalls.length === 0) return "";

  return toolCalls.map((tc, idx) => {
    const errorClass = tc.is_error ? " tool-call-error" : "";
    const toolName = escapeHtml(tc.tool_name || "unknown");
    const errorLabel = tc.is_error ? " [ERROR]" : "";

    return `<div class="tool-call-row${errorClass}" data-tool-idx="${idx}">` +
      `<span class="tool-call-row-icon">\u25B6</span>` +
      `<span class="tool-call-row-name">${toolName}${errorLabel}</span>` +
      `</div>` +
      `<div class="tool-call-detail" data-tool-idx="${idx}" style="display:none;">` +
      _renderToolCallDetail(tc) +
      `</div>`;
  }).join("");
}

function _renderToolCallDetail(tc) {
  let html = "";

  const input = tc.input || "";
  if (input) {
    const inputStr = typeof input === "string" ? input : JSON.stringify(input, null, 2);
    html += `<div class="tool-call-label">${t("ws.tool_input")}</div>`;
    html += `<div class="tool-call-content">${escapeHtml(inputStr)}</div>`;
  }

  const result = tc.result || "";
  if (result) {
    const resultStr = typeof result === "string" ? result : JSON.stringify(result, null, 2);
    html += `<div class="tool-call-label">${t("ws.tool_result")}</div>`;
    if (resultStr.length > TOOL_RESULT_TRUNCATE) {
      const truncated = resultStr.slice(0, TOOL_RESULT_TRUNCATE);
      html += `<div class="tool-call-content" data-full-result="${escapeHtml(resultStr)}">${escapeHtml(truncated)}...</div>`;
      html += `<button class="tool-call-show-more">${t("ws.show_more")}</button>`;
    } else {
      html += `<div class="tool-call-content">${escapeHtml(resultStr)}</div>`;
    }
  }

  return html;
}

function _bindToolCallHandlers(container) {
  if (!container) return;

  container.querySelectorAll(".tool-call-row").forEach(row => {
    row.addEventListener("click", () => {
      const idx = row.dataset.toolIdx;
      const detail = row.nextElementSibling;
      if (!detail || detail.dataset.toolIdx !== idx) return;

      const isExpanded = row.classList.contains("expanded");
      if (isExpanded) {
        row.classList.remove("expanded");
        detail.style.display = "none";
      } else {
        row.classList.add("expanded");
        detail.style.display = "";
      }
    });
  });

  container.querySelectorAll(".tool-call-show-more").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const contentEl = btn.previousElementSibling;
      if (!contentEl) return;
      const fullResult = contentEl.dataset.fullResult;
      if (fullResult) {
        contentEl.textContent = fullResult;
        delete contentEl.dataset.fullResult;
        btn.remove();
      }
    });
  });
}

function renderConversationSessions(data) {
  const body = getDetailBody();
  if (!body) return;

  if (!data || !data.sessions || data.sessions.length === 0) {
    body.innerHTML = `<div class="loading-placeholder">${t("chat.history_no_data")}</div>`;
    return;
  }

  let html = "";
  for (const session of data.sessions) {
    const startTs = session.session_start ? timeStr(session.session_start) : "";
    const triggerLabel = session.trigger === "heartbeat" ? t("ws.trigger_heartbeat")
      : session.trigger === "cron" ? t("ws.trigger_cron") : "";
    html += `<div class="session-divider">${triggerLabel ? `<span class="session-trigger">${escapeHtml(triggerLabel)}</span> ` : ""}${startTs}</div>`;

    if (session.messages) {
      for (const msg of session.messages) {
        const ts = msg.ts ? timeStr(msg.ts) : "";
        const roleClass = msg.role === "assistant" ? "assistant" : msg.role === "system" ? "system" : "user";
        const roleLabel = msg.role === "human" ? t("chat.role_human") : msg.role === "system" ? t("chat.role_system") : msg.role;
        const content = msg.role === "assistant" ? renderSimpleMarkdown(msg.content || "") : escapeHtml(msg.content || "");
        const toolHtml = msg.role === "assistant" ? _renderToolCalls(msg.tool_calls) : "";
        html += `
          <div class="session-turn">
            <div class="session-turn-meta">${ts} - ${escapeHtml(roleLabel)}</div>
            <div class="session-turn-bubble ${roleClass}">${content}${toolHtml}</div>
          </div>`;
      }
    }
  }

  if (!html) html = `<div class="loading-placeholder">${t("chat.history_no_data")}</div>`;
  body.innerHTML = html;
  _bindToolCallHandlers(body);
  body.scrollTop = body.scrollHeight;
}

async function loadActiveConversation() {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  showDetail(t("ws.conversation_active"));
  const body = getDetailBody();
  if (body) body.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const data = await api.fetchConversationHistory(selectedAnima);
    renderConversationSessions(data);
  } catch (err) {
    console.error("Failed to load active conversation:", err);
    if (body) body.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

async function loadArchivedSession(sessionId) {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  showDetail(t("chat.session_detail", { id: sessionId }));
  const body = getDetailBody();
  if (body) body.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const data = await api.fetchSession(selectedAnima, sessionId);
    renderArchivedDetail(data);
  } catch (err) {
    console.error("Failed to load archived session:", err);
    if (body) body.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

function renderArchivedDetail(data) {
  const body = getDetailBody();
  if (!body) return;

  if (data.markdown) {
    body.innerHTML = `<div class="session-markdown">${renderSimpleMarkdown(data.markdown)}</div>`;
    return;
  }

  if (data.data) {
    const d = data.data;
    let html = `
      <div class="session-meta-block">
        <div><strong>${t("chat.state_label")}</strong> ${escapeHtml(d.trigger || t("common.unknown"))}</div>
        <div><strong>${t("chat.turn_count")}</strong> ${d.turn_count || 0}</div>
        <div><strong>${t("chat.context_usage")}</strong> ${((d.context_usage_ratio || 0) * 100).toFixed(0)}%</div>
      </div>`;

    if (d.original_prompt) {
      html += `
        <div class="session-section-block">
          <div class="session-section-label">${t("chat.request_label")}</div>
          <pre class="session-pre">${escapeHtml(d.original_prompt)}</pre>
        </div>`;
    }

    if (d.accumulated_response) {
      html += `
        <div class="session-section-block">
          <div class="session-section-label">${t("chat.response_label")}</div>
          <div>${renderSimpleMarkdown(d.accumulated_response)}</div>
        </div>`;
    }

    body.innerHTML = html;
    return;
  }

  body.innerHTML = `<div class="loading-placeholder">${t("chat.no_data")}</div>`;
}

async function loadTranscript(date) {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  showDetail(t("chat.transcript_detail", { date }));
  const body = getDetailBody();
  if (body) body.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const data = await api.fetchTranscript(selectedAnima, date);
    renderTurns(data);
  } catch (err) {
    console.error("Failed to load transcript:", err);
    if (body) body.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

async function loadEpisode(date) {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  showDetail(t("chat.episode_detail", { date }));
  const body = getDetailBody();
  if (body) body.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const data = await api.fetchEpisode(selectedAnima, date);
    body.innerHTML = `<div class="session-markdown">${renderSimpleMarkdown(data.content || t("chat.no_content"))}</div>`;
  } catch (err) {
    console.error("Failed to load episode:", err);
    if (body) body.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}
