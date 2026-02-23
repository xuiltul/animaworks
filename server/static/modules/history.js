/* ── History Panel ─────────────────────────── */

import { state, dom, escapeHtml, renderMarkdown, smartTimestamp } from "./state.js";
import { api } from "./api.js";
import {
  renderHistoryMessage,
  renderSessionDivider,
  bindToolCallHandlers,
  fetchConversationHistory,
} from "./chat.js";

// ── Local State ──────────────────────────────

let _sessions = [];
let _hasMore = false;
let _nextBefore = null;
let _loading = false;
let _scrollObserver = null;

const HISTORY_PAGE_SIZE = 50;

// ── DOM Helpers ─────────────────────────────

function getHistoryConv() {
  return dom.historyConversation || document.getElementById("historyConversation");
}

// ── Session List ────────────────────────────

export async function loadSessionList() {
  const sessionList = dom.historySessionList || document.getElementById("historySessionList");
  if (!sessionList) return; // History panel not in DOM

  const name = state.selectedAnima;
  if (!name) {
    sessionList.innerHTML = '<div class="loading-placeholder">Anima\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044</div>';
    return;
  }

  // Show the conversation history view directly (no session list)
  // The history panel now shows the same activity_log-based conversation view
  sessionList.innerHTML = '';

  // Reset state
  _sessions = [];
  _hasMore = false;
  _nextBefore = null;
  _loading = false;

  await _loadConversationHistory(name, true);
}

// ── Conversation History Loading ────────────

async function _loadConversationHistory(animaName, initial = false) {
  const sessionList = dom.historySessionList || document.getElementById("historySessionList");
  if (!sessionList) return;

  if (_loading) return;
  _loading = true;

  if (initial) {
    sessionList.innerHTML = '<div class="loading-placeholder">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
  }

  const data = await fetchConversationHistory(animaName, HISTORY_PAGE_SIZE, initial ? null : _nextBefore);

  if (data && data.sessions && data.sessions.length > 0) {
    if (initial) {
      _sessions = data.sessions;
    } else {
      _sessions = [..._sessions, ...data.sessions];
    }
    _hasMore = data.has_more || false;
    _nextBefore = data.next_before || null;
  } else if (initial) {
    _sessions = [];
    _hasMore = false;
    _nextBefore = null;
  } else {
    _hasMore = false;
  }

  _loading = false;
  _renderConversationView(sessionList);
}

// ── Conversation View Rendering ──────────────

function _renderConversationView(container) {
  if (!container) return;

  if (_sessions.length === 0 && !_loading) {
    container.innerHTML = '<div class="loading-placeholder">\u4F1A\u8A71\u5C65\u6B74\u304C\u3042\u308A\u307E\u305B\u3093</div>';
    return;
  }

  let html = "";

  // Render sessions
  for (let si = 0; si < _sessions.length; si++) {
    const session = _sessions[si];
    html += renderSessionDivider(session, si === 0);

    if (session.messages) {
      for (const msg of session.messages) {
        html += renderHistoryMessage(msg);
      }
    }
  }

  // "Load more" button at bottom
  if (_hasMore) {
    html += '<div class="history-load-more-wrap" style="text-align:center; padding:12px;">';
    html += '<button class="activity-load-more" id="historyLoadMoreBtn">\u3082\u3063\u3068\u8AAD\u307F\u8FBC\u3080</button>';
    html += '</div>';
  }

  container.innerHTML = html;

  // Bind tool call handlers
  bindToolCallHandlers(container);

  // Bind "load more" button
  const loadMoreBtn = container.querySelector("#historyLoadMoreBtn");
  if (loadMoreBtn) {
    loadMoreBtn.addEventListener("click", async () => {
      loadMoreBtn.disabled = true;
      loadMoreBtn.textContent = "\u8AAD\u307F\u8FBC\u307F\u4E2D...";
      const name = state.selectedAnima;
      if (name) {
        await _loadConversationHistory(name, false);
      }
    });
  }

  // Scroll to bottom on initial load
  container.scrollTop = container.scrollHeight;
}

// ── Detail View ─────────────────────────────

export function showHistoryDetail(title) {
  const sessionList = dom.historySessionList || document.getElementById("historySessionList");
  const detail = dom.historyDetail || document.getElementById("historyDetail");
  const detailTitle = dom.historyDetailTitle || document.getElementById("historyDetailTitle");
  if (sessionList) sessionList.style.display = "none";
  if (detail) detail.style.display = "";
  if (detailTitle) detailTitle.textContent = title;
}

export function hideHistoryDetail() {
  const detail = dom.historyDetail || document.getElementById("historyDetail");
  const sessionList = dom.historySessionList || document.getElementById("historySessionList");
  if (detail) detail.style.display = "none";
  if (sessionList) sessionList.style.display = "";
}
