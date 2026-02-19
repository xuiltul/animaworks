// ── Session & Conversation History ──────────────────────
// Session list, active conversation, archived sessions, transcripts, episodes.

import { getState, setState } from "./state.js";
import * as api from "./api.js";
import { escapeHtml, renderSimpleMarkdown, timeStr } from "./utils.js";

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
        <button class="session-back-btn">&larr; Back</button>
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
    listArea.innerHTML = '<div class="loading-placeholder">Anima を選択してください</div>';
    return;
  }

  listArea.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const data = await api.fetchSessions(selectedAnima);
    setState({ sessionList: data });
    renderItems(data);
  } catch (err) {
    console.error("Failed to load sessions:", err);
    listArea.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
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
        <div class="section-header">現在の会話</div>
        <div class="session-item session-active" data-type="active">
          <div class="session-item-title">進行中の会話</div>
          <div class="session-item-meta">
            ${ac.total_turn_count || 0}ターン ${ac.has_summary ? "(要約あり)" : ""}
            | 最終: ${lastTime}
          </div>
        </div>
      </div>`;
  }

  // 2. Archived sessions
  if (data.archived_sessions && data.archived_sessions.length > 0) {
    html += '<div class="session-section"><div class="section-header">セッションアーカイブ</div>';
    for (const s of data.archived_sessions) {
      const ts = s.timestamp ? timeStr(s.timestamp) : s.id;
      html += `
        <div class="session-item" data-type="archive" data-id="${escapeHtml(s.id)}">
          <div class="session-item-title">${escapeHtml(s.trigger || "セッション")} (${s.turn_count || 0}ターン)</div>
          <div class="session-item-meta">${ts}</div>
        </div>`;
    }
    html += "</div>";
  }

  // 3. Transcripts
  if (data.transcripts && data.transcripts.length > 0) {
    html += '<div class="session-section"><div class="section-header">会話ログ</div>';
    for (const t of data.transcripts) {
      html += `
        <div class="session-item" data-type="transcript" data-date="${escapeHtml(t.date)}">
          <div class="session-item-title">${escapeHtml(t.date)}</div>
          <div class="session-item-meta">${t.message_count || 0}メッセージ</div>
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
    html = '<div class="loading-placeholder">履歴がありません</div>';
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
        <div class="history-summary-label">要約 (${data.compressed_turn_count || 0}ターン分)</div>
        <div class="history-summary-body">${renderSimpleMarkdown(data.compressed_summary)}</div>
      </div>`;
  }

  // Turn list
  if (data.turns && data.turns.length > 0) {
    for (const t of data.turns) {
      const ts = t.timestamp ? timeStr(t.timestamp) : "";
      const roleClass = t.role === "assistant" ? "assistant" : "user";
      const roleLabel = t.role === "human" ? "ユーザー" : t.role;
      const content =
        t.role === "assistant" ? renderSimpleMarkdown(t.content || "") : escapeHtml(t.content || "");
      html += `
        <div class="session-turn">
          <div class="session-turn-meta">${ts} - ${escapeHtml(roleLabel)}</div>
          <div class="session-turn-bubble ${roleClass}">${content}</div>
        </div>`;
    }
  }

  if (!html) {
    html = '<div class="loading-placeholder">会話データがありません</div>';
  }

  body.innerHTML = html;
  body.scrollTop = body.scrollHeight;
}

async function loadActiveConversation() {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  showDetail("進行中の会話");
  const body = getDetailBody();
  if (body) body.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const data = await api.fetchConversationFull(selectedAnima);
    renderTurns(data);
  } catch (err) {
    console.error("Failed to load active conversation:", err);
    if (body) body.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
  }
}

async function loadArchivedSession(sessionId) {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  showDetail(`セッション: ${sessionId}`);
  const body = getDetailBody();
  if (body) body.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const data = await api.fetchSession(selectedAnima, sessionId);
    renderArchivedDetail(data);
  } catch (err) {
    console.error("Failed to load archived session:", err);
    if (body) body.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
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
        <div><strong>トリガー:</strong> ${escapeHtml(d.trigger || "不明")}</div>
        <div><strong>ターン数:</strong> ${d.turn_count || 0}</div>
        <div><strong>コンテキスト使用率:</strong> ${((d.context_usage_ratio || 0) * 100).toFixed(0)}%</div>
      </div>`;

    if (d.original_prompt) {
      html += `
        <div class="session-section-block">
          <div class="session-section-label">依頼内容</div>
          <pre class="session-pre">${escapeHtml(d.original_prompt)}</pre>
        </div>`;
    }

    if (d.accumulated_response) {
      html += `
        <div class="session-section-block">
          <div class="session-section-label">応答</div>
          <div>${renderSimpleMarkdown(d.accumulated_response)}</div>
        </div>`;
    }

    body.innerHTML = html;
    return;
  }

  body.innerHTML = '<div class="loading-placeholder">データがありません</div>';
}

async function loadTranscript(date) {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  showDetail(`会話ログ: ${date}`);
  const body = getDetailBody();
  if (body) body.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const data = await api.fetchTranscript(selectedAnima, date);
    renderTurns(data);
  } catch (err) {
    console.error("Failed to load transcript:", err);
    if (body) body.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
  }
}

async function loadEpisode(date) {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  showDetail(`エピソード: ${date}`);
  const body = getDetailBody();
  if (body) body.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const data = await api.fetchEpisode(selectedAnima, date);
    body.innerHTML = `<div class="session-markdown">${renderSimpleMarkdown(data.content || "(内容なし)")}</div>`;
  } catch (err) {
    console.error("Failed to load episode:", err);
    if (body) body.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
  }
}
