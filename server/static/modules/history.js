/* ── History Panel ─────────────────────────── */

import { state, dom, timeStr, escapeHtml, renderMarkdown } from "./state.js";
import { api } from "./api.js";

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
    sessionList.innerHTML = '<div class="loading-placeholder">Animaを選択してください</div>';
    return;
  }
  sessionList.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';
  try {
    const data = await api(`/api/animas/${encodeURIComponent(name)}/sessions`);
    state.sessionList = data;
    renderSessionList(data);
  } catch (err) {
    sessionList.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
  }
}

function renderSessionList(data) {
  let html = "";

  // Active Conversation
  if (data.active_conversation) {
    const ac = data.active_conversation;
    const lastTime = ac.last_timestamp ? timeStr(ac.last_timestamp) : "--:--";
    html += `
      <div class="session-section-header">現在の会話</div>
      <div class="session-item session-active" data-type="active">
        <div class="session-item-title">進行中の会話</div>
        <div class="session-item-meta">
          ${ac.total_turn_count}ターン ${ac.has_summary ? "(要約あり)" : ""}
          | 最終: ${lastTime}
        </div>
      </div>`;
  }

  // Archived Sessions
  if (data.archived_sessions && data.archived_sessions.length > 0) {
    html += '<div class="session-section-header">セッションアーカイブ</div>';
    for (const s of data.archived_sessions) {
      const ts = s.timestamp ? timeStr(s.timestamp) : s.id;
      html += `
        <div class="session-item" data-type="archive" data-id="${escapeHtml(s.id)}">
          <div class="session-item-title">${escapeHtml(s.trigger || "セッション")} (${s.turn_count}ターン)</div>
          <div class="session-item-meta">${ts} | ctx: ${(s.context_usage_ratio * 100).toFixed(0)}%</div>
          ${s.original_prompt_preview ? `<div class="session-item-preview">${escapeHtml(s.original_prompt_preview)}</div>` : ""}
        </div>`;
    }
  }

  // Transcripts
  if (data.transcripts && data.transcripts.length > 0) {
    html += '<div class="session-section-header">会話ログ</div>';
    for (const t of data.transcripts) {
      html += `
        <div class="session-item" data-type="transcript" data-date="${escapeHtml(t.date)}">
          <div class="session-item-title">${escapeHtml(t.date)}</div>
          <div class="session-item-meta">${t.message_count}メッセージ</div>
        </div>`;
    }
  }

  // Episodes
  if (data.episodes && data.episodes.length > 0) {
    html += '<div class="session-section-header">エピソードログ</div>';
    for (const e of data.episodes) {
      html += `
        <div class="session-item" data-type="episode" data-date="${escapeHtml(e.date)}">
          <div class="session-item-title">${escapeHtml(e.date)}</div>
          <div class="session-item-preview">${escapeHtml(e.preview)}</div>
        </div>`;
    }
  }

  if (!html) {
    html = '<div class="loading-placeholder">履歴がありません</div>';
  }

  const sessionListEl = dom.historySessionList || document.getElementById("historySessionList");
  if (!sessionListEl) return;
  sessionListEl.innerHTML = html;

  // Bind click handlers
  sessionListEl.querySelectorAll(".session-item").forEach((item) => {
    item.addEventListener("click", () => {
      const type = item.dataset.type;
      if (type === "active") loadActiveConversation();
      else if (type === "archive") loadArchivedSession(item.dataset.id);
      else if (type === "transcript") loadTranscriptInHistory(item.dataset.date);
      else if (type === "episode") loadEpisodeInHistory(item.dataset.date);
    });
  });
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

// ── Conversation Loading ────────────────────

async function loadActiveConversation() {
  const name = state.selectedAnima;
  if (!name) return;

  showHistoryDetail("進行中の会話");
  const conv = getHistoryConv();
  if (conv) conv.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const data = await api(`/api/animas/${encodeURIComponent(name)}/conversation/full?limit=50`);
    renderConversationDetail(data);
  } catch (err) {
    if (conv) conv.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
  }
}

function renderConversationDetail(data) {
  const conv = getHistoryConv();
  if (!conv) return;

  let html = "";

  if (data.has_summary && data.compressed_summary) {
    html += `<div class="history-summary">
      <div class="history-summary-label">要約 (${data.compressed_turn_count}ターン分)</div>
      <div class="history-summary-body">${renderMarkdown(data.compressed_summary)}</div>
    </div>`;
  }

  if (data.turns && data.turns.length > 0) {
    for (const t of data.turns) {
      const ts = t.timestamp ? timeStr(t.timestamp) : "";
      const bubbleClass = t.role === "assistant" ? "assistant" : "user";
      const roleLabel = t.role === "human" ? "ユーザー" : t.role;
      const content = t.role === "assistant" ? renderMarkdown(t.content) : escapeHtml(t.content);
      html += `
        <div class="history-turn">
          <div class="history-turn-meta">${ts} - ${escapeHtml(roleLabel)}</div>
          <div class="chat-bubble ${bubbleClass}">${content}</div>
        </div>`;
    }
  }

  if (!html) {
    html = '<div class="loading-placeholder">会話データがありません</div>';
  }

  conv.innerHTML = html;
  conv.scrollTop = conv.scrollHeight;
}

async function loadArchivedSession(sessionId) {
  const name = state.selectedAnima;
  if (!name) return;

  showHistoryDetail(`セッション: ${sessionId}`);
  const conv = getHistoryConv();
  if (conv) conv.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const data = await api(`/api/animas/${encodeURIComponent(name)}/sessions/${encodeURIComponent(sessionId)}`);
    renderArchivedSessionDetail(data);
  } catch (err) {
    if (conv) conv.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
  }
}

function renderArchivedSessionDetail(data) {
  const conv = getHistoryConv();
  if (!conv) return;

  if (data.markdown) {
    conv.innerHTML = `<div class="history-markdown">${renderMarkdown(data.markdown)}</div>`;
  } else if (data.data) {
    const d = data.data;
    let html = `<div class="history-session-meta">
      <div><strong>トリガー:</strong> ${escapeHtml(d.trigger || "不明")}</div>
      <div><strong>ターン数:</strong> ${d.turn_count || 0}</div>
      <div><strong>コンテキスト使用率:</strong> ${((d.context_usage_ratio || 0) * 100).toFixed(0)}%</div>
    </div>`;
    if (d.original_prompt) {
      html += `<div class="history-section"><div class="history-section-label">依頼内容</div><pre class="history-pre">${escapeHtml(d.original_prompt)}</pre></div>`;
    }
    if (d.accumulated_response) {
      html += `<div class="history-section"><div class="history-section-label">応答</div><div>${renderMarkdown(d.accumulated_response)}</div></div>`;
    }
    conv.innerHTML = html;
  } else {
    conv.innerHTML = '<div class="loading-placeholder">データがありません</div>';
  }
}

async function loadTranscriptInHistory(date) {
  const name = state.selectedAnima;
  if (!name) return;

  showHistoryDetail(`会話ログ: ${date}`);
  const conv = getHistoryConv();
  if (conv) conv.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const data = await api(`/api/animas/${encodeURIComponent(name)}/transcripts/${encodeURIComponent(date)}`);
    renderConversationDetail(data);
  } catch (err) {
    if (conv) conv.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
  }
}

async function loadEpisodeInHistory(date) {
  const name = state.selectedAnima;
  if (!name) return;

  showHistoryDetail(`エピソード: ${date}`);
  const conv = getHistoryConv();
  if (conv) conv.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const data = await api(`/api/animas/${encodeURIComponent(name)}/episodes/${encodeURIComponent(date)}`);
    if (conv) conv.innerHTML = `<div class="history-markdown">${renderMarkdown(data.content || "(内容なし)")}</div>`;
  } catch (err) {
    if (conv) conv.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
  }
}
