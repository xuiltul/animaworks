// ── History Panel Controller ──────────────────
import { $ } from "./ctx.js";

export function createHistoryController(ctx) {
  const { state, deps } = ctx;
  const { api, t, escapeHtml, renderMarkdown, timeStr } = deps;

  function showHistoryDetail(title) {
    const list = $("chatHistorySessionList");
    const detail = $("chatHistoryDetail");
    const titleEl = $("chatHistoryDetailTitle");
    if (list) list.style.display = "none";
    if (detail) detail.style.display = "";
    if (titleEl) titleEl.textContent = title;
  }

  async function loadSessionList() {
    const list = $("chatHistorySessionList");
    if (!list || !state.selectedAnima) {
      if (list) list.innerHTML = `<div class="loading-placeholder">${t("chat.anima_select_first")}</div>`;
      return;
    }

    list.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

    try {
      const data = await api(`/api/animas/${encodeURIComponent(state.selectedAnima)}/sessions`);
      let html = "";

      if (data.active_conversation) {
        const ac = data.active_conversation;
        const lastTime = ac.last_timestamp ? timeStr(ac.last_timestamp) : "--:--";
        html += `<div class="session-section-header">${t("chat.history_current")}</div>
          <div class="session-item session-active" data-type="active">
            <div class="session-item-title">${t("ws.conversation_active")}</div>
            <div class="session-item-meta">${t("chat.session_turns", { count: ac.total_turn_count })} ${ac.has_summary ? t("chat.session_summary") : ""} | ${t("chat.last_label")}: ${lastTime}</div>
          </div>`;
      }

      if (data.archived_sessions?.length > 0) {
        html += `<div class="session-section-header">${t("chat.history_archive")}</div>`;
        for (const s of data.archived_sessions) {
          const ts = s.timestamp ? timeStr(s.timestamp) : s.id;
          html += `<div class="session-item" data-type="archive" data-id="${escapeHtml(s.id)}">
            <div class="session-item-title">${escapeHtml(s.trigger || t("chat.session"))} (${t("chat.session_turns", { count: s.turn_count })})</div>
            <div class="session-item-meta">${ts} | ctx: ${(s.context_usage_ratio * 100).toFixed(0)}%</div>
            ${s.original_prompt_preview ? `<div class="session-item-preview">${escapeHtml(s.original_prompt_preview)}</div>` : ""}
          </div>`;
        }
      }

      if (data.transcripts?.length > 0) {
        html += `<div class="session-section-header">${t("chat.history_transcript")}</div>`;
        for (const tr of data.transcripts) {
          html += `<div class="session-item" data-type="transcript" data-date="${escapeHtml(tr.date)}">
            <div class="session-item-title">${escapeHtml(tr.date)}</div>
            <div class="session-item-meta">${tr.message_count} messages</div>
          </div>`;
        }
      }

      if (data.episodes?.length > 0) {
        html += `<div class="session-section-header">${t("chat.episode_log")}</div>`;
        for (const e of data.episodes) {
          html += `<div class="session-item" data-type="episode" data-date="${escapeHtml(e.date)}">
            <div class="session-item-title">${escapeHtml(e.date)}</div>
            <div class="session-item-preview">${escapeHtml(e.preview)}</div>
          </div>`;
        }
      }

      if (!html) html = `<div class="loading-placeholder">${t("chat.history_empty")}</div>`;
      list.innerHTML = html;

      list.querySelectorAll(".session-item").forEach(item => {
        item.addEventListener("click", () => {
          const type = item.dataset.type;
          if (type === "active") loadActiveConversation();
          else if (type === "archive") loadArchivedSession(item.dataset.id);
          else if (type === "transcript") loadTranscript(item.dataset.date);
          else if (type === "episode") loadEpisode(item.dataset.date);
        });
      });
    } catch (err) {
      list.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderConversationHistoryDetail(data) {
    const conv = $("chatHistoryConversation");
    if (!conv) return;
    if (!data?.sessions?.length) {
      conv.innerHTML = '<div class="loading-placeholder">\u4F1A\u8A71\u30C7\u30FC\u30BF\u304C\u3042\u308A\u307E\u305B\u3093</div>';
      return;
    }
    let html = "";
    for (let si = 0; si < data.sessions.length; si++) {
      const session = data.sessions[si];
      html += ctx.controllers.renderer.renderSessionDivider(session, si === 0);
      if (session.messages) {
        for (const msg of session.messages) html += ctx.controllers.renderer.renderHistoryMessage(msg);
      }
    }
    if (!html) html = '<div class="loading-placeholder">\u4F1A\u8A71\u30C7\u30FC\u30BF\u304C\u3042\u308A\u307E\u305B\u3093</div>';
    conv.innerHTML = html;
    ctx.controllers.renderer.bindToolCallHandlers(conv);
    conv.scrollTop = conv.scrollHeight;
  }

  async function loadActiveConversation() {
    if (!state.selectedAnima) return;
    showHistoryDetail(t("chat.history_detail_title"));
    const conv = $("chatHistoryConversation");
    if (conv) conv.innerHTML = '<div class="loading-placeholder">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
    try {
      const data = await ctx.controllers.renderer.fetchConversationHistory(state.selectedAnima, 50, null, state.selectedThreadId);
      renderConversationHistoryDetail(data);
    } catch {
      if (conv) conv.innerHTML = '<div class="loading-placeholder">\u8AAD\u307F\u8FBC\u307F\u5931\u6557</div>';
    }
  }

  async function loadArchivedSession(sessionId) {
    if (!state.selectedAnima) return;
    showHistoryDetail(t("chat.session_detail", { id: sessionId }));
    const conv = $("chatHistoryConversation");
    if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;
    try {
      const data = await api(`/api/animas/${encodeURIComponent(state.selectedAnima)}/sessions/${encodeURIComponent(sessionId)}`);
      if (data.markdown) {
        if (conv) conv.innerHTML = `<div class="history-markdown">${renderMarkdown(data.markdown, state.selectedAnima)}</div>`;
      } else if (data.data) {
        const d = data.data;
        let html = `<div class="history-session-meta"><div><strong>${t("chat.state_label")}</strong> ${escapeHtml(d.trigger || t("chat.state_unknown"))}</div><div><strong>${t("chat.turn_count")}</strong> ${d.turn_count || 0}</div><div><strong>${t("chat.context_usage")}</strong> ${((d.context_usage_ratio || 0) * 100).toFixed(0)}%</div></div>`;
        if (d.original_prompt) html += `<div class="history-section"><div class="history-section-label">${t("chat.request_label")}</div><pre class="history-pre">${escapeHtml(d.original_prompt)}</pre></div>`;
        if (d.accumulated_response) html += `<div class="history-section"><div class="history-section-label">${t("chat.response_label")}</div><div>${renderMarkdown(d.accumulated_response, state.selectedAnima)}</div></div>`;
        if (conv) conv.innerHTML = html;
      } else {
        if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("chat.no_data")}</div>`;
      }
    } catch {
      if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}</div>`;
    }
  }

  function renderTranscriptDetail(data) {
    const conv = $("chatHistoryConversation");
    if (!conv) return;
    let html = "";
    if (data.turns?.length > 0) {
      for (const turn of data.turns) {
        const ts = turn.timestamp ? timeStr(turn.timestamp) : "";
        const bubbleClass = turn.role === "assistant" ? "assistant" : "user";
        const roleLabel = turn.role === "human" ? t("chat.role_human") : turn.role;
        const content = turn.role === "assistant" ? renderMarkdown(turn.content || "", state.selectedAnima) : escapeHtml(turn.content || "");
        html += `<div class="history-turn"><div class="history-turn-meta">${ts} - ${escapeHtml(roleLabel)}</div><div class="chat-bubble ${bubbleClass}">${content}</div></div>`;
      }
    }
    if (!html) html = `<div class="loading-placeholder">${t("chat.history_no_data")}</div>`;
    conv.innerHTML = html;
    conv.scrollTop = conv.scrollHeight;
  }

  async function loadTranscript(date) {
    if (!state.selectedAnima) return;
    showHistoryDetail(t("chat.transcript_detail", { date }));
    const conv = $("chatHistoryConversation");
    if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;
    try {
      const data = await api(`/api/animas/${encodeURIComponent(state.selectedAnima)}/transcripts/${encodeURIComponent(date)}`);
      renderTranscriptDetail(data);
    } catch {
      if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}</div>`;
    }
  }

  async function loadEpisode(date) {
    if (!state.selectedAnima) return;
    showHistoryDetail(t("chat.episode_detail", { date }));
    const conv = $("chatHistoryConversation");
    if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;
    try {
      const data = await api(`/api/animas/${encodeURIComponent(state.selectedAnima)}/episodes/${encodeURIComponent(date)}`);
      if (conv) conv.innerHTML = `<div class="history-markdown">${renderMarkdown(data.content || t("chat.no_content"), state.selectedAnima)}</div>`;
    } catch {
      if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}</div>`;
    }
  }

  return { loadSessionList, loadActiveConversation, loadArchivedSession, loadTranscript, loadEpisode };
}
