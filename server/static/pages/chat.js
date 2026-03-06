// ── Chat Page (Orchestrator) ──────────────────
import { t } from "/shared/i18n.js";
import { createPaneHost } from "./chat/pane-host.js";

let _host = null;

export function render(container) {
  container.innerHTML = `
    <!-- Mobile Tab Bar -->
    <nav class="chat-mobile-tabs" data-chat-id="chatMobileTabs">
      <button class="chat-mobile-tab active" data-panel="chat" data-chat-id="chatMobileTabChat">${t("nav.chat")}</button>
      <button class="chat-mobile-tab" data-panel="info" data-chat-id="chatMobileTabInfo">${t("chat.character_summary")}</button>
    </nav>

    <div class="chat-page-layout" data-chat-id="chatPageLayout">
      <!-- Pane Host -->
      <div class="chat-pane-host" data-chat-id="chatPaneHost"></div>

      <div class="chat-right-pane-handle">
        <button
          type="button"
          data-chat-id="chatRightPaneToggleBtn"
          class="chat-right-pane-toggle-btn"
          aria-label="${t("pane.sidebar_toggle")}"
          title="${t("pane.sidebar_toggle")}"
        >
          <svg class="chat-right-pane-toggle-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M15 6l-6 6 6 6" />
          </svg>
        </button>
      </div>

      <!-- Right: Sidebar -->
      <div class="chat-page-sidebar mobile-hidden">
        <nav class="right-tabs" style="display:flex; border-bottom:1px solid var(--border-color, #eee);">
          <button class="right-tab active" data-tab="state" data-chat-id="chatTabState">${t("chat.state_current")}</button>
          <button class="right-tab" data-tab="activity" data-chat-id="chatTabActivity">${t("nav.activity")}</button>
          <button class="right-tab" data-tab="history" data-chat-id="chatTabHistory">${t("chat.history_conversation")}</button>
        </nav>

        <div data-chat-id="chatRightTabContent" style="padding:0.75rem;">
          <div data-chat-id="chatPaneState">
            <pre class="state-content" data-chat-id="chatAnimaState" style="white-space:pre-wrap; word-break:break-word; margin:0;">${t("chat.anima_select_first")}</pre>
          </div>
          <div data-chat-id="chatPaneActivity" style="display:none;">
            <div data-chat-id="chatActivityFeed" class="activity-feed">
              <div class="activity-empty">${t("activity.feed_empty")}</div>
            </div>
          </div>
          <div data-chat-id="chatPaneHistory" style="display:none;">
            <div data-chat-id="chatHistorySessionList">
              <div class="loading-placeholder">${t("chat.anima_select_first")}</div>
            </div>
            <div data-chat-id="chatHistoryDetail" style="display:none;">
              <button class="memory-back-btn" data-chat-id="chatHistoryBackBtn">&larr; ${t("chat.back_list")}</button>
              <h3 data-chat-id="chatHistoryDetailTitle" style="margin:0.5rem 0;"></h3>
              <div data-chat-id="chatHistoryConversation" style="max-height:400px; overflow-y:auto;"></div>
            </div>
          </div>

          <div class="chat-memory-section">
            <nav class="memory-tabs" style="display:flex; border-bottom:1px solid var(--border-color, #eee);">
              <button class="memory-tab active" data-tab="episodes">${t("chat.memory_episodes")}</button>
              <button class="memory-tab" data-tab="knowledge">${t("chat.memory_knowledge")}</button>
              <button class="memory-tab" data-tab="procedures">${t("chat.memory_procedures")}</button>
            </nav>
            <div data-chat-id="chatMemoryFileList" class="memory-file-list" style="padding:0.5rem;">
              <div class="loading-placeholder">${t("chat.anima_select")}</div>
            </div>
            <div data-chat-id="chatMemoryContentArea" style="display:none; padding:0.5rem;">
              <button class="memory-back-btn" data-chat-id="chatMemoryBackBtn">&larr; ${t("chat.memory_back")}</button>
              <h3 data-chat-id="chatMemoryContentTitle" style="margin:0.5rem 0;"></h3>
              <pre data-chat-id="chatMemoryContentBody" class="memory-content-body" style="white-space:pre-wrap; word-break:break-word;"></pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  _host = createPaneHost(container);
  _host.bindSharedEvents();
  _host.restoreLayout();
  if (_host.panes.length === 0) _host.addPane();
}

export function destroy() {
  if (!_host) return;
  _host.destroy();
  _host = null;
}

export function splitPane() {
  return _host?.splitPane() ?? null;
}

export function getPaneHost() {
  return _host;
}
